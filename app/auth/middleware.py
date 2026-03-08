
import time
import logging
import ipaddress
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.datastructures import MutableHeaders
from sqlalchemy import select

from app.database.db import AsyncSessionLocal
from app.auth.security import CSRF_COOKIE_NAME

logger = logging.getLogger(__name__)


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    Check incoming request IPs against the whitelist.
    If no entries exist in the whitelist table, all IPs are allowed (open mode).
    """

    # Paths that bypass IP check (healthcheck, etc.)
    BYPASS_PATHS = {"/health", "/auth/login", "/auth/register"}

    async def dispatch(self, request: Request, call_next):
        # Always allow bypass paths
        if request.url.path in self.BYPASS_PATHS:
            return await call_next(request)

        try:
            from app.auth.models import IPWhitelistEntry
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(IPWhitelistEntry).where(IPWhitelistEntry.is_active == True)
                )
                entries = result.scalars().all()

            # If no whitelist entries, open mode — allow all
            if not entries:
                return await call_next(request)

            client_ip = self._get_client_ip(request)
            try:
                client_addr = ipaddress.ip_address(client_ip)
            except ValueError:
                return JSONResponse({"detail": "Invalid client IP"}, status_code=403)

            for entry in entries:
                try:
                    network = ipaddress.ip_network(entry.cidr, strict=False)
                    if client_addr in network:
                        return await call_next(request)
                except ValueError:
                    continue

            logger.warning(f"IP {client_ip} blocked by whitelist")
            return JSONResponse({"detail": "Access denied: IP not in whitelist"}, status_code=403)

        except Exception as e:
            # If DB is not ready yet, allow through
            logger.debug(f"IP whitelist check skipped: {e}")
            return await call_next(request)

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "0.0.0.0"


class AuditMiddleware(BaseHTTPMiddleware):
    """Log state-changing requests (POST, PUT, DELETE) to audit log."""

    # Only audit these methods
    AUDIT_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

    # Paths to skip (high-frequency polling)
    SKIP_PATHS = {"/dashboard/stats", "/dashboard/orders", "/dashboard/fills", "/dashboard/price", "/health"}

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if request.method in self.AUDIT_METHODS and request.url.path not in self.SKIP_PATHS:
            try:
                from app.auth.security import decode_access_token, COOKIE_NAME
                from app.auth.models import AuditLog

                token = request.cookies.get(COOKIE_NAME)
                user_id = None
                if token:
                    payload = decode_access_token(token)
                    if payload and payload.get("sub"):
                        # Quick lookup by username to get id
                        from app.auth.models import User
                        async with AsyncSessionLocal() as session:
                            result = await session.execute(
                                select(User.id).where(User.username == payload["sub"])
                            )
                            row = result.first()
                            if row:
                                user_id = row[0]

                ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
                    request.client.host if request.client else "unknown"
                )

                async with AsyncSessionLocal() as session:
                    entry = AuditLog(
                        user_id=user_id,
                        action=f"{request.method} {request.url.path}",
                        detail=f"status={response.status_code}",
                        ip_address=ip,
                        user_agent=request.headers.get("user-agent", "")[:500],
                        timestamp=time.time()
                    )
                    session.add(entry)
                    await session.commit()
            except Exception as e:
                logger.debug(f"Audit log failed: {e}")

        return response


class SessionTimeoutMiddleware(BaseHTTPMiddleware):
    """Validate that JWT token hasn't expired beyond session timeout."""

    UNPROTECTED_PATHS = {"/auth/login", "/auth/register", "/auth/logout", "/health", "/static"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow static and auth paths
        if any(path.startswith(p) for p in self.UNPROTECTED_PATHS):
            return await call_next(request)

        from app.auth.security import COOKIE_NAME, decode_access_token

        token = request.cookies.get(COOKIE_NAME)
        if token:
            payload = decode_access_token(token)
            if payload:
                exp = payload.get("exp", 0)
                if time.time() > exp:
                    # Token expired — redirect to login
                    from starlette.responses import RedirectResponse
                    response = RedirectResponse(url="/auth/login", status_code=303)
                    response.delete_cookie(COOKIE_NAME)
                    return response

        return await call_next(request)


class CSRFMiddleware:
    """
    Standard ASGI middleware for CSRF protection.
    Ensures that state-changing requests (POST, PUT, DELETE, PATCH)
    contain a token that matches the one stored in a cookie.
    
    This implementation avoids the body-consumption issue of BaseHTTPMiddleware
    by manually handling the receive stream when form data validation is required.
    """

    SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}
    # Paths that don't need CSRF (API, webhooks, or explicit bypass)
    EXEMPT_PATHS = {"/api/", "/ws/", "/webhook/"}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        # 1. Skip validation for safe methods
        if request.method in self.SAFE_METHODS:
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Check if cookie exists in request
                    if not request.cookies.get(CSRF_COOKIE_NAME):
                        from app.auth.security import generate_csrf_token
                        from starlette.datastructures import MutableHeaders
                        headers = MutableHeaders(scope=message)
                        # We can't easily use response.set_cookie here because we're at ASGI level
                        # but we can add the Set-Cookie header manually
                        cookie_val = f"{CSRF_COOKIE_NAME}={generate_csrf_token()}; Path=/; SameSite=lax"
                        headers.append("Set-Cookie", cookie_val)
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
            return

        # 2. Check for exemptions
        if any(request.url.path.startswith(p) for p in self.EXEMPT_PATHS):
            await self.app(scope, receive, send)
            return

        # 3. Validate Token
        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
        if not cookie_token:
            logger.warning(f"CSRF failure: Missing cookie for {request.url.path}")
            response = JSONResponse({"detail": "CSRF cookie missing"}, status_code=403)
            await response(scope, receive, send)
            return

        # Handle body consumption for token validation
        submitted_token = request.headers.get("X-CSRF-Token")
        
        if not submitted_token:
            # We need to peek at the body if it's a form
            content_type = request.headers.get("Content-Type", "")
            if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
                # Capture the body
                body_chunks = []
                async def wrapped_receive():
                    if body_chunks:
                        return body_chunks.pop(0)
                    return await receive()

                # Read the body to parse form (this is the tricky part in ASGI)
                # For simplicity and to avoid complex stream management, 
                # we'll use a more robust approach: capture all chunks, parse, then re-emit.
                entire_body = b""
                more_body = True
                while more_body:
                    message = await receive()
                    entire_body += message.get("body", b"")
                    more_body = message.get("more_body", False)
                    body_chunks.append(message)

                # Now we have the body, we can try to parse the token
                try:
                    # Temporary request to parse form from captured body
                    temp_scope = scope.copy()
                    async def temp_receive():
                        return {"type": "http.request", "body": entire_body, "more_body": False}
                    
                    temp_request = Request(temp_scope, temp_receive)
                    form = await temp_request.form()
                    submitted_token = form.get("csrf_token")
                except Exception as e:
                    logger.debug(f"Form parsing failed during CSRF check: {e}")

                # Use the wrapped receive for the rest of the application
                receive = wrapped_receive

        if not submitted_token or submitted_token != cookie_token:
            logger.warning(f"CSRF failure: Invalid token for {request.url.path}")
            response = JSONResponse({"detail": "CSRF token invalid or missing"}, status_code=403)
            await response(scope, receive, send)
            return

        # Continue with (potentially wrapped) receive
        await self.app(scope, receive, send)
