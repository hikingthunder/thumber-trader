
import time
import logging
import ipaddress
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
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


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    Stateful CSRF protection.
    Ensures that state-changing requests (POST, PUT, DELETE, PATCH)
    contain a token that matches the one stored in a cookie.
    """

    SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}
    # Paths that don't need CSRF (API, webhooks, or explicit bypass)
    EXEMPT_PATHS = {"/api/", "/ws/", "/webhook/"}

    async def dispatch(self, request: Request, call_next):
        # 1. Skip validation for safe methods
        if request.method in self.SAFE_METHODS:
            response = await call_next(request)
            
            # Ensure CSRF cookie is present
            if not request.cookies.get(CSRF_COOKIE_NAME):
                from app.auth.security import generate_csrf_token
                response.set_cookie(
                    CSRF_COOKIE_NAME,
                    generate_csrf_token(),
                    httponly=False,  # Accessible by frontend JS if needed
                    samesite="lax",
                    secure=False  # Should be True in production with HTTPS
                )
            return response

        # 2. Check for exemptions
        if any(request.url.path.startswith(p) for p in self.EXEMPT_PATHS):
            return await call_next(request)

        # 3. Validate Token
        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
        if not cookie_token:
            logger.warning(f"CSRF failure: Missing cookie for {request.url.path}")
            return JSONResponse({"detail": "CSRF cookie missing"}, status_code=403)

        # Check header first, then form data
        submitted_token = request.headers.get("X-CSRF-Token")
        if not submitted_token:
            try:
                # We need to swallow exceptions as not all POSTs are forms
                form = await request.form()
                submitted_token = form.get("csrf_token")
            except Exception:
                pass

        if not submitted_token or submitted_token != cookie_token:
            logger.warning(f"CSRF failure: Invalid token for {request.url.path}")
            return JSONResponse({"detail": "CSRF token invalid or missing"}, status_code=403)

        return await call_next(request)
