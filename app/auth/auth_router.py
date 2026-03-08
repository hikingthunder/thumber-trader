
import time
import logging
from fastapi import APIRouter, Request, HTTPException, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func

from app.database.db import AsyncSessionLocal
from app.auth.models import User
from app.auth.security import (
    hash_password, verify_password, create_access_token,
    generate_totp_secret, get_totp_uri, verify_totp, generate_qr_code_base64,
    generate_csrf_token, get_current_user, log_audit,
    COOKIE_NAME, CSRF_COOKIE_NAME, ACCESS_TOKEN_EXPIRE_MINUTES
)

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="/auth", tags=["auth"])
templates = Jinja2Templates(directory="app/web/templates")


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@auth_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render the login page."""
    csrf_token = request.cookies.get(CSRF_COOKIE_NAME) or generate_csrf_token()
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": None,
        "show_totp": False,
        "show_register": False,
        "csrf_token": csrf_token
    })


@auth_router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    totp_code: str = Form(default="")
):
    """Authenticate user with username/password + optional TOTP."""
    ip = _client_ip(request)
    ua = request.headers.get("user-agent", "")

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        await log_audit(None, "login_failed", f"username={username}", ip, ua)
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid username or password",
            "show_totp": False,
            "show_register": False,
            "csrf_token": request.cookies.get(CSRF_COOKIE_NAME)
        }, status_code=401)

    if not user.is_active:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Account is deactivated",
            "show_totp": False,
            "show_register": False,
            "csrf_token": request.cookies.get(CSRF_COOKIE_NAME)
        }, status_code=403)

    # Check TOTP if enabled
    if user.totp_enabled:
        if not totp_code:
            return templates.TemplateResponse("login.html", {
                "request": request,
                "error": None,
                "show_totp": True,
                "show_register": False,
                "username": username,
                "password": password
            })
        if not user.totp_secret or not verify_totp(user.totp_secret, totp_code):
            await log_audit(user.id, "totp_failed", "", ip, ua)
            return templates.TemplateResponse("login.html", {
                "request": request,
                "error": "Invalid authenticator code",
                "show_totp": True,
                "show_register": False,
                "username": username,
                "password": password
            }, status_code=401)

    # Issue JWT
    token = create_access_token(data={"sub": user.username, "role": user.role})

    # Update last login
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user.id))
        db_user = result.scalar_one()
        db_user.last_login = time.time()
        await session.commit()

    await log_audit(user.id, "login_success", "", ip, ua)

    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        secure=False  # Set to True in production with HTTPS
    )
    # Set CSRF token
    csrf = generate_csrf_token()
    response.set_cookie(key=CSRF_COOKIE_NAME, value=csrf, samesite="lax", max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    return response


@auth_router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Show registration page — only if no users exist (first-run setup)."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(func.count(User.id)))
        count = result.scalar()

    if count > 0:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Registration is disabled. Contact your administrator.",
            "show_totp": False,
            "show_register": False
        })

    csrf_token = request.cookies.get(CSRF_COOKIE_NAME) or generate_csrf_token()
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": None,
        "show_totp": False,
        "show_register": True,
        "csrf_token": csrf_token
    })


@auth_router.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...)
):
    """Create the first admin user. Locked after first user is created."""
    ip = _client_ip(request)

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(func.count(User.id)))
        count = result.scalar()

    if count > 0:
        raise HTTPException(status_code=403, detail="Registration is closed")

    if password != password_confirm:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Passwords do not match",
            "show_totp": False,
            "show_register": True,
            "csrf_token": request.cookies.get(CSRF_COOKIE_NAME)
        }, status_code=400)

    if len(password) < 8:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Password must be at least 8 characters",
            "show_totp": False,
            "show_register": True,
            "csrf_token": request.cookies.get(CSRF_COOKIE_NAME)
        }, status_code=400)

    async with AsyncSessionLocal() as session:
        new_user = User(
            username=username,
            hashed_password=hash_password(password),
            role="admin",
            created_at=time.time()
        )
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

    await log_audit(new_user.id, "user_registered", f"Admin user created: {username}", ip)
    logger.info(f"Admin user '{username}' created successfully.")

    return RedirectResponse(url="/auth/login", status_code=303)


@auth_router.get("/totp-setup", response_class=HTMLResponse)
async def totp_setup_page(request: Request, user=Depends(get_current_user)):
    """Show TOTP setup page with QR code."""
    secret = generate_totp_secret()
    uri = get_totp_uri(secret, user.username)
    qr_b64 = generate_qr_code_base64(uri)

    return templates.TemplateResponse("totp_setup.html", {
        "request": request,
        "qr_code": qr_b64,
        "secret": secret,
        "error": None,
        "user": user
    })


@auth_router.post("/totp-verify")
async def totp_verify(
    request: Request,
    secret: str = Form(...),
    code: str = Form(...),
    user=Depends(get_current_user)
):
    """Verify TOTP code and enable 2FA for the user."""
    if not verify_totp(secret, code):
        uri = get_totp_uri(secret, user.username)
        qr_b64 = generate_qr_code_base64(uri)
        return templates.TemplateResponse("totp_setup.html", {
            "request": request,
            "qr_code": qr_b64,
            "secret": secret,
            "error": "Invalid code. Try again.",
            "user": user
        })

    # Save TOTP secret and enable
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user.id))
        db_user = result.scalar_one()
        db_user.totp_secret = secret
        db_user.totp_enabled = True
        await session.commit()

    await log_audit(user.id, "totp_enabled", "", _client_ip(request))

    return RedirectResponse(url="/?msg=2FA+enabled+successfully", status_code=303)


@auth_router.get("/logout")
async def logout(request: Request):
    """Clear session and redirect to login."""
    user = None
    try:
        user = await get_current_user(request)
    except Exception:
        pass

    if user:
        await log_audit(user.id, "logout", "", _client_ip(request))

    response = RedirectResponse(url="/auth/login", status_code=303)
    response.delete_cookie(COOKIE_NAME)
    response.delete_cookie(CSRF_COOKIE_NAME)
    return response


@auth_router.get("/audit-log", response_class=HTMLResponse)
async def audit_log_page(request: Request, user=Depends(get_current_user)):
    """Show the audit log (admin only)."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    from app.auth.models import AuditLog
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(200)
        )
        logs = result.scalars().all()

    return templates.TemplateResponse("audit_log.html", {
        "request": request,
        "logs": logs,
        "user": user
    })
