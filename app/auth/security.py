
import os
import time
import logging
import io
import base64
from typing import Optional
from datetime import timedelta

from passlib.context import CryptContext
from jose import JWTError, jwt
import pyotp
import qrcode

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import AsyncSessionLocal

logger = logging.getLogger(__name__)

# --- Configuration ---
SECRET_KEY = os.getenv("JWT_SECRET_KEY", os.urandom(32).hex())
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
COOKIE_NAME = "thumber_access_token"
CSRF_COOKIE_NAME = "thumber_csrf_token"

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# --- JWT Tokens ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = time.time() + (expires_delta.total_seconds() if expires_delta else ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    to_encode.update({"exp": expire, "iat": time.time()})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# --- TOTP ---
def generate_totp_secret() -> str:
    return pyotp.random_base32()


def get_totp_uri(secret: str, username: str, issuer: str = "ThumberTrader") -> str:
    return pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name=issuer)


def verify_totp(secret: str, code: str) -> bool:
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)  # Allow 30s window


def generate_qr_code_base64(uri: str) -> str:
    """Generate a QR code as a base64-encoded PNG for embedding in HTML."""
    qr = qrcode.QRCode(version=1, box_size=6, border=2)
    qr.add_data(uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="white", back_color="#0f172a")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# --- CSRF ---
def generate_csrf_token() -> str:
    return os.urandom(32).hex()


# --- FastAPI Dependencies ---
async def get_current_user(request: Request):
    """Extract and validate JWT from cookie. Returns User or redirects to login."""
    from app.auth.models import User  # Deferred import to avoid circular

    token = request.cookies.get(COOKIE_NAME)
    if not token:
        # For API endpoints, raise 401; for pages, redirect to login
        if request.url.path.startswith("/api/") or request.url.path.startswith("/ws/"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        raise HTTPException(status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                            headers={"Location": "/auth/login"})

    payload = decode_access_token(token)
    if not payload:
        if request.url.path.startswith("/api/") or request.url.path.startswith("/ws/"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired or invalid")
        raise HTTPException(status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                            headers={"Location": "/auth/login"})

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return user


async def get_optional_user(request: Request):
    """Try to get user from cookie, return None if not authenticated."""
    try:
        return await get_current_user(request)
    except HTTPException:
        return None


# --- Audit Logging Helper ---
async def log_audit(user_id: Optional[int], action: str, detail: str = "", ip_address: str = "", user_agent: str = ""):
    """Write an entry to the audit log."""
    from app.auth.models import AuditLog

    async with AsyncSessionLocal() as session:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            detail=detail,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=time.time()
        )
        session.add(entry)
        await session.commit()
