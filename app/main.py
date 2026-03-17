
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.database.db import init_db
from app.core.manager import manager
from app.web.router import router as web_router
from app.auth.auth_router import auth_router
from app.web.ws_router import ws_router
from app.web.webhook_router import router as webhook_router

from app.auth.middleware import IPWhitelistMiddleware, AuditMiddleware, SessionTimeoutMiddleware, CSRFMiddleware

# Setup Logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Rate Limiter (SlowAPI)
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application starting up...")

    # Initialize Database (creates all tables including auth tables)
    await init_db()

    # Check if first-run (no users exist) and log it
    try:
        from sqlalchemy import select, func
        from app.database.db import AsyncSessionLocal
        from app.auth.models import User
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(func.count(User.id)))
            count = result.scalar()
        if count == 0:
            logger.info("⚠️  No users found — first-run mode. Navigate to /auth/register to create admin account.")
    except Exception:
        pass

    # Initialize and Start Strategies
    try:
        await manager.initialize()
        if settings.auto_start:
            await manager.start()
    except Exception as e:
        logger.error(f"Failed to start strategies: {e}")

    yield

    # Shutdown
    logger.info("Application shutting down...")
    await manager.stop()

app = FastAPI(
    title="Thumber Trader",
    description="Institutional-grade grid trading bot with FastAPI and HTMX",
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None,  # Disable Swagger UI in production
    redoc_url=None,
)

# --- Rate Limiter ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Middleware Stack (order matters: last added = first executed) ---
# CORS
cors_allowed_origins = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
allow_credentials = "*" not in cors_allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session timeout (checks JWT expiry)
app.add_middleware(SessionTimeoutMiddleware)

# CSRF Protection
app.add_middleware(CSRFMiddleware)

# Audit logging (logs POST/PUT/DELETE)
app.add_middleware(AuditMiddleware)

# IP whitelist (blocks unauthorized IPs)
app.add_middleware(IPWhitelistMiddleware)

# --- Routers ---
app.include_router(auth_router)
app.include_router(web_router)
app.include_router(ws_router)
app.include_router(webhook_router)

# --- Static Files ---
static_dir = os.path.join(os.path.dirname(__file__), "static")
branding_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "branding")
if os.path.isdir(branding_dir):
    app.mount("/static/branding", StaticFiles(directory=branding_dir), name="branding")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# --- Root Redirect ---
@app.get("/", include_in_schema=False)
async def root_redirect(request: Request):
    """Redirect root to dashboard, but check auth first."""
    from app.auth.security import get_optional_user
    user = await get_optional_user(request)
    if not user:
        # Check if any users exist
        try:
            from sqlalchemy import select, func
            from app.database.db import AsyncSessionLocal
            from app.auth.models import User
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(func.count(User.id)))
                count = result.scalar()
            if count == 0:
                return RedirectResponse(url="/auth/register", status_code=303)
        except Exception:
            pass
        return RedirectResponse(url="/auth/login", status_code=303)
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/health")
async def health_check():
    return {"status": "ok", "manager_running": manager.running}


@app.get("/metrics")
async def metrics():
    """
    Expose bot metrics in Prometheus format.
    Simplified version without external prometheus_client dependency.
    """
    if not settings.prometheus_enabled:
        return ""

    stats = await manager.get_global_stats()
    lines = [
        "# HELP thumber_trader_running Boolean if manager is running",
        "# TYPE thumber_trader_running gauge",
        f"thumber_trader_running {1 if stats['running'] else 0}",

        "# HELP thumber_trader_pnl_realized_usd Realized PnL in USD",
        "# TYPE thumber_trader_pnl_realized_usd gauge",
        f"thumber_trader_pnl_realized_usd {stats['total_realized_pnl']}",

        "# HELP thumber_trader_pnl_unrealized_usd Unrealized PnL in USD",
        "# TYPE thumber_trader_pnl_unrealized_usd gauge",
        f"thumber_trader_pnl_unrealized_usd {stats['total_unrealized_pnl']}"
    ]

    for pid, s_stats in stats["strategies"].items():
        lines.extend([
            f'thumber_trader_strategy_orders_count{{product_id="{pid}"}} {s_stats["orders_count"]}',
            f'thumber_trader_strategy_last_price{{product_id="{pid}"}} {s_stats["last_price"]}',
            f'thumber_trader_strategy_inventory_base{{product_id="{pid}"}} {s_stats["inventory_base"]}'
        ])

    return "\n".join(lines) + "\n"
