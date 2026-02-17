
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database.db import init_db
from app.core.manager import manager
from app.web.router import router as web_router

# Setup Logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application starting up...")
    
    # Initialize Database
    await init_db()
    
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
    description="High-frequency grid trading bot with FastAPI and HTMX",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(web_router)

# Mount Static if we had any (placeholder)
# app.mount("/static", StaticFiles(directory="app/static"), name="static")

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
