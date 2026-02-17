
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
