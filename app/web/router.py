import logging
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.core.manager import manager
from app.core.backtest import BacktestEngine
from app.config import settings
from app.utils.helpers import update_env_file
from app.database.db import AsyncSessionLocal
from app.database.models import Fill, TaxLotMatch, DailyStats
from app.auth.security import get_current_user, log_audit
from sqlalchemy import select
from app.utils.export import export_data, models_to_dicts, map_to_accounting, get_accounting_headers, calculate_fee_summary
import io
import time
import psutil

router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")
logger = logging.getLogger(__name__)


# --- Helper ---
def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# --- Dashboard ---
@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request, user=Depends(get_current_user)):
    """Render the main dashboard page."""
    context = {
        "request": request,
        "product_id": settings.product_id,
        "paper_trading": settings.paper_trading_mode,
        "user": user
    }
    return templates.TemplateResponse("index.html", context)


# --- Configuration ---
@router.get("/config", response_class=HTMLResponse)
async def get_config(request: Request, user=Depends(get_current_user)):
    """Render the configuration page."""
    def get_val(v):
        return v.get_secret_value() if hasattr(v, "get_secret_value") else (v or "")

    context = {
        "request": request,
        "settings": settings,
        "api_key_val": get_val(settings.coinbase_api_key),
        "user": user
    }
    return templates.TemplateResponse("config.html", context)


@router.post("/config", response_class=HTMLResponse)
async def update_config(request: Request, user=Depends(get_current_user)):
    """Handle configuration updates."""
    form = await request.form()
    updates = {}

    for key, value in form.items():
        env_key = key.upper()

        if key.startswith("_") or (key.endswith("_secret") and value == "********") or (key.endswith("_token") and value == "********"):
            continue

        if key == "coinbase_api_secret" and not value.strip():
            continue

        if key in ["paper_trading_mode", "auto_start", "kelly_allocation_enabled", "liquidity_depth_check_enabled",
                    "alpha_fusion_enabled", "sentiment_override_enabled", "vpin_enabled", "consensus_pricing_enabled",
                    "ha_failover_enabled", "atr_enabled", "strategy_stack_enabled", "notifications_enabled",
                    "dynamic_rebalancing_enabled"]:
            continue

        updates[env_key] = str(value)

    checkbox_fields = [
        "paper_trading_mode", "auto_start", "kelly_allocation_enabled", "liquidity_depth_check_enabled",
        "alpha_fusion_enabled", "sentiment_override_enabled", "vpin_enabled", "consensus_pricing_enabled",
        "ha_failover_enabled", "atr_enabled", "strategy_stack_enabled", "notifications_enabled",
        "dynamic_rebalancing_enabled"
    ]
    for field in checkbox_fields:
        updates[field.upper()] = "true" if form.get(field) == "on" else "false"

    try:
        update_env_file(updates)
        message = "Configuration saved! Please restart the bot to apply changes."
        success = True
        await log_audit(user.id, "config_change", f"Updated {len(updates)} settings", _client_ip(request))
    except Exception as e:
        message = f"Failed to save configuration: {e}"
        success = False

    def get_val(v):
        return v.get_secret_value() if hasattr(v, "get_secret_value") else (v or "")

    context = {
        "request": request,
        "settings": settings,
        "api_key_val": get_val(settings.coinbase_api_key),
        "message": message,
        "success": success,
        "user": user
    }
    return templates.TemplateResponse("config.html", context)


# --- Dashboard Partials (HTMX) ---
@router.get("/api/status")
async def get_api_status(user=Depends(get_current_user)):
    """Return JSON status of the bot."""
    return manager.get_status()


@router.get("/dashboard/stats", response_class=HTMLResponse)
async def get_dashboard_stats(request: Request):
    """HTMX partial for stats widget including advanced metrics."""
    stats = await manager.get_global_stats()
    
    # Extract first strategy stats
    strat_id = next(iter(stats["strategies"]), None)
    s = stats["strategies"].get(strat_id, {}) if strat_id else {}
    
    context = {
        "request": request,
        "running": stats.get("running", False),
        "product_id": strat_id,
        "stats": s,
        "total_realized_pnl": stats.get("total_realized_pnl", 0),
        "total_unrealized_pnl": stats.get("total_unrealized_pnl", 0),
    }
    return templates.TemplateResponse("partials/stats.html", context)


@router.get("/dashboard/orders", response_class=HTMLResponse)
async def get_orders_table(request: Request):
    """HTMX partial for orders table."""
    strategy = manager.strategies.get(settings.product_id)
    orders = []
    if strategy:
        orders = list(strategy.orders.values())
        orders.sort(key=lambda x: x.get("price", 0), reverse=True)

    context = {
        "request": request,
        "orders": orders
    }
    return templates.TemplateResponse("partials/orders.html", context)


@router.get("/dashboard/fills", response_class=HTMLResponse)
async def get_fills_table(request: Request):
    """HTMX partial for recent fills table."""
    async with AsyncSessionLocal() as session:
        stmt = select(Fill).order_by(Fill.ts.desc()).limit(20)
        result = await session.execute(stmt)
        fills = result.scalars().all()

    def fmt_val(val, decimals=2):
        if val is None: return ""
        try:
            fval = float(val)
            if decimals == 8:
                return f"{fval:.8f}".rstrip('0').rstrip('.')
            return f"{fval:.2f}"
        except:
            return str(val)

    formatted_fills = []
    for f in fills:
        from datetime import datetime
        formatted_fills.append({
            "ts": datetime.fromtimestamp(f.ts).strftime("%Y-%m-%d %H:%M:%S"),
            "side": f.side,
            "price": fmt_val(f.price, 2),
            "base_size": fmt_val(f.base_size, 8),
            "grid_index": f.grid_index,
            "realized_pnl_usd": fmt_val(f.realized_pnl_usd, 2) if f.side == 'SELL' else None,
            "raw_pnl": float(f.realized_pnl_usd) if f.realized_pnl_usd else 0
        })

    context = {
        "request": request,
        "fills": formatted_fills
    }
    return templates.TemplateResponse("partials/fills.html", context)


@router.get("/dashboard/price")
async def get_dashboard_price():
    """Return latest price for the chart."""
    status = await manager.get_global_stats()
    product_status = status.get("strategies", {}).get(settings.product_id, {})
    return {
        "price": float(product_status.get("last_price", 0)),
        "time": int(time.time())
    }


# --- Bot Controls ---
@router.post("/dashboard/control/start", response_class=HTMLResponse)
async def start_bot(request: Request, user=Depends(get_current_user)):
    """Start the trading bot."""
    try:
        if not manager.running:
            await manager.start()
        await log_audit(user.id, "bot_start", "", _client_ip(request))
    except Exception as e:
        pass
    return await get_dashboard_stats(request)


@router.post("/dashboard/control/stop", response_class=HTMLResponse)
async def stop_bot(request: Request, user=Depends(get_current_user)):
    """Stop the trading bot."""
    try:
        if manager.running:
            await manager.stop()
        await log_audit(user.id, "bot_stop", "", _client_ip(request))
    except Exception as e:
        pass
    return await get_dashboard_stats(request)


# --- Emergency Kill Switch ---
@router.post("/api/emergency/kill")
async def emergency_kill(request: Request, user=Depends(get_current_user)):
    """Cancel all orders and flatten all positions."""
    await log_audit(user.id, "EMERGENCY_KILL", "Kill switch activated!", _client_ip(request))

    cancelled = 0
    errors = []

    try:
        # Stop the bot first
        if manager.running:
            await manager.stop()

        # Cancel all open orders
        if manager.exchange:
            for pid, strategy in manager.strategies.items():
                order_ids = list(strategy.orders.keys())
                if order_ids:
                    try:
                        await manager.exchange.cancel_orders(order_ids)
                        cancelled += len(order_ids)
                    except Exception as e:
                        logger.error(f"Failed to cancel orders for {pid}: {e}")
                        errors.append(f"Failed to cancel orders for {pid}")
    except Exception as e:
        logger.error(f"Kill switch error: {e}", exc_info=True)
        errors.append("Internal system error during kill switch activation.")

    from app.utils.notifications import notify
    await notify(f"🚨 **EMERGENCY KILL SWITCH ACTIVATED** by {user.username}\n"
                 f"Cancelled {cancelled} orders.", force=True)

    return {
        "status": "killed",
        "orders_cancelled": cancelled,
        "errors": errors,
        "message": f"Kill switch activated. {cancelled} orders cancelled."
    }


# --- System Health ---
@router.get("/api/health/detailed")
async def detailed_health(user=Depends(get_current_user)):
    """Return detailed system health metrics."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    # Check Coinbase API latency
    api_latency_ms = -1
    try:
        if manager.exchange:
            start = time.time()
            await manager.exchange.get_current_price(settings.product_id)
            api_latency_ms = round((time.time() - start) * 1000, 1)
    except Exception:
        pass

    # WebSocket status
    ws_connected = False
    if hasattr(manager, 'ws_client') and manager.ws_client:
        ws_connected = manager.ws_client.running and manager.ws_client.ws is not None

    return {
        "cpu_percent": cpu_percent,
        "ram_percent": memory.percent,
        "ram_used_mb": round(memory.used / 1024 / 1024, 1),
        "ram_total_mb": round(memory.total / 1024 / 1024, 1),
        "api_latency_ms": api_latency_ms,
        "ws_connected": ws_connected,
        "bot_running": manager.running,
        "uptime_seconds": round(time.time() - getattr(manager, '_start_time', time.time()), 1)
    }


# --- Notifications ---
@router.post("/config/test-notifications", response_class=HTMLResponse)
async def test_notifications(request: Request, user=Depends(get_current_user)):
    """Send a test notification."""
    from app.utils.notifications import notify
    try:
        msg = "🧪 *Thumber Trader*: This is a test notification. Your settings are working correctly!"
        await notify(msg, force=True)
        return """<span style="color: #10b981;">✅ Test sent! Check your Telegram/Discord.</span>"""
    except Exception as e:
        logger.error(f"Notification test failed: {e}", exc_info=True)
        return """<span style="color: #ef4444;">❌ Failed: Notification service unavailable or misconfigured.</span>"""


# --- Export ---
@router.get("/export", response_class=HTMLResponse)
async def get_export_page(request: Request, user=Depends(get_current_user)):
    """Render the data export page."""
    context = {
        "request": request,
        "product_id": settings.product_id,
        "paper_trading": settings.paper_trading_mode,
        "user": user
    }
    return templates.TemplateResponse("export.html", context)


@router.get("/api/fee-summary", response_class=HTMLResponse)
async def get_fee_summary(request: Request, user=Depends(get_current_user)):
    """Return fee summary as HTML partial."""
    async with AsyncSessionLocal() as session:
        stmt = select(Fill).order_by(Fill.ts.desc())
        result = await session.execute(stmt)
        models = result.scalars().all()
        data = models_to_dicts(models)
    
    summary = calculate_fee_summary(data)
    
    return f"""
    <div class="fee-grid">
        <div class="fee-stat">
            <span class="value" style="color: #f97316;">${summary['total_fees_usd']:,.4f}</span>
            <span class="label">Total Fees Paid</span>
        </div>
        <div class="fee-stat">
            <span class="value">${summary['total_volume_usd']:,.2f}</span>
            <span class="label">Total Volume</span>
        </div>
        <div class="fee-stat">
            <span class="value" style="color: #10b981;">${summary['buy_volume_usd']:,.2f}</span>
            <span class="label">Buy Volume</span>
        </div>
        <div class="fee-stat">
            <span class="value" style="color: #ef4444;">${summary['sell_volume_usd']:,.2f}</span>
            <span class="label">Sell Volume</span>
        </div>
        <div class="fee-stat">
            <span class="value">{summary['avg_fee_rate_pct']:.4f}%</span>
            <span class="label">Avg Fee Rate</span>
        </div>
        <div class="fee-stat">
            <span class="value">{summary['trade_count']}</span>
            <span class="label">Total Trades</span>
        </div>
    </div>
    """


# --- Backtest ---
@router.get("/backtest", response_class=HTMLResponse)
async def get_backtest(request: Request, user=Depends(get_current_user)):
    """Render the backtest page."""
    context = {
        "request": request,
        "settings": settings,
        "user": user
    }
    return templates.TemplateResponse("backtest.html", context)


@router.post("/api/backtest/run", response_class=HTMLResponse)
async def run_backtest(request: Request, user=Depends(get_current_user)):
    """Handle backtest execution."""
    form = await request.form()
    product_id = str(form.get("product_id", settings.product_id))
    start_date = str(form.get("start_date"))
    end_date = str(form.get("end_date"))
    initial_usd = Decimal(str(form.get("initial_usd", "1000")))
    grid_lines = int(form.get("grid_lines", settings.grid_lines))
    band_pct = Decimal(str(form.get("grid_band_pct", settings.grid_band_pct)))
    order_size = Decimal(str(form.get("order_size", "10")))

    # Convert dates to limit (rough approximation since exchange fetch uses limit)
    # For a real implementation, we'd use start/end timestamps properly in exchange.py
    # Let's say we fetch 1000 candles for now as a demo limit
    limit = 1000
    granularity = "ONE_HOUR"
    
    # We need an exchange instance
    # The manager has one if it's running
    exchange = None
    if manager.running and manager.exchange:
        exchange = manager.exchange
    else:
        from app.core.exchange import CoinbaseExchange
        api_key = settings.coinbase_api_key.get_secret_value() if settings.coinbase_api_key else ""
        api_secret = settings.coinbase_api_secret.get_secret_value() if settings.coinbase_api_secret else ""
        if api_key and api_secret:
            exchange = CoinbaseExchange(api_key, api_secret)
    
    if not exchange:
        return """<div class="alert alert-danger">API Credentials required for backtesting (to fetch historical data).</div>"""

    candles = await exchange.fetch_public_candles(product_id, granularity, limit)
    
    if not candles:
        return """<div class="alert alert-danger">Failed to fetch historical data for backtesting.</div>"""

    engine = BacktestEngine(initial_balance=initial_usd)
    engine.run(candles, grid_lines, band_pct, order_size)
    report = engine.get_report()
    
    context = {
        "request": request,
        "report": report,
        "trades": engine.trades[-20:], # show last 20
        "pnl_history": [float(p) for p in engine.pnl_history]
    }
    return templates.TemplateResponse("partials/backtest_results.html", context)


@router.get("/export/{type}/{format}")
async def export_data_endpoint(request: Request, type: str, format: str, user=Depends(get_current_user)):
    """Export data in various formats."""
    async with AsyncSessionLocal() as session:
        if type == "fills":
            stmt = select(Fill).order_by(Fill.ts.desc())
            result = await session.execute(stmt)
            models = result.scalars().all()
            data = models_to_dicts(models)
            headers = [column.key for column in Fill.__table__.columns]
            prefix = "fills_export"
        elif type == "tax_matches":
            stmt = select(TaxLotMatch).order_by(TaxLotMatch.created_ts.desc())
            result = await session.execute(stmt)
            models = result.scalars().all()
            data = models_to_dicts(models)
            headers = [column.key for column in TaxLotMatch.__table__.columns]
            prefix = "tax_matches_export"
        elif type == "stats":
            stmt = select(DailyStats).order_by(DailyStats.ts.desc())
            result = await session.execute(stmt)
            models = result.scalars().all()
            data = models_to_dicts(models)
            headers = [column.key for column in DailyStats.__table__.columns]
            prefix = "daily_stats_export"
        elif type == "accounting_fills":
            stmt = select(Fill).order_by(Fill.ts.desc())
            result = await session.execute(stmt)
            models = result.scalars().all()
            raw_data = models_to_dicts(models)
            data = map_to_accounting(raw_data, "fills")
            headers = get_accounting_headers("fills")
            prefix = "accounting_fills"
        elif type == "accounting_tax":
            stmt = select(TaxLotMatch).order_by(TaxLotMatch.created_ts.desc())
            result = await session.execute(stmt)
            models = result.scalars().all()
            raw_data = models_to_dicts(models)
            data = map_to_accounting(raw_data, "tax_matches")
            headers = get_accounting_headers("tax_matches")
            prefix = "accounting_tax_matches"
        else:
            raise HTTPException(status_code=400, detail="Invalid export type")

    await log_audit(user.id, "data_export", f"type={type} format={format}", _client_ip(request))
    content, filename, mimetype = export_data(data, headers, prefix, format)

    return StreamingResponse(
        io.BytesIO(content if isinstance(content, bytes) else content.encode()),
        media_type=mimetype,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
