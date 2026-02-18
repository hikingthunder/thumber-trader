
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.manager import manager
from app.config import settings
from app.utils.helpers import update_env_file
from app.database.db import AsyncSessionLocal
from app.database.models import Fill, TaxLotMatch, DailyStats
from sqlalchemy import select
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from app.utils.export import export_data, models_to_dicts, map_to_accounting, get_accounting_headers
import io

router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")

@router.get("/config", response_class=HTMLResponse)
async def get_config(request: Request):
    """Render the configuration page."""
    # We pass the current settings to the template
    # Note: sensitive keys like secrets should probably be masked in a real app
    # Safe extraction of secret values for display
    def get_val(v):
        return v.get_secret_value() if hasattr(v, "get_secret_value") else (v or "")

    context = {
        "request": request,
        "settings": settings,
        "api_key_val": get_val(settings.coinbase_api_key)
    }
    return templates.TemplateResponse("config.html", context)

@router.post("/config", response_class=HTMLResponse)
async def update_config(request: Request):
    """Handle configuration updates."""
    form = await request.form()
    updates = {}
    
    # We dynamically map form fields to settings
    # This allows us to handle many settings without a massive if/else block
    for key, value in form.items():
        # Map back to ENV_VAR names (UPPER_CASE)
        env_key = key.upper()
        
        # Skip internal form fields or empty secrets
        if key.startswith("_") or (key.endswith("_secret") and value == "********") or (key.endswith("_token") and value == "********"):
            continue
            
        # Handle special case for coinbase_api_secret (PEM files usually multi-line)
        if key == "coinbase_api_secret" and not value.strip():
            continue

        # For checkboxes, they only appear if checked
        # We handle them separately below for clarity as they need a True/False string
        if key in ["paper_trading_mode", "auto_start", "kelly_allocation_enabled", "liquidity_depth_check_enabled", 
                  "alpha_fusion_enabled", "sentiment_override_enabled", "vpin_enabled", "consensus_pricing_enabled", 
                  "ha_failover_enabled", "atr_enabled", "strategy_stack_enabled", "notifications_enabled"]:
            continue
            
        updates[env_key] = str(value)

    # Handle all expected checkboxes
    checkbox_fields = [
        "paper_trading_mode", "auto_start", "kelly_allocation_enabled", "liquidity_depth_check_enabled", 
        "alpha_fusion_enabled", "sentiment_override_enabled", "vpin_enabled", "consensus_pricing_enabled", 
        "ha_failover_enabled", "atr_enabled", "strategy_stack_enabled", "notifications_enabled"
    ]
    for field in checkbox_fields:
        updates[field.upper()] = "true" if form.get(field) == "on" else "false"
    
    # Save to .env
    try:
        update_env_file(updates)
        message = "Configuration saved! Please restart the bot to apply changes."
        success = True
    except Exception as e:
        message = f"Failed to save configuration: {e}"
        success = False
        
    # Safe extraction of secret values for display
    def get_val(v):
        return v.get_secret_value() if hasattr(v, "get_secret_value") else (v or "")

    context = {
        "request": request,
        "settings": settings, # This will still show old settings until reload
        "api_key_val": get_val(settings.coinbase_api_key),
        "message": message,
        "success": success
    }
    return templates.TemplateResponse("config.html", context)

@router.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Render the main dashboard page."""
    context = {
        "request": request,
        "product_id": settings.product_id,
        "paper_trading": settings.paper_trading_mode
    }
    return templates.TemplateResponse("index.html", context)

@router.get("/api/status")
async def get_api_status():
    """Return JSON status of the bot."""
    return manager.get_status()

@router.get("/dashboard/stats", response_class=HTMLResponse)
async def get_dashboard_stats(request: Request):
    """HTMX partial for stats widget."""
    # Use the new async method for full stats
    status = await manager.get_global_stats()
    
    strategies = status.get("strategies", {})
    balances = status.get("balances", {})
    
    # Simple aggregation for single product
    product_status = strategies.get(settings.product_id, {})
    
    # Calculate total portfolio value roughly (Base + Quote)
    # This is a bit tricky without real-time prices for all assets, 
    # but for this pair we have the price.
    base_currency = settings.product_id.split("-")[0]
    quote_currency = settings.product_id.split("-")[1]
    
    base_bal = balances.get(base_currency, 0)
    quote_bal = balances.get(quote_currency, 0)
    
    import logging
    logging.info(f"Dashboard Stats - Product: {settings.product_id}, Found Bales: {list(balances.keys())}")
    logging.info(f"Extracted: {base_currency}={base_bal}, {quote_currency}={quote_bal}")
    
    # Format for display: 2 decimals for fiat, 8 for crypto
    def fmt_balance(d, symbol):
        val = float(d)
        if symbol in ["USD", "EUR", "GBP", "USDC", "USDT"]:
            return f"{val:.2f}"
        return f"{val:.8f}".rstrip('0').rstrip('.') if val > 0 else "0.00"

    context = {
        "request": request,
        "running": status.get("running", False),
        "orders_count": product_status.get("orders_count", 0),
        "last_price": f"{float(product_status.get('last_price', 0)):.2f}",
        "realized_pnl": f"{float(status.get('total_realized_pnl', 0)):.2f}",
        "unrealized_pnl": f"{float(status.get('total_unrealized_pnl', 0)):.2f}",
        "base_bal": fmt_balance(base_bal, base_currency),
        "quote_bal": fmt_balance(quote_bal, quote_currency),
        "base_currency": base_currency,
        "quote_currency": quote_currency
    }
    return templates.TemplateResponse("partials/stats.html", context)

@router.get("/dashboard/orders", response_class=HTMLResponse)
async def get_orders_table(request: Request):
    """HTMX partial for orders table."""
    strategy = manager.strategies.get(settings.product_id)
    orders = []
    if strategy:
        # Convert dict to list and sort
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

    # Pre-format values for the template to keep it simple and avoid Jinja2 syntax issues
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
    import time
    status = await manager.get_global_stats()
    product_status = status.get("strategies", {}).get(settings.product_id, {})
    return {
        "price": float(product_status.get("last_price", 0)),
        "time": int(time.time())
    }

@router.post("/config/test-notifications", response_class=HTMLResponse)
async def test_notifications(request: Request):
    """Send a test notification."""
    from app.utils.notifications import notify
    try:
        msg = "🧪 *Thumber Trader*: This is a test notification. Your settings are working correctly!"
        # We force notification even if disabled in settings for the test
        await notify(msg, force=True)
        return """<span style="color: #10b981;">✅ Test sent! Check your Telegram/Discord.</span>"""
    except Exception as e:
        return f"""<span style="color: #ef4444;">❌ Failed: {str(e)}</span>"""

@router.get("/export", response_class=HTMLResponse)
async def get_export_page(request: Request):
    """Render the data export page."""
    context = {
        "request": request,
        "product_id": settings.product_id,
        "paper_trading": settings.paper_trading_mode
    }
    return templates.TemplateResponse("export.html", context)

@router.get("/export/{type}/{format}")
async def export_data_endpoint(request: Request, type: str, format: str):
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
            type = "fills" # Reset for helper logic if needed
        elif type == "accounting_tax":
            stmt = select(TaxLotMatch).order_by(TaxLotMatch.created_ts.desc())
            result = await session.execute(stmt)
            models = result.scalars().all()
            raw_data = models_to_dicts(models)
            data = map_to_accounting(raw_data, "tax_matches")
            headers = get_accounting_headers("tax_matches")
            prefix = "accounting_tax_matches"
            type = "tax_matches"
        else:
            raise HTTPException(status_code=400, detail="Invalid export type")

    content, filename, mimetype = export_data(data, headers, prefix, format)
    
    return StreamingResponse(
        io.BytesIO(content if isinstance(content, bytes) else content.encode()),
        media_type=mimetype,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
