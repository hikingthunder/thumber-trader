import logging
import io
import json
import time
import psutil
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from sqlalchemy import select, func

from app.core.manager import manager
from app.core.backtest import BacktestEngine
from app.config import settings, Settings
from app.utils.helpers import update_env_file, encrypt_value, decrypt_value, resolve_env_path
from app.database.db import AsyncSessionLocal
from app.database.models import Fill, TaxLotMatch, DailyStats, ConfigVersion
from app.auth.security import get_current_user, log_audit
from app.auth.models import AuditLog
from app.utils.export import export_data, models_to_dicts, map_to_accounting, get_accounting_headers, calculate_fee_summary
from app.utils.notifications import notify

router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")
logger = logging.getLogger(__name__)


# --- Helper ---
def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _require_admin(user) -> None:
    """Enforce admin role for state-changing control endpoints."""
    if getattr(user, "role", None) != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")



def _get_env_text() -> str:
    env_path = resolve_env_path()
    if not env_path.exists():
        return ""
    return env_path.read_text()


def _save_env_text(snapshot: str) -> None:
    env_path = resolve_env_path()
    text = snapshot if snapshot.endswith("\n") or not snapshot else snapshot + "\n"
    env_path.write_text(text)
    try:
        env_path.chmod(0o600)
    except Exception as exc:
        logger.debug(f"Unable to enforce 0600 on .env during save: {exc}")


def _build_change_summary(changed_keys: list[str], rollback_from_id: Optional[int] = None) -> str:
    ordered_keys = sorted(changed_keys)
    preview = ", ".join(ordered_keys[:6])
    suffix = "" if len(ordered_keys) <= 6 else f" (+{len(ordered_keys)-6} more)"
    if rollback_from_id is not None:
        return f"Rollback to config version {rollback_from_id}: {preview}{suffix}"
    return f"Updated {len(ordered_keys)} keys: {preview}{suffix}"


async def _record_config_version(user, changed_keys: list[str], env_snapshot: str, rollback_from_id: Optional[int] = None):
    actor_name = getattr(user, "username", None) or f"user-{getattr(user, 'id', 'unknown')}"
    encrypted_snapshot = encrypt_value(env_snapshot)
    record = ConfigVersion(
        created_ts=time.time(),
        actor_user_id=getattr(user, "id", None),
        actor_username=actor_name,
        change_summary=_build_change_summary(changed_keys, rollback_from_id=rollback_from_id),
        changed_keys=json.dumps(sorted(changed_keys)),
        env_snapshot=encrypted_snapshot,
        rollback_from_id=rollback_from_id,
    )
    async with AsyncSessionLocal() as session:
        session.add(record)
        await session.commit()


async def _get_recent_config_versions(limit: int = 12) -> list[ConfigVersion]:
    async with AsyncSessionLocal() as session:
        stmt = select(ConfigVersion).order_by(ConfigVersion.created_ts.desc()).limit(limit)
        result = await session.execute(stmt)
        rows = list(result.scalars().all())
        for row in rows:
            row.created_at_label = datetime.fromtimestamp(row.created_ts).strftime("%Y-%m-%d %H:%M:%S")
        return rows


# --- Dashboard ---
@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request, user=Depends(get_current_user)):
    """Render the main dashboard page."""
    context = {
        "request": request,
        "product_id": settings.product_id,
        "paper_trading": settings.is_simulated_execution(),
        "execution_mode": settings.normalized_execution_mode(),
        "user": user
    }
    return templates.TemplateResponse("index.html", context)


# --- Configuration ---
@router.get("/config", response_class=HTMLResponse)
async def get_config(request: Request, user=Depends(get_current_user)):
    """Render the configuration page."""
    # Ensure coinbase_api_key is treated correctly for the password field
    api_key_val = settings.coinbase_api_key
    if hasattr(api_key_val, "get_secret_value"):
        api_key_val = api_key_val.get_secret_value()
        
    recommended_defaults = {
        name: (str(field.default) if field.default is not None else "")
        for name, field in Settings.model_fields.items()
        if field.default is not None
    }
    context = {
        "request": request,
        "settings": settings,
        "api_key_val": api_key_val or "",
        "config_versions": await _get_recent_config_versions(),
        "recommended_defaults": recommended_defaults,
        "paper_trading": settings.is_simulated_execution(),
        "user": user
    }
    return templates.TemplateResponse("config.html", context)


@router.post("/config/save")
async def save_config(request: Request, user=Depends(get_current_user)):
    """Save updated configuration to .env file."""
    _require_admin(user)
    form_data = await request.form()
    updates = {k.upper(): v for k, v in form_data.items() if v and k != "csrf_token"}

    execution_mode = str(form_data.get("execution_mode", "")).strip().lower()
    if execution_mode in {"live", "paper", "shadow_live"}:
        updates["EXECUTION_MODE"] = execution_mode

    bool_fields = {
        "paper_trading_mode",
        "auto_start",
        "kelly_allocation_enabled",
        "liquidity_depth_check_enabled",
        "alpha_fusion_enabled",
        "sentiment_override_enabled",
        "vpin_enabled",
        "dynamic_rebalancing_enabled",
        "strategy_stack_enabled",
        "notifications_enabled",
        "notify_on_trade",
        "notify_on_grid_breach",
        "notify_on_error",
        "notify_on_daily_summary",
        "session_timeout_enabled"
    }
    for field in bool_fields:
        updates[field.upper()] = "true" if field in form_data else "false"

    if execution_mode in {"paper", "shadow_live"}:
        updates["PAPER_TRADING_MODE"] = "true"
    elif execution_mode == "live":
        updates["PAPER_TRADING_MODE"] = "false"
    
    try:
        success = update_env_file(updates)
    except PermissionError:
        logger.exception("Configuration save failed due to env file permissions")
        env_path = resolve_env_path()
        return HTMLResponse(
            f"<div class='alert alert-danger'>Failed to save configuration: permission denied for {env_path}. Update file ownership/permissions for the runtime user.</div>",
            status_code=500,
        )
    except Exception as exc:
        logger.exception("Configuration save failed due to unexpected error")
        return HTMLResponse(
            f"<div class='alert alert-danger'>Failed to save configuration: {type(exc).__name__}. Check server logs for details.</div>",
            status_code=500,
        )
    if success:
        env_snapshot = _get_env_text()
        await _record_config_version(user, list(updates.keys()), env_snapshot)
        await log_audit(user.id, "config_change", f"Updated keys: {list(updates.keys())}", _client_ip(request))
        return HTMLResponse("<div class='alert alert-success'>Configuration saved and versioned. Restart required.</div>")
    return HTMLResponse("<div class='alert alert-danger'>Failed to save configuration.</div>", status_code=500)




@router.post("/config/rollback")
async def rollback_config(request: Request, user=Depends(get_current_user)):
    """Restore a previous .env snapshot from config version history."""
    _require_admin(user)
    form_data = await request.form()
    target_id = form_data.get("version_id")
    if not target_id:
        return HTMLResponse("<div class='alert alert-danger'>Missing version_id.</div>", status_code=400)

    try:
        version_id = int(target_id)
    except (TypeError, ValueError):
        return HTMLResponse("<div class='alert alert-danger'>Invalid version_id.</div>", status_code=400)

    async with AsyncSessionLocal() as session:
        stmt = select(ConfigVersion).where(ConfigVersion.id == version_id)
        result = await session.execute(stmt)
        target = result.scalar_one_or_none()

    if target is None:
        return HTMLResponse("<div class='alert alert-danger'>Config version not found.</div>", status_code=404)

    snapshot_text = decrypt_value(target.env_snapshot)
    _save_env_text(snapshot_text)
    env_snapshot = _get_env_text()
    changed_keys = json.loads(target.changed_keys) if target.changed_keys else []
    await _record_config_version(user, changed_keys, env_snapshot, rollback_from_id=target.id)
    await log_audit(user.id, "config_rollback", f"Restored config version #{target.id}", _client_ip(request))
    return HTMLResponse("<div class='alert alert-success'>Configuration rollback applied. Restart required.</div>")


@router.post("/config/test-notifications")
async def test_notifications(request: Request, user=Depends(get_current_user)):
    """Send a test notification across currently configured channels."""
    _require_admin(user)
    await notify("✅ Thumber Trader notification test: channel connectivity verified.", force=True)
    await log_audit(user.id, "notification_test", "Triggered test notification dispatch", _client_ip(request))
    return HTMLResponse("<div class='alert alert-success'>Notification test dispatched to configured channels.</div>")


# --- Backtesting ---
@router.get("/backtest", response_class=HTMLResponse)
async def get_backtest(request: Request, user=Depends(get_current_user)):
    """Render the backtesting page."""
    context = {
        "request": request,
        "settings": settings,
        "paper_trading": settings.is_simulated_execution(),
        "user": user
    }
    return templates.TemplateResponse("backtest.html", context)


@router.post("/backtest/run")
async def run_backtest(request: Request, user=Depends(get_current_user)):
    """Execute a backtest."""
    _require_admin(user)
    form_data = await request.form()
    engine = BacktestEngine(
        product_id=form_data.get("product_id", settings.product_ids.split(',')[0]),
        start_date=form_data.get("start_date"),
        end_date=form_data.get("end_date"),
        initial_capital=float(form_data.get("initial_usd", 1000))
    )
    report = await engine.run()
    await log_audit(user.id, "backtest_run", f"Product: {engine.product_id}", _client_ip(request))
    return templates.TemplateResponse("partials/backtest_results.html", {
        "request": request, 
        "report": report,
        "trades": engine.trades[-20:], # Show last 20 trades
        "pnl_history": engine.pnl_history
    })


@router.get("/dashboard/stats", response_class=HTMLResponse)
async def get_dashboard_stats(request: Request, user=Depends(get_current_user)):
    """HTMX partial for stats widget including advanced metrics."""
    stats = await manager.get_global_stats()
    
    # Calculate total fees from all fills
    async with AsyncSessionLocal() as session:
        fee_stmt = select(func.sum(Fill.fee_paid))
        fee_result = await session.execute(fee_stmt)
        total_fees = float(fee_result.scalar() or 0)
    
    # Extract first strategy stats
    strat_id = next(iter(stats["strategies"]), None)
    s = stats["strategies"].get(strat_id, {}) if strat_id else {}
    
    balances = stats.get("balances", {})
    product_symbol = (strat_id or settings.product_id).split("-")[0]
    quote_symbol = (strat_id or settings.product_id).split("-")[-1]
    context = {
        "request": request,
        "running": stats.get("running", False),
        "product_id": strat_id,
        "stats": s,
        "total_realized_pnl": stats.get("total_realized_pnl", 0),
        "total_unrealized_pnl": stats.get("total_unrealized_pnl", 0),
        "total_fees": total_fees,
        "execution_mode": (s.get("execution_mode") if s else settings.normalized_execution_mode()),
        "base_balance": float(balances.get(product_symbol, 0) or 0),
        "quote_balance": float(balances.get(quote_symbol, 0) or 0),
        "last_price": float(s.get("last_price", 0) or 0),
        "exchange_health": manager.get_exchange_health(),
    }
    return templates.TemplateResponse("partials/stats.html", context)


@router.get("/dashboard/orders", response_class=HTMLResponse)
async def get_orders_table(request: Request, user=Depends(get_current_user)):
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
async def get_fills_table(request: Request, user=Depends(get_current_user)):
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


@router.get("/dashboard/signals", response_class=HTMLResponse)
async def get_dashboard_signals(request: Request, user=Depends(get_current_user)):
    """HTMX partial for trading signals."""
    from app.core.analysis import rsi, macd_histogram, vpin
    
    strategy = manager.strategies.get(settings.product_id)
    if not strategy or not strategy.candles:
        return "<p style='color: var(--text-muted);'>Insufficient data for analysis</p>"
        
    prices = [c[3] for c in strategy.candles]  # Closes
    rsi_val = rsi(prices, 14)
    macd_val = macd_histogram(prices, 12, 26, 9)
    current_vpin = vpin(strategy.candles, 50)
    
    def get_color(val, neutral=50, threshold=20):
        if val > neutral + threshold: return "var(--success-color)"
        if val < neutral - threshold: return "var(--danger-color)"
        return "var(--text-muted)"

    return f"""
    <div style="display: grid; gap: 10px;">
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 0.85rem;">RSI (14)</span>
            <span class="font-mono" style="color: {get_color(float(rsi_val), 50, 20)}; font-weight: 700;">{rsi_val:.2f}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 0.85rem;">MACD Hist</span>
            <span class="font-mono" style="color: {'var(--success-color)' if macd_val > 0 else 'var(--danger-color)'}; font-weight: 700;">{macd_val:.4f}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 0.85rem;">Toxicity (VPIN)</span>
            <span class="font-mono" style="color: {'var(--danger-color)' if current_vpin > 0.7 else 'var(--success-color)'}; font-weight: 700;">{current_vpin*100:.1f}%</span>
        </div>
    </div>
    """

@router.get("/dashboard/performance-mini", response_class=HTMLResponse)
async def get_performance_mini(request: Request, user=Depends(get_current_user)):
    """HTMX partial for performance summary."""
    async with AsyncSessionLocal() as session:
        cutoff = time.time() - (7 * 86400)
        stmt = select(DailyStats).where(DailyStats.ts > cutoff).order_by(DailyStats.ts.desc())
        result = await session.execute(stmt)
        stats = result.scalars().all()
        
    total_pnl = sum(float(s.pnl_per_1k) for s in stats)
    trade_count = len(stats)
    
    return f"""
    <div style="display: grid; gap: 8px;">
        <div style="font-size: 1.5rem; font-weight: 800; color: {'var(--success-color)' if total_pnl >= 0 else 'var(--danger-color)'};">
            {'+' if total_pnl > 0 else ''}${total_pnl:.2f}
            <span style="font-size: 0.8rem; color: var(--text-muted); font-weight: 500;">(7d)</span>
        </div>
        <div style="color: var(--text-muted); font-size: 0.85rem; font-weight: 600;">
            Recent Performance Trend
        </div>
    </div>
    """

@router.get("/dashboard/health-stats")
async def get_health_stats(user=Depends(get_current_user)):
    """Unified JSON for system health card."""
    async with AsyncSessionLocal() as session:
        audit_count_stmt = select(func.count(AuditLog.id))
        audit_result = await session.execute(audit_count_stmt)
        audit_count = audit_result.scalar() or 0
        
    return {
        "load": psutil.getloadavg()[0],
        "ram": psutil.virtual_memory().percent,
        "audit_count": audit_count,
        "latency": int(time.time() * 1000) % 50,
    }


@router.get("/dashboard/performance-data")
async def get_performance_data(user=Depends(get_current_user)):
    """Return JSON data for 24h PnL chart."""
    async with AsyncSessionLocal() as session:
        cutoff = time.time() - 86400
        stmt = select(DailyStats).where(DailyStats.ts > cutoff).order_by(DailyStats.ts.asc())
        result = await session.execute(stmt)
        stats = result.scalars().all()
        
    return {
        "labels": [datetime.fromtimestamp(s.ts).strftime("%H:%M") for s in stats],
        "values": [float(s.pnl_per_1k) for s in stats]
    }


@router.get("/dashboard/audit-events", response_class=HTMLResponse)
async def get_recent_audit_events(request: Request, user=Depends(get_current_user)):
    """HTMX partial for last 3 security events."""
    async with AsyncSessionLocal() as session:
        stmt = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(3)
        result = await session.execute(stmt)
        logs = result.scalars().all()
        
    if not logs:
        return "<p style='color: var(--text-muted); font-size: 0.8rem;'>No recent security events</p>"
        
    items = []
    for l in logs:
        ts = datetime.fromtimestamp(l.timestamp).strftime("%H:%M:%S")
        items.append(f"""
        <div style="font-size: 0.75rem; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
            <span style="color: var(--accent-color);">{ts}</span>: 
            <span style="color: var(--text-color);">{l.action}</span>
            <div style="color: var(--text-muted); font-size: 0.7rem;">{l.ip_address}</div>
        </div>
        """)
    return "".join(items)




@router.get("/dashboard/depth")
async def get_dashboard_depth(user=Depends(get_current_user)):
    """Return latest cached order-book depth payload for fallback polling."""
    payload = manager.last_depth_payload.get(settings.product_id)
    if payload:
        return payload
    return {"type": "depth", "product_id": settings.product_id, "density": [], "vpin": 0.0, "time": 0}


@router.get("/dashboard/price")
async def get_dashboard_price(user=Depends(get_current_user)):
    """Return latest price for the chart."""
    status = await manager.get_global_stats()
    product_status = status.get("strategies", {}).get(settings.product_id, {})
    return {
        "price": float(product_status.get("last_price", 0)),
        "time": int(time.time())
    }


@router.post("/dashboard/control/{action}")
async def control_bot(action: str, request: Request, user=Depends(get_current_user)):
    """Start or stop the trading engine."""
    _require_admin(user)
    success = False
    if action == "start":
        success = await manager.start()
    elif action == "stop":
        success = await manager.stop()
    
    if success:
        await log_audit(user.id, f"bot_{action}", f"Engine {action}ed manually", _client_ip(request))
    
    return await get_dashboard_stats(request)


# --- Data Export ---
@router.get("/export", response_class=HTMLResponse)
async def get_export_view(request: Request, user=Depends(get_current_user)):
    """Render the data export page."""
    context = {
        "request": request,
        "user": user,
        "product_id": settings.product_id,
        "paper_trading": settings.is_simulated_execution(),
        "execution_mode": settings.normalized_execution_mode(),
        "formats": ["csv", "xlsx", "ods"],
        "tax_methods": ["FIFO", "LIFO", "HIFO"]
    }
    return templates.TemplateResponse("export.html", context)


@router.get("/dashboard/exchange-health")
async def get_exchange_health(user=Depends(get_current_user)):
    """Expose current exchange connectivity metadata for operational visibility."""
    return manager.get_exchange_health()


@router.post("/export/generate")
async def generate_export(request: Request, user=Depends(get_current_user)):
    """Generate and download execution data."""
    form_data = await request.form()
    file_format = form_data.get("format", "csv").lower()
    tax_method = form_data.get("tax_method", "FIFO")
    
    async with AsyncSessionLocal() as session:
        # Fetch data
        fills_stmt = select(Fill).order_by(Fill.ts.asc())
        fills_res = await session.execute(fills_stmt)
        fills = fills_res.scalars().all()
        
        matches_stmt = select(TaxLotMatch).order_by(TaxLotMatch.created_ts.asc())
        matches_res = await session.execute(matches_stmt)
        matches = matches_res.scalars().all()
        
        stats_stmt = select(DailyStats).order_by(DailyStats.ts.asc())
        stats_res = await session.execute(stats_stmt)
        stats = stats_res.scalars().all()

        # Convert to dicts for export
        data = {
            "Fills": models_to_dicts(fills),
            "Tax_Lots": models_to_dicts(matches),
            "Daily_Performance": models_to_dicts(stats),
            "Fee_Summary": calculate_fee_summary(fills)
        }

    output = export_data(data, file_format)
    if isinstance(output, str):
        output = output.encode("utf-8")
    await log_audit(user.id, "data_export", f"Format: {file_format}, Method: {tax_method}", _client_ip(request))
    
    media_types = {
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "ods": "application/vnd.oasis.opendocument.spreadsheet"
    }
    
    filename = f"thumber_trader_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_format}"
    return StreamingResponse(
        io.BytesIO(output),
        media_type=media_types.get(file_format, "application/octet-stream"),
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
