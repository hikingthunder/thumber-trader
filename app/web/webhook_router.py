"""TradingView Webhook Inbound — receives alerts and maps them to grid actions."""

import logging
import json
from fastapi import APIRouter, Request, HTTPException
from app.config import settings
from app.core.manager import manager
from app.utils.notifications import notify

logger = logging.getLogger(__name__)
router = APIRouter(tags=["webhooks"])


@router.post("/api/webhooks/tradingview")
async def tradingview_webhook(request: Request):
    """Receive TradingView alerts and map to grid actions.
    
    Expected JSON payload:
    {
        "secret": "<shared_secret>",
        "action": "pause" | "resume" | "adjust_band" | "kill",
        "product_id": "BTC-USD",  (optional, defaults to settings)
        "value": 0.05  (optional, for adjust_band)
    }
    """
    # Validate shared secret
    webhook_secret = settings.tradingview_webhook_secret
    if not webhook_secret:
        raise HTTPException(status_code=403, detail="Webhook not configured")
    
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    secret = body.get("secret", "")
    expected = webhook_secret.get_secret_value() if hasattr(webhook_secret, "get_secret_value") else str(webhook_secret)
    
    if secret != expected:
        logger.warning(f"Webhook auth failed from {request.client.host}")
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    action = body.get("action", "").lower()
    product_id = body.get("product_id", settings.product_id)
    value = body.get("value")
    
    result = {"status": "ok", "action": action}
    
    if action == "pause":
        if product_id in manager.strategies:
            manager.strategies[product_id].paused = True
            msg = f"⏸️ *Webhook*: Strategy paused for {product_id} (TradingView alert)"
            await notify(msg, event_type="trade")
            result["detail"] = f"Paused {product_id}"
        else:
            result["detail"] = f"No active strategy for {product_id}"
    
    elif action == "resume":
        if product_id in manager.strategies:
            manager.strategies[product_id].paused = False
            msg = f"▶️ *Webhook*: Strategy resumed for {product_id} (TradingView alert)"
            await notify(msg, event_type="trade")
            result["detail"] = f"Resumed {product_id}"
        else:
            result["detail"] = f"No active strategy for {product_id}"
    
    elif action == "adjust_band":
        if value is not None:
            from decimal import Decimal
            # This will take effect on next grid re-initialization
            # For now we notify and log — a more complete implementation would
            # update settings dynamically
            msg = f"📐 *Webhook*: Band adjustment signal received. Value: {value} (TradingView alert)"
            await notify(msg, event_type="grid_breach")
            result["detail"] = f"Band adjustment signal logged: {value}"
        else:
            result["detail"] = "Missing 'value' for adjust_band action"
    
    elif action == "kill":
        await manager.emergency_kill()
        msg = "🛑 *Webhook*: EMERGENCY KILL triggered via TradingView alert!"
        await notify(msg, force=True)
        result["detail"] = "Emergency kill executed"
    
    else:
        result["status"] = "error"
        result["detail"] = f"Unknown action: {action}"
    
    logger.info(f"Webhook processed: {action} -> {result.get('detail')}")
    return result
