import logging
import json
import urllib.request
import urllib.parse
import asyncio
from typing import Optional
from app.config import settings

logger = logging.getLogger(__name__)

def get_secret(v):
    """Safely extract secret value if it's a SecretStr, otherwise return as is."""
    return v.get_secret_value() if hasattr(v, "get_secret_value") else v

async def send_telegram_message(message: str) -> bool:
    """Send a message via Telegram Bot API."""
    token = get_secret(settings.telegram_bot_token)
    chat_id = get_secret(settings.telegram_chat_id)

    if not token or not chat_id:
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    def _send():
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ThumberTraderBot/1.0"
        }
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.getcode() == 200

    try:
        return await asyncio.to_thread(_send)
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

async def send_discord_message(message: str) -> bool:
    """Send a message via Discord Webhook."""
    webhook_url = get_secret(settings.discord_webhook_url)
    if not webhook_url:
        return False
        
    payload = {
        "content": message
    }
    
    def _send():
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ThumberTraderBot/1.0"
        }
        req = urllib.request.Request(webhook_url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.getcode() in [200, 204]

    try:
        return await asyncio.to_thread(_send)
    except Exception as e:
        logger.error(f"Failed to send Discord message: {e}")
        return False

async def send_slack_message(message: str) -> bool:
    """Send a message via Slack Incoming Webhook."""
    webhook_url = get_secret(settings.slack_webhook_url) if settings.slack_webhook_url else None
    if not webhook_url:
        return False
    
    payload = {
        "text": message,
        "username": "Thumber Trader",
        "icon_emoji": ":chart_with_upwards_trend:"
    }
    
    def _send():
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ThumberTraderBot/2.0"
        }
        req = urllib.request.Request(webhook_url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.getcode() == 200
    
    try:
        return await asyncio.to_thread(_send)
    except Exception as e:
        logger.error(f"Failed to send Slack message: {e}")
        return False

async def send_pagerduty_alert(message: str, severity: str = "warning") -> bool:
    """Send an alert via PagerDuty Events API v2."""
    routing_key = get_secret(settings.pagerduty_routing_key) if settings.pagerduty_routing_key else None
    if not routing_key:
        return False
    
    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": message[:1024],  # PD max summary length
            "source": "thumber-trader",
            "severity": severity,  # critical, error, warning, info
            "component": "trading-bot",
            "custom_details": {
                "product_id": settings.product_id,
                "paper_mode": settings.paper_trading_mode
            }
        }
    }
    
    url = "https://events.pagerduty.com/v2/enqueue"
    
    def _send():
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ThumberTraderBot/2.0"
        }
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.getcode() == 202
    
    try:
        return await asyncio.to_thread(_send)
    except Exception as e:
        logger.error(f"Failed to send PagerDuty alert: {e}")
        return False

async def notify(message: str, force: bool = False, event_type: Optional[str] = None):
    """Dispatch message to all enabled notification channels.
    
    event_type can be: 'trade', 'grid_breach', 'error', 'daily_summary'
    If event_type is provided, only dispatch if that event type toggle is enabled.
    """
    if not settings.notifications_enabled and not force:
        return

    # Granular event filtering
    if event_type and not force:
        event_toggles = {
            "trade": getattr(settings, "notify_on_trade", True),
            "grid_breach": getattr(settings, "notify_on_grid_breach", True),
            "error": getattr(settings, "notify_on_error", True),
            "daily_summary": getattr(settings, "notify_on_daily_summary", True),
        }
        if not event_toggles.get(event_type, True):
            return  # This event type is disabled

    tasks = []
    if settings.telegram_bot_token and settings.telegram_chat_id:
        tasks.append(send_telegram_message(message))
    
    if settings.discord_webhook_url:
        tasks.append(send_discord_message(message))
    
    if settings.slack_webhook_url:
        tasks.append(send_slack_message(message))
    
    # PagerDuty only for errors and critical events
    if settings.pagerduty_routing_key and event_type in ("error", None):
        severity = "critical" if force else "warning"
        tasks.append(send_pagerduty_alert(message, severity))
        
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

