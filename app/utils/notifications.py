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

async def notify(message: str, force: bool = False):
    """Dispatch message to all enabled notification channels."""
    if not settings.notifications_enabled and not force:
        return

    tasks = []
    if settings.telegram_bot_token and settings.telegram_chat_id:
        tasks.append(send_telegram_message(message))
    
    if settings.discord_webhook_url:
        tasks.append(send_discord_message(message))
        
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
