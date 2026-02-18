import asyncio
import json
import logging
import time
import hmac
import hashlib
from typing import Dict, Any, List, Optional, Callable, Set
import websockets
from app.config import settings

logger = logging.getLogger(__name__)

class WSUserClient:
    """
    Coinbase Advanced Trade WebSocket Client.
    Handles 'user' channel for fills and 'ticker' channel for price updates.
    """
    URL = "wss://advanced-trade-ws.coinbase.com"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.subscriptions: Set[str] = set()
        self.handlers: Dict[str, List[Callable]] = {
            "fills": [],
            "ticker": []
        }
        self.running = False
        self.retry_count = 0
        self.max_retries = 10
        
        # Sequence tracking
        self.last_sequences: Dict[str, int] = {}

    def add_handler(self, channel: str, handler: Callable):
        if channel in self.handlers:
            self.handlers[channel].append(handler)

    async def subscribe(self, product_ids: List[str], channel: str = "ticker"):
        self.subscriptions.add(channel)
        if self.ws and self.ws.open:
            await self._send_subscription(product_ids, channel)

    async def start(self):
        self.running = True
        while self.running:
            try:
                async with websockets.connect(self.URL) as ws:
                    self.ws = ws
                    self.retry_count = 0
                    logger.info("WebSocket connected to Coinbase.")
                    
                    # Resubscribe to channels
                    product_ids = settings.product_ids.split(",")
                    await self._send_subscription(product_ids, "user")
                    await self._send_subscription(product_ids, "ticker")
                    
                    async for message in ws:
                        await self._handle_message(message)
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    self.retry_count += 1
                    wait = min(60, 2 ** self.retry_count)
                    logger.info(f"Retrying WebSocket in {wait} seconds...")
                    await asyncio.sleep(wait)

    async def stop(self):
        self.running = False
        if self.ws:
            await self.ws.close()

    async def _send_subscription(self, product_ids: List[str], channel: str):
        timestamp = str(int(time.time()))
        message = f"{timestamp}{channel}{','.join(product_ids)}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            digestmod=hashlib.sha256
        ).hexdigest()

        subscribe_msg = {
            "type": "subscribe",
            "channel": channel,
            "product_ids": product_ids,
            "api_key": self.api_key,
            "timestamp": timestamp,
            "signature": signature
        }
        await self.ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {channel} for {product_ids}")

    async def _handle_message(self, message: str):
        data = json.loads(message)
        channel = data.get("channel")
        
        # sequence reconciliation if supported by channel
        sequence = data.get("sequence")
        if sequence and channel:
            expected = self.last_sequences.get(channel, sequence - 1) + 1
            if sequence > expected:
                logger.warning(f"Gap detected in {channel} sequence: expected {expected}, got {sequence}")
                # Trigger reconciliation event
                for handler in self.handlers.get("reconcile", []):
                    await handler(channel, expected, sequence)
            self.last_sequences[channel] = sequence

        if channel == "user":
            # Handle fills
            events = data.get("events", [])
            for event in events:
                if event.get("type") == "fill":
                    for handler in self.handlers["fills"]:
                        await handler(event)
        
        elif channel == "ticker":
            events = data.get("events", [])
            for event in events:
                tickers = event.get("tickers", [])
                for ticker in tickers:
                    for handler in self.handlers["ticker"]:
                        await handler(ticker)
