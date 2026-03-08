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
    # Dedicated Endpoints
    MARKET_URL = "wss://advanced-trade-ws.coinbase.com"
    USER_URL = "wss://advanced-trade-ws-user.coinbase.com"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Ensure private key is properly formatted (newlines)
        if "-----BEGIN" in self.api_secret and "\\n" in self.api_secret:
            self.api_secret = self.api_secret.replace("\\n", "\n")

        self.market_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.user_ws: Optional[websockets.WebSocketClientProtocol] = None
        
        self.handlers: Dict[str, List[Callable]] = {
            "fills": [],
            "ticker": [],
            "l2": []
        }
        self.running = False
        
        # Sequence tracking
        self.last_sequences: Dict[str, int] = {}
        self._msg_count = 0

    def add_handler(self, channel: str, handler: Callable):
        if channel in self.handlers:
            self.handlers[channel].append(handler)

    async def start(self):
        self.running = True
        logger.info("Starting Dual WebSocket Client...")
        
        # Run both loops concurrently
        await asyncio.gather(
            self._connection_loop("market", self.MARKET_URL, ["ticker", "l2_data"]),
            self._connection_loop("user", self.USER_URL, ["user"])
        )

    async def _connection_loop(self, name: str, url: str, channels: List[str]):
        """Maintain a persistent connection to a specific endpoint."""
        retry_count = 0
        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    if name == "market":
                        self.market_ws = ws
                    else:
                        self.user_ws = ws
                        
                    retry_count = 0
                    logger.info(f"WebSocket connected to {name.upper()} endpoint: {url}")
                    
                    # Resubscribe to channels
                    product_ids = settings.product_ids.split(",")
                    for channel in channels:
                        # User channel doesn't strictly need product_ids and it's safer to omit if auth fails
                        pids = product_ids if channel != "user" else []
                        await self._send_subscription(ws, pids, channel, name)
                    
                    async for message in ws:
                        await self._handle_message(message, name)
                        
            except Exception as e:
                logger.error(f"WebSocket {name.upper()} connection error: {e}")
                if self.running:
                    retry_count += 1
                    wait = min(60, 2 ** retry_count)
                    logger.info(f"Retrying {name.upper()} WebSocket in {wait} seconds...")
                    await asyncio.sleep(wait)

    async def stop(self):
        self.running = False
        if self.market_ws:
            await self.market_ws.close()
        if self.user_ws:
            await self.user_ws.close()

    async def _send_subscription(self, ws: websockets.WebSocketClientProtocol, product_ids: List[str], channel: str, name: str):
        """
        Send a subscription message.
        Uses JWT authentication for Cloud API Keys (standard for Adv Trade).
        """
        try:
            from coinbase import jwt_generator
            jwt_token = jwt_generator.build_ws_jwt(self.api_key, self.api_secret)
            
            subscribe_msg = {
                "type": "subscribe",
                "channel": channel
            }
            if product_ids:
                subscribe_msg["product_ids"] = product_ids
            
            # Only use JWT for USER endpoint (private data)
            # Market endpoint channels (ticker, l2_data) are public and often reject JWT
            if name == "user":
                subscribe_msg["jwt"] = jwt_token

            await ws.send(json.dumps(subscribe_msg))
            logger.info(f"Sent {'authenticated ' if name == 'user' else ''}{name.upper()} subscription for {channel}")
        except Exception as e:
            logger.error(f"Failed to generate JWT for {name.upper()} subscription: {e}")
            # Fallback to HMAC signature for legacy API Keys
            timestamp = str(int(time.time()))
            import hmac
            import hashlib
            msg = f"{timestamp}{channel}{','.join(product_ids)}"
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                msg.encode("utf-8"),
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
            await ws.send(json.dumps(subscribe_msg))
            logger.info(f"Sent Legacy signature {name.upper()} subscription for {channel}")

    async def _handle_message(self, message: str, name: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode message from {name.upper()}: {message[:100]}")
            return

        msg_type = data.get("type")
        channel = data.get("channel")
        
        if msg_type == "error":
            logger.error(f"Coinbase WS {name.upper()} Error: {data.get('message')} (Reason: {data.get('reason')})")
            return

        if msg_type == "subscriptions":
             logger.info(f"Confirmed {name.upper()} subscriptions: {data.get('channels')}")
             return

        # Debug log for throughput (every 100th message or specific channels)
        self._msg_count += 1
        if self._msg_count % 500 == 0 or channel in ["user"]:
            logger.debug(f"WS Message: channel={channel}, type={msg_type}")

        # sequence reconciliation if supported by channel
        sequence = data.get("sequence")
        if sequence and channel:
            expected = self.last_sequences.get(channel, sequence - 1) + 1
            if sequence > expected:
                logger.warning(f"Gap detected in {channel} sequence: expected {expected}, got {sequence}")
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
        
        elif channel == "l2_data":
            # Pass full L2 event to handlers
            for handler in self.handlers["l2"]:
                await handler(data)
