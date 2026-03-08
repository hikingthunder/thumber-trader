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
    # For Advanced Trade, the user endpoint can handle all channels if authenticated
    URL = "wss://advanced-trade-ws-user.coinbase.com"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Ensure private key is properly formatted (newlines)
        if "-----BEGIN" in self.api_secret and "\\n" in self.api_secret:
            self.api_secret = self.api_secret.replace("\\n", "\n")

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.subscriptions: Set[str] = set()
        self.handlers: Dict[str, List[Callable]] = {
            "fills": [],
            "ticker": [],
            "l2": []
        }
        self.running = False
        self.retry_count = 0
        self.max_retries = 10
        
        # Sequence tracking
        self.last_sequences: Dict[str, int] = {}

    def add_handler(self, channel: str, handler: Callable):
        if channel in self.handlers:
            self.handlers[channel].append(handler)

    async def start(self):
        self.running = True
        while self.running:
            try:
                # Use the user-authenticated endpoint which handles both public and private data
                async with websockets.connect(self.URL) as ws:
                    self.ws = ws
                    self.retry_count = 0
                    logger.info(f"WebSocket connected to {self.URL}")
                    
                    # Resubscribe to channels
                    product_ids = settings.product_ids.split(",")
                    
                    # Subscribe to public and private channels
                    # Advanced Trade V3 requires authentication for ALL channels 
                    # if using a Cloud API Key on the user endpoint.
                    await self._send_subscription(product_ids, "ticker")
                    await self._send_subscription(product_ids, "l2_data")
                    await self._send_subscription(product_ids, "user")
                    
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
        """
        Send a subscription message.
        Uses JWT authentication for Cloud API Keys (standard for Adv Trade).
        """
        try:
            from coinbase import jwt_generator
            # Correct function found via inspection: build_ws_jwt
            jwt_token = jwt_generator.build_ws_jwt(self.api_key, self.api_secret)
            
            subscribe_msg = {
                "type": "subscribe",
                "channel": channel,
                "product_ids": product_ids,
                "jwt": jwt_token
            }
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Sent authenticated subscription for {channel} product_ids: {product_ids}")
        except Exception as e:
            logger.error(f"Failed to generate JWT for subscription: {e}")
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
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Sent Legacy signature subscription for {channel}")

    async def _handle_message(self, message: str):
        data = json.loads(message)
        msg_type = data.get("type")
        channel = data.get("channel")
        
        if msg_type == "error":
            logger.error(f"Coinbase WS Error: {data.get('message')} (Reason: {data.get('reason')})")
            return

        if msg_type == "subscriptions":
             logger.info(f"Confirmed subscriptions: {data.get('channels')}")
             return

        # Debug log for throughput (every 100th message or specific channels)
        if not hasattr(self, "_msg_count"): self._msg_count = 0
        self._msg_count += 1
        if self._msg_count % 100 == 0 or channel in ["ticker", "user"]:
            logger.debug(f"WS Message: channel={channel}, type={msg_type}")

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
        
        elif channel == "l2_data":
            # Pass full L2 event to handlers
            for handler in self.handlers["l2"]:
                await handler(data)
