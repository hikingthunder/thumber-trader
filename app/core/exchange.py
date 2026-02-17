import asyncio
import json
import logging
import time
import urllib.request
import urllib.parse
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Callable
import uuid

# Try to import official client, handle failure gracefully (though it should be installed)
try:
    from coinbase.rest import RESTClient
except ImportError:
    RESTClient = None

class CoinbaseExchange:
    def __init__(self, api_key: str, api_secret: str):
        if not api_key or not api_secret:
            raise ValueError("API Key and Secret are required for CoinbaseExchange")
        
        if RESTClient is None:
             raise ImportError("coinbase-python library not found. Please install it.")

        # Handle potential escaped newlines from env vars or UI input
        if "-----BEGIN" in api_secret and "\\n" in api_secret:
            api_secret = api_secret.replace("\\n", "\n")

        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self._latency_observations: List[float] = []

    async def _run_async(self, func: Callable, *args, **kwargs) -> Any:
        """Run a synchronous blocking function in a thread pool."""
        return await asyncio.to_thread(func, *args, **kwargs)

    async def get_product(self, product_id: str) -> Dict[str, Any]:
        return await self._run_async(self.client.get_product, product_id=product_id)

    async def get_current_price(self, product_id: str) -> Decimal:
        try:
            product = await self.get_product(product_id)
            
            # Helper to get value from either dict or object
            def get_v(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            price_str = get_v(product, "price")
            if price_str:
                return Decimal(str(price_str))
            
            # Fallback for different API versions or nested structures
            return Decimal("0")
        except Exception as e:
            logging.warning(f"Could not parse price for {product_id}: {e}")
            return Decimal("0")


    async def get_account_balances(self) -> Dict[str, Decimal]:
        try:
            response = await self._run_async(self.client.get_accounts)
            
            # The library can return a dict or an object with 'accounts' attribute
            if isinstance(response, dict):
                accounts = response.get("accounts", [])
            else:
                accounts = getattr(response, "accounts", [])
            
            logging.info(f"Exchange found {len(accounts)} accounts")
            
            balances = {}
            for account in accounts:
                # Helper to get value from either dict or object
                def get_v(obj, key, default=None):
                    if isinstance(obj, dict):
                        return obj.get(key, default)
                    return getattr(obj, key, default)

                currency = get_v(account, "currency")
                available_obj = get_v(account, "available_balance")
                
                if currency and available_obj:
                    available_val = get_v(available_obj, "value", 0)
                    balances[str(currency)] = Decimal(str(available_val))
            
            # Debug log for a few balances to verify
            if balances:
                sample = {k: str(v) for k, v in list(balances.items())[:3]}
                logging.info(f"Sample balances: {sample}")
            else:
                logging.warning("No non-zero balances found or parsing failed.")
                
            return balances
        except Exception as e:
            logging.error(f"Error fetching account balances: {e}")
            return {}

    async def create_order(self, 
                           product_id: str, 
                           side: str, 
                           order_configuration: Dict[str, Any],
                           client_order_id: Optional[str] = None) -> Dict[str, Any]:
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())
            
        return await self._run_async(
            self.client.create_order,
            client_order_id=client_order_id,
            product_id=product_id,
            side=side,
            order_configuration=order_configuration
        )

    async def cancel_orders(self, order_ids: List[str]) -> Any:
        return await self._run_async(self.client.cancel_orders, order_ids=order_ids)
    
    async def get_open_orders(self, product_id: str) -> List[Dict[str, Any]]:
        # This might return a generator or paginated response, handle carefully
        response = await self._run_async(
             self.client.get_orders, 
             product_id=product_id, 
             order_status=["OPEN", "PENDING", "ACTIVE"]
        )
        orders = response.get("orders", []) if isinstance(response, dict) else getattr(response, "orders", [])
        return orders

    # Public Endpoints using urllib (as per original design for speed/simplicity on public data)
    # properly wrapped in async to avoid blocking
    async def fetch_public_candles(self, product_id: str, granularity: str, limit: int = 300) -> List[Tuple[int, Decimal, Decimal, Decimal, Decimal]]:
        # Map granularity string to what API expects if needed, or assume caller passes correct value
        # The library method is likely get_candles(product_id, start_time, end_time, granularity)
        # However, for simplicity and to match the 'limit' semantic, we might need to calculate start/end
        # But wait, looking at the official SDK, it usually takes start/end.
        # Let's try to find a way to get "last N candles".
        # As a fallback, we can calculate start time based on granularity * limit.
        
        # Approximate start time calculation
        # Granularity is typically "ONE_MINUTE", "FIVE_MINUTE", etc.
        seconds_map = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800,
            "ONE_HOUR": 3600,
            "TWO_HOUR": 7200,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400
        }
        
        seconds = seconds_map.get(granularity, 3600)
        end_time = int(time.time())
        start_time = end_time - (seconds * limit)

        try:
            # The SDK method signature for get_candles in coinbase-advanced-py (which wraps the official one)
            # typically expects start and end as unix timestamps (int or string).
            
            # Using the client to handle auth
            response = await self._run_async(
                self.client.get_candles,
                product_id=product_id,
                start=str(start_time),
                end=str(end_time),
                granularity=granularity
            )
            
            # SDK returns a dict with 'candles' key
            candles = response.get("candles", []) if isinstance(response, dict) else getattr(response, "candles", [])

            rows = []
            for candle in candles:
                try:
                    # Candle object attributes: start, low, high, open, close, volume
                    # If it's a dict
                    if isinstance(candle, dict):
                         start = int(candle.get("start", 0))
                         high = Decimal(str(candle.get("high", 0)))
                         low = Decimal(str(candle.get("low", 0)))
                         close = Decimal(str(candle.get("close", 0)))
                         volume = Decimal(str(candle.get("volume", 0)))
                    else:
                         # If it's an object
                         start = int(getattr(candle, "start", 0))
                         high = Decimal(str(getattr(candle, "high", 0)))
                         low = Decimal(str(getattr(candle, "low", 0)))
                         close = Decimal(str(getattr(candle, "close", 0)))
                         volume = Decimal(str(getattr(candle, "volume", 0)))
                    
                    rows.append((start, high, low, close, volume))
                except Exception:
                    continue
            
            rows.sort(key=lambda x: x[0])
            return rows

        except Exception as e:
            logging.warning(f"Failed to fetch candles via client for {product_id}: {e}")
            return []

    async def get_market_trades(self, product_id: str, limit: int = 50) -> List[Dict[str, Any]]:
         # Similar implementation for trades if needed
         # Original code used: https://api.exchange.coinbase.com/products/{product_id}/trades
         url = f"https://api.exchange.coinbase.com/products/{product_id}/trades"
         
         def _fetch():
            req = urllib.request.Request(url, headers={"User-Agent": "thumber-trader/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

         try:
            data = await self._run_async(_fetch)
            return data[:limit]
         except Exception as e:
            logging.warning(f"Failed to fetch trades: {e}")
            return []

