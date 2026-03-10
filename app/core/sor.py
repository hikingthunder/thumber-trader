"""Smart Order Routing (SOR) — splits large orders to minimize slippage and market impact."""

import asyncio
import logging
import random
import uuid
from decimal import Decimal, ROUND_DOWN
from typing import Optional

from app.core.exchange import CoinbaseExchange
from app.config import settings

logger = logging.getLogger(__name__)


class SmartOrderRouter:
    """Split large orders into smaller randomized-timing chunks."""
    
    def __init__(self, exchange: CoinbaseExchange):
        self.exchange = exchange
    
    async def execute_split_order(
        self,
        product_id: str,
        side: str,
        total_notional_usd: Decimal,
        current_price: Decimal,
        num_slices: int = 5,
        max_delay_ms: int = 2000,
        price_limit_pct: Optional[Decimal] = None
    ) -> dict:
        """Execute a large order split into randomized-timing slices.
        
        Args:
            product_id: Trading pair (e.g., "BTC-USD")
            side: "BUY" or "SELL" 
            total_notional_usd: Total USD value of the order
            current_price: Current market price
            num_slices: Number of child orders
            max_delay_ms: Maximum random delay between slices (milliseconds)
            price_limit_pct: Maximum deviation from current price (e.g., 0.005 = 0.5%)
        
        Returns:
            Dict with execution results summary
        """
        if current_price <= 0:
            return {"status": "error", "detail": "Invalid price"}
        
        slice_notional = total_notional_usd / Decimal(num_slices)
        total_filled = Decimal("0")
        child_orders = []
        total_fees = Decimal("0")
        
        logger.info(f"SOR: Splitting {side} ${total_notional_usd} into {num_slices} slices of ${slice_notional}")
        
        for i in range(num_slices):
            # Fetch real-time price for this specific slice to prevent slippage
            try:
                slice_price = await self.exchange.get_current_price(product_id)
                if slice_price <= 0:
                    logger.warning(f"SOR slice {i+1} got invalid price: {slice_price}. Using last known.")
                    slice_price = current_price
                else:
                    current_price = slice_price # Update last known
            except Exception as e:
                logger.warning(f"SOR slice {i+1} price fetch failed: {e}. Using last known.")
                slice_price = current_price

            # Calculate size for this slice
            size = (slice_notional / slice_price).quantize(Decimal("0.00001"), rounding=ROUND_DOWN)
            
            if size <= 0:
                continue
            
            # Calculate limit price with slippage protection
            if price_limit_pct:
                if side == "BUY":
                    limit_price = slice_price * (1 + price_limit_pct)
                else:
                    limit_price = slice_price * (1 - price_limit_pct)
            else:
                limit_price = slice_price
            
            limit_price = limit_price.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            
            try:
                if settings.is_simulated_execution():
                    child_id = f"sor-paper-{uuid.uuid4()}"
                    child_orders.append({
                        "order_id": child_id,
                        "slice": i + 1,
                        "side": side,
                        "size": str(size),
                        "price": str(limit_price),
                        "status": "filled_paper"
                    })
                    total_filled += size
                else:
                    config = {
                        "limit_limit_gtc": {
                            "base_size": str(size),
                            "limit_price": str(limit_price),
                            "post_only": True
                        }
                    }
                    response = await self.exchange.create_order(product_id, side, config)
                    child_id = response.get("order_id", "unknown")
                    child_orders.append({
                        "order_id": child_id,
                        "slice": i + 1,
                        "side": side,
                        "size": str(size),
                        "price": str(limit_price),
                        "status": "placed"
                    })
                    total_filled += size
                    
            except Exception as e:
                logger.error(f"SOR slice {i+1} failed: {e}")
                child_orders.append({
                    "slice": i + 1,
                    "status": "failed",
                    "error": str(e)
                })
            
            # Random delay between slices (except last)
            if i < num_slices - 1:
                delay_ms = random.randint(100, max_delay_ms)
                await asyncio.sleep(delay_ms / 1000.0)
        
        result = {
            "status": "completed",
            "side": side,
            "total_notional_usd": str(total_notional_usd),
            "slices_attempted": num_slices,
            "slices_filled": sum(1 for o in child_orders if "failed" not in o.get("status", "")),
            "total_base_filled": str(total_filled),
            "child_orders": child_orders
        }
        
        logger.info(f"SOR complete: {result['slices_filled']}/{num_slices} slices filled, total {total_filled} base")
        return result
