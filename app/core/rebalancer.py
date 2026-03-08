"""Portfolio Rebalancer — maintains target weights across multiple trading pairs."""

import asyncio
import logging
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional

from app.core.exchange import CoinbaseExchange
from app.config import settings

logger = logging.getLogger(__name__)


class PortfolioRebalancer:
    """Multi-pair portfolio weight management with periodic drift checks."""
    
    def __init__(self, exchange: CoinbaseExchange):
        self.exchange = exchange
        self.target_weights: Dict[str, Decimal] = {}  # e.g. {"BTC": 0.60, "ETH": 0.30, "USD": 0.10}
        self.last_rebalance_ts: float = 0.0
        self.drift_threshold: Decimal = Decimal("0.05")  # 5% drift tolerance
    
    def set_targets(self, weights: Dict[str, Decimal]):
        """Set target portfolio weights. Must sum to 1.0."""
        total = sum(weights.values())
        if abs(total - Decimal("1.0")) > Decimal("0.001"):
            raise ValueError(f"Target weights must sum to 1.0, got {total}")
        self.target_weights = weights
    
    async def get_current_weights(self) -> Dict[str, Any]:
        """Calculate current portfolio weights based on account balances."""
        balances = await self.exchange.get_account_balances()
        
        # Get USD values for each asset
        usd_values: Dict[str, Decimal] = {}
        total_usd = Decimal("0")
        
        for currency, balance in balances.items():
            if balance <= 0:
                continue
            
            if currency == "USD":
                usd_values["USD"] = balance
                total_usd += balance
            elif balance > 0:
                try:
                    price = await self.exchange.get_current_price(f"{currency}-USD")
                    if price > 0:
                        value = balance * price
                        usd_values[currency] = value
                        total_usd += value
                except Exception:
                    continue
        
        # Calculate current weights
        current_weights = {}
        if total_usd > 0:
            for currency, value in usd_values.items():
                current_weights[currency] = (value / total_usd).quantize(Decimal("0.0001"))
        
        return {
            "balances": {k: str(v) for k, v in usd_values.items()},
            "weights": {k: str(v) for k, v in current_weights.items()},
            "total_usd": str(total_usd)
        }
    
    async def check_drift(self) -> Dict[str, Any]:
        """Check if any asset has drifted beyond the threshold."""
        if not self.target_weights:
            return {"needs_rebalance": False, "reason": "No targets set"}
        
        portfolio = await self.get_current_weights()
        current = portfolio["weights"]
        
        drifts = {}
        needs_rebalance = False
        
        for asset, target in self.target_weights.items():
            current_weight = Decimal(current.get(asset, "0"))
            drift = current_weight - target
            drift_pct = abs(drift)
            
            drifts[asset] = {
                "target": str(target),
                "current": str(current_weight),
                "drift": str(drift),
                "over_threshold": drift_pct > self.drift_threshold
            }
            
            if drift_pct > self.drift_threshold:
                needs_rebalance = True
        
        return {
            "needs_rebalance": needs_rebalance,
            "drifts": drifts,
            "total_usd": portfolio["total_usd"]
        }
    
    async def generate_rebalance_orders(self) -> List[Dict]:
        """Generate the orders needed to rebalance the portfolio."""
        drift_result = await self.check_drift()
        
        if not drift_result["needs_rebalance"]:
            return []
        
        total_usd = Decimal(drift_result["total_usd"])
        orders = []
        
        for asset, info in drift_result["drifts"].items():
            if asset == "USD":
                continue  # Can't trade USD directly
            
            drift = Decimal(info["drift"])
            
            if abs(drift) <= self.drift_threshold:
                continue  # Within tolerance
            
            # Calculate USD amount to trade
            trade_usd = abs(drift) * total_usd
            
            if trade_usd < settings.min_notional_usd:
                continue  # Too small
            
            side = "SELL" if drift > 0 else "BUY"
            
            orders.append({
                "product_id": f"{asset}-USD",
                "side": side,
                "notional_usd": str(trade_usd.quantize(Decimal("0.01"))),
                "reason": f"Rebalance: {asset} drifted {drift*100:.2f}% from target"
            })
        
        return orders
