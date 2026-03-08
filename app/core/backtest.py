import logging
import asyncio
import time
from decimal import Decimal
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

from app.core.manager import manager
from app.core import indicators

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(
        self,
        product_id: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 1000.0,
        maker_fee: float = 0.001,
    ):
        self.product_id = product_id
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = Decimal(str(initial_capital))
        self.maker_fee = Decimal(str(maker_fee))
        
        self.balance = self.initial_balance
        self.inventory = Decimal("0")
        self.trades: List[Dict[str, Any]] = []
        self.pnl_history: List[float] = []
        self.active_orders: List[Dict[str, Any]] = []

    async def run(self):
        """Fetch data and run the simulation."""
        # 1. Convert dates to timestamps
        try:
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
        except Exception as e:
            logger.error(f"Invalid dates for backtest: {e}")
            return None

        # 2. Fetch candles from exchange
        # We'll use 1-hour candles for backtest to cover longer ranges efficiently
        candles = await manager.exchange.fetch_public_candles(
            product_id=self.product_id,
            granularity="ONE_HOUR",
            limit=300 
        )
        
        if not candles:
            logger.warning(f"No candles found for {self.product_id}")
            return None

        # 3. Simulation Logic (Grid)
        from app.config import settings
        grid_lines = settings.grid_lines
        band_pct = Decimal(str(settings.grid_band_pct))
        order_notional = Decimal(str(settings.base_order_notional_usd))

        mid_price = candles[0][3]
        upper = mid_price * (1 + band_pct)
        lower = mid_price * (1 - band_pct)
        step = (upper - lower) / (grid_lines - 1)
        grid_levels = [lower + (step * i) for i in range(grid_lines)]
        
        for i, level in enumerate(grid_levels):
            if level > mid_price:
                self.active_orders.append({"side": "SELL", "price": level, "size": order_notional / level, "index": i})
            elif level < mid_price:
                self.active_orders.append({"side": "BUY", "price": level, "size": order_notional / level, "index": i})

        for i in range(1, len(candles)):
            ts, high, low, current_price, volume = candles[i]
            
            filled_orders = []
            for order in self.active_orders:
                if order["side"] == "BUY" and low <= order["price"]:
                    filled_orders.append(order)
                elif order["side"] == "SELL" and high >= order["price"]:
                    filled_orders.append(order)
            
            for order in filled_orders:
                self.active_orders.remove(order)
                self._handle_fill(order, current_price, grid_levels)
            
            # Track PnL (Total Equity)
            equity = self.balance + (self.inventory * current_price)
            self.pnl_history.append(float(equity - self.initial_balance))

        return self.get_report()

    def _handle_fill(self, order: Dict[str, Any], current_price: Decimal, grid_levels: List[Decimal]):
        side = order["side"]
        price = order["price"]
        size = order["size"]
        
        cost = size * price
        fee = cost * self.maker_fee
        
        if side == "BUY":
            self.balance -= (cost + fee)
            self.inventory += size
        else:
            self.balance += (cost - fee)
            self.inventory -= size
            
        self.trades.append({
            "side": side,
            "price": float(price),
            "size": float(size),
            "fee": float(fee),
            "balance": float(self.balance),
            "inventory": float(self.inventory)
        })
        
        # Place replacement order
        new_side = "SELL" if side == "BUY" else "BUY"
        new_index = order["index"] + 1 if side == "BUY" else order["index"] - 1
        
        if 0 <= new_index < len(grid_levels):
            new_price = grid_levels[new_index]
            self.active_orders.append({
                "side": new_side, 
                "price": new_price, 
                "size": size, 
                "index": new_index
            })

    def get_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.pnl_history:
            return {}
            
        total_pnl = Decimal(str(self.pnl_history[-1]))
        max_drawdown = Decimal("0")
        peak = Decimal("-Infinity")
        
        for pnl_val in self.pnl_history:
            pnl = Decimal(str(pnl_val))
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        return {
            "initial_balance": float(self.initial_balance),
            "final_equity": float(self.initial_balance + total_pnl),
            "total_pnl": float(total_pnl),
            "max_drawdown": float(max_drawdown),
            "trade_count": len(self.trades)
        }
