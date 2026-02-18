import logging
import csv
import json
from decimal import Decimal, ROUND_DOWN
from typing import List, Tuple, Dict, Any, Optional
from app.core import indicators

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(
        self,
        initial_balance: Decimal = Decimal("1000"),
        maker_fee: Decimal = Decimal("0.004"),
        slippage: Decimal = Decimal("0.0001")
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.inventory = Decimal("0")
        self.maker_fee = maker_fee
        self.slippage = slippage
        
        self.trades: List[Dict[str, Any]] = []
        self.pnl_history: List[Decimal] = []
        
        self.active_orders: List[Dict[str, Any]] = []

    def load_candles_from_csv(self, filepath: str) -> List[Tuple[int, Decimal, Decimal, Decimal, Decimal]]:
        """Load candles from a CSV file (ts, high, low, close, volume)."""
        candles = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                next(reader) # skip header
                for row in reader:
                    # ts, high, low, close, volume
                    candles.append((
                        int(row[0]),
                        Decimal(row[1]),
                        Decimal(row[2]),
                        Decimal(row[3]),
                        Decimal(row[4])
                    ))
        except Exception as e:
            logger.error(f"Failed to load candles: {e}")
        return candles

    def run(self, candles: List[Tuple[int, Decimal, Decimal, Decimal, Decimal]], grid_lines: int, band_pct: Decimal, order_notional: Decimal):
        """Run the backtest simulation."""
        if not candles:
            return
            
        # Initial grid setup
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
            _ts, high, low, current_price, volume = candles[i]
            
            # Check for fills
            filled_orders = []
            for order in self.active_orders:
                if order["side"] == "BUY" and low <= order["price"]:
                    filled_orders.append(order)
                elif order["side"] == "SELL" and high >= order["price"]:
                    filled_orders.append(order)
            
            for order in filled_orders:
                self.active_orders.remove(order)
                self._handle_fill(order, current_price, grid_levels)
            
            # Track PnL
            unrealized_pnl = self.inventory * (current_price - (self.trades[-1]["price"] if self.trades else current_price))
            self.pnl_history.append(self.balance + (self.inventory * current_price) - self.initial_balance)

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
            "price": price,
            "size": size,
            "fee": fee,
            "balance": self.balance,
            "inventory": self.inventory
        })
        
        # Place replacement order
        new_side = "SELL" if side == "BUY" else "BUY"
        new_index = order["index"] + 1 if side == "BUY" else order["index"] - 1
        
        if 0 <= new_index < len(grid_levels):
            new_price = grid_levels[new_index]
            self.active_orders.append({
                "side": new_side, 
                "price": new_price, 
                "size": size, # Simplified
                "index": new_index
            })

    def get_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.pnl_history:
            return {}
            
        total_pnl = self.pnl_history[-1]
        max_drawdown = Decimal("0")
        peak = Decimal("-Infinity")
        
        for pnl in self.pnl_history:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        return {
            "initial_balance": self.initial_balance,
            "final_equity": self.initial_balance + total_pnl,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "trade_count": len(self.trades)
        }
