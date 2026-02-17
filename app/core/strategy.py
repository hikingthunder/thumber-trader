
import asyncio
import logging
import uuid
import time
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple, Any

from app.config import settings
from app.core.engine import StrategyEngine
from app.core.exchange import CoinbaseExchange
from app.database.db import get_db, AsyncSessionLocal
from app.database.models import Order, Fill, DailyStats, TaxLot, TaxLotMatch
from app.core.state import SharedRiskState
from app.core import analysis
from sqlalchemy import select, update, delete
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class GridStrategy(StrategyEngine):
    def __init__(self, product_id: str, exchange: CoinbaseExchange, shared_risk_state: Optional[SharedRiskState] = None):
        self.product_id = product_id
        self.exchange = exchange
        self.shared_risk_state = shared_risk_state
        self.running = False
        
        # State variables
        self.grid_levels: List[Decimal] = []
        self.orders: Dict[str, Dict[str, Any]] = {} # Cache of active orders
        self.active_order_ids: set = set()
        
        # Market data cache
        self.last_price: Decimal = Decimal("0")
        self.candles: List[Tuple[int, Decimal, Decimal, Decimal, Decimal]] = [] # OHLCV
        
        # PnL State
        self.realized_pnl: Decimal = Decimal("0")
        
        # Configuration shortcuts
        self.base_currency = product_id.split("-")[0]
        self.quote_currency = product_id.split("-")[1]
        
    async def run(self):
        logger.info(f"Starting GridStrategy for {self.product_id}")
        self.running = True
        
        # Initial setup: load orders, check balance, set initial grid if needed
        await self._load_state()
        
        while self.running:
            try:
                loop_start = time.time()
                
                # 1. Update Market Data
                await self._update_market_data()
                
                # 2. Check Order Statuses (Fills)
                await self._process_fills()
                
                # 3. Update Indicators / Analysis
                metrics = self._analyze_market()
                
                # 4. Core Logic: Place/Cancel Orders
                await self._execute_grid_logic(metrics)

                # 5. Risk Checks
                await self._check_risk_parameters()
                
                # Sleep for remaining poll interval
                elapsed = time.time() - loop_start
                sleep_time = max(0.1, settings.poll_seconds - elapsed)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in strategy loop for {self.product_id}: {e}", exc_info=True)
                await asyncio.sleep(5) # Backoff on error

    def stop(self):
        self.running = False
        logger.info(f"Stopping GridStrategy for {self.product_id}")

    async def _load_state(self):
        """Load active orders from DB on startup."""
        async with AsyncSessionLocal() as session:
            stmt = select(Order).where(Order.product_id == self.product_id)
            result = await session.execute(stmt)
            db_orders = result.scalars().all()
            
            for order in db_orders:
                self.orders[order.order_id] = {
                    "id": order.order_id,
                    "side": order.side,
                    "price": Decimal(order.price),
                    "base_size": Decimal(order.base_size),
                    "grid_index": order.grid_index
                }
                self.active_order_ids.add(order.order_id)
            logger.info(f"Loaded {len(self.orders)} active orders from database.")

    async def _update_market_data(self):
        """Fetch latest price and candles."""
        try:
            self.last_price = await self.exchange.get_current_price(self.product_id)
            # Fetch enough candles for indicators (e.g. RSI 14, EMA 200)
            self.candles = await self.exchange.fetch_public_candles(
                self.product_id, 
                granularity=settings.trend_candle_granularity,
                limit=300
            )
        except Exception as e:
            logger.warning(f"Market data update failed: {e}")

    async def _process_fills(self):
        """Check status of open orders against API/Sim logic."""
        if not self.orders:
            return

        # For simplicity in this v1, we fetch all open orders from exchange to reconcile
        # In a real high-freq bot, we would use websocket stream for updates
        # Here we do a REST poll every cycle (simulated/real)
        
        # If paper trading, simulate fills based on last price
        if settings.paper_trading_mode:
             await self._simulate_fills()
             return

        # Real trading reconciliation
        try:
            open_orders_api = await self.exchange.get_open_orders(self.product_id)
            api_order_ids = set(o["order_id"] for o in open_orders_api)
            
            # Identify filled orders (present in local but missing in API open orders)
            # Note: This is a simplified check. A robust system checks return values or a fill feed.
            filled_ids = []
            for oid in list(self.active_order_ids):
                if oid not in api_order_ids:
                    # Order is no longer open in API. It might be filled or canceled.
                    # We assume filled if we didn't cancel it ourselves (or check fills endpoint).
                    # For now, treat as filled and verify with fills endpoint if possible or move on.
                    # Using get_order would be safer but more API calls.
                    filled_ids.append(oid)
            
            for oid in filled_ids:
                await self._handle_fill(oid)

        except Exception as e:
            logger.warning(f"Error processing fills: {e}")

    async def _simulate_fills(self):
        """Simulate fills in paper trading mode."""
        filled_ids = []
        for oid, order in self.orders.items():
            side = order["side"]
            price = order["price"]
            
            if side == "BUY" and self.last_price <= price:
                filled_ids.append(oid)
            elif side == "SELL" and self.last_price >= price:
                filled_ids.append(oid)
        
        for oid in filled_ids:
            # Add artificial delay or slippage logic here if needed
            await self._handle_fill(oid)

    async def _handle_fill(self, order_id: str):
        """Record fill to DB, update state, and trigger logic."""
        if order_id not in self.orders:
            return
            
        order = self.orders[order_id]
        logger.info(f"Order {order_id} ({order['side']} @ {order['price']}) filled.")
        
        # RECORD FILL
        async with AsyncSessionLocal() as session:
            # Verify it exists in DB to avoid FK errors or duplicate processing
            db_order = await session.get(Order, order_id)
            if db_order:
                # Create Fill record
                fill = Fill(
                    ts=time.time(),
                    product_id=self.product_id,
                    side=order["side"],
                    price=str(order["price"]),
                    base_size=str(order["base_size"]),
                    fee_paid="0", # TODO: Calculate or fetch fee
                    grid_index=order["grid_index"],
                    order_id=order_id
                )
                session.add(fill)
                await session.flush() # Get fill.id

                # TAX LOT LOGIC
                if order["side"] == "BUY":
                    # Create new Tax Lot
                    new_lot = TaxLot(
                        buy_fill_id=fill.id,
                        acquired_ts=time.time(),
                        product_id=self.product_id,
                        buy_price=str(order["price"]),
                        original_base_size=str(order["base_size"]),
                        remaining_base_size=str(order["base_size"]),
                        fee_paid_usd="0", # TODO
                        created_ts=time.time(),
                        updated_ts=time.time()
                    )
                    session.add(new_lot)
                
                elif order["side"] == "SELL":
                    # Match against existing open tax lots (FIFO)
                    sell_size = Decimal(order["base_size"])
                    sell_price = Decimal(order["price"])
                    
                    # Fetch open lots (oldest first)
                    stmt = select(TaxLot).where(
                        TaxLot.product_id == self.product_id,
                        TaxLot.remaining_base_size != "0"
                    ).order_by(TaxLot.acquired_ts.asc())
                    
                    open_lots = (await session.execute(stmt)).scalars().all()
                    
                    total_pnl = Decimal("0")
                    
                    for lot in open_lots:
                        if sell_size <= 0:
                            break
                            
                        lot_remaining = Decimal(lot.remaining_base_size)
                        match_size = min(sell_size, lot_remaining)
                        
                        # Update lot
                        lot.remaining_base_size = str(lot_remaining - match_size)
                        if (lot_remaining - match_size) == 0:
                            lot.closed_ts = time.time()
                        lot.updated_ts = time.time()
                        
                        # Calculate PnL for this match
                        cost_basis = match_size * Decimal(lot.buy_price)
                        proceeds = match_size * sell_price
                        pnl = proceeds - cost_basis
                        total_pnl += pnl
                        
                        # Create match record
                        match = TaxLotMatch(
                            sell_fill_id=fill.id,
                            lot_id=lot.id,
                            matched_base_size=str(match_size),
                            buy_price=lot.buy_price,
                            sell_price=str(sell_price),
                            proceeds_usd=str(proceeds),
                            cost_basis_usd=str(cost_basis),
                            realized_pnl_usd=str(pnl),
                            acquired_ts=lot.acquired_ts,
                            created_ts=time.time()
                        )
                        session.add(match)
                        
                        sell_size -= match_size
                    
                    fill.realized_pnl_usd = str(total_pnl)
                    self.realized_pnl += total_pnl

                # Remove from Orders table
                await session.delete(db_order)
                await session.commit()
        
        # Update in-memory state
        del self.orders[order_id]
        self.active_order_ids.discard(order_id)
        
        # Trigger replacement logic (Grid Logic)
        await self._place_replacement_order(order)

    async def _place_replacement_order(self, filled_order: Dict[str, Any]):
        """Place the opposite order for the grid."""
        side = filled_order["side"]
        price = filled_order["price"]
        grid_index = filled_order["grid_index"]
        # Basic logic: 
        # If BUY filled at index i, place SELL at index i+1
        # If SELL filled at index i, place BUY at index i-1
        
        new_side = "SELL" if side == "BUY" else "BUY"
        new_index = grid_index + 1 if side == "BUY" else grid_index - 1
        
        # Check boundary
        # If we haven't initialized grid levels locally, we can't determine price.
        # We need to ensure grid_levels are maintained.
        if not self.grid_levels:
             # If grid is lost, re-anchor or derive?
             # For now, assume simple relative spacing
             step_pct = settings.grid_band_pct / Decimal(settings.grid_lines) # simplified
             step = price * step_pct
             new_price = price + step if new_side == "SELL" else price - step
        else:
             if 0 <= new_index < len(self.grid_levels):
                  new_price = self.grid_levels[new_index]
             else:
                  logger.info(f"Grid boundary reached at index {new_index}, no replacement order.")
                  return

        # Place new order
        base_size = filled_order["base_size"] # Keep same size for simplicity or adjust
        
        await self._place_limit_order(new_side, new_price, base_size, new_index)

    async def _place_limit_order(self, side: str, price: Decimal, size: Decimal, grid_index: int):
        """Execute order placement via exchange and save to DB."""
        # Round/Quantize
        price = price.quantize(Decimal("0.01"), rounding=ROUND_DOWN) # TODO use product quantum
        size = size.quantize(Decimal("0.00001"), rounding=ROUND_DOWN) # TODO use product quantum

        if size * price < settings.min_notional_usd:
             logger.warning(f"Skipping {side} order, notional too small: {size*price}")
             return

        try:
            if settings.paper_trading_mode:
                client_order_id = f"paper-{uuid.uuid4()}"
                # Mimic response
                response = {"id": client_order_id, "status": "open"}
            else:
                # Real API call
                config = {
                    "limit_limit_gtc": {
                        "base_size": str(size),
                        "limit_price": str(price),
                        "post_only": True
                    }
                }
                response = await self.exchange.create_order(self.product_id, side, config)
                # Parse response for real ID
                # client_order_id might be in response or use what we sent
                client_order_id = response.get("order_id")

            if client_order_id:
                # Save to DB
                async with AsyncSessionLocal() as session:
                    new_order = Order(
                        order_id=client_order_id,
                        side=side,
                        price=str(price),
                        base_size=str(size),
                        grid_index=grid_index,
                        product_id=self.product_id,
                        created_ts=time.time(),
                        eligible_fill_ts=time.time()
                    )
                    session.add(new_order)
                    await session.commit()
                
                # Update memory
                self.orders[client_order_id] = {
                    "id": client_order_id,
                    "side": side,
                    "price": price,
                    "base_size": size,
                    "grid_index": grid_index
                }
                self.active_order_ids.add(client_order_id)
                logger.info(f"Placed {side} order {client_order_id} @ {price}")

        except Exception as e:
            logger.error(f"Failed to place {side} limit order: {e}")

    def _analyze_market(self) -> Dict[str, Any]:
        """Calculate indicators using analysis module."""
        if not self.candles:
             return {}
        
        closes = [c[3] for c in self.candles]
        
        rsi_val = analysis.rsi(closes, settings.alpha_rsi_period)
        ema_fast = analysis.ema(closes, settings.trend_ema_fast)
        ema_slow = analysis.ema(closes, settings.trend_ema_slow)
        
        # Trend detection
        trend = "NEUTRAL"
        if ema_slow > 0:
             diff_pct = (ema_fast - ema_slow) / ema_slow
             if diff_pct > settings.trend_strength_threshold:
                  trend = "UP"
             elif diff_pct < -settings.trend_strength_threshold:
                  trend = "DOWN"
        
        logger.debug(f"Analysis: RSI={rsi_val:.2f}, Trend={trend}")
        return {
             "rsi": rsi_val,
             "trend": trend,
             "price": self.last_price
        }

    async def _execute_grid_logic(self, metrics: Dict[str, Any]):
        """Main decision loop for initial placement or rebalancing logic."""
        # If no orders exist, we might need to initialize the grid (First Run)
        if not self.orders and self.running and self.last_price > 0:
             # Only verify checks and place if we intentionally want to start fresh 
             # (e.g. if DB was empty). 
             # For safety, manual trigger or explicit config required usually?
             # For this refactor, we'll assume if empty & auto-start enabled -> place grid
             
             # Calculate levels
             mid_price = self.last_price
             # Create a simple arithmetic grid around mid_price
             upper = mid_price * (1 + settings.grid_band_pct)
             lower = mid_price * (1 - settings.grid_band_pct)
             step = (upper - lower) / (settings.grid_lines - 1)
             
             self.grid_levels = [lower + (step * i) for i in range(settings.grid_lines)]
             
             logger.info(f"Initializing Grid: {self.grid_levels[0]:.2f} to {self.grid_levels[-1]:.2f}")
             
             # Place initial orders
             for i, level in enumerate(self.grid_levels):
                  # Determine side: Sell above mid, Buy below mid
                  # But typically grid bot places orders at all levels except current?
                  if level > mid_price * Decimal("1.001"): # Slightly above loop
                       side = "SELL"
                       # Calculate size based on balance/config
                       size = (settings.base_order_notional_usd / level)
                       await self._place_limit_order(side, level, size, i)
                  elif level < mid_price * Decimal("0.999"): # Slightly below loop
                       side = "BUY"
                       size = (settings.base_order_notional_usd / level)
                       await self._place_limit_order(side, level, size, i)

    async def get_stats(self) -> Dict[str, Any]:
        """Return strategy statistics including PnL."""
        unrealized_pnl = Decimal("0")
        inventory_base = Decimal("0")
        
        if self.last_price > 0:
            async with AsyncSessionLocal() as session:
                # Calculate Unrealized PnL from open tax lots
                stmt = select(TaxLot).where(
                    TaxLot.product_id == self.product_id,
                    TaxLot.remaining_base_size != "0"
                )
                result = await session.execute(stmt)
                open_lots = result.scalars().all()
                
                for lot in open_lots:
                    remaining = Decimal(lot.remaining_base_size)
                    buy_price = Decimal(lot.buy_price)
                    
                    # Value diff
                    current_val = remaining * self.last_price
                    cost_basis = remaining * buy_price
                    unrealized_pnl += (current_val - cost_basis)
                    inventory_base += remaining

        return {
            "running": self.running,
            "orders_count": len(self.orders),
            "last_price": self.last_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "inventory_base": inventory_base
        }

    async def _check_risk_parameters(self):
        """Global risk checks."""
        pass

