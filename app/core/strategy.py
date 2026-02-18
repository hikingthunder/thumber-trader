
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
from app.utils.notifications import notify
from sqlalchemy import select, update, delete
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class GridStrategy(StrategyEngine):
    def __init__(self, product_id: str, exchange: CoinbaseExchange, shared_risk_state: Optional[SharedRiskState] = None):
        self.product_id = product_id
        self.exchange = exchange
        self.shared_risk_state = shared_risk_state
        self.running = False
        self.paused = False
        
        # Quantitative tools
        self.kalman = analysis.KalmanFilter()
        
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
                if self.paused:
                    logger.debug(f"Strategy for {self.product_id} is paused. Skipping loop.")
                    await asyncio.sleep(5)
                    continue

                # 0. HA Check (Persistent via DB)
                if settings.ha_failover_enabled:
                    is_master = await self._acquire_ha_lock_db()
                    if not is_master:
                        logger.debug(f"HA: STANDBY mode for {settings.ha_instance_id}. Sleeping.")
                        await asyncio.sleep(settings.ha_standby_sleep_seconds)
                        continue
                    else:
                        logger.debug(f"HA: MASTER mode for {settings.ha_instance_id}.")

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
        """Fetch latest price and candles with optional consensus pricing."""
        try:
            cb_price = await self.exchange.get_current_price(self.product_id)
            self.last_price = cb_price
            
            # Consensus Pricing
            if settings.consensus_pricing_enabled:
                external_prices = []
                exchanges = settings.consensus_exchanges.split(",")
                for ex in exchanges:
                    if ex.strip().lower() == "coinbase":
                        continue
                    p = await self.exchange.get_external_price(ex.strip(), self.product_id)
                    if p:
                        external_prices.append(p)
                
                if external_prices:
                    avg_external = sum(external_prices) / len(external_prices)
                    # Check deviation
                    deviation = abs(cb_price - avg_external) / avg_external
                    if deviation > settings.consensus_max_deviation_pct:
                        logger.warning(f"Consensus price deviation too high ({deviation*100:.2f}%). Using average external price: {avg_external}")
                        self.last_price = avg_external
                    else:
                        # Blend consensus (weighted 50/50 for now)
                        self.last_price = (cb_price + avg_external) / 2

            # Fetch candles
            self.candles = await self.exchange.fetch_public_candles(
                self.product_id, 
                granularity=settings.trend_candle_granularity,
                limit=300
            )
            
            # Update Kalman Filter with latest price
            if self.last_price > 0:
                self.last_price = Decimal(str(self.kalman.update(float(self.last_price))))

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
        msg = f"🔔 *Order Filled*: {order['side']} {order['base_size']} {self.product_id} @ {order['price']}"
        await notify(msg)
        logger.info(f"Order {order_id} ({order['side']} @ {order['price']}) filled.")
        
        realized_pnl = Decimal("0") # Initialize PnL for this fill
        fee = Decimal("0") # TODO: Calculate or fetch fee

        async with AsyncSessionLocal() as session:
            # Verify it exists in DB to avoid FK errors or duplicate processing
            db_order = await session.get(Order, order_id)
            if db_order:
                # TAX LOT LOGIC
                if order["side"] == "BUY":
                    # Create new Tax Lot
                    new_lot = TaxLot(
                        buy_fill_id=None, # Will be updated after fill is added
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
                    
                    # Fetch open lots based on tax lot method
                    order_by = TaxLot.acquired_ts.asc() # Default FIFO
                    if settings.tax_lot_method == "LIFO":
                        order_by = TaxLot.acquired_ts.desc()
                    elif settings.tax_lot_method == "HIFO":
                        order_by = TaxLot.buy_price.desc()
                    
                    stmt = select(TaxLot).where(
                        TaxLot.product_id == self.product_id,
                        TaxLot.remaining_base_size != "0"
                    ).order_by(order_by)
                    
                    open_lots = (await session.execute(stmt)).scalars().all()
                    
                    total_pnl_for_sell = Decimal("0")
                    
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
                        total_pnl_for_sell += pnl
                        
                        # Create match record
                        match = TaxLotMatch(
                            sell_fill_id=None, # Will be updated after fill is added
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
                    
                    realized_pnl = total_pnl_for_sell # Assign to the outer variable
                    self.realized_pnl += total_pnl_for_sell

                # Manage reserves with the calculated PnL
                pnl_to_record = self.shared_risk_state.manage_reserves(self.product_id, realized_pnl)

                # Create Fill record
                new_fill = Fill(
                    ts=time.time(),
                    product_id=self.product_id,
                    side=order["side"],
                    price=str(order["price"]),
                    base_size=str(order["base_size"]),
                    fee_paid=str(fee),
                    grid_index=order["grid_index"],
                    order_id=order_id,
                    tax_lot_method=settings.tax_lot_method,
                    realized_pnl_usd=str(pnl_to_record)
                )
                session.add(new_fill)
                await session.flush() # Get fill.id

                # Update buy_fill_id for new_lot and sell_fill_id for matches if they were created
                if order["side"] == "BUY":
                    if 'new_lot' in locals(): # Check if new_lot was created
                        new_lot.buy_fill_id = new_fill.id
                elif order["side"] == "SELL":
                    for match_obj in session.new:
                        if isinstance(match_obj, TaxLotMatch):
                            match_obj.sell_fill_id = new_fill.id

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
        
        # MACD
        macd_hist = analysis.macd_histogram(
            closes, 
            settings.alpha_macd_fast, 
            settings.alpha_macd_slow, 
            settings.alpha_macd_signal
        )
        
        # ATR
        atr_val = analysis.atr(self.candles, settings.atr_period)
        
        # VPIN (Order Flow Toxicity)
        vpin_val = analysis.vpin(self.candles, settings.vpin_rolling_buckets)

        logger.debug(f"Analysis: RSI={rsi_val:.2f}, Trend={trend}, ATR={atr_val}, MACD_Hist={macd_hist:.4f}, VPIN={vpin_val:.2f}")
        return {
             "rsi": rsi_val,
             "trend": trend,
             "atr": atr_val,
             "macd_hist": macd_hist,
             "vpin": vpin_val,
             "price": self.last_price
        }

    def _detect_regime(self) -> str:
        """Use HMM to detect current market regime."""
        if not settings.hmm_regime_detection_enabled or len(self.candles) < settings.hmm_lookback:
            return "NORMAL"
            
        try:
            closes = [c[3] for c in self.candles[-settings.hmm_lookback:]]
            returns = analysis.calculate_returns(closes)
            
            # Simple HMM fitting (initially) or using pre-trained model
            # For now, we'll use the analysis module's fit function
            res = analysis.fit_gaussian_hmm(
                returns, 
                n_states=settings.hmm_states,
                iterations=settings.hmm_iterations,
                min_variance=settings.hmm_min_variance
            )
            
            if not res:
                return "NORMAL"
                
            init_out, trans_out, means, variances = res
            
            # Identify "Volatile" vs "Quiet" states
            # Quiet: High return/low variance or Mid return/mid variance
            # Volatile: High variance
            idx_max_var = variances.index(max(variances))
            idx_min_var = variances.index(min(variances))
            
            current_probs = analysis.hmm_filter_probabilities(returns[-5:], init_out, trans_out, means, variances)
            if not current_probs:
                return "NORMAL"
                
            current_state = current_probs.index(max(current_probs))
            
            if current_state == idx_max_var:
                return "VOLATILE"
            elif current_state == idx_min_var:
                return "QUIET"
            else:
                return "NORMAL"
                
        except Exception as e:
            logger.warning(f"HMM Regime Detection failed: {e}")
            return "NORMAL"

    def _calculate_grid_levels(self, mid_price: Decimal, band_pct: Decimal) -> List[Decimal]:
        """Calculate grid levels based on spacing mode."""
        upper = mid_price * (1 + band_pct)
        lower = mid_price * (1 - band_pct)
        
        if settings.grid_spacing_mode == "geometric":
            return analysis.calculate_geometric_levels(lower, upper, settings.grid_lines)
        else: # arithmetic
            step = (upper - lower) / (settings.grid_lines - 1)
            return [lower + (step * i) for i in range(settings.grid_lines)]

    async def _execute_grid_logic(self, metrics: Dict[str, Any]):
        """Main decision loop for grid placement and sizing."""
        if not settings.alpha_fusion_enabled:
             # Fallback to basic grid if alpha fusion disabled
             return await self._execute_basic_grid_logic(metrics)

        # 1. Alpha Fusion Sentiment
        rsi_score = Decimal("0")
        if metrics["rsi"] < 30: rsi_score = Decimal("1")
        elif metrics["rsi"] > 70: rsi_score = Decimal("-1")
        
        macd_score = Decimal("1") if metrics["macd_hist"] > 0 else Decimal("-1")
        
        sentiment = (rsi_score * settings.alpha_weight_rsi + macd_score * settings.alpha_weight_macd).quantize(Decimal("0.01"))
        
        # Black-Litterman Adjustment
        bl_view = self.shared_risk_state.black_litterman_views.get(self.product_id, Decimal("0"))
        if bl_view != 0:
            sentiment = (sentiment + bl_view) / Decimal("2")
            logger.debug(f"Black-Litterman Adjusted Sentiment: {sentiment:.2f}")
        
        # Dynamic Sizing with Kelly Criterion
        base_notional = settings.base_order_notional_usd
        if settings.kelly_allocation_enabled:
             # Basic implementation: scale size by sentiment and Kelly fraction
             # In a real bot, we'd use historical win rate/ratios from DB
             kelly_frac = settings.kelly_min_allocation_frac + (abs(sentiment) * (settings.kelly_max_allocation_frac - settings.kelly_min_allocation_frac))
             base_notional = base_notional * kelly_frac
             logger.debug(f"Kelly Adjusted Notional: {base_notional:.2f} USD (Frac={kelly_frac:.2f})")

        # Update dynamic bands if ATR is enabled
        current_band_pct = settings.grid_band_pct
        if settings.atr_enabled:
             atr_val = metrics.get("atr", Decimal("0"))
             if atr_val > 0 and self.last_price > 0:
                  atr_band = (atr_val * settings.atr_band_multiplier) / self.last_price
                  current_band_pct = max(settings.atr_min_band_pct, min(settings.atr_max_band_pct, atr_band))

        # Widen bands if VPIN indicates high toxicity
        if settings.vpin_enabled:
             vpin_val = metrics.get("vpin", Decimal("0"))
             if vpin_val > settings.vpin_threshold_percentile: # threshold used as absolute for now
                  current_band_pct = current_band_pct * settings.vpin_widen_band_multiplier
                  logger.warning(f"High Toxicity detected (VPIN={vpin_val:.2f}). Widening bands to {current_band_pct*100:.2f}%")

        # HMM Regime Adjustment
        regime = self._detect_regime()
        if regime == "VOLATILE":
            current_band_pct = current_band_pct * Decimal("1.25")
            logger.info(f"HMM detected VOLATILE regime. Widening bands to {current_band_pct*100:.2f}%")
        elif regime == "QUIET":
            current_band_pct = current_band_pct * Decimal("0.85")
            logger.info(f"HMM detected QUIET regime. Narrowing bands to {current_band_pct*100:.2f}%")

        # Sentiment Overrides
        if settings.sentiment_override_enabled:
             # This would typically fetch from settings.sentiment_source_url
             # For now, we'll use a placeholder or log the intent
             logger.debug("Sentiment overrides enabled but external source not implemented. Using Alpha Fusion.")

        # Initial Grid Placement
        if not self.orders and self.running and self.last_price > 0:
             mid_price = self.last_price
             self.grid_levels = self._calculate_grid_levels(mid_price, current_band_pct)
             
             logger.info(f"Initializing {settings.grid_spacing_mode.capitalize()} Grid with Alpha Fusion (Sentiment: {sentiment})")
             
             for i, level in enumerate(self.grid_levels):
                  # Bias placement based on sentiment: tilt grid towards buy/sell
                  if level > mid_price * (Decimal("1.001") - sentiment * Decimal("0.005")): 
                       side = "SELL"
                       size = (base_notional / level)
                       await self._place_limit_order(side, level, size, i)
                  elif level < mid_price * (Decimal("0.999") - sentiment * Decimal("0.005")): 
                       side = "BUY"
                       size = (base_notional / level)
                       await self._place_limit_order(side, level, size, i)

    async def _execute_basic_grid_logic(self, metrics: Dict[str, Any]):
        """Legacy/Basic grid logic for when alpha fusion is disabled."""
        current_band_pct = settings.grid_band_pct
        if settings.atr_enabled:
             atr_val = metrics.get("atr", Decimal("0"))
             if atr_val > 0 and self.last_price > 0:
                  atr_band = (atr_val * settings.atr_band_multiplier) / self.last_price
                  current_band_pct = max(settings.atr_min_band_pct, min(settings.atr_max_band_pct, atr_band))

        if not self.orders and self.running and self.last_price > 0:
             mid_price = self.last_price
             self.grid_levels = self._calculate_grid_levels(mid_price, current_band_pct)
             for i, level in enumerate(self.grid_levels):
                  if level > mid_price * Decimal("1.001"): 
                       side = "SELL"
                       size = (settings.base_order_notional_usd / level)
                       await self._place_limit_order(side, level, size, i)
                  elif level < mid_price * Decimal("0.999"): 
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
        # 1. Hard Stop-Loss Check
        stats = await self.get_stats()
        unrealized_pnl = stats["unrealized_pnl"]
        inventory_base = stats["inventory_base"]
        
        if inventory_base > 0 and self.last_price > 0:
            total_value = inventory_base * self.last_price
            if total_value > 0:
                pnl_pct = unrealized_pnl / total_value
                if pnl_pct < -settings.hard_stop_loss_pct:
                    msg = f"⚠️ *CRITICAL*: Hard Stop-Loss reached ({pnl_pct*100:.2f}% < -{settings.hard_stop_loss_pct*100:.2f}%). Stopping strategy and canceling all orders."
                    logger.warning(msg)
                    await notify(msg, force=True)
                    self.stop()
                    # Cancel all open orders for this product
                    order_ids = list(self.active_order_ids)
                    if order_ids:
                        await self.exchange.cancel_orders(order_ids)
                        # TODO: Market sell inventory if required?

    async def on_ws_fill(self, event: Dict[str, Any]):
        """Handle real-time fill from WebSocket."""
        # WebSocket events are faster than REST polling
        # Reconcile if order is in our active set
        order_id = event.get("order_id")
        if isinstance(order_id, str) and order_id in self.orders:
            logger.info(f"WS Fill received for {order_id}")
            await self._handle_fill(order_id)

    async def on_ws_ticker(self, ticker: Dict[str, Any]):
        """Handle real-time price update from WebSocket."""
        price = Decimal(ticker.get("price", "0"))
        if price > 0:
            # Update last_price via Kalman filter for smoothing
            self.last_price = Decimal(str(self.kalman.update(float(price))))
            
            # Micro-price calculation if depth is available
            best_bid = Decimal(ticker.get("best_bid", "0"))
            best_ask = Decimal(ticker.get("best_ask", "0"))
            if best_bid > 0 and best_ask > 0:
                # Note: ticker doesn't always have depth sizes in simple format
                # If available, we'd use analysis.calculate_micro_price(best_bid, bid_size, best_ask, ask_size)
                pass

    def stop(self):
        """Signal the strategy loop to stop."""
        self.running = False

    async def _acquire_ha_lock_db(self) -> bool:
        """Acquire or renew the HA master lock in the database."""
        try:
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    # Check if lock exists
                    from app.database.models import HALock
                    stmt = select(HALock).where(HALock.lock_name == "master")
                    lock = (await session.execute(stmt)).scalar_one_or_none()
                    
                    now = time.time()
                    instance_id = settings.ha_instance_id
                    lease = settings.ha_lock_lease_seconds
                    
                    if not lock:
                        # Create new lock
                        lock = HALock(
                            lock_name="master",
                            holder_id=instance_id,
                            lease_expires_ts=now + lease,
                            holder_ws_sequence_ts=now,
                            updated_ts=now
                        )
                        session.add(lock)
                        return True
                    
                    # Check if existing lock is expired or held by us
                    if lock.lease_expires_ts < now or lock.holder_id == instance_id:
                        lock.holder_id = instance_id
                        lock.lease_expires_ts = now + lease
                        lock.updated_ts = now
                        return True
                    
                    return False
        except Exception as e:
            logger.error(f"Failed to acquire HA lock from DB: {e}")
            return False
