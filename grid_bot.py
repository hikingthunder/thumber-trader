#!/usr/bin/env python3
"""Adaptive Grid Trading Bot for Coinbase Advanced Trade.

Core behavior:
- Neutral grid by default (symmetric around startup price).
- Trend-aware bias using public Coinbase candle data (EMA fast/slow).
- Risk controls to avoid oversized bag-holding exposure.
- Post-only limit orders, local order persistence, and fill replacement.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sqlite3
import stat
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from coinbase.rest import RESTClient

getcontext().prec = 28



@dataclass
class BotConfig:
    product_id: str = os.getenv("PRODUCT_ID", "BTC-USD")
    grid_lines: int = int(os.getenv("GRID_LINES", "8"))
    grid_band_pct: Decimal = Decimal(os.getenv("GRID_BAND_PCT", "0.15"))
    min_notional_usd: Decimal = Decimal(os.getenv("MIN_NOTIONAL_USD", "6"))
    min_grid_profit_pct: Decimal = Decimal(os.getenv("MIN_GRID_PROFIT_PCT", "0.015"))
    maker_fee_pct: Decimal = Decimal(os.getenv("MAKER_FEE_PCT", "0.004"))
    target_net_profit_pct: Decimal = Decimal(os.getenv("TARGET_NET_PROFIT_PCT", "0.002"))
    poll_seconds: int = int(os.getenv("POLL_SECONDS", "60"))

    # live fee adaptation
    dynamic_fee_tracking_enabled: bool = os.getenv("DYNAMIC_FEE_TRACKING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    fee_refresh_seconds: int = int(os.getenv("FEE_REFRESH_SECONDS", "3600"))

    # optional exchange-side bracket attachment for entry buys
    use_exchange_bracket_orders: bool = os.getenv("USE_EXCHANGE_BRACKET_ORDERS", "false").strip().lower() in {"1", "true", "yes", "on"}
    bracket_take_profit_pct: Decimal = Decimal(os.getenv("BRACKET_TAKE_PROFIT_PCT", "0.01"))
    bracket_stop_loss_pct: Decimal = Decimal(os.getenv("BRACKET_STOP_LOSS_PCT", "0.01"))

    # grid shape
    grid_spacing_mode: str = os.getenv("GRID_SPACING_MODE", "arithmetic").strip().lower()

    # volatility-adaptive grid
    atr_enabled: bool = os.getenv("ATR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    atr_period: int = int(os.getenv("ATR_PERIOD", "14"))
    atr_band_multiplier: Decimal = Decimal(os.getenv("ATR_BAND_MULTIPLIER", "4"))
    atr_min_band_pct: Decimal = Decimal(os.getenv("ATR_MIN_BAND_PCT", "0.03"))
    atr_max_band_pct: Decimal = Decimal(os.getenv("ATR_MAX_BAND_PCT", "0.35"))

    # capital and risk controls
    base_order_notional_usd: Decimal = Decimal(os.getenv("BASE_ORDER_NOTIONAL_USD", "10"))
    quote_reserve_pct: Decimal = Decimal(os.getenv("QUOTE_RESERVE_PCT", "0.25"))
    max_btc_inventory_pct: Decimal = Decimal(os.getenv("MAX_BTC_INVENTORY_PCT", "0.65"))
    hard_stop_loss_pct: Decimal = Decimal(os.getenv("HARD_STOP_LOSS_PCT", "0.08"))

    # trend signal controls
    trend_candle_granularity: str = os.getenv("TREND_GRANULARITY", "ONE_HOUR")
    trend_candle_limit: int = int(os.getenv("TREND_CANDLE_LIMIT", "72"))
    trend_ema_fast: int = int(os.getenv("TREND_EMA_FAST", "9"))
    trend_ema_slow: int = int(os.getenv("TREND_EMA_SLOW", "21"))
    trend_strength_threshold: Decimal = Decimal(os.getenv("TREND_STRENGTH_THRESHOLD", "0.003"))
    dynamic_inventory_cap_enabled: bool = os.getenv("DYNAMIC_INVENTORY_CAP_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    inventory_cap_min_pct: Decimal = Decimal(os.getenv("INVENTORY_CAP_MIN_PCT", "0.30"))
    inventory_cap_max_pct: Decimal = Decimal(os.getenv("INVENTORY_CAP_MAX_PCT", "0.80"))

    # execution mode
    paper_trading_mode: bool = os.getenv("PAPER_TRADING_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
    paper_start_usd: Decimal = Decimal(os.getenv("PAPER_START_USD", "1000"))
    paper_start_btc: Decimal = Decimal(os.getenv("PAPER_START_BTC", "0"))
    paper_fill_exceed_pct: Decimal = Decimal(os.getenv("PAPER_FILL_EXCEED_PCT", "0.0001"))
    paper_fill_delay_seconds: int = int(os.getenv("PAPER_FILL_DELAY_SECONDS", "5"))
    paper_slippage_pct: Decimal = Decimal(os.getenv("PAPER_SLIPPAGE_PCT", "0.0001"))

    # local dashboard
    dashboard_enabled: bool = os.getenv("DASHBOARD_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    dashboard_host: str = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8080"))

    # trailing grid behavior
    trailing_grid_enabled: bool = os.getenv("TRAILING_GRID_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    trailing_trigger_levels: int = int(os.getenv("TRAILING_TRIGGER_LEVELS", "2"))

    # persistence
    state_db_path: str = os.getenv("STATE_DB_PATH", "grid_state.db")

    # safe start
    safe_start_enabled: bool = os.getenv("SAFE_START_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    base_buy_mode: str = os.getenv("BASE_BUY_MODE", "off").strip().lower()


class GridBot:
    def __init__(self, client: RESTClient, config: BotConfig, orders_path: Path):
        self.client = client
        self.config = config
        self.orders_path = orders_path
        self.orders: Dict[str, Dict[str, Any]] = self._load_orders()
        self._running = True
        self.grid_levels: List[Decimal] = []
        self.grid_anchor_price = Decimal("0")
        self.price_increment = Decimal("0.01")
        self.base_increment = Decimal("0.00000001")
        self.start_ts = time.time()
        self.loop_count = 0
        self.last_price = Decimal("0")
        self.last_trend_bias = "NEUTRAL"
        self.fill_count = 0
        self.recent_events: List[str] = []
        self._dashboard_server: Optional[ThreadingHTTPServer] = None
        self._state_lock = threading.RLock()
        self._cached_trend_strength = Decimal("0")
        self._active_inventory_cap_pct = self.config.max_btc_inventory_pct
        self.paper_balances = {
            "USD": self.config.paper_start_usd,
            "BTC": self.config.paper_start_btc,
        }
        self._effective_maker_fee_pct = self.config.maker_fee_pct
        self._last_fee_refresh_ts = 0.0
        self._daily_realized_pnl = Decimal("0")
        self._daily_turnover_usd = Decimal("0")
        self._daily_day_index = int(time.time() // 86400)
        self._paper_inventory_cost_usd = self.config.paper_start_btc * self.grid_anchor_price
        self._emergency_stop_triggered = False
        self._db = sqlite3.connect(self.config.state_db_path, check_same_thread=False)
        self._db_lock = threading.RLock()
        self._init_db()
        self._migrate_orders_json_if_needed()

    def run(self) -> None:
        if self.config.dashboard_enabled:
            self._start_dashboard_server()
        self._load_product_metadata()
        self._refresh_maker_fee(force=True)
        current_price = self._get_current_price()
        self.grid_anchor_price = current_price
        if self.config.paper_trading_mode and self.config.paper_start_btc > 0:
            self._paper_inventory_cost_usd = self.config.paper_start_btc * current_price
        self.grid_levels = self._build_grid_levels(current_price)
        self._run_safe_start_checks(current_price)

        if not self.orders:
            logging.info("No persisted orders found; placing initial adaptive grid.")
            self._place_initial_grid_orders(current_price)
            self._save_orders()
        else:
            logging.info("Loaded %s persisted active orders.", len(self.orders))

        while self._running:
            try:
                current_price = self._get_current_price()
                self._refresh_maker_fee()
                trend_bias = self._get_trend_bias()
                with self._state_lock:
                    self.loop_count += 1
                    self.last_price = current_price
                    self.last_trend_bias = trend_bias
                self._risk_monitor(current_price)
                self._maybe_roll_grid(current_price)
                self._process_open_orders(current_price=current_price, trend_bias=trend_bias)
                self._save_orders()
                self._record_daily_stats_snapshot(current_price)
            except Exception as exc:
                logging.exception("Loop error: %s", exc)
            if self._running:
                time.sleep(self.config.poll_seconds)

        self._stop_dashboard_server()

    def _load_product_metadata(self) -> None:
        product = _as_dict(self.client.get_product(product_id=self.config.product_id))
        if product.get("quote_increment"):
            self.price_increment = Decimal(str(product["quote_increment"]))
        if product.get("base_increment"):
            self.base_increment = Decimal(str(product["base_increment"]))
        logging.info("Metadata: quote_increment=%s base_increment=%s", self.price_increment, self.base_increment)

    def _get_current_price(self) -> Decimal:
        product = _as_dict(self.client.get_product(product_id=self.config.product_id))
        raw = product.get("price")
        if raw is None:
            raise RuntimeError(f"Unable to read product price: {product}")
        return Decimal(str(raw))

    def _build_grid_levels(self, current_price: Decimal) -> List[Decimal]:
        if self.config.grid_lines < 2:
            raise ValueError("GRID_LINES must be >= 2")

        effective_band_pct = self._effective_grid_band_pct(current_price)
        lower = current_price * (Decimal("1") - effective_band_pct)
        upper = current_price * (Decimal("1") + effective_band_pct)

        min_required_step_pct = max(
            self.config.min_grid_profit_pct,
            (self._effective_maker_fee_pct * Decimal("2")) + self.config.target_net_profit_pct,
        )

        if self.config.grid_spacing_mode == "geometric":
            ratio = (upper / lower) ** (Decimal("1") / Decimal(self.config.grid_lines - 1))
            levels = [self._q_price(lower * (ratio ** Decimal(i))) for i in range(self.config.grid_lines)]
        else:
            step = (upper - lower) / Decimal(self.config.grid_lines - 1)
            levels = [self._q_price(lower + step * Decimal(i)) for i in range(self.config.grid_lines)]

        levels = sorted(set(levels))
        if len(levels) < 2:
            raise ValueError("Grid precision/increments collapsed levels below usable range; widen band or reduce lines.")

        spacing_pct = (levels[1] - levels[0]) / current_price
        if spacing_pct < min_required_step_pct:
            raise ValueError(
                f"Grid spacing {spacing_pct:.4%} is below required minimum {min_required_step_pct:.4%}."
            )

        logging.info(
            "Grid(%s): lower=%s upper=%s lines=%s step≈%.2f%% band≈%.2f%%",
            self.config.grid_spacing_mode,
            levels[0],
            levels[-1],
            len(levels),
            float(spacing_pct * 100),
            float(effective_band_pct * 100),
        )
        return levels

    def _effective_grid_band_pct(self, current_price: Decimal) -> Decimal:
        if not self.config.atr_enabled:
            return self.config.grid_band_pct

        candles = self._fetch_public_candles()
        atr = _atr(candles, self.config.atr_period)
        if atr <= 0:
            return self.config.grid_band_pct

        dynamic_pct = (atr * self.config.atr_band_multiplier) / current_price
        return max(self.config.atr_min_band_pct, min(dynamic_pct, self.config.atr_max_band_pct))

    def _place_initial_grid_orders(self, current_price: Decimal) -> None:
        trend_bias = self._get_trend_bias()
        buy_levels = [p for p in self.grid_levels if p < current_price]
        sell_levels = [p for p in self.grid_levels if p > current_price]

        if trend_bias == "DOWN":
            # capital defense: avoid catching falling knife aggressively.
            buy_levels = buy_levels[: max(1, len(buy_levels) // 2)]
        elif trend_bias == "UP":
            # still neutral, but keep more room to distribute sells on strength.
            sell_levels = sell_levels[:]

        usd_available = self._get_available_balance("USD")
        btc_available = self._get_available_balance("BTC")
        deployable_usd = usd_available * (Decimal("1") - self.config.quote_reserve_pct)

        buy_budget_per_order = max(self.config.min_notional_usd, self.config.base_order_notional_usd)
        max_buys = int(deployable_usd // buy_budget_per_order)
        buy_levels = buy_levels[:max_buys] if max_buys >= 0 else []

        for level in buy_levels:
            self._place_grid_order(side="BUY", price=level, usd_notional=buy_budget_per_order)

        if sell_levels and btc_available > Decimal("0"):
            btc_per_sell = self._q_base(btc_available / Decimal(len(sell_levels)))
            for level in sell_levels:
                if btc_per_sell * level < self.config.min_notional_usd:
                    continue
                self._place_grid_order(side="SELL", price=level, base_size=btc_per_sell)

        logging.info("Initial grid placed with trend bias=%s (buys=%s sells=%s).", trend_bias, len(buy_levels), len(sell_levels))

    def _process_open_orders(self, current_price: Decimal, trend_bias: str) -> None:
        if not self.orders:
            return

        for order_id, record in list(self.orders.items()):
            status = self._order_status(order_id=order_id, record=record, current_price=current_price)
            if status not in {"FILLED", "DONE", "COMPLETED"}:
                continue

            side = record["side"]
            grid_index = int(record["grid_index"])
            fill_price = Decimal(str(record["price"]))
            self.orders.pop(order_id, None)
            self.fill_count += 1
            self._add_event(f"{side} filled @ {fill_price}")
            self._apply_fill_to_paper_balances(side=side, base_size=Decimal(str(record["base_size"])), price=fill_price)
            self._record_fill(order_id=order_id, record=record, fill_price=fill_price)

            if side == "BUY" and grid_index + 1 < len(self.grid_levels):
                if trend_bias == "DOWN":
                    logging.info("Buy filled at %s; downtrend active, delaying replacement sell check.", fill_price)
                new_price = self.grid_levels[grid_index + 1]
                base_size = self._q_base(Decimal(str(record["base_size"])))
                new_id = self._place_grid_order(side="SELL", price=new_price, base_size=base_size, grid_index=grid_index + 1)
                if new_id:
                    logging.info("Buy Filled at $%s! Placed Sell at $%s.", fill_price, new_price)
                    self._add_event(f"Replacement SELL placed @ {new_price}")

            elif side == "SELL" and grid_index - 1 >= 0:
                if trend_bias == "DOWN":
                    logging.info("Sell filled at %s in downtrend; preserving capital (skip replacement buy).", fill_price)
                    continue
                new_price = self.grid_levels[grid_index - 1]
                usd_notional = max(self.config.min_notional_usd, Decimal(str(record["base_size"])) * fill_price)
                new_id = self._place_grid_order(side="BUY", price=new_price, usd_notional=usd_notional, grid_index=grid_index - 1)
                if new_id:
                    logging.info("Sell Filled at $%s! Placed Buy at $%s.", fill_price, new_price)
                    self._add_event(f"Replacement BUY placed @ {new_price}")

    def _order_status(self, order_id: str, record: Dict[str, Any], current_price: Decimal) -> str:
        if not self.config.paper_trading_mode:
            order = _as_dict(self.client.get_order(order_id=order_id))
            return str(order.get("status", "")).upper()

        side = str(record["side"]).upper()
        price = Decimal(str(record["price"]))
        if time.time() < float(record.get("eligible_fill_ts", 0)):
            return "OPEN"

        exceed = self.config.paper_fill_exceed_pct
        if side == "BUY" and current_price <= price * (Decimal("1") - exceed):
            return "FILLED"
        if side == "SELL" and current_price >= price * (Decimal("1") + exceed):
            return "FILLED"
        return "OPEN"

    def _risk_monitor(self, current_price: Decimal) -> None:
        usd_bal = self._get_available_balance("USD")
        btc_bal = self._get_available_balance("BTC")
        btc_notional = btc_bal * current_price
        portfolio_value = usd_bal + btc_notional

        if portfolio_value <= Decimal("0"):
            return

        btc_ratio = btc_notional / portfolio_value
        if btc_ratio > self._active_inventory_cap_pct:
            logging.warning(
                "Risk: BTC inventory %.2f%% exceeds limit %.2f%%. New buy placements are throttled.",
                float(btc_ratio * 100),
                float(self._active_inventory_cap_pct * 100),
            )

        stop_price = self.grid_anchor_price * (Decimal("1") - self.config.hard_stop_loss_pct)
        if current_price < stop_price:
            logging.warning(
                "Hard stop-loss zone breached (price=%s < %s). Buy replacements remain suspended in downtrend.",
                current_price,
                self._q_price(stop_price),
            )
            if not self._emergency_stop_triggered and self.orders:
                self._emergency_stop_triggered = True
                self._add_event("Hard stop breached: cancelling outstanding orders in batch")
                self._cancel_open_orders_batch()

    def _place_grid_order(
        self,
        side: str,
        price: Decimal,
        grid_index: Optional[int] = None,
        usd_notional: Optional[Decimal] = None,
        base_size: Optional[Decimal] = None,
    ) -> Optional[str]:
        side = side.upper()
        if grid_index is None:
            grid_index = self.grid_levels.index(price)

        price = self._q_price(price)
        if side == "BUY":
            if usd_notional is None:
                raise ValueError("usd_notional is required for BUY")

            # Exposure gate: do not add BTC if portfolio already BTC-heavy.
            if self._btc_inventory_ratio(price) > self._active_inventory_cap_pct:
                logging.warning("Skipped BUY at %s due to BTC inventory cap.", price)
                return None

            usd_notional = max(usd_notional, self.config.min_notional_usd)
            base_size = self._q_base(usd_notional / price)
            if self.config.paper_trading_mode:
                available_usd = self._get_available_balance("USD")
                required = base_size * price
                if required > available_usd:
                    logging.warning("Skipped BUY at %s in paper mode; insufficient USD.", price)
                    return None
        elif side == "SELL":
            if base_size is None:
                raise ValueError("base_size is required for SELL")
            base_size = self._q_base(base_size)
            if self.config.paper_trading_mode and base_size > self._get_available_balance("BTC"):
                logging.warning("Skipped SELL at %s in paper mode; insufficient BTC.", price)
                return None
        else:
            raise ValueError(f"Unsupported side {side}")

        notional = base_size * price
        if notional < self.config.min_notional_usd:
            logging.warning("Skipped %s at %s, notional %s below min %s", side, price, notional, self.config.min_notional_usd)
            return None

        payload = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": self.config.product_id,
            "side": side,
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": format(base_size, "f"),
                    "limit_price": format(price, "f"),
                    "post_only": True,
                }
            },
        }
        if (
            not self.config.paper_trading_mode
            and side == "BUY"
            and self.config.use_exchange_bracket_orders
        ):
            payload["attached_order_configuration"] = {
                "trigger_bracket_gtc": {
                    "base_size": format(base_size, "f"),
                    "limit_price": format(self._q_price(price * (Decimal("1") + self.config.bracket_take_profit_pct)), "f"),
                    "stop_trigger_price": format(self._q_price(price * (Decimal("1") - self.config.bracket_stop_loss_pct)), "f"),
                }
            }
        if self.config.paper_trading_mode:
            order_id = f"paper-{uuid.uuid4()}"
        else:
            response = _as_dict(self.client.create_order(**payload))
            if not bool(response.get("success", True)):
                logging.error("Order rejected: %s", response)
                return None

            order_id = response.get("order_id") or response.get("success_response", {}).get("order_id")
            if not order_id:
                raise RuntimeError(f"Order id missing from response: {response}")

        now_ts = time.time()
        self.orders[order_id] = {
            "side": side,
            "price": str(price),
            "base_size": str(base_size),
            "grid_index": grid_index,
            "product_id": self.config.product_id,
            "created_ts": now_ts,
            "eligible_fill_ts": now_ts + self.config.paper_fill_delay_seconds,
        }
        logging.info("Placed %s LIMIT post-only: %s @ %s", side, base_size, price)
        self._add_event(f"{side} order placed @ {price} (id={order_id[:12]})")
        return order_id

    def _btc_inventory_ratio(self, ref_price: Decimal) -> Decimal:
        usd_bal = self._get_available_balance("USD")
        btc_bal = self._get_available_balance("BTC")
        btc_notional = btc_bal * ref_price
        total = usd_bal + btc_notional
        return Decimal("0") if total <= 0 else btc_notional / total

    def _get_trend_bias(self) -> str:
        closes = self._fetch_public_candle_closes()
        if len(closes) < max(self.config.trend_ema_fast, self.config.trend_ema_slow):
            logging.info("Trend data unavailable/insufficient; using NEUTRAL bias.")
            self._cached_trend_strength = Decimal("0")
            self._active_inventory_cap_pct = self._compute_dynamic_inventory_cap(Decimal("0"))
            return "NEUTRAL"

        ema_fast = _ema(closes, self.config.trend_ema_fast)
        ema_slow = _ema(closes, self.config.trend_ema_slow)
        if ema_slow <= 0:
            self._cached_trend_strength = Decimal("0")
            self._active_inventory_cap_pct = self._compute_dynamic_inventory_cap(Decimal("0"))
            return "NEUTRAL"

        strength = (ema_fast - ema_slow) / ema_slow
        self._cached_trend_strength = strength
        self._active_inventory_cap_pct = self._compute_dynamic_inventory_cap(strength)

        if strength >= self.config.trend_strength_threshold:
            return "UP"
        if strength <= -self.config.trend_strength_threshold:
            return "DOWN"
        return "NEUTRAL"

    def _fetch_public_candle_closes(self) -> List[Decimal]:
        candles = self._fetch_public_candles()
        return [c[3] for c in candles]

    def _fetch_public_candles(self) -> List[Tuple[int, Decimal, Decimal, Decimal]]:
        params = urllib.parse.urlencode(
            {
                "granularity": self.config.trend_candle_granularity,
                "limit": str(self.config.trend_candle_limit),
            }
        )
        url = f"https://api.coinbase.com/api/v3/brokerage/products/{self.config.product_id}/candles?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logging.warning("Trend candle fetch failed: %s", exc)
            return []

        candles = payload.get("candles", [])
        rows: List[Tuple[int, Decimal, Decimal, Decimal]] = []
        for candle in candles:
            close = candle.get("close")
            start = candle.get("start")
            high = candle.get("high")
            low = candle.get("low")
            if close is None or start is None or high is None or low is None:
                continue
            rows.append((int(start), Decimal(str(high)), Decimal(str(low)), Decimal(str(close))))

        rows.sort(key=lambda x: x[0])
        return rows

    def _compute_dynamic_inventory_cap(self, trend_strength: Decimal) -> Decimal:
        if not self.config.dynamic_inventory_cap_enabled:
            return self.config.max_btc_inventory_pct

        cap_min = min(self.config.inventory_cap_min_pct, self.config.inventory_cap_max_pct)
        cap_max = max(self.config.inventory_cap_min_pct, self.config.inventory_cap_max_pct)
        threshold = self.config.trend_strength_threshold if self.config.trend_strength_threshold > 0 else Decimal("0.001")
        normalized = max(Decimal("-1"), min(Decimal("1"), trend_strength / (threshold * Decimal("3"))))
        weight = (normalized + Decimal("1")) / Decimal("2")
        return cap_min + ((cap_max - cap_min) * weight)

    def _q_price(self, value: Decimal) -> Decimal:
        return value.quantize(self.price_increment, rounding=ROUND_DOWN)

    def _q_base(self, value: Decimal) -> Decimal:
        return value.quantize(self.base_increment, rounding=ROUND_DOWN)

    def _get_available_balance(self, currency: str) -> Decimal:
        if self.config.paper_trading_mode:
            return self.paper_balances.get(currency, Decimal("0"))

        payload = _as_dict(self.client.get_accounts())
        for account in payload.get("accounts", []):
            if account.get("currency") == currency:
                raw = account.get("available_balance", {}).get("value")
                if raw is not None:
                    return Decimal(str(raw))
        return Decimal("0")

    def _load_orders(self) -> Dict[str, Dict[str, Any]]:
        db_orders = self._load_orders_from_db()
        if db_orders:
            return db_orders
        if not self.orders_path.exists():
            return {}
        try:
            return json.loads(self.orders_path.read_text())
        except Exception as exc:
            logging.warning("Failed reading %s: %s", self.orders_path, exc)
            return {}

    def _save_orders(self) -> None:
        self._save_orders_to_db()
        self.orders_path.write_text(json.dumps(self.orders, indent=2, sort_keys=True))
        try:
            self.orders_path.chmod(0o600)
        except Exception:
            pass



    def stop(self) -> None:
        self._running = False

    def _apply_fill_to_paper_balances(self, side: str, base_size: Decimal, price: Decimal) -> None:
        if not self.config.paper_trading_mode:
            return

        self._roll_daily_metrics_window()
        slippage = self.config.paper_slippage_pct
        exec_price = price * (Decimal("1") + slippage if side == "BUY" else Decimal("1") - slippage)
        exec_price = self._q_price(exec_price)
        notional = base_size * exec_price
        fee_paid = notional * self._effective_maker_fee_pct
        self._daily_turnover_usd += notional

        if side == "BUY":
            self.paper_balances["USD"] = self.paper_balances["USD"] - notional - fee_paid
            self.paper_balances["BTC"] = self.paper_balances["BTC"] + base_size
            self._paper_inventory_cost_usd += notional + fee_paid
        elif side == "SELL":
            btc_before = self.paper_balances["BTC"]
            avg_cost = (self._paper_inventory_cost_usd / btc_before) if btc_before > 0 else Decimal("0")
            cost_basis = avg_cost * base_size
            proceeds = notional - fee_paid
            self.paper_balances["USD"] = self.paper_balances["USD"] + proceeds
            self.paper_balances["BTC"] = self.paper_balances["BTC"] - base_size
            self._paper_inventory_cost_usd = max(Decimal("0"), self._paper_inventory_cost_usd - cost_basis)
            self._daily_realized_pnl += proceeds - cost_basis

    def _add_event(self, message: str) -> None:
        with self._state_lock:
            self.recent_events.append(f"{time.strftime('%H:%M:%S')} | {message}")
            self.recent_events = self.recent_events[-25:]

    def _roll_daily_metrics_window(self) -> None:
        day_idx = int(time.time() // 86400)
        if day_idx != self._daily_day_index:
            self._daily_day_index = day_idx
            self._daily_realized_pnl = Decimal("0")
            self._daily_turnover_usd = Decimal("0")

    def _refresh_maker_fee(self, force: bool = False) -> None:
        if not self.config.dynamic_fee_tracking_enabled or self.config.paper_trading_mode:
            self._effective_maker_fee_pct = self.config.maker_fee_pct
            return

        now = time.time()
        if not force and now - self._last_fee_refresh_ts < self.config.fee_refresh_seconds:
            return

        self._last_fee_refresh_ts = now
        try:
            fetcher = getattr(self.client, "get_transaction_summary", None)
            payload = _as_dict(fetcher()) if callable(fetcher) else {}
            maker = _find_decimal(payload, ["maker_fee_rate", "maker_fee", "maker_fee_rate_bps"])
            if maker is not None:
                if maker > Decimal("1"):
                    maker = maker / Decimal("10000")
                self._effective_maker_fee_pct = maker
                logging.info("Fee tracker updated maker fee to %.4f%%", float(maker * 100))
            else:
                self._effective_maker_fee_pct = self.config.maker_fee_pct
        except Exception as exc:
            logging.warning("Fee refresh failed (%s); falling back to configured MAKER_FEE_PCT.", exc)
            self._effective_maker_fee_pct = self.config.maker_fee_pct

    def _cancel_open_orders_batch(self) -> None:
        if self.config.paper_trading_mode:
            self.orders.clear()
            return

        order_ids = list(self.orders.keys())
        if not order_ids:
            return

        try:
            self.client.cancel_orders(order_ids=order_ids)
        except Exception:
            logging.warning("Batch cancel unavailable, falling back to single-order cancellation.")
            for order_id in order_ids:
                try:
                    self.client.cancel_orders(order_ids=[order_id])
                except Exception as exc:
                    logging.warning("Failed to cancel order %s: %s", order_id, exc)
        self.orders.clear()

    def _cancel_single_order(self, order_id: str) -> None:
        if self.config.paper_trading_mode:
            self.orders.pop(order_id, None)
            return
        try:
            self.client.cancel_orders(order_ids=[order_id])
        except Exception as exc:
            logging.warning("Failed to cancel order %s: %s", order_id, exc)

    def _maybe_roll_grid(self, current_price: Decimal) -> None:
        if not self.config.trailing_grid_enabled or len(self.grid_levels) < 2:
            return

        step = self.grid_levels[1] - self.grid_levels[0]
        if step <= 0:
            return
        trigger = step * Decimal(self.config.trailing_trigger_levels)
        moved = False

        while current_price > self.grid_levels[-1] + trigger:
            self._roll_grid_up(step)
            moved = True
        while current_price < self.grid_levels[0] - trigger:
            self._roll_grid_down(step)
            moved = True

        if moved:
            self._save_orders()

    def _roll_grid_up(self, step: Decimal) -> None:
        removed_index = 0
        removed_order = self._find_order_by_grid_index_and_side(removed_index, "BUY")
        if removed_order:
            self._cancel_single_order(removed_order)

        new_levels = self.grid_levels[1:] + [self._q_price(self.grid_levels[-1] + step)]
        self.grid_levels = new_levels
        self._reindex_orders_after_shift(direction="up")
        self._place_grid_order(side="SELL", price=self.grid_levels[-1], base_size=self._q_base(self.config.base_order_notional_usd / self.grid_levels[-1]), grid_index=len(self.grid_levels) - 1)
        self._add_event(f"Trailing roll up -> new top {self.grid_levels[-1]}")

    def _roll_grid_down(self, step: Decimal) -> None:
        removed_index = len(self.grid_levels) - 1
        removed_order = self._find_order_by_grid_index_and_side(removed_index, "SELL")
        if removed_order:
            self._cancel_single_order(removed_order)

        new_levels = [self._q_price(self.grid_levels[0] - step)] + self.grid_levels[:-1]
        self.grid_levels = new_levels
        self._reindex_orders_after_shift(direction="down")
        self._place_grid_order(side="BUY", price=self.grid_levels[0], usd_notional=self.config.base_order_notional_usd, grid_index=0)
        self._add_event(f"Trailing roll down -> new bottom {self.grid_levels[0]}")

    def _find_order_by_grid_index_and_side(self, grid_index: int, side: str) -> Optional[str]:
        for order_id, record in self.orders.items():
            if int(record.get("grid_index", -1)) == grid_index and str(record.get("side", "")).upper() == side:
                return order_id
        return None

    def _reindex_orders_after_shift(self, direction: str) -> None:
        to_remove: List[str] = []
        for order_id, record in self.orders.items():
            idx = int(record.get("grid_index", -1))
            if direction == "up":
                if idx <= 0:
                    to_remove.append(order_id)
                else:
                    record["grid_index"] = idx - 1
            else:
                if idx >= len(self.grid_levels) - 1:
                    to_remove.append(order_id)
                else:
                    record["grid_index"] = idx + 1
        for order_id in to_remove:
            self.orders.pop(order_id, None)

    def _risk_metrics(self) -> Dict[str, str]:
        candles = self._fetch_public_candles()
        closes = [c[3] for c in candles]
        if len(closes) < 3:
            return {"var_95_24h_pct": "0", "cvar_95_24h_pct": "0"}

        returns: List[Decimal] = []
        for prev, cur in zip(closes, closes[1:]):
            if prev > 0:
                returns.append((cur - prev) / prev)
        if not returns:
            return {"var_95_24h_pct": "0", "cvar_95_24h_pct": "0"}

        returns.sort()
        idx = max(0, int((len(returns) - 1) * 0.05))
        var = returns[idx]
        tail = returns[: idx + 1]
        cvar = sum(tail, Decimal("0")) / Decimal(len(tail))
        return {
            "var_95_24h_pct": str(var),
            "cvar_95_24h_pct": str(cvar),
        }

    def _status_snapshot(self) -> Dict[str, Any]:
        with self._state_lock:
            price = self.last_price
            usd_bal = self._get_available_balance("USD")
            btc_bal = self._get_available_balance("BTC")
            portfolio = usd_bal + (btc_bal * price if price > 0 else Decimal("0"))
            self._roll_daily_metrics_window()
            capital_used = max(Decimal("1"), portfolio)
            pnl_per_1k = (self._daily_realized_pnl / capital_used) * Decimal("1000")
            turnover_ratio = self._daily_turnover_usd / capital_used
            risk = self._risk_metrics()
            return {
                "product_id": self.config.product_id,
                "paper_trading_mode": self.config.paper_trading_mode,
                "runtime_seconds": int(time.time() - self.start_ts),
                "loop_count": self.loop_count,
                "last_price": str(price),
                "trend_bias": self.last_trend_bias,
                "active_orders": len(self.orders),
                "fills": self.fill_count,
                "balances": {"USD": str(usd_bal), "BTC": str(btc_bal)},
                "portfolio_value_usd": str(portfolio),
                "inventory_cap_pct": str(self._active_inventory_cap_pct),
                "trend_strength": str(self._cached_trend_strength),
                "maker_fee_pct": str(self._effective_maker_fee_pct),
                "daily_realized_pnl_usd": str(self._daily_realized_pnl),
                "daily_pnl_per_1k": str(pnl_per_1k),
                "daily_turnover_ratio": str(turnover_ratio),
                "risk_metrics": risk,
                "recent_events": list(self.recent_events),
                "orders": self.orders,
            }

    def _start_dashboard_server(self) -> None:
        bot = self

        class DashboardHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/status":
                    payload = json.dumps(bot._status_snapshot(), indent=2).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if self.path != "/":
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                    return

                snapshot = bot._status_snapshot()
                html = _render_dashboard_html(snapshot).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/api/action":
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
                    return

                action = str(payload.get("action", "")).strip().lower()
                result = bot._handle_dashboard_action(action=action, payload=payload)
                body = json.dumps(result, indent=2).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self._dashboard_server = ThreadingHTTPServer((self.config.dashboard_host, self.config.dashboard_port), DashboardHandler)
        thread = threading.Thread(target=self._dashboard_server.serve_forever, name="dashboard-server", daemon=True)
        thread.start()
        logging.info("Dashboard available at http://%s:%s", self.config.dashboard_host, self.config.dashboard_port)

    def _stop_dashboard_server(self) -> None:
        if self._dashboard_server is None:
            return
        self._dashboard_server.shutdown()
        self._dashboard_server.server_close()
        self._dashboard_server = None

    def _handle_dashboard_action(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if action == "kill_switch":
            self._add_event("Kill switch activated from dashboard")
            self._cancel_open_orders_batch()
            self.stop()
            return {"ok": True, "action": action}
        if action == "reanchor":
            current_price = self._get_current_price()
            self.grid_anchor_price = current_price
            self.grid_levels = self._build_grid_levels(current_price)
            self._cancel_open_orders_batch()
            self._place_initial_grid_orders(current_price)
            self._add_event(f"Manual re-anchor at {current_price}")
            return {"ok": True, "action": action, "anchor": str(current_price)}
        if action == "reload_config":
            updates = payload.get("updates", {})
            applied = self._apply_runtime_config_updates(updates)
            self._add_event(f"Config hot reload applied: {','.join(applied)}")
            return {"ok": True, "action": action, "applied": applied}
        return {"ok": False, "error": f"unsupported action {action}"}

    def _apply_runtime_config_updates(self, updates: Dict[str, Any]) -> List[str]:
        allowed = {
            "max_btc_inventory_pct": Decimal,
            "target_net_profit_pct": Decimal,
            "base_order_notional_usd": Decimal,
            "quote_reserve_pct": Decimal,
        }
        applied: List[str] = []
        for key, cast in allowed.items():
            if key not in updates:
                continue
            setattr(self.config, key, cast(str(updates[key])))
            applied.append(key)
        return applied

    def _init_db(self) -> None:
        with self._db_lock:
            cur = self._db.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    side TEXT NOT NULL,
                    price TEXT NOT NULL,
                    base_size TEXT NOT NULL,
                    grid_index INTEGER NOT NULL,
                    product_id TEXT NOT NULL,
                    created_ts REAL NOT NULL,
                    eligible_fill_ts REAL NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    product_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price TEXT NOT NULL,
                    base_size TEXT NOT NULL,
                    fee_paid TEXT NOT NULL,
                    grid_index INTEGER NOT NULL,
                    order_id TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    product_id TEXT NOT NULL,
                    pnl_per_1k TEXT NOT NULL,
                    var_95_24h_pct TEXT NOT NULL,
                    turnover_ratio TEXT NOT NULL
                )
            """)
            self._db.commit()

    def _save_orders_to_db(self) -> None:
        with self._db_lock:
            cur = self._db.cursor()
            cur.execute("DELETE FROM orders")
            for order_id, record in self.orders.items():
                cur.execute(
                    """
                    INSERT INTO orders(order_id, side, price, base_size, grid_index, product_id, created_ts, eligible_fill_ts)
                    VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        order_id,
                        str(record.get("side", "")),
                        str(record.get("price", "0")),
                        str(record.get("base_size", "0")),
                        int(record.get("grid_index", 0)),
                        str(record.get("product_id", self.config.product_id)),
                        float(record.get("created_ts", time.time())),
                        float(record.get("eligible_fill_ts", time.time())),
                    ),
                )
            self._db.commit()

    def _load_orders_from_db(self) -> Dict[str, Dict[str, Any]]:
        with self._db_lock:
            cur = self._db.cursor()
            rows = cur.execute("SELECT order_id, side, price, base_size, grid_index, product_id, created_ts, eligible_fill_ts FROM orders").fetchall()
        orders: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            orders[row[0]] = {
                "side": row[1],
                "price": row[2],
                "base_size": row[3],
                "grid_index": row[4],
                "product_id": row[5],
                "created_ts": row[6],
                "eligible_fill_ts": row[7],
            }
        return orders

    def _record_fill(self, order_id: str, record: Dict[str, Any], fill_price: Decimal) -> None:
        fee_paid = fill_price * Decimal(str(record.get("base_size", "0"))) * self._effective_maker_fee_pct
        with self._db_lock:
            self._db.execute(
                """
                INSERT INTO fills(ts, product_id, side, price, base_size, fee_paid, grid_index, order_id)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    time.time(),
                    self.config.product_id,
                    str(record.get("side", "")),
                    str(fill_price),
                    str(record.get("base_size", "0")),
                    str(fee_paid),
                    int(record.get("grid_index", 0)),
                    order_id,
                ),
            )
            self._db.commit()

    def _record_daily_stats_snapshot(self, current_price: Decimal) -> None:
        usd_bal = self._get_available_balance("USD")
        btc_bal = self._get_available_balance("BTC")
        portfolio = usd_bal + (btc_bal * current_price if current_price > 0 else Decimal("0"))
        capital_used = max(Decimal("1"), portfolio)
        pnl_per_1k = (self._daily_realized_pnl / capital_used) * Decimal("1000")
        turnover_ratio = self._daily_turnover_usd / capital_used
        risk = self._risk_metrics()
        with self._db_lock:
            self._db.execute(
                """
                INSERT INTO daily_stats(ts, product_id, pnl_per_1k, var_95_24h_pct, turnover_ratio)
                VALUES(?,?,?,?,?)
                """,
                (time.time(), self.config.product_id, str(pnl_per_1k), str(risk.get("var_95_24h_pct", "0")), str(turnover_ratio)),
            )
            self._db.commit()

    def _migrate_orders_json_if_needed(self) -> None:
        if self._load_orders_from_db() or not self.orders_path.exists():
            return
        try:
            self.orders = json.loads(self.orders_path.read_text())
            self._save_orders_to_db()
        except Exception as exc:
            logging.warning("Orders JSON migration skipped: %s", exc)

    def _run_safe_start_checks(self, current_price: Decimal) -> None:
        if not self.config.safe_start_enabled:
            return
        min_step_pct = (self.config.grid_band_pct * Decimal("2")) / Decimal(max(1, self.config.grid_lines - 1))
        required = (self._effective_maker_fee_pct * Decimal("2")) + self.config.target_net_profit_pct
        if required >= min_step_pct:
            raise ValueError(f"Fee viability failed: required step {required:.4%} >= estimated grid step {min_step_pct:.4%}")

        buy_levels = len([p for p in self.grid_levels if p < current_price])
        sell_levels = len([p for p in self.grid_levels if p > current_price])
        needed_usd = self.config.base_order_notional_usd * Decimal(max(1, buy_levels))
        usd_bal = self._get_available_balance("USD")
        if usd_bal < needed_usd:
            raise ValueError(f"Safe-start failed: USD balance {usd_bal} < required buy-side capital {needed_usd}")

        needed_btc = Decimal("0")
        if sell_levels > 0:
            needed_btc = (self.config.base_order_notional_usd / current_price) * Decimal(sell_levels)
        btc_bal = self._get_available_balance("BTC")
        if btc_bal < needed_btc:
            if self.config.base_buy_mode == "auto":
                shortfall = needed_btc - btc_bal
                self._execute_base_buy(shortfall, current_price)
            else:
                raise ValueError(
                    f"Safe-start failed: BTC balance {btc_bal} < required sell-side inventory {needed_btc}. "
                    "Set BASE_BUY_MODE=auto to acquire initial base inventory."
                )

    def _execute_base_buy(self, base_size: Decimal, current_price: Decimal) -> None:
        if base_size <= 0:
            return
        quote_size = self._q_price(base_size * current_price)
        if self.config.paper_trading_mode:
            self.paper_balances["USD"] -= quote_size
            self.paper_balances["BTC"] += base_size
            self._add_event(f"Auto base-buy (paper): {base_size} BTC")
            return
        payload = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": self.config.product_id,
            "side": "BUY",
            "order_configuration": {"market_market_ioc": {"quote_size": format(quote_size, "f")}},
        }
        _ = self.client.create_order(**payload)
        self._add_event(f"Auto base-buy executed for {base_size} BTC")


def _validate_config(config: BotConfig) -> None:
    if config.grid_lines < 2:
        raise ValueError("GRID_LINES must be >= 2")
    if config.poll_seconds < 5:
        raise ValueError("POLL_SECONDS must be >= 5")
    if config.fee_refresh_seconds < 60:
        raise ValueError("FEE_REFRESH_SECONDS must be >= 60")
    if config.grid_spacing_mode not in {"arithmetic", "geometric"}:
        raise ValueError("GRID_SPACING_MODE must be arithmetic or geometric")
    if config.maker_fee_pct < 0 or config.target_net_profit_pct < 0:
        raise ValueError("MAKER_FEE_PCT and TARGET_NET_PROFIT_PCT must be >= 0")
    if config.atr_period <= 1:
        raise ValueError("ATR_PERIOD must be > 1")
    if config.atr_band_multiplier <= 0:
        raise ValueError("ATR_BAND_MULTIPLIER must be > 0")
    if not (Decimal("0") < config.atr_min_band_pct <= config.atr_max_band_pct < Decimal("1")):
        raise ValueError("ATR_MIN_BAND_PCT and ATR_MAX_BAND_PCT must satisfy 0 < min <= max < 1")
    if config.bracket_take_profit_pct < 0 or config.bracket_stop_loss_pct < 0:
        raise ValueError("Bracket TP/SL percentages must be >= 0")
    if config.base_order_notional_usd <= 0:
        raise ValueError("BASE_ORDER_NOTIONAL_USD must be > 0")
    if config.min_notional_usd <= 0:
        raise ValueError("MIN_NOTIONAL_USD must be > 0")
    if not (Decimal("0") <= config.quote_reserve_pct < Decimal("1")):
        raise ValueError("QUOTE_RESERVE_PCT must be in [0,1)")
    if config.paper_fill_delay_seconds < 0:
        raise ValueError("PAPER_FILL_DELAY_SECONDS must be >= 0")
    if config.paper_fill_exceed_pct < 0 or config.paper_slippage_pct < 0:
        raise ValueError("Paper simulation percentages must be >= 0")
    if not (Decimal("0") < config.max_btc_inventory_pct <= Decimal("1")):
        raise ValueError("MAX_BTC_INVENTORY_PCT must be in (0,1]")
    if not (Decimal("0") < config.inventory_cap_min_pct <= Decimal("1")):
        raise ValueError("INVENTORY_CAP_MIN_PCT must be in (0,1]")
    if not (Decimal("0") < config.inventory_cap_max_pct <= Decimal("1")):
        raise ValueError("INVENTORY_CAP_MAX_PCT must be in (0,1]")
    if config.dashboard_port <= 0:
        raise ValueError("DASHBOARD_PORT must be > 0")
    if config.paper_start_usd < 0 or config.paper_start_btc < 0:
        raise ValueError("PAPER_START_USD and PAPER_START_BTC must be >= 0")
    if config.trailing_trigger_levels < 1:
        raise ValueError("TRAILING_TRIGGER_LEVELS must be >= 1")
    if config.base_buy_mode not in {"off", "auto"}:
        raise ValueError("BASE_BUY_MODE must be off or auto")


def _orders_path() -> Path:
    return Path(os.getenv("ORDERS_PATH", "orders.json"))


def _ema(values: List[Decimal], period: int) -> Decimal:
    if period <= 0 or len(values) < period:
        return Decimal("0")
    alpha = Decimal("2") / Decimal(period + 1)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = (v * alpha) + (ema_val * (Decimal("1") - alpha))
    return ema_val



def _atr(candles: List[Tuple[int, Decimal, Decimal, Decimal]], period: int) -> Decimal:
    if period <= 0 or len(candles) <= period:
        return Decimal("0")

    true_ranges: List[Decimal] = []
    prev_close = candles[0][3]
    for _ts, high, low, close in candles[1:]:
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
        prev_close = close

    if len(true_ranges) < period:
        return Decimal("0")
    return sum(true_ranges[-period:], Decimal("0")) / Decimal(period)

def _read_secret(var_name: str, file_var_name: str) -> Optional[str]:
    direct = os.getenv(var_name)
    if direct:
        return direct.strip()

    secret_path = os.getenv(file_var_name)
    if not secret_path:
        return None

    path = Path(secret_path)
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"Secret file {path} must not be group/world accessible")
    return path.read_text().strip()



def _find_decimal(payload: Any, candidate_keys: List[str]) -> Optional[Decimal]:
    if isinstance(payload, dict):
        for key in candidate_keys:
            if key in payload:
                try:
                    return Decimal(str(payload[key]))
                except Exception:
                    pass
        for value in payload.values():
            found = _find_decimal(value, candidate_keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_decimal(item, candidate_keys)
            if found is not None:
                return found
    return None


def _as_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "__dict__"):
        return dict(response.__dict__)
    raise TypeError(f"Unsupported response type: {type(response)!r}")


def _render_dashboard_html(snapshot: Dict[str, Any]) -> str:
    rows = []
    for oid, order in snapshot.get("orders", {}).items():
        rows.append(
            f"<tr><td>{oid}</td><td>{order.get('side')}</td><td>{order.get('price')}</td>"
            f"<td>{order.get('base_size')}</td><td>{order.get('grid_index')}</td></tr>"
        )
    events = "".join(f"<li>{event}</li>" for event in snapshot.get("recent_events", []))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="refresh" content="10" />
  <title>Thumber Trader Dashboard</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #121a2b;
      --panel-soft: #1a2439;
      --border: #2d3a58;
      --text: #e8eefc;
      --muted: #93a0bf;
      --accent: #4f8cff;
      --danger: #d95f5f;
      --success: #3fb27f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      margin: 0;
      background: radial-gradient(circle at top, #14213a 0%, var(--bg) 48%);
      color: var(--text);
      line-height: 1.4;
    }}
    .container {{ max-width: 1120px; margin: 24px auto; padding: 0 16px 24px; }}
    .header {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 16px; }}
    h1, h2 {{ margin: 0; }}
    .muted {{ color: var(--muted); }}
    .badge {{ background: var(--panel-soft); border: 1px solid var(--border); border-radius: 999px; padding: 6px 10px; font-size: 12px; color: var(--muted); }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin: 14px 0 18px; }}
    .card {{ background: linear-gradient(180deg, #16223a 0%, var(--panel) 100%); border: 1px solid var(--border); padding: 12px; border-radius: 10px; box-shadow: 0 8px 20px rgba(0,0,0,0.22); }}
    .k {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    .v {{ font-size: 18px; font-weight: 600; }}
    .section {{ margin-top: 18px; }}
    .section-header {{ display:flex; align-items:center; justify-content:space-between; gap: 10px; margin-bottom: 8px; }}
    .section-links a {{ color: var(--accent); text-decoration: none; font-size: 12px; margin-left: 10px; }}
    .section-links a:hover {{ text-decoration: underline; }}
    .table-wrap {{ max-height: 300px; overflow: auto; border-radius: 10px; }}
    table {{ width:100%; border-collapse: collapse; font-size:14px; background: var(--panel); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
    td, th {{ border-bottom:1px solid var(--border); padding:10px 12px; text-align:left; }}
    th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }}
    tr:hover td {{ background: rgba(79, 140, 255, 0.06); }}
    .controls {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    .control-item {{ background: var(--panel-soft); border: 1px solid var(--border); border-radius: 10px; padding: 10px; }}
    .control-item p {{ margin: 6px 0 0; font-size: 12px; color: var(--muted); }}
    button {{ width: 100%; border: 1px solid transparent; border-radius: 8px; padding: 10px 12px; color: white; font-weight: 600; cursor: pointer; }}
    button:hover {{ filter: brightness(1.08); }}
    .btn-danger {{ background: var(--danger); }}
    .btn-primary {{ background: var(--accent); }}
    .btn-neutral {{ background: #5f6f95; }}
    #action-status {{ display: none; margin-top: 12px; padding: 10px; border-radius: 8px; font-size: 13px; }}
    #action-status.ok {{ display: block; background: rgba(63,178,127,.15); border: 1px solid rgba(63,178,127,.45); color: #b9f0d6; }}
    #action-status.err {{ display: block; background: rgba(217,95,95,.15); border: 1px solid rgba(217,95,95,.45); color: #ffd1d1; }}
    ul {{ margin-top: 10px; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Thumber Trader Dashboard</h1>
      <span class="badge">Mode: <strong>{'PAPER' if snapshot.get('paper_trading_mode') else 'LIVE'}</strong></span>
    </div>
    <p class="muted">Operational control panel for monitoring open grid orders and runtime bot health.</p>

    <div class="grid">
      <div class="card"><div class="k">Product</div><div class="v">{snapshot.get('product_id')}</div></div>
      <div class="card"><div class="k">Last Price</div><div class="v">{snapshot.get('last_price')}</div></div>
      <div class="card"><div class="k">Trend</div><div class="v">{snapshot.get('trend_bias')}</div></div>
      <div class="card"><div class="k">Portfolio (USD)</div><div class="v">{snapshot.get('portfolio_value_usd')}</div></div>
      <div class="card"><div class="k">Active Orders</div><div class="v">{snapshot.get('active_orders')}</div></div>
      <div class="card"><div class="k">Fills</div><div class="v">{snapshot.get('fills')}</div></div>
    </div>

    <div class="section" id="controls">
      <div class="section-header">
        <h2>Controls</h2>
        <div class="section-links"><a href="#open-orders">Jump to Open Orders</a><a href="#recent-events">Jump to Recent Events</a></div>
      </div>
      <div class="controls">
        <div class="control-item">
          <button class="btn-danger" title="Cancel all active orders and stop the bot loop immediately" onclick="sendAction('kill_switch')">Emergency Kill Switch</button>
          <p>Immediately cancel all open orders and halt execution.</p>
        </div>
        <div class="control-item">
          <button class="btn-primary" title="Rebuild the full grid around current market price" onclick="sendAction('reanchor')">Manual Re-anchor</button>
          <p>Re-center grid levels to current market and republish orders.</p>
        </div>
        <div class="control-item">
          <button class="btn-neutral" title="Change selected runtime limits without restarting the service" onclick="reloadConfig()">Hot Reload Config</button>
          <p>Update selected risk and profit knobs while bot is running.</p>
        </div>
      </div>
      <div id="action-status" aria-live="polite"></div>
    </div>

    <div class="section" id="open-orders">
      <div class="section-header"><h2>Open Orders</h2></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Order ID</th><th>Side</th><th>Price</th><th>Base Size</th><th>Grid Index</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </div>

    <div class="section" id="recent-events">
      <h2>Recent Events</h2>
      <ul>{events}</ul>
      <p><a href="/api/status">JSON API</a></p>
    </div>
  </div>

  <script>
    function renderStatus(msg, ok=true) {{
      const el = document.getElementById('action-status');
      el.className = ok ? 'ok' : 'err';
      el.textContent = msg;
    }}

    async function sendAction(action, updates) {{
      try {{
        const resp = await fetch('/api/action', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{action, updates}})
        }});
        const data = await resp.json();
        const ok = Boolean(data.ok);
        renderStatus(ok ? `Action "${{action}}" completed.` : (data.error || 'Action failed'), ok);
      }} catch (err) {{
        renderStatus(`Request failed: ${{err}}`, false);
      }}
    }}

    function reloadConfig() {{
      const target = prompt('TARGET_NET_PROFIT_PCT override (blank to skip):', '0.002');
      const cap = prompt('MAX_BTC_INVENTORY_PCT override (blank to skip):', '0.65');
      const updates = {{}};
      if (target) updates.target_net_profit_pct = target;
      if (cap) updates.max_btc_inventory_pct = cap;
      if (Object.keys(updates).length === 0) {{
        renderStatus('No config changes were provided.', false);
        return;
      }}
      sendAction('reload_config', updates);
    }}
  </script>
</body>
</html>"""


def build_client() -> RESTClient:
    api_key = _read_secret("COINBASE_API_KEY", "COINBASE_API_KEY_FILE")
    api_secret = _read_secret("COINBASE_API_SECRET", "COINBASE_API_SECRET_FILE")
    if not api_key or not api_secret:
        raise EnvironmentError(
            "Set COINBASE_API_KEY/COINBASE_API_SECRET (or *_FILE variants with chmod 600)."
        )
    return RESTClient(api_key=api_key, api_secret=api_secret)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    client = build_client()
    config = BotConfig()
    _validate_config(config)
    bot = GridBot(client, config, _orders_path())

    def _handle_signal(signum: int, _frame: Any) -> None:
        logging.info("Received signal %s, shutting down cleanly.", signum)
        bot.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    bot.run()


if __name__ == "__main__":
    main()
