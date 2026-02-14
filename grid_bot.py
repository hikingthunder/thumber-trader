#!/usr/bin/env python3
"""Adaptive Grid Trading Bot for Coinbase Advanced Trade.

Core behavior:
- Neutral grid by default (symmetric around startup price).
- Trend-aware bias using public Coinbase candle data (EMA fast/slow).
- Risk controls to avoid oversized bag-holding exposure.
- Post-only limit orders, local order persistence, and fill replacement.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import contextlib
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
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_DOWN, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config_schema import CONFIG_FIELDS
from dashboard_views import render_config_html, render_dashboard_home_html


getcontext().prec = 28





@dataclass
class BotConfig:
    product_id: str = os.getenv("PRODUCT_ID", "BTC-USD")
    product_ids: str = os.getenv("PRODUCT_IDS", os.getenv("PRODUCT_ID", "BTC-USD"))
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
    adx_period: int = int(os.getenv("ADX_PERIOD", "14"))
    adx_ranging_threshold: Decimal = Decimal(os.getenv("ADX_RANGING_THRESHOLD", "20"))
    adx_trending_threshold: Decimal = Decimal(os.getenv("ADX_TRENDING_THRESHOLD", "25"))
    adx_range_band_multiplier: Decimal = Decimal(os.getenv("ADX_RANGE_BAND_MULTIPLIER", "0.8"))
    adx_trend_band_multiplier: Decimal = Decimal(os.getenv("ADX_TREND_BAND_MULTIPLIER", "1.25"))
    adx_trend_order_size_multiplier: Decimal = Decimal(os.getenv("ADX_TREND_ORDER_SIZE_MULTIPLIER", "0.7"))
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
    paper_start_base: Decimal = Decimal(os.getenv("PAPER_START_BASE", os.getenv("PAPER_START_BTC", "0")))
    paper_fill_exceed_pct: Decimal = Decimal(os.getenv("PAPER_FILL_EXCEED_PCT", "0.0001"))
    paper_fill_delay_seconds: int = int(os.getenv("PAPER_FILL_DELAY_SECONDS", "5"))
    paper_slippage_pct: Decimal = Decimal(os.getenv("PAPER_SLIPPAGE_PCT", "0.0001"))

    # local dashboard
    dashboard_enabled: bool = os.getenv("DASHBOARD_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    dashboard_host: str = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8080"))
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    prometheus_path: str = os.getenv("PROMETHEUS_PATH", "/metrics")

    # trailing grid behavior
    trailing_grid_enabled: bool = os.getenv("TRAILING_GRID_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    trailing_trigger_levels: int = int(os.getenv("TRAILING_TRIGGER_LEVELS", "2"))

    # persistence
    state_db_path: str = os.getenv("STATE_DB_PATH", "grid_state.db")
    legacy_orders_json_enabled: bool = os.getenv("LEGACY_ORDERS_JSON_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # safe start
    safe_start_enabled: bool = os.getenv("SAFE_START_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    base_buy_mode: str = os.getenv("BASE_BUY_MODE", "off").strip().lower()
    shared_usd_reserve_enabled: bool = os.getenv("SHARED_USD_RESERVE_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # cross-asset risk controls
    cross_asset_correlation_enabled: bool = os.getenv("CROSS_ASSET_CORRELATION_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    cross_asset_correlation_threshold: Decimal = Decimal(os.getenv("CROSS_ASSET_CORRELATION_THRESHOLD", "0.85"))
    cross_asset_leader_inventory_trigger_pct: Decimal = Decimal(os.getenv("CROSS_ASSET_LEADER_INVENTORY_TRIGGER_PCT", "0.80"))
    cross_asset_inventory_tightening_factor: Decimal = Decimal(os.getenv("CROSS_ASSET_INVENTORY_TIGHTENING_FACTOR", "0.65"))
    cross_asset_inventory_min_pct: Decimal = Decimal(os.getenv("CROSS_ASSET_INVENTORY_MIN_PCT", "0.20"))
    cross_asset_candle_lookback: int = int(os.getenv("CROSS_ASSET_CANDLE_LOOKBACK", "48"))
    cross_asset_refresh_seconds: int = int(os.getenv("CROSS_ASSET_REFRESH_SECONDS", "300"))


class SharedPaperPortfolio:
    def __init__(self, usd: Decimal):
        self.balances: Dict[str, Decimal] = {"USD": usd}
        self.lock = threading.RLock()

    def get_balance(self, currency: str) -> Decimal:
        with self.lock:
            return self.balances.get(currency, Decimal("0"))

    def apply_delta(self, currency: str, delta: Decimal) -> None:
        with self.lock:
            self.balances[currency] = self.balances.get(currency, Decimal("0")) + delta


class SharedRiskState:
    def __init__(self):
        self.lock = threading.RLock()
        self.cross_asset_inventory_caps: Dict[str, Decimal] = {}
        self.pairwise_correlations: Dict[Tuple[str, str], Decimal] = {}
        self.portfolio_beta = Decimal("0")

    def set_inventory_cap(self, product_id: str, cap: Decimal) -> None:
        with self.lock:
            self.cross_asset_inventory_caps[product_id] = cap

    def get_inventory_cap(self, product_id: str) -> Optional[Decimal]:
        with self.lock:
            return self.cross_asset_inventory_caps.get(product_id)

    def set_correlation(self, left: str, right: str, value: Decimal) -> None:
        key = tuple(sorted((left, right)))
        with self.lock:
            self.pairwise_correlations[key] = value

    def get_correlation(self, left: str, right: str) -> Optional[Decimal]:
        key = tuple(sorted((left, right)))
        with self.lock:
            return self.pairwise_correlations.get(key)

    def set_portfolio_beta(self, beta: Decimal) -> None:
        with self.lock:
            self.portfolio_beta = beta

    def get_portfolio_beta(self) -> Decimal:
        with self.lock:
            return self.portfolio_beta


class GridBot:
    def __init__(
        self,
        client: Any,
        config: BotConfig,
        orders_path: Path,
        shared_paper_portfolio: Optional[SharedPaperPortfolio] = None,
        shared_risk_state: Optional[SharedRiskState] = None,
    ):
        self.client = client
        self.config = config
        self.base_currency = self.config.product_id.split("-")[0]
        self.shared_paper_portfolio = shared_paper_portfolio
        self.shared_risk_state = shared_risk_state
        self.orders_path = orders_path
        self._db = sqlite3.connect(self.config.state_db_path, check_same_thread=False)
        self._db_lock = threading.RLock()
        self._init_db()
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
        self._ws_queue: asyncio.Queue[Dict[str, str]] = asyncio.Queue()
        self._ws_client: Optional[Any] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_ws_sequence: Optional[int] = self._load_last_ws_sequence()
        self._cached_trend_strength = Decimal("0")
        self._cached_atr_pct = Decimal("0")
        self._cached_adx = Decimal("0")
        self._market_regime = "UNKNOWN"
        self._active_inventory_cap_pct = self.config.max_btc_inventory_pct
        self.paper_balances = {
            "USD": self.config.paper_start_usd,
            self.base_currency: self.config.paper_start_base,
        }
        self._effective_maker_fee_pct = self.config.maker_fee_pct
        self._last_fee_refresh_ts = 0.0
        self._daily_realized_pnl = Decimal("0")
        self._daily_turnover_usd = Decimal("0")
        self._daily_day_index = int(time.time() // 86400)
        self._paper_inventory_cost_usd = self.config.paper_start_base * self.grid_anchor_price
        self._realized_pnl_total = Decimal("0")
        self._api_latency_buckets_ms = [50, 100, 250, 500, 1000, 2000, 5000, 10000]
        self._api_latency_bucket_counts = [0 for _ in self._api_latency_buckets_ms]
        self._api_latency_observation_count = 0
        self._api_latency_observation_sum_ms = 0.0
        self._emergency_stop_triggered = False
        self._migrate_orders_json_if_needed()

    async def run(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        if self.config.dashboard_enabled:
            self._start_dashboard_server()
        self._load_product_metadata()
        self._refresh_maker_fee(force=True)
        current_price = self._get_current_price()
        self.grid_anchor_price = current_price
        if self.config.paper_trading_mode and self.config.paper_start_base > 0:
            self._paper_inventory_cost_usd = self.config.paper_start_base * current_price
        self.grid_levels = self._build_grid_levels(current_price)
        self._run_safe_start_checks(current_price)

        if not self.orders:
            logging.info("No persisted orders found; placing initial adaptive grid.")
            self._place_initial_grid_orders(current_price)
            self._save_orders()
        else:
            logging.info("Loaded %s persisted active orders.", len(self.orders))

        tasks = [
            asyncio.create_task(self._market_poll_loop(), name="market-poll-loop"),
            asyncio.create_task(self._risk_monitor_loop(), name="risk-monitor-loop"),
            asyncio.create_task(self._ws_user_listener_loop(), name="ws-user-listener"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            self._running = False
            self._close_ws_client()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._stop_dashboard_server()

    async def _market_poll_loop(self) -> None:
        while self._running:
            try:
                current_price = await asyncio.to_thread(self._get_current_price)
                await asyncio.to_thread(self._refresh_maker_fee)
                trend_bias = await asyncio.to_thread(self._get_trend_bias)
                with self._state_lock:
                    self.loop_count += 1
                    self.last_price = current_price
                    self.last_trend_bias = trend_bias
                await asyncio.to_thread(self._maybe_roll_grid, current_price)
                await asyncio.to_thread(self._process_open_orders, current_price, trend_bias)
                await asyncio.to_thread(self._save_orders)
                await asyncio.to_thread(self._record_daily_stats_snapshot, current_price)
            except Exception as exc:
                logging.exception("Market loop error: %s", exc)
            if self._running:
                await asyncio.sleep(self.config.poll_seconds)

    async def _risk_monitor_loop(self) -> None:
        while self._running:
            try:
                current_price = self.last_price or await asyncio.to_thread(self._get_current_price)
                await asyncio.to_thread(self._risk_monitor, current_price)
            except Exception as exc:
                logging.exception("Risk loop error: %s", exc)
            await asyncio.sleep(max(5, self.config.poll_seconds // 2))

    async def _ws_user_listener_loop(self) -> None:
        if self.config.paper_trading_mode:
            return
        while self._running:
            try:
                client = await asyncio.to_thread(self._build_ws_user_client)
                if client is None:
                    logging.info("WSUserClient unavailable; falling back to polling-only fills.")
                    return
                self._ws_client = client
                await asyncio.to_thread(self._ws_subscribe_user_channel, client)
                while self._running:
                    ws_event = await asyncio.wait_for(self._ws_queue.get(), timeout=1.0)
                    if not ws_event:
                        continue
                    event_type = ws_event.get("type", "")
                    if event_type == "reconcile":
                        await asyncio.to_thread(self._reconcile_state_from_exchange)
                        continue
                    filled_order_id = ws_event.get("order_id", "")
                    if filled_order_id and filled_order_id in self.orders:
                        current_price = self.last_price or await asyncio.to_thread(self._get_current_price)
                        trend_bias = self.last_trend_bias or "NEUTRAL"
                        await asyncio.to_thread(self._process_open_orders, current_price, trend_bias)
                        await asyncio.to_thread(self._save_orders)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logging.warning("WebSocket listener error: %s", exc)
                self._close_ws_client()
                await asyncio.sleep(2)

    def _load_product_metadata(self) -> None:
        product = _as_dict(self._coinbase_api_call("get_product", self.client.get_product, product_id=self.config.product_id))
        if product.get("quote_increment"):
            self.price_increment = Decimal(str(product["quote_increment"]))
        if product.get("base_increment"):
            self.base_increment = Decimal(str(product["base_increment"]))
        logging.info("Metadata: quote_increment=%s base_increment=%s", self.price_increment, self.base_increment)

    def _get_current_price(self) -> Decimal:
        product = _as_dict(self._coinbase_api_call("get_product", self.client.get_product, product_id=self.config.product_id))
        raw = product.get("price")
        if raw is None:
            raise RuntimeError(f"Unable to read product price: {product}")
        return Decimal(str(raw))

    def _build_ws_user_client(self) -> Optional[Any]:
        try:
            from coinbase.websocket import WSUserClient
        except Exception as exc:
            logging.warning("WSUserClient import failed: %s", exc)
            return None

        api_key = _read_secret("COINBASE_API_KEY", "COINBASE_API_KEY_FILE")
        api_secret = _read_secret("COINBASE_API_SECRET", "COINBASE_API_SECRET_FILE")
        if not api_key or not api_secret:
            logging.warning("WebSocket disabled: missing Coinbase API credentials.")
            return None

        def _on_message(msg: Any) -> None:
            try:
                payload = msg if isinstance(msg, dict) else json.loads(str(msg))
            except Exception:
                return

            sequence = self._extract_ws_sequence(payload)
            if sequence is not None and self._ws_sequence_gap_detected(sequence):
                logging.warning("WebSocket sequence gap detected; queueing REST reconciliation.")
                if self._event_loop is not None:
                    self._event_loop.call_soon_threadsafe(self._ws_queue.put_nowait, {"type": "reconcile"})
                return

            events = payload.get("events", []) if isinstance(payload, dict) else []
            for event in events:
                if str(event.get("type", "")).lower() != "update":
                    continue
                for order in event.get("orders", []):
                    status = str(order.get("status", "")).upper()
                    if status in {"FILLED", "DONE", "COMPLETED"}:
                        order_id = str(order.get("order_id", "")).strip()
                        if order_id:
                            if self._event_loop is not None:
                                self._event_loop.call_soon_threadsafe(
                                    self._ws_queue.put_nowait,
                                    {"type": "filled", "order_id": order_id},
                                )

        try:
            return WSUserClient(api_key=api_key, api_secret=api_secret, on_message=_on_message)
        except TypeError:
            return WSUserClient(api_key=api_key, api_secret=api_secret, on_message=_on_message, verbose=False)

    def _ws_subscribe_user_channel(self, ws_client: Any) -> None:
        subscribe = getattr(ws_client, "subscribe", None)
        open_fn = getattr(ws_client, "open", None)
        start_fn = getattr(ws_client, "run_forever", None)

        if callable(open_fn):
            open_fn()
        if callable(subscribe):
            subscribe(channels=["user"], product_ids=[self.config.product_id])
        subscribe_user = getattr(ws_client, "subscribe_user", None)
        if callable(subscribe_user):
            subscribe_user(product_ids=[self.config.product_id])
        if callable(start_fn):
            start_fn()

    def _close_ws_client(self) -> None:
        if self._ws_client is None:
            return
        close_fn = getattr(self._ws_client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        self._ws_client = None

    def _extract_ws_sequence(self, payload: Dict[str, Any]) -> Optional[int]:
        for key in ("sequence_num", "sequence", "sequence_number"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return None

    def _ws_sequence_gap_detected(self, sequence: int) -> bool:
        with self._state_lock:
            if self._last_ws_sequence is None:
                self._last_ws_sequence = sequence
                self._save_last_ws_sequence(sequence)
                return False

            if sequence <= self._last_ws_sequence:
                return False

            expected = self._last_ws_sequence + 1
            gap_detected = sequence != expected
            self._last_ws_sequence = sequence
            self._save_last_ws_sequence(sequence)
            return gap_detected

    def _reconcile_state_from_exchange(self) -> None:
        get_orders = getattr(self.client, "get_orders", None)
        if not callable(get_orders):
            logging.warning("State reconciliation skipped: REST client has no get_orders method.")
            return

        payload = _as_dict(self._coinbase_api_call("get_orders", get_orders))
        remote_orders = payload.get("orders", []) if isinstance(payload, dict) else []
        statuses_open = {"OPEN", "PENDING", "ACTIVE", "QUEUED"}
        reconciled: Dict[str, Dict[str, Any]] = {}
        now = time.time()

        for order in remote_orders:
            if str(order.get("product_id", "")).upper() != self.config.product_id.upper():
                continue
            status = str(order.get("status", "")).upper()
            if status and status not in statuses_open:
                continue

            order_id = str(order.get("order_id", "")).strip()
            side = str(order.get("side", "")).upper()
            if not order_id or side not in {"BUY", "SELL"}:
                continue

            order_cfg = order.get("order_configuration", {}).get("limit_limit_gtc", {})
            limit_price = order_cfg.get("limit_price") or order.get("limit_price") or order.get("price")
            base_size = order_cfg.get("base_size") or order.get("base_size")
            if limit_price is None or base_size is None:
                continue

            price = self._q_price(Decimal(str(limit_price)))
            grid_index = min(range(len(self.grid_levels)), key=lambda idx: abs(self.grid_levels[idx] - price)) if self.grid_levels else 0
            reconciled[order_id] = {
                "side": side,
                "price": str(price),
                "base_size": str(self._q_base(Decimal(str(base_size)))),
                "grid_index": grid_index,
                "product_id": self.config.product_id,
                "created_ts": now,
                "eligible_fill_ts": now,
            }

        self.orders = reconciled
        self._save_orders()
        self._add_event(f"State reconciled via REST (open orders: {len(self.orders)})")
        logging.info("State reconciliation complete for %s (%s open orders).", self.config.product_id, len(self.orders))

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
        regime_band_multiplier = self._regime_band_multiplier()
        if not self.config.atr_enabled:
            return self.config.grid_band_pct * regime_band_multiplier

        candles = self._fetch_public_candles()
        atr = _atr(candles, self.config.atr_period)
        if atr <= 0:
            return self.config.grid_band_pct * regime_band_multiplier

        dynamic_pct = (atr * self.config.atr_band_multiplier) / current_price
        atr_band = max(self.config.atr_min_band_pct, min(dynamic_pct, self.config.atr_max_band_pct))
        return atr_band * regime_band_multiplier

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
        base_available = self._get_available_balance(self.base_currency)
        deployable_usd = usd_available * (Decimal("1") - self.config.quote_reserve_pct)

        buy_budget_per_order = self._regime_adjusted_buy_notional(self.config.base_order_notional_usd)
        max_buys = int(deployable_usd // buy_budget_per_order)
        buy_levels = buy_levels[:max_buys] if max_buys >= 0 else []

        for level in buy_levels:
            self._place_grid_order(side="BUY", price=level, usd_notional=buy_budget_per_order)

        if sell_levels and base_available > Decimal("0"):
            base_per_sell = self._q_base(base_available / Decimal(len(sell_levels)))
            for level in sell_levels:
                if base_per_sell * level < self.config.min_notional_usd:
                    continue
                self._place_grid_order(side="SELL", price=level, base_size=base_per_sell)

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
                usd_notional = self._regime_adjusted_buy_notional(Decimal(str(record["base_size"])) * fill_price)
                new_id = self._place_grid_order(side="BUY", price=new_price, usd_notional=usd_notional, grid_index=grid_index - 1)
                if new_id:
                    logging.info("Sell Filled at $%s! Placed Buy at $%s.", fill_price, new_price)
                    self._add_event(f"Replacement BUY placed @ {new_price}")

    def _order_status(self, order_id: str, record: Dict[str, Any], current_price: Decimal) -> str:
        if not self.config.paper_trading_mode:
            order = _as_dict(self._coinbase_api_call("get_order", self.client.get_order, order_id=order_id))
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
        base_bal = self._get_available_balance(self.base_currency)
        base_notional = base_bal * current_price
        portfolio_value = usd_bal + base_notional

        if portfolio_value <= Decimal("0"):
            return

        base_ratio = base_notional / portfolio_value
        effective_cap = self._effective_inventory_cap_pct()
        if base_ratio > effective_cap:
            logging.warning(
                "Risk: %s inventory %.2f%% exceeds limit %.2f%%. New buy placements are throttled.",
                self.base_currency,
                float(base_ratio * 100),
                float(effective_cap * 100),
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

            # Exposure gate: do not add base asset if portfolio is already base-heavy.
            if self._btc_inventory_ratio(price) > self._effective_inventory_cap_pct():
                logging.warning("Skipped BUY at %s due to %s inventory cap.", price, self.base_currency)
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
            if self.config.paper_trading_mode and base_size > self._get_available_balance(self.base_currency):
                logging.warning("Skipped SELL at %s in paper mode; insufficient %s.", price, self.base_currency)
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
        base_bal = self._get_available_balance(self.base_currency)
        base_notional = base_bal * ref_price
        total = usd_bal + base_notional
        return Decimal("0") if total <= 0 else base_notional / total

    def _effective_inventory_cap_pct(self) -> Decimal:
        cap = self._active_inventory_cap_pct
        if self.shared_risk_state is not None:
            shared_cap = self.shared_risk_state.get_inventory_cap(self.config.product_id)
            if shared_cap is not None:
                cap = min(cap, shared_cap)
        return cap

    def _get_trend_bias(self) -> str:
        candles = self._fetch_public_candles()
        closes = [c[3] for c in candles]
        self._cached_atr_pct = self._estimate_atr_pct(candles)
        self._cached_adx = _adx(candles, self.config.adx_period)
        self._market_regime = self._classify_market_regime(self._cached_adx)
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

    def _estimate_atr_pct(self, candles: List[Tuple[int, Decimal, Decimal, Decimal]]) -> Decimal:
        period = max(1, self.config.atr_period)
        if len(candles) < period + 1:
            return Decimal("0")
        trs: List[Decimal] = []
        prev_close = candles[0][3]
        for _ts, high, low, close in candles[1:]:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
            prev_close = close
        window = trs[-period:]
        if not window:
            return Decimal("0")
        atr = sum(window, Decimal("0")) / Decimal(len(window))
        close = candles[-1][3]
        return Decimal("0") if close <= 0 else atr / close

    def ranging_score(self) -> Decimal:
        trend_strength = abs(self._cached_trend_strength)
        return max(Decimal("0"), self._cached_atr_pct - trend_strength)

    def _classify_market_regime(self, adx_value: Decimal) -> str:
        if adx_value >= self.config.adx_trending_threshold:
            return "TRENDING"
        if adx_value <= self.config.adx_ranging_threshold:
            return "RANGING"
        return "TRANSITION"

    def _regime_band_multiplier(self) -> Decimal:
        if self._market_regime == "TRENDING":
            return self.config.adx_trend_band_multiplier
        if self._market_regime == "RANGING":
            return self.config.adx_range_band_multiplier
        return Decimal("1")

    def _regime_adjusted_buy_notional(self, usd_notional: Decimal) -> Decimal:
        adjusted = usd_notional
        if self._market_regime == "TRENDING":
            adjusted = usd_notional * self.config.adx_trend_order_size_multiplier
        return max(self.config.min_notional_usd, adjusted)

    def _fetch_public_candles(self) -> List[Tuple[int, Decimal, Decimal, Decimal]]:
        params = urllib.parse.urlencode(
            {
                "granularity": self.config.trend_candle_granularity,
                "limit": str(self.config.trend_candle_limit),
            }
        )
        url = f"https://api.coinbase.com/api/v3/brokerage/products/{self.config.product_id}/candles?{params}"
        try:
            with self._coinbase_api_call("public_candles", urllib.request.urlopen, url, timeout=15) as resp:
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
            if self.shared_paper_portfolio is not None:
                return self.shared_paper_portfolio.get_balance(currency)
            return self.paper_balances.get(currency, Decimal("0"))

        payload = _as_dict(self._coinbase_api_call("get_accounts", self.client.get_accounts))
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
        if self.config.legacy_orders_json_enabled:
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
            if self.shared_paper_portfolio is not None:
                self.shared_paper_portfolio.apply_delta("USD", -(notional + fee_paid))
                self.shared_paper_portfolio.apply_delta(self.base_currency, base_size)
            self.paper_balances["USD"] = self.paper_balances.get("USD", Decimal("0")) - notional - fee_paid
            self.paper_balances[self.base_currency] = self.paper_balances.get(self.base_currency, Decimal("0")) + base_size
            self._paper_inventory_cost_usd += notional + fee_paid
        elif side == "SELL":
            base_before = self.paper_balances.get(self.base_currency, Decimal("0"))
            avg_cost = (self._paper_inventory_cost_usd / base_before) if base_before > 0 else Decimal("0")
            cost_basis = avg_cost * base_size
            proceeds = notional - fee_paid
            if self.shared_paper_portfolio is not None:
                self.shared_paper_portfolio.apply_delta("USD", proceeds)
                self.shared_paper_portfolio.apply_delta(self.base_currency, -base_size)
            self.paper_balances["USD"] = self.paper_balances.get("USD", Decimal("0")) + proceeds
            self.paper_balances[self.base_currency] = self.paper_balances.get(self.base_currency, Decimal("0")) - base_size
            self._paper_inventory_cost_usd = max(Decimal("0"), self._paper_inventory_cost_usd - cost_basis)
            realized = proceeds - cost_basis
            self._daily_realized_pnl += realized
            self._realized_pnl_total += realized

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
            base_bal = self._get_available_balance(self.base_currency)
            portfolio = usd_bal + (base_bal * price if price > 0 else Decimal("0"))
            self._roll_daily_metrics_window()
            capital_used = max(Decimal("1"), portfolio)
            pnl_per_1k = (self._daily_realized_pnl / capital_used) * Decimal("1000")
            portfolio_beta = self.shared_risk_state.get_portfolio_beta() if self.shared_risk_state is not None else Decimal("0")
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
                "balances": {"USD": str(usd_bal), self.base_currency: str(base_bal)},
                "portfolio_value_usd": str(portfolio),
                "inventory_cap_pct": str(self._effective_inventory_cap_pct()),
                "trend_strength": str(self._cached_trend_strength),
                "adx": str(self._cached_adx),
                "market_regime": self._market_regime,
                "atr_pct": str(self._cached_atr_pct),
                "ranging_score": str(self.ranging_score()),
                "maker_fee_pct": str(self._effective_maker_fee_pct),
                "daily_realized_pnl_usd": str(self._daily_realized_pnl),
                "realized_pnl_total_usd": str(self._realized_pnl_total),
                "daily_pnl_per_1k": str(pnl_per_1k),
                "daily_turnover_ratio": str(turnover_ratio),
                "risk_metrics": risk,
                "portfolio_beta": str(portfolio_beta),
                "recent_events": list(self.recent_events),
                "orders": self.orders,
                "config": self._config_snapshot(),
            }

    def _tax_report_csv(self, year: Optional[int] = None) -> str:
        rows = self._fetch_fills(year=year)
        lines = ["ts_iso,product_id,side,price,base_size,gross_notional_usd,fee_paid_usd,net_notional_usd,order_id,grid_index"]
        for row in rows:
            ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(row[0])))
            price = Decimal(str(row[3]))
            base_size = Decimal(str(row[4]))
            gross = price * base_size
            fee = Decimal(str(row[5]))
            net = gross - fee if str(row[2]).upper() == "SELL" else gross + fee
            lines.append(
                ",".join(
                    [
                        ts_iso,
                        str(row[1]),
                        str(row[2]),
                        str(row[3]),
                        str(row[4]),
                        str(gross),
                        str(row[5]),
                        str(net),
                        str(row[6]),
                        str(row[7]),
                    ]
                )
            )
        return "\n".join(lines) + "\n"

    def _fetch_fills(self, year: Optional[int] = None) -> List[Tuple[Any, ...]]:
        query = "SELECT ts, product_id, side, price, base_size, fee_paid, order_id, grid_index FROM fills"
        params: Tuple[Any, ...] = ()
        if year is not None:
            start_ts = time.mktime(time.strptime(f"{year}-01-01", "%Y-%m-%d"))
            end_ts = time.mktime(time.strptime(f"{year + 1}-01-01", "%Y-%m-%d"))
            query += " WHERE ts >= ? AND ts < ?"
            params = (start_ts, end_ts)
        query += " ORDER BY ts ASC"
        with self._db_lock:
            cur = self._db.cursor()
            return cur.execute(query, params).fetchall()

    def _config_snapshot(self) -> Dict[str, str]:
        snapshot: Dict[str, str] = {}
        for field in CONFIG_FIELDS:
            value = getattr(self.config, field["attr"])
            snapshot[field["env"]] = str(value)
        return snapshot

    def _parse_config_value(self, raw_value: str, value_type: str) -> Any:
        value = raw_value.strip()
        if value_type == "str":
            return value
        if value_type == "int":
            return int(value)
        if value_type == "decimal":
            return Decimal(value)
        if value_type == "bool":
            return value.lower() in {"1", "true", "yes", "on"}
        raise ValueError(f"Unsupported config type: {value_type}")

    def _start_dashboard_server(self) -> None:
        bot = self

        class DashboardHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/api/status":
                    payload = json.dumps(bot._status_snapshot(), indent=2).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if bot.config.prometheus_enabled and parsed.path == bot.config.prometheus_path:
                    payload = bot._prometheus_metrics_payload().encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if parsed.path == "/api/tax_report.csv":
                    year = None
                    params = urllib.parse.parse_qs(parsed.query)
                    if "year" in params and params["year"]:
                        try:
                            year = int(params["year"][0])
                        except ValueError:
                            self.send_error(HTTPStatus.BAD_REQUEST, "year must be an integer")
                            return
                    payload = bot._tax_report_csv(year=year).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", "attachment; filename=tax_report.csv")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if parsed.path == "/config":
                    snapshot = bot._status_snapshot()
                    params = urllib.parse.parse_qs(parsed.query)
                    popup_mode = params.get("popup", ["0"])[0] in {"1", "true", "yes", "on"}
                    html_page = render_config_html(snapshot, popup_mode=popup_mode).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html_page)))
                    self.end_headers()
                    self.wfile.write(html_page)
                    return

                if parsed.path != "/":
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                    return

                snapshot = bot._status_snapshot()
                html = render_dashboard_home_html(snapshot).encode("utf-8")
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

    def _coinbase_api_call(self, operation: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
            self._observe_api_latency(elapsed_ms)

    def _observe_api_latency(self, elapsed_ms: float) -> None:
        with self._state_lock:
            self._api_latency_observation_count += 1
            self._api_latency_observation_sum_ms += elapsed_ms
            idx = bisect.bisect_left(self._api_latency_buckets_ms, elapsed_ms)
            if idx >= len(self._api_latency_bucket_counts):
                return
            self._api_latency_bucket_counts[idx] += 1

    def _prometheus_metrics_payload(self) -> str:
        with self._state_lock:
            price = self.last_price
            usd_bal = self._get_available_balance("USD")
            base_bal = self._get_available_balance(self.base_currency)
            portfolio = usd_bal + (base_bal * price if price > 0 else Decimal("0"))
            inventory_ratio = self._btc_inventory_ratio(price if price > 0 else Decimal("0"))
            capital_used = max(Decimal("1"), portfolio)
            pnl_per_1k = (self._daily_realized_pnl / capital_used) * Decimal("1000")
            portfolio_beta = self.shared_risk_state.get_portfolio_beta() if self.shared_risk_state is not None else Decimal("0")

            labels = f'product_id="{self.config.product_id}"'
            lines = [
                "# HELP bot_realized_pnl_usd Total realized profit in USD.",
                "# TYPE bot_realized_pnl_usd gauge",
                f"bot_realized_pnl_usd{{{labels}}} {float(self._realized_pnl_total)}",
                "# HELP bot_inventory_ratio Base asset notional / total portfolio value ratio.",
                "# TYPE bot_inventory_ratio gauge",
                f"bot_inventory_ratio{{{labels}}} {float(inventory_ratio)}",
                "# HELP bot_equity_curve_usd Mark-to-market equity in USD.",
                "# TYPE bot_equity_curve_usd gauge",
                f"bot_equity_curve_usd{{{labels}}} {float(portfolio)}",
                "# HELP bot_pnl_per_1k Daily realized PnL per $1k of capital in USD.",
                "# TYPE bot_pnl_per_1k gauge",
                f"bot_pnl_per_1k{{{labels}}} {float(pnl_per_1k)}",
                "# HELP bot_portfolio_beta Portfolio beta versus BTC benchmark returns.",
                "# TYPE bot_portfolio_beta gauge",
                f"bot_portfolio_beta{{{labels}}} {float(portfolio_beta)}",
                "# HELP api_latency_milliseconds Coinbase API call latency histogram in milliseconds.",
                "# TYPE api_latency_milliseconds histogram",
            ]

            cumulative = 0
            for limit, count in zip(self._api_latency_buckets_ms, self._api_latency_bucket_counts):
                cumulative += count
                lines.append(f'api_latency_milliseconds_bucket{{{labels},le="{limit}"}} {cumulative}')
            lines.append(f'api_latency_milliseconds_bucket{{{labels},le="+Inf"}} {self._api_latency_observation_count}')
            lines.append(f"api_latency_milliseconds_sum{{{labels}}} {self._api_latency_observation_sum_ms}")
            lines.append(f"api_latency_milliseconds_count{{{labels}}} {self._api_latency_observation_count}")
            return "\n".join(lines) + "\n"

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
        if action == "save_config":
            updates = payload.get("updates", {})
            env_path = str(payload.get("env_path", os.getenv("BOT_ENV_PATH", ".env"))).strip() or ".env"
            saved = self._save_config_updates(Path(env_path), updates)
            self._add_event(f"Config file updated: {','.join(saved)}")
            return {"ok": True, "action": action, "saved": saved, "env_path": env_path}
        return {"ok": False, "error": f"unsupported action {action}"}

    def _apply_runtime_config_updates(self, updates: Dict[str, Any]) -> List[str]:
        applied: List[str] = []
        field_map = {f["env"]: f for f in CONFIG_FIELDS}
        for env_name, raw in updates.items():
            field = field_map.get(env_name)
            if field is None:
                continue
            parsed = self._parse_config_value(str(raw), field["type"])
            setattr(self.config, field["attr"], parsed)
            applied.append(env_name)
        _validate_config(self.config)
        return applied

    def _save_config_updates(self, env_path: Path, updates: Dict[str, Any]) -> List[str]:
        if not updates:
            return []
        field_map = {f["env"]: f for f in CONFIG_FIELDS}
        normalized: Dict[str, str] = {}
        for env_name, raw in updates.items():
            field = field_map.get(env_name)
            if field is None:
                continue
            parsed = self._parse_config_value(str(raw), field["type"])
            setattr(self.config, field["attr"], parsed)
            normalized[env_name] = str(parsed).lower() if field["type"] == "bool" else str(parsed)

        _validate_config(self.config)
        existing = env_path.read_text().splitlines() if env_path.exists() else []
        output_lines: List[str] = []
        consumed = set()
        for line in existing:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in line:
                output_lines.append(line)
                continue
            key, _val = line.split("=", 1)
            key = key.strip()
            if key in normalized:
                output_lines.append(f"{key}={normalized[key]}")
                consumed.add(key)
            else:
                output_lines.append(line)

        if normalized and output_lines and output_lines[-1].strip() != "":
            output_lines.append("")
        for key in sorted(normalized.keys() - consumed):
            output_lines.append(f"{key}={normalized[key]}")

        env_path.write_text("\n".join(output_lines).rstrip() + "\n")
        return sorted(normalized.keys())

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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS state_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            self._db.commit()

    def _load_last_ws_sequence(self) -> Optional[int]:
        with self._db_lock:
            row = self._db.execute("SELECT value FROM state_meta WHERE key='last_ws_sequence'").fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return int(str(row[0]))
        except (TypeError, ValueError):
            return None

    def _save_last_ws_sequence(self, sequence: int) -> None:
        with self._db_lock:
            self._db.execute(
                "INSERT OR REPLACE INTO state_meta(key, value) VALUES('last_ws_sequence', ?)",
                (str(sequence),),
            )
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
        base_bal = self._get_available_balance(self.base_currency)
        portfolio = usd_bal + (base_bal * current_price if current_price > 0 else Decimal("0"))
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
        base_bal = self._get_available_balance(self.base_currency)
        if base_bal < needed_btc:
            if self.config.base_buy_mode == "auto":
                shortfall = needed_btc - base_bal
                self._execute_base_buy(shortfall, current_price)
            else:
                raise ValueError(
                    f"Safe-start failed: {self.base_currency} balance {base_bal} < required sell-side inventory {needed_btc}. "
                    "Set BASE_BUY_MODE=auto to acquire initial base inventory."
                )

    def _execute_base_buy(self, base_size: Decimal, current_price: Decimal) -> None:
        if base_size <= 0:
            return
        quote_size = self._q_price(base_size * current_price)
        if self.config.paper_trading_mode:
            self.paper_balances["USD"] -= quote_size
            self.paper_balances[self.base_currency] += base_size
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
    if config.adx_period <= 1:
        raise ValueError("ADX_PERIOD must be > 1")
    if config.adx_ranging_threshold < 0 or config.adx_trending_threshold < 0:
        raise ValueError("ADX thresholds must be >= 0")
    if config.adx_ranging_threshold >= config.adx_trending_threshold:
        raise ValueError("ADX_RANGING_THRESHOLD must be less than ADX_TRENDING_THRESHOLD")
    if config.adx_range_band_multiplier <= 0 or config.adx_trend_band_multiplier <= 0:
        raise ValueError("ADX band multipliers must be > 0")
    if config.adx_trend_order_size_multiplier <= 0:
        raise ValueError("ADX_TREND_ORDER_SIZE_MULTIPLIER must be > 0")
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
    if config.paper_start_usd < 0 or config.paper_start_btc < 0 or config.paper_start_base < 0:
        raise ValueError("PAPER_START_USD, PAPER_START_BTC, and PAPER_START_BASE must be >= 0")
    if config.trailing_trigger_levels < 1:
        raise ValueError("TRAILING_TRIGGER_LEVELS must be >= 1")
    if config.base_buy_mode not in {"off", "auto"}:
        raise ValueError("BASE_BUY_MODE must be off or auto")
    if not any(item.strip() for item in config.product_ids.replace(";", ",").split(",")):
        raise ValueError("PRODUCT_IDS must include at least one product id")
    if not (Decimal("0") <= config.cross_asset_correlation_threshold <= Decimal("1")):
        raise ValueError("CROSS_ASSET_CORRELATION_THRESHOLD must be in [0,1]")
    if not (Decimal("0") < config.cross_asset_leader_inventory_trigger_pct <= Decimal("1.5")):
        raise ValueError("CROSS_ASSET_LEADER_INVENTORY_TRIGGER_PCT must be in (0,1.5]")
    if not (Decimal("0") < config.cross_asset_inventory_tightening_factor <= Decimal("1")):
        raise ValueError("CROSS_ASSET_INVENTORY_TIGHTENING_FACTOR must be in (0,1]")
    if not (Decimal("0") < config.cross_asset_inventory_min_pct <= Decimal("1")):
        raise ValueError("CROSS_ASSET_INVENTORY_MIN_PCT must be in (0,1]")
    if config.cross_asset_candle_lookback < 2:
        raise ValueError("CROSS_ASSET_CANDLE_LOOKBACK must be >= 2")
    if config.cross_asset_refresh_seconds < 5:
        raise ValueError("CROSS_ASSET_REFRESH_SECONDS must be >= 5")


def _orders_path() -> Path:
    return Path(os.getenv("ORDERS_PATH", "orders.json"))


def export_tax_report_csv(db_path: str, output_path: Path, year: Optional[int] = None) -> int:
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT ts, product_id, side, price, base_size, fee_paid, order_id, grid_index FROM fills"
        params: Tuple[Any, ...] = ()
        if year is not None:
            start_ts = time.mktime(time.strptime(f"{year}-01-01", "%Y-%m-%d"))
            end_ts = time.mktime(time.strptime(f"{year + 1}-01-01", "%Y-%m-%d"))
            query += " WHERE ts >= ? AND ts < ?"
            params = (start_ts, end_ts)
        query += " ORDER BY ts ASC"
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    lines = ["ts_iso,product_id,side,price,base_size,gross_notional_usd,fee_paid_usd,net_notional_usd,order_id,grid_index"]
    for row in rows:
        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(row[0])))
        price = Decimal(str(row[3]))
        base_size = Decimal(str(row[4]))
        gross = price * base_size
        fee = Decimal(str(row[5]))
        net = gross - fee if str(row[2]).upper() == "SELL" else gross + fee
        lines.append(
            ",".join(
                [
                    ts_iso,
                    str(row[1]),
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                    str(gross),
                    str(row[5]),
                    str(net),
                    str(row[6]),
                    str(row[7]),
                ]
            )
        )

    output_path.write_text("\n".join(lines) + "\n")
    return len(rows)


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

def _adx(candles: List[Tuple[int, Decimal, Decimal, Decimal]], period: int) -> Decimal:
    if period <= 1 or len(candles) < (period * 2):
        return Decimal("0")

    plus_dm_values: List[Decimal] = []
    minus_dm_values: List[Decimal] = []
    tr_values: List[Decimal] = []

    prev_high = candles[0][1]
    prev_low = candles[0][2]
    prev_close = candles[0][3]

    for _ts, high, low, close in candles[1:]:
        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = up_move if up_move > down_move and up_move > 0 else Decimal("0")
        minus_dm = down_move if down_move > up_move and down_move > 0 else Decimal("0")
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

        plus_dm_values.append(plus_dm)
        minus_dm_values.append(minus_dm)
        tr_values.append(tr)

        prev_high = high
        prev_low = low
        prev_close = close

    if len(tr_values) < period:
        return Decimal("0")

    atr = sum(tr_values[:period], Decimal("0"))
    plus_dm_smoothed = sum(plus_dm_values[:period], Decimal("0"))
    minus_dm_smoothed = sum(minus_dm_values[:period], Decimal("0"))

    dx_values: List[Decimal] = []

    def _append_dx(current_atr: Decimal, current_plus_dm: Decimal, current_minus_dm: Decimal) -> None:
        if current_atr <= 0:
            return
        plus_di = (current_plus_dm / current_atr) * Decimal("100")
        minus_di = (current_minus_dm / current_atr) * Decimal("100")
        denominator = plus_di + minus_di
        if denominator <= 0:
            return
        dx_values.append((abs(plus_di - minus_di) / denominator) * Decimal("100"))

    _append_dx(atr, plus_dm_smoothed, minus_dm_smoothed)

    for idx in range(period, len(tr_values)):
        atr = atr - (atr / Decimal(period)) + tr_values[idx]
        plus_dm_smoothed = plus_dm_smoothed - (plus_dm_smoothed / Decimal(period)) + plus_dm_values[idx]
        minus_dm_smoothed = minus_dm_smoothed - (minus_dm_smoothed / Decimal(period)) + minus_dm_values[idx]
        _append_dx(atr, plus_dm_smoothed, minus_dm_smoothed)

    if not dx_values:
        return Decimal("0")

    lookback = min(period, len(dx_values))
    return sum(dx_values[-lookback:], Decimal("0")) / Decimal(lookback)


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


def _returns_from_closes(closes: List[Decimal]) -> List[Decimal]:
    returns: List[Decimal] = []
    for prev, cur in zip(closes, closes[1:]):
        if prev > 0:
            returns.append((cur - prev) / prev)
    return returns


def _pearson_corr(xs: List[Decimal], ys: List[Decimal]) -> Decimal:
    n = min(len(xs), len(ys))
    if n < 2:
        return Decimal("0")
    xs = xs[-n:]
    ys = ys[-n:]
    mean_x = sum(xs, Decimal("0")) / Decimal(n)
    mean_y = sum(ys, Decimal("0")) / Decimal(n)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / Decimal(n)
    var_x = sum((x - mean_x) * (x - mean_x) for x in xs) / Decimal(n)
    var_y = sum((y - mean_y) * (y - mean_y) for y in ys) / Decimal(n)
    if var_x <= 0 or var_y <= 0:
        return Decimal("0")
    return cov / ((var_x.sqrt()) * (var_y.sqrt()))


def _beta(asset_returns: List[Decimal], benchmark_returns: List[Decimal]) -> Decimal:
    n = min(len(asset_returns), len(benchmark_returns))
    if n < 2:
        return Decimal("0")
    x = benchmark_returns[-n:]
    y = asset_returns[-n:]
    mean_x = sum(x, Decimal("0")) / Decimal(n)
    mean_y = sum(y, Decimal("0")) / Decimal(n)
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / Decimal(n)
    var_x = sum((xi - mean_x) * (xi - mean_x) for xi in x) / Decimal(n)
    if var_x <= 0:
        return Decimal("0")
    return cov / var_x


def _as_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "__dict__"):
        return dict(response.__dict__)
    raise TypeError(f"Unsupported response type: {type(response)!r}")


class GridManager:
    def __init__(self, client: Any, config: BotConfig):
        self.client = client
        self.config = config
        self.engines: List[GridBot] = []
        self._shared_risk_state = SharedRiskState()
        self._risk_task: Optional[asyncio.Task[Any]] = None

    def _product_ids(self) -> List[str]:
        raw = self.config.product_ids or self.config.product_id
        ids = [item.strip().upper() for item in raw.replace(";", ",").split(",") if item.strip()]
        return ids or [self.config.product_id]

    def _orders_path_for(self, product_id: str) -> Path:
        base = _orders_path()
        if len(self._product_ids()) <= 1:
            return base
        return base.with_name(f"{base.stem}_{product_id.lower()}{base.suffix}")

    def _db_path_for(self, product_id: str) -> str:
        base = Path(self.config.state_db_path)
        if len(self._product_ids()) <= 1:
            return str(base)
        return str(base.with_name(f"{base.stem}_{product_id.lower()}{base.suffix}"))

    def _shared_paper_portfolio(self) -> Optional[SharedPaperPortfolio]:
        if not self.config.paper_trading_mode or not self.config.shared_usd_reserve_enabled:
            return None
        portfolio = SharedPaperPortfolio(self.config.paper_start_usd)
        for product_id in self._product_ids():
            base = product_id.split("-")[0]
            portfolio.apply_delta(base, self.config.paper_start_base)
        return portfolio

    def _fetch_closes_for_product(self, product_id: str) -> List[Decimal]:
        params = urllib.parse.urlencode(
            {
                "granularity": self.config.trend_candle_granularity,
                "limit": str(max(2, self.config.cross_asset_candle_lookback)),
            }
        )
        url = f"https://api.coinbase.com/api/v3/brokerage/products/{product_id}/candles?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logging.warning("Cross-asset candle fetch failed for %s: %s", product_id, exc)
            return []

        rows: List[Tuple[int, Decimal]] = []
        for candle in payload.get("candles", []):
            start = candle.get("start")
            close = candle.get("close")
            if start is None or close is None:
                continue
            rows.append((int(start), Decimal(str(close))))
        rows.sort(key=lambda item: item[0])
        return [close for _ts, close in rows]

    def _refresh_cross_asset_risk(self) -> None:
        if not self.config.cross_asset_correlation_enabled or len(self.engines) < 2:
            for engine in self.engines:
                self._shared_risk_state.set_inventory_cap(engine.config.product_id, engine.config.max_btc_inventory_pct)
            self._shared_risk_state.set_portfolio_beta(Decimal("0"))
            return

        product_ids = [engine.config.product_id for engine in self.engines]
        closes_by_product = {pid: self._fetch_closes_for_product(pid) for pid in product_ids}
        returns_by_product = {pid: _returns_from_closes(closes) for pid, closes in closes_by_product.items()}

        for engine in self.engines:
            self._shared_risk_state.set_inventory_cap(engine.config.product_id, engine.config.max_btc_inventory_pct)

        leader = None
        leader_usage = Decimal("0")
        for engine in self.engines:
            price = engine.last_price
            usage = engine._btc_inventory_ratio(price if price > 0 else Decimal("0"))
            cap = max(Decimal("0.0001"), engine.config.max_btc_inventory_pct)
            normalized = usage / cap
            if normalized > leader_usage:
                leader_usage = normalized
                leader = engine

        if leader is not None and leader_usage >= self.config.cross_asset_leader_inventory_trigger_pct:
            leader_id = leader.config.product_id
            leader_returns = returns_by_product.get(leader_id, [])
            for engine in self.engines:
                target_id = engine.config.product_id
                if target_id == leader_id:
                    continue
                target_returns = returns_by_product.get(target_id, [])
                corr = _pearson_corr(leader_returns, target_returns)
                self._shared_risk_state.set_correlation(leader_id, target_id, corr)
                if corr >= self.config.cross_asset_correlation_threshold:
                    tightened = engine.config.max_btc_inventory_pct * self.config.cross_asset_inventory_tightening_factor
                    tightened = max(self.config.cross_asset_inventory_min_pct, tightened)
                    self._shared_risk_state.set_inventory_cap(target_id, tightened)
                    logging.info(
                        "Cross-asset cap tightened for %s: corr(%s,%s)=%.3f leader_usage=%.2f%% cap=%.2f%%",
                        target_id,
                        leader_id,
                        target_id,
                        float(corr),
                        float(leader_usage * 100),
                        float(tightened * 100),
                    )

        benchmark_returns = returns_by_product.get("BTC-USD")
        if not benchmark_returns:
            first = product_ids[0] if product_ids else ""
            benchmark_returns = returns_by_product.get(first, [])

        portfolio_value = Decimal("0")
        weighted_beta = Decimal("0")
        for engine in self.engines:
            price = engine.last_price
            usd_bal = engine._get_available_balance("USD")
            base_bal = engine._get_available_balance(engine.base_currency)
            value = usd_bal + (base_bal * price if price > 0 else Decimal("0"))
            portfolio_value += value
            asset_beta = _beta(returns_by_product.get(engine.config.product_id, []), benchmark_returns)
            weighted_beta += value * asset_beta

        if portfolio_value > 0:
            self._shared_risk_state.set_portfolio_beta(weighted_beta / portfolio_value)
        else:
            self._shared_risk_state.set_portfolio_beta(Decimal("0"))

    async def _cross_asset_risk_loop(self) -> None:
        while any(engine._running for engine in self.engines):
            await asyncio.to_thread(self._refresh_cross_asset_risk)
            await asyncio.sleep(max(5, self.config.cross_asset_refresh_seconds))

    async def run(self) -> None:
        shared_portfolio = self._shared_paper_portfolio()
        for product_id in self._product_ids():
            engine_cfg = replace(self.config, product_id=product_id, state_db_path=self._db_path_for(product_id))
            if shared_portfolio is not None:
                engine_cfg = replace(engine_cfg, paper_start_usd=Decimal("0"), paper_start_base=Decimal("0"), paper_start_btc=Decimal("0"))
            self.engines.append(
                GridBot(
                    client=self.client,
                    config=engine_cfg,
                    orders_path=self._orders_path_for(product_id),
                    shared_paper_portfolio=shared_portfolio,
                    shared_risk_state=self._shared_risk_state,
                )
            )

        if len(self.engines) == 1:
            await self.engines[0].run()
            return

        names = ", ".join(e.config.product_id for e in self.engines)
        logging.info("GridManager starting %s engines: %s", len(self.engines), names)
        self._risk_task = asyncio.create_task(self._cross_asset_risk_loop(), name="cross-asset-risk-loop")
        try:
            await asyncio.gather(*(engine.run() for engine in self.engines))
        finally:
            if self._risk_task is not None:
                self._risk_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._risk_task

    def stop(self) -> None:
        for engine in self.engines:
            engine.stop()


def build_client() -> Any:
    api_key = _read_secret("COINBASE_API_KEY", "COINBASE_API_KEY_FILE")
    api_secret = _read_secret("COINBASE_API_SECRET", "COINBASE_API_SECRET_FILE")
    if not api_key or not api_secret:
        raise EnvironmentError(
            "Set COINBASE_API_KEY/COINBASE_API_SECRET (or *_FILE variants with chmod 600)."
        )
    from coinbase.rest import RESTClient

    return RESTClient(api_key=api_key, api_secret=api_secret)


def main() -> None:
    parser = argparse.ArgumentParser(description="Thumber Trader grid bot")
    parser.add_argument("--export-tax-report", dest="export_tax_report", type=Path, help="write fills tax CSV to this path")
    parser.add_argument("--tax-year", dest="tax_year", type=int, help="optional tax year filter for CSV export")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = BotConfig()
    _validate_config(config)

    if args.export_tax_report:
        count = export_tax_report_csv(config.state_db_path, args.export_tax_report, year=args.tax_year)
        logging.info("Exported %s fills to %s", count, args.export_tax_report)
        return

    client = build_client()
    manager = GridManager(client, config)

    def _handle_signal(signum: int, _frame: Any) -> None:
        logging.info("Received signal %s, shutting down cleanly.", signum)
        manager.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    asyncio.run(manager.run())


if __name__ == "__main__":
    main()
