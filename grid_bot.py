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
import collections
import json
import importlib
import logging
import math
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

    # multi-venue market data
    consensus_pricing_enabled: bool = os.getenv("CONSENSUS_PRICING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    consensus_exchanges: str = os.getenv("CONSENSUS_EXCHANGES", "coinbase,binance,kraken,bybit")
    consensus_max_deviation_pct: Decimal = Decimal(os.getenv("CONSENSUS_MAX_DEVIATION_PCT", "0.02"))

    # alpha fusion
    alpha_fusion_enabled: bool = os.getenv("ALPHA_FUSION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    alpha_rsi_period: int = int(os.getenv("ALPHA_RSI_PERIOD", "14"))
    alpha_macd_fast: int = int(os.getenv("ALPHA_MACD_FAST", "12"))
    alpha_macd_slow: int = int(os.getenv("ALPHA_MACD_SLOW", "26"))
    alpha_macd_signal: int = int(os.getenv("ALPHA_MACD_SIGNAL", "9"))
    alpha_weight_rsi: Decimal = Decimal(os.getenv("ALPHA_WEIGHT_RSI", "0.3"))
    alpha_weight_macd: Decimal = Decimal(os.getenv("ALPHA_WEIGHT_MACD", "0.3"))
    alpha_weight_imbalance: Decimal = Decimal(os.getenv("ALPHA_WEIGHT_IMBALANCE", "0.4"))

    # order-flow toxicity (VPIN)
    vpin_enabled: bool = os.getenv("VPIN_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    vpin_bucket_volume_base: Decimal = Decimal(os.getenv("VPIN_BUCKET_VOLUME_BASE", "0.25"))
    vpin_rolling_buckets: int = int(os.getenv("VPIN_ROLLING_BUCKETS", "50"))
    vpin_history_size: int = int(os.getenv("VPIN_HISTORY_SIZE", "300"))
    vpin_threshold_percentile: Decimal = Decimal(os.getenv("VPIN_THRESHOLD_PERCENTILE", "0.95"))
    vpin_response_mode: str = os.getenv("VPIN_RESPONSE_MODE", "widen").strip().lower()
    vpin_widen_band_multiplier: Decimal = Decimal(os.getenv("VPIN_WIDEN_BAND_MULTIPLIER", "1.5"))

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
    kelly_allocation_enabled: bool = os.getenv("KELLY_ALLOCATION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    kelly_refresh_seconds: int = int(os.getenv("KELLY_REFRESH_SECONDS", "900"))
    kelly_lookback_fills: int = int(os.getenv("KELLY_LOOKBACK_FILLS", "300"))
    kelly_min_closed_trades: int = int(os.getenv("KELLY_MIN_CLOSED_TRADES", "20"))
    kelly_min_allocation_frac: Decimal = Decimal(os.getenv("KELLY_MIN_ALLOCATION_FRAC", "0.25"))
    kelly_max_allocation_frac: Decimal = Decimal(os.getenv("KELLY_MAX_ALLOCATION_FRAC", "2.50"))
    quote_reserve_pct: Decimal = Decimal(os.getenv("QUOTE_RESERVE_PCT", "0.25"))
    max_btc_inventory_pct: Decimal = Decimal(os.getenv("MAX_BTC_INVENTORY_PCT", "0.65"))
    hard_stop_loss_pct: Decimal = Decimal(os.getenv("HARD_STOP_LOSS_PCT", "0.08"))
    liquidity_depth_check_enabled: bool = os.getenv("LIQUIDITY_DEPTH_CHECK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    liquidity_depth_levels: int = int(os.getenv("LIQUIDITY_DEPTH_LEVELS", "50"))
    liquidity_max_book_share_pct: Decimal = Decimal(os.getenv("LIQUIDITY_MAX_BOOK_SHARE_PCT", "0.20"))

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
    hmm_regime_detection_enabled: bool = os.getenv("HMM_REGIME_DETECTION_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    hmm_states: int = int(os.getenv("HMM_STATES", "3"))
    hmm_lookback: int = int(os.getenv("HMM_LOOKBACK", "120"))
    hmm_iterations: int = int(os.getenv("HMM_ITERATIONS", "12"))
    hmm_min_variance: Decimal = Decimal(os.getenv("HMM_MIN_VARIANCE", "0.00000001"))
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

    # operational circuit breaker
    api_circuit_breaker_enabled: bool = os.getenv("API_CIRCUIT_BREAKER_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    api_latency_p95_threshold_ms: float = float(os.getenv("API_LATENCY_P95_THRESHOLD_MS", "2000"))
    api_failure_rate_threshold_pct: Decimal = Decimal(os.getenv("API_FAILURE_RATE_THRESHOLD_PCT", "0.05"))
    api_health_window_seconds: int = int(os.getenv("API_HEALTH_WINDOW_SECONDS", "300"))
    api_recovery_consecutive_minutes: int = int(os.getenv("API_RECOVERY_CONSECUTIVE_MINUTES", "5"))

    # optional sentiment override module
    sentiment_override_enabled: bool = os.getenv("SENTIMENT_OVERRIDE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    sentiment_source_url: str = os.getenv("SENTIMENT_SOURCE_URL", "")
    sentiment_api_bearer_token: str = os.getenv("SENTIMENT_API_BEARER_TOKEN", "")
    sentiment_json_path: str = os.getenv("SENTIMENT_JSON_PATH", "score")
    sentiment_asset_query_param: str = os.getenv("SENTIMENT_ASSET_QUERY_PARAM", "symbol")
    sentiment_refresh_seconds: int = int(os.getenv("SENTIMENT_REFRESH_SECONDS", "300"))
    sentiment_lookback_seconds: int = int(os.getenv("SENTIMENT_LOOKBACK_SECONDS", "3600"))
    sentiment_negative_threshold: Decimal = Decimal(os.getenv("SENTIMENT_NEGATIVE_THRESHOLD", "-0.6"))
    sentiment_safe_inventory_cap_pct: Decimal = Decimal(os.getenv("SENTIMENT_SAFE_INVENTORY_CAP_PCT", "0.20"))

    # optional Telegram notifications
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    telegram_whitelist_chat_id: str = os.getenv("TELEGRAM_WHITELIST_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID", ""))


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


class StrategyEngine:
    """Base interface for pluggable trading strategy engines."""

    strategy_name = "base"

    async def run(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class GridStrategy(StrategyEngine):
    """Concrete grid strategy implementation for a single product."""

    strategy_name = "grid"

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
        self._last_portfolio_value_usd = Decimal("0")
        self.recent_events: List[str] = []
        self._dashboard_server: Optional[ThreadingHTTPServer] = None
        self._state_lock = threading.RLock()
        self._dashboard_update_cond = threading.Condition(self._state_lock)
        self._dashboard_update_seq = 0
        self._ws_queue: asyncio.Queue[Dict[str, str]] = asyncio.Queue()
        self._ws_client: Optional[Any] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_ws_sequence: Optional[int] = self._load_last_ws_sequence()
        self._cached_trend_strength = Decimal("0")
        self._cached_atr_pct = Decimal("0")
        self._cached_adx = Decimal("0")
        self._market_regime = "UNKNOWN"
        self._market_regime_source = "ADX"
        self._market_regime_confidence = Decimal("0")
        self._active_inventory_cap_pct = self.config.max_btc_inventory_pct
        self._capital_allocation_multiplier = Decimal("1")
        self._capital_allocation_kelly_fraction = Decimal("0")
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
        self._api_health_samples: collections.deque[Tuple[float, float, bool]] = collections.deque()
        self._api_safe_mode = False
        self._sentiment_samples: collections.deque[Tuple[float, Decimal]] = collections.deque()
        self._sentiment_safe_mode = False
        self._sentiment_inventory_cap_pct_override: Optional[Decimal] = None
        self._last_sentiment_report = {
            "sample_count": 0,
            "latest_score": "0",
            "score_1h": "0",
            "healthy": True,
        }
        self._paused = False
        self._api_recovery_healthy_minutes = 0
        self._last_api_health_report = {
            "sample_count": 0,
            "failure_rate": 0.0,
            "p95_latency_ms": 0.0,
            "healthy": True,
        }
        self._emergency_stop_triggered = False
        self._last_consensus_components: Dict[str, str] = {}
        self._alpha_confidence_scores: Dict[int, Decimal] = {}
        self._alpha_signal_snapshot: Dict[str, str] = {"rsi": "0", "macd_hist": "0", "book_imbalance": "0", "vpin": "0"}
        self._vpin_bucket_fill = Decimal("0")
        self._vpin_bucket_buy = Decimal("0")
        self._vpin_bucket_sell = Decimal("0")
        self._vpin_bucket_imbalances: collections.deque[Decimal] = collections.deque(maxlen=max(5, self.config.vpin_history_size))
        self._vpin_history: collections.deque[Decimal] = collections.deque(maxlen=max(20, self.config.vpin_history_size))
        self._vpin_last_trade_id: Optional[str] = None
        self._vpin_value = Decimal("0")
        self._vpin_threshold = Decimal("0")
        self._vpin_toxic_flow = False
        self._vpin_pause_entries = False
        self._vpin_band_multiplier = Decimal("1")
        self._dashboard_candles: List[Dict[str, float]] = []
        self._dashboard_candles_refresh_ts = 0.0
        self._dashboard_recent_fills: List[Dict[str, str]] = self._load_recent_fills_for_dashboard(limit=40)
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
            asyncio.create_task(self._api_health_monitor_loop(), name="api-health-monitor"),
        ]
        if self.config.sentiment_override_enabled:
            tasks.append(asyncio.create_task(self._sentiment_monitor_loop(), name="sentiment-monitor"))
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
                await asyncio.to_thread(self._refresh_vpin_signal)
                await asyncio.to_thread(self._refresh_alpha_confidence, current_price)
                with self._state_lock:
                    self.loop_count += 1
                    self.last_price = current_price
                    self.last_trend_bias = trend_bias
                    self._notify_dashboard_update_locked()
                if not self._paused:
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

    async def _api_health_monitor_loop(self) -> None:
        if not self.config.api_circuit_breaker_enabled:
            return
        while self._running:
            try:
                await asyncio.to_thread(self._evaluate_api_health)
            except Exception as exc:
                logging.exception("API health monitor loop error: %s", exc)
            await asyncio.sleep(60)

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

    async def _sentiment_monitor_loop(self) -> None:
        while self._running:
            try:
                await asyncio.to_thread(self._evaluate_sentiment_override)
            except Exception as exc:
                logging.exception("Sentiment monitor loop error: %s", exc)
            await asyncio.sleep(max(30, self.config.sentiment_refresh_seconds))

    def _load_product_metadata(self) -> None:
        product = _as_dict(self._coinbase_api_call("get_product", self.client.get_product, product_id=self.config.product_id))
        if product.get("quote_increment"):
            self.price_increment = Decimal(str(product["quote_increment"]))
        if product.get("base_increment"):
            self.base_increment = Decimal(str(product["base_increment"]))
        logging.info("Metadata: quote_increment=%s base_increment=%s", self.price_increment, self.base_increment)

    def _get_current_price(self) -> Decimal:
        if self.config.consensus_pricing_enabled:
            consensus = self._get_consensus_price()
            if consensus is not None:
                return consensus
        product = _as_dict(self._coinbase_api_call("get_product", self.client.get_product, product_id=self.config.product_id))
        raw = product.get("price")
        if raw is None:
            raise RuntimeError(f"Unable to read product price: {product}")
        return Decimal(str(raw))

    def _get_consensus_price(self) -> Optional[Decimal]:
        mids = self._fetch_exchange_mid_prices()
        if not mids:
            return None

        values = list(mids.values())
        pivot = sorted(values)[len(values) // 2]
        allowed = max(Decimal("0"), self.config.consensus_max_deviation_pct)
        filtered = [v for v in values if pivot > 0 and abs(v - pivot) / pivot <= allowed]
        if not filtered:
            filtered = values
        self._last_consensus_components = {name: str(price) for name, price in mids.items()}
        return sum(filtered, Decimal("0")) / Decimal(len(filtered))

    def _fetch_exchange_mid_prices(self) -> Dict[str, Decimal]:
        exchange_names = [part.strip().lower() for part in self.config.consensus_exchanges.split(",") if part.strip()]
        mids: Dict[str, Decimal] = {}
        for venue in exchange_names:
            try:
                mid = self._fetch_venue_mid_price(venue)
                if mid is not None and mid > 0:
                    mids[venue] = mid
            except Exception as exc:
                logging.debug("Consensus pricing venue %s unavailable: %s", venue, exc)
        return mids

    def _fetch_venue_mid_price(self, venue: str) -> Optional[Decimal]:
        base, quote = self.config.product_id.split("-")
        if venue == "coinbase":
            url = f"https://api.exchange.coinbase.com/products/{base}-{quote}/ticker"
            with self._coinbase_api_call("public_ticker_coinbase", urllib.request.urlopen, url, timeout=8) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            bid = Decimal(str(payload.get("bid") or "0"))
            ask = Decimal(str(payload.get("ask") or "0"))
            return (bid + ask) / Decimal("2") if bid > 0 and ask > 0 else None
        if venue == "binance":
            symbol = f"{base}USDT" if quote == "USD" else f"{base}{quote}"
            url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={symbol}"
            with urllib.request.urlopen(url, timeout=8) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            bid = Decimal(str(payload.get("bidPrice") or "0"))
            ask = Decimal(str(payload.get("askPrice") or "0"))
            return (bid + ask) / Decimal("2") if bid > 0 and ask > 0 else None
        if venue == "kraken":
            kraken_base = "XBT" if base == "BTC" else base
            pair = f"{kraken_base}{quote}"
            url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
            with urllib.request.urlopen(url, timeout=8) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            result = payload.get("result", {})
            if not result:
                return None
            ticker = next(iter(result.values()))
            bid = Decimal(str((ticker.get("b") or ["0"])[0]))
            ask = Decimal(str((ticker.get("a") or ["0"])[0]))
            return (bid + ask) / Decimal("2") if bid > 0 and ask > 0 else None
        if venue == "bybit":
            symbol = f"{base}USDT" if quote == "USD" else f"{base}{quote}"
            url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
            with urllib.request.urlopen(url, timeout=8) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            rows = payload.get("result", {}).get("list", [])
            if not rows:
                return None
            row = rows[0]
            bid = Decimal(str(row.get("bid1Price") or "0"))
            ask = Decimal(str(row.get("ask1Price") or "0"))
            return (bid + ask) / Decimal("2") if bid > 0 and ask > 0 else None
        return None

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
            return self.config.grid_band_pct * regime_band_multiplier * self._vpin_band_multiplier

        candles = self._fetch_public_candles()
        atr = _atr(candles, self.config.atr_period)
        if atr <= 0:
            return self.config.grid_band_pct * regime_band_multiplier * self._vpin_band_multiplier

        dynamic_pct = (atr * self.config.atr_band_multiplier) / current_price
        atr_band = max(self.config.atr_min_band_pct, min(dynamic_pct, self.config.atr_max_band_pct))
        return atr_band * regime_band_multiplier * self._vpin_band_multiplier

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

        buy_budget_per_order = self._regime_adjusted_buy_notional(self._allocated_base_order_notional())
        max_buys = int(deployable_usd // buy_budget_per_order)
        buy_levels = buy_levels[:max_buys] if max_buys >= 0 else []

        for level in buy_levels:
            confidence = self._confidence_for_price(level)
            self._place_grid_order(
                side="BUY",
                price=level,
                usd_notional=buy_budget_per_order * self._confidence_order_multiplier(confidence),
            )

        if sell_levels and base_available > Decimal("0"):
            base_per_sell = self._q_base(base_available / Decimal(len(sell_levels)))
            for level in sell_levels:
                scaled = self._q_base(base_per_sell * self._confidence_order_multiplier(self._confidence_for_price(level)))
                if scaled * level < self.config.min_notional_usd:
                    continue
                self._place_grid_order(side="SELL", price=level, base_size=scaled)

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
                scaled_base = self._q_base(base_size * self._confidence_order_multiplier(self._confidence_for_price(new_price)))
                new_id = self._place_grid_order(side="SELL", price=new_price, base_size=scaled_base, grid_index=grid_index + 1)
                if new_id:
                    logging.info("Buy Filled at $%s! Placed Sell at $%s.", fill_price, new_price)
                    self._add_event(f"Replacement SELL placed @ {new_price}")

            elif side == "SELL" and grid_index - 1 >= 0:
                if trend_bias == "DOWN":
                    logging.info("Sell filled at %s in downtrend; preserving capital (skip replacement buy).", fill_price)
                    continue
                new_price = self.grid_levels[grid_index - 1]
                usd_notional = self._regime_adjusted_buy_notional(Decimal(str(record["base_size"])) * fill_price)
                confidence = self._confidence_for_price(new_price)
                new_id = self._place_grid_order(
                    side="BUY",
                    price=new_price,
                    usd_notional=usd_notional * self._confidence_order_multiplier(confidence),
                    grid_index=grid_index - 1,
                )
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
            if self._vpin_pause_entries:
                logging.warning("Skipped BUY at %s because VPIN toxicity gate is active.", price)
                return None
            if self._buy_safe_mode_active():
                logging.warning("Skipped BUY at %s because safe mode is active.", price)
                return None
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
            if self._vpin_pause_entries:
                logging.warning("Skipped SELL at %s because VPIN toxicity gate is active.", price)
                return None
            if base_size is None:
                raise ValueError("base_size is required for SELL")
            base_size = self._q_base(base_size)
            base_size = self._liquidity_adjusted_sell_size(price, base_size)
            if base_size is None:
                return None
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
            response = _as_dict(self._coinbase_api_call("create_order", self.client.create_order, **payload))
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
        if self._sentiment_inventory_cap_pct_override is not None:
            cap = min(cap, self._sentiment_inventory_cap_pct_override)
        if self.shared_risk_state is not None:
            shared_cap = self.shared_risk_state.get_inventory_cap(self.config.product_id)
            if shared_cap is not None:
                cap = min(cap, shared_cap)
        return cap

    def _buy_safe_mode_active(self) -> bool:
        return self._api_safe_mode or self._sentiment_safe_mode

    def _get_trend_bias(self) -> str:
        candles = self._fetch_public_candles()
        closes = [c[3] for c in candles]
        self._cached_atr_pct = self._estimate_atr_pct(candles)
        self._cached_adx = _adx(candles, self.config.adx_period)
        hmm_regime = self._classify_market_regime_from_hmm(closes)
        if hmm_regime is not None:
            self._market_regime, self._market_regime_confidence = hmm_regime
            self._market_regime_source = "HMM"
        else:
            self._market_regime = self._classify_market_regime(self._cached_adx)
            self._market_regime_source = "ADX"
            self._market_regime_confidence = Decimal("0")
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

    def _classify_market_regime_from_hmm(self, closes: List[Decimal]) -> Optional[Tuple[str, Decimal]]:
        if not self.config.hmm_regime_detection_enabled:
            return None

        returns = _returns(closes[-(self.config.hmm_lookback + 1) :])
        min_obs = max(20, self.config.hmm_states * 4)
        if len(returns) < min_obs:
            logging.debug("HMM regime detection skipped; need at least %s returns (have %s).", min_obs, len(returns))
            return None

        model = _fit_gaussian_hmm(
            returns,
            n_states=self.config.hmm_states,
            iterations=self.config.hmm_iterations,
            min_variance=self.config.hmm_min_variance,
        )
        if model is None:
            return None

        probs, means, variances = model
        winner = max(range(len(probs)), key=lambda idx: probs[idx])
        confidence = probs[winner]

        vol_rank = sorted(range(len(variances)), key=lambda idx: variances[idx])
        low_vol_state = vol_rank[0]
        high_vol_state = vol_rank[-1]
        if winner == low_vol_state:
            regime = "RANGING"
        elif winner == high_vol_state:
            regime = "TRENDING"
        else:
            regime = "TRANSITION"

        logging.info(
            "HMM regime=%s confidence=%.3f state=%s mean=%.6f sigma=%.6f",
            regime,
            confidence,
            winner,
            means[winner],
            math.sqrt(max(variances[winner], 0.0)),
        )
        return regime, Decimal(str(confidence))

    def _regime_band_multiplier(self) -> Decimal:
        if self._market_regime == "TRENDING":
            return self.config.adx_trend_band_multiplier
        if self._market_regime == "RANGING":
            return self.config.adx_range_band_multiplier
        return Decimal("1")

    def set_capital_allocation(self, multiplier: Decimal, kelly_fraction: Decimal) -> None:
        with self._state_lock:
            self._capital_allocation_multiplier = max(Decimal("0"), multiplier)
            self._capital_allocation_kelly_fraction = max(Decimal("0"), kelly_fraction)

    def _allocated_base_order_notional(self) -> Decimal:
        sized = self.config.base_order_notional_usd * self._capital_allocation_multiplier
        return max(self.config.min_notional_usd, sized)

    def _regime_adjusted_buy_notional(self, usd_notional: Decimal) -> Decimal:
        adjusted = usd_notional
        if self._market_regime == "TRENDING":
            adjusted = usd_notional * self.config.adx_trend_order_size_multiplier
        return max(self.config.min_notional_usd, adjusted)

    def _liquidity_adjusted_sell_size(self, limit_price: Decimal, base_size: Decimal) -> Optional[Decimal]:
        if not self.config.liquidity_depth_check_enabled or base_size <= 0:
            return base_size

        ref_price = self.last_price if self.last_price > 0 else self.grid_anchor_price
        if ref_price <= 0:
            ref_price = self._get_current_price()

        ask_depth = self._fetch_cumulative_ask_depth(min(ref_price, limit_price), limit_price)
        if ask_depth <= 0:
            logging.warning("Skipped SELL at %s; no visible ask depth in level2 snapshot.", limit_price)
            return None

        capped = self._q_base(ask_depth * self.config.liquidity_max_book_share_pct)
        if capped <= 0:
            logging.warning("Skipped SELL at %s; liquidity cap collapsed below base increment.", limit_price)
            return None

        if base_size <= capped:
            return base_size

        if capped * limit_price < self.config.min_notional_usd:
            logging.warning(
                "Skipped SELL at %s; order-book cap %s falls below minimum notional %s.",
                limit_price,
                capped,
                self.config.min_notional_usd,
            )
            return None

        logging.info(
            "Liquidity-aware sizing reduced SELL @ %s from %s to %s base units (visible asks=%s).",
            limit_price,
            base_size,
            capped,
            ask_depth,
        )
        return capped

    def _fetch_cumulative_ask_depth(self, lower_price: Decimal, upper_price: Decimal) -> Decimal:
        if upper_price <= 0 or lower_price >= upper_price:
            return Decimal("0")

        asks = self._fetch_level2_asks()
        if not asks:
            return Decimal("0")

        total = Decimal("0")
        for price, size in asks:
            if price < lower_price:
                continue
            if price > upper_price:
                break
            total += size
        return total

    def _fetch_level2_asks(self) -> List[Tuple[Decimal, Decimal]]:
        limit = max(1, self.config.liquidity_depth_levels)
        endpoints = [
            f"https://api.coinbase.com/api/v3/brokerage/product_book?product_id={self.config.product_id}&limit={limit}",
            f"https://api.exchange.coinbase.com/products/{self.config.product_id}/book?level=2",
        ]
        for url in endpoints:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "thumber-trader/1.0"})
                with self._coinbase_api_call("public_level2", urllib.request.urlopen, req, timeout=10) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except Exception:
                continue

            asks = self._parse_level2_asks(payload)
            if asks:
                return asks
        logging.warning("Level2 depth fetch failed for %s.", self.config.product_id)
        return []

    def _parse_level2_asks(self, payload: Dict[str, Any]) -> List[Tuple[Decimal, Decimal]]:
        raw_asks = payload.get("pricebook", {}).get("asks")
        if raw_asks is None:
            raw_asks = payload.get("asks", [])

        parsed: List[Tuple[Decimal, Decimal]] = []
        for level in raw_asks or []:
            if isinstance(level, dict):
                price_raw = level.get("price")
                size_raw = level.get("size") or level.get("quantity")
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                price_raw = level[0]
                size_raw = level[1]
            else:
                continue

            if price_raw is None or size_raw is None:
                continue
            try:
                price = Decimal(str(price_raw))
                size = Decimal(str(size_raw))
            except Exception:
                continue
            if price <= 0 or size <= 0:
                continue
            parsed.append((price, size))

        parsed.sort(key=lambda x: x[0])
        return parsed

    def _fetch_level2_bids(self) -> List[Tuple[Decimal, Decimal]]:
        limit = max(1, self.config.liquidity_depth_levels)
        endpoints = [
            f"https://api.coinbase.com/api/v3/brokerage/product_book?product_id={self.config.product_id}&limit={limit}",
            f"https://api.exchange.coinbase.com/products/{self.config.product_id}/book?level=2",
        ]
        for url in endpoints:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "thumber-trader/1.0"})
                with self._coinbase_api_call("public_level2", urllib.request.urlopen, req, timeout=10) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except Exception:
                continue

            bids = self._parse_level2_bids(payload)
            if bids:
                return bids
        return []

    def _parse_level2_bids(self, payload: Dict[str, Any]) -> List[Tuple[Decimal, Decimal]]:
        raw_bids = payload.get("pricebook", {}).get("bids")
        if raw_bids is None:
            raw_bids = payload.get("bids", [])

        parsed: List[Tuple[Decimal, Decimal]] = []
        for level in raw_bids or []:
            if isinstance(level, dict):
                price_raw = level.get("price")
                size_raw = level.get("size") or level.get("quantity")
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                price_raw = level[0]
                size_raw = level[1]
            else:
                continue

            if price_raw is None or size_raw is None:
                continue
            try:
                price = Decimal(str(price_raw))
                size = Decimal(str(size_raw))
            except Exception:
                continue
            if price <= 0 or size <= 0:
                continue
            parsed.append((price, size))

        parsed.sort(key=lambda x: x[0], reverse=True)
        return parsed

    def _book_imbalance(self) -> Decimal:
        asks = self._fetch_level2_asks()
        bids = self._fetch_level2_bids()
        top = max(1, min(10, self.config.liquidity_depth_levels))
        ask_size = sum((size for _price, size in asks[:top]), Decimal("0"))
        bid_size = sum((size for _price, size in bids[:top]), Decimal("0"))
        denom = ask_size + bid_size
        if denom <= 0:
            return Decimal("0")
        return (bid_size - ask_size) / denom

    def _refresh_alpha_confidence(self, current_price: Decimal) -> None:
        if not self.config.alpha_fusion_enabled or not self.grid_levels:
            self._alpha_confidence_scores = {}
            return

        candles = self._fetch_public_candles()
        closes = [c[3] for c in candles]
        rsi = _rsi(closes, self.config.alpha_rsi_period)
        macd_hist = _macd_histogram(closes, self.config.alpha_macd_fast, self.config.alpha_macd_slow, self.config.alpha_macd_signal)
        imbalance = self._book_imbalance()
        self._alpha_signal_snapshot = {
            "rsi": str(rsi),
            "macd_hist": str(macd_hist),
            "book_imbalance": str(imbalance),
            "vpin": str(self._vpin_value),
        }

        w_rsi = self.config.alpha_weight_rsi
        w_macd = self.config.alpha_weight_macd
        w_imb = self.config.alpha_weight_imbalance
        denom = w_rsi + w_macd + w_imb
        if denom <= 0:
            denom = Decimal("1")

        rsi_component = (rsi - Decimal("50")) / Decimal("50")
        price_ref = current_price if current_price > 0 else (closes[-1] if closes else Decimal("0"))
        macd_component = Decimal("0") if price_ref <= 0 else max(Decimal("-1"), min(Decimal("1"), macd_hist / (price_ref * Decimal("0.01"))))

        scores: Dict[int, Decimal] = {}
        for idx, level in enumerate(self.grid_levels):
            side_sign = Decimal("1") if level >= current_price else Decimal("-1")
            fused = ((rsi_component * w_rsi) + (macd_component * w_macd) + (imbalance * side_sign * w_imb)) / denom
            confidence = max(Decimal("0"), min(Decimal("1"), (fused + Decimal("1")) / Decimal("2")))
            scores[idx] = confidence
        self._alpha_confidence_scores = scores

    def _confidence_for_price(self, price: Decimal) -> Decimal:
        if not self._alpha_confidence_scores or not self.grid_levels:
            return Decimal("0.5")
        try:
            idx = self.grid_levels.index(price)
        except ValueError:
            idx = min(range(len(self.grid_levels)), key=lambda i: abs(self.grid_levels[i] - price))
        return self._alpha_confidence_scores.get(idx, Decimal("0.5"))

    def _confidence_order_multiplier(self, confidence: Decimal) -> Decimal:
        return Decimal("0.5") + max(Decimal("0"), min(Decimal("1"), confidence))

    def _refresh_vpin_signal(self) -> None:
        if not self.config.vpin_enabled:
            self._vpin_toxic_flow = False
            self._vpin_pause_entries = False
            self._vpin_band_multiplier = Decimal("1")
            return

        trades = self._fetch_recent_trades()
        if not trades:
            return

        bucket_size = max(Decimal("0.00000001"), self.config.vpin_bucket_volume_base)
        new_trade_seen = False
        for trade_id, side, size in trades:
            if self._vpin_last_trade_id is not None and trade_id <= self._vpin_last_trade_id:
                continue
            new_trade_seen = True
            remaining = size
            while remaining > 0:
                room = bucket_size - self._vpin_bucket_fill
                chunk = remaining if remaining <= room else room
                if side == "buy":
                    self._vpin_bucket_buy += chunk
                else:
                    self._vpin_bucket_sell += chunk
                self._vpin_bucket_fill += chunk
                remaining -= chunk
                if self._vpin_bucket_fill >= bucket_size:
                    imbalance = abs(self._vpin_bucket_buy - self._vpin_bucket_sell) / bucket_size
                    self._vpin_bucket_imbalances.append(imbalance)
                    self._vpin_bucket_fill = Decimal("0")
                    self._vpin_bucket_buy = Decimal("0")
                    self._vpin_bucket_sell = Decimal("0")
            self._vpin_last_trade_id = trade_id

        if not new_trade_seen or len(self._vpin_bucket_imbalances) < max(5, self.config.vpin_rolling_buckets // 2):
            return

        window = list(self._vpin_bucket_imbalances)[-self.config.vpin_rolling_buckets :]
        self._vpin_value = sum(window, Decimal("0")) / Decimal(len(window))
        self._vpin_history.append(self._vpin_value)
        self._vpin_threshold = _decimal_percentile(list(self._vpin_history), self.config.vpin_threshold_percentile)
        toxic_flow = len(self._vpin_history) >= 20 and self._vpin_value >= self._vpin_threshold > 0
        self._set_vpin_controls(toxic_flow)

    def _set_vpin_controls(self, toxic_flow: bool) -> None:
        mode = self.config.vpin_response_mode
        pause_entries = toxic_flow and mode in {"pause", "pause_entries", "both"}
        band_multiplier = self.config.vpin_widen_band_multiplier if toxic_flow and mode in {"widen", "both"} else Decimal("1")
        band_multiplier = max(Decimal("1"), band_multiplier)

        state_changed = toxic_flow != self._vpin_toxic_flow or pause_entries != self._vpin_pause_entries or band_multiplier != self._vpin_band_multiplier
        self._vpin_toxic_flow = toxic_flow
        self._vpin_pause_entries = pause_entries
        self._vpin_band_multiplier = band_multiplier
        if state_changed:
            action = f"band x{self._vpin_band_multiplier}" if self._vpin_toxic_flow else "normal mode"
            if self._vpin_pause_entries:
                action += ", entry pause"
            self._add_event(f"VPIN state changed: toxic={self._vpin_toxic_flow}, {action}")

    def _fetch_recent_trades(self) -> List[Tuple[str, str, Decimal]]:
        url = f"https://api.exchange.coinbase.com/products/{self.config.product_id}/trades"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "thumber-trader/1.0"})
            with self._coinbase_api_call("public_trades", urllib.request.urlopen, req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []

        rows: List[Tuple[str, str, Decimal]] = []
        for trade in reversed(payload if isinstance(payload, list) else []):
            trade_id = str(trade.get("trade_id") or "")
            side = str(trade.get("side") or "").lower()
            size_raw = trade.get("size")
            if not trade_id or side not in {"buy", "sell"} or size_raw is None:
                continue
            try:
                size = Decimal(str(size_raw))
            except Exception:
                continue
            if size <= 0:
                continue
            rows.append((trade_id, side, size))
        return rows

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

    def _fetch_public_candles_ohlc(self, granularity: str = "FIVE_MINUTE", limit: int = 120) -> List[Dict[str, float]]:
        params = urllib.parse.urlencode({"granularity": granularity, "limit": str(limit)})
        url = f"https://api.coinbase.com/api/v3/brokerage/products/{self.config.product_id}/candles?{params}"
        try:
            with self._coinbase_api_call("dashboard_candles", urllib.request.urlopen, url, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logging.warning("Dashboard candle fetch failed: %s", exc)
            return []

        rows: List[Dict[str, float]] = []
        for candle in payload.get("candles", []):
            start = candle.get("start")
            open_price = candle.get("open")
            high = candle.get("high")
            low = candle.get("low")
            close = candle.get("close")
            if start is None or open_price is None or high is None or low is None or close is None:
                continue
            rows.append(
                {
                    "time": int(start),
                    "open": float(open_price),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                }
            )
        rows.sort(key=lambda row: row["time"])
        return rows

    def _dashboard_chart_snapshot(self) -> Dict[str, Any]:
        candles: List[Dict[str, float]]
        now = time.time()
        with self._state_lock:
            stale = (now - self._dashboard_candles_refresh_ts) > 15
            candles = list(self._dashboard_candles)
            recent_fills = list(self._dashboard_recent_fills)

        if stale or not candles:
            refreshed = self._fetch_public_candles_ohlc()
            if refreshed:
                candles = refreshed
                with self._state_lock:
                    self._dashboard_candles = list(refreshed)
                    self._dashboard_candles_refresh_ts = now

        return {
            "candles": candles,
            "recent_fills": recent_fills,
        }

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

    def pause_trading(self) -> None:
        self._paused = True
        self._add_event("Trading paused by Telegram command")

    def is_paused(self) -> bool:
        return self._paused

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
            self._notify_dashboard_update_locked()

    def _notify_dashboard_update_locked(self) -> None:
        self._dashboard_update_seq += 1
        self._dashboard_update_cond.notify_all()

    def _dashboard_stream_snapshot(self) -> Dict[str, Any]:
        chart = self._dashboard_chart_snapshot()
        with self._state_lock:
            return {
                "product_id": self.config.product_id,
                "last_price": str(self.last_price),
                "trend_bias": self.last_trend_bias,
                "portfolio_value_usd": str(self._last_portfolio_value_usd),
                "active_orders": len(self.orders),
                "fills": self.fill_count,
                "orders": self.orders,
                "recent_events": list(self.recent_events),
                "chart": chart,
            }

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

    def _cancel_pending_buy_orders(self) -> int:
        buy_order_ids = [oid for oid, rec in self.orders.items() if str(rec.get("side", "")).upper() == "BUY"]
        if not buy_order_ids:
            return 0
        if self.config.paper_trading_mode:
            for oid in buy_order_ids:
                self.orders.pop(oid, None)
            return len(buy_order_ids)
        try:
            self.client.cancel_orders(order_ids=buy_order_ids)
        except Exception:
            logging.warning("Batch BUY cancel unavailable, falling back to single-order cancellation.")
            for oid in buy_order_ids:
                try:
                    self.client.cancel_orders(order_ids=[oid])
                except Exception as exc:
                    logging.warning("Failed to cancel BUY order %s: %s", oid, exc)
        for oid in buy_order_ids:
            self.orders.pop(oid, None)
        return len(buy_order_ids)

    def _fetch_sentiment_score(self) -> Optional[Decimal]:
        url = self.config.sentiment_source_url.strip()
        if not url:
            return None

        query_key = self.config.sentiment_asset_query_param.strip()
        if query_key:
            split = urllib.parse.urlsplit(url)
            query = urllib.parse.parse_qs(split.query, keep_blank_values=True)
            query[query_key] = [self.base_currency]
            url = urllib.parse.urlunsplit(
                (
                    split.scheme,
                    split.netloc,
                    split.path,
                    urllib.parse.urlencode(query, doseq=True),
                    split.fragment,
                )
            )

        headers = {"Accept": "application/json"}
        token = self.config.sentiment_api_bearer_token.strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        path = [p for p in self.config.sentiment_json_path.split(".") if p]
        node: Any = payload
        for key in path:
            if isinstance(node, dict):
                node = node.get(key)
            else:
                node = None
            if node is None:
                break
        if node is None:
            return None
        return Decimal(str(node))

    def _trim_sentiment_samples_locked(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        window_start = now - self.config.sentiment_lookback_seconds
        while self._sentiment_samples and self._sentiment_samples[0][0] < window_start:
            self._sentiment_samples.popleft()

    def _evaluate_sentiment_override(self) -> None:
        score = self._fetch_sentiment_score()
        if score is None:
            return
        now = time.time()
        with self._state_lock:
            self._sentiment_samples.append((now, score))
            self._trim_sentiment_samples_locked(now)
            sample_count = len(self._sentiment_samples)
            score_1h = sum((s for _, s in self._sentiment_samples), Decimal("0")) / Decimal(sample_count)
            healthy = score_1h > self.config.sentiment_negative_threshold
            self._last_sentiment_report = {
                "sample_count": sample_count,
                "latest_score": str(score),
                "score_1h": str(score_1h),
                "healthy": healthy,
            }

            if not healthy:
                if not self._sentiment_safe_mode:
                    self._sentiment_safe_mode = True
                    self._sentiment_inventory_cap_pct_override = min(
                        max(self.config.sentiment_safe_inventory_cap_pct, Decimal("0")),
                        Decimal("1"),
                    )
                    canceled = self._cancel_pending_buy_orders()
                    self._add_event(
                        f"Sentiment panic detected (1h={score_1h}): safe mode ON, canceled {canceled} BUY orders"
                    )
                    logging.warning(
                        "Sentiment score dropped to %s (threshold=%s). Entering safe mode and biasing inventory toward USD.",
                        score_1h,
                        self.config.sentiment_negative_threshold,
                    )
                    self._send_telegram_alert("⚠️ Sentiment Panic Detected: Entering Safe Mode and canceling BUY orders")
            elif self._sentiment_safe_mode:
                self._sentiment_safe_mode = False
                self._sentiment_inventory_cap_pct_override = None
                self._add_event(f"Sentiment recovered (1h={score_1h}): safe mode OFF")
                logging.info("Sentiment recovered above threshold (%s > %s).", score_1h, self.config.sentiment_negative_threshold)

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
        trailing_notional = self._allocated_base_order_notional()
        self._place_grid_order(side="SELL", price=self.grid_levels[-1], base_size=self._q_base(trailing_notional / self.grid_levels[-1]), grid_index=len(self.grid_levels) - 1)
        self._add_event(f"Trailing roll up -> new top {self.grid_levels[-1]}")

    def _roll_grid_down(self, step: Decimal) -> None:
        removed_index = len(self.grid_levels) - 1
        removed_order = self._find_order_by_grid_index_and_side(removed_index, "SELL")
        if removed_order:
            self._cancel_single_order(removed_order)

        new_levels = [self._q_price(self.grid_levels[0] - step)] + self.grid_levels[:-1]
        self.grid_levels = new_levels
        self._reindex_orders_after_shift(direction="down")
        self._place_grid_order(side="BUY", price=self.grid_levels[0], usd_notional=self._allocated_base_order_notional(), grid_index=0)
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
        chart = self._dashboard_chart_snapshot()
        with self._state_lock:
            price = self.last_price
            usd_bal = self._get_available_balance("USD")
            base_bal = self._get_available_balance(self.base_currency)
            portfolio = usd_bal + (base_bal * price if price > 0 else Decimal("0"))
            self._last_portfolio_value_usd = portfolio
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
                "capital_allocation_multiplier": str(self._capital_allocation_multiplier),
                "kelly_fraction": str(self._capital_allocation_kelly_fraction),
                "trend_strength": str(self._cached_trend_strength),
                "adx": str(self._cached_adx),
                "market_regime": self._market_regime,
                "market_regime_source": self._market_regime_source,
                "market_regime_confidence": str(self._market_regime_confidence),
                "atr_pct": str(self._cached_atr_pct),
                "consensus_components": dict(self._last_consensus_components),
                "alpha_confidence_scores": {str(k): str(v) for k, v in self._alpha_confidence_scores.items()},
                "alpha_signals": dict(self._alpha_signal_snapshot),
                "vpin": {
                    "value": str(self._vpin_value),
                    "threshold": str(self._vpin_threshold),
                    "toxic_flow": self._vpin_toxic_flow,
                    "pause_entries": self._vpin_pause_entries,
                    "band_multiplier": str(self._vpin_band_multiplier),
                },
                "ranging_score": str(self.ranging_score()),
                "maker_fee_pct": str(self._effective_maker_fee_pct),
                "daily_realized_pnl_usd": str(self._daily_realized_pnl),
                "realized_pnl_total_usd": str(self._realized_pnl_total),
                "daily_pnl_per_1k": str(pnl_per_1k),
                "daily_turnover_ratio": str(turnover_ratio),
                "api_safe_mode": self._api_safe_mode,
                "sentiment_safe_mode": self._sentiment_safe_mode,
                "paused": self._paused,
                "api_health": dict(self._last_api_health_report),
                "sentiment_health": dict(self._last_sentiment_report),
                "risk_metrics": risk,
                "portfolio_beta": str(portfolio_beta),
                "recent_events": list(self.recent_events),
                "orders": self.orders,
                "chart": chart,
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

    def _load_recent_fills_for_dashboard(self, limit: int = 40) -> List[Dict[str, str]]:
        rows = self._fetch_fills()
        fills: List[Dict[str, str]] = []
        for row in rows[-limit:]:
            fills.append(
                {
                    "ts": str(row[0]),
                    "side": str(row[2]),
                    "price": str(row[3]),
                    "base_size": str(row[4]),
                    "order_id": str(row[6]),
                }
            )
        return fills

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
        if value_type == "float":
            return float(value)
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

                if parsed.path == "/api/stream":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()

                    seq = -1
                    while bot._running:
                        with bot._dashboard_update_cond:
                            if seq >= 0:
                                bot._dashboard_update_cond.wait_for(lambda: bot._dashboard_update_seq != seq, timeout=15)
                            seq = bot._dashboard_update_seq
                        payload = json.dumps(bot._dashboard_stream_snapshot(), separators=(",", ":"))
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
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
                with bot._dashboard_update_cond:
                    bot._notify_dashboard_update_locked()

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
        succeeded = False
        try:
            result = func(*args, **kwargs)
            succeeded = True
            return result
        finally:
            elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
            self._observe_api_latency(elapsed_ms)
            self._record_api_health_sample(elapsed_ms=elapsed_ms, succeeded=succeeded)

    def _observe_api_latency(self, elapsed_ms: float) -> None:
        with self._state_lock:
            self._api_latency_observation_count += 1
            self._api_latency_observation_sum_ms += elapsed_ms
            idx = bisect.bisect_left(self._api_latency_buckets_ms, elapsed_ms)
            if idx >= len(self._api_latency_bucket_counts):
                return
            self._api_latency_bucket_counts[idx] += 1

    def _record_api_health_sample(self, elapsed_ms: float, succeeded: bool) -> None:
        now = time.time()
        with self._state_lock:
            self._api_health_samples.append((now, elapsed_ms, succeeded))
            self._trim_api_health_samples_locked(now)

    def _trim_api_health_samples_locked(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        window_start = now - self.config.api_health_window_seconds
        while self._api_health_samples and self._api_health_samples[0][0] < window_start:
            self._api_health_samples.popleft()

    def _evaluate_api_health(self) -> None:
        with self._state_lock:
            self._trim_api_health_samples_locked()
            samples = list(self._api_health_samples)

        sample_count = len(samples)
        if sample_count == 0:
            return

        failures = sum(1 for _, _, ok in samples if not ok)
        failure_rate = failures / sample_count
        latencies = sorted(lat for _, lat, _ in samples)
        p95_index = min(len(latencies) - 1, max(0, int(len(latencies) * 0.95) - 1))
        p95_latency = latencies[p95_index]
        healthy = (
            p95_latency <= self.config.api_latency_p95_threshold_ms
            and failure_rate <= float(self.config.api_failure_rate_threshold_pct)
        )

        with self._state_lock:
            self._last_api_health_report = {
                "sample_count": sample_count,
                "failure_rate": failure_rate,
                "p95_latency_ms": p95_latency,
                "healthy": healthy,
            }

            if not healthy:
                self._api_recovery_healthy_minutes = 0
                if not self._api_safe_mode:
                    self._api_safe_mode = True
                    self._add_event("API instability detected: entering safe mode")
                    logging.warning(
                        "API instability detected (p95=%.2fms, failure_rate=%.2f%%): entering safe mode.",
                        p95_latency,
                        failure_rate * 100,
                    )
                    self._send_telegram_alert("🚨 API Instability Detected: Entering Safe Mode")
            elif self._api_safe_mode:
                self._api_recovery_healthy_minutes += 1
                if self._api_recovery_healthy_minutes >= self.config.api_recovery_consecutive_minutes:
                    self._api_safe_mode = False
                    self._api_recovery_healthy_minutes = 0
                    self._add_event("API health recovered: exiting safe mode")
                    logging.info(
                        "API health recovered for %s consecutive minutes. Exiting safe mode.",
                        self.config.api_recovery_consecutive_minutes,
                    )

    def _send_telegram_alert(self, text: str) -> None:
        token = self.config.telegram_bot_token.strip()
        chat_id = self.config.telegram_chat_id.strip()
        if not token or not chat_id:
            return
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "disable_notification": "false"}).encode("utf-8")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status >= 400:
                    logging.warning("Telegram alert failed with HTTP %s", resp.status)
        except Exception as exc:
            logging.warning("Telegram alert failed: %s", exc)

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
                "# HELP bot_api_safe_mode Whether API circuit breaker safe mode is active (1=active).",
                "# TYPE bot_api_safe_mode gauge",
                f"bot_api_safe_mode{{{labels}}} {1 if self._api_safe_mode else 0}",
                "# HELP bot_sentiment_safe_mode Whether sentiment safe mode is active (1=active).",
                "# TYPE bot_sentiment_safe_mode gauge",
                f"bot_sentiment_safe_mode{{{labels}}} {1 if self._sentiment_safe_mode else 0}",
                "# HELP bot_sentiment_score_1h One-hour rolling sentiment score.",
                "# TYPE bot_sentiment_score_1h gauge",
                f"bot_sentiment_score_1h{{{labels}}} {self._last_sentiment_report.get('score_1h', '0')}",
                "# HELP bot_vpin Volume-synchronized probability of informed trading.",
                "# TYPE bot_vpin gauge",
                f"bot_vpin{{{labels}}} {float(self._vpin_value)}",
                "# HELP bot_vpin_toxic_flow Whether VPIN toxicity regime is active (1=active).",
                "# TYPE bot_vpin_toxic_flow gauge",
                f"bot_vpin_toxic_flow{{{labels}}} {1 if self._vpin_toxic_flow else 0}",
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
        fill_ts = time.time()
        fee_paid = fill_price * Decimal(str(record.get("base_size", "0"))) * self._effective_maker_fee_pct
        with self._db_lock:
            self._db.execute(
                """
                INSERT INTO fills(ts, product_id, side, price, base_size, fee_paid, grid_index, order_id)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    fill_ts,
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

        with self._state_lock:
            self._dashboard_recent_fills.append(
                {
                    "ts": str(fill_ts),
                    "side": str(record.get("side", "")),
                    "price": str(fill_price),
                    "base_size": str(record.get("base_size", "0")),
                    "order_id": order_id,
                }
            )
            self._dashboard_recent_fills = self._dashboard_recent_fills[-40:]

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
        _ = self._coinbase_api_call("create_order", self.client.create_order, **payload)
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
    if config.hmm_states < 2:
        raise ValueError("HMM_STATES must be >= 2")
    if config.hmm_lookback < 30:
        raise ValueError("HMM_LOOKBACK must be >= 30")
    if config.hmm_iterations < 1:
        raise ValueError("HMM_ITERATIONS must be >= 1")
    if config.hmm_min_variance <= 0:
        raise ValueError("HMM_MIN_VARIANCE must be > 0")
    if config.base_order_notional_usd <= 0:
        raise ValueError("BASE_ORDER_NOTIONAL_USD must be > 0")
    if config.kelly_refresh_seconds < 30:
        raise ValueError("KELLY_REFRESH_SECONDS must be >= 30")
    if config.kelly_lookback_fills < 20:
        raise ValueError("KELLY_LOOKBACK_FILLS must be >= 20")
    if config.kelly_min_closed_trades < 1:
        raise ValueError("KELLY_MIN_CLOSED_TRADES must be >= 1")
    if not (Decimal("0") < config.kelly_min_allocation_frac <= config.kelly_max_allocation_frac):
        raise ValueError("KELLY_MIN_ALLOCATION_FRAC and KELLY_MAX_ALLOCATION_FRAC must satisfy 0 < min <= max")
    if config.min_notional_usd <= 0:
        raise ValueError("MIN_NOTIONAL_USD must be > 0")
    if not (Decimal("0") <= config.quote_reserve_pct < Decimal("1")):
        raise ValueError("QUOTE_RESERVE_PCT must be in [0,1)")
    if config.liquidity_depth_levels < 1:
        raise ValueError("LIQUIDITY_DEPTH_LEVELS must be >= 1")
    if not (Decimal("0") < config.liquidity_max_book_share_pct <= Decimal("1")):
        raise ValueError("LIQUIDITY_MAX_BOOK_SHARE_PCT must be in (0,1]")
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
    if config.api_latency_p95_threshold_ms <= 0:
        raise ValueError("API_LATENCY_P95_THRESHOLD_MS must be > 0")
    if not (Decimal("0") <= config.api_failure_rate_threshold_pct <= Decimal("1")):
        raise ValueError("API_FAILURE_RATE_THRESHOLD_PCT must be in [0,1]")
    if config.api_health_window_seconds < 60:
        raise ValueError("API_HEALTH_WINDOW_SECONDS must be >= 60")
    if config.api_recovery_consecutive_minutes < 1:
        raise ValueError("API_RECOVERY_CONSECUTIVE_MINUTES must be >= 1")


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


def _rsi(values: List[Decimal], period: int) -> Decimal:
    if period <= 0 or len(values) < period + 1:
        return Decimal("50")

    gains: List[Decimal] = []
    losses: List[Decimal] = []
    for prev, cur in zip(values, values[1:]):
        delta = cur - prev
        gains.append(max(Decimal("0"), delta))
        losses.append(max(Decimal("0"), -delta))

    avg_gain = sum(gains[-period:], Decimal("0")) / Decimal(period)
    avg_loss = sum(losses[-period:], Decimal("0")) / Decimal(period)
    if avg_loss <= 0:
        return Decimal("100") if avg_gain > 0 else Decimal("50")
    rs = avg_gain / avg_loss
    return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))


def _macd_histogram(values: List[Decimal], fast: int, slow: int, signal: int) -> Decimal:
    if len(values) < max(fast, slow) + signal:
        return Decimal("0")

    alpha_fast = Decimal("2") / Decimal(fast + 1)
    alpha_slow = Decimal("2") / Decimal(slow + 1)
    alpha_signal = Decimal("2") / Decimal(signal + 1)

    fast_ema = values[0]
    slow_ema = values[0]
    macd_series: List[Decimal] = []
    for val in values:
        fast_ema = (val * alpha_fast) + (fast_ema * (Decimal("1") - alpha_fast))
        slow_ema = (val * alpha_slow) + (slow_ema * (Decimal("1") - alpha_slow))
        macd_series.append(fast_ema - slow_ema)

    signal_ema = macd_series[0]
    for m in macd_series[1:]:
        signal_ema = (m * alpha_signal) + (signal_ema * (Decimal("1") - alpha_signal))
    return macd_series[-1] - signal_ema


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


def _returns(closes: List[Decimal]) -> List[float]:
    if len(closes) < 2:
        return []
    rets: List[float] = []
    prev = closes[0]
    for current in closes[1:]:
        if prev > 0 and current > 0:
            rets.append(float((current / prev) - Decimal("1")))
        prev = current
    return rets


def _fit_gaussian_hmm(
    observations: List[float],
    n_states: int,
    iterations: int,
    min_variance: Decimal,
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    t_len = len(observations)
    if t_len < 2 or n_states < 2:
        return None

    eps = 1e-12
    min_var = float(min_variance)
    sorted_obs = sorted(observations)

    means = [sorted_obs[int((idx + 1) * (t_len / (n_states + 1)))] for idx in range(n_states)]
    obs_mean = sum(observations) / t_len
    obs_var = sum((v - obs_mean) ** 2 for v in observations) / t_len
    variances = [max(obs_var, min_var) for _ in range(n_states)]
    init_prob = [1.0 / n_states for _ in range(n_states)]
    trans = []
    for i in range(n_states):
        row = []
        for j in range(n_states):
            row.append(0.85 if i == j else 0.15 / max(1, n_states - 1))
        trans.append(row)

    def emission(x: float, mean: float, var: float) -> float:
        var = max(var, min_var)
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var)
        exponent = math.exp(-((x - mean) ** 2) / (2.0 * var))
        return max(eps, coeff * exponent)

    for _ in range(iterations):
        b = [[emission(observations[t], means[i], variances[i]) for i in range(n_states)] for t in range(t_len)]
        alpha = [[0.0 for _ in range(n_states)] for _ in range(t_len)]
        beta = [[0.0 for _ in range(n_states)] for _ in range(t_len)]
        scales = [0.0 for _ in range(t_len)]

        for i in range(n_states):
            alpha[0][i] = init_prob[i] * b[0][i]
        scales[0] = max(eps, sum(alpha[0]))
        for i in range(n_states):
            alpha[0][i] /= scales[0]

        for t in range(1, t_len):
            for j in range(n_states):
                alpha[t][j] = sum(alpha[t - 1][i] * trans[i][j] for i in range(n_states)) * b[t][j]
            scales[t] = max(eps, sum(alpha[t]))
            for j in range(n_states):
                alpha[t][j] /= scales[t]

        for i in range(n_states):
            beta[-1][i] = 1.0
        for t in range(t_len - 2, -1, -1):
            for i in range(n_states):
                beta[t][i] = sum(trans[i][j] * b[t + 1][j] * beta[t + 1][j] for j in range(n_states))
                beta[t][i] /= max(eps, scales[t + 1])

        gamma = [[0.0 for _ in range(n_states)] for _ in range(t_len)]
        xi_sum = [[0.0 for _ in range(n_states)] for _ in range(n_states)]
        for t in range(t_len):
            norm = max(eps, sum(alpha[t][i] * beta[t][i] for i in range(n_states)))
            for i in range(n_states):
                gamma[t][i] = (alpha[t][i] * beta[t][i]) / norm

        for t in range(t_len - 1):
            denom = 0.0
            for i in range(n_states):
                for j in range(n_states):
                    denom += alpha[t][i] * trans[i][j] * b[t + 1][j] * beta[t + 1][j]
            denom = max(eps, denom)
            for i in range(n_states):
                for j in range(n_states):
                    numer = alpha[t][i] * trans[i][j] * b[t + 1][j] * beta[t + 1][j]
                    xi_sum[i][j] += numer / denom

        for i in range(n_states):
            init_prob[i] = gamma[0][i]
            trans_denom = max(eps, sum(gamma[t][i] for t in range(t_len - 1)))
            for j in range(n_states):
                trans[i][j] = xi_sum[i][j] / trans_denom
            row_sum = max(eps, sum(trans[i]))
            for j in range(n_states):
                trans[i][j] /= row_sum

            gamma_sum = max(eps, sum(gamma[t][i] for t in range(t_len)))
            means[i] = sum(gamma[t][i] * observations[t] for t in range(t_len)) / gamma_sum
            var = sum(gamma[t][i] * ((observations[t] - means[i]) ** 2) for t in range(t_len)) / gamma_sum
            variances[i] = max(var, min_var)

    last_probs = gamma[-1]
    total = max(eps, sum(last_probs))
    return [p / total for p in last_probs], means, variances


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


def _decimal_percentile(values: List[Decimal], percentile: Decimal) -> Decimal:
    if not values:
        return Decimal("0")
    pct = max(Decimal("0"), min(Decimal("1"), percentile))
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = int((len(ordered) - 1) * float(pct))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def _as_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "__dict__"):
        return dict(response.__dict__)
    raise TypeError(f"Unsupported response type: {type(response)!r}")


class StrategyManager:
    def __init__(self, client: Any, config: BotConfig):
        self.client = client
        self.config = config
        self.engines: List[StrategyEngine] = []
        self._shared_risk_state = SharedRiskState()
        self._risk_task: Optional[asyncio.Task[Any]] = None
        self._telegram_thread: Optional[threading.Thread] = None
        self._telegram_stop = threading.Event()
        self._telegram_ready = threading.Event()
        self._telegram_application_builder: Optional[Any] = None
        self._telegram_command_handler_cls: Optional[Any] = None

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

    def _compute_kelly_fraction(self, engine: StrategyEngine) -> Optional[Decimal]:
        with engine._db_lock:
            rows = engine._db.execute(
                """
                SELECT side, price, base_size, fee_paid
                FROM fills
                WHERE product_id = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (engine.config.product_id, int(self.config.kelly_lookback_fills)),
            ).fetchall()

        if not rows:
            return None

        fills = list(reversed(rows))
        open_buys: List[Tuple[Decimal, Decimal]] = []
        closed_results: List[Decimal] = []

        for side, price_raw, base_raw, fee_raw in fills:
            side_u = str(side).upper()
            price = Decimal(str(price_raw))
            base_size = Decimal(str(base_raw))
            fee_paid = Decimal(str(fee_raw))
            if price <= 0 or base_size <= 0:
                continue

            notional = price * base_size
            if side_u == "BUY":
                open_buys.append((base_size, notional + fee_paid))
                continue
            if side_u != "SELL":
                continue

            sell_remaining = base_size
            sell_proceeds = notional - fee_paid
            matched_cost = Decimal("0")
            matched_proceeds = Decimal("0")

            while sell_remaining > 0 and open_buys:
                buy_base, buy_cost = open_buys[0]
                if buy_base <= 0:
                    open_buys.pop(0)
                    continue
                matched = min(sell_remaining, buy_base)
                ratio = matched / buy_base
                cost_alloc = buy_cost * ratio
                proceeds_alloc = sell_proceeds * (matched / base_size)
                matched_cost += cost_alloc
                matched_proceeds += proceeds_alloc
                sell_remaining -= matched
                leftover_base = buy_base - matched
                leftover_cost = buy_cost - cost_alloc
                if leftover_base > 0:
                    open_buys[0] = (leftover_base, leftover_cost)
                else:
                    open_buys.pop(0)

            if matched_cost > 0:
                closed_results.append(matched_proceeds - matched_cost)

        if len(closed_results) < self.config.kelly_min_closed_trades:
            return None

        wins = [x for x in closed_results if x > 0]
        losses = [x for x in closed_results if x < 0]
        if not wins or not losses:
            return None

        p = Decimal(len(wins)) / Decimal(len(closed_results))
        avg_win = sum(wins, Decimal("0")) / Decimal(len(wins))
        avg_loss = sum((-x for x in losses), Decimal("0")) / Decimal(len(losses))
        if avg_loss <= 0:
            return None

        r = avg_win / avg_loss
        if r <= 0:
            return None

        kelly = (p * (r + Decimal("1")) - Decimal("1")) / r
        return max(Decimal("0"), kelly)

    def _refresh_kelly_allocations(self) -> None:
        if not self.config.kelly_allocation_enabled:
            for engine in self.engines:
                engine.set_capital_allocation(Decimal("1"), Decimal("0"))
            return

        for engine in self.engines:
            kelly_fraction = self._compute_kelly_fraction(engine)
            if kelly_fraction is None:
                engine.set_capital_allocation(Decimal("1"), Decimal("0"))
                continue

            multiplier = max(self.config.kelly_min_allocation_frac, min(kelly_fraction, self.config.kelly_max_allocation_frac))
            engine.set_capital_allocation(multiplier, kelly_fraction)

    async def _cross_asset_risk_loop(self) -> None:
        kelly_next_refresh = 0.0
        while any(engine._running for engine in self.engines):
            await asyncio.to_thread(self._refresh_cross_asset_risk)
            now = time.time()
            if now >= kelly_next_refresh:
                await asyncio.to_thread(self._refresh_kelly_allocations)
                kelly_next_refresh = now + max(30, self.config.kelly_refresh_seconds)
            await asyncio.sleep(max(5, self.config.cross_asset_refresh_seconds))

    async def run(self) -> None:
        shared_portfolio = self._shared_paper_portfolio()
        for product_id in self._product_ids():
            engine_cfg = replace(self.config, product_id=product_id, state_db_path=self._db_path_for(product_id))
            if shared_portfolio is not None:
                engine_cfg = replace(engine_cfg, paper_start_usd=Decimal("0"), paper_start_base=Decimal("0"), paper_start_btc=Decimal("0"))
            self.engines.append(
                GridStrategy(
                    client=self.client,
                    config=engine_cfg,
                    orders_path=self._orders_path_for(product_id),
                    shared_paper_portfolio=shared_portfolio,
                    shared_risk_state=self._shared_risk_state,
                )
            )

        self._start_telegram_controller()

        if len(self.engines) == 1:
            try:
                await self.engines[0].run()
            finally:
                self._stop_telegram_controller()
            return

        names = ", ".join(e.config.product_id for e in self.engines)
        logging.info("StrategyManager starting %s engines: %s", len(self.engines), names)
        self._risk_task = asyncio.create_task(self._cross_asset_risk_loop(), name="cross-asset-risk-loop")
        try:
            await asyncio.gather(*(engine.run() for engine in self.engines))
        finally:
            self._stop_telegram_controller()
            if self._risk_task is not None:
                self._risk_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._risk_task

    def stop(self) -> None:
        self._stop_telegram_controller()
        for engine in self.engines:
            engine.stop()

    def _find_engine(self, product_id: str) -> Optional[StrategyEngine]:
        wanted = product_id.strip().upper()
        for engine in self.engines:
            if engine.config.product_id.upper() == wanted:
                return engine
        return None

    def _start_telegram_controller(self) -> None:
        if self._telegram_thread is not None:
            return
        if not self.config.telegram_bot_token.strip():
            return
        try:
            telegram_ext = importlib.import_module("telegram.ext")
            self._telegram_application_builder = telegram_ext.Application.builder
            self._telegram_command_handler_cls = telegram_ext.CommandHandler
        except Exception:
            logging.warning("TELEGRAM_BOT_TOKEN set but python-telegram-bot is unavailable; Telegram commands disabled.")
            return
        self._telegram_stop.clear()
        self._telegram_ready.clear()
        self._telegram_thread = threading.Thread(target=self._telegram_thread_main, name="telegram-command-loop", daemon=True)
        self._telegram_thread.start()
        self._telegram_ready.wait(timeout=5)

    def _stop_telegram_controller(self) -> None:
        self._telegram_stop.set()
        if self._telegram_thread is not None:
            self._telegram_thread.join(timeout=10)
            self._telegram_thread = None

    def _telegram_thread_main(self) -> None:
        try:
            asyncio.run(self._run_telegram_controller())
        except Exception as exc:
            logging.warning("Telegram controller stopped: %s", exc)

    async def _run_telegram_controller(self) -> None:
        app = self._telegram_application_builder().token(self.config.telegram_bot_token.strip()).build()
        app.add_handler(self._telegram_command_handler_cls("pause", self._telegram_pause))
        app.add_handler(self._telegram_command_handler_cls("set_risk", self._telegram_set_risk))
        app.add_handler(self._telegram_command_handler_cls("report", self._telegram_report))
        await app.initialize()
        await app.start()
        if app.updater is None:
            raise RuntimeError("Telegram updater unavailable")
        await app.updater.start_polling(drop_pending_updates=True)
        self._telegram_ready.set()
        try:
            while not self._telegram_stop.is_set() and any(engine._running for engine in self.engines):
                await asyncio.sleep(1)
        finally:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()

    def _is_whitelisted_chat(self, update: Any) -> bool:
        allowed = self.config.telegram_whitelist_chat_id.strip()
        chat_id = ""
        if getattr(update, "effective_chat", None) is not None:
            chat_id = str(getattr(update.effective_chat, "id", ""))
        if not allowed:
            logging.warning("Telegram command received but TELEGRAM_WHITELIST_CHAT_ID is empty; rejecting command.")
            return False
        return chat_id == allowed

    async def _telegram_pause(self, update: Any, context: Any) -> None:
        if not self._is_whitelisted_chat(update):
            await update.effective_message.reply_text("Unauthorized chat.")
            return
        if not self.engines:
            await update.effective_message.reply_text("No active engines.")
            return
        target = (context.args[0] if context.args else "").strip().upper()
        if not target and len(self.engines) == 1:
            target = self.engines[0].config.product_id
        if not target:
            await update.effective_message.reply_text("Usage: /pause <PRODUCT_ID>")
            return
        engine = self._find_engine(target)
        if engine is None:
            await update.effective_message.reply_text(f"Unknown product_id: {target}")
            return
        engine.pause_trading()
        await update.effective_message.reply_text(f"Paused trading for {engine.config.product_id}.")

    async def _telegram_set_risk(self, update: Any, context: Any) -> None:
        if not self._is_whitelisted_chat(update):
            await update.effective_message.reply_text("Unauthorized chat.")
            return
        if not context.args:
            await update.effective_message.reply_text("Usage: /set_risk <0.0-1.0>")
            return
        try:
            new_cap = Decimal(str(context.args[0]))
        except Exception:
            await update.effective_message.reply_text("Invalid number. Usage: /set_risk <0.0-1.0>")
            return
        if not (Decimal("0") <= new_cap <= Decimal("1")):
            await update.effective_message.reply_text("Risk cap must be within [0.0, 1.0].")
            return

        for engine in self.engines:
            with engine._state_lock:
                engine.config.max_btc_inventory_pct = new_cap
                engine._active_inventory_cap_pct = new_cap
                engine._add_event(f"Inventory cap updated via Telegram: {new_cap}")
            self._shared_risk_state.set_inventory_cap(engine.config.product_id, new_cap)
        await update.effective_message.reply_text(
            f"Updated max inventory cap to {new_cap} across {len(self.engines)} engine(s)."
        )

    async def _telegram_report(self, update: Any, _context: Any) -> None:
        if not self._is_whitelisted_chat(update):
            await update.effective_message.reply_text("Unauthorized chat.")
            return
        if not self.engines:
            await update.effective_message.reply_text("No active engines.")
            return
        lines = []
        for engine in self.engines:
            snapshot = engine._status_snapshot()
            risk = snapshot.get("risk_metrics", {})
            lines.append(
                "\n".join(
                    [
                        f"{snapshot.get('product_id')}:",
                        f"  realized_pnl_total_usd={snapshot.get('realized_pnl_total_usd')}",
                        f"  var_95_24h_pct={risk.get('var_95_24h_pct', '0')}",
                        f"  paused={snapshot.get('paused')}",
                    ]
                )
            )
        await update.effective_message.reply_text("\n\n".join(lines))



# Backward-compatible aliases for existing integrations.
GridBot = GridStrategy
GridManager = StrategyManager

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
    manager = StrategyManager(client, config)

    def _handle_signal(signum: int, _frame: Any) -> None:
        logging.info("Received signal %s, shutting down cleanly.", signum)
        manager.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    asyncio.run(manager.run())


if __name__ == "__main__":
    main()
