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



@dataclass(frozen=True)
class BotConfig:
    product_id: str = os.getenv("PRODUCT_ID", "BTC-USD")
    grid_lines: int = int(os.getenv("GRID_LINES", "8"))
    grid_band_pct: Decimal = Decimal(os.getenv("GRID_BAND_PCT", "0.15"))
    min_notional_usd: Decimal = Decimal(os.getenv("MIN_NOTIONAL_USD", "6"))
    min_grid_profit_pct: Decimal = Decimal(os.getenv("MIN_GRID_PROFIT_PCT", "0.015"))
    poll_seconds: int = int(os.getenv("POLL_SECONDS", "60"))

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

    # execution mode
    paper_trading_mode: bool = os.getenv("PAPER_TRADING_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
    paper_start_usd: Decimal = Decimal(os.getenv("PAPER_START_USD", "1000"))
    paper_start_btc: Decimal = Decimal(os.getenv("PAPER_START_BTC", "0"))

    # local dashboard
    dashboard_enabled: bool = os.getenv("DASHBOARD_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    dashboard_host: str = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8080"))


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
        self.paper_balances = {
            "USD": self.config.paper_start_usd,
            "BTC": self.config.paper_start_btc,
        }

    def run(self) -> None:
        if self.config.dashboard_enabled:
            self._start_dashboard_server()
        self._load_product_metadata()
        current_price = self._get_current_price()
        self.grid_anchor_price = current_price
        self.grid_levels = self._build_grid_levels(current_price)

        if not self.orders:
            logging.info("No persisted orders found; placing initial adaptive grid.")
            self._place_initial_grid_orders(current_price)
            self._save_orders()
        else:
            logging.info("Loaded %s persisted active orders.", len(self.orders))

        while self._running:
            try:
                current_price = self._get_current_price()
                trend_bias = self._get_trend_bias()
                with self._state_lock:
                    self.loop_count += 1
                    self.last_price = current_price
                    self.last_trend_bias = trend_bias
                self._risk_monitor(current_price)
                self._process_open_orders(current_price=current_price, trend_bias=trend_bias)
                self._save_orders()
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

        lower = current_price * (Decimal("1") - self.config.grid_band_pct)
        upper = current_price * (Decimal("1") + self.config.grid_band_pct)
        step = (upper - lower) / Decimal(self.config.grid_lines - 1)
        step_pct = step / current_price
        if step_pct < self.config.min_grid_profit_pct:
            raise ValueError(
                f"Grid spacing {step_pct:.4%} is below required minimum {self.config.min_grid_profit_pct:.4%}."
            )

        levels = [self._q_price(lower + step * Decimal(i)) for i in range(self.config.grid_lines)]
        logging.info("Grid: lower=%s upper=%s lines=%s step≈%.2f%%", levels[0], levels[-1], len(levels), float(step_pct * 100))
        return levels

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
        if side == "BUY" and current_price <= price:
            return "FILLED"
        if side == "SELL" and current_price >= price:
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
        if btc_ratio > self.config.max_btc_inventory_pct:
            logging.warning(
                "Risk: BTC inventory %.2f%% exceeds limit %.2f%%. New buy placements are throttled.",
                float(btc_ratio * 100),
                float(self.config.max_btc_inventory_pct * 100),
            )

        stop_price = self.grid_anchor_price * (Decimal("1") - self.config.hard_stop_loss_pct)
        if current_price < stop_price:
            logging.warning(
                "Hard stop-loss zone breached (price=%s < %s). Buy replacements remain suspended in downtrend.",
                current_price,
                self._q_price(stop_price),
            )

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
            if self._btc_inventory_ratio(price) > self.config.max_btc_inventory_pct:
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

        self.orders[order_id] = {
            "side": side,
            "price": str(price),
            "base_size": str(base_size),
            "grid_index": grid_index,
            "product_id": self.config.product_id,
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
            return "NEUTRAL"

        ema_fast = _ema(closes, self.config.trend_ema_fast)
        ema_slow = _ema(closes, self.config.trend_ema_slow)
        if ema_slow <= 0:
            return "NEUTRAL"

        strength = (ema_fast - ema_slow) / ema_slow
        if strength >= self.config.trend_strength_threshold:
            return "UP"
        if strength <= -self.config.trend_strength_threshold:
            return "DOWN"
        return "NEUTRAL"

    def _fetch_public_candle_closes(self) -> List[Decimal]:
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
        rows: List[Tuple[int, Decimal]] = []
        for candle in candles:
            close = candle.get("close")
            start = candle.get("start")
            if close is None or start is None:
                continue
            rows.append((int(start), Decimal(str(close))))

        rows.sort(key=lambda x: x[0])
        return [r[1] for r in rows]

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
        if not self.orders_path.exists():
            return {}
        try:
            return json.loads(self.orders_path.read_text())
        except Exception as exc:
            logging.warning("Failed reading %s: %s", self.orders_path, exc)
            return {}

    def _save_orders(self) -> None:
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
        notional = base_size * price
        if side == "BUY":
            self.paper_balances["USD"] = self.paper_balances["USD"] - notional
            self.paper_balances["BTC"] = self.paper_balances["BTC"] + base_size
        elif side == "SELL":
            self.paper_balances["USD"] = self.paper_balances["USD"] + notional
            self.paper_balances["BTC"] = self.paper_balances["BTC"] - base_size

    def _add_event(self, message: str) -> None:
        with self._state_lock:
            self.recent_events.append(f"{time.strftime('%H:%M:%S')} | {message}")
            self.recent_events = self.recent_events[-25:]

    def _status_snapshot(self) -> Dict[str, Any]:
        with self._state_lock:
            price = self.last_price
            usd_bal = self._get_available_balance("USD")
            btc_bal = self._get_available_balance("BTC")
            portfolio = usd_bal + (btc_bal * price if price > 0 else Decimal("0"))
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


def _validate_config(config: BotConfig) -> None:
    if config.grid_lines < 2:
        raise ValueError("GRID_LINES must be >= 2")
    if config.poll_seconds < 5:
        raise ValueError("POLL_SECONDS must be >= 5")
    if config.base_order_notional_usd <= 0:
        raise ValueError("BASE_ORDER_NOTIONAL_USD must be > 0")
    if config.min_notional_usd <= 0:
        raise ValueError("MIN_NOTIONAL_USD must be > 0")
    if not (Decimal("0") <= config.quote_reserve_pct < Decimal("1")):
        raise ValueError("QUOTE_RESERVE_PCT must be in [0,1)")
    if not (Decimal("0") < config.max_btc_inventory_pct <= Decimal("1")):
        raise ValueError("MAX_BTC_INVENTORY_PCT must be in (0,1]")
    if config.dashboard_port <= 0:
        raise ValueError("DASHBOARD_PORT must be > 0")
    if config.paper_start_usd < 0 or config.paper_start_btc < 0:
        raise ValueError("PAPER_START_USD and PAPER_START_BTC must be >= 0")


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
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta http-equiv=\"refresh\" content=\"10\" />
  <title>Thumber Trader Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background:#111; color:#e6e6e6; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:16px; }}
    .card {{ background:#1d1d1d; padding:12px; border-radius:8px; }}
    table {{ width:100%; border-collapse: collapse; font-size:14px; }}
    td, th {{ border-bottom:1px solid #333; padding:8px; text-align:left; }}
  </style>
</head>
<body>
  <h1>Thumber Trader Dashboard</h1>
  <p>Mode: <strong>{'PAPER' if snapshot.get('paper_trading_mode') else 'LIVE'}</strong></p>
  <div class=\"grid\">
    <div class=\"card\"><div>Product</div><strong>{snapshot.get('product_id')}</strong></div>
    <div class=\"card\"><div>Last Price</div><strong>{snapshot.get('last_price')}</strong></div>
    <div class=\"card\"><div>Trend</div><strong>{snapshot.get('trend_bias')}</strong></div>
    <div class=\"card\"><div>Portfolio (USD)</div><strong>{snapshot.get('portfolio_value_usd')}</strong></div>
    <div class=\"card\"><div>Active Orders</div><strong>{snapshot.get('active_orders')}</strong></div>
    <div class=\"card\"><div>Fills</div><strong>{snapshot.get('fills')}</strong></div>
  </div>
  <h2>Open Orders</h2>
  <table>
    <thead><tr><th>Order ID</th><th>Side</th><th>Price</th><th>Base Size</th><th>Grid Index</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Recent Events</h2>
  <ul>{events}</ul>
  <p><a href=\"/api/status\">JSON API</a></p>
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
