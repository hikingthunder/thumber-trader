#!/usr/bin/env python3
"""Native historical backtester for the adaptive grid strategy.

This script reuses GridBot execution logic in paper mode, but replaces live
Coinbase REST calls with a mock client that walks through historical 1-minute
OHLCV candles.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import tempfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from grid_bot import BotConfig, GridBot, _validate_config


Candle = Tuple[int, Decimal, Decimal, Decimal, Decimal, Decimal]


class HistoricalRESTClient:
    """Mock replacement for Coinbase RESTClient backed by historical candles."""

    def __init__(self, product_id: str, candles: Sequence[Candle]):
        if not candles:
            raise ValueError("HistoricalRESTClient requires at least one candle")
        self.product_id = product_id
        self._candles = list(candles)
        self._idx = 0

    def set_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._candles):
            raise IndexError(f"Candle index out of range: {idx}")
        self._idx = idx

    def candle_count(self) -> int:
        return len(self._candles)

    def current_candle(self) -> Candle:
        return self._candles[self._idx]

    def get_product(self, product_id: str) -> Dict[str, Any]:
        if product_id != self.product_id:
            raise ValueError(f"Unsupported product requested: {product_id}")
        _ts, _open, _high, _low, close, _volume = self.current_candle()
        return {
            "product_id": product_id,
            "price": str(close),
            "quote_increment": "0.01",
            "base_increment": "0.00000001",
        }


class BacktestGridBot(GridBot):
    """GridBot variant that sources trend candles from in-memory history."""

    def __init__(self, *args: Any, historical_candles: Sequence[Candle], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._historical_candles = list(historical_candles)
        self._cursor = 0

    def set_cursor(self, cursor: int) -> None:
        self._cursor = cursor

    def _fetch_public_candles(self) -> List[Tuple[int, Decimal, Decimal, Decimal]]:
        # GridBot expects tuples: (timestamp, high, low, close)
        if not self._historical_candles:
            return []
        start_idx = max(0, self._cursor - self.config.trend_candle_limit + 1)
        window = self._historical_candles[start_idx : self._cursor + 1]
        return [(ts, high, low, close) for ts, _open, high, low, close, _volume in window]


def _parse_decimal(raw: str) -> Decimal:
    return Decimal(str(raw).strip())


def load_ohlcv_csv(path: Path) -> List[Candle]:
    """Load 1-minute OHLCV candles from CSV.

    Expected header columns (case-insensitive):
      - timestamp (or ts/time)
      - open
      - high
      - low
      - close
      - volume
    """

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV must include a header row")
        fields = {name.lower().strip(): name for name in reader.fieldnames}

        def _required(*names: str) -> str:
            for name in names:
                if name in fields:
                    return fields[name]
            raise ValueError(f"Missing required CSV column. expected one of: {names}")

        ts_col = _required("timestamp", "ts", "time")
        open_col = _required("open")
        high_col = _required("high")
        low_col = _required("low")
        close_col = _required("close")
        volume_col = _required("volume")

        rows: List[Candle] = []
        for row in reader:
            ts = int(str(row[ts_col]).strip())
            open_px = _parse_decimal(row[open_col])
            high_px = _parse_decimal(row[high_col])
            low_px = _parse_decimal(row[low_col])
            close_px = _parse_decimal(row[close_col])
            volume = _parse_decimal(row[volume_col])
            rows.append((ts, open_px, high_px, low_px, close_px, volume))

    rows.sort(key=lambda item: item[0])
    if len(rows) < 10:
        raise ValueError("Not enough candles to backtest (need at least 10 rows)")
    return rows


def _format_decimal(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000001")), "f")


def run_backtest(config: BotConfig, candles: Sequence[Candle], start_index: int, end_index: int) -> Dict[str, Any]:
    client = HistoricalRESTClient(product_id=config.product_id, candles=candles)

    with tempfile.TemporaryDirectory(prefix="thumber-backtest-") as tmp_dir:
        tmp = Path(tmp_dir)
        bot = BacktestGridBot(
            client=client,
            config=config,
            orders_path=tmp / "orders.json",
            historical_candles=candles,
        )

        client.set_index(start_index)
        bot.set_cursor(start_index)

        bot._load_product_metadata()
        bot._refresh_maker_fee(force=True)
        initial_price = bot._get_current_price()
        bot.grid_anchor_price = initial_price
        if bot.config.paper_trading_mode and bot.config.paper_start_base > 0:
            bot._paper_inventory_cost_usd = bot.config.paper_start_base * initial_price
        bot.grid_levels = bot._build_grid_levels(initial_price)
        bot._run_safe_start_checks(initial_price)
        bot._place_initial_grid_orders(initial_price)

        for idx in range(start_index, end_index + 1):
            client.set_index(idx)
            bot.set_cursor(idx)
            _ts, _open, _high, _low, close, _volume = candles[idx]
            current_price = close
            bot.last_price = current_price
            trend_bias = bot._get_trend_bias()
            bot.last_trend_bias = trend_bias
            bot.loop_count += 1
            bot._maybe_roll_grid(current_price)
            bot._process_open_orders(current_price, trend_bias)
            bot._risk_monitor(current_price)

        end_price = candles[end_index][4]
        usd_balance = bot._get_available_balance("USD")
        base_balance = bot._get_available_balance(bot.base_currency)
        equity = usd_balance + (base_balance * end_price)
        pnl = equity - (config.paper_start_usd + (config.paper_start_base * initial_price))

        return {
            "product_id": config.product_id,
            "grid_lines": config.grid_lines,
            "candles_processed": (end_index - start_index) + 1,
            "start_ts": candles[start_index][0],
            "end_ts": candles[end_index][0],
            "start_price": _format_decimal(initial_price),
            "end_price": _format_decimal(end_price),
            "ending_usd": _format_decimal(usd_balance),
            "ending_base": _format_decimal(base_balance),
            "ending_equity_usd": _format_decimal(equity),
            "net_pnl_usd": _format_decimal(pnl),
            "fills": bot.fill_count,
            "open_orders_remaining": len(bot.orders),
            "final_trend_bias": bot.last_trend_bias,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the grid bot over historical 1-minute OHLCV data")
    parser.add_argument("--csv", type=Path, required=True, help="Path to OHLCV CSV (timestamp,open,high,low,close,volume)")
    parser.add_argument("--product-id", default="BTC-USD", help="Trading pair identifier")
    parser.add_argument("--grid-lines", type=int, default=8, help="Grid line count to test")
    parser.add_argument(
        "--compare-grid-lines",
        default="",
        help="Optional comma-separated grid lines to compare in one run (example: 8,10,12)",
    )
    parser.add_argument("--lookback-minutes", type=int, default=30 * 24 * 60, help="Trailing minutes to replay")
    parser.add_argument("--paper-start-usd", type=Decimal, default=Decimal("1000"), help="Starting paper USD balance")
    parser.add_argument("--paper-start-base", type=Decimal, default=Decimal("0"), help="Starting paper base-asset balance")
    parser.add_argument("--base-order-notional", type=Decimal, default=Decimal("10"), help="Base notional per buy order")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Bot poll interval (used for config validation only)")
    parser.add_argument(
        "--wfo-enabled",
        action="store_true",
        help="Enable walk-forward optimization (train on in-sample window, validate on out-of-sample window)",
    )
    parser.add_argument(
        "--wfo-in-sample-days",
        type=int,
        default=7,
        help="In-sample training window size in days when --wfo-enabled is used",
    )
    parser.add_argument(
        "--wfo-out-sample-days",
        type=int,
        default=2,
        help="Out-of-sample validation window size in days when --wfo-enabled is used",
    )
    parser.add_argument(
        "--wfo-atr-multipliers",
        default="3.0,4.0,5.0",
        help="Comma-separated ATR band multiplier candidates for WFO",
    )
    parser.add_argument(
        "--wfo-adx-range-multipliers",
        default="0.7,0.8,0.9",
        help="Comma-separated ADX ranging band multiplier candidates for WFO",
    )
    parser.add_argument(
        "--wfo-adx-trend-multipliers",
        default="1.1,1.25,1.4",
        help="Comma-separated ADX trending band multiplier candidates for WFO",
    )
    return parser.parse_args()


def _grid_line_scenarios(primary_grid_lines: int, compare_csv: str) -> List[int]:
    if not compare_csv.strip():
        return [primary_grid_lines]
    values = [int(item.strip()) for item in compare_csv.split(",") if item.strip()]
    if primary_grid_lines not in values:
        values.append(primary_grid_lines)
    return sorted(set(values))


def _build_backtest_config(args: argparse.Namespace, grid_lines: int, state_db_path: str) -> BotConfig:
    cfg = BotConfig()
    cfg = replace(
        cfg,
        product_id=args.product_id.upper(),
        product_ids=args.product_id.upper(),
        grid_lines=grid_lines,
        poll_seconds=args.poll_seconds,
        paper_trading_mode=True,
        paper_start_usd=args.paper_start_usd,
        paper_start_base=args.paper_start_base,
        paper_start_btc=args.paper_start_base,
        base_order_notional_usd=args.base_order_notional,
        paper_fill_delay_seconds=0,
        paper_fill_exceed_pct=Decimal("0"),
        dashboard_enabled=False,
        prometheus_enabled=False,
        safe_start_enabled=False,
        state_db_path=state_db_path,
    )
    _validate_config(cfg)
    return cfg


def _parse_decimal_candidates(raw: str, field_name: str) -> List[Decimal]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{field_name} requires at least one candidate value")
    return [Decimal(value) for value in values]


def _run_walk_forward(
    *,
    args: argparse.Namespace,
    candles: Sequence[Candle],
    start_index: int,
    end_index: int,
    grid_lines: int,
) -> Dict[str, Any]:
    in_sample_minutes = args.wfo_in_sample_days * 24 * 60
    out_sample_minutes = args.wfo_out_sample_days * 24 * 60
    if in_sample_minutes <= 0 or out_sample_minutes <= 0:
        raise ValueError("WFO windows must be > 0 days")

    atr_candidates = _parse_decimal_candidates(args.wfo_atr_multipliers, "--wfo-atr-multipliers")
    adx_range_candidates = _parse_decimal_candidates(args.wfo_adx_range_multipliers, "--wfo-adx-range-multipliers")
    adx_trend_candidates = _parse_decimal_candidates(args.wfo_adx_trend_multipliers, "--wfo-adx-trend-multipliers")

    total_needed = in_sample_minutes + out_sample_minutes
    total_available = (end_index - start_index) + 1
    if total_available < total_needed:
        raise ValueError(
            f"Not enough candles for WFO: need at least {total_needed}, found {total_available} in requested lookback"
        )

    folds: List[Dict[str, Any]] = []
    cursor = start_index
    fold_id = 1
    while True:
        in_start = cursor
        in_end = in_start + in_sample_minutes - 1
        out_start = in_end + 1
        out_end = out_start + out_sample_minutes - 1
        if out_end > end_index:
            break

        best_train_result: Dict[str, Any] | None = None
        best_params: Dict[str, str] | None = None

        for atr_mult in atr_candidates:
            for adx_range_mult in adx_range_candidates:
                for adx_trend_mult in adx_trend_candidates:
                    with tempfile.TemporaryDirectory(prefix="thumber-backtest-state-") as tmp_state:
                        base_cfg = _build_backtest_config(
                            args=args,
                            grid_lines=grid_lines,
                            state_db_path=str(Path(tmp_state) / f"wfo_train_{fold_id}.db"),
                        )
                        tuned_cfg = replace(
                            base_cfg,
                            atr_enabled=True,
                            atr_band_multiplier=atr_mult,
                            adx_range_band_multiplier=adx_range_mult,
                            adx_trend_band_multiplier=adx_trend_mult,
                        )
                        train_result = run_backtest(
                            config=tuned_cfg,
                            candles=candles,
                            start_index=in_start,
                            end_index=in_end,
                        )
                    train_pnl = Decimal(train_result["net_pnl_usd"])
                    if best_train_result is None or train_pnl > Decimal(best_train_result["net_pnl_usd"]):
                        best_train_result = train_result
                        best_params = {
                            "atr_band_multiplier": str(atr_mult),
                            "adx_range_band_multiplier": str(adx_range_mult),
                            "adx_trend_band_multiplier": str(adx_trend_mult),
                        }

        if best_train_result is None or best_params is None:
            raise RuntimeError("WFO failed to evaluate parameter candidates")

        with tempfile.TemporaryDirectory(prefix="thumber-backtest-state-") as tmp_state:
            fold_cfg = _build_backtest_config(
                args=args,
                grid_lines=grid_lines,
                state_db_path=str(Path(tmp_state) / f"wfo_test_{fold_id}.db"),
            )
            fold_cfg = replace(
                fold_cfg,
                atr_enabled=True,
                atr_band_multiplier=Decimal(best_params["atr_band_multiplier"]),
                adx_range_band_multiplier=Decimal(best_params["adx_range_band_multiplier"]),
                adx_trend_band_multiplier=Decimal(best_params["adx_trend_band_multiplier"]),
            )
            out_result = run_backtest(config=fold_cfg, candles=candles, start_index=out_start, end_index=out_end)

        folds.append(
            {
                "fold": fold_id,
                "in_sample_window": {"start_ts": candles[in_start][0], "end_ts": candles[in_end][0]},
                "out_sample_window": {"start_ts": candles[out_start][0], "end_ts": candles[out_end][0]},
                "best_params": best_params,
                "in_sample_result": best_train_result,
                "out_sample_result": out_result,
            }
        )

        fold_id += 1
        cursor = out_start

    if not folds:
        raise ValueError("WFO produced zero folds; increase lookback or reduce WFO window sizes")

    total_out_sample_pnl = sum(Decimal(fold["out_sample_result"]["net_pnl_usd"]) for fold in folds)
    avg_out_sample_pnl = total_out_sample_pnl / Decimal(len(folds))
    profitable_folds = sum(1 for fold in folds if Decimal(fold["out_sample_result"]["net_pnl_usd"]) > 0)

    return {
        "mode": "walk_forward_optimization",
        "product_id": args.product_id.upper(),
        "grid_lines": grid_lines,
        "window": {"start_ts": candles[start_index][0], "end_ts": candles[end_index][0]},
        "settings": {
            "in_sample_days": args.wfo_in_sample_days,
            "out_sample_days": args.wfo_out_sample_days,
            "atr_band_multiplier_candidates": [str(value) for value in atr_candidates],
            "adx_range_band_multiplier_candidates": [str(value) for value in adx_range_candidates],
            "adx_trend_band_multiplier_candidates": [str(value) for value in adx_trend_candidates],
        },
        "summary": {
            "fold_count": len(folds),
            "out_sample_profitable_folds": profitable_folds,
            "out_sample_win_rate": f"{(profitable_folds / len(folds)):.4f}",
            "total_out_sample_pnl_usd": _format_decimal(total_out_sample_pnl),
            "avg_out_sample_pnl_usd": _format_decimal(avg_out_sample_pnl),
        },
        "folds": folds,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    candles = load_ohlcv_csv(args.csv)
    if args.lookback_minutes <= 0:
        raise ValueError("--lookback-minutes must be > 0")

    start_index = max(0, len(candles) - args.lookback_minutes)
    end_index = len(candles) - 1
    if start_index >= end_index:
        raise ValueError("Not enough candles available for requested lookback window")

    scenarios = _grid_line_scenarios(args.grid_lines, args.compare_grid_lines)
    results: List[Dict[str, Any]] = []

    for grid_lines in scenarios:
        if args.wfo_enabled:
            result = _run_walk_forward(
                args=args,
                candles=candles,
                start_index=start_index,
                end_index=end_index,
                grid_lines=grid_lines,
            )
            results.append(result)
            continue

        with tempfile.TemporaryDirectory(prefix="thumber-backtest-state-") as tmp_state:
            config = _build_backtest_config(
                args=args,
                grid_lines=grid_lines,
                state_db_path=str(Path(tmp_state) / f"grid_state_{grid_lines}.db"),
            )
            result = run_backtest(config=config, candles=candles, start_index=start_index, end_index=end_index)
            results.append(result)

    if len(results) == 1:
        print(json.dumps(results[0], indent=2))
        return

    print(json.dumps({"window": {"start_ts": candles[start_index][0], "end_ts": candles[end_index][0]}, "results": results}, indent=2))


if __name__ == "__main__":
    main()
