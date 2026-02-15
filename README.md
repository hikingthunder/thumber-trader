# Coinbase Adaptive Grid Bot (BTC-USD default)

`grid_bot.py` is a Python bot for Coinbase Advanced Trade using `coinbase-advanced-py`.

This README is written for a **Debian 13 (Trixie) Proxmox LXC service deployment** with practical, copy/paste steps.

## What this bot does

- Uses a neutral grid anchor at startup.
- Creates a default ±15% band around the startup price.
- Uses 8 grid lines by default and enforces a minimum grid spacing floor.
- Supports arithmetic or geometric spacing (`GRID_SPACING_MODE`).
- Can auto-size grid bands using ATR (`ATR_ENABLED=true`).
- Enforces fee-aware spacing floor via maker-fee + target-net-profit settings.
- Places **post-only** limit orders (maker intent).
- Persists state in SQLite (`grid_state.db`) for restart recovery and reporting.
- Runs the core engine on `asyncio` with concurrent market polling, risk monitoring, and user-stream handling.
- Uses Coinbase `WSUserClient` for low-latency fill notifications and immediate replacement orders.
- Supports multi-source consensus pricing by blending mid-prices from Coinbase/Binance/Kraken/Bybit (configurable + outlier filtering).
- Computes an alpha fusion confidence score per grid level by combining RSI, MACD histogram, and order book imbalance.
- Detects order-flow toxicity with VPIN (volume-time buckets), then auto-widens grid bands or pauses entries when VPIN breaches its rolling 95th-percentile threshold.
- Tracks Coinbase WebSocket sequence numbers; on any gap, automatically triggers REST `get_orders` reconciliation so local SQLite state recovers before normal execution resumes.
- Keeps periodic polling (60s default) as a resilience backstop and market-data refresh path.
- Supports optional exchange-side attached bracket exits for BUY entries.
- Refreshes maker fee data hourly (configurable) and auto-updates spacing economics.
- Enhances paper trading with queue-delay, exceed-threshold, and slippage simulation.
- Exposes daily normalized PnL, turnover, and VaR/CVaR risk metrics in dashboard status.
- Exposes Prometheus metrics (`/metrics` by default) for realized PnL, inventory skew, equity curve, portfolio beta, Coinbase API latency histogram, and API safe-mode state.
- Includes an API-health operational circuit breaker that enters Safe Mode (pauses new BUY/entry orders) when latency/failure thresholds are breached, then auto-resumes after sustained recovery.
- Adds optional sentiment-driven Safe Mode overrides (e.g., X/Twitter or news API feed) that cancel pending BUY orders and tighten inventory caps toward USD during panic regimes.
- Sends a high-priority Telegram alert when API instability forces Safe Mode (optional token/chat-id configuration).
- Applies risk controls (USD reserve, BTC inventory cap, stop-loss awareness).
- Adds trend bias (EMA fast/slow over Coinbase candles) to be more defensive in downtrends.
- Detects market regime via ADX (or optional HMM probabilistic states) and auto-adjusts grid width + buy order notional for trending vs ranging conditions.
- In multi-asset mode, tightens correlated products' inventory caps when one asset is already near its own inventory limit.

## 1) Debian 13 / Proxmox LXC setup

### Install system packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git curl ca-certificates
```

### Create a dedicated runtime user (recommended)

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin gridbot
```

### Place app files

```bash
sudo mkdir -p /opt/thumber-trader
sudo chown -R gridbot:gridbot /opt/thumber-trader
# Copy grid_bot.py and this README into /opt/thumber-trader
```

### Create virtual environment and install dependency

```bash
sudo -u gridbot python3 -m venv /opt/thumber-trader/.venv
sudo -u gridbot /opt/thumber-trader/.venv/bin/pip install --upgrade pip
sudo -u gridbot /opt/thumber-trader/.venv/bin/pip install coinbase-advanced-py
```

## 2) Secure Coinbase credentials

Use file-based secrets with strict permissions:

```bash
sudo -u gridbot mkdir -p /opt/thumber-trader/secrets
sudo -u gridbot bash -lc 'umask 077 && cat > /opt/thumber-trader/secrets/cb_api_key'
sudo -u gridbot bash -lc 'umask 077 && cat > /opt/thumber-trader/secrets/cb_api_secret'
```

Verify permissions:

```bash
stat -c '%a %n' /opt/thumber-trader/secrets/cb_api_key /opt/thumber-trader/secrets/cb_api_secret
# should print 600 for both files
```

The bot refuses group/world-readable secret files.

## 3) Create service environment file

Create `/opt/thumber-trader/.env`:

```bash
sudo -u gridbot tee /opt/thumber-trader/.env >/dev/null <<'ENV'
COINBASE_API_KEY_FILE=/opt/thumber-trader/secrets/cb_api_key
COINBASE_API_SECRET_FILE=/opt/thumber-trader/secrets/cb_api_secret

PRODUCT_ID=BTC-USD
# Optional multi-asset mode (comma-separated). Example: BTC-USD,ETH-USD,SOL-USD
PRODUCT_IDS=BTC-USD
GRID_LINES=8
GRID_BAND_PCT=0.15
MIN_NOTIONAL_USD=6
MIN_GRID_PROFIT_PCT=0.015
MAKER_FEE_PCT=0.004
TARGET_NET_PROFIT_PCT=0.002
POLL_SECONDS=60

# Multi-venue consensus pricing
CONSENSUS_PRICING_ENABLED=true
CONSENSUS_EXCHANGES=coinbase,binance,kraken,bybit
CONSENSUS_MAX_DEVIATION_PCT=0.02

# Alpha fusion (RSI + MACD + order-book imbalance)
ALPHA_FUSION_ENABLED=true
ALPHA_RSI_PERIOD=14
ALPHA_MACD_FAST=12
ALPHA_MACD_SLOW=26
ALPHA_MACD_SIGNAL=9
ALPHA_WEIGHT_RSI=0.3
ALPHA_WEIGHT_MACD=0.3
ALPHA_WEIGHT_IMBALANCE=0.4

# VPIN toxicity detection (widen|pause|both)
VPIN_ENABLED=true
VPIN_BUCKET_VOLUME_BASE=0.25
VPIN_ROLLING_BUCKETS=50
VPIN_HISTORY_SIZE=300
VPIN_THRESHOLD_PERCENTILE=0.95
VPIN_RESPONSE_MODE=widen
VPIN_WIDEN_BAND_MULTIPLIER=1.5

DYNAMIC_FEE_TRACKING_ENABLED=true
FEE_REFRESH_SECONDS=3600

# Optional native bracket attachment (BUY entries only)
USE_EXCHANGE_BRACKET_ORDERS=false
BRACKET_TAKE_PROFIT_PCT=0.01
BRACKET_STOP_LOSS_PCT=0.01

# Grid model: arithmetic|geometric
GRID_SPACING_MODE=arithmetic

# Optional ATR-adaptive grid band
ATR_ENABLED=false
ATR_PERIOD=14
ATR_BAND_MULTIPLIER=4
ATR_MIN_BAND_PCT=0.03
ATR_MAX_BAND_PCT=0.35

# Optional local dashboard
DASHBOARD_ENABLED=true
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=8080

# Prometheus exporter endpoint (scrape with Prometheus; visualize in Grafana)
PROMETHEUS_ENABLED=true
PROMETHEUS_PATH=/metrics

# API operational circuit breaker
API_CIRCUIT_BREAKER_ENABLED=true
API_LATENCY_P95_THRESHOLD_MS=2000
API_FAILURE_RATE_THRESHOLD_PCT=0.05
API_HEALTH_WINDOW_SECONDS=300
API_RECOVERY_CONSECUTIVE_MINUTES=5

# Optional sentiment override (expects JSON score from SENTIMENT_SOURCE_URL)
SENTIMENT_OVERRIDE_ENABLED=false
SENTIMENT_SOURCE_URL=
SENTIMENT_API_BEARER_TOKEN=
SENTIMENT_JSON_PATH=score
SENTIMENT_ASSET_QUERY_PARAM=symbol
SENTIMENT_REFRESH_SECONDS=300
SENTIMENT_LOOKBACK_SECONDS=3600
SENTIMENT_NEGATIVE_THRESHOLD=-0.6
SENTIMENT_SAFE_INVENTORY_CAP_PCT=0.20

# Optional Telegram alerting + command-and-control
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
# Required for secure remote commands (/pause, /set_risk, /report)
TELEGRAM_WHITELIST_CHAT_ID=

# Safer dry-run mode (simulated fills + balances)
PAPER_TRADING_MODE=false
PAPER_START_USD=1000
PAPER_START_BTC=0
# Alias used by multi-asset engines for each product base currency in paper mode
PAPER_START_BASE=0
SHARED_USD_RESERVE_ENABLED=true

# Paper fill realism controls
PAPER_FILL_EXCEED_PCT=0.0001
PAPER_FILL_DELAY_SECONDS=5
PAPER_SLIPPAGE_PCT=0.0001

BASE_ORDER_NOTIONAL_USD=10
QUOTE_RESERVE_PCT=0.25
MAX_BTC_INVENTORY_PCT=0.65
HARD_STOP_LOSS_PCT=0.08

# Optional liquidity-aware sell sizing using level2 order book depth
LIQUIDITY_DEPTH_CHECK_ENABLED=true
LIQUIDITY_DEPTH_LEVELS=50
LIQUIDITY_MAX_BOOK_SHARE_PCT=0.20

# Optional dynamic inventory cap scaling by trend strength
DYNAMIC_INVENTORY_CAP_ENABLED=false
INVENTORY_CAP_MIN_PCT=0.30
INVENTORY_CAP_MAX_PCT=0.80

# Optional cross-asset correlation risk filter
CROSS_ASSET_CORRELATION_ENABLED=true
CROSS_ASSET_CORRELATION_THRESHOLD=0.85
CROSS_ASSET_LEADER_INVENTORY_TRIGGER_PCT=0.80
CROSS_ASSET_INVENTORY_TIGHTENING_FACTOR=0.65
CROSS_ASSET_INVENTORY_MIN_PCT=0.20
CROSS_ASSET_CANDLE_LOOKBACK=48
CROSS_ASSET_REFRESH_SECONDS=300

TREND_GRANULARITY=ONE_HOUR
TREND_CANDLE_LIMIT=72
TREND_EMA_FAST=9
TREND_EMA_SLOW=21
TREND_STRENGTH_THRESHOLD=0.003

# Market regime guardrail (ADX)
ADX_PERIOD=14
ADX_RANGING_THRESHOLD=20
ADX_TRENDING_THRESHOLD=25
ADX_RANGE_BAND_MULTIPLIER=0.8
ADX_TREND_BAND_MULTIPLIER=1.25
ADX_TREND_ORDER_SIZE_MULTIPLIER=0.7

# Optional probabilistic market regime classification via Gaussian HMM
HMM_REGIME_DETECTION_ENABLED=false
HMM_STATES=3
HMM_LOOKBACK=120
HMM_ITERATIONS=12
HMM_MIN_VARIANCE=0.00000001

STATE_DB_PATH=/opt/thumber-trader/grid_state.db
# Optional legacy mirror for old tooling (disabled by default)
LEGACY_ORDERS_JSON_ENABLED=false
# Optional: legacy order-state mirror path when enabled
# ORDERS_PATH=/opt/thumber-trader/orders.json
ENV
```

Lock it down:

```bash
sudo chmod 600 /opt/thumber-trader/.env
sudo chown gridbot:gridbot /opt/thumber-trader/.env
```

## 4) Run once manually (smoke test)

```bash
sudo -u gridbot bash -lc 'cd /opt/thumber-trader && set -a && source .env && set +a && .venv/bin/python grid_bot.py'
```

Watch logs for initial grid placement and no permission errors.

If dashboard is enabled, open `http://127.0.0.1:8080` (or your configured host/port) to monitor runtime status, open orders, recent events, and JSON status (`/api/status`).

The dashboard now exposes configuration on a dedicated page at `/config` (and a popup launcher from `/`) so runtime controls stay clean while edits happen in a focused view. You can apply all bot env variables live and persist them back to `.env` (or `BOT_ENV_PATH`) without using SSH editors like nano/vim.

### Telegram command-and-control (optional)

Install the Telegram SDK if you want bidirectional control:

```bash
/opt/thumber-trader/.venv/bin/pip install python-telegram-bot
```

Security: set `TELEGRAM_WHITELIST_CHAT_ID` to your personal chat ID. Commands from any other chat are rejected.

Supported commands:
- `/pause <PRODUCT_ID>`: pause trading loop actions for one engine without stopping the process.
- `/set_risk <0.0-1.0>`: update `MAX_BTC_INVENTORY_PCT` live across all running engines.
- `/report`: return per-engine realized PnL and 95% VaR snapshot.


Use `PAPER_TRADING_MODE=true` for a live-data dry run without sending exchange orders.



## Native backtesting (historical replay)

Use `backtest.py` to replay historical 1-minute candles through the existing grid logic in paper mode.
It swaps the live Coinbase REST client for a mock client backed by your CSV data.

Expected CSV header columns (case-insensitive):

- `timestamp` (or `ts` / `time`)
- `open`
- `high`
- `low`
- `close`
- `volume`

Run a single scenario:

```bash
python backtest.py \
  --csv data/btc_usd_1m.csv \
  --product-id BTC-USD \
  --grid-lines 8 \
  --lookback-minutes 43200
```

Compare multiple grid densities over the same window (for example 8 vs 10 lines over ~30 days):

```bash
python backtest.py \
  --csv data/btc_usd_1m.csv \
  --product-id BTC-USD \
  --compare-grid-lines 8,10 \
  --lookback-minutes 43200
```

The script prints JSON metrics per scenario, including ending equity, net PnL, fills, and residual open orders.

### Walk-Forward Optimization (WFO)

Use walk-forward mode to reduce overfitting by repeatedly training parameters on an in-sample window, then validating them on the immediately following out-of-sample window.

Default behavior in WFO mode:

- In-sample window: 7 days
- Out-of-sample window: 2 days
- Optimization targets: `ATR_BAND_MULTIPLIER`, `ADX_RANGE_BAND_MULTIPLIER`, `ADX_TREND_BAND_MULTIPLIER`

Example (30-day lookback, 7d train + 2d test folds):

```bash
python backtest.py \
  --csv data/btc_usd_1m.csv \
  --product-id BTC-USD \
  --grid-lines 8 \
  --lookback-minutes 43200 \
  --wfo-enabled \
  --wfo-in-sample-days 7 \
  --wfo-out-sample-days 2 \
  --wfo-atr-multipliers 3.0,4.0,5.0 \
  --wfo-adx-range-multipliers 0.7,0.8,0.9 \
  --wfo-adx-trend-multipliers 1.1,1.25,1.4
```

Output includes per-fold selected parameters, in-sample/out-of-sample metrics, plus summary robustness stats (fold win rate and aggregate out-of-sample PnL).

## Prometheus + Grafana observability

When dashboard is enabled, the bot also serves a Prometheus endpoint (default: `http://127.0.0.1:8080/metrics`).

Included metrics:

- `bot_realized_pnl_usd` (gauge): cumulative realized PnL in USD.
- `bot_inventory_ratio` (gauge): base-asset notional / total portfolio value.
- `bot_equity_curve_usd` (gauge): mark-to-market portfolio equity in USD.
- `bot_pnl_per_1k` (gauge): daily realized PnL normalized per $1k capital.
- `bot_portfolio_beta` (gauge): portfolio beta versus BTC benchmark returns.
- `api_latency_milliseconds` (histogram): latency distribution for Coinbase REST/public calls.
- `bot_api_safe_mode` (gauge): `1` when API circuit breaker Safe Mode is active and new entry (BUY) orders are paused.
- `bot_sentiment_safe_mode` (gauge): `1` when sentiment panic Safe Mode is active.
- `bot_sentiment_score_1h` (gauge): rolling one-hour sentiment score used by the override.

This makes it straightforward to build a Grafana dashboard that overlays strategy outputs (equity / PnL per $1k) with API health (latency).

## 5) Install as systemd service

Create `/etc/systemd/system/thumber-gridbot.service`:

```ini
[Unit]
Description=Thumber Trader Coinbase Adaptive Grid Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=gridbot
Group=gridbot
WorkingDirectory=/opt/thumber-trader
EnvironmentFile=/opt/thumber-trader/.env
ExecStart=/opt/thumber-trader/.venv/bin/python /opt/thumber-trader/grid_bot.py
Restart=always
RestartSec=10

# Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/thumber-trader

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now thumber-gridbot.service
```

Inspect status/logs:

```bash
systemctl status thumber-gridbot.service
journalctl -u thumber-gridbot.service -f
```

## Tuning guidelines (scaling up safely)

- Increase `BASE_ORDER_NOTIONAL_USD` gradually as balance grows.
- Keep `MIN_NOTIONAL_USD` at or above exchange/API minimum constraints.
- Keep `QUOTE_RESERVE_PCT` > 0 to avoid fully deploying cash.
- Lower `MAX_BTC_INVENTORY_PCT` if you want stricter anti-bag-holding behavior.
- Widen `GRID_BAND_PCT` for higher volatility, tighten for range-bound periods.
- If fees increase, raise `MIN_GRID_PROFIT_PCT` accordingly.

## Operational notes

- Trend bias is computed from Coinbase market data; sentiment-safe-mode integration is optional and disabled by default.
- No strategy can guarantee profit or fully eliminate drawdown risk.
- Start with small size, validate behavior in logs, then scale.

## Tax-time export

All fills are stored in SQLite (`fills` table) and can be exported to CSV for accounting/tax workflows.

Export all fills:

```bash
cd /opt/thumber-trader
set -a && source .env && set +a
.venv/bin/python grid_bot.py --export-tax-report fills_tax_report.csv
```

Export a single tax year:

```bash
cd /opt/thumber-trader
set -a && source .env && set +a
.venv/bin/python grid_bot.py --export-tax-report fills_tax_report_2025.csv --tax-year 2025
```

If dashboard is enabled, you can also download CSV directly:

- `http://127.0.0.1:8080/api/tax_report.csv`
- `http://127.0.0.1:8080/api/tax_report.csv?year=2025`


## Code layout (modularized)

- `grid_bot.py`: trading loop, exchange interactions, persistence, and HTTP routing.
- `dashboard_views.py`: dashboard and config-page HTML renderers (web UI layer).
- `config_schema.py`: shared env/config field schema used by runtime updates and UI rendering.

This keeps the project on a clearer path toward separate web/backend/trading concerns as it grows.
