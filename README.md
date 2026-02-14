# Coinbase Adaptive Grid Bot (BTC-USD default)

`grid_bot.py` is a Python bot for Coinbase Advanced Trade using `coinbase-advanced-py`.

This README is written for a **Debian 13 (Trixie) Proxmox LXC service deployment** with practical, copy/paste steps.

## What this bot does

- Uses a neutral grid anchor at startup.
- Creates a default ±15% band around the startup price.
- Uses 8 grid lines by default and enforces a minimum grid spacing floor.
- Places **post-only** limit orders (maker intent).
- Saves open order state to `orders.json` so restart recovery works.
- Polls every 60 seconds by default, detects fills, and places replacement orders.
- Applies risk controls (USD reserve, BTC inventory cap, stop-loss awareness).
- Adds trend bias (EMA fast/slow over Coinbase candles) to be more defensive in downtrends.

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
GRID_LINES=8
GRID_BAND_PCT=0.15
MIN_NOTIONAL_USD=6
MIN_GRID_PROFIT_PCT=0.015
POLL_SECONDS=60

# Optional local dashboard
DASHBOARD_ENABLED=true
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=8080

# Safer dry-run mode (simulated fills + balances)
PAPER_TRADING_MODE=false
PAPER_START_USD=1000
PAPER_START_BTC=0

BASE_ORDER_NOTIONAL_USD=10
QUOTE_RESERVE_PCT=0.25
MAX_BTC_INVENTORY_PCT=0.65
HARD_STOP_LOSS_PCT=0.08

TREND_GRANULARITY=ONE_HOUR
TREND_CANDLE_LIMIT=72
TREND_EMA_FAST=9
TREND_EMA_SLOW=21
TREND_STRENGTH_THRESHOLD=0.003

# Optional: override order state file path
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

Use `PAPER_TRADING_MODE=true` for a live-data dry run without sending exchange orders.

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

- This bot uses only Coinbase data for trend bias; no third-party sentiment feed is required.
- No strategy can guarantee profit or fully eliminate drawdown risk.
- Start with small size, validate behavior in logs, then scale.
