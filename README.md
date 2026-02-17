# Thumber Trader v2.0

Thumber Trader is a high-frequency, adaptive grid trading bot designed for Coinbase Advanced Trade. Now evolved into a modular FastAPI application, it features a modern web dashboard, robust risk management, and advanced market analysis capabilities.

## 🚀 Key Features

- **Modular FastAPI Architecture**: High-performance backend with an HTMX-powered web dashboard.
- **Arithmetic Grid Engine**: Reliable grid trading with arithmetic spacing.
- **Basic Analysis**: Integrated RSI and EMA indicators for market context.
- **Tax-Ready Logging**: SQLite persistence with FIFO tax lot matching for reporting.

## 🛠 Prerequisites

- **Python 3.10+** (if running natively)
- **Coinbase Advanced Trade API Key** (with Trade and View permissions)
- **Docker & Docker Compose** (optional, for containerized deployment)

---

## 📦 Installation Options

### 1. Docker (Recommended for Proxmox/LXC/VPS)
The easiest way to get started.

```bash
# Clone the repository
git clone https://github.com/hikingthunder/thumber-trader.git
cd thumber-trader

# Create and edit your .env file (see Configuration section)
cp .env.example .env

# Start the application
docker-compose up -d
```

### 2. Debian/Ubuntu (Native Linux)
Perfect for Proxmox LXC containers or dedicated servers.

```bash
# Install system packages
sudo apt update && sudo apt install -y python3 python3-venv git curl

# Setup directory
sudo mkdir -p /opt/thumber-trader
sudo chown $USER:$USER /opt/thumber-trader
cd /opt/thumber-trader

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run the app (ensure .env is present)
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 3. Windows
Running natively on Windows.

```powershell
# In PowerShell
git clone https://github.com/hikingthunder/thumber-trader.git
cd thumber-trader

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

---

## ⚙️ Configuration

Copy the `.env` file and populate it with your credentials and preferences.

| Variable | Description | Default |
|----------|-------------|---------|
| `COINBASE_API_KEY` | Your Coinbase API Key | |
| `COINBASE_API_SECRET` | Your Coinbase API Secret | |
| `PRODUCT_ID` | Default trading pair (e.g., BTC-USD) | `BTC-USD` |
| `GRID_LINES` | Number of grid levels | `8` |
| `GRID_BAND_PCT` | Total width of the grid (±%) | `0.15` (15%) |
| `AUTO_START` | Automatically start the bot on launch| `True` |
| `PAPER_TRADING_MODE` | Simulate trades without real funds | `False` |

*Refer to `app/config.py` for a full list of over 100 configuration options.*

---

## 📊 Monitoring & Controls

Once running, access the dashboard at:
`http://localhost:8080` (or your server IP)

- **Main Dashboard**: Real-time status, open orders, and PnL metrics.
- **Config Editor**: Edit bot settings live without restarting.
- **Health Check**: Available at `/health`.

---

Basic analysis tools are located in `app/core/analysis.py`.

---

## ⚖️ Disclaimer

Trading cryptocurrencies involves significant risk. This bot is provided as-is without any guarantees of profit. Use `PAPER_TRADING_MODE=True` before risking real capital. Not financial advice.
