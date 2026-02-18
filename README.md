# ⚡ Thumber Trader v2.0: The Institutional-Grade Grid Bot

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Thumber Trader** is a high-frequency, ultra-adaptive grid trading engine built for the modern crypto market. Evolved from a simple script into a robust **FastAPI ecosystem**, it combines industrial-strength risk management with cutting-edge market analysis to keep you ahead of the spread.

> [!IMPORTANT]
> **Why Thumber Trader?** Most bots react to the market. Thumber Trader *anticipates* it using order flow toxicity (VPIN) and signal fusion.

---

## 🔥 Key Competitive Edge

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Adaptive Grid** | Arithmetic & Geometric spacing with ATR-adaptive bands. | Maximizes yield in both ranging and trending markets. |
| **Alpha Fusion** | Real-time RSI + MACD + Order Book Imbalance signals. | Filters noise and avoids "catching the falling knife." |
| **VPIN Engine** | Detects "Toxic Flow" before a breakout happens. | Automatically halts or widens bands to prevent stop-outs. |
| **Kelly Criterion** | Mathematical position sizing based on historical performance. | Optimizes capital growth while managing ruin risk. |
| **Multi-Venue Pricing**| Consolidation from Coinbase, Binance, Kraken, and Bybit. | Ensures accurate execution prices and prevents oracle manipulation. |
| **High Availability** | Active-Passive failover with database-backed election. | 99.9% uptime for your capital. |

---

## 🚀 Get Started in 5 Minutes

### 1. The Container Way (Recommended)
Perfect for Proxmox, VPS, or Docker-hardened environments.

```bash
git clone https://github.com/hikingthunder/thumber-trader.git
cd thumber-trader
cp .env.example .env
# Edit .env with your Coinbase Keys
docker-compose up -d
```

### 2. Native Linux (Debian/Ubuntu/LXC)
The choice for performance purists.

```bash
# Install dependencies
sudo apt update && sudo apt install -y python3-venv git curl
python3 -m venv .venv && source .venv/activate
pip install -r requirements.txt

# Run the engine
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 3. Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

---

## ⚙️ Advanced Configuration (No Files Needed!)

Forget digging through `.env` files. Thumber Trader features a **Premium Configuration Dashboard** where you can tune:

- **Product Management**: Trade BTC, ETH, SOL and more simultaneously.
- **Toxicity Controls**: Tune VPIN sensitivity and response modes.
- **Strategy Stacks**: Layer core grid logic with alpha and hedging layers.
- **Notifications**: Instant alerts via Telegram or Discord Webhooks.
- **Safe Start**: Execute "Base Buys" with VWAP algorithms to enter positions smoothly.

---

## 📈 Dashboard & Monitoring

Access your command center at `http://localhost:8080`:

- **Real-time Charting**: Visualize your grid levels against live price action.
- **PnL Tracking**: Comprehensive daily stats and realized/unrealized metrics.
- **Tax-Ready Exports**: One-click accounting exports (CSV, XLSX, ODS) with FIFO support.
- **Prometheus Metrics**: Ready-to-go `/metrics` endpoint for Grafana integration.

---

## ❓ FAQ

**Q: Can I use this for Paper Trading?**  
A: Yes! Enable `PAPER_TRADING_MODE=True` in the dashboard to test your strategies with zero risk.

**Q: Which API Keys do I need?**  
A: You need **Coinbase Advanced Trade** keys with `Trade` and `View` permissions.

**Q: How do I handle Proxmox LXC?**  
A: Use the Native Linux instructions. Thumber Trader is extremely lightweight and runs perfectly in a 1-core, 512MB RAM Debian LXC.

---

## ⚖️ Disclaimer

Trading cryptocurrencies involves significant risk. Thumber Trader is a tool, not a guarantee. **Always test in paper mode first.** Not financial advice.

---

*Made with ❤️ by the hikingthunder team.*
