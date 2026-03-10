# Thumber Trader v2.0

Institutional-style grid trading bot with FastAPI dashboard, strategy controls, risk guardrails, and paper/live execution modes.

## What this project includes

- Adaptive grid engine (`arithmetic` or `geometric` spacing)
- Execution modes: Live, Paper, and Shadow Live (simulated execution on live market data)
- Alpha fusion controls (RSI, MACD, imbalance weighting)
- VPIN toxicity controls and risk guardrails
- Web dashboard with:
  - auth (register/login/session) + audit log filtering
  - runtime stats, fills widgets, and multi-style live market charts (line/area, candlestick, OHLC, Heikin Ashi + additional chart-mode fallbacks)
  - config editor that writes to `.env` with encrypted version history + rollback and inline recommended defaults
  - account & security page for username/password updates and 2FA management
  - backtest page
  - export page (CSV/XLSX/ODS)
- Notifications (Telegram/Discord/Slack/PagerDuty)
- TradingView webhook endpoint for pause/resume/kill actions
- Prometheus-style metrics endpoint (`/metrics`)
- Docker + Compose + Podman workflows

---

## Quick start

### 1) Clone and configure

```bash
git clone https://github.com/hikingthunder/thumber-trader.git
cd thumber-trader
cp .env.example .env
# edit .env with your keys/secrets before live trading
```

### 2) Run with Docker Compose

```bash
docker compose up -d --build
```

Open: `http://localhost:8080`

### 3) Run with Podman Compose

```bash
podman compose up -d --build
```

If your host uses rootless Podman with SELinux, add relabel flags to bind mounts (`:Z`) or use named volumes.

---

## Installation matrix (major OS / platforms)

### Debian / Ubuntu / Proxmox LXC (recommended)

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Fedora / RHEL / Rocky / AlmaLinux

```bash
sudo dnf install -y git python3 python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Arch Linux / Manjaro

```bash
sudo pacman -Sy --needed git python python-pip
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### openSUSE

```bash
sudo zypper install -y git python311 python311-pip
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### macOS (Homebrew)

```bash
brew install git python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

### Windows 10/11 (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

### WSL2

Use the Debian/Ubuntu instructions inside your WSL distro.

### Virtualization targets

- **Proxmox VE VM**: any Linux section above applies
- **Proxmox LXC (Debian)**: use helper script below
- **VMware / VirtualBox / KVM**: use distro-native section

---

## Proxmox Debian LXC one-line installer

From a fresh Debian LXC shell (as root):

```bash
bash -lc "$(curl -fsSL https://raw.githubusercontent.com/hikingthunder/thumber-trader/main/scripts/install_lxc.sh)"
```

What it does:
- installs Python + Git + venv tooling
- clones/updates repo to `/opt/thumber-trader`
- creates virtualenv and installs requirements
- creates `/etc/thumber-trader/thumber-trader.env` if missing
- installs and enables `thumber-trader.service` (runs as dedicated non-root `thumber` user)

---

## Container usage

### Docker

```bash
docker compose up -d --build
docker compose logs -f app
```

### Podman

```bash
podman compose up -d --build
podman compose logs -f app
```

### Notes

- Default app port: `8080`
- Environment loaded from `.env`
- Persistent runtime state in `data/` and sqlite DB files

---

## Security hygiene checklist

- `.env` is gitignored
- DB and runtime dirs are gitignored
- added ignores for private keys/certs and secret artifacts
- keep API keys only in `.env` or external secret manager
- leave `JWT_SECRET_KEY` empty for ephemeral sessions, or set a long random value for persistent logins
- rotate credentials after any accidental exposure

---

## Core endpoints

- `GET /health` – app + manager state
- `GET /metrics` – Prometheus text metrics
- `GET /dashboard` – primary UI
- `GET/POST /config` – runtime config editor
- `GET/POST /backtest` – backtesting UI/actions
- `POST /webhook/tradingview` – TradingView actions (`pause|resume|adjust_band|kill`)
- `WS /ws/dashboard` – realtime dashboard channel

---


## Release enhancement roadmap

Paper trading is stable enough to begin the next hardening wave. The staged plan for shadow-live validation, config rollback, portfolio risk budgets, walk-forward robustness, incident replay, and pre-live readiness gates is documented in `docs/release_enhancement_plan.md`, including owner/status/milestone tracking, implementation checklists, migration notes, and validation/security sign-off criteria.

## Updating

### Git-based deployment

```bash
git pull --ff-only
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart thumber-trader
```

If you deployed with `scripts/install_lxc.sh`, update as:

```bash
sudo -u thumber git -C /opt/thumber-trader pull --ff-only
sudo -u thumber /opt/thumber-trader/.venv/bin/pip install -r /opt/thumber-trader/requirements.txt
sudo systemctl restart thumber-trader
```

### Docker / Podman deployment

```bash
git pull --ff-only
docker compose up -d --build
# or
podman compose up -d --build
```

---

## Development roles

Role ownership and delegation guidance lives in [`AGENTS.md`](AGENTS.md).
Templates for other AI-agent runtimes live in [`docs/agents/templates/`](docs/agents/templates/).

---

## Disclaimer

Use Thumber Trader at your own risk. This software is provided for operational tooling and research workflows only and is **not** financial, investment, legal, or tax advice.

Crypto trading is high risk. Start in paper mode, validate behavior in your environment, and never deploy live capital you cannot afford to lose.
