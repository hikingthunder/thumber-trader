#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/thumber-trader"
ENV_DIR="/etc/thumber-trader"
ENV_FILE="$ENV_DIR/thumber-trader.env"
SERVICE_FILE="/etc/systemd/system/thumber-trader.service"
REPO_URL="https://github.com/hikingthunder/thumber-trader.git"
APP_USER="thumber"
APP_GROUP="thumber"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git curl ca-certificates python3 python3-venv python3-pip

if ! getent group "$APP_GROUP" >/dev/null 2>&1; then
  groupadd --system "$APP_GROUP"
fi

if ! id -u "$APP_USER" >/dev/null 2>&1; then
  useradd --system --gid "$APP_GROUP" --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
fi

if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone "$REPO_URL" "$APP_DIR"
else
  git -C "$APP_DIR" pull --ff-only
fi

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

mkdir -p "$ENV_DIR"
if [[ ! -f "$ENV_FILE" ]]; then
  cp "$APP_DIR/.env.example" "$ENV_FILE"
fi

chown -R "$APP_USER:$APP_GROUP" "$APP_DIR"
chown root:"$APP_GROUP" "$ENV_FILE"
chmod 640 "$ENV_FILE"

cat > "$SERVICE_FILE" <<UNIT
[Unit]
Description=Thumber Trader FastAPI Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_GROUP
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$APP_DIR/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now thumber-trader.service
systemctl --no-pager status thumber-trader.service || true

echo "Install complete. Edit $ENV_FILE, then: systemctl restart thumber-trader"
