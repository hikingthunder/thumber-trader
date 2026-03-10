#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/thumber-trader"
ENV_FILE="/etc/thumber-trader/thumber-trader.env"
SERVICE_NAME="thumber-trader.service"
APP_USER="thumber"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "This updater must be run as root." >&2
  exit 1
fi

if [[ ! -d "$APP_DIR/.git" ]]; then
  echo "Repository not found at $APP_DIR. Run scripts/install_lxc.sh first." >&2
  exit 1
fi

if [[ ! -x "$APP_DIR/.venv/bin/pip" ]]; then
  echo "Virtualenv missing at $APP_DIR/.venv. Run scripts/install_lxc.sh first." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file missing at $ENV_FILE. Run scripts/install_lxc.sh first." >&2
  exit 1
fi

echo "[1/5] Updating repository..."
sudo -u "$APP_USER" git -C "$APP_DIR" fetch --all --prune
sudo -u "$APP_USER" git -C "$APP_DIR" pull --ff-only

echo "[2/5] Updating Python dependencies..."
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "[3/5] Reloading systemd unit files..."
systemctl daemon-reload

echo "[4/5] Restarting $SERVICE_NAME..."
systemctl restart "$SERVICE_NAME"

echo "[5/5] Current service status:"
systemctl --no-pager --full status "$SERVICE_NAME" || true

echo "Update complete. If settings changed, review $ENV_FILE and restart again if needed."
