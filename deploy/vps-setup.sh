#!/usr/bin/env bash
# Run as root on a fresh Ubuntu 24.04 VPS:
#   curl -fsSL https://raw.githubusercontent.com/YOUR_GITHUB/alphavedha/main/deploy/vps-setup.sh | bash
set -euo pipefail

APP_DIR="/opt/alphavedha"
GITHUB_USER="${GITHUB_USER:-}"

echo "==> [1/5] Installing Docker"
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable --now docker

echo "==> [2/5] Installing Tailscale"
curl -fsSL https://tailscale.com/install.sh | sh
echo ""
echo "    Run this now to connect to your tailnet:"
echo "    tailscale up"
echo "    Then note the 100.x.x.x IP shown."
echo ""

echo "==> [3/5] Creating app directory"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

echo "==> [4/5] Cloning repos"
if [ -n "$GITHUB_USER" ]; then
  git clone "https://github.com/$GITHUB_USER/alphavedha.git" alphavedha
  git clone "https://github.com/$GITHUB_USER/alphavedha-ui.git" alphavedha-ui
else
  echo "    GITHUB_USER not set — clone repos manually into $APP_DIR/alphavedha and $APP_DIR/alphavedha-ui"
fi

echo "==> [5/5] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. tailscale up    (note your 100.x.x.x IP)"
echo "  2. cd $APP_DIR/alphavedha"
echo "  3. cp .env.vps.example .env.vps"
echo "  4. nano .env.vps   (fill in passwords)"
echo "  5. docker compose -f docker-compose.vps.yml up -d --build"
echo "  6. docker compose -f docker-compose.vps.yml exec api alembic upgrade head"
echo "  7. Open http://<tailscale-ip> in your browser"
