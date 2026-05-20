#!/usr/bin/env bash
set -euo pipefail

# AlphaVedha VPS deployment setup
# Run as root on a fresh Ubuntu 22.04+ server

INSTALL_DIR="/opt/alphavedha"
LOG_DIR="/var/log/alphavedha"
DOMAIN="${1:-}"

if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 <domain>"
    echo "Example: $0 alphavedha.example.com"
    exit 1
fi

echo "==> Installing system packages"
apt-get update
apt-get install -y python3.12 python3.12-venv python3.12-dev \
    postgresql-client libpq-dev nginx certbot python3-certbot-nginx \
    git curl

echo "==> Creating alphavedha user"
id -u alphavedha &>/dev/null || useradd -r -m -d "$INSTALL_DIR" -s /bin/bash alphavedha

echo "==> Setting up directories"
mkdir -p "$INSTALL_DIR" "$LOG_DIR" "$INSTALL_DIR/models/artifacts"
chown -R alphavedha:alphavedha "$INSTALL_DIR" "$LOG_DIR"

echo "==> Setting up Python environment"
sudo -u alphavedha bash -c "
    cd $INSTALL_DIR
    python3.12 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -e .
"

echo "==> Installing systemd services"
cp "$INSTALL_DIR/deploy/alphavedha-api.service" /etc/systemd/system/
cp "$INSTALL_DIR/deploy/alphavedha-scheduler.service" /etc/systemd/system/
systemctl daemon-reload

echo "==> Configuring nginx"
DOMAIN="$DOMAIN" envsubst '${DOMAIN}' < "$INSTALL_DIR/deploy/nginx.conf" \
    > /etc/nginx/sites-available/alphavedha
ln -sf /etc/nginx/sites-available/alphavedha /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

echo "==> Obtaining SSL certificate"
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email "${EMAIL:-admin@$DOMAIN}"

echo "==> Enabling services"
systemctl enable alphavedha-api alphavedha-scheduler
nginx -t && systemctl reload nginx

echo ""
echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy .env.prod.example to $INSTALL_DIR/.env.prod and fill in values"
echo "  2. Start PostgreSQL + Redis (docker compose or system packages)"
echo "  3. Run migrations: cd $INSTALL_DIR && .venv/bin/alembic upgrade head"
echo "  4. Start services:"
echo "     systemctl start alphavedha-api"
echo "     systemctl start alphavedha-scheduler"
echo "  5. Check status: systemctl status alphavedha-api"
