#!/usr/bin/env bash
# Run on the VPS from /opt/alphavedha/alphavedha/:
#   ./deploy/deploy.sh
set -euo pipefail

COMPOSE="docker compose -f docker-compose.vps.yml"

echo "==> Pulling latest code"
git pull origin main
cd ../alphavedha-ui && git pull origin main && cd ../alphavedha

echo "==> Rebuilding containers"
$COMPOSE build --no-cache api scheduler ui

echo "==> Restarting services"
$COMPOSE up -d

echo "==> Running DB migrations"
$COMPOSE exec api alembic upgrade head

echo "==> Checking service health"
sleep 5
$COMPOSE ps

echo ""
echo "==> Deploy complete. Access the UI at http://$(tailscale ip -4 2>/dev/null || echo '<tailscale-ip>')"
