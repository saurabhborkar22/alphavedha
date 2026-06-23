#!/usr/bin/env bash
# Run on the VPS from /opt/alphavedha/alphavedha/:
#   ./deploy/deploy.sh
set -euo pipefail

COMPOSE="docker compose -f docker-compose.vps.yml --env-file .env.vps"

echo "==> Pulling latest code"
git pull --ff-only origin main
export GIT_SHA=$(git rev-parse HEAD)
(cd ../alphavedha-ui && git pull --ff-only origin main)

echo "==> Rebuilding containers"
$COMPOSE build api scheduler ui

echo "==> Running DB migrations (before restart)"
$COMPOSE run --rm -T api alembic upgrade head

echo "==> Restarting services"
$COMPOSE up -d --force-recreate api scheduler
$COMPOSE restart nginx

echo "==> Checking service health"
$COMPOSE ps

echo ""
echo "==> Deploy complete. Access the UI at http://$(tailscale ip -4 2>/dev/null || echo '<tailscale-ip>')"
