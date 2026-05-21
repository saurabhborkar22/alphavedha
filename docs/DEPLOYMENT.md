# AlphaVedha — Deployment Guide

## Prerequisites

- Ubuntu 22.04+ or Debian 12+
- Docker + Docker Compose v2
- 4GB+ RAM, 2 CPU cores minimum
- Domain name (for SSL) or use IP directly

## Quick Start (Docker)

```bash
# 1. Clone and configure
git clone https://github.com/saurabhborkar22/alphavedha.git
cd alphavedha
cp .env.prod.example .env.prod

# 2. Edit .env.prod with real values
# IMPORTANT: Change ALL placeholder passwords
nano .env.prod

# 3. Validate configuration
python scripts/validate_env.py

# 4. Start services
docker compose -f docker-compose.prod.yml up -d

# 5. Verify
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://user:pass@postgres:5432/alphavedha` |
| `REDIS_URL` | Redis connection string | `redis://:password@redis:6379/0` |
| `ALPHAVEDHA_API_KEY` | Primary API key (32+ chars) | `av_prod_a1b2c3d4...` |
| `POSTGRES_USER` | PostgreSQL user | `alphavedha` |
| `POSTGRES_PASSWORD` | PostgreSQL password | (strong random) |
| `POSTGRES_DB` | Database name | `alphavedha` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPHAVEDHA_API_KEY_SECONDARY` | (empty) | Rollover key for zero-downtime rotation |
| `ALPHAVEDHA_ENV` | `production` | Environment name |
| `ALPHAVEDHA_CORS_ORIGINS` | (empty) | Comma-separated allowed origins |
| `DB_POOL_SIZE` | `10` | SQLAlchemy pool size |
| `DB_MAX_OVERFLOW` | `20` | Max overflow connections |
| `DB_POOL_TIMEOUT` | `30` | Pool checkout timeout (seconds) |
| `DB_POOL_RECYCLE` | `1800` | Connection recycle interval (seconds) |
| `REDIS_PASSWORD` | (empty) | Redis auth password |
| `FINNHUB_API_KEY` | (empty) | For news sentiment |
| `ALERT_EMAIL_ENABLED` | `false` | Enable email alerts |
| `ALERT_SMTP_HOST` | `smtp.gmail.com` | SMTP server |
| `ALERT_SMTP_PORT` | `587` | SMTP port |

## Architecture

```
[nginx :443] → [FastAPI :8000] → [PostgreSQL :5432]
                    ↓                    ↑
              [Redis :6379]        [TimescaleDB]
                    ↓
            [Scheduler (cron)]
```

### Services

| Service | Container | Purpose |
|---------|-----------|---------|
| `app` | alphavedha-app | FastAPI API server |
| `scheduler` | alphavedha-scheduler | Background jobs (predictions, drift, retraining) |
| `postgres` | alphavedha-db | TimescaleDB (PostgreSQL 16 + time-series) |
| `redis` | alphavedha-redis | Feature cache, rate limiting |

## VPS Deployment (systemd)

For bare-metal or VPS without Docker:

```bash
# 1. Run setup script (installs Python, PostgreSQL, Redis, nginx, certbot)
chmod +x deploy/setup.sh
sudo ./deploy/setup.sh

# 2. Install application
cd /opt/alphavedha
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Configure environment
cp .env.prod.example .env.prod
nano .env.prod

# 4. Run database migrations
alembic upgrade head

# 5. Install systemd services
sudo cp deploy/alphavedha-api.service /etc/systemd/system/
sudo cp deploy/alphavedha-scheduler.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now alphavedha-api alphavedha-scheduler

# 6. Configure nginx
sudo cp deploy/nginx.conf /etc/nginx/sites-available/alphavedha
sudo ln -s /etc/nginx/sites-available/alphavedha /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 7. SSL certificate
sudo certbot --nginx -d your-domain.com
```

## Database

### Migrations

```bash
# Apply pending migrations
alembic upgrade head

# Check migration status
python scripts/check_migrations.py

# Create new migration after model changes
alembic revision --autogenerate -m "description"
```

### Backups

```bash
# Manual backup
./scripts/backup_db.sh ./backups

# Automated daily backup (add to crontab)
# crontab -e
0 2 * * * /opt/alphavedha/scripts/backup_db.sh /opt/alphavedha/backups

# Restore from backup
./scripts/restore_db.sh backups/alphavedha_20260521_020000.sql.gz
```

### Connection Pool Tuning

For production workloads, adjust via environment variables:

| Workload | DB_POOL_SIZE | DB_MAX_OVERFLOW | Notes |
|----------|-------------|-----------------|-------|
| Light (personal) | 5 | 10 | Default is fine |
| Medium (shared) | 10 | 20 | Current default |
| Heavy (many users) | 20 | 40 | Monitor with `pg_stat_activity` |

## API Key Rotation

Zero-downtime key rotation:

```bash
# 1. Generate new key
NEW_KEY=$(openssl rand -hex 32)

# 2. Set as secondary (both keys now work)
echo "ALPHAVEDHA_API_KEY_SECONDARY=$NEW_KEY" >> .env.prod
docker compose -f docker-compose.prod.yml restart app

# 3. Update all clients to use new key

# 4. Promote new key to primary, remove old
# In .env.prod:
#   ALPHAVEDHA_API_KEY=$NEW_KEY
#   ALPHAVEDHA_API_KEY_SECONDARY=
docker compose -f docker-compose.prod.yml restart app
```

## Monitoring

### Health Checks

```bash
# Liveness (always works)
curl http://localhost:8000/health

# Readiness (checks DB + Redis + models)
curl http://localhost:8000/ready

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Logs

```bash
# Docker logs
docker compose -f docker-compose.prod.yml logs -f app

# systemd logs
journalctl -u alphavedha-api -f

# Application logs (if file logging enabled)
tail -f /opt/alphavedha/logs/alphavedha.log
tail -f /opt/alphavedha/logs/alphavedha-error.log
```

## Troubleshooting

| Symptom | Check | Fix |
|---------|-------|-----|
| 503 on all endpoints | `curl /ready` | Models not loaded — check model artifacts |
| DB connection refused | `docker compose ps postgres` | Restart postgres, check credentials |
| Redis timeout | `redis-cli ping` | Restart redis, check REDIS_URL |
| High latency | `/metrics` (histogram) | Check DB pool exhaustion, add caching |
| Migration errors | `python scripts/check_migrations.py` | `alembic upgrade head` |
| Placeholder passwords | `python scripts/validate_env.py` | Update .env.prod with real values |
