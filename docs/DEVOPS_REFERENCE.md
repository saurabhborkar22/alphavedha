# AlphaVedha — DevOps & Infrastructure Reference

## Infrastructure Overview

| Component | Spec | Cost |
|---|---|---|
| VPS (serving) | Hetzner CX23 — 2 vCPU, 4 GB RAM | €3.99/mo |
| VPS (training) | Hetzner CX43 — 8 vCPU, 16 GB RAM | On-demand only |
| Scaling time | poweroff → change_type → poweron | ~5 min downtime |
| VPS path | `/opt/alphavedha/` | — |
| Admin access | Tailscale overlay (100.x.x.x) | — |

Scale-up cost for a 6-hour training run: ~€0.12. Scale-down is guaranteed by `if: always()` in GitHub Actions.

---

## Docker Services

### docker-compose.yml (local dev — infrastructure only)

| Service | Image | Ports |
|---|---|---|
| postgres | timescale/timescaledb:latest-pg16 | 5435:5432 |
| redis | redis:7-alpine | 6379:6379 |

No app containers — dev runs Python venv locally against these.

---

### docker-compose.vps.yml (VPS production — full stack)

| Service | Build | Ports/Expose | Key Env Vars | Health Check |
|---|---|---|---|---|
| nginx | nginx:alpine | 80:80 | — | disabled |
| api | Dockerfile (INSTALL_TARGET=vps) | expose 8000 | DATABASE_URL, REDIS_URL, ALPHAVEDHA_API_KEY, ALPHAVEDHA_DEMO | curl /health every 30s, start_period 15s |
| scheduler | Dockerfile (INSTALL_TARGET=vps) | — | TZ=Asia/Kolkata | disabled |
| trainer | Dockerfile (INSTALL_TARGET=vps) | — | ALPHAVEDHA_HEAVY_TRAINING=1 | none; restart: "no" — profile: training |
| ui | ../alphavedha-ui/Dockerfile | expose 3000 | NEXT_PUBLIC_API_URL=/api | node http check every 30s, start_period 20s |
| postgres | timescale/timescaledb:latest-pg16 | expose 5432 (internal only) | POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB | pg_isready every 10s |
| redis | redis:7-alpine | expose 6379 (internal only) | REDIS_PASSWORD | redis-cli -a ping every 10s |

**Shared volumes:**
- `model-artifacts` — shared between api, scheduler, trainer (trained models reach serving container this way)
- `app-logs` — shared between api, scheduler
- `pgdata` — PostgreSQL data
- `redisdata` — Redis data

**Postgres and Redis** not exposed publicly — internal Docker network only.
**Trainer** only launched on-demand (profile: training). Started manually or via GitHub Actions.

---

### docker-compose.prod.yml (generic production — no UI)

Same as VPS but without nginx and ui services. Postgres/Redis ports bound to 127.0.0.1 (not public).

---

## Dockerfiles

### Backend Dockerfile (two-stage)

**Stage 1 (builder: python:3.12-slim):**
- Installs gcc, libpq-dev
- Copies source
- `ARG INSTALL_TARGET=vps` controls dependencies:
  - `vps`: torch CPU-only from pytorch.org/whl/cpu + requirements-vps.txt
  - `full`: torch + torchvision CPU + full pyproject.toml

**Stage 2 (runtime: python:3.12-slim):**
- libpq5 + curl only
- Creates `alphavedha` group/user (runs as non-root)
- Copies /install + source
- Creates `/app/models/artifacts` and `/app/logs`
- `ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app`
- `EXPOSE 8000`
- `HEALTHCHECK: curl /health every 30s`
- `CMD: uvicorn alphavedha.api.app:app --host 0.0.0.0 --port 8000`

### UI Dockerfile (two-stage)

**Stage 1 (builder: node:20-slim):**
- npm ci
- `NODE_OPTIONS=--max_old_space_size=2048` (prevents OOM on VPS)
- `ARG NEXT_PUBLIC_API_URL=/api` — baked into client bundle
- `npm run build` → `output: "standalone"`

**Stage 2 (runner: node:20-slim):**
- Creates `nextjs` user (uid 1001, non-root)
- Copies `.next/standalone/`, `.next/static/`, `public/`
- `EXPOSE 3000`
- `CMD: node server.js`

---

## nginx Configuration (deploy/nginx-vps.conf)

```
/          → proxy_pass http://ui:3000    (Next.js UI)
/api/      → proxy_pass http://api:8000  (FastAPI, strips /api prefix)
/api/ws/   → proxy_pass http://api:8000/ws/  (WebSocket upgrade)
           → proxy_read_timeout 300s     (scan endpoint headroom)
           → WebSocket upgrade map (connection_upgrade)
```

Rate limiting (nginx.conf for HTTPS): zone `api_predict` 5 r/s, burst 10 → 429 on exceed.

---

## CI/CD Pipelines

### ci.yml — Main CI

**Triggers:** push to main, PR targeting main. Concurrency: cancel-in-progress per branch.

| Job | When | Steps |
|---|---|---|
| lint | every trigger | ruff check + ruff format check + mypy (continue-on-error) |
| test | PR only | pytest tests/unit/ with coverage (fail-under=0 — tests must pass, no min coverage) |
| security | every trigger | pip-audit on compiled lockfile (ignores CVE-2025-3000: torch local-only issue) |
| deploy | push to main only (needs lint + security) | SSH to VPS: git checkout → docker build api/scheduler/trainer → docker up → alembic upgrade head |

**Deploy steps on VPS:**
```bash
git fetch && git checkout $GITHUB_SHA
docker compose build api scheduler trainer
docker compose up -d --force-recreate api scheduler
docker compose restart nginx
alembic upgrade head
```
Trainer is built but NOT started — available for on-demand training.

---

### train.yml — Train Models on VPS

**Triggers:** cron `0 17 * * 6` (Saturday 22:30 IST = 17:00 UTC); `workflow_dispatch` with model choice + scale_server option.

**Concurrency group:** `train-vps` (serialized with sim.yml — only one training run at a time).

**Scale-up sequence:**
```
poweroff → wait 60s → change_type to cx43 → wait 15s → poweron → wait 90s
```

**Training command (timeout 8h):**
```bash
docker compose --profile training run --rm \
  -e ALPHAVEDHA_HEAVY_TRAINING=1 \
  trainer python -m alphavedha.cli.main train "$TRAIN_MODEL" --tier large
```

**Scale-down:** `if: always()` — guaranteed even on failure/cancellation. CX43→CX23 (poweroff → change_type → poweron).

---

### sim.yml — Historical Simulation

**Triggers:** `workflow_dispatch` only. Inputs: cutoff, end, scale_server, deploy_ui, regime_overlay.

**Concurrency group:** `train-vps` (serialized with train.yml).

**Steps:** Scale up → SSH → run `sim_paper_trading.py` → build sim_artifact.json → optionally rebuild UI → verify via curl → scale down (guaranteed).

---

## Monitoring System

### Drift Detection (monitoring/drift.py)

Algorithm: PSI (Population Stability Index) + KS test

**PSI formula:** `sum((current_pct - ref_pct) * ln(current_pct / ref_pct))` — 10 bins, epsilon guard 1e-6

| PSI Value | Action |
|---|---|
| < 0.1 | Normal |
| 0.1 – 0.2 | WARNING — logged |
| > 0.2 | ALERT — email + `requires_retrain=True` |

All numeric features checked. KS test (scipy.stats.ks_2samp) also computed per feature.
Prometheus gauge: `alphavedha_drift_psi{feature_group}` updated per group.

---

### Performance Tracking (monitoring/performance.py)

Rolling windows: 7d, 30d, 90d.

Per-window metrics:
- accuracy: fraction where predicted_direction == actual_direction
- precision_buy, precision_sell: accuracy within each direction subset
- magnitude_mae: mean absolute error of predicted vs actual return magnitude
- profitable_pct: fraction where direction×return > 0

`requires_retrain = True` if any window accuracy < 0.52 (52% threshold).

---

### Retraining Manager (monitoring/retrainer.py)

**Trigger priority (evaluated in order):**
1. `drift_report.requires_retrain` (PSI > 0.2) → reason: "drift_detected"
2. `performance_report.requires_retrain` (accuracy < 52%) → reason: "performance_degraded"
3. `days_since_last_train >= schedule_interval` → reason: "scheduled" (weekly=7, monthly=30)

**Version lifecycle:** shadow → active → retired

Promotion logic (`compare_models()`):
- acc_delta >= 0.01 AND f1_delta >= 0.01 → "promote"
- either delta <= -0.02 → "discard"
- otherwise → "extend_shadow"

Cleanup: keeps active + N most recent non-active (default 5). Uses `shutil.rmtree` on extras.

---

### Alerts (monitoring/alerts.py)

Channel: Email via SMTP (Gmail or configurable). Env vars: `ALERT_SMTP_HOST`, `ALERT_SMTP_PORT=587`, `ALERT_SMTP_SENDER`, `ALERT_SMTP_PASSWORD`, `ALERT_SMTP_RECIPIENT`, `ALERT_EMAIL_ENABLED=1`.

| Alert Method | Level | When |
|---|---|---|
| scheduler_job_failed(job, error) | CRITICAL | Any scheduler job exception |
| drift_detected(feature_group, psi) | WARNING | PSI > psi_alert (0.2) |
| accuracy_drop(window, accuracy, threshold) | WARNING | Accuracy < 52% in any rolling window |
| api_error_spike(count, window_minutes) | CRITICAL | 5xx count spike |
| data_quality_failed(report) | CRITICAL | Any critical data quality failures |

Subject format: `[AlphaVedha {LEVEL}] {subject}`. Uses STARTTLS (port 587).

---

### Experiment Tracker (monitoring/experiment_tracker.py)

Storage: JSON files in `models/artifacts/runs/{run_id}.json`
Run ID format: `{YYYYMMDD}_{HHMMSS}_{microseconds}_{model_name}`

Per-run fields: run_id, model_name, started_at, duration_seconds, hyperparams, train_metrics, val_metrics, data_range, n_train_rows, n_val_rows, n_symbols, feature_count, artifact_path.

Operations: `log_run()`, `list_runs(model_name, limit=20)`, `get_run(run_id)`, `compare_runs(a, b)` → delta dict.

---

### Track Record (monitoring/track_record.py)

Three tracks computed:

| Track | Selection Criteria |
|---|---|
| all | Every prediction with direction != 0 and actual_return not null |
| gate_passed | is_tradeable=True (persisted flag); falls back to confidence >= 0.50 for legacy rows |
| top_k | Top-5 highest-confidence directional predictions per prediction_date |

Cohort model: equal-weight 15-trading-day bet per prediction date. Net return = gross return minus round_trip_cost_pct.

Annualization factor: `sqrt(252/15)` — 15-day cohort horizon.

Per-track stats: n_selected, n_evaluated, win_rate_net, avg_return_gross, avg_return_net, total_return_net, profit_factor_net, sharpe_net, max_drawdown_net.

---

## Risk Management

### Position Sizing — Half Kelly (risk/position_sizing.py)

```python
q = 1 - p                          # P(incorrect)
b = magnitude / magnitude_loss_ref  # win/loss ratio (magnitude_loss_ref = 2%)
kelly_fraction = p - q / b          # generalized Kelly
position_pct = min(kelly_fraction * 0.5 * 100, max_single_stock_pct)
```

Returns 0 if: confidence < 0.55, magnitude <= 0, or kelly_fraction <= 0.

---

### Circuit Breaker (risk/circuit_breaker.py)

Evaluated on current portfolio drawdown.

| Level | Drawdown | Action |
|---|---|---|
| 0 | < 10% | Normal |
| 1 | ≥ 10% | Halve all proposed positions (× 0.5) |
| 2 | ≥ 15% | Block new entries; halve positions |
| 3 | ≥ 20% | All positions → 0 |

Recovery: `current_value >= peak_value * 0.95` → reset to level 0.

---

### Portfolio Constraints (risk/portfolio.py)

Checks in order (each can block or reduce position):

1. **Holding period** (sells): min_holding_days=3. Violation → block.
2. **Liquidity** (buys): avg_daily_turnover >= 5 crore. Violation → block.
3. **Correlation** (buys): abs(corr_60d) > 0.7 with any existing holding → block.
4. **Sector cap** (buys): sector_weight + proposed <= 25%. Excess → reduce proposed (may reach 0).

---

### Market Impact Model (risk/impact_model.py)

Almgren-Chriss inspired. Per cap tier coefficients:

| Tier | eta | gamma | bid-ask spread |
|---|---|---|---|
| large | 0.05 | 0.01 | 0.05% |
| mid | 0.15 | 0.03 | 0.10% |
| small | 0.40 | 0.08 | 0.20% |

```
participation_rate = order_size_shares / avg_daily_volume
temporary_impact = eta × sigma × (participation_rate ^ 0.6)
permanent_impact = gamma × sigma × participation_rate
execution_cost = temporary + permanent + bid_ask_cost
```

Feasibility: participation_rate < 20% ADV.

Execution recommendations:
- < 5%: "execute normally"
- 5–10%: "split into tranches"
- 10–20%: "use VWAP/TWAP over 2+ hours"
- > 20%: "order too large for daily volume"

---

## Signals Module (alphavedha/signals/)

### ExecutionEngine (signals/execution.py)
- Generates optimal trade execution plans for Indian equities
- Optimal windows: 10:30–11:30, 14:00–14:45 IST
- Avoid: 9:15–9:30 (opening noise), 15:20–15:30 (closing manipulation risk)
- Order type: market (large), limit (mid), vwap (small)
- `is_expiry_day()` — last Thursday of month (monthly F&O expiry)
- `is_weekly_expiry()` — every Thursday (Nifty/BankNifty weekly expiry)

### PairsTrader (signals/pairs.py)
- Market-neutral spread trading via OLS cointegration
- Entry at z-score > 2.0 or < -2.0; exit at |z| < 0.5; stop at |z| > 3.5
- `backtest_pair()` — full trade-level P&L with Sharpe and max drawdown

### Pre-defined Sector Pairs (signals/pairs_universe.py)
10 pairs: HDFCBANK/ICICIBANK, TCS/INFY, RELIANCE/ONGC, MARUTI/M&M, SUNPHARMA/DRREDDY, HINDALCO/TATASTEEL, BAJFINANCE/BAJAJFINSV, NTPC/POWERGRID, HDFCBANK/KOTAKBANK, BHARTIARTL/TECHM.

Validation requirements: Engle-Granger cointegration p < 0.05, correlation > 0.6, half-life < 60 days, ≥ 120 common data points.

---

## Sentiment Module (alphavedha/sentiment/)

**SentimentAggregator** — `aggregate(symbol, lookback_days=7)`:
- Fetches from RSS + Reddit concurrently (asyncio.gather)
- FinBERT scoring: positive - negative prob per article; overall = mean net score
- Momentum: recent_half_mean - earlier_half_mean
- Data quality: min(1.0, post_count / 10)

**RSS sources:** Moneycontrol buzzingstocks RSS, ET Markets RSS, Business Standard markets RSS (3 feeds, concurrent fetch)

**Reddit source:** r/IndiaInvestments, r/niftyoptions, r/DalalStreetTalks, r/IndianStockMarket (PRAW, 25 posts each). Requires `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET`. Degrades to empty list if unavailable.

---

## Sectors Module (alphavedha/sectors/)

**compute_sector_rotation()** — RRG (Relative Rotation Graph) analysis:
- 12 NSE sector indices vs Nifty 50 benchmark
- RS-Ratio: (sector/benchmark) × 100, smoothed 10-day rolling mean, normalized to baseline=100
- RS-Momentum: 1-day ROC of RS-Ratio, smoothed 5-day, normalized to 100=flat
- Phase classification: Leading (RS-R≥100 AND RS-M≥100), Weakening (RS-R≥100 AND RS-M<100), Lagging (RS-R<100 AND RS-M<100), Improving (RS-R<100 AND RS-M≥100)
- Also computes 1m and 3m absolute + relative returns

---

## Logging (monitoring/logging.py)

**Development** (`ALPHAVEDHA_ENV != production`): ConsoleRenderer with colors to stderr.

**Production:** JSONRenderer to stderr + `logs/alphavedha.log` (50 MB, 5 backups) + `logs/alphavedha-error.log` (20 MB, 3 backups, errors only).

Processors: merge_contextvars → add_log_level → add_logger_name → TimeStamper(fmt="iso") → StackInfoRenderer → UnicodeDecoder → JSONRenderer.

---

## Key Environment Variables

| Variable | Required | Description |
|---|---|---|
| DATABASE_URL | Yes | `postgresql+asyncpg://user:pass@postgres:5432/alphavedha` |
| REDIS_URL | Yes | `redis://:password@redis:6379/0` |
| ALPHAVEDHA_API_KEY | No | Primary API key (32+ chars recommended); if unset, all requests pass |
| ALPHAVEDHA_API_KEY_SECONDARY | No | Rotation key for zero-downtime key change |
| ALPHAVEDHA_DEMO | No | `1` = demo mode (no model weights needed) |
| ALPHAVEDHA_HEAVY_TRAINING | No | `1` = enable LSTM/TFT retraining in scheduler |
| ALPHAVEDHA_REGIME_OVERLAY | No | `1` = apply regime overlay to position sizing |
| ALPHAVEDHA_CORS_ORIGINS | No | Comma-separated allowed origins |
| ALERT_EMAIL_ENABLED | No | `1` = enable email alerts |
| ALERT_SMTP_HOST | No | default: smtp.gmail.com |
| ALERT_SMTP_PORT | No | default: 587 |
| ALERT_SMTP_SENDER | No | Sender email address |
| ALERT_SMTP_PASSWORD | No | SMTP password / app password |
| ALERT_SMTP_RECIPIENT | No | Alert recipient email |
| REDDIT_CLIENT_ID | No | Reddit API (for sentiment) |
| REDDIT_CLIENT_SECRET | No | Reddit API secret |
| FINNHUB_API_KEY | No | News sentiment (alternative) |
| MARKETAUX_API_KEY | No | News sentiment (alternative) |
