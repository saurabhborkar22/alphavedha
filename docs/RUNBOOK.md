# Incident Runbook

Procedures for diagnosing and resolving common operational issues.

## Alert Reference

AlphaVedha sends email alerts (when `ALERT_EMAIL_ENABLED=true`) for these events:

| Alert | Level | Trigger |
|-------|-------|---------|
| Scheduler job failed | CRITICAL | Any scheduled job throws an unhandled exception |
| Feature drift detected | WARNING | PSI > 0.2 for any feature group |
| Accuracy drop | WARNING | Rolling accuracy below threshold (default 52%) |
| API error spike | CRITICAL | High error count within a time window |

---

## Incident 1: Feature Drift Detected

**Alert:** `[AlphaVedha WARNING] Feature drift detected: {feature_group}`

### What it means

The distribution of input features has shifted significantly compared to the training data. PSI (Population Stability Index) above 0.2 indicates the model's inputs look different from what it was trained on.

### Diagnosis

```bash
# Check which features drifted
alphavedha scheduler run-now weekly_drift_check

# Check logs for details
grep "drift" logs/alphavedha.log | tail -20
```

Common causes:
- **Market regime change** — e.g., sudden volatility spike, sector rotation
- **Data source issue** — upstream API changed format, missing fields
- **Calendar effect** — budget season, election, RBI policy announcement

### Remediation

**If PSI 0.2–0.5 (moderate drift):**
- Monitor for 2-3 more days. Drift may be transient (event-driven).
- No immediate action needed unless accuracy is also dropping.

**If PSI > 0.5 (severe drift):**
- Trigger a manual retrain:
  ```bash
  alphavedha train all --tier large
  ```
- The new model is deployed as shadow automatically. Verify predictions look reasonable before promoting.

**If drift is from a data source issue:**
- Check the data provider (yfinance, NSE):
  ```bash
  alphavedha data status
  alphavedha data refresh --tier large --days 5
  ```
- If a provider is down, the scheduler will retry on the next run. Features using that provider will be NaN-filled gracefully.

---

## Incident 2: Prediction Accuracy Drop

**Alert:** `[AlphaVedha WARNING] Prediction accuracy below threshold (30d)`

### What it means

Rolling 30-day directional accuracy has fallen below 52%. At this level, the model is barely better than random and should not be trusted for trading decisions.

### Diagnosis

```bash
# Check current performance metrics
alphavedha scheduler run-now daily_evaluation

# Review recent predictions vs outcomes
grep "evaluation" logs/alphavedha.log | tail -30
```

Check if the drop is:
- **Across all stocks** → systemic issue (regime change, model staleness)
- **Concentrated in a sector** → sector-specific event (earnings season, policy change)
- **Only for one direction** → buy signals still accurate but sell signals failing, or vice versa

### Remediation

**Step 1: Check for drift**
```bash
alphavedha scheduler run-now weekly_drift_check
```
If drift is detected alongside accuracy drop, retrain first.

**Step 2: Retrain if accuracy persists below 52% for 5+ days**
```bash
alphavedha train all --tier large
```

**Step 3: Compare new model vs old**
```bash
alphavedha model compare
```
The comparison shows val accuracy delta. Promote only if the new model improves metrics:
- Accuracy +0.01 → promote
- Accuracy -0.02 → discard and investigate further
- Marginal → extend shadow period to 20 more days

**Step 4: If retraining doesn't help**
- Check data quality: are there missing days, stale prices, or corrupt records?
- Check if market conditions are genuinely unpredictable (high VIX, geopolitical event).
- Consider pausing live predictions until conditions normalize. The circuit breaker (3-level drawdown at 10%/15%/20%) should automatically reduce position sizes.

---

## Incident 3: Scheduler Job Failed

**Alert:** `[AlphaVedha CRITICAL] Scheduler job failed: {job_name}`

### What it means

One of the five scheduled jobs threw an unhandled exception:

| Job | Schedule (IST) | Purpose |
|-----|---------------|---------|
| `daily_predictions` | 8:30 AM | Generate pre-market predictions |
| `daily_evaluation` | 3:45 PM | Score yesterday's predictions |
| `weekly_drift_check` | Saturday 8 PM | Drift detection + performance eval |
| `monthly_retrain` | 1st Saturday 10 PM | Model retraining (if triggered) |
| `quarterly_rebalance_check` | March/September | Nifty composition check |

### Diagnosis

```bash
# Check scheduler status
alphavedha scheduler status

# Check logs for the specific failure
grep "{job_name}" logs/alphavedha.log | tail -20
```

### Common Failures and Fixes

**`daily_predictions` failed:**
- Usually a database connection issue or model artifact not found.
- Fix: check PostgreSQL is running, check Redis is running, verify model files exist:
  ```bash
  ls models/artifacts/xgboost/latest/
  ls models/artifacts/ensemble/latest/
  ```
- Retry manually: `alphavedha scheduler run-now daily_predictions`

**`daily_evaluation` failed:**
- Usually no predictions to evaluate (if `daily_predictions` also failed earlier).
- Fix: resolve the prediction failure first, then evaluation will work on the next cycle.

**`weekly_drift_check` failed:**
- Usually insufficient data for PSI computation (need both reference and current windows).
- Fix: ensure at least 30 days of predictions exist. This self-resolves over time.

**`monthly_retrain` failed:**
- Usually a training pipeline error (OOM, data issue, model convergence failure).
- Fix: check `logs/alphavedha.log` for the specific model that failed. The pipeline is fault-tolerant — individual model failures don't stop the rest:
  ```bash
  grep "train_all" logs/alphavedha.log | tail -30
  ```
- Retry manually: `alphavedha scheduler run-now monthly_retrain`

---

## Incident 4: API Error Spike

**Alert:** `[AlphaVedha CRITICAL] API error spike: {count} errors in {window}min`

### What it means

The API is returning a high number of 5xx errors, indicating a server-side issue.

### Diagnosis

```bash
# Check API health
curl http://localhost:8000/health
curl http://localhost:8000/ready

# Check API logs
tail -100 logs/alphavedha.log | grep ERROR

# Check Prometheus metrics (if configured)
curl http://localhost:8000/metrics | grep http_requests_total
```

### Common Causes and Fixes

**Database connection errors (503 on /ready):**
```bash
# Check PostgreSQL
systemctl status alphavedha-db   # or: docker ps
# Check connection pool
grep "pool" logs/alphavedha.log | tail -10
```
Fix: restart PostgreSQL. If pool exhaustion, increase `DB_POOL_SIZE` (default: 5) or `DB_MAX_OVERFLOW` (default: 10) in environment.

**Redis connection errors:**
```bash
redis-cli ping
```
Fix: restart Redis. Predictions still work without Redis (cache miss, slower but functional).

**Model loading errors (500 on /predict):**
```bash
grep "model_load\|ModelNotFoundError" logs/alphavedha.log | tail -10
```
Fix: verify model artifacts exist. If corrupted, retrain:
```bash
alphavedha train all --tier large
```
Then restart the API server to reload models.

**Out of memory:**
```bash
free -h
grep "oom\|killed" /var/log/syslog | tail -5
```
Fix: reduce batch prediction concurrency. Set `PREDICTION_SEMAPHORE=5` (default: 10) in environment. Or add swap:
```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

**Rate limiting (429 responses):**

Not a server error — the client is exceeding rate limits. Limits:
- General: 100 requests/minute per IP
- Batch/scan: 10 requests/minute per IP
- Nginx proxy: 30 requests/second (if deployed)

No server-side fix needed. Client should implement backoff.

---

## Incident 5: Data Source Outage

**No automated alert** — detected when `data refresh` returns 0 rows or `data status` shows stale dates.

### Diagnosis

```bash
# Check data freshness
alphavedha data status

# Try a manual refresh
alphavedha data refresh --tier large --days 1
```

### Common Causes

**yfinance rate limited or down:**
- yfinance is the primary data source (free, no API key needed).
- NSE sometimes blocks IPs that make too many requests.
- Fix: wait 15-30 minutes and retry. The rate limiter (1 request per symbol per second) usually prevents this.

**NSE website changed format:**
- NSE periodically changes their HTML/API format, breaking scrapers.
- Fix: check the NSE provider (`alphavedha/data/providers/nse.py`) against the current NSE website. Update parsing logic if needed.

**Market holiday:**
- No data is expected on weekends, NSE holidays, or settlement holidays.
- The scheduler does not run predictions on non-trading days.
- Verify: check NSE holiday calendar at niftyindices.com.

### Impact

- **Missing 1-2 days:** Features use lookback windows (5d, 10d, 20d) so short gaps are tolerated. Predictions will still work but may be slightly less accurate.
- **Missing 5+ days:** Feature quality degrades significantly. Suspend predictions until data is caught up.

---

## Incident 6: Service Won't Start

### API server won't start

```bash
# Check systemd status
systemctl status alphavedha-api

# Check logs
journalctl -u alphavedha-api --since "10 minutes ago"
```

Common causes:
- **Port 8000 already in use:** `lsof -i :8000` → kill the old process or change the port.
- **Model warm-up failed:** The server runs a test prediction on startup. If model files are corrupt, it crashes. Check logs, retrain if needed.
- **Database migration pending:** `alembic upgrade head`

### Scheduler won't start

```bash
systemctl status alphavedha-scheduler
journalctl -u alphavedha-scheduler --since "10 minutes ago"
```

Common cause: another scheduler instance is already running. The scheduler doesn't use a lock file, so two instances will duplicate jobs.

```bash
# Find and kill duplicate
ps aux | grep "scheduler start"
```

---

## General Debugging

### Log locations

| Log | Path | Contents |
|-----|------|----------|
| Main | `logs/alphavedha.log` | All application logs (50MB, 5 rotations) |
| Errors | `logs/alphavedha-error.log` | ERROR level only (20MB, 3 rotations) |
| systemd | `journalctl -u alphavedha-*` | Service start/stop, crashes |
| nginx | `/var/log/nginx/alphavedha-*.log` | HTTP access and errors (if deployed) |

### Useful log queries

```bash
# All errors in the last hour
grep "ERROR" logs/alphavedha.log | tail -50

# Training events
grep "train_all" logs/alphavedha.log | tail -20

# Prediction performance
grep "evaluation\|accuracy" logs/alphavedha.log | tail -20

# Drift alerts
grep "drift\|psi" logs/alphavedha.log | tail -20
```

### Health check endpoints

```bash
curl localhost:8000/health     # Liveness (always responds if server is up)
curl localhost:8000/ready      # Readiness (checks DB, Redis, models)
curl localhost:8000/metrics    # Prometheus metrics
```

### Database backup (before risky operations)

```bash
scripts/backup_db.sh           # Creates timestamped pg_dump in backups/
scripts/restore_db.sh <file>   # Restore from backup
```
