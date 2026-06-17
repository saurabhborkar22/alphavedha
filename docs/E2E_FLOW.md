# AlphaVedha — End-to-End Flow

This document covers: (1) the complete daily operational lifecycle of the system, and (2) the step-by-step lifecycle of a single API prediction request.

---

## System Architecture Overview

```
User Browser
     │
     ▼
nginx (port 80)
├── / → Next.js UI (port 3000)
├── /api/* → FastAPI (port 8000)  [HTTP]
└── /api/ws/* → FastAPI (port 8000)  [WebSocket upgrade]
     │
     ├── FastAPI (uvicorn)
     │   ├── PredictionService → PredictionEngine
     │   │   ├── XGBoostModel
     │   │   ├── LSTMModel
     │   │   ├── TemporalAttentionModel (TFT)
     │   │   ├── GNNModel (optional)
     │   │   ├── RegimeDetector (HMM)
     │   │   ├── StackingEnsemble (Ridge)
     │   │   ├── MetaLabelingModel (XGB binary)
     │   │   └── ConformalPredictor (MAPIE)
     │   └── RiskManager → CircuitBreaker + Kelly
     │
     ├── Scheduler (separate process)
     │   └── 12 jobs across daily/weekly/monthly cadence
     │
     ├── PostgreSQL 16 + TimescaleDB
     │   └── 17 tables (8 hypertables)
     │
     └── Redis 7
         └── Prediction cache (TTL-aware)

Model Artifacts: shared Docker volume between api + scheduler + trainer
```

---

## Daily Operational Flow

### Pre-Market (before 9:15 AM IST)

**08:30 AM — Daily Predictions Job** (`scheduler.run_daily_predictions`)
1. Skips weekends
2. Builds fresh `PredictionService` (no Redis cache — ensures fresh model weights)
3. Calls `service.predict_tier("large")` → all 50 Nifty symbols
4. `StockRanker.rank(predictions, top_n=50)` — ranks all signals
5. Persists ALL predictions as `paper_trades` rows:
   - predicted_direction, predicted_magnitude, confidence, model_version, regime, is_tradeable
   - entry_price = last close (fetched from OHLCV, 10-day lookback)
   - prediction_date = today
6. If any prediction fails → `scheduler_job_failed` email alert

---

### Market Open (9:15 AM IST)

- No additional automated events
- Predictions from 8:30 AM are stored; API is live serving those cached predictions
- Redis prediction TTL = 300s (5 minutes) during market hours

---

### During Trading Hours (9:15 AM – 3:30 PM IST)

**Every 2 minutes — Intraday Poll** (`scheduler.run_intraday_poll`)
- `LiveDataPoller.poll_once()` via `asyncio.to_thread`
- Polls `yfinance.Ticker.fast_info` for all stocks in stocks.yaml
- Upserts `intraday_ohlcv`: GREATEST(high), LEAST(low), last_price, tick_count += 1
- Every 5th tick: deletes `predict:{symbol}:*` Redis keys → forces fresh prediction on next request

**On-demand predictions (API requests):**
- Cache hit → instant response from Redis
- Cache miss → full prediction pipeline (see Request Lifecycle below) → write to Redis

**WebSocket streams (active connections to /ws/live/{symbol}):**
- Every 5s: fetch latest tick via `asyncio.to_thread` → broadcast to connected clients

---

### Market Close (3:30 PM IST) + After Close

**3:45 PM — Daily Evaluation Job** (`scheduler.run_daily_evaluation`)
1. Loads open paper trades where `prediction_date <= today - 21 calendar days` (ensures ~15 trading days elapsed)
2. For each trade: fetches actual OHLCV close at 15-trading-day horizon via yfinance
3. Computes: `actual_return = (exit_price / entry_price) - 1`
4. `is_correct = (predicted_direction * actual_return > 0)` — directional accuracy
5. Updates paper_trades: exit_price, actual_return, is_correct
6. `_store_pnl_summary()` → writes to `daily_pnl` table

**3:50 PM — Quality Check Job** (`scheduler.run_quality_check`)
- `QualityChecker.run_full_check(today)` across completeness / freshness / consistency / anomaly
- Persists to `data_quality_reports`
- Sends `data_quality_failed` email alert if any critical failures

**5:00 PM — Data Refresh Job** (`scheduler.run_data_refresh`)
- `refresh_latest(tier, lookback_days=5)` — ingests last 5 trading days of OHLCV

**6:30 PM — FII/DII Ingestion** (`scheduler.run_fii_dii_ingestion`)
- NSE publishes FII/DII data ~17:30 IST
- `ingest_fii_dii()` → upserts into `institutional_flows`
- Feeds `macro_fii_net`, `macro_dii_net`, `macro_fii_cum_5d`, `macro_dii_cum_5d` features

**11:30 PM — Daily XGBoost Retrain** (`scheduler.run_daily_xgboost_retrain`)
- `train_xgboost(tier)` — fits within CX23 budget (2 vCPU, 4 GB RAM)
- After-hours Redis TTL extends to: seconds until next 9:15 AM IST

---

### Weekly Jobs (Saturday)

**8:00 PM — Drift Check** (`scheduler.run_drift_check`)
- `DriftDetector.check_drift(reference_df, current_df)` across all numeric features
- PSI > 0.1 → WARNING, PSI > 0.2 → ALERT + email + `requires_retrain=True`
- KS test also computed per feature

**10:00 PM — Monthly Retrain Check** (1st Saturday only: `day <= 7`)
- `RetrainingManager().should_retrain()` — evaluates drift + performance triggers
- If triggered: runs full `train_all()` or targeted retraining

**10:30 PM — Weekly LSTM + TFT Retrain** (`ALPHAVEDHA_HEAVY_TRAINING=1` gated)
- Runs only when env var is set (typically via GitHub Actions train.yml workflow)
- `train_lstm(tier)` + `train_tft(tier)` — RAM-intensive (needs CX43: 8 vCPU, 16 GB)

**Saturday 10:30 PM UTC (10:30 PM IST = 17:00 UTC) — GitHub Actions train.yml**
- Scales Hetzner VPS: CX23 → CX43 (poweroff → change_type → poweron, ~5 min downtime)
- SSHes to VPS: `docker compose run trainer python -m alphavedha.cli.main train all --tier large`
- Scales back to CX23 on `if: always()` — guaranteed scale-down even on failure
- Timeout: 8 hours

---

### Sunday Jobs

**9:00 PM — BSE Ingestion** (`scheduler.run_bse_ingestion`)
- `ingest_bse_announcements(symbols, today-7, today)`
- Upserts `corporate_announcements`

**9:30 PM — Trends Ingestion** (`scheduler.run_trends_ingestion`)
- Google Trends for 5 market sectors

---

### Monday (Market Open Week)

**7:00 AM — Rebalance Check** (March/September only)
- Fetches live Nifty 50 composition from niftyindices.com
- Compares with current `index_constituents` records
- Logs additions/removals (does not auto-update stocks.yaml — manual step)

---

### Quarterly (March / September)

Semi-annual Nifty 50 index rebalance:
1. Run `alphavedha data refresh` (or `refresh_universe()`)
2. New constituents added to `index_constituents` with `effective_from=today`, `effective_to=NULL`
3. Removed constituents: update `effective_to=today`
4. Point-in-time queries now reflect the new composition going forward

---

## Complete Request Lifecycle: GET /predict/TCS

### 1. nginx (port 80)
```
GET /api/predict/TCS
→ strip /api prefix
→ proxy_pass http://api:8000/predict/TCS
   proxy_read_timeout 300s
   (WebSocket upgrade headers also set for /api/ws/)
```

### 2. FastAPI — uvicorn receives request
- App singleton created at startup via `create_app()`
- `slowapi.Limiter` checks per-IP rate limit (100/min default)
- If exceeded → 429 with `Retry-After: 60`

### 3. Auth check (deps.py: verify_api_key)
- Reads `X-API-Key` header
- No env var configured → pass (dev mode)
- Configured → `hmac.compare_digest(key, stored_key)` constant-time compare
- Failure → 401 (missing) or 403 (invalid)

### 4. Route handler (routes/predictions.py)
- `get_service()` → returns singleton `PredictionService` (set at lifespan startup)
- Service not initialized → 503

### 5. PredictionService.predict_single("TCS")
- **Cache lookup:** Redis GET `predict:TCS:{model_version}`
  - HIT → deserialize → return immediately (CACHE_HITS++ Prometheus counter)
  - MISS → CACHE_MISSES++ → proceed to step 6

### 6. Feature Loading (_get_features → _load_real_features)
- Load `_feature_window_rows=60` rows from `features` table (most recent, looking back 2×60=120 calendar days)
- Feature version filter: `feature_version = config.feature_version`
- Fallback: `_compute_features_on_the_fly("TCS")`:
  - Loads OHLCV from `daily_ohlcv`
  - Runs `compute_all_features()` in thread pool (`asyncio.to_thread`)
  - Writes result back to feature store (best-effort)

### 7. Market Features (_get_market_features)
- Per-day in-memory cache of equal-weight portfolio log returns + 20-day realized vol
- Needed by RegimeDetector
- If unavailable → regime="unknown", probs=[0.25, 0.25, 0.25, 0.25]

### 8. PredictionEngine.predict("TCS", features, sector, market_features, ...)

**8a. RegimeDetector.predict(returns, vol)**
→ regime_name, regime_probs[4]

**8b. RegimeStrategySelector.select(regime_name)**
→ kelly_fraction, meta_confidence_threshold, require_all_models_agree

**8c. Base model inference** (XGBoost + LSTM + TFT + optional GNN)
- Each model: `model.predict(features) → PredictionResult`
- XGBoost: tabular features
- LSTM: top-30 feature subset, 60-day sequence, last row output
- TFT: same subset, multi-horizon, uses 15d horizon for direction
- Failed model → neutral placeholder; need ≥ 2 successes or PredictionError

**8d. StackingEnsemble.predict(base_predictions, regime_probs)**
- Builds 14-col meta-feature matrix: 3×proba + 3×proba + 3×proba + 4×regime + disagreement
- Ridge.decision_function() → softmax → direction, confidence, magnitude (confidence-weighted)

**8e. MetaLabelingModel.predict(features, ensemble_direction, ensemble_confidence)**
→ meta_confidence = P(ensemble correct), is_tradeable = meta_confidence > threshold

**8f. Regime threshold gate**
- meta_confidence < strategy.meta_confidence_threshold → is_tradeable = False

**8g. ConformalPredictor.predict(features)**
→ price_low, price_mid, price_high (as return%, converted to price via last_close)

**8h. CompositeScorer.score(ensemble, regime, features)**
→ composite_score [0-100] from 6 weighted sub-scores

### 9. RiskManager.assess(meta_confidence, magnitude)
- `compute_position_size()` → generalized half-Kelly: `fraction = p - (1-p)/b; position = fraction × 0.5 × 100%`
- Cap at `max_single_stock_pct=10%`
- Portfolio constraints (if portfolio state provided): sector cap, correlation, liquidity, holding period
- CircuitBreaker: drawdown-based level (0-3); adjusts position by multiplier

### 10. ATR Trade Levels (_compute_atr_levels)
- Long: stop = entry - 1.5×ATR, target = entry + 2.0×ATR
- Short: stop = entry + 1.5×ATR, target = entry - 2.0×ATR

### 11. Redis Cache Write
- `_compute_ttl()`: 300s during 09:15-15:30 IST weekdays, else seconds to next 9:15 IST
- `redis.setex(f"predict:TCS:{model_version}", ttl, json_payload)`
- Redis unavailable → silent no-op (API still functions)

### 12. Response
```json
{
  "symbol": "TCS",
  "direction": 1,
  "direction_label": "BUY",
  "magnitude": 0.025,
  "composite_score": 72.4,
  "meta_confidence": 0.68,
  "is_tradeable": true,
  "regime": "bull",
  "price_targets": {"low": 3850.0, "mid": 3950.0, "high": 4050.0},
  "risk": {"position_size_pct": 5.2, "model_disagreement": 0.12},
  "trade_setup": {"entry_price": 3900.0, "stop_loss_price": 3820.0, "take_profit_price": 3980.0},
  "model_version": "ensemble_v1.2",
  "generated_at": "2026-06-17T09:30:00+05:30",
  "warnings": []
}
```

---

## Data Flow: Market Data → Features → DB

```
yfinance (.NS suffix)
    │  rate_limit: 2 req/s, 3 retries
    ▼
OHLCVProvider.fetch_ohlcv()
    │
    ▼
Preprocessing Pipeline (in order):
  1. corporate_actions.py   → adjusts split/bonus/rights backwards; stores adj_close
  2. circuit_handler.py     → sets circuit_hit flag (upper/lower/None); no rows dropped
  3. missing_data.py        → forward-fills holidays (is_filled=True); no interpolation
  [4. fractional_diff.py]   → skipped at ingestion time; runs at feature compute time
  [5. outlier_treatment.py] → skipped at ingestion time; winsorizes [1%, 99%]
    │
    ▼
store_ohlcv()  →  daily_ohlcv (TimescaleDB hypertable)
    │
    ▼
[At prediction/training time]
compute_all_features()  →  164-col DataFrame
    │
    ▼
store_features()  →  features (TimescaleDB hypertable, JSON blob per row)
```

---

## Data Flow: Model Training → Artifacts → Serving

```
PostgreSQL (features + daily_ohlcv + labels)
    │
    ▼
TrainingPipeline.train_all()
  Step 1: Load tier data → 3-way temporal split (70/15/15 + 20-day embargos)
  Step 2: Train XGBoost → xgboost/latest/
  Step 3: Select top-30 features (from XGB feature_importance.csv)
  Step 4: Train LSTM → lstm/latest/
  Step 5: Train TFT → tft/latest/
  Step 6: Train Regime (HMM) → regime/latest/
  Step 7: Train Ensemble (Ridge on OOF) → ensemble/latest/
  Step 8: Train Meta-Labeling (XGB binary on OOF) → meta_labeling/latest/
  Step 9: Train Conformal (MAPIE on OOF) → conformal/latest/
    │
    ▼ (shared Docker volume: model-artifacts)
    │
    ▼
ModelRegistry.get_prediction_engine()
  → loads all 7+ model artifacts at startup
  → PredictionEngine singleton
    │
    ▼
PredictionService (uses engine per request)
```

---

## Track Record Flow (paper trades → UI)

```
08:30 AM: Daily predictions persisted to paper_trades
  (predicted_direction, confidence, is_tradeable, entry_price=last_close)
    │
    ▼  [15+ trading days later]
    │
3:45 PM: Daily evaluation job
  → actual_return = (exit_price / entry_price) - 1
  → is_correct = True/False
  → updates paper_trades
    │
    ▼
compute_track_record(trades, round_trip_cost_pct, gate_confidence, top_k=5)
  → 3 tracks: all / gate_passed (is_tradeable=True) / top_k (top-5 by confidence)
  → per-track: win_rate_net, avg_return_net, sharpe_net, max_drawdown_net
    │
    ▼
GET /public/track-record  →  /track UI page
GET /paper/dashboard      →  /paper UI page
GET /paper/simulation     →  /backtest UI page (from sim_artifact.json)
```

---

## Historical Simulation Flow (one-time, manual trigger)

```
GitHub Actions: sim.yml (workflow_dispatch)
  inputs: cutoff (train end date), end (last sim day), regime_overlay, deploy_ui
    │
    ▼
Scale up: CX23 → CX43
    │
    ▼
VPS: scripts/sim_paper_trading.py --cutoff DATE --end DATE
  │
  ├── as_of seam: PredictionService(as_of=date) — loads only data known before cutoff
  │
  ├── Day-by-day loop (cutoff+1 → end):
  │   ├── Predict all symbols using models trained up to cutoff
  │   ├── Store predictions with prediction_date=loop_date
  │   └── Next day: evaluate outcomes (actual close at horizon)
  │
  ├── Build sim_artifact.json (schema_version=2):
  │   ├── track_record: 3-track cost-adjusted results
  │   ├── backtest: equity curve, monthly heatmap, return distribution, rolling Sharpe
  │   ├── diagnostics: calibration curve (10 deciles), cost sensitivity (0x/0.5x/1x/2x)
  │   └── meta: cutoff, end, model_version, schema_version
  │
  └── API reads artifact via load_sim_artifact() (mtime-based cache)
    │
    ▼
Scale down: CX43 → CX23 (guaranteed by if: always())
```
