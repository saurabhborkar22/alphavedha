# AlphaVedha — Master Progress Checklist

> Last updated: 2026-06-05
> Total tests: 858+ | Source LOC: ~19,500 | Test LOC: ~11,500

---

## Foundation (Weeks 1-8) — COMPLETE

### Week 1: Data Pipeline
- [x] Data providers (yfinance, jugaad-data) with rate limiting and retry
- [x] OHLCV ingestion for Nifty 50 (80,264 rows, Jan 2020 — May 2026)
- [x] PostgreSQL + TimescaleDB schema (DailyOHLCV, CorporateAction, IndexConstituent)
- [x] Preprocessing pipeline (corporate actions, circuit detection, missing data, fractional diff, outliers)
- [x] Redis caching layer
- [x] Data validation (validate_ohlcv) and error handling

### Week 2: Feature Engineering (PR #1)
- [x] Technical features (SMA, EMA, RSI, MACD, Bollinger, ATR, ADX, CCI, Stochastic, Williams %R)
- [x] Macro features (India VIX, FII/DII flows, Nifty indices, yield curve, oil, USD/INR)
- [x] Fundamental features (P/E, P/B, ROE, Debt/Equity, Dividend yield, EPS)
- [x] Derivatives features (OI, PCR, Greeks, futures basis, IV)
- [x] Sentiment features (Finnhub/MarketAux news sentiment)
- [x] Calendar features (day of week, month, earnings dates, holidays)
- [x] Returns features (log returns, rolling volatility, Sharpe, Sortino)
- [x] Microstructure features (delivery %, volume profile, VWAP)
- [x] Feature pipeline orchestrator (159 features total)

### Week 3: Labeling + XGBoost + CPCV + Backtest (PR #2)
- [x] Triple barrier labeling (asymmetric: 2.0x up, 1.5x down, ATR-based)
- [x] Sample weight computation (decay-based, trend enforcement)
- [x] BaseModel ABC (fit/predict/save/load interface)
- [x] XGBoost model (dual classification + regression heads, joblib serialization)
- [x] CPCV validation (Combinatorial Purged Cross-Validation)
- [x] Backtest engine (VectorBT, Indian cost model)
- [x] 70 tests

### Week 4: LSTM + Temporal Attention (PR #3)
- [x] LSTM model (2-layer, 128 hidden, safetensors serialization)
- [x] Temporal Fusion Transformer (GRN, VSN, interpretable multi-head attention)
- [x] Multi-horizon prediction (7d/15d/30d)
- [x] Shared sequence utilities (EarlyStopping, SequenceDataset, padding)
- [x] 44 tests (228 total)

### Week 5: HMM Regime + Conformal Prediction (PR #4)
- [x] HMM Regime Detector (4-state: bull/bear/sideways/high-volatility)
- [x] Conformal Predictor (MAPIE jackknife+, 90% coverage target)
- [x] 26 tests (254 total)

### Week 6: Ensemble Stacking + Meta-Labeling (PR #5)
- [x] Stacking Ensemble (RidgeClassifier meta-learner, 14 meta-features, OOF training)
- [x] Meta-Labeling Model (XGBClassifier binary gate, 0.55 threshold, filters 30-40% of signals)
- [x] 24 tests (278 total)

### Week 7: Prediction Engine + Risk Management (PR #6)
- [x] Position sizing (Half-Kelly, 10% cap)
- [x] Portfolio constraints (sector 25% cap, correlation 0.7, liquidity 5cr, 3d min holding)
- [x] Circuit breaker (3-level drawdown: 10%/15%/20%)
- [x] Risk manager orchestrator
- [x] Composite scorer (6 weighted sub-scores)
- [x] Stock ranker (buy/sell separation, score-based ranking)
- [x] Prediction engine (full pipeline orchestrator, graceful degradation)
- [x] 47 tests (325 total)

### Week 8: API + CLI (PR #7)
- [x] FastAPI app factory (lifespan, exception handlers, structlog)
- [x] API authentication (X-API-Key header)
- [x] Rate limiting (slowapi, 100/min default, 10/min for batch)
- [x] Health + readiness endpoints
- [x] Prediction routes (predict, batch, scan)
- [x] Demo mode (works without DB/models)
- [x] Typer CLI (predict, scan, serve commands)
- [x] Rich formatters (panels, tables, colored output)
- [x] Service layer (ModelRegistry, PredictionCache, PredictionService)
- [x] 57 tests (382 total)

---

## Phase A: Immediate Impact — COMPLETE (PRs #9, #10)

### A1. FII/DII Flow Data Ingestion
- [x] NSE provider (direct NSE fetches, FII/DII flows, circuit data, bhavcopy)
- [x] Rate limiting (1 req/2s for NSE)
- [x] FII/DII features activated (fii_net_flow_5d, fii_net_flow_10d, dii_net_flow_5d, fii_dii_divergence, fii_buying_streak)

### A2. F&O Data Ingestion
- [x] Derivatives data ingestion (futures OI, PCR, max pain, IV)
- [x] Options chain parsing
- [x] Derivatives features activated (futures_oi_change_pct, futures_basis, pcr_oi, pcr_volume, max_pain_distance, iv_percentile, oi_concentration)

### A3. Delivery Percentage Signals
- [x] Delivery % features (micro_delivery_pct_rank, micro_delivery_vol_combo, micro_high_delivery_breakout)
- [x] Statistical significance validation

### A4. Earnings Surprise Tracker
- [x] Earnings provider (NSE SEBI filings, Screener.in)
- [x] Quarterly/annual results parser
- [x] Earnings features (earnings_surprise_pct, days_since_earnings, revenue_growth_qoq)

### A5. Regime-Conditional Strategy Selection
- [x] Regime-aware strategy selection (different configs per bull/bear/sideways/high-vol)
- [x] Position sizing varies by regime

### A6. Walk-Forward Backtest
- [x] Walk-forward engine (time-series cross-validation, train/test splits, embargo)
- [x] Indian cost model (STT, brokerage, slippage)
- [x] Equity curve generation
- [x] Monthly returns breakdown

---

## Phase B: Strategic Edge — COMPLETE (PR #11)

### B1. Pairs Trading Engine
- [x] Cointegration detection and hedge ratios
- [x] Pairs signal generation (spread z-score trading)
- [x] Pairs universe discovery (sector-based correlation)

### B2. Promoter Pledging & Insider Activity Tracker
- [x] SEBI provider (corporate actions, shareholding patterns, insider trading)
- [x] Promoter pledging features
- [x] Insider activity tracking

### B3. Indian Financial News Sentiment
- [x] News provider (Finnhub/MarketAux integration)
- [x] Sentiment scoring pipeline
- [x] News volume z-score (event detection)

### B4. Alternative Data: Auto Sales & Cement Dispatch
- [x] Alt data provider (auto sales, cement dispatch, PMI, credit growth)
- [x] Sector-stock mapping for alt data features

### B5. Paper Trading Dashboard
- [x] Paper trading API routes (open/close positions, portfolio, trades, performance)
- [x] Dashboard API (summary, allocations, sector breakdown, top gainers/losers)
- [x] Slippage modeling in paper trades

### B6. RL Portfolio Optimizer (Prototype)
- [x] ActorCritic network (PPO architecture)
- [x] Trading environment (state: features + positions + portfolio, action: position weights)
- [x] Reward function (PnL - costs + Sharpe bonus - drawdown penalty)
- [x] Complete PPO training loop (implemented in D5.4)

---

## Phase C: World-Class — COMPLETE (PR #12)

### C1. Graph Neural Network
- [x] Pure PyTorch GraphSAGE (no torch_geometric dependency)
- [x] Stock graph builder (sector, correlation, promoter group edges)
- [x] Edge deduplication
- [x] Full-batch training with dual heads (classification + regression)
- [x] safetensors serialization
- [x] YAML-driven stock config (configs/stocks.yaml)

### C2. Online Learning
- [x] Drift detection (PSI + KS test per feature)
- [x] Performance monitoring (rolling accuracy 7d/30d/90d, precision, alpha vs benchmark)
- [x] Auto-retrainer (calendar/drift/performance triggered, shadow deployment 20d)

### C3. Full Alternative Data Pipeline
- [x] Extended alt data provider (13 sources: GST, auto sales, cement, power, port cargo, UPI, credit growth, forex reserves, crude oil, US overnight, job postings, PMI, steel)
- [x] 5 new macro features integrated (EXPECTED_FEATURE_COUNT: 159)

### C4. Intelligent Execution Engine
- [x] Almgren-Chriss market impact model (Indian cap tier calibration)
- [x] Bid-ask cost modeling (5bps large / 10bps mid / 20bps small)
- [x] F&O expiry detection (monthly + weekly)
- [x] Execution timing rules (avoid opening/closing auction, prefer stability windows)
- [x] Participation rate checks

### C5. Public Track Record Dashboard
- [x] Track record API (accuracy breakdowns by regime/sector/cap/confidence)
- [x] Predictions API (paginated, filtered, searchable)
- [x] Equity curve endpoint
- [x] Monthly returns endpoint
- [x] CSV/JSON export
- [x] Model info endpoint
- [x] Deterministic demo data (MD5 hash seeding per symbol+date)

---

## Production & Operations

### D1. Production Infrastructure — COMPLETE (PR #15)

#### D1.1 Containerization
- [x] Dockerfile (multi-stage build, Python 3.12-slim, non-root user, health check)
- [x] docker-compose.prod.yml (app + scheduler + postgres + redis, health checks, env_file)
- [x] .dockerignore, .env.prod.example

#### D1.2 Process Management
- [x] systemd service template for API server (gunicorn + uvicorn workers)
- [x] systemd service template for background scheduler
- [x] Auto-restart, security hardening (NoNewPrivileges, ProtectSystem=strict)

#### D1.3 Reverse Proxy & SSL
- [x] nginx configuration (SSL termination, gzip, security headers)
- [x] Let's Encrypt / certbot integration
- [x] Rate limiting at proxy level (30r/s general, 5r/s predictions)
- [x] VPS setup script (deploy/setup.sh)

#### D1.4 CI/CD Pipeline
- [x] GitHub Actions workflow: ruff lint + format check + mypy on all PRs
- [x] GitHub Actions workflow: pytest unit tests with coverage on all PRs
- [x] Coverage artifact upload on main push
- [x] Dependabot (weekly pip, monthly Actions)

### D2. Observability & Monitoring — COMPLETE (PR #16)

#### D2.1 Metrics
- [x] Prometheus `/metrics` endpoint (prometheus-fastapi-instrumentator)
- [x] 11 custom metrics: request latency, prediction count/confidence, model load time
- [x] Feature computation time, scheduler job duration/status, drift PSI
- [x] Active positions, cache hit/miss counters

#### D2.2 Alerting
- [x] Email SMTP alerting (Gmail App Password compatible)
- [x] Alert methods: scheduler_job_failed, drift_detected, accuracy_drop, api_error_spike
- [x] AlertConfig from environment variables

#### D2.3 Logging
- [x] Structured JSON logging (structlog) for production
- [x] Colored console logging for development
- [x] RotatingFileHandler (50MB main/5 backups, 20MB error/3 backups)

#### D2.4 Health Checks
- [x] Extended `/ready` — checks DB connectivity, Redis, model status
- [x] Returns model_version in readiness response

### D3. Database & Migrations — COMPLETE (PR #18)

#### D3.1 Schema Management
- [x] Initialize Alembic (async-aware env.py with asyncpg)
- [x] Generate initial migration from ORM models (13 tables)
- [x] Migration check script (scripts/check_migrations.py)

#### D3.2 Performance
- [x] Connection pool tuning (env-configurable: DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_POOL_TIMEOUT, DB_POOL_RECYCLE)

#### D3.3 Backup & Recovery
- [x] Automated daily backup script (scripts/backup_db.sh — pg_dump + gzip + pruning)
- [x] Database restore script (scripts/restore_db.sh — drop/recreate + pg_restore + alembic stamp)

### D4. Security Hardening — COMPLETE (PR #18)

#### D4.1 Authentication & Authorization
- [x] API key rotation (dual-key: ALPHAVEDHA_API_KEY + ALPHAVEDHA_API_KEY_SECONDARY)
- [x] Timing-safe key comparison (hmac.compare_digest)
- [x] CORS configuration (ALPHAVEDHA_CORS_ORIGINS env var, comma-separated)
- [x] API key hashing utility for safe logging

#### D4.2 Secrets Management
- [x] Environment validation script (scripts/validate_env.py — detects placeholder values)
- [x] .env.prod.example updated with all new config vars

#### D4.3 Security Audit
- [x] Dependency vulnerability scan in CI (pip-audit in GitHub Actions)
- [x] Input sanitization: symbol regex validation, tier allowlist, top_n bounds
- [x] Invalid API key attempts logged with prefix
- [x] 11 new tests (API key rotation, input validation, hash utility)

### D5. ML Operations Improvements — COMPLETE (PR #25)

#### D5.1 Experiment Tracking
- [x] JSON-based ExperimentTracker (file-based, zero external dependencies)
- [x] RunRecord dataclass with hyperparams, metrics, data range, artifact path
- [x] Run logging integrated into training pipeline (all models auto-log)
- [x] Run comparison (side-by-side val metric deltas)
- [x] CLI: `experiment list`, `experiment compare`

#### D5.2 Model Serving
- [x] Real feature loading in PredictionService (ModelRegistry loads saved models, FeatureEngine computes live features)
- [x] Model warm-up on server start (runs single prediction to exercise full path)
- [x] Async batch prediction with semaphore (10 concurrent, `asyncio.gather`)
- [x] Concurrent `scan_tier` using batch prediction

#### D5.3 Automated Model Comparison
- [x] ComparisonResult dataclass with promote/discard/extend_shadow recommendations
- [x] `RetrainingManager.compare_models()` — compares active vs shadow metrics
- [x] Threshold-based logic (accuracy +0.01 → promote, -0.02 → discard, marginal → extend)
- [x] CLI: `model compare`

#### D5.4 Complete RL Agent
- [x] PPO training loop in `train_rl_agent()` (episode-based, GAE advantages)
- [x] Walk-forward validation (`walk_forward_rl` with configurable windows)
- [x] WalkForwardResult with avg Sharpe, return, max drawdown
- [x] RL integrated into `train_all()` as Step 10

#### D5.5 Model Export Fixes
- [x] Export GNN model in `models/__init__.py`
- [x] Export Conformal model in `models/__init__.py`
- [x] Verify all models are importable via public API

#### D5.6 TimescaleDB Hypertables
- [x] Alembic migration converting 8 tables to hypertables (monthly chunks)
- [x] Composite natural PKs replacing serial id columns
- [x] Compression policies (daily_ohlcv: 6mo, features: 3mo)
- [x] Colab training notebook (private repo support with PAT)

### D6. Testing Gaps — COMPLETE (PR #22, #27)

#### D6.1 Missing Unit Tests — COMPLETE
- [x] `data/stock_graph.py` — stock relationship graph (17 tests)
- [x] `data/ingestion.py` — data ingestion orchestration (8 tests)
- [x] `data/store.py` — feature store read/write (27 tests)
- [x] `data/database.py` — DB connection logic (3 tests)
- [x] `training/pipeline.py` — temporal splits, feature selection, data prep (18 tests)
- [x] `training/gnn_pipeline.py` — GNN training (6 tests)
- [x] `training/rl_pipeline.py` — RL training (4 tests)
- [x] `models/trading_env.py` — RL trading environment (14 tests)
- [x] `signals/pairs_universe.py` — cointegration, hedge ratio, half-life, scanning (16 tests)
- [x] `api/deps.py` — API auth and dependency injection (8 tests)
- [x] `exceptions.py` — all 11 exception classes (6 tests)

#### D6.2 Integration Tests — COMPLETE
- [x] Data pipeline end-to-end (store → load → verify, upsert idempotent, date range, multi-symbol isolation, delete)
- [x] API with real service wiring (health, ready, predict, batch, scan, invalid symbol, invalid tier — 7 tests, demo mode)
- [x] Feature store consistency (save/load round-trip, versioning, date range filtering — 3 tests)
- [x] Model save/load round-trip for all 8 model types (XGBoost, LSTM, TFT, HMM, Ensemble, Meta-Labeling, Conformal, PPO)
- [x] Docker test environment (docker-compose.test.yml, port 5433, tmpfs for speed)
- [x] Graceful skip when test DB unavailable (credential-verified probe, not just port check)

#### D6.3 Quality — COMPLETE
- [x] Coverage report (`make coverage` — pytest-cov with HTML report, 80% threshold)
- [x] Pre-commit hooks (.pre-commit-config.yaml — ruff check+fix, ruff format, mypy)
- [x] Pre-existing test failure resolved (test_universe.py was already passing — note removed)

### D7. Data Pipeline Enhancements — COMPLETE (PR #28)

#### D7.1 Live Data
- [x] Real-time OHLCV polling via yfinance fast_info (2-min interval during market hours)
- [x] LiveDataPoller: upsert IntradayOHLCV with GREATEST/LEAST high/low tracking
- [x] Redis prediction cache invalidation every 5 ticks (predict:{symbol}:* pattern)
- [x] is_market_open() IST-aware check (9:15-15:30, Mon-Fri)
- [x] Intraday poll scheduler job (every 2 min, skips when market closed)
- [x] CLI: `data live-status` (shows today's intraday OHLCV from DB)

#### D7.2 Data Quality
- [x] QualityChecker with 4 check types: completeness, freshness, consistency, anomaly
- [x] DataLineage ORM model + ingestion lineage tracking (_write_lineage helper)
- [x] DataQualityReport ORM model + persist_report()
- [x] QualityReport dataclass with n_passed / n_warnings / n_critical properties
- [x] EmailAlerter.data_quality_failed() integration for critical failures
- [x] Nightly quality check scheduler job (15:50 IST)
- [x] CLI: `data quality-check [--date YYYY-MM-DD]`

#### D7.3 Additional Data Sources
- [x] BSEProvider: corporate announcements (board meetings, dividends, AGM, bonus, etc.)
- [x] CorporateAnnouncement ORM model + upsert via uq_corp_announcement constraint
- [x] CLI: `data fetch-bse <symbols...> [--days N]`
- [x] GoogleTrendsProvider: 5 sectors (banking, IT, pharma, auto, FMCG) via pytrends
- [x] CLI: `data fetch-trends [--demo]`
- [x] Weekly Sunday night jobs: BSE ingestion (21:00), Google Trends (21:30)
- [x] 3 corporate event features: corp_days_to_next_board, corp_days_since_dividend, corp_event_this_week
- [x] 2 Google Trends features: trends_sector_7d, trends_sector_change
- [x] EXPECTED_FEATURE_COUNT: 159 → 164
- [x] 53+ new tests

### D8. Background Job Scheduling

- [x] Task scheduler setup (`schedule` library, lightweight in-process, IST-aware)
- [x] Daily 8:30 AM IST: pre-market predictions
- [x] Daily 3:45 PM IST: prediction outcome evaluation
- [x] Weekly: drift detection + performance evaluation (Saturday 8 PM)
- [x] Monthly: model retraining — first Saturday of month, 10 PM (if triggered)
- [x] CLI commands: `scheduler start`, `scheduler run-now <job>`, `scheduler status`
- [x] 21 unit tests for scheduler
- [x] Quarterly: index rebalancing check (Nifty composition changes, March/September)

### D9. Documentation — COMPLETE

- [x] DEPLOYMENT.md (production setup guide: Docker, VPS/systemd, env config, backups, key rotation, troubleshooting)
- [x] CONTRIBUTING.md (branch naming, commit conventions, code style, PR process, testing rules, financial data rules)
- [x] API_GUIDE.md (authentication, all endpoints with curl examples, error codes, rate limits, demo mode)
- [x] Architecture decision records (ADR.md — 9 ADRs: ensemble architecture, triple barrier labeling, meta-labeling, TimescaleDB, HMM regime, conformal prediction, demo mode, experiment tracking, async-first)
- [x] Model training guide (TRAINING_GUIDE.md — data ingestion → feature engineering → training all 10 models → validation → deployment, with CLI commands and troubleshooting)
- [x] Runbook for incidents (RUNBOOK.md — 6 incident types: drift, accuracy drop, scheduler failure, API errors, data outage, service startup)

### D10. UI/UX (Separate Repo: alphavedha-ui) — COMPLETE

#### D10.1 Setup
- [x] Initialize Next.js 15 + TailwindCSS + shadcn/ui project
- [x] Configure API client (lib/api/ — predictions, paper, public, system)
- [x] Authentication flow (login page + Zustand auth store)

#### D10.2 Core Pages
- [x] Dashboard (signal cards, live market strip, corporate events widget)
- [x] Scanner (filterable stock scan with filter-panel component)
- [x] Stock detail page (stock/[symbol] — confidence ring, feature bars, neural viz)
- [x] Backtest results viewer (equity curve, monthly returns)
- [x] ML Ops page (model status, drift, experiment tracking)

#### D10.3 Public Pages
- [x] Public track record page (track-record/ + track/)
- [x] Accuracy breakdown charts (donut, area, radar charts)
- [x] By-confidence breakdowns

#### D10.4 Advanced
- [x] Live intraday page (live/page.tsx)
- [x] Mobile-responsive design (mobile-bottom-nav.tsx)
- [x] Dark mode (dark theme throughout)
- [x] Notification system (notifications store + notifications/page.tsx)
- [x] Paper trading UI (paper/page.tsx + trade-modal)
- [x] Command palette (Cmd+K navigation)
- [x] Events page (corporate events calendar)
- [x] Trends page (Google Trends data)
- [x] Data quality page (data/page.tsx)
- [x] Settings page

### D11. Compliance & Legal

#### D11.1 SEBI (When Going Public)
- [ ] SEBI Research Analyst (RA) registration
- [ ] NISM Series XV certification
- [ ] Compliance officer appointment
- [ ] Disclaimer on all predictions
- [ ] Holdings disclosure for recommended stocks
- [ ] 5-year record maintenance

#### D11.2 Audit Trail
- [ ] Immutable prediction log (append-only PostgreSQL table)
- [ ] Prediction timestamp verification (before market open)
- [ ] Model version tracked with every prediction
- [ ] Access control logging

#### D11.3 Data Governance
- [ ] Data retention policy (define periods per data type)
- [ ] Data archival strategy (cold storage for old predictions)
- [ ] Privacy policy (if collecting user data via UI)

### D12. Model Training (End-to-End)

> These are not code tasks — they are operational tasks to run the existing code on real data.

- [ ] Ingest full historical data (2005-present, all Nifty 50 + Midcap 150)
- [ ] Train XGBoost on real features
- [ ] Train LSTM on real features
- [ ] Train TFT on real features
- [ ] Train HMM Regime Detector
- [ ] Train Ensemble (stacking meta-learner with OOF predictions)
- [ ] Train Meta-Labeling model
- [ ] Calibrate Conformal Predictor
- [ ] Train GNN with real stock graph
- [ ] Run walk-forward backtest on all models
- [ ] Validate: Sharpe > 1.0, max drawdown < 15%, beats Nifty 50 by 5%+
- [ ] Run paper trading for 30+ days to build track record

---

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Walk-forward Sharpe | > 1.5 | Not yet tested |
| Max Drawdown | < 12% | Not yet tested |
| Annual Return (after costs) | > Nifty 50 + 10% | Not yet tested |
| Win Rate | > 55% | Not yet tested |
| Profit Factor | > 1.8 | Not yet tested |
| Meta-label filter rate | 30-40% | Code ready, not tested on real data |
| Post-meta-label accuracy | > 55% | Code ready, not tested on real data |
| Paper trading track record | 90+ days verified | Not started |
| Drift detection latency | < 7 days | Code ready |
| Alternative data sources | 5+ | 13 sources implemented |
| Test count | 500+ | 797 |
| Source LOC | — | ~17,800 |

---

## Summary

| Area | Status | Progress |
|------|--------|----------|
| Foundation (Weeks 1-8) | COMPLETE | 100% |
| Phase A: Immediate Impact | COMPLETE | 100% |
| Phase B: Strategic Edge | COMPLETE | 100% |
| Phase C: World-Class | COMPLETE | 100% |
| D1: Production Infrastructure | COMPLETE | 100% |
| D2: Observability & Monitoring | COMPLETE | 100% |
| D3: Database & Migrations | COMPLETE | 100% |
| D4: Security Hardening | COMPLETE | 100% |
| D5: ML Ops Improvements | COMPLETE | 100% (experiment tracking, model serving, comparison, RL pipeline, TimescaleDB) |
| D6: Testing Gaps | COMPLETE | 100% (unit tests, integration tests, model round-trips, pre-commit hooks, coverage) |
| D7: Data Pipeline Enhancements | COMPLETE | 100% (quality checks, BSE/Trends data, live polling, 164 features) |
| D8: Background Scheduling | COMPLETE | 100% |
| D9: Documentation | COMPLETE | 100% (DEPLOYMENT.md, CONTRIBUTING.md, API_GUIDE.md, ADR.md, TRAINING_GUIDE.md, RUNBOOK.md) |
| D10: UI/UX | COMPLETE | 100% (14 pages, 23 components, API client, Zustand stores — repo: alphavedha-ui) |
| D11: Compliance & Legal | NOT STARTED | 0% (personal use only — SEBI not required) |
| D12: Model Training | NOT STARTED | 0% (operational task — run make train on real data) |
| D13: VPS Deployment | COMPLETE | 100% (Docker Compose stack, nginx, Tailscale, smoke-tested locally, deploy scripts ready) |

**Overall: Full-stack AI stock prediction platform complete end-to-end. ML engine (XGBoost + LSTM + TFT + HMM ensemble, 164 features), production API (FastAPI + Redis + TimescaleDB), Next.js UI (14 screens, dark mode, mobile-responsive), VPS deployment stack (Docker Compose, Hetzner CX22 target, Tailscale access). 858+ tests. Remaining: deploy to real Hetzner VPS and run real model training on live NSE data.**
