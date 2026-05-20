# AlphaVedha — Master Progress Checklist

> Last updated: 2026-05-20
> Total tests: 548 | Source LOC: ~15,800 | Test LOC: ~7,500

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
- [ ] Complete PPO training loop (partially implemented — architecture only)

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

## Remaining Work — NOT STARTED

### D1. Production Infrastructure

#### D1.1 Containerization
- [ ] Dockerfile (multi-stage build, slim Python 3.12 base)
- [ ] docker-compose prod override (secrets from .env.prod, no hardcoded credentials)
- [ ] Container health checks

#### D1.2 Process Management
- [ ] systemd service template for API server
- [ ] systemd service template for background scheduler
- [ ] Auto-restart and failure recovery

#### D1.3 Reverse Proxy & SSL
- [ ] nginx configuration (SSL termination, gzip, caching headers)
- [ ] Let's Encrypt certificate automation
- [ ] Rate limiting at proxy level

#### D1.4 CI/CD Pipeline
- [ ] GitHub Actions workflow: lint + typecheck on all PRs
- [ ] GitHub Actions workflow: pytest unit tests on all PRs
- [ ] GitHub Actions workflow: integration tests on main
- [ ] Coverage reporting and badge
- [ ] Automated dependency updates (Dependabot/Renovate)

### D2. Observability & Monitoring

#### D2.1 Metrics
- [ ] Prometheus `/metrics` endpoint
- [ ] Request latency histograms
- [ ] Prediction count/accuracy gauges
- [ ] Model inference time tracking
- [ ] Feature computation time tracking

#### D2.2 Error Tracking
- [ ] Sentry integration (or alternative: Axiom, Datadog)
- [ ] Alert channels (email/Slack/PagerDuty) for critical failures
- [ ] Error rate dashboards

#### D2.3 Logging
- [ ] Centralized log aggregation (ELK, Loki, or cloud provider)
- [ ] Log rotation and retention policy
- [ ] Structured log indexing for search

#### D2.4 Health Checks
- [ ] Extend `/ready` to check DB connectivity
- [ ] Extend `/ready` to check feature store health
- [ ] Extend `/ready` to check Redis connectivity
- [ ] Uptime monitoring (UptimeRobot, Pingdom)

### D3. Database & Migrations

#### D3.1 Schema Management
- [ ] Initialize Alembic
- [ ] Generate initial migration from ORM models
- [ ] Migration CI check (prevent unapplied migrations)

#### D3.2 Performance
- [ ] Index optimization audit (query explain plans)
- [ ] Partition strategy validation (TimescaleDB chunk intervals)
- [ ] Connection pool tuning for production load

#### D3.3 Backup & Recovery
- [ ] Automated daily backups (pg_dump or WAL archiving)
- [ ] Backup verification tests
- [ ] Point-in-time recovery procedure documented
- [ ] Disaster recovery runbook

### D4. Security Hardening

#### D4.1 Authentication & Authorization
- [ ] API key rotation mechanism
- [ ] Role-based access control (admin vs read-only)
- [ ] OAuth2/JWT support (if exposing to external users)
- [ ] CORS configuration (when frontend is deployed)

#### D4.2 Secrets Management
- [ ] Remove hardcoded docker credentials
- [ ] Vault integration (HashiCorp Vault, AWS Secrets Manager, or equivalent)
- [ ] Environment-specific secret injection

#### D4.3 Security Audit
- [ ] Dependency vulnerability scan (pip-audit, safety)
- [ ] OWASP security review of API endpoints
- [ ] Input sanitization review
- [ ] SQL injection prevention audit

### D5. ML Operations Improvements

#### D5.1 Experiment Tracking
- [ ] MLflow or Weights & Biases integration
- [ ] Hyperparameter logging per training run
- [ ] Run comparison dashboard
- [ ] Model performance history

#### D5.2 Model Serving
- [ ] Real feature loading in PredictionService (currently raises NotImplementedError — only demo works)
- [ ] Model warm-up on server start
- [ ] Model inference caching (Redis with market-hours TTL)
- [ ] Batch prediction optimization

#### D5.3 A/B Testing & Shadow Deployment
- [ ] Shadow model serving (run new model alongside production)
- [ ] Traffic splitting for A/B tests
- [ ] Automated performance comparison between model versions

#### D5.4 Complete RL Agent
- [ ] Finish PPO training loop implementation
- [ ] RL agent training pipeline (rl_pipeline.py)
- [ ] Walk-forward validation for RL agent
- [ ] Integration with ensemble (optional: add as 5th base model)

#### D5.5 Model Export Fixes
- [ ] Export GNN model in `models/__init__.py`
- [ ] Export Conformal model in `models/__init__.py`
- [ ] Verify all models are importable via public API

### D6. Testing Gaps

#### D6.1 Missing Unit Tests
- [ ] `data/stock_graph.py` — stock relationship graph
- [ ] `data/ingestion.py` — data ingestion orchestration
- [ ] `data/store.py` — feature store read/write
- [ ] `data/database.py` — DB connection logic
- [ ] `training/pipeline.py` — training orchestration
- [ ] `training/gnn_pipeline.py` — GNN training
- [ ] `training/rl_pipeline.py` — RL training
- [ ] `models/trading_env.py` — RL trading environment
- [ ] `signals/pairs_universe.py` — pairs discovery
- [ ] `api/deps.py` — API auth and dependency injection

#### D6.2 Integration Tests
- [ ] Data pipeline end-to-end (fetch -> preprocess -> store -> query)
- [ ] API with real prediction engine (not demo mode)
- [ ] Feature store consistency (training vs serving features match)
- [ ] Model save/load round-trip for all model types

#### D6.3 Quality
- [ ] Run coverage report and document per-module coverage
- [ ] Add pre-commit hooks (ruff, mypy, pytest)
- [ ] Fix the 1 pre-existing test failure (test_universe.py)

### D7. Data Pipeline Enhancements

#### D7.1 Live Data
- [ ] Real-time OHLCV updates during market hours
- [ ] WebSocket feed for live prices (Kite/Angel/Dhan broker API)
- [ ] Feature recomputation on new data arrival

#### D7.2 Data Quality
- [ ] Automated data quality checks (completeness, freshness, consistency)
- [ ] Data lineage tracking
- [ ] Anomaly detection on incoming data

#### D7.3 Additional Data Sources
- [ ] BSE corporate announcements (board meetings, dividends)
- [ ] MCA filings (Ministry of Corporate Affairs)
- [ ] Google Trends for sector interest
- [ ] Satellite data for infrastructure/real-estate stocks

### D8. Background Job Scheduling

- [ ] Task scheduler setup (Celery + Redis, or APScheduler)
- [ ] Daily 8:30 AM IST: pre-market predictions
- [ ] Daily 3:45 PM IST: prediction outcome evaluation
- [ ] Weekly: drift detection + performance evaluation
- [ ] Monthly: model retraining (if triggered)
- [ ] Quarterly: index rebalancing check (Nifty composition changes)

### D9. Documentation

- [ ] DEPLOYMENT.md (production setup guide, HA, scaling, environment config)
- [ ] CONTRIBUTING.md (branch naming, commit conventions, PR process, testing)
- [ ] Architecture decision records (ADRs) for key design choices
- [ ] API usage guide with curl examples
- [ ] Model training guide (end-to-end from data to deployment)
- [ ] Runbook for incidents (model drift, data outage, API errors)

### D10. UI/UX (Separate Repo: alphavedha-ui)

#### D10.1 Setup
- [ ] Initialize Next.js + TailwindCSS + shadcn/ui project
- [ ] Configure API client (fetch from AlphaVedha API)
- [ ] Authentication flow (API key management)

#### D10.2 Core Pages
- [ ] Dashboard (portfolio summary, today's predictions, equity curve)
- [ ] Predictions table (filterable, sortable, paginated)
- [ ] Stock detail page (prediction history, feature importance, confidence)
- [ ] Backtest results viewer (equity curve, monthly returns, trade log)
- [ ] Model performance page (accuracy over time, drift alerts, regime)

#### D10.3 Public Pages
- [ ] Public track record page (no auth required)
- [ ] Monthly returns card
- [ ] Accuracy breakdown charts
- [ ] Downloadable prediction CSV

#### D10.4 Advanced
- [ ] Real-time WebSocket updates during market hours
- [ ] Mobile-responsive design
- [ ] Dark mode
- [ ] Notification system (drift alerts, retraining events)

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
| Test count | 500+ | 548 |
| Source LOC | — | ~15,800 |

---

## Summary

| Area | Status | Progress |
|------|--------|----------|
| Foundation (Weeks 1-8) | COMPLETE | 100% |
| Phase A: Immediate Impact | COMPLETE | 100% |
| Phase B: Strategic Edge | COMPLETE | ~95% (RL training loop partial) |
| Phase C: World-Class | COMPLETE | 100% |
| D1: Production Infrastructure | NOT STARTED | 0% |
| D2: Observability | NOT STARTED | 0% |
| D3: Database & Migrations | NOT STARTED | 0% |
| D4: Security Hardening | NOT STARTED | 0% |
| D5: ML Ops Improvements | NOT STARTED | ~10% (some pieces exist) |
| D6: Testing Gaps | NOT STARTED | 0% |
| D7: Data Pipeline Enhancements | NOT STARTED | 0% |
| D8: Background Scheduling | NOT STARTED | 0% |
| D9: Documentation | PARTIAL | ~20% (README, CLAUDE.md exist) |
| D10: UI/UX | NOT STARTED | 0% (design prompts exist) |
| D11: Compliance & Legal | NOT STARTED | 0% |
| D12: Model Training | NOT STARTED | 0% |

**Overall: Core ML engine is complete (~88% of code). Production deployment, operations, training, and UI remain.**
