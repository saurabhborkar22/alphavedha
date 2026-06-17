# AlphaVedha — Quick Reference (Claude Context Primer)

AI-powered Indian stock market prediction engine. Python 3.12, FastAPI, PostgreSQL 16 + TimescaleDB, Redis 7, Next.js 16 UI.

---

## Module Map

| Path | Purpose |
|---|---|
| `alphavedha/api/` | FastAPI app: app.py (factory), deps.py (auth), schemas.py, routes/ (10 modules) |
| `alphavedha/services/` | prediction_service.py (orchestrator), model_registry.py (artifact loader), cache.py (Redis), ui_data.py (UI helpers) |
| `alphavedha/prediction/` | engine.py (15-step inference), scorer.py (composite 0-100), ranker.py, regime_strategy.py |
| `alphavedha/models/` | xgboost_model.py, lstm_model.py, temporal_attention.py (TFT), gnn_model.py, regime.py (HMM), ensemble.py (Ridge), meta_model.py, conformal.py (MAPIE), rl_agent.py |
| `alphavedha/features/` | pipeline.py (entry point), technical.py, returns.py, calendar_features.py, microstructure.py, macro.py, derivatives.py, sentiment.py, fundamental_features.py, corporate_events.py, trends_features.py |
| `alphavedha/training/` | pipeline.py (train_all, 10-step chain), gnn_pipeline.py, rl_pipeline.py |
| `alphavedha/labels/` | triple_barrier.py (labels), sample_weights.py (uniqueness + recency) |
| `alphavedha/data/` | database.py (asyncpg pool), models.py (ORM), store.py (CRUD), ingestion.py (yfinance/NSE/BSE), universe.py (Nifty composition), live_feed.py (intraday poll), quality.py, stock_graph.py (GNN edges) |
| `alphavedha/backtest/` | engine.py, costs.py (Indian cost model), cpcv.py (15 paths), walk_forward.py, sim_views.py |
| `alphavedha/monitoring/` | drift.py (PSI), performance.py (rolling accuracy), alerts.py (SMTP), track_record.py (3 tracks), retrainer.py, experiment_tracker.py, logging.py (structlog), metrics.py (Prometheus) |
| `alphavedha/risk/` | position_sizing.py (half-Kelly), circuit_breaker.py (3 levels), portfolio.py (sector/corr/liquidity), impact_model.py (Almgren-Chriss), risk_manager.py |
| `alphavedha/signals/` | execution.py (optimal windows), pairs.py (cointegration), pairs_universe.py (10 sector pairs) |
| `alphavedha/fundamental/` | analyzer.py, beneish.py (M-Score manipulation), altman.py (Z'-Score distress), fetcher.py |
| `alphavedha/sentiment/` | aggregator.py (SentimentAggregator), sources.py (RSS + Reddit) |
| `alphavedha/sectors/` | rotation.py (RRG analysis, 12 sector indices) |
| `alphavedha/scheduler.py` | AlphaVedhaScheduler — 12 jobs (daily/weekly/monthly) |
| `alphavedha/config.py` | AppConfig (Pydantic v2, cached singleton from default.yaml) |
| `alphavedha/exceptions.py` | AlphaVedhaError hierarchy (10 specific exception types) |
| `alphavedha/cli/main.py` | Typer CLI: predict, scan, serve, data, train, backtest, scheduler, experiment, model |
| `configs/default.yaml` | All model/data/risk/API defaults |
| `configs/stocks.yaml` | 11 sectors, 5 promoter groups, 49 Screener.in slugs |
| `alembic/versions/` | 4 migrations (13 tables → TimescaleDB → 4 new tables → is_tradeable) |

---

## Model Training Order (dependency chain)

```
Step 1: XGBoost          ← trains first (tabular, no deps)
Step 1: GNN              ← optional 4th base learner (parallel with XGBoost)
Step 1: RegimeDetector   ← HMM on portfolio returns+vol (parallel with XGBoost)
Step 2: LSTM             ← needs XGB feature_importance.csv (top-30 selection)
Step 2: TFT              ← same (parallel with LSTM)
Step 3: StackingEnsemble ← needs XGB+LSTM+TFT+Regime trained; trains on OOF predictions
Step 4: MetaLabeling     ← needs Ensemble trained; binary XGB gate on OOF
Step 5: Conformal        ← MAPIE on OOF; can be parallel (uses only OOF data)
Step 6: PPO RL           ← last; TradingEnvironment
```

Artifacts: `models/artifacts/{name}/latest/` (shared Docker volume).
Required for serving: xgboost, lstm, tft, regime, ensemble, meta_labeling, conformal (7 models).

---

## Key APIs (all relative to base URL)

| Endpoint | Auth | Purpose |
|---|---|---|
| GET /health | No | Liveness: `{"status":"ok"}` |
| GET /ready | No | Readiness: models + DB + Redis status |
| GET /predict/{symbol} | Yes | Single prediction (Redis-cached, 300s TTL market hours) |
| POST /predict/batch | Yes | Up to 20 symbols |
| GET /scan/{tier} | Yes | Scan large/mid/small/all universe |
| GET /scan/{tier} | No | Same route, no auth (ui_support.py — different handler) |
| WS /ws/live/{symbol} | No | Real-time tick stream (5s interval) |
| GET /paper/dashboard | No | Track record (3 tracks: all/gate/top_k) |
| GET /paper/simulation | No | Historical simulation artifact |
| GET /public/track-record | No | Public track record page data |
| GET /models/status | No | Model health + drift + system resources |
| GET /backtest/range | No | Per-day + date-range P&L from sim artifact |
| GET /fundamental/analyze/{symbol} | Yes | Beneish M-Score + Altman Z'-Score |
| GET /sentiment/{symbol} | Yes | FinBERT sentiment (7-day lookback default) |
| GET /sectors/trends | No | RRG sector rotation analysis |
| GET /metrics | No | Prometheus metrics |

Auth: `X-API-Key` header. No env var = dev pass-through. `hmac.compare_digest` constant-time.

---

## Database (17 tables)

Hypertables (8): `daily_ohlcv`, `features`, `derivatives_data`, `institutional_flows`, `daily_pnl`, `paper_trades`, `news_articles`, `insider_trades`

Non-hypertable (9): `corporate_actions`, `index_constituents`, `earnings_results`, `promoter_holdings`, `alternative_data`, `corporate_announcements`, `data_lineage`, `data_quality_reports`, `intraday_ohlcv`

Key: `features` table stores 164-feature JSON blob per (symbol, date, feature_version). Load: most recent 60 rows (sequence_length). `paper_trades.is_tradeable` distinguishes all-predictions from gate-passed trades.

Connection pool: asyncpg, pool_size=10, max_overflow=20, pool_recycle=1800s.

---

## Features: 164 declared, 148 effective

| Category | Count | Key Source |
|---|---|---|
| Technical | 40 | OHLCV via `ta` library |
| Returns | 21 | Close price (log returns, frac diff) |
| Calendar | 18 | DatetimeIndex (F&O expiry, RBI, seasons) |
| Microstructure | 13 | NSE delivery % from jugaad-data |
| Macro | 30 | yfinance + FII/DII DB (9 are stubs) |
| Derivatives | 20 | F&O OI, IV, PCR, max pain (6 are stubs) |
| Sentiment | 8 | FinBERT on news + Reddit |
| Fundamental | 9 | Earnings, pledging, insider trades |
| Corporate Events | 3 | BSE announcements |
| Trends | 2 | Google Trends (both stubs) |

16 stubs dropped at training time by `_STUB_FEATURES` frozenset in `training/pipeline.py`.

---

## Critical Rules (always enforce)

- **Timestamps:** ALL timezone-aware (Asia/Kolkata). Never naive datetime.
- **Train/val split:** Temporal only (never random). 20-day purge + 20-day embargo at each boundary.
- **Look-ahead bias:** Only data where `announced_date <= as_of_date` used for fundamental features.
- **No mock DB in integration tests.** Unit tests mock yfinance, not the database.
- **No auto-commit:** Always show changes first, never commit without explicit user request.
- **No direct push to main:** Every change needs a PR from a feature branch.
- **Hetzner API:** Read-only access only. No server changes without explicit per-message approval.
- **UI has no CI:** Deploy by rebuilding the `ui` Docker container manually on VPS.
- **Repo must be public:** Private repo would incur GitHub Actions metered billing.

---

## Daily Schedule (IST)

| Time | Job |
|---|---|
| 08:30 | Predict all Nifty 50 → store as paper_trades |
| 09:15 | Market opens (no automated events) |
| ~09:15–15:30 | Intraday poll every 2 min; Redis TTL=300s |
| 15:45 | Evaluate paper_trades at 15-day horizon |
| 15:50 | Data quality check |
| 17:00 | Data refresh (last 5 days OHLCV) |
| 18:30 | FII/DII ingestion |
| 23:30 | XGBoost retrain (CX23-safe) |
| Saturday 20:00 | Drift check (PSI + KS) |
| Saturday 22:30 | LSTM + TFT retrain (if ALPHAVEDHA_HEAVY_TRAINING=1) |
| Saturday 22:30 UTC | GitHub Actions train.yml (scales CX23→CX43→CX23) |

---

## Reference Documents

| Doc | Content |
|---|---|
| `docs/SYSTEM_ARCHITECTURE.md` | Full system map, repo structure, design decisions, data sources |
| `docs/BACKEND_REFERENCE.md` | All API routes with request/response schemas, services, scheduler, config |
| `docs/DATABASE_SCHEMA.md` | All 17 tables with column types, indexes, TimescaleDB config, ingestion pipeline |
| `docs/ML_ARCHITECTURE.md` | All 9 models, training pipeline steps, ensemble logic, meta-labeling |
| `docs/FEATURE_CATALOG.md` | All 164 features by category with lookback, source, stub flag |
| `docs/UI_ARCHITECTURE.md` | All 16 routes, components, API calls per page, real-time features, auth |
| `docs/DEVOPS_REFERENCE.md` | Docker services, CI/CD workflows, monitoring, risk, signals, sentiment |
| `docs/E2E_FLOW.md` | Day-in-the-life operational flow + complete request lifecycle |

Training/backtest/deployment runbooks: `docs/TRAINING_GUIDE.md`, `docs/RUNBOOK.md`, `docs/DEPLOYMENT.md`
