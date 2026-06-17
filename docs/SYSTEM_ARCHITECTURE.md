# AlphaVedha — System Architecture

## What It Is

AI-powered Indian stock market prediction platform for NSE/BSE (Nifty 50 + Midcap 150). Predicts direction, magnitude, and price targets for individual stocks using an ensemble of 8 ML models trained on 164 features. Paper-trades its own signals to maintain a verifiable track record.

**Not a trading bot.** No broker API integration. Prediction-only + paper trading for now.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Hetzner VPS (CX23)                      │
│                                                                  │
│  ┌──────────┐    ┌─────────────────────────────────────────┐    │
│  │          │    │              Docker Network              │    │
│  │ Internet │    │                                          │    │
│  │  :80     │───▶│  nginx                                  │    │
│  │          │    │   ├── /        → ui:3000 (Next.js)      │    │
│  └──────────┘    │   ├── /api/    → api:8000 (FastAPI)     │    │
│                  │   └── /api/ws/ → api:8000/ws/ (WS)      │    │
│                  │                                          │    │
│                  │  api (uvicorn)     scheduler             │    │
│                  │   ├── PredictionService                  │    │
│                  │   │   └── PredictionEngine               │    │
│                  │   │       ├── XGBoostModel               │    │
│                  │   │       ├── LSTMModel                  │    │
│                  │   │       ├── TFT (TemporalAttention)    │    │
│                  │   │       ├── GNNModel (optional)        │    │
│                  │   │       ├── RegimeDetector (HMM)       │    │
│                  │   │       ├── StackingEnsemble (Ridge)   │    │
│                  │   │       ├── MetaLabelingModel          │    │
│                  │   │       └── ConformalPredictor         │    │
│                  │   ├── RiskManager                        │    │
│                  │   └── Redis (prediction cache)           │    │
│                  │                                          │    │
│                  │  postgres (TimescaleDB)  redis           │    │
│                  │  model-artifacts (shared volume)         │    │
│                  └─────────────────────────────────────────┘    │
│                                                                  │
│  Trainer (CX43, on-demand)                                       │
│   └── TrainingPipeline.train_all() — 8-step dependency chain    │
└─────────────────────────────────────────────────────────────────┘

GitHub Actions
  ├── ci.yml     — lint + test + deploy (on push to main)
  ├── train.yml  — weekly model training (Saturday 10:30 PM IST)
  └── sim.yml    — historical simulation (manual trigger)

alphavedha-ui (separate repo, no CI — manual Docker rebuild)
```

---

## Repository Structure

```
alphavedha/                       # Backend (this repo)
├── alphavedha/
│   ├── api/                      # FastAPI app factory + routes
│   │   ├── app.py                # create_app(), lifespan startup
│   │   ├── deps.py               # auth (verify_api_key), DI helpers
│   │   ├── schemas.py            # Pydantic request/response schemas
│   │   └── routes/               # 10 route modules
│   ├── services/
│   │   ├── prediction_service.py # Central orchestrator (cache + features + engine)
│   │   ├── model_registry.py     # Artifact loading + demo mock models
│   │   ├── cache.py              # Redis + in-process LRU prediction cache
│   │   └── ui_data.py            # Data helpers for UI endpoints
│   ├── prediction/
│   │   ├── engine.py             # PredictionEngine — 15-step inference pipeline
│   │   ├── scorer.py             # CompositeScorer (6 sub-scores → 0-100)
│   │   ├── ranker.py             # StockRanker — sorts by composite score
│   │   └── regime_strategy.py   # Per-regime kelly/threshold params
│   ├── models/                   # All ML model classes
│   │   ├── base.py               # BaseModel ABC
│   │   ├── xgboost_model.py      # XGBoost (tabular)
│   │   ├── lstm_model.py         # LSTM (temporal)
│   │   ├── temporal_attention.py # TFT (multi-horizon + VSN)
│   │   ├── gnn_model.py          # GraphSAGE (stock relationships)
│   │   ├── regime.py             # HMM regime detector
│   │   ├── ensemble.py           # Stacking ensemble (Ridge)
│   │   ├── meta_model.py         # Meta-labeling (XGB binary gate)
│   │   └── conformal.py          # MAPIE conformal prediction intervals
│   ├── features/                 # Feature engineering (164 features)
│   │   ├── pipeline.py           # compute_all_features() entry point
│   │   ├── technical.py          # 40 features: TA-lib indicators
│   │   ├── returns.py            # 21 features: log returns, momentum
│   │   ├── calendar_features.py  # 18 features: expiry, season, RBI
│   │   ├── microstructure.py     # 13 features: NSE delivery %
│   │   ├── macro.py              # 30 features: VIX, FII/DII, forex, rates
│   │   ├── derivatives.py        # 20 features: OI, IV, PCR, max pain
│   │   ├── sentiment.py          # 8 features: FinBERT on news/Reddit
│   │   ├── fundamental_features.py # 9 features: earnings, pledging, insider
│   │   ├── corporate_events.py   # 3 features: board meetings, dividends
│   │   └── trends_features.py    # 2 features: Google Trends (stub)
│   ├── training/
│   │   ├── pipeline.py           # train_all() — 10-step training orchestrator
│   │   ├── gnn_pipeline.py       # GNN-specific training
│   │   └── rl_pipeline.py        # PPO RL training
│   ├── labels/
│   │   ├── triple_barrier.py     # Triple-barrier label generation
│   │   └── sample_weights.py     # Uniqueness + recency weights
│   ├── data/
│   │   ├── database.py           # asyncpg connection pool
│   │   ├── models.py             # SQLAlchemy ORM models (17 tables)
│   │   ├── store.py              # OHLCV + features read/write
│   │   ├── ingestion.py          # Data pipeline (yfinance, NSE, BSE)
│   │   ├── universe.py           # Nifty/Midcap composition management
│   │   ├── live_feed.py          # Intraday poll (LiveDataPoller)
│   │   ├── quality.py            # Data quality checks
│   │   └── stock_graph.py        # GNN edge construction (sector+corr+promoter)
│   ├── backtest/
│   │   ├── engine.py             # Backtesting loop (cost-adjusted P&L)
│   │   ├── costs.py              # Indian trading cost model
│   │   ├── cpcv.py               # Combinatorial Purged CV (15 paths)
│   │   ├── walk_forward.py       # Walk-forward validation
│   │   └── sim_views.py          # Simulation artifact view builders
│   ├── monitoring/               # MLOps
│   │   ├── drift.py              # PSI + KS drift detection
│   │   ├── performance.py        # Rolling accuracy tracking
│   │   ├── alerts.py             # Email alerts (SMTP)
│   │   ├── track_record.py       # 3-track paper trade analysis
│   │   ├── retrainer.py          # Retrain trigger + version lifecycle
│   │   ├── experiment_tracker.py # Run metadata (JSON files)
│   │   ├── logging.py            # structlog structured logging
│   │   └── metrics.py            # Prometheus metrics
│   ├── risk/
│   │   ├── position_sizing.py    # Generalized half-Kelly
│   │   ├── circuit_breaker.py    # 3-level drawdown protection
│   │   ├── portfolio.py          # Sector/correlation/liquidity constraints
│   │   ├── impact_model.py       # Almgren-Chriss market impact
│   │   └── risk_manager.py       # Orchestrates all risk components
│   ├── signals/
│   │   ├── execution.py          # Optimal execution windows + VWAP
│   │   ├── pairs.py              # Cointegration pairs trader
│   │   └── pairs_universe.py     # 10 pre-defined sector pairs
│   ├── fundamental/
│   │   ├── analyzer.py           # FundamentalAnalyzer orchestrator
│   │   ├── fetcher.py            # yfinance financial statements
│   │   ├── beneish.py            # Beneish M-Score (8-variable manipulation detector)
│   │   └── altman.py             # Altman Z'-Score (distress predictor)
│   ├── sentiment/
│   │   ├── aggregator.py         # SentimentAggregator (FinBERT + RSS + Reddit)
│   │   └── sources.py            # RSS and Reddit data sources
│   ├── sectors/
│   │   └── rotation.py           # RRG sector rotation analysis
│   ├── scheduler.py              # AlphaVedhaScheduler (12 jobs)
│   ├── config.py                 # AppConfig (Pydantic v2, cached singleton)
│   └── exceptions.py             # Custom exception hierarchy
├── tests/
│   ├── unit/                     # Unit tests per module
│   ├── integration/              # DB integration tests (real DB, no mocks)
│   └── backtest/                 # Strategy validation tests
├── alembic/versions/             # 4 migrations
├── configs/
│   ├── default.yaml              # All model/data/risk/API defaults
│   ├── features.yaml             # Feature registry
│   └── stocks.yaml               # Nifty 50 sectors, promoter groups, Screener slugs
├── deploy/
│   ├── nginx-vps.conf            # Production nginx config
│   └── nginx.conf                # Alternative nginx (HTTPS)
├── scripts/                      # Utility scripts (sim_paper_trading, hetzner_scale, etc.)
└── docs/                         # Architecture reference docs (this directory)

alphavedha-ui/                    # Frontend (separate repo)
├── app/                          # Next.js 16 App Router
│   ├── (app)/                    # Auth-protected route group (12 pages)
│   ├── login/                    # Public login page
│   └── track/                    # Public track record
├── components/
│   ├── layout/                   # NavBar, MobileBottomNav, CommandPalette
│   ├── dashboard/                # SignalCard, LiveMarketStrip, CorporateEventsWidget
│   ├── scanner/                  # FilterPanel
│   ├── paper/                    # TradeModal
│   ├── neural/                   # NeuralViz (SVG model animation)
│   ├── core/                     # GlassCard, StatCard, ConfidenceRing, DirectionBadge, etc.
│   └── charts/                   # AreaChart, CandlestickChart, FeatureBars, etc. (pure SVG)
└── lib/
    ├── api/                      # client.ts, predictions.ts, paper.ts, system.ts, public.ts
    ├── store/                    # Zustand: auth, watchlist, notifications
    ├── use-live-stream.ts        # WebSocket hook
    └── utils.ts + glossary.ts    # Shared utilities + metric definitions
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| All 'use client' in Next.js | Simpler state, no RSC hydration complexity; prediction data changes frequently |
| Redis with in-process LRU fallback | API stays functional when Redis is down; no hard dependency |
| TimescaleDB hypertables | Automatic partitioning + compression on time-series tables; no manual partitioning |
| Triple-barrier labels (not simple forward returns) | Point-in-time exit via volatility-adjusted barriers; avoids look-ahead bias |
| 20-day purge + embargo | Triple-barrier labels can span up to 15 days; 20-day gap ensures zero overlap |
| Stacking over voting | Ridge meta-learner learns optimal base model weights from OOF; adapts to regime |
| Meta-labeling gate | Separates "which direction?" (ensemble) from "should I trade?" (meta-model); improves precision |
| Conformal prediction intervals | Valid coverage guarantees (90%) without distributional assumptions |
| Fractional differentiation | Makes price series stationary while preserving long-range memory (vs log-differencing) |
| Half-Kelly sizing | Avoids Kelly ruin risk while still scaling position with edge |
| CX23 → CX43 scale for training | LSTM/TFT need 16 GB RAM; CX23 serves 24/7 for €3.99/mo |
| No CI for UI | Avoids GitHub Actions metered billing; manual Docker rebuild is acceptable |
| Repo public | Free GitHub Actions minutes (CI/CD pipeline) |

---

## Data Sources

| Source | Data | Rate Limit | Used For |
|---|---|---|---|
| yfinance (.NS suffix) | OHLCV, fast_info, financials, calendar | 2 req/s | Primary OHLCV, macro, live feed, corporate events |
| niftyindices.com | Index compositions (CSV) | — | Universe management |
| NSE (jugaad-data) | Daily bhavcopy, F&O | 0.5 req/s | delivery_pct, derivatives |
| BSE | Corporate announcements bulk | — | corporate_announcements table |
| Screener.in | Quarterly earnings | sequential | earnings_results table |
| Moneycontrol/ET/BS RSS | News headlines | — | FinBERT sentiment |
| Reddit (PRAW) | r/IndiaInvestments + 3 others | 25 posts/sub | FinBERT sentiment |
| Google Trends | Sector search interest | — | trends_* features (stub for now) |
| RBI | G-Sec yield, PMI | — | Stub features (not live yet) |

---

## Indian Market Constants (hardcoded in calendar_features.py / signals/execution.py)

- Market hours: 09:15 – 15:30 IST, Monday–Friday
- Pre-open: 09:00 – 09:15 IST
- F&O monthly expiry: last Thursday of month
- F&O weekly expiry: every Thursday (Nifty/BankNifty)
- Optimal execution window: 10:30–11:30 and 14:00–14:45 IST
- Avoid: 09:15–09:30 (opening noise), 15:20–15:30 (closing manipulation risk)
- STT: 0.1% on delivery buy+sell
- Settlement: T+1 (moving to T+0 for select stocks)
- Circuit limits: 5% / 10% / 20% on individual stocks; 10% / 15% / 20% on indices
- Nifty 50 rebalance: March and September (semi-annual)
