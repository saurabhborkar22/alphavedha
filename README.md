# AlphaVedha (अल्फावेध)

AI-powered Indian stock market prediction engine for NSE/BSE.

Combines ensemble ML models (XGBoost + LSTM + TFT + GNN), India-specific microstructure signals, graph neural networks, alternative data, and quantitative finance techniques to predict stock direction, magnitude, and price targets.

**548 tests | 159 features | 8 ML models | 13 alt data sources | ~16K LOC**

## Architecture

```
Data Providers (NSE, jugaad, yfinance, SEBI, news, alt data)
    ↓
Preprocessing (corp actions → circuits → missing data → FFD → outliers)
    ↓
Feature Engine (159 features: technical, macro, fundamental, derivatives, sentiment, calendar, returns, microstructure, alt data)
    ↓
Base Models (XGBoost + LSTM + TFT) → Stacking Ensemble → Meta-Labeling Gate → Conformal Prediction
    ↑                                                                              ↓
HMM Regime Detector ─── regime-conditional strategy ───────────── Risk Manager (Kelly + constraints + circuit breaker)
    ↑                                                                              ↓
GNN (GraphSAGE) ─── stock relationship modeling ──────────────── Execution Engine (Almgren-Chriss impact model)
                                                                                   ↓
                                                                    API (FastAPI) + CLI (Typer + Rich)
```

## Setup

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Start PostgreSQL (TimescaleDB) + Redis
docker compose up -d

# 3. Run tests
pytest tests/unit/ -v
```

## Quick Commands

```bash
# Development
make lint               # ruff check + mypy
make test               # pytest with coverage
make test-unit          # Unit tests only (no DB needed)

# Data
make data-refresh       # Fetch latest market data
make data-backfill      # Backfill historical data (20 years)

# ML
make train              # Train all models
make train-xgboost      # Train XGBoost only
make train-lstm         # Train LSTM only
make train-tft          # Train TFT only
make train-regime       # Train HMM regime detector
make validate           # Run CPCV validation

# Prediction
make predict SYMBOL=TCS.NS    # Predict one stock
make scan TIER=large           # Scan Nifty 50

# CLI (direct)
alphavedha predict TCS --demo          # Predict with Rich output
alphavedha predict TCS --demo --json   # JSON output
alphavedha scan large --demo --top-n 5 # Scan and rank stocks
alphavedha serve --demo                # Start API in demo mode

# API
make serve              # Start FastAPI dev server (http://localhost:8000/docs)

# Backtest
make backtest           # Full walk-forward backtest
```

## Project Structure

```
alphavedha/
├── alphavedha/               # Main package (~15,800 LOC)
│   ├── config.py             # Pydantic config loader (configs/default.yaml)
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── data/                 # Data ingestion, preprocessing, storage
│   │   ├── providers/        # 6 providers: jugaad, yfinance, NSE, SEBI, earnings, news, alt data
│   │   ├── preprocessing/    # 5-stage pipeline: corp actions, circuits, missing, FFD, outliers
│   │   ├── stock_graph.py    # GNN graph builder (sector + correlation + promoter edges)
│   │   ├── database.py       # Async SQLAlchemy + connection pooling
│   │   ├── models.py         # ORM models (TimescaleDB hypertables)
│   │   ├── universe.py       # Nifty 50/Midcap 150/Smallcap index tracking
│   │   └── store.py          # Feature store with upsert
│   ├── features/             # 159 features across 9 modules
│   │   ├── technical.py      # SMA, EMA, RSI, MACD, Bollinger, ATR, ADX, CCI
│   │   ├── macro.py          # India VIX, FII/DII, indices, currency, commodities
│   │   ├── fundamental.py    # P/E, P/B, ROE, Debt/Equity, earnings surprise
│   │   ├── derivatives.py    # OI, PCR, Greeks, futures basis, IV
│   │   ├── sentiment.py      # FinBERT news sentiment
│   │   ├── calendar.py       # Day patterns, F&O expiry, holidays
│   │   ├── returns.py        # Log returns, volatility, Sharpe, Sortino
│   │   ├── microstructure.py # Delivery %, volume profile, VWAP
│   │   └── pipeline.py       # Feature pipeline orchestrator
│   ├── labels/               # Triple barrier labeling + sample weights
│   ├── models/               # 8 ML models
│   │   ├── xgboost_model.py  # Gradient boosted trees (classification + regression)
│   │   ├── lstm_model.py     # 2-layer LSTM with dual heads
│   │   ├── tft_model.py      # Temporal Fusion Transformer (GRN + VSN + attention)
│   │   ├── gnn_model.py      # Pure PyTorch GraphSAGE (no torch_geometric)
│   │   ├── regime.py         # HMM 4-state regime detector
│   │   ├── ensemble.py       # RidgeClassifier stacking meta-learner
│   │   ├── meta_model.py     # XGBClassifier meta-labeling gate
│   │   ├── conformal.py      # MAPIE jackknife+ prediction intervals
│   │   └── rl_agent.py       # PPO actor-critic (prototype)
│   ├── prediction/           # Prediction engine
│   │   ├── engine.py         # Full pipeline orchestrator (10 steps)
│   │   ├── scorer.py         # 6-factor composite scorer
│   │   ├── ranker.py         # Buy/sell ranking by score
│   │   └── regime_strategy.py # Regime-conditional strategy selection
│   ├── risk/                 # Risk management
│   │   ├── position_sizing.py # Half-Kelly criterion
│   │   ├── portfolio.py      # Sector cap, correlation, liquidity constraints
│   │   ├── circuit_breaker.py # 3-level drawdown protection
│   │   ├── risk_manager.py   # Orchestrator (Kelly → constraints → CB)
│   │   └── impact_model.py   # Almgren-Chriss market impact (Indian cap tiers)
│   ├── signals/              # Signal generation
│   │   ├── pairs.py          # Pairs trading (cointegration + spread trading)
│   │   └── execution.py      # Order execution + timing
│   ├── monitoring/           # Online learning
│   │   ├── drift.py          # PSI + KS test drift detection
│   │   ├── performance.py    # Rolling accuracy, precision, alpha tracking
│   │   └── retrainer.py      # Auto-retraining (calendar/drift/performance triggers)
│   ├── training/             # Training pipelines for all models
│   ├── backtest/             # Walk-forward, CPCV, Indian cost model
│   ├── services/             # ModelRegistry, PredictionCache, PredictionService
│   ├── api/                  # FastAPI (health, predictions, paper trading, public dashboard)
│   └── cli/                  # Typer CLI (predict, scan, serve, data, train, backtest)
├── tests/                    # 548 tests
├── configs/                  # YAML configuration
│   ├── default.yaml          # All parameters
│   └── stocks.yaml           # Sectors, promoter groups, screener symbols
├── docs/                     # Design specs, architecture docs
├── docker-compose.yml        # PostgreSQL (TimescaleDB) + Redis
├── Makefile                  # Build, test, train, serve targets
└── pyproject.toml            # Dependencies, ruff, mypy, pytest config
```

## ML Pipeline

| Component | Description |
|-----------|-------------|
| **XGBoost** | Tabular features, dual classification + regression heads |
| **LSTM** | 2-layer, 128 hidden, 30-day sequences |
| **TFT** | Temporal Fusion Transformer, multi-horizon (7d/15d/30d) |
| **GNN** | GraphSAGE on stock relationship graph (sector + correlation + promoter edges) |
| **HMM** | 4-state regime detector (bull/bear/sideways/high-volatility) |
| **Stacking Ensemble** | RidgeClassifier on 14 meta-features (model probs + regime + disagreement) |
| **Meta-Labeling** | Binary gate: filters 30-40% of low-confidence signals |
| **Conformal** | MAPIE jackknife+: 90% coverage prediction intervals |

## Risk Management

| Layer | Description |
|-------|-------------|
| **Position Sizing** | Half-Kelly criterion, capped at 10% per stock |
| **Portfolio Constraints** | Sector 25% cap, correlation 0.7, liquidity Rs 5cr, 3d min holding |
| **Circuit Breaker** | L1 (10% DD): halve sizes, L2 (15%): block entries, L3 (20%): close all |
| **Execution** | Almgren-Chriss impact model, F&O expiry avoidance, timing rules |

## API Endpoints

| Route | Auth | Description |
|-------|------|-------------|
| `GET /health` | No | Liveness probe |
| `GET /predict/{symbol}` | Yes | Single stock prediction |
| `POST /predict/batch` | Yes | Batch prediction (up to 20) |
| `GET /scan/{tier}` | Yes | Scan Nifty 50 / Midcap / Smallcap |
| `POST /paper/open-position` | Yes | Open paper trade |
| `GET /paper/portfolio` | Yes | Paper trading portfolio |
| `GET /dashboard/summary` | Yes | Portfolio overview |
| `GET /public/track-record` | No | Full performance with breakdowns |
| `GET /public/predictions` | No | Historical predictions (paginated) |
| `GET /public/equity-curve` | No | Portfolio vs benchmark |
| `GET /public/predictions/export` | No | Download CSV/JSON |

## Build Progress

| Phase | Focus | Status |
|-------|-------|--------|
| Week 1 | Data pipeline, preprocessing, DB models | Done |
| Week 2 | Feature engineering (159 features, 9 modules) | Done |
| Week 3 | Triple barrier labeling, XGBoost, CPCV, backtest | Done |
| Week 4 | LSTM, Temporal Fusion Transformer | Done |
| Week 5 | HMM regime detection, conformal prediction | Done |
| Week 6 | Ensemble stacking, meta-labeling | Done |
| Week 7 | Prediction engine, risk management | Done |
| Week 8 | FastAPI REST API, Typer CLI | Done |
| Phase A | NSE data (FII/DII, F&O), delivery signals, earnings, walk-forward backtest | Done |
| Phase B | Pairs trading, promoter pledging, news sentiment, alt data, paper trading, RL prototype | Done |
| Phase C | GNN (GraphSAGE), online learning, 13 alt data sources, execution engine, public dashboard | Done |

See [docs/PROGRESS.md](docs/PROGRESS.md) for the full checklist including remaining work.

## Tech Stack

Python 3.12 | PyTorch | XGBoost | scikit-learn | hmmlearn | MAPIE | FastAPI | Typer | Rich | PostgreSQL + TimescaleDB | Redis | SQLAlchemy 2.0 | Pydantic v2 | structlog | safetensors | VectorBT

## License

MIT
