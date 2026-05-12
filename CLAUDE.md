# AlphaVedha — AI-Powered Indian Stock Market Prediction Engine

## Project Overview

AlphaVedha (अल्फावेध) is a Python-based stock market analysis and prediction platform for Indian markets (NSE/BSE). It combines ensemble ML models (XGBoost + LSTM + Temporal Fusion Transformer), India-specific microstructure signals, and quantitative finance techniques to predict stock direction, magnitude, and price targets.

**Phase 1:** Prediction engine with 141-feature ensemble ML
**Phase 2:** AI-powered balance sheet analysis (Beneish M-Score, Altman Z-Score)
**Phase 3:** Signal timing engine with entry/exit optimization

## Tech Stack

- **Python 3.12** — core language
- **FastAPI** — REST API
- **PostgreSQL 16 + TimescaleDB** — time-series storage
- **Redis 7** — feature cache, rate limiting
- **XGBoost** — tabular feature model
- **PyTorch** — LSTM, Temporal Fusion Transformer
- **hmmlearn** — HMM regime detection
- **MAPIE** — conformal prediction intervals
- **VectorBT** — backtesting
- **FinBERT (HuggingFace)** — news sentiment
- **Typer** — CLI
- **Docker + docker-compose** — containerization

## Quick Commands
 
```bash
# Setup
make setup              # Create venv, install deps, setup DB
make docker-up          # Start PostgreSQL + Redis

# Development
make lint               # ruff check + mypy
make test               # pytest with coverage
make test-unit          # Unit tests only
make test-integration   # Integration tests only

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
make scan TIER=mid             # Scan Midcap 150

# API
make serve              # Start FastAPI dev server
make serve-prod         # Start with gunicorn

# Backtest
make backtest           # Full strategy backtest
```

## Project Structure

```
alphavedha/
├── alphavedha/           # Main package
│   ├── data/             # Data ingestion & preprocessing
│   ├── features/         # Feature engineering (141 features)
│   ├── labels/           # Triple barrier + meta-labeling
│   ├── models/           # ML models (XGBoost, LSTM, TFT, HMM)
│   ├── prediction/       # Prediction engine orchestrator
│   ├── risk/             # Risk management (Kelly, circuit breakers)
│   ├── backtest/         # VectorBT backtesting + CPCV validation
│   ├── monitoring/       # MLOps (drift detection, versioning)
│   ├── fundamental/      # Phase 2: Balance sheet AI
│   ├── signals/          # Phase 3: Signal timing
│   ├── api/              # FastAPI endpoints
│   └── cli/              # Typer CLI
├── tests/                # pytest test suite
├── configs/              # YAML configuration files
├── agents/               # Specialized Claude Code agent configs
├── notebooks/            # Jupyter experimentation
├── scripts/              # Utility scripts
└── docs/                 # Design specs and documentation
```

## Code Conventions

### Python Style
- Strict type hints on ALL function signatures — no `Any` unless documented why
- Use `from __future__ import annotations` in every file
- Pydantic v2 for all data models and config validation
- Use `pathlib.Path` not `os.path`
- Context managers for all resources (DB connections, file handles)
- Early returns to reduce nesting
- No mutable default arguments

### Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: prefix with `_`
- Type aliases: `PascalCase` with descriptive names

### Import Order
1. `__future__` annotations
2. Standard library
3. Third-party
4. Local (`from alphavedha.xxx import yyy`)

### Error Handling
- Custom exceptions in `alphavedha/exceptions.py`
- Raise specific exceptions, never bare `Exception`
- Log errors before raising (structured logging with `structlog`)
- Data pipeline errors: log + skip row + continue (never crash on bad data)
- Model errors: log + return None prediction + alert

### Testing
- Every module has corresponding tests in `tests/unit/`
- Integration tests in `tests/integration/` for data pipeline + DB
- Backtest tests in `tests/backtest/` for strategy validation
- Use `pytest` with `pytest-asyncio` mode `auto`
- Fixtures in `conftest.py` at each test directory level
- Mock external APIs (yfinance, NSE) in unit tests — use recorded fixtures
- NEVER mock the database in integration tests

### Financial Data Rules — CRITICAL
- ALL timestamps must be timezone-aware (Asia/Kolkata for Indian market data)
- NEVER use random train/test splits for time-series data
- ALWAYS use point-in-time data — no look-ahead bias
- Corporate action adjustments MUST be applied before feature computation
- Prices must be adjusted for splits, bonuses, rights issues
- Circuit-hit days must be flagged, not silently included
- Include delisted stocks in historical data (survivorship bias protection)

### ML Model Rules
- Every model must implement the `BaseModel` interface
- Models must be serializable (use `safetensors` for PyTorch, `joblib` for sklearn/xgboost)
- Every training run must log: features used, hyperparameters, train/val metrics, timestamp
- NEVER train on data that overlaps with validation (use purge + embargo)
- Minimum 20-day embargo between train and validation sets
- Use fractionally differentiated series, not raw prices
- All features must be computed using only past data at prediction time

### Database Rules
- Use TimescaleDB hypertables for all time-series tables
- Partition by time (monthly chunks)
- Index on (symbol, timestamp) for all market data tables
- Use connection pooling (asyncpg)
- Migrations via Alembic

### API Rules
- All endpoints return JSON with consistent schema
- Include `model_version` and `generated_at` in every prediction response
- Rate limit external data provider calls (respect their limits)
- Cache computed features in Redis (TTL: market hours = 5 min, after hours = until next open)

## Git Workflow

- Branch naming: `feature/xxx`, `fix/xxx`, `data/xxx`, `model/xxx`
- Commit messages: conventional commits (`feat:`, `fix:`, `data:`, `model:`, `test:`, `docs:`)
- Never commit model weights, data files, or credentials
- `.env` files are in `.gitignore` — use `.env.example` as template

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://alphavedha:password@localhost:5432/alphavedha
REDIS_URL=redis://localhost:6379/0

# Data Providers (optional — free sources work without keys)
FINNHUB_API_KEY=           # For news sentiment
MARKETAUX_API_KEY=         # For news sentiment (alternative)

# Broker API (Phase 3 — automated trading)
# KITE_API_KEY=
# KITE_API_SECRET=
```

## Domain Knowledge Reference

- **Indian market hours:** 9:15 AM - 3:30 PM IST (pre-open: 9:00-9:15)
- **Settlement:** T+1 (moving to T+0 for select stocks)
- **Circuit limits:** 5%, 10%, 20% on individual stocks; 10%, 15%, 20% on indices
- **F&O expiry:** Last Thursday of month (monthly), every Thursday (weekly for Nifty/BankNifty)
- **STT:** 0.1% on delivery buy+sell, 0.025% on intraday sell, 0.0125% on F&O
- **Nifty 50 rebalancing:** Semi-annual (March, September)
- **Key data sources:** yfinance (.NS suffix), jugaad-data (NSE), niftyindices.com (index compositions)
