# AlphaVedha (अल्फावेध)

AI-powered Indian stock market prediction engine for NSE/BSE.

Combines ensemble ML models (XGBoost + LSTM + TFT), India-specific microstructure signals, and quantitative finance techniques to predict stock direction, magnitude, and price targets.

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
make lint               # ruff check + mypy
make test               # pytest with coverage
make test-unit          # Unit tests only (no DB needed)
make data-refresh       # Fetch latest market data
make predict SYMBOL=TCS.NS   # Predict one stock
make scan TIER=large          # Scan Nifty 50
make serve              # Start FastAPI dev server
```

## Project Structure

```
alphavedha/
├── config.py             # Pydantic config loader (configs/default.yaml)
├── exceptions.py         # Custom exception hierarchy
├── data/                 # Data ingestion, preprocessing, storage
│   ├── providers/        # yfinance, jugaad-data fetchers
│   ├── preprocessing/    # Corporate actions, circuits, missing data, frac diff
│   ├── database.py       # Async SQLAlchemy engine + session
│   ├── models.py         # ORM models (6 tables)
│   ├── universe.py       # Nifty 50/150/250 index compositions
│   └── store.py          # Feature + OHLCV storage with upsert
├── features/             # Feature engineering (141 features, 7 groups)
├── labels/               # Triple barrier + meta-labeling
├── models/               # ML models (XGBoost, LSTM, TFT, HMM)
├── prediction/           # Prediction engine orchestrator
├── risk/                 # Risk management (Kelly, circuit breakers)
├── backtest/             # VectorBT backtesting + CPCV validation
├── monitoring/           # MLOps (drift detection, versioning)
├── api/                  # FastAPI endpoints
└── cli/                  # Typer CLI
```

## Build Progress

| Week | Focus | Status |
|------|-------|--------|
| 1 | Data pipeline + preprocessing + DB models | Done |
| 2 | Feature engineering (technical + returns + calendar) | Pending |
| 3 | Triple barrier labeling + XGBoost + CPCV + backtest | Pending |
| 4 | LSTM + HMM regime + derivatives + macro features | Pending |
| 5 | TFT + ensemble + meta-labeling + conformal + sentiment | Pending |
| 6 | FastAPI + risk management + MLOps + CLI + Docker | Pending |

## License

MIT
