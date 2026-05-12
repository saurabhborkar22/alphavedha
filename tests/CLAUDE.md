# Tests — AlphaVedha

## Structure
```
tests/
├── conftest.py              # Shared fixtures (DB connection, sample data, mock providers)
├── unit/                    # Fast, no external deps
│   ├── data/                # Data provider and preprocessing tests
│   ├── features/            # Feature computation tests
│   ├── labels/              # Triple barrier and meta-labeling tests
│   ├── models/              # Model interface and prediction shape tests
│   ├── prediction/          # Prediction engine orchestration tests
│   ├── risk/                # Risk management logic tests
│   └── api/                 # API endpoint tests (mocked prediction engine)
├── integration/             # Requires DB, slower
│   ├── data/                # Full data pipeline: fetch → preprocess → store
│   ├── features/            # Feature store read/write consistency
│   └── api/                 # API with real prediction engine
└── backtest/                # Strategy validation
    ├── test_no_lookahead.py # Verify no future data leakage
    ├── test_costs.py        # Verify all Indian market costs included
    ├── test_survivorship.py # Verify delisted stocks included
    └── test_cpcv.py         # Verify CPCV validation protocol
```

## Running Tests
```bash
pytest tests/unit/ -v                          # Unit tests (fast, no deps)
pytest tests/integration/ -v                    # Integration (needs DB)
pytest tests/backtest/ -v                       # Backtest validation
pytest tests/ -v --cov=alphavedha --cov-report=term-missing  # Full suite with coverage
```

## Rules
- Unit tests: mock ALL external APIs (yfinance, NSE, Finnhub)
- Integration tests: use real PostgreSQL/TimescaleDB (from docker-compose)
- NEVER mock the database in integration tests
- Use `pytest-asyncio` with mode `auto`
- Fixtures for sample data: use real market data from known dates for reproducibility
- Every bug fix must include a regression test

## Key Test Scenarios
1. **Look-ahead bias detection**: verify features at time T don't use data > T
2. **Corporate action adjustment**: verify prices are correctly adjusted for known splits
3. **Circuit limit handling**: verify circuit days are flagged, not dropped
4. **Model serialization**: save model, load model, verify identical predictions
5. **API response schema**: verify all required fields present in every response
6. **Triple barrier labels**: verify labels match manual calculation on known price paths
7. **Feature store consistency**: features computed and stored match features retrieved

## Coverage Target
- Minimum 80% line coverage
- 100% coverage on: labels/, risk/, backtest/costs.py
