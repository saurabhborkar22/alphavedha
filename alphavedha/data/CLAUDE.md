# Data Layer — AlphaVedha

## Responsibility
Ingestion, preprocessing, and storage of all market data. This layer is the foundation — every model depends on clean, correctly adjusted, point-in-time data.

## Architecture

```
database.py       → Async SQLAlchemy engine, session factory, health check
models.py         → ORM models (DailyOHLCV, CorporateAction, IndexConstituent, etc.)
providers/        → Fetch raw data from external sources
  base.py         → DataProvider protocol, RateLimiter, validate_ohlcv, retry logic
  yfinance_provider.py  → Yahoo Finance (.NS suffix, bulk download)
  jugaad_provider.py    → NSE daily data with delivery %
preprocessing/    → Clean, adjust, transform raw data
  corporate_actions.py  → Split/bonus/rights adjustment, raw price preservation
  circuit_handler.py    → 5%/10%/20% circuit detection
  missing_data.py       → Forward-fill with flags, suspension detection
  fractional_diff.py    → FFD transform, auto min-d via ADF test
  outlier_treatment.py  → Winsorization (skips prices/returns)
  pipeline.py           → Orchestrates all 5 steps in correct order
universe.py       → Manage point-in-time index constituents
store.py          → Feature store + OHLCV store with upsert support
```

## Providers

Each provider implements the `DataProvider` protocol (see `providers/base.py`):

```python
class DataProvider(Protocol):
    @property
    def name(self) -> str: ...
    async def fetch_ohlcv(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...
    async def fetch_bulk(self, symbols: list[str], start: date, end: date) -> dict[str, FetchResult]: ...
    async def health_check(self) -> bool: ...
```

### Provider priority (fallback chain)
1. **jugaad-data** — primary for NSE daily data (most reliable for Indian markets)
2. **yfinance** — fallback, also primary for 20+ year historical backfill
3. **nse_provider** — for FII/DII, bhavcopy, OI data, corporate actions (direct NSE)
4. **news_provider** — Finnhub/MarketAux for sentiment data

### Rate Limiting
- yfinance: max 2 requests/second, with exponential backoff
- NSE website: max 1 request/2 seconds, rotate user-agents, respect 403s
- Finnhub free: 60 calls/minute
- Rate limiter: in-memory token bucket (`RateLimiter` in `base.py`), configurable via `configs/default.yaml`
- All providers use `fetch_with_retry()` for automatic retry with exponential backoff (3 retries default)

## Preprocessing — CRITICAL RULES

### Corporate Action Adjustment (corporate_actions.py)
- MUST adjust historical prices for: stock splits, bonus issues, rights issues, dividends
- Use adjustment factors from BSE/NSE, cross-validated with yfinance adjusted close
- Apply adjustments BACKWARDS from current price (standard method)
- Log every adjustment applied with details
- Store BOTH raw and adjusted prices (raw for audit, adjusted for features)

### Circuit Limit Handling (circuit_handler.py)
- Detect circuit hits: compare day's range to previous close ± circuit %
- Add `circuit_hit` column: "upper", "lower", or None
- Circuit-hit days get a flag feature — do NOT exclude them from data
- Volume on circuit days is unreliable — flag but don't drop

### Missing Data (missing_data.py)
- Market holidays: expected gaps, forward-fill prices, set volume to 0
- Suspended stocks: flag with `is_suspended=True`, do NOT interpolate
- Data provider outages: retry 3x with backoff, then log gap and continue
- NEVER interpolate prices — only forward-fill with a `is_filled` flag

### Fractional Differentiation (fractional_diff.py)
- Compute minimum d per stock that passes ADF test (p < 0.05)
- Typical range: d ~ 0.3 to 0.5
- Store optimal d per symbol in config, recompute monthly
- Use fixed-width window (max 100 lags) to avoid infinite memory

### Outlier Treatment (outlier_treatment.py)
- Winsorize features at 1st and 99th percentile
- Do NOT winsorize prices or returns — only computed features
- Log outlier counts per feature for drift monitoring

## Universe Manager (universe.py)
- Fetch current Nifty 50, Midcap 150, Smallcap 250 compositions from niftyindices.com
- Store historical compositions with effective dates (point-in-time)
- When computing features for a past date, use the index composition AS OF that date
- Track additions/removals for rebalancing signals

## Feature Store (store.py)
- PostgreSQL-backed feature store
- Ensures identical features in training and serving (no training-serving skew)
- Key: (symbol, timestamp, feature_version)
- TTL: recompute daily after market close
- Batch compute: process all symbols in universe after each market close

## Data Tables (TimescaleDB)

```sql
-- Daily OHLCV (hypertable, chunked monthly)
daily_ohlcv (symbol, date, open, high, low, close, adj_close, volume,
             delivery_pct, circuit_hit, is_adjusted, is_filled)

-- Features (hypertable)
features (symbol, date, feature_version, feature_json)

-- Corporate actions
corporate_actions (symbol, ex_date, action_type, ratio, details)

-- Universe compositions (point-in-time)
index_constituents (index_name, symbol, effective_from, effective_to)

-- FII/DII flows
institutional_flows (date, category, buy_value, sell_value, net_value)

-- Derivatives data
derivatives_data (symbol, date, futures_oi, futures_price, options_data_json)
```

## Database Layer

### Engine (`database.py`)
- Async SQLAlchemy with asyncpg (connection pooling: 10 pool, 20 overflow)
- `get_engine()` / `get_session_factory()` — module-level singletons
- `DATABASE_URL` env var or auto-constructed from config
- `create_tables()` for dev/testing, Alembic for production migrations

### ORM Models (`models.py`)
All tables have `created_at` with server default, proper indexes, and unique constraints.
Key: every time-series query should use the `(symbol, date)` composite index.

## Testing
- Unit test each provider with recorded API response fixtures
- Integration test the full pipeline: fetch → preprocess → store → retrieve
- Validate: no future data leakage, no unadjusted prices in feature store
- Test circuit limit detection against known historical circuit hits

### Existing Tests (tests/unit/data/)
- `test_providers.py` — OHLCV validation, rate limiter, mocked yfinance provider
- `test_preprocessing.py` — Corporate actions, circuit detection, missing data, fractional diff, outliers
- `test_pipeline.py` — End-to-end preprocessing pipeline orchestration
- `test_universe.py` — Index URLs, config loading and validation

### Test Fixtures (tests/conftest.py)
- `sample_ohlcv` — 20 trading days of realistic TCS data
- `sample_ohlcv_with_gaps` — OHLCV with 3 missing days
- `sample_ohlcv_with_circuit` — OHLCV with simulated 5% upper circuit
- `sample_ohlcv_long` — 252 trading days for fractional diff tests
- `sample_corporate_actions` — Split + bonus action records
