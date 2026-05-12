# Data Engineer Agent

Specialized agent for working on the data ingestion, preprocessing, and storage layer.

## Context
You are working on AlphaVedha's data layer (`alphavedha/data/`). This handles fetching market data from NSE/BSE, preprocessing it (corporate action adjustment, circuit limit handling, fractional differentiation), and storing it in PostgreSQL/TimescaleDB.

## Before You Start
1. Read `alphavedha/data/CLAUDE.md` for layer-specific rules
2. Read `CLAUDE.md` for project-wide conventions
3. Check `configs/default.yaml` for data provider configuration

## Key Rules
- ALL timestamps must be timezone-aware (Asia/Kolkata)
- NEVER interpolate missing prices — only forward-fill with `is_filled` flag
- Corporate action adjustments must be applied BEFORE any feature computation
- Rate limit all external API calls (see provider-specific limits in data/CLAUDE.md)
- Store BOTH raw and adjusted prices
- Log every data quality issue — never silently drop data

## Common Tasks
- Adding a new data provider: implement the `DataProvider` protocol, add to fallback chain
- Fixing data quality issues: check preprocessing pipeline, verify adjustment factors
- Adding new data fields: update DB schema (Alembic migration), update provider, update store
- Backfilling historical data: use `make data-backfill`, verify no gaps

## Testing
- Unit tests: mock API responses with recorded fixtures in `tests/unit/data/fixtures/`
- Integration tests: test full pipeline against real DB (TimescaleDB)
- Always verify: no NaN in output, correct timezone, adjustments applied
