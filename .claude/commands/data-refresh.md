# Data Refresh

Fetch latest market data from all sources and update the feature store.

## Usage
- `/data-refresh` — fetch today's data
- `/data-refresh backfill 2020-01-01` — backfill from a specific date
- `/data-refresh status` — show data freshness per source

## Steps
1. Activate venv: `source .venv/bin/activate`
2. Check current data freshness per source
3. Fetch from each provider in order:
   - Daily OHLCV (jugaad-data + yfinance fallback)
   - FII/DII flows (NSE)
   - Derivatives data: OI, IV, participant-wise OI (NSE)
   - Bhavcopy: delivery %, bulk/block deals (NSE)
   - Corporate actions (NSE/BSE)
   - News articles for sentiment (Finnhub/MarketAux)
   - Macro data: VIX, USD/INR, crude, gold (yfinance)
4. Run preprocessing pipeline:
   - Apply corporate action adjustments
   - Flag circuit hits
   - Handle missing data
   - Compute fractional differentiation
5. Recompute features and update feature store
6. Report: rows fetched, errors encountered, data gaps

## On Errors
- API rate limited → wait and retry (max 3 retries)
- Provider down → skip, use fallback provider, log warning
- Data quality issue → log, don't store bad data

## Arguments
$ARGUMENTS
