# Feature Engineering — AlphaVedha

## Responsibility
Compute all 141 features from raw market data. Features are grouped into 7 categories, each in its own module.

## ABSOLUTE RULES
1. **No future data.** Every feature at time T must use ONLY data available at or before T.
2. **No NaN propagation.** Handle NaN explicitly — forward-fill, default value, or exclude. Never let NaN silently enter the model.
3. **Deterministic.** Same input data → same features. No randomness in feature computation.
4. **Documented units.** Every feature must document its unit/scale (percentage, ratio, z-score, raw value).

## Feature Groups

### technical.py (40 features)
- Use `ta` library where possible, custom implementations only when `ta` doesn't support it
- All indicators computed on ADJUSTED close prices
- Multi-window indicators (RSI, SMA, EMA) compute all windows in one pass for efficiency
- Return a DataFrame with column naming: `{indicator}_{window}` (e.g., `rsi_14`, `sma_50`)

### derivatives.py (20 features)
- Source: NSE options chain and futures data
- Participant-wise OI from NSE daily reports
- IV computed from ATM options using Black-Scholes
- Handle stocks without F&O by filling with market-level aggregates
- Column naming: `deriv_{indicator}` (e.g., `deriv_iv_rank`, `deriv_fii_oi`)

### macro.py (25 features)
- Most macro features are market-wide (same value for all stocks on a given day)
- Sector-relative features are per-stock (stock return vs sector return)
- Monthly data (GST, PMI, auto sales): forward-fill to daily, add `macro_{name}_staleness_days` feature
- Column naming: `macro_{indicator}` (e.g., `macro_fii_net_flow`, `macro_vix`)

### microstructure.py (10 features)
- India-specific signals — this is our differentiator
- Delivery %: normalize as z-score against 20-day rolling mean
- Promoter/FPI data: quarterly, forward-fill between filings
- Bulk/block deals: binary flag + deal size as % of daily volume
- Column naming: `micro_{indicator}` (e.g., `micro_delivery_zscore`, `micro_promoter_pledge_pct`)

### sentiment.py (8 features)
- FinBERT model loaded once, shared across all stocks
- Batch process all news articles for the day, then aggregate per stock
- If no news for a stock on a day, use neutral sentiment (0.0) with a `no_news_flag`
- Social sentiment: aggregate from available sources, degrade gracefully if API is down
- Column naming: `sent_{indicator}` (e.g., `sent_news_score`, `sent_velocity_zscore`)

### calendar_features.py (18 features)
- Pure computation from dates — no external data needed (except monsoon from IMD)
- F&O expiry: compute using last Thursday rule, handle exchange holidays
- Corporate action calendar: earnings dates from BSE/NSE announcements
- Column naming: `cal_{indicator}` (e.g., `cal_days_to_expiry`, `cal_monsoon_flag`)

### returns.py (20 features)
- Log returns, NOT simple returns (log returns are additive across time)
- Fractionally differentiated series from preprocessing layer
- Regime label: output from HMM model (computed in models/regime.py, consumed here)
- Column naming: `ret_{indicator}` (e.g., `ret_log_1d`, `ret_frac_diff`, `ret_regime`)

## pipeline.py — Orchestrator
- Calls all 7 feature groups in order
- Concatenates into single DataFrame: (n_stocks × n_dates) × 141 features
- Validates: no NaN, no inf, correct dtypes, correct date alignment
- Writes to feature store
- Logs feature computation time and any warnings

## Adding New Features
1. Add to the appropriate group module
2. Update `configs/features.yaml` with feature metadata (name, window, source, description)
3. Add unit test verifying computation against known values
4. Run `make validate` to check impact on model performance
5. Update feature count in root CLAUDE.md
