# AlphaVedha — Database Schema Reference

PostgreSQL 16 + TimescaleDB. All timestamps are timezone-aware (Asia/Kolkata). Connection via asyncpg + SQLAlchemy async.

## Migration Chain

```
05c23a1b9653  initial schema (13 tables)
a1b2c3d4e5f6  convert 8 tables to TimescaleDB hypertables
6f2d6044726f  add 4 new tables (corporate_announcements, data_lineage, data_quality_reports, intraday_ohlcv)
b8c1d2e3f4a5  add is_tradeable column to paper_trades
```

## Connection Pool (database.py)

| Param | Default | Env override |
|---|---|---|
| pool_size | 10 | DB_POOL_SIZE |
| max_overflow | 20 | DB_MAX_OVERFLOW |
| pool_timeout | 30s | DB_POOL_TIMEOUT |
| pool_recycle | 1800s | DB_POOL_RECYCLE |
| pool_pre_ping | True | — |
| max concurrent connections | 30 | — |

Session factory: `async_sessionmaker(expire_on_commit=False)`. Engine singleton pattern — created once on first call.

---

## Tables

### 1. daily_ohlcv ⚡ HYPERTABLE
Partition by: `date` | Chunk: 1 month | Compress: segments=symbol, order=date DESC, after 6 months

| Column | Type | Nullable | Notes |
|---|---|---|---|
| symbol | String(20) | NO | PK (composite) |
| date | Date | NO | PK (composite) |
| open | Float | NO | |
| high | Float | NO | |
| low | Float | NO | |
| close | Float | NO | |
| adj_close | Float | NO | Corporate-action-adjusted close |
| volume | Integer | NO | |
| delivery_pct | Float | YES | NSDL delivery % from NSE bhavcopy |
| circuit_hit | String(10) | YES | "upper" / "lower" / NULL |
| is_adjusted | Boolean | NO | False = raw, True = adjusted |
| is_filled | Boolean | NO | True = forward-filled from prior day (holiday) |
| created_at | DateTime | NO | now() |

Index: `ix_daily_ohlcv_date` on (date DESC)

---

### 2. features ⚡ HYPERTABLE
Partition by: `date` | Chunk: 1 month | Compress: segments=symbol, order=date DESC, after 3 months

| Column | Type | Nullable | Notes |
|---|---|---|---|
| symbol | String(20) | NO | PK (composite) |
| date | Date | NO | PK (composite) |
| feature_version | String(20) | NO | PK (composite) e.g. "v1" |
| feature_json | JSON | NO | Full 164-feature dict for that symbol-date |
| created_at | DateTime | NO | now() |

Upsert on conflict (symbol, date, feature_version). Load/store via `store.py:store_features()` / `load_features()`.

---

### 3. derivatives_data ⚡ HYPERTABLE
Partition by: `date` | Chunk: 1 month

| Column | Type | Nullable | Notes |
|---|---|---|---|
| symbol | String(20) | NO | PK (composite) |
| date | Date | NO | PK (composite) |
| futures_oi | Integer | YES | Near-month futures open interest |
| futures_price | Float | YES | Near-month futures price |
| options_data_json | JSON | YES | Full options chain snapshot |
| created_at | DateTime | NO | now() |

---

### 4. institutional_flows ⚡ HYPERTABLE
Partition by: `date` | Chunk: 1 month

| Column | Type | Nullable | Notes |
|---|---|---|---|
| date | Date | NO | PK (composite) |
| category | String(10) | NO | PK (composite) — "FII" or "DII" |
| buy_value | Float | NO | Daily buy in crore INR |
| sell_value | Float | NO | Daily sell in crore INR |
| net_value | Float | NO | buy_value - sell_value |
| created_at | DateTime | NO | now() |

Index: `ix_institutional_flows_date` on (date DESC)

---

### 5. daily_pnl ⚡ HYPERTABLE
Partition by: `date` | Chunk: 1 month

| Column | Type | Nullable | Notes |
|---|---|---|---|
| date | Date | NO | PK |
| portfolio_value | Float | NO | Total paper portfolio value |
| daily_return | Float | NO | Day return fraction |
| cumulative_return | Float | NO | Return since inception |
| n_positions | Integer | NO | Open positions count |
| n_correct | Integer | NO | Correct directional predictions today |
| n_total_predictions | Integer | NO | Total predictions today |
| benchmark_return | Float | NO | 0.0 — Nifty 50 return same day |
| created_at | DateTime | NO | now() |

---

### 6. paper_trades ⚡ HYPERTABLE
Partition by: `prediction_date` | Chunk: 1 month

| Column | Type | Nullable | Notes |
|---|---|---|---|
| symbol | String(20) | NO | PK (composite) |
| prediction_date | Date | NO | PK (composite) — date prediction was made (before open) |
| predicted_direction | Integer | NO | 1=up, -1=down, 0=neutral |
| predicted_magnitude | Float | NO | Expected % move |
| confidence | Float | NO | Ensemble meta-model confidence 0–1 |
| model_version | String(50) | NO | e.g. "ensemble_v1.2" |
| regime | String(20) | YES | HMM regime at prediction time |
| is_tradeable | Boolean | YES | Meta-labeling gate decision. NULL for pre-migration rows |
| entry_price | Float | YES | Filled at open if tradeable |
| exit_price | Float | YES | Filled after holding period |
| actual_return | Float | YES | Realized return |
| is_correct | Boolean | YES | True if direction matched |
| created_at | DateTime | NO | now() |

Indexes: `ix_paper_trades_date` on (prediction_date DESC), `ix_paper_trades_symbol` on (symbol)

**Critical note:** `is_tradeable` distinguishes "all predictions" from "trades the strategy actually takes." Without it the track record can't split the 3 tracks (all / gate_passed / top_k).

---

### 7. news_articles ⚡ HYPERTABLE
Partition by: `published_date` | Chunk: 1 month

| Column | Type | Nullable | Notes |
|---|---|---|---|
| content_hash | String(64) | NO | PK (composite) — SHA-256 dedup key |
| published_date | Date | NO | PK (composite) |
| symbol | String(20) | YES | NULL for market-wide news |
| source | String(50) | NO | Finnhub / MarketAux etc. |
| title | String(500) | NO | |
| url | String(1000) | YES | |
| sentiment_score | Float | YES | FinBERT score -1 to +1 |
| created_at | DateTime | NO | now() |

Indexes: `ix_news_articles_symbol_date` on (symbol, published_date DESC), `ix_news_articles_date` on (published_date DESC)
Upsert conflict on (content_hash, published_date) — updates only sentiment_score.

---

### 8. insider_trades ⚡ HYPERTABLE
Partition by: `trade_date` | Chunk: 1 month

| Column | Type | Nullable | Notes |
|---|---|---|---|
| symbol | String(20) | NO | PK (composite) |
| trade_date | Date | NO | PK (composite) |
| person_name | String(200) | NO | PK (composite) — SAST filer name |
| person_category | String(100) | YES | e.g. "Promoter", "Director" |
| trade_type | String(10) | NO | "buy" or "sell" |
| shares | Integer | NO | |
| value_lakhs | Float | NO | 0.0 — Trade value in lakh INR |
| created_at | DateTime | NO | now() |

Index: `ix_insider_trades_symbol` on (symbol). Note: append-only, no upsert.

---

### 9. corporate_actions (not a hypertable)

| Column | Type | Nullable | Notes |
|---|---|---|---|
| id | Integer | NO | PK autoincrement |
| symbol | String(20) | NO | |
| ex_date | Date | NO | Ex-date of action |
| action_type | String(20) | NO | "split", "bonus", "dividend", "rights" |
| ratio | Float | NO | Adjustment ratio |
| details | String(500) | YES | Human-readable description |
| created_at | DateTime | NO | now() |

Unique: (symbol, ex_date, action_type). Index: `ix_corporate_actions_symbol`.

---

### 10. index_constituents (not a hypertable)

| Column | Type | Nullable | Notes |
|---|---|---|---|
| id | Integer | NO | PK autoincrement |
| index_name | String(50) | NO | "NIFTY 50" / "NIFTY MIDCAP 150" / "NIFTY SMALLCAP 250" |
| symbol | String(20) | NO | NSE symbol without .NS suffix |
| company_name | String(200) | YES | |
| sector | String(100) | YES | Sector from niftyindices.com CSV |
| effective_from | Date | NO | Date composition was fetched |
| effective_to | Date | YES | NULL = currently active |
| created_at | DateTime | NO | now() |

Indexes: `ix_index_constituents_lookup` on (index_name, effective_from), `ix_index_constituents_symbol` on (symbol)
Point-in-time query: `effective_from <= as_of AND (effective_to IS NULL OR effective_to >= as_of)` — survivorship-bias-free.

---

### 11. earnings_results (not a hypertable)

| Column | Type | Nullable | Notes |
|---|---|---|---|
| id | Integer | NO | PK autoincrement |
| symbol | String(20) | NO | |
| quarter | Integer | NO | 1–4 |
| year | Integer | NO | Fiscal year |
| revenue_actual | Float | YES | |
| revenue_estimate | Float | YES | Analyst consensus |
| revenue_surprise_pct | Float | YES | (actual-estimate)/estimate |
| profit_actual | Float | YES | Net profit |
| profit_estimate | Float | YES | |
| profit_surprise_pct | Float | YES | |
| expenses | Float | YES | Total expenses |
| announced_date | Date | YES | |
| created_at | DateTime | NO | now() |

Unique: (symbol, quarter, year). Indexes: `ix_earnings_results_symbol`, `ix_earnings_results_announced`.

---

### 12. promoter_holdings (not a hypertable)

| Column | Type | Nullable | Notes |
|---|---|---|---|
| id | Integer | NO | PK autoincrement |
| symbol | String(20) | NO | |
| quarter_end | Date | NO | Quarter-end (SEBI filing date) |
| promoter_pct | Float | NO | % held by promoters |
| pledge_pct | Float | NO | 0.0 — % of promoter holding pledged |
| public_pct | Float | NO | 0.0 |
| fii_pct | Float | NO | 0.0 |
| dii_pct | Float | NO | 0.0 |
| created_at | DateTime | NO | now() |

Unique: (symbol, quarter_end). Indexes: `ix_promoter_holdings_symbol`, `ix_promoter_holdings_quarter`.

---

### 13. alternative_data (not a hypertable)

| Column | Type | Nullable | Notes |
|---|---|---|---|
| id | Integer | NO | PK autoincrement |
| data_type | String(50) | NO | "auto_sales", "cement_production", "pmi", "credit_growth" |
| period_date | Date | NO | Month-end date |
| value | Float | NO | |
| yoy_change | Float | YES | YoY % change |
| sector | String(100) | YES | |
| source | String(100) | YES | |
| created_at | DateTime | NO | now() |

Unique: (data_type, period_date). Index: `ix_alternative_data_type_date`.

---

### 14. corporate_announcements (not a hypertable) — added D7

| Column | Type | Nullable | Notes |
|---|---|---|---|
| id | Integer | NO | PK autoincrement |
| symbol | String(20) | NO | |
| announced_date | Date | NO | BSE/NSE announcement date |
| ex_date | Date | YES | For dividend/split actions |
| event_type | String(20) | NO | "dividend", "agm", "board_meeting", "results" |
| description | String(500) | NO | |
| created_at | DateTime | NO | now() |

Unique: (symbol, announced_date, event_type). Indexes: `ix_corp_ann_symbol`, `ix_corp_ann_date`.

---

### 15. data_lineage (not a hypertable) — added D7

| Column | Type | Nullable | Notes |
|---|---|---|---|
| id | Integer | NO | PK autoincrement |
| symbol | String(20) | YES | NULL for index-level data |
| date | Date | NO | |
| table_name | String(50) | NO | Which table was written |
| provider | String(50) | NO | "yfinance", "nse", "jugaad" |
| fetched_at | DateTime | NO | Wall-clock time of fetch |
| row_count | Integer | NO | Rows written in that batch |
| created_at | DateTime | NO | now() |

Index: `ix_data_lineage_symbol_date` on (symbol, date).

---

### 16. data_quality_reports (not a hypertable) — added D7

| Column | Type | Nullable | Notes |
|---|---|---|---|
| id | Integer | NO | PK autoincrement |
| symbol | String(20) | YES | NULL for portfolio-level checks |
| report_date | Date | NO | |
| check_type | String(30) | NO | "completeness" / "freshness" / "consistency" / "anomaly" |
| passed | Boolean | NO | |
| severity | String(10) | NO | "ok" / "warning" / "critical" |
| detail | String(1000) | NO | Human-readable explanation |
| created_at | DateTime | NO | now() |

Indexes: `ix_dqr_date` on (report_date), `ix_dqr_symbol` on (symbol).

---

### 17. intraday_ohlcv (not a hypertable) — added D7

| Column | Type | Nullable | Notes |
|---|---|---|---|
| symbol | String(20) | NO | PK (composite) |
| date | Date | NO | PK (composite) — one row per symbol per day |
| open | Float | NO | Captured at first tick |
| high | Float | NO | Running day high (GREATEST on upsert) |
| low | Float | NO | Running day low (LEAST on upsert) |
| last_price | Float | NO | Last traded price |
| volume | Integer | NO | Latest total volume |
| tick_count | Integer | NO | 0 — number of poll ticks so far |
| last_updated | DateTime | NO | IST timestamp of last poll |
| created_at | DateTime | NO | now() |

---

## TimescaleDB Summary

| Table | Time Column | Chunk | Compress Segments | Compress After |
|---|---|---|---|---|
| daily_ohlcv | date | 1 month | symbol | 6 months |
| features | date | 1 month | symbol | 3 months |
| derivatives_data | date | 1 month | — | — |
| institutional_flows | date | 1 month | — | — |
| daily_pnl | date | 1 month | — | — |
| paper_trades | prediction_date | 1 month | — | — |
| news_articles | published_date | 1 month | — | — |
| insider_trades | trade_date | 1 month | — | — |

Extension: `CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE`. All conversions use `migrate_data => TRUE`. No continuous aggregates defined.

---

## Data Ingestion Pipeline

### OHLCV (daily)
1. `get_symbols_for_tier(tier)` → queries `index_constituents` with point-in-time filter
2. `YFinanceProvider.fetch_ohlcv(symbol+".NS", start, end)` — rate limited: 2 req/s, 3 retries with exponential backoff
3. Preprocessing pipeline (in order):
   - `corporate_actions.py` — applies split/bonus/rights/dividend adjustments backwards; stores both raw + adj_close
   - `circuit_handler.py` — sets `circuit_hit` flag (upper/lower/None); does NOT drop circuit days
   - `missing_data.py` — forward-fills market holidays (volume=0, `is_filled=True`); no interpolation ever; alert if gap > 10 days
   - `fractional_diff.py` — skipped during ingestion; runs at feature-compute time
   - `outlier_treatment.py` — skipped during ingestion; winsorizes features at [1%, 99%]
4. `store_ohlcv()` — upsert via `ON CONFLICT (symbol, date) DO UPDATE`
5. `_write_lineage()` — records audit trail in `data_lineage`

Concurrency: `asyncio.Semaphore(3)` — max 3 symbols fetched simultaneously.

### Other ingestion jobs
| Job | Provider | Rate limit | Upsert key |
|---|---|---|---|
| FII/DII | NSE | 0.5 req/s | (date, category) |
| Derivatives (F&O) | NSE | 0.5 req/s, sequential | (symbol, date) |
| Earnings | Screener.in | sequential | (symbol, quarter, year) |
| BSE Announcements | BSE bulk | — | ON CONFLICT DO NOTHING |
| Google Trends | Google Trends | — | NOT persisted to DB — in-memory at feature time |

---

## Universe Management

Sources: niftyindices.com CSV downloads

| Tier | Index Name | Count |
|---|---|---|
| large | NIFTY 50 | 50 |
| mid | NIFTY MIDCAP 150 | 150 |
| small | NIFTY SMALLCAP 250 | 250 |

Point-in-time query: `effective_from <= as_of AND (effective_to IS NULL OR effective_to >= as_of)` — prevents survivorship bias.

Nifty 50 rebalances semi-annually (March, September). Requires manual `refresh_universe()` to capture additions/removals.

---

## Live Feed (live_feed.py)

- `LiveDataPoller` polls `yfinance.Ticker.fast_info` every 120s via `asyncio.to_thread`
- Gate: `is_market_open()` — IST 09:15–15:30, Mon–Fri
- Upsert: `GREATEST(high)`, `LEAST(low)`, replace last_price/volume, increment tick_count
- Zero price guard: skips upsert if last_price == 0
- Cache invalidation: every 5th tick deletes `predict:{symbol}:*` Redis keys

---

## Backtest Cost Model (costs.py)

| Cost Component | Rate |
|---|---|
| STT (delivery) | 0.1% buy+sell |
| Brokerage | ₹20 flat per order |
| Exchange transaction | 0.00345% per order |
| GST | 18% on (brokerage + exchange_txn) |
| SEBI turnover | 0.0001% per order |
| Stamp duty | 0.015% buy side only |
| Slippage — large cap | 0.1% |
| Slippage — mid cap | 0.3% |
| Slippage — small cap | 0.5% |

Reference value for round-trip cost %: ₹1,00,000.

## CPCV Splits (cpcv.py)

- n_segments=6 → C(6,2)=15 test paths
- k_test_segments=2
- purge_days=20 before each test segment
- embargo_days=20 after each test segment
- Acceptance: median_sharpe >= 0.8 AND worst_sharpe >= 0.3

---

## Fundamental Analysis (in-memory, not stored)

### Beneish M-Score — 8 indices
`M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI`

| Index | Signals Manipulation When |
|---|---|
| DSRI (Receivables/Sales ratio change) | High — receivables growing faster than sales |
| GMI (Gross Margin Index) | > 1 — margins deteriorating |
| AQI (Asset Quality Index) | High — intangibles/off-balance-sheet growing |
| SGI (Sales Growth Index) | High — fast growth → higher manipulation risk |
| DEPI (Depreciation Index) | > 1 — depreciation rate slowing |
| SGAI (SGA Expense Index) | > 1 — SGA growing faster than revenue |
| TATA (Total Accruals/Total Assets) | High — net income diverging from OCF |
| LVGI (Leverage Index) | > 1 — leverage increasing |

Thresholds: M > -1.78 = manipulator, M between -2.22 and -1.78 = grey_zone, M ≤ -2.22 = safe.

### Altman Z'-Score — 4 components (non-manufacturing variant)
`Z' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4`
Thresholds: Z' > 2.60 = safe, 1.10–2.60 = grey_zone, ≤ 1.10 = distress.

---

## Stock Graph (stock_graph.py)

Used by GNN model. Three edge types:

| Type | Code | Description | Weight |
|---|---|---|---|
| SECTOR | 0 | Same sector (from stocks.yaml) | 1.0 binary |
| CORRELATION | 1 | abs(Pearson) >= 0.6 on returns | abs(corr) |
| PROMOTER | 2 | Same promoter group (from stocks.yaml) | 1.0 binary |

5 promoter groups: tata (TCS/TATAMOTORS/TATASTEEL/TATACONSUM), adani (ADANIENT/ADANIPORTS), bajaj (BAJFINANCE/BAJAJFINSV/BAJAJ-AUTO), mahindra (M&M/TECHM), birla (HINDALCO/GRASIM/ULTRACEMCO).
