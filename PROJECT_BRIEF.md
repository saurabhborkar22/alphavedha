# AlphaVedha — Project Brief & Session Context

> **Purpose:** Read this file at the start of every new Claude Code session. It gives full context on what's built, what's next, architecture decisions, and implementation details so you can continue without re-discovery.

---

## 1. What Is AlphaVedha?

An AI-powered Indian stock market prediction engine for NSE/BSE. Predicts stock direction, magnitude, confidence, and price target ranges using ensemble ML (XGBoost + LSTM + TFT) with 141 India-specific features.

- **Solo developer:** Saurabh Borkar (borkarsaurabh22@gmail.com)
- **Language:** Python 3.12, async-first
- **Repo:** `/home/lenovo/alphavedha/`
- **Build system:** Hatch (pyproject.toml)
- **Linter:** ruff (no TCH rule — removed as too pedantic)
- **Tests:** pytest with pytest-asyncio mode=auto

---

## 2. Build Roadmap (6 weeks)

| Week | Focus | Status |
|------|-------|--------|
| 1 | Data pipeline + preprocessing + DB | **DONE** |
| 2 | Feature engineering (all 141 features) | **NEXT** |
| 3 | Triple barrier labeling + XGBoost + CPCV validation + VectorBT backtest | Planned |
| 4 | LSTM + HMM regime + derivatives features + macro features | Planned |
| 5 | TFT + stacking ensemble + meta-labeling + conformal prediction + sentiment | Planned |
| 6 | FastAPI + risk management + MLOps monitoring + CLI + Docker + tests | Planned |

---

## 3. Week 1 — COMPLETED (Data Pipeline)

### Files created (all in `alphavedha/data/`):

```
database.py           — Async SQLAlchemy engine, session factory (asyncpg, pool=10, overflow=20)
models.py             — 6 ORM models: DailyOHLCV, CorporateAction, IndexConstituent, 
                        InstitutionalFlow, DerivativesData, Feature
store.py              — Feature store + OHLCV store with PostgreSQL upsert
universe.py           — Nifty 50/150/250 compositions from niftyindices.com, point-in-time

providers/
  base.py             — DataProvider protocol, RateLimiter (token bucket), fetch_with_retry(), validate_ohlcv()
  yfinance_provider.py — YFinanceProvider (.NS suffix, bulk via asyncio.Semaphore(5))
  jugaad_provider.py  — JugaadProvider (delivery_pct, 0.5 req/sec rate limit)

preprocessing/
  corporate_actions.py — Backward adjustment factors, raw_* column preservation
  circuit_handler.py   — 5%/10%/20% detection (largest-threshold-first)
  missing_data.py      — Forward-fill with is_filled flag, suspension detection
  fractional_diff.py   — FFD transform, auto min-d via ADF test (linear scan, not binary search)
  outlier_treatment.py — Winsorization (skips price and return columns)
  pipeline.py          — Orchestrates all 5 steps, returns PreprocessingResult
```

### Config: `alphavedha/config.py`
- Pydantic v2 nested BaseModel classes for every section of `configs/default.yaml`
- `get_config()` is cached singleton via `@functools.lru_cache`

### Tests: 49 unit tests, all passing (1.34s)
- `tests/unit/data/test_preprocessing.py` — 18 tests
- `tests/unit/data/test_providers.py` — 7 tests
- `tests/unit/data/test_universe.py` — 7 tests
- `tests/unit/data/test_pipeline.py` — 6 tests
- `tests/conftest.py` — 5 fixtures: sample_ohlcv (20d), with_gaps, with_circuit, long (252d), corporate_actions

### Bugs caught and fixed during Week 1:
- Circuit handler iterated smallest-threshold-first → 10% moves flagged as 5%. Fixed: `sorted(thresholds, reverse=True)`
- `frac_diff_dataframe` docstring said "binary search" but was linear scan. Fixed docstring.
- Double computation of find_min_d — pipeline called it again after frac_diff_dataframe already did. Fixed: return tuple `(df, dict[str, float])`.
- Dead dividend branch in corporate_actions logged "adjustment applied" but did nothing. Added `continue`.
- `pd.io.common.StringIO` is private API. Switched to `io.StringIO`.

### Git commits:
- `c66ba36` — feat: implement Week 1 data pipeline (25 files, 2634 insertions)
- `e48e99a` — docs: update README and data layer docs

---

## 4. Week 2 — Feature Engineering (141 Features)

### What to build

7 feature modules in `alphavedha/features/` + pipeline orchestrator. Each module is a standalone function that takes a preprocessed OHLCV DataFrame and returns a DataFrame of named feature columns.

### Module breakdown:

#### 4.1 technical.py — 40 features
Uses the `ta` library (already in pyproject.toml). All computed on adjusted close.

**Momentum (12):**
- RSI: windows 7, 14, 21 → `rsi_7`, `rsi_14`, `rsi_21`
- Stochastic: %K(14), %D(14) → `stoch_k_14`, `stoch_d_14`
- MACD: `macd_12_26`, `macd_signal_12_26`, `macd_hist_12_26`
- Williams %R: `willr_14`
- ROC: 10, 20 → `roc_10`, `roc_20`
- CCI: `cci_20`

**Trend (10):**
- SMA: 20, 50, 200 → `sma_20`, `sma_50`, `sma_200`
- EMA: 9, 21 → `ema_9`, `ema_21`
- Price ratios: `price_to_sma_20`, `price_to_sma_50`
- ADX: `adx_14`, `dip_14` (DI+), `dim_14` (DI-)

**Volatility (8):**
- Bollinger: `bb_upper_20`, `bb_lower_20`, `bb_width_20`, `bb_pct_20`
- ATR: `atr_14`, `natr_14` (normalized ATR = ATR/close × 100)
- Historical vol: `hvol_20`, `hvol_60` (annualized log-return std)

**Volume (10):**
- OBV: `obv`, `obv_ema_20`
- Volume: `vol_sma_20`, `vol_ratio_20`
- VWAP: `vwap_20` (rolling), `price_to_vwap_20`
- MFI: `mfi_14`
- A/D line: `ad`
- Chaikin Oscillator: `cho_3_10`
- Force Index: `fi_13`

**Key function signature:**
```python
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 40 technical indicators. Input must have OHLCV columns."""
```

#### 4.2 returns.py — 20 features
Log returns (NOT simple), fractionally differentiated series from preprocessing.

- Log returns: `ret_log_1d`, `ret_log_5d`, `ret_log_10d`, `ret_log_20d`
- Rolling stats: `ret_mean_5d`, `ret_mean_20d`, `ret_std_5d`, `ret_std_20d`
- Higher moments: `ret_skew_20d`, `ret_kurt_20d`
- Risk metrics: `ret_sharpe_20d`, `ret_max_dd_20d`
- Ratios: `ret_up_ratio_20d`
- Momentum: `ret_mom_5d`, `ret_mom_20d`, `ret_mom_60d`
- Z-score: `ret_zscore_20d`
- Frac-diff: `ret_frac_diff` (from preprocessing output)
- 52-week: `ret_52w_high_dist`, `ret_52w_low_dist`
- Regime: `ret_regime` (from HMM model when available, default=1 sideways)

**Key function signature:**
```python
def compute_return_features(
    df: pd.DataFrame,
    frac_diff_col: str | None = None,
) -> pd.DataFrame:
```

#### 4.3 calendar_features.py — 18 features
Pure date math, no external data (except monsoon = June-Sep flag).

- Time: `cal_dow` (0-4), `cal_month`, `cal_quarter`, `cal_week_of_month`
- F&O: `cal_days_to_monthly_expiry`, `cal_is_expiry_week`, `cal_is_expiry_day`
- Events: `cal_days_to_rbi` (bi-monthly RBI policy), `cal_is_budget_month` (February)
- Seasonal: `cal_is_january`, `cal_is_december`, `cal_monsoon_flag` (Jun-Sep)
- Results: `cal_is_result_season` (Jan/Apr/Jul/Oct)
- Calendar: `cal_doy`, `cal_year`, `cal_week_of_year`
- Holiday: `cal_is_monday`, `cal_days_in_quarter`

**Helper needed:** `last_thursday_of_month(year, month) -> date` for F&O expiry calculation.

**Key function signature:**
```python
def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 18 calendar features from DatetimeIndex."""
```

#### 4.4 microstructure.py — 10 features
India-specific delivery % signals. Requires `delivery_pct` column from jugaad-data.

- Delivery: `micro_delivery_pct`, `micro_delivery_zscore` (20d rolling z-score)
- Delivery trend: `micro_delivery_to_ma5`, `micro_delivery_trend_5d`, `micro_delivery_accel`
- Volume: `micro_vol_anomaly` (today/20d avg)
- Combined signals: `micro_hd_up` (high delivery + up move), `micro_hd_down`, `micro_ld_up`
- Rolling: `micro_delivery_rolling_10d`

**Graceful degradation:** If `delivery_pct` column is missing, return all zeros with a warning log.

**Key function signature:**
```python
def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
```

#### 4.5 macro.py — 25 features
Market-wide features. Data sources: yfinance for global (VIX, FX, commodities), `institutional_flows` table for FII/DII.

**Market data (fetched via yfinance, cached):**
- India VIX: `macro_vix`, `macro_vix_change_1d`, `macro_vix_zscore_20d`
- Nifty returns: `macro_nifty_ret_1d`, `macro_nifty_ret_5d`
- USD/INR: `macro_usdinr`, `macro_usdinr_change_1d`
- Brent crude: `macro_brent`, `macro_brent_change_1d`
- Gold: `macro_gold`
- Yields: `macro_gsec_10y`, `macro_gsec_change_1d`, `macro_us10y`

**FII/DII (from DB table `institutional_flows`):**
- `macro_fii_net`, `macro_dii_net`
- `macro_fii_cum_5d`, `macro_dii_cum_5d`

**Sector-relative (computed from OHLCV):**
- `macro_sector_ret_1d`, `macro_sector_rel_ret_1d`

**Monthly (forward-filled):**
- `macro_pmi`, `macro_pmi_staleness_days`
- `macro_breadth_200sma`, `macro_adv_dec_ratio`

**Design:** Two-layer approach:
1. `fetch_macro_data(start, end) -> pd.DataFrame` — fetches/caches market-wide data
2. `compute_macro_features(stock_df, macro_df, sector_df) -> pd.DataFrame` — merges and computes

#### 4.6 derivatives.py — 20 features
F&O data from `derivatives_data` table.

- Futures: `deriv_futures_oi`, `deriv_futures_oi_change`, `deriv_futures_premium`
- IV: `deriv_atm_iv` (Black-Scholes via scipy.optimize.brentq), `deriv_iv_rank`, `deriv_iv_pctile`
- PCR: `deriv_pcr_oi`, `deriv_pcr_vol`
- Max pain: `deriv_max_pain`, `deriv_dist_max_pain`
- Participant OI: `deriv_fii_futures_oi`, `deriv_fii_options_oi`, `deriv_pro_futures_net`, `deriv_retail_futures_net`
- OI interpretation: `deriv_oi_buildup`, `deriv_oi_unwind`, `deriv_short_cover`, `deriv_short_build`
- Greeks: `deriv_gex`, `deriv_delta_oi`

**Graceful degradation:** Stocks without F&O return NaN (filled by pipeline with market-level aggregates or 0).

**Black-Scholes IV helper:**
```python
def implied_volatility(market_price, S, K, T, r=0.065, option_type="call") -> float:
    """Compute IV using Brent's method. Returns NaN if no solution."""
```

#### 4.7 sentiment.py — 8 features
FinBERT for news sentiment. Model loaded lazily.

- `sent_news_score` (mean FinBERT score for day's articles)
- `sent_news_score_5d` (5-day MA)
- `sent_velocity`, `sent_velocity_zscore`
- `sent_article_count`
- `sent_no_news_flag` (1 if no articles)
- `sent_pos_ratio`, `sent_neg_ratio`

**Graceful degradation:** If transformers not installed or no news API key, return neutral (0.0) with `sent_no_news_flag=1`.

#### 4.8 features/pipeline.py — Orchestrator
Calls all 7 modules, concatenates, validates, stores to feature store.

```python
@dataclass
class FeatureResult:
    df: pd.DataFrame           # (n_dates × 141 features)
    symbol: str
    feature_count: int
    nan_count: int
    computation_time_ms: float

async def compute_all_features(
    symbol: str,
    ohlcv_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    derivatives_df: pd.DataFrame | None = None,
    news_articles: list[dict] | None = None,
    frac_diff_col: str | None = None,
    store_results: bool = True,
    feature_version: str = "v1",
) -> FeatureResult:
```

**Pipeline order:**
1. technical (40) — needs only OHLCV
2. returns (20) — needs OHLCV + optional frac_diff
3. calendar (18) — needs only DatetimeIndex
4. microstructure (10) — needs delivery_pct column
5. macro (25) — needs macro_df (market-wide data)
6. derivatives (20) — needs derivatives_df
7. sentiment (8) — needs news articles

**Validation after concat:**
- Assert 141 columns
- Replace inf with NaN, then forward-fill remaining NaN
- Log any columns with >50% NaN as warnings
- Store to feature store via `store.store_features()`

### Tests to write (in `tests/unit/features/`):

| Test file | Tests | Key assertions |
|-----------|-------|----------------|
| `test_technical.py` | ~12 | RSI in [0,100], BB upper > lower, ATR > 0, no look-ahead |
| `test_returns.py` | ~8 | Log return matches manual calc, rolling std > 0, sharpe computable |
| `test_calendar.py` | ~8 | Expiry day detection, monsoon flag Jun-Sep, DOW 0-4 |
| `test_microstructure.py` | ~6 | Z-score range, binary flags, graceful when no delivery_pct |
| `test_macro.py` | ~6 | Handles empty FII data, sector return correct, VIX features present |
| `test_derivatives.py` | ~6 | IV in [0.05, 2.0], PCR > 0, graceful with no data |
| `test_sentiment.py` | ~6 | Neutral when no news, scores in [-1,1], no_news_flag |
| `test_pipeline.py` | ~6 | 141 columns, no NaN after fill, no inf, timing logged |

### Conftest fixture needed:
```python
@pytest.fixture
def sample_ohlcv_with_delivery(sample_ohlcv_long):
    """252-day OHLCV with delivery_pct column for microstructure tests."""
    df = sample_ohlcv_long.copy()
    rng = np.random.default_rng(42)
    df["delivery_pct"] = rng.uniform(0.3, 0.8, size=len(df))
    return df
```

---

## 5. Architecture Decisions (locked in)

| Decision | Choice | Why |
|----------|--------|-----|
| Returns | Log returns, not simple | Additive across time, better statistical properties |
| Frac-diff | FFD with auto min-d via ADF | Preserve memory while achieving stationarity |
| Feature naming | `{group}_{indicator}_{window}` | Grep-friendly, group-sortable |
| NaN handling | Forward-fill after feature computation | Never let NaN reach the model |
| Look-ahead | Every feature at time T uses only data ≤ T | Enforced by rolling windows, tested explicitly |
| Circuit days | Flag, don't drop | Volume unreliable but price action informative |
| DB | Async SQLAlchemy + asyncpg | Consistent with data layer, non-blocking |
| Feature store | PostgreSQL JSON column | Simple, queryable, versioned |
| Config | Pydantic v2 + YAML | Type-safe, validated at startup |
| Logging | structlog | Structured, JSON-serializable |

---

## 6. Code Conventions (quick reference)

- `from __future__ import annotations` in every file
- Type hints on all function signatures
- Early returns to reduce nesting
- `structlog.get_logger(__name__)` for logging
- Custom exceptions in `alphavedha/exceptions.py`
- Import order: future → stdlib → third-party → local
- Column naming: `{group_prefix}_{indicator}` (e.g., `rsi_14`, `ret_log_1d`, `cal_dow`, `micro_delivery_zscore`)
- No mutable default arguments
- Context managers for DB sessions

---

## 7. Key File Locations

```
alphavedha/
├── CLAUDE.md                          # Root instructions (read first)
├── PROJECT_BRIEF.md                   # THIS FILE
├── configs/default.yaml               # All configuration
├── alphavedha/
│   ├── config.py                      # Pydantic config loader
│   ├── exceptions.py                  # Custom exceptions (create if missing)
│   ├── data/
│   │   ├── CLAUDE.md                  # Data layer docs
│   │   ├── database.py                # DB engine + sessions
│   │   ├── models.py                  # 6 ORM models
│   │   ├── store.py                   # Feature + OHLCV store (upsert)
│   │   ├── universe.py                # Index compositions
│   │   ├── providers/                 # yfinance, jugaad
│   │   └── preprocessing/            # 5-step pipeline
│   └── features/
│       ├── CLAUDE.md                  # Feature engineering docs
│       ├── __init__.py                # Exports
│       ├── technical.py               # 40 indicators (ta library)
│       ├── returns.py                 # 20 return features
│       ├── calendar_features.py       # 18 calendar features
│       ├── microstructure.py          # 10 delivery/volume features
│       ├── macro.py                   # 25 macro/market features
│       ├── derivatives.py             # 20 F&O features
│       ├── sentiment.py               # 8 news sentiment features
│       └── pipeline.py                # Orchestrator
├── tests/
│   ├── conftest.py                    # Shared fixtures
│   ├── unit/data/                     # 49 tests (passing)
│   └── unit/features/                 # Feature tests (Week 2)
└── docs/superpowers/specs/
    └── 2026-05-12-alphavedha-prediction-engine-design.md  # Full design spec
```

---

## 8. DB Models Reference

```python
# DailyOHLCV — (symbol, date) unique
#   open, high, low, close, adj_close, volume, delivery_pct, circuit_hit, is_adjusted, is_filled

# Feature — (symbol, date, feature_version) unique
#   feature_json: dict (all 141 features as JSON)

# InstitutionalFlow — (date, category) unique
#   category: "FII" or "DII", buy_value, sell_value, net_value

# DerivativesData — (symbol, date) unique
#   futures_oi, futures_price, options_data_json: dict

# IndexConstituent — index_name, symbol, effective_from, effective_to
# CorporateAction — symbol, ex_date, action_type, ratio, details
```

---

## 9. Running Things

```bash
# Activate venv
source .venv/bin/activate      # or create: python3 -m venv .venv && pip install -e ".[dev]"

# Run tests
pytest tests/unit/ -v           # All unit tests
pytest tests/unit/data/ -v      # Data layer only
pytest tests/unit/features/ -v  # Feature tests (Week 2)

# Lint
ruff check alphavedha/
ruff format alphavedha/

# Type check
# mypy alphavedha/              # Strict mode enabled in pyproject.toml
```

---

## 10. Weeks 3-6 Preview

### Week 3: Labeling + XGBoost + Validation
- `alphavedha/labels/` — Triple barrier method, meta-labeling, sample weights
- `alphavedha/models/xgboost_model.py` — XGBoost classifier
- `alphavedha/backtest/cpcv.py` — Combinatorial purged cross-validation (6 segments, k=2, 15 paths)
- `alphavedha/backtest/engine.py` — VectorBT with Indian market costs (STT, brokerage, GST, stamp duty)

### Week 4: LSTM + HMM + More Features
- `alphavedha/models/lstm_model.py` — 2-layer LSTM, 60-day sequence, top-30 features
- `alphavedha/models/regime.py` — 4-state HMM on Nifty returns + VIX
- Wire derivatives + macro features to live NSE data feeds

### Week 5: TFT + Ensemble + Conformal
- `alphavedha/models/tft_model.py` — Temporal Fusion Transformer (7d/15d/30d horizons)
- `alphavedha/models/ensemble.py` — Ridge stacking meta-learner
- `alphavedha/models/conformal.py` — MAPIE conformal prediction (90% coverage)
- Sentiment features with FinBERT

### Week 6: API + Risk + MLOps
- `alphavedha/api/` — FastAPI endpoints (predict, scan, health)
- `alphavedha/risk/` — Half-Kelly sizing, sector caps, drawdown circuit breakers
- `alphavedha/monitoring/` — PSI drift detection, model versioning, auto-retraining
- `alphavedha/cli/` — Typer CLI (/predict, /scan, /train, /backtest)
- Docker + docker-compose

---

## 11. Things to Watch Out For

1. **No look-ahead bias** — Every feature at time T uses only data ≤ T. Test this explicitly.
2. **NaN propagation** — Handle NaN explicitly in every module. Pipeline does final forward-fill.
3. **Rate limits** — yfinance (2/sec), NSE (0.5/sec), Finnhub (60/min). Always use RateLimiter.
4. **Circuit days** — Flag, never drop. Volume is unreliable on circuit days.
5. **Graceful degradation** — Macro, derivatives, sentiment modules must work when data is unavailable (return zeros/NaN).
6. **PEP 668** — System uses externally managed Python. Always use venv.
7. **pytest-asyncio mode=auto** — No need for `@pytest.mark.asyncio` decorator.
8. **ruff TCH rule removed** — Was flagging runtime imports as type-checking-only. Too pedantic.
9. **Don't commit spec docs** — User preference: don't commit design/spec documents without explicit approval.
