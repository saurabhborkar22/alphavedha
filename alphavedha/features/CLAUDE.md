# Feature Engineering — AlphaVedha

## Responsibility
Compute all 154 features from raw market data. Features are grouped into 8 categories, each in its own module.

## ABSOLUTE RULES
1. **No future data.** Every feature at time T must use ONLY data available at or before T.
2. **No NaN propagation.** Handle NaN explicitly — forward-fill, default value, or exclude. Never let NaN silently enter the model.
3. **Deterministic.** Same input data → same features. No randomness in feature computation.
4. **Documented units.** Every feature must document its unit/scale in `configs/features.yaml`.

## Architecture

```
features/
├── __init__.py            # Exports: FeatureResult, compute_all_features
├── technical.py           # 40 features (ta library): RSI, MACD, BB, ATR, OBV, ADX, etc.
├── returns.py             # 21 features: log returns, rolling stats, momentum, regime
├── calendar_features.py   # 18 features: F&O expiry, monsoon, RBI, result season
├── microstructure.py      # 13 features: delivery %, z-score, volume anomaly, breakout
├── macro.py               # 25 features: VIX, FII/DII, FX, commodities, sector-relative
├── derivatives.py         # 20 features: futures OI, IV (Black-Scholes), PCR, max pain
├── sentiment.py           # 8 features: FinBERT news scores, velocity, article count
└── pipeline.py            # Orchestrator: concat, validate, fill NaN, FeatureResult
```

## Feature Groups

### technical.py (40 features)
- Uses `ta` library (ta.momentum, ta.trend, ta.volatility, ta.volume)
- Custom: historical volatility (hvol), rolling VWAP, Chaikin Oscillator
- ADX guarded for short data (needs > 28 rows)
- **Momentum (12):** rsi_{7,14,21}, stoch_{k,d}_14, macd_12_26, macd_signal/hist, willr_14, roc_{10,20}, cci_20
- **Trend (10):** sma_{20,50,200}, ema_{9,21}, price_to_sma_{20,50}, adx_14, dip_14, dim_14
- **Volatility (8):** bb_{upper,lower,width,pct}_20, atr_14, natr_14, hvol_{20,60}
- **Volume (10):** obv, obv_ema_20, vol_sma_20, vol_ratio_20, vwap_20, price_to_vwap_20, mfi_14, ad, cho_3_10, fi_13

### returns.py (21 features)
- Log returns (NOT simple) — additive across time
- Fractionally differentiated series from preprocessing (pass `frac_diff_col` param)
- Regime label: defaults to 1 (sideways) until HMM model is built (Week 4)
- **Features:** ret_log_{1,5,10,20}d, ret_mean/std_{5,20}d, ret_skew/kurt_20d, ret_sharpe_20d, ret_max_dd_20d, ret_up_ratio_20d, ret_mom_{5,20,60}d, ret_zscore_20d, ret_frac_diff, ret_52w_{high,low}_dist, ret_regime

### calendar_features.py (18 features)
- Pure date math, no external data
- F&O expiry: `_last_thursday_of_month()` helper
- **Features:** cal_dow, cal_month, cal_quarter, cal_week_of_month, cal_days_to_monthly_expiry, cal_is_expiry_{week,day}, cal_days_to_rbi, cal_is_budget_month, cal_is_{january,december}, cal_monsoon_flag, cal_is_result_season, cal_doy, cal_year, cal_week_of_year, cal_is_monday, cal_days_in_quarter

### microstructure.py (13 features)
- Requires `delivery_pct` column from jugaad-data provider
- Graceful degradation: returns zeros + warning if delivery_pct missing
- **Features:** micro_delivery_pct, micro_delivery_zscore, micro_delivery_to_ma5, micro_delivery_trend_5d, micro_delivery_accel, micro_vol_anomaly, micro_hd_{up,down}, micro_ld_up, micro_delivery_rolling_10d, micro_delivery_pct_rank, micro_delivery_vol_combo, micro_high_delivery_breakout

### macro.py (25 features)
- Two-layer: `fetch_macro_data()` fetches via yfinance, `compute_macro_features()` computes
- Market-wide data: India VIX, Nifty, USD/INR, Brent, Gold, US 10Y
- FII/DII from `institutional_flows` DB table
- Sector-relative returns computed from sector OHLCV
- **6 stub features** (NaN): macro_pmi, macro_pmi_staleness_days, macro_breadth_200sma, macro_adv_dec_ratio, macro_index_cpr, macro_mktcap_flow

### derivatives.py (20 features)
- Black-Scholes IV via `scipy.optimize.brentq` (risk-free rate: 6.5%)
- Options chain parsed from `options_data_json` column in `derivatives_data` table
- OI interpretation: buildup, unwinding, short cover, short build (binary flags)
- **6 stub features** (NaN): deriv_fii/pro/retail futures/options OI, deriv_gex, deriv_delta_oi

### sentiment.py (8 features)
- FinBERT (`ProsusAI/finbert`) loaded lazily via `_get_finbert()`
- Graceful degradation: returns neutral (0.0) + `sent_no_news_flag=1` if no data or no model
- Input: `daily_articles` dict mapping date strings to article text lists
- **Features:** sent_news_score, sent_news_score_5d, sent_velocity, sent_velocity_zscore, sent_article_count, sent_no_news_flag, sent_pos_ratio, sent_neg_ratio

## pipeline.py — Orchestrator
- `compute_all_features()` calls all 7 modules in order, concatenates into 142-column DataFrame
- Validates: replaces inf with NaN, forward-fill, back-fill, fill remaining with 0.0
- Warns if any column has >50% NaN
- Returns `FeatureResult` with df, symbol, feature_count, nan_filled_count, computation_time_ms, warnings
- Feature store integration: caller is responsible for calling `store.store_features()` after

## Stub Features (16 total)
Features that return NaN because their data sources aren't built yet:
- **macro (8):** gsec_10y (hardcoded 7.0), gsec_change, pmi, pmi_staleness, breadth_200sma, adv_dec_ratio, index_cpr, mktcap_flow
- **derivatives (6):** fii_futures_oi, fii_options_oi, pro_futures_net, retail_futures_net, gex, delta_oi
- **returns (1):** ret_regime (hardcoded 1, needs HMM model from Week 4)
- **Config:** All stubs marked with `stub: true` + `stub_reason` in `configs/features.yaml`

## Testing
- Unit tests: `tests/unit/features/` — 70 tests covering all 7 modules + pipeline
- Key assertions: bounded values (RSI 0-100, ratios 0-1), no look-ahead, graceful degradation, correct column counts
- Run: `pytest tests/unit/features/ -v`

## Adding New Features
1. Add to the appropriate group module
2. Update `configs/features.yaml` with feature metadata (name, window, source, unit)
3. Update the module's `*_FEATURE_COUNT` constant
4. Update `EXPECTED_FEATURE_COUNT` in pipeline.py
5. Add unit test verifying computation against known values
6. Run `make validate` to check impact on model performance
