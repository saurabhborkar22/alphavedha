# AlphaVedha — Feature Catalog

Total declared: **164 features** across 9 categories.
Effective at training time: **148** (16 stubs hardcoded to zero are dropped by `_STUB_FEATURES` in `training/pipeline.py`).

Feature computation entry point: `features/pipeline.py:compute_all_features(df, symbol, config, as_of, **kwargs)`

---

## Category Summary

| Category | Count | Source | File |
|---|---|---|---|
| Technical | 40 | OHLCV via `ta` library | `features/technical.py` |
| Returns | 21 | Close price (log returns) | `features/returns.py` |
| Calendar | 18 | DatetimeIndex only | `features/calendar_features.py` |
| Microstructure | 13 | delivery_pct from jugaad-data | `features/microstructure.py` |
| Macro | 30 | yfinance, DB, stubs | `features/macro.py` |
| Derivatives | 20 | derivatives_data DB table | `features/derivatives.py` |
| Sentiment | 8 | FinBERT (ProsusAI/finbert) | `features/sentiment.py` |
| Fundamental | 9 | earnings_results, promoter_holdings, insider_trades | `features/fundamental_features.py` |
| Corporate Events | 3 | corporate_announcements DB | `features/corporate_events.py` |
| Trends | 2 | Google Trends (STUB) | `features/trends_features.py` |
| **Total** | **164** | | |

---

## Technical Features (40)

### Momentum (12)

| Feature | Lookback | Description |
|---|---|---|
| rsi_7 | 7 | RSI 7-day |
| rsi_14 | 14 | RSI 14-day |
| rsi_21 | 21 | RSI 21-day |
| stoch_k_14 | 14 | Stochastic %K (window=14, smooth=3) |
| stoch_d_14 | 14/3 | Stochastic %D signal |
| macd_12_26 | 12/26 | MACD line (EMA12 - EMA26) |
| macd_signal_12_26 | 9 | MACD signal (9-day EMA of MACD) |
| macd_hist_12_26 | — | MACD histogram (MACD - signal) |
| willr_14 | 14 | Williams %R |
| roc_10 | 10 | Rate of Change 10-day |
| roc_20 | 20 | Rate of Change 20-day |
| cci_20 | 20 | Commodity Channel Index |

### Trend (10)

| Feature | Lookback | Description |
|---|---|---|
| sma_20 | 20 | Simple Moving Average |
| sma_50 | 50 | Simple Moving Average |
| sma_200 | 200 | Simple Moving Average |
| ema_9 | 9 | Exponential Moving Average |
| ema_21 | 21 | Exponential Moving Average |
| price_to_sma_20 | 20 | Close / SMA20 |
| price_to_sma_50 | 50 | Close / SMA50 |
| adx_14 | 14 | Average Directional Index (NaN guard: needs ≥ 28 rows) |
| dip_14 | 14 | DI+ (positive directional movement) |
| dim_14 | 14 | DI- (negative directional movement) |

### Volatility (8)

| Feature | Lookback | Description |
|---|---|---|
| bb_upper_20 | 20 | Bollinger upper band |
| bb_lower_20 | 20 | Bollinger lower band |
| bb_width_20 | 20 | Bollinger band width |
| bb_pct_20 | 20 | Bollinger %B |
| atr_14 | 14 | Average True Range |
| natr_14 | 14 | Normalized ATR = ATR/close × 100 |
| hvol_20 | 20 | Historical volatility (annualized log returns) |
| hvol_60 | 60 | Historical volatility (annualized log returns) |

### Volume (10)

| Feature | Lookback | Description |
|---|---|---|
| obv | cumulative | On-Balance Volume |
| obv_ema_20 | 20 | 20-day EMA of OBV |
| vol_sma_20 | 20 | Volume 20-day SMA |
| vol_ratio_20 | 20 | Volume / VolumeMA20 |
| vwap_20 | 20 | 20-day rolling VWAP |
| price_to_vwap_20 | 20 | Close / VWAP20 |
| mfi_14 | 14 | Money Flow Index |
| ad | cumulative | Accumulation/Distribution Index |
| cho_3_10 | 3/10 | Chaikin Oscillator (EMA3(AD) - EMA10(AD)) |
| fi_13 | 13 | Force Index |

---

## Returns Features (21)

All use log returns (not simple returns). Source: close price column.

| Feature | Lookback | Description |
|---|---|---|
| ret_log_1d | 1 | 1-day log return |
| ret_log_5d | 5 | 5-day log return |
| ret_log_10d | 10 | 10-day log return |
| ret_log_20d | 20 | 20-day log return |
| ret_mean_5d | 5 | 5-day rolling mean of daily log returns |
| ret_mean_20d | 20 | 20-day rolling mean |
| ret_std_5d | 5 | 5-day rolling std |
| ret_std_20d | 20 | 20-day rolling std |
| ret_skew_20d | 20 | 20-day rolling skewness |
| ret_kurt_20d | 20 | 20-day rolling kurtosis |
| ret_sharpe_20d | 20 | 20-day rolling Sharpe (annualized) |
| ret_max_dd_20d | 20 | 20-day rolling max drawdown |
| ret_up_ratio_20d | 20 | Fraction of days with positive log return (20-day) |
| ret_mom_5d | 5 | Simple momentum: close/close[t-5] - 1 |
| ret_mom_20d | 20 | Simple momentum 20-day |
| ret_mom_60d | 60 | Simple momentum 60-day |
| ret_zscore_20d | 20 | Z-score of today's log return vs 20-day distribution |
| ret_frac_diff | varies | Fractionally differentiated price series (min d passing ADF test, typical d≈0.3–0.5) |
| ret_52w_high_dist | 252 | (close - 52w_high) / 52w_high |
| ret_52w_low_dist | 252 | (close - 52w_low) / 52w_low |
| ret_regime | — | **STUB** — hardcoded to 1 (HMM not applied per-symbol at feature time) |

---

## Calendar Features (18)

Pure time-based features. Zero external data required.

| Feature | Description |
|---|---|
| cal_dow | Day of week (0=Monday, 4=Friday) |
| cal_month | Month (1-12) |
| cal_quarter | Quarter (1-4) |
| cal_week_of_month | Week within month (1-5) |
| cal_days_to_monthly_expiry | Days to next F&O monthly expiry (last Thursday of month) |
| cal_is_expiry_week | Binary: date is in expiry week |
| cal_is_expiry_day | Binary: date IS the last Thursday |
| cal_days_to_rbi | Approximate days to next RBI bi-monthly policy meeting |
| cal_is_budget_month | Binary: February |
| cal_is_january | Binary: January |
| cal_is_december | Binary: December |
| cal_monsoon_flag | Binary: June–September |
| cal_is_result_season | Binary: January/April/July/October |
| cal_doy | Day of year (1-366) |
| cal_year | Year |
| cal_week_of_year | ISO week (1-53) |
| cal_is_monday | Binary: Monday |
| cal_days_in_quarter | Days elapsed since start of current quarter |

---

## Microstructure Features (13)

Source: `delivery_pct` from NSE bhavcopy via jugaad-data. Degrades gracefully to zeros if data unavailable (circuit days, newly listed stocks).

| Feature | Lookback | Description |
|---|---|---|
| micro_delivery_pct | 1 | Raw delivery % (>50% = institutional buying signal) |
| micro_delivery_zscore | 20 | Z-score of delivery_pct vs 20-day window |
| micro_delivery_to_ma5 | 5 | delivery_pct / 5-day MA |
| micro_delivery_trend_5d | 5 | delivery_pct - delivery_pct[t-5] |
| micro_delivery_accel | 2 | Second difference of delivery_pct |
| micro_vol_anomaly | 20 | Volume / 20-day volume MA |
| micro_hd_up | 1 | Binary: delivery>50% AND close>open (high delivery, price up) |
| micro_hd_down | 1 | Binary: delivery>50% AND close<open (high delivery, price down) |
| micro_ld_up | 1 | Binary: delivery<30% AND close>open (speculative buying) |
| micro_delivery_rolling_10d | 10 | 10-day rolling mean of delivery_pct |
| micro_delivery_pct_rank | 60 | Percentile rank within 60-day window |
| micro_delivery_vol_combo | 20 | Binary: delivery_zscore>2 AND vol_anomaly>1.5 |
| micro_high_delivery_breakout | 20 | Binary: delivery>60% AND close>20-day high (prior) |

---

## Macro Features (30)

| Feature | Source | Stub? | Description |
|---|---|---|---|
| macro_vix | yfinance ^INDIAVIX | No | India VIX level |
| macro_vix_change_1d | VIX | No | Daily % change |
| macro_vix_zscore_20d | VIX | No | 20-day z-score |
| macro_nifty_ret_1d | yfinance ^NSEI | No | Nifty 50 1-day log return |
| macro_nifty_ret_5d | yfinance ^NSEI | No | Nifty 50 5-day log return |
| macro_usdinr | yfinance USDINR=X | No | USD/INR spot |
| macro_usdinr_change_1d | USDINR | No | Daily % change |
| macro_brent | yfinance BZ=F | No | Brent crude price |
| macro_brent_change_1d | Brent | No | Daily % change |
| macro_gold | yfinance GC=F | No | Gold price USD |
| macro_us10y | yfinance ^TNX | No | US 10Y Treasury yield |
| macro_gsec_10y | RBI | **STUB** | India 10Y G-Sec yield (hardcoded 7.0) |
| macro_gsec_change_1d | G-Sec | **STUB** | Daily change (hardcoded 0.0) |
| macro_fii_net | institutional_flows DB | No | FII net buy/sell (crore INR) |
| macro_fii_cum_5d | FII net | No | 5-day cumulative FII flow |
| macro_dii_net | institutional_flows DB | No | DII net buy/sell (crore INR) |
| macro_dii_cum_5d | DII net | No | 5-day cumulative DII flow |
| macro_sector_ret_1d | sector index OHLCV | No | Sector index 1-day log return |
| macro_sector_rel_ret_1d | stock vs sector | No | Stock return minus sector return |
| macro_pmi | gov releases | **STUB** | India Manufacturing PMI |
| macro_pmi_staleness_days | PMI | **STUB** | Days since last PMI |
| macro_breadth_200sma | universe prices | **STUB** | % stocks above 200-day SMA |
| macro_adv_dec_ratio | NSE bhavcopy | **STUB** | Advance/Decline ratio |
| macro_index_cpr | NSE derivatives | **STUB** | Index central pivot range |
| macro_mktcap_flow | computed | **STUB** | FII-DII differential |
| macro_crude_oil | alternative_data DB | No | Crude oil price |
| macro_crude_oil_change_5d | crude_oil | No | 5-day % change |
| macro_us_overnight_return | alternative_data DB | No | US market overnight return |
| macro_forex_reserves | alternative_data DB | No | India forex reserves |
| macro_forex_reserves_change | forex_reserves | No | % change |

---

## Derivatives Features (20)

Source: `derivatives_data` DB table (NSE F&O data). Options chain stored as JSON.

| Feature | Stub? | Description |
|---|---|---|
| deriv_futures_oi | No | Futures open interest (contracts) |
| deriv_futures_oi_change | No | Futures OI % daily change |
| deriv_futures_premium | No | Futures premium over spot (%) |
| deriv_atm_iv | No | ATM implied volatility via Black-Scholes (scipy.brentq, r=6.5%) |
| deriv_iv_rank | No | IV rank: (current - 252d_min)/(252d_max - 252d_min) |
| deriv_iv_pctile | No | IV percentile within 252-day window |
| deriv_pcr_oi | No | Put/Call ratio by open interest |
| deriv_pcr_vol | No | Put/Call ratio by volume |
| deriv_max_pain | No | Strike with max aggregate OI loss to option writers |
| deriv_dist_max_pain | No | (close - max_pain) / close × 100 |
| deriv_fii_futures_oi | **STUB** | FII futures OI |
| deriv_fii_options_oi | **STUB** | FII options OI |
| deriv_pro_futures_net | **STUB** | Proprietary desk net futures |
| deriv_retail_futures_net | **STUB** | Retail net futures |
| deriv_oi_buildup | No | Binary: OI↑ AND price↑ (long buildup) |
| deriv_oi_unwind | No | Binary: OI↓ AND price↓ (long unwinding) |
| deriv_short_cover | No | Binary: OI↓ AND price↑ (short covering) |
| deriv_short_build | No | Binary: OI↑ AND price↓ (short buildup) |
| deriv_gex | **STUB** | Gamma exposure (needs full Greeks) |
| deriv_delta_oi | **STUB** | Delta-adjusted OI |

---

## Sentiment Features (8)

Source: FinBERT (`ProsusAI/finbert`) lazy-loaded on first use. News from Moneycontrol RSS, ET Markets RSS, Business Standard RSS + Reddit (r/IndiaInvestments, r/niftyoptions, r/DalalStreetTalks, r/IndianStockMarket).

| Feature | Lookback | Description |
|---|---|---|
| sent_news_score | 1 | Daily mean FinBERT net score (positive prob - negative prob) |
| sent_news_score_5d | 5 | 5-day rolling mean of news_score |
| sent_velocity | 1 | First difference of news_score |
| sent_velocity_zscore | 20 | 20-day z-score of velocity |
| sent_article_count | 1 | Daily article count |
| sent_no_news_flag | 1 | Binary: zero articles today |
| sent_pos_ratio | 1 | Ratio of articles with net score > 0.1 |
| sent_neg_ratio | 1 | Ratio of articles with net score < -0.1 |

---

## Fundamental Features (9)

All computed point-in-time: only data where `announced_date <= as_of_date` is used.

| Feature | Update Freq | Description |
|---|---|---|
| fund_earnings_surprise_pct | Quarterly | (actual - estimate) / |estimate| × 100; falls back to YoY growth if no estimate |
| fund_days_since_earnings | Daily | Days since last earnings announcement |
| fund_earnings_surprise_streak | Quarterly | Consecutive quarterly beats (+) or misses (-) |
| fund_revenue_growth_qoq | Quarterly | QoQ revenue growth % |
| fund_profit_margin_change | Quarterly | Profit margin change vs prior quarter (ppt) |
| fund_promoter_pledge_pct | Quarterly | Current promoter pledge % |
| fund_pledge_change_qoq | Quarterly | QoQ change in pledge % |
| fund_promoter_buying_30d | Daily | Binary: promoter bought shares in last 30 days |
| fund_insider_buy_sell_ratio | Daily | Insider buy/(buy+sell) value in trailing 30 days |

---

## Corporate Events Features (3)

Source: `corporate_announcements` DB table (BSE announcements).

| Feature | Description |
|---|---|
| corp_days_to_next_board | Days until next board meeting (999 if none scheduled) |
| corp_days_since_dividend | Days since last dividend (999 if none in history) |
| corp_event_this_week | Binary: any event (board/dividend/bonus/rights/buyback/split/AGM/EGM) within next 7 days |

---

## Trends Features (2) — STUB

Source: Google Trends API. Not wired at training time.

| Feature | Lookback | Description |
|---|---|---|
| trends_sector_7d | 7 | Google Trends sector search interest 7-day avg (0-100 scale) |
| trends_sector_change | 7 | Recent 7d avg minus prior 7d avg |

---

## Stub Features List (16 dropped at training time)

These are returned by compute_all_features() but excluded by `_STUB_FEATURES` frozenset before any model training or prediction:

```python
_STUB_FEATURES = frozenset({
    "macro_gsec_10y",        # India G-Sec yield — no live source
    "macro_gsec_change_1d",  # G-Sec change
    "macro_pmi",             # India PMI — no live source
    "macro_pmi_staleness_days",
    "macro_breadth_200sma",  # universe not passed per-symbol
    "macro_adv_dec_ratio",   # NSE bhavcopy not integrated
    "macro_index_cpr",       # NSE derivatives pivot
    "macro_mktcap_flow",     # FII-DII differential computed differently
    "deriv_fii_futures_oi",  # NSE participant-level data
    "deriv_fii_options_oi",
    "deriv_pro_futures_net",
    "deriv_retail_futures_net",
    "deriv_gex",             # needs full options Greeks
    "deriv_delta_oi",        # needs delta-adjusted options
    "ret_regime",            # HMM applied at engine level, not per-symbol
    "trends_sector_7d",      # Google Trends not wired
    "trends_sector_change",
})
```

---

## Feature Storage

Features stored as JSON blob in `features` table:
```sql
(symbol, date, feature_version) PRIMARY KEY
feature_json: JSON  -- full 164-feature dict
```

Load for prediction: most recent 60 rows (max of lstm.sequence_length, tft.sequence_length).
Upsert key: `(symbol, date, feature_version)` — version string like "v1" allows clean invalidation when feature engineering changes.

---

## Fractional Differentiation

Applied to price series (not returns) to achieve stationarity while preserving memory:
- Per-symbol: find minimum d where ADF test p-value < 0.05 (default range: d ∈ [0.1, 0.8])
- Typical d ≈ 0.3–0.5 for Indian stocks
- Fixed-width window: max 100 lags (FFD — Fixed-width window Fractionally Differentiated)
- Recomputed monthly per config
- Result stored as `ret_frac_diff` feature
