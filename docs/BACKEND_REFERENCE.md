# AlphaVedha — Backend Reference

FastAPI app. Entry: `alphavedha/api/app.py`. Startup: `create_app(demo=bool)`. CLI: `alphavedha/cli/main.py`.

## Startup Sequence (lifespan)

1. `get_config()` — loads `configs/default.yaml`, Pydantic v2 validation, cached singleton
2. `ModelRegistry(demo=demo)` — prepares artifact paths (no weights loaded yet)
3. `aioredis.from_url(REDIS_URL).ping()` — connect Redis; if unreachable `redis_client = None`
4. `PredictionCache(redis_client)` — in-process LRU fallback if Redis down
5. `PredictionService(registry, cache, config)` → `registry.get_prediction_engine()` — **loads model weights here**
6. `set_service(service)` — stores singleton in module global
7. If not demo: `service.warm_up()` — single prediction to prime inference
8. Yield (app serves)
9. Shutdown: `redis_client.aclose()`

---

## Authentication (deps.py)

- **Header:** `X-API-Key` (FastAPI `APIKeyHeader`, `auto_error=False`)
- **Zero-config dev mode:** if no `ALPHAVEDHA_API_KEY` env var → all requests pass
- **Comparison:** `hmac.compare_digest()` (constant-time, prevents timing attacks)
- **Rotation:** `ALPHAVEDHA_API_KEY_SECONDARY` for zero-downtime key rotation
- **Responses:** 401 (missing), 403 (invalid)
- **Safe logging:** only first 4 chars + "..." logged on invalid attempts

---

## CORS

Controlled by `ALPHAVEDHA_CORS_ORIGINS` env var (comma-separated). Not added if unset. When configured: `allow_methods=["GET","POST"]`, `allow_headers=["X-API-Key","Content-Type"]`, `allow_credentials=True`.

---

## Rate Limiting

`slowapi.Limiter(key_func=get_remote_address)` keyed by client IP. 429 response: `{"error": {"code": "RATE_LIMITED", "message": "...", "details": {}}}` with `Retry-After: 60` header. Config: `api.rate_limit.default_per_minute=100`, `api.rate_limit.batch_per_minute=10`.

---

## Error Responses (global handlers)

All errors: `{"error": {"code": "...", "message": "...", "details": {}}}`

| Exception | HTTP | Code |
|---|---|---|
| SymbolNotFoundError | 404 | SYMBOL_NOT_FOUND |
| PredictionError | 500 | PREDICTION_FAILED |
| ModelNotFoundError | 503 | MODELS_NOT_LOADED |
| RateLimitExceeded | 429 | RATE_LIMITED |

---

## API Routes

### Health (`routes/health.py`) — no auth

| Method | Path | Response |
|---|---|---|
| GET | /health | `{"status": "ok", "version": "0.1.0"}` |
| GET | /ready | `{ready, models_loaded, cache_available, database_available, model_version}` |

---

### Predictions (`routes/predictions.py`) — require `X-API-Key`

| Method | Path | Request | Response |
|---|---|---|---|
| GET | /predict/{symbol} | path: symbol (regex `^[A-Z0-9&_.-]{1,20}$`) | `PredictionResponse` |
| POST | /predict/batch | body: `{symbols: list[str], max 20}` | `BatchResponse` (per-symbol failures in `failed[]`) |
| GET | /scan/{tier} | path: tier ∈ {large,mid,small,all}; query: top_n (1-50, default 10) | `ScanResponse` |

**Cache:** Redis key `predict:{symbol}:{model_version}`. TTL: 300s market hours, seconds-to-next-open otherwise.

---

### Paper Trading (`routes/paper_trading.py`) — no auth — prefix `/paper`

| Method | Path | Request | Response |
|---|---|---|---|
| POST | /paper/predict | `PaperTradeRequest` | `PaperTradeResponse` |
| POST | /paper/outcome | `TradeOutcomeRequest` | `{status, symbol, date}` |
| GET | /paper/dashboard | — | `DashboardSummary` (3 tracks: all/gate_passed/top_k) |
| GET | /paper/trades | query: symbol?, limit=100 | `list[PredictionRecord]` |
| GET | /paper/simulation | — | `{available, track_record, diagnostics, meta, generated_at}` |
| GET | /paper/simulations | — | `{runs, count}` (historical simulation archive) |
| GET | /paper/simulation/{slug} | path: slug (alphanum regex) | `{available, slug, track_record, diagnostics, backtest, meta}` |

---

### Dashboard (`routes/dashboard.py`) — no auth — prefix `/dashboard`

| Method | Path | Response |
|---|---|---|
| GET | /dashboard/track-record | `PublicTrackRecord` |
| GET | /dashboard/equity-curve | `list[DailyPnLRecord]` |

---

### Live / WebSocket (`routes/live.py`) — no auth

| Method | Path | Messages |
|---|---|---|
| WS | /ws/live/{symbol} | `snapshot` (candles + tick), `tick` (LTP update every 5s), `market_closed` |
| WS | /ws/market | `market_summary` for NIFTY50/BANKNIFTY/SENSEX every 5s |

nginx proxies `/api/ws/` → `/ws/` on the FastAPI container with WebSocket upgrade headers.

---

### Sectors (`routes/sectors.py`) — require auth — prefix `/sectors`

| Method | Path | Response |
|---|---|---|
| GET | /sectors/rotation | `{rotation_message, top_sectors, avoid_sectors, benchmark, sectors[], data_quality, generated_at}` — RRG methodology (12 sector indices vs Nifty 50) |

---

### Sentiment (`routes/sentiment.py`) — require auth — prefix `/sentiment`

| Method | Path | Request | Response |
|---|---|---|---|
| GET | /sentiment/{symbol} | query: lookback_days (1-30, default 7) | `{symbol, score, momentum, verdict, post_count, source_counts, data_quality, generated_at}` |

Verdict thresholds: bullish ≥ 0.15, cautiously_bullish ≥ 0.05, neutral, cautiously_bearish ≤ -0.05, bearish ≤ -0.15.

---

### Signals (`routes/signals.py`) — require auth — prefix `/signals`

| Method | Path | Request | Response |
|---|---|---|---|
| GET | /signals/timing | — | `{is_good_to_trade, timing_quality_reason, is_expiry_day, next_fo_expiry, optimal_windows[]}` |
| GET | /signals/execution/{symbol} | query: cap_tier, avg_daily_volume, order_size_shares, current_spread_pct | `{order_type, n_tranches, tranche_interval_minutes, estimated_slippage_pct, recommended_windows[]}` |
| GET | /signals/buy-sell/{symbol} | query: cap_tier, avg_daily_volume, order_size_shares | Prediction + execution plan combined |

---

### Fundamental (`routes/fundamental.py`) — require auth — prefix `/fundamental`

| Method | Path | Response |
|---|---|---|
| GET | /fundamental/analyze/{symbol} | `{overall_verdict, beneish_m_score, altman_z_score, summary, data_quality}` |

Verdicts: `healthy / caution / red_flag / insufficient_data`.

---

### UI Support (`routes/ui_support.py`) — no auth (used by Next.js UI)

All endpoints have demo mode (deterministic synthetic data) and real mode (live DB/model data). Demo mode: `ALPHAVEDHA_DEMO=1` env var.

| Method | Path | Request | Response | Cache |
|---|---|---|---|---|
| GET | /scan/{tier} | query: top_n | `ScanResponseModel` with buy/sell/all_candidates | Redis per prediction |
| GET | /predict/{symbol}/explain | — | `ExplainResponse` (XGB feature importances + prediction) | None |
| GET | /portfolio/summary | — | `PortfolioSummaryResponse` | None |
| GET | /models/status | — | `ModelsStatusResponse` (7 model statuses + ensemble summary) | None |
| GET | /backtest/summary | — | `BacktestSummaryResponse` | mtime-based file cache |
| GET | /backtest/equity | — | `{strategy: [{y, date}], benchmark: [{y, date}]}` | mtime-based file cache |
| GET | /backtest/monthly | — | `list[{year, month, return_pct}]` | mtime-based file cache |
| GET | /backtest/distribution | — | `list[{label, count}]` | mtime-based file cache |
| GET | /backtest/rolling-sharpe | — | `list[{y}]` | mtime-based file cache |
| GET | /backtest/range | query: start?, end? | Per-day + date-range slicing from sim artifact | None |
| GET | /stocks/search | query: q | `{results: [StockSearchResult]}` max 10 | None |
| GET | /intraday/live | query: symbol?, tier? | `{symbol, ltp, open, high, low, change_pct, candles[], recent_ticks[]}` | 2 min in-process |
| GET | /system/health | — | `SystemResources` (CPU, memory, disk, GPU=0.0) | None |
| GET | /system/data-quality | — | `{overall_score, symbols_covered, missing_bars, last_updated, symbol_quality[]}` | None |
| GET | /features/drift | — | `list[DriftFeature]` (always [] in real mode currently) | None |
| GET | /experiments | query: model? | `list[ExperimentRun]` from ExperimentTracker | None |
| GET | /events/corporate | query: days=30, symbol?, type? | `list[CorporateEvent]` | 6h in-process |
| GET | /sectors/trends | — | `{sectors[], trends_signals[]}` (RRG sector data) | 10 min in-process |
| GET | /notifications | — | `list[{id, type, title, body, read, created_at}]` (high-confidence paper trades) | None |
| POST | /notifications/read-all | — | `{"status": "ok"}` (no-op) | None |
| GET | /paper/positions | — | `list[{id, symbol, side, quantity, entry_price, ltp, unrealized_pnl}]` | None |
| GET | /paper/orders | — | `list[{id, symbol, side, quantity, price, status, timestamp}]` last 50 | None |
| GET | /paper/equity-history | — | `list[{y}]` last 30 rows from daily_pnl | None |
| POST | /paper/orders | body: `{symbol, side, quantity?, price?, confidence?, magnitude?}` | `{id}` | None |
| POST | /paper/positions/{position_id}/close | path: `{symbol}:{date}` | `{status, id}` | None |
| GET | /public/track-record | — | full track-record dict | None |

---

### Public (`routes/public.py`) — no auth — prefix `/public`

| Method | Path | Request | Response |
|---|---|---|---|
| GET | /public/track-record | — | Track record with 3 tracks, monthly returns, accuracy over time |
| GET | /public/predictions | query: start_date?, end_date?, symbol?, direction?, min_confidence?, page=1, page_size=50 | Paginated predictions (max page_size 200) |
| GET | /public/equity-curve | query: start_date? | `{points[], start_value, current_value}` |
| GET | /public/monthly-returns | — | `{returns: [MonthlyReturn]}` |
| GET | /public/predictions/export | query: format=csv\|json | StreamingResponse (CSV) or JSONResponse |
| GET | /public/model-info | — | `{model_version, architecture, base_models[], feature_count, last_retrain_date, validation_sharpe}` |

---

## Service Layer

### PredictionService (services/prediction_service.py)

Central orchestrator. Shared by API routes, CLI, and scheduler.

Key methods:

| Method | Description |
|---|---|
| `predict_single(symbol, sector="")` | Cache-first: Redis → features → engine.predict() → write cache |
| `scan_tier(tier, top_n=10)` | Deduplicates concurrent requests via asyncio.Task per `{tier}:{top_n}` key |
| `predict_batch(symbols)` | Concurrent with `asyncio.Semaphore(10)` |
| `warm_up()` | One prediction at startup to warm JIT/model loading |
| `_get_features(symbol)` | Demo: mock features; Real: load from feature store → fallback to on-the-fly compute |
| `_compute_features_on_the_fly(symbol)` | OHLCV → compute_all_features() in thread + optional macro → write back to store |

**`_as_of: date | None`** — injected for historical simulation; None = live (today).
**`_feature_window_rows`** — max(lstm.sequence_length, tft.sequence_length) = 60.

---

### ModelRegistry (services/model_registry.py)

Loads model artifacts from disk (or constructs deterministic demo mocks).

Key methods:

| Method | Description |
|---|---|
| `models_available()` | Checks `{name}/latest/metadata.json` exists per required model (no weight loading) |
| `get_prediction_engine()` | Builds PredictionEngine with real or demo models |

**Required artifacts:** xgboost, lstm, tft, regime, ensemble, meta_labeling, conformal.
**Optional:** gnn (loaded if artifact exists).

Demo model behaviors (deterministic from MD5 seed of input):
- `_DemoBaseModel`: fixed direction/magnitude/proba from MD5 seed
- `_DemoRegime`: always "bull", proba [0.6, 0.1, 0.2, 0.1]
- `_DemoMeta`: 0.72 confidence for non-hold, 0.45 for hold
- `_DemoConformal`: fixed price_low=95, price_mid=100, price_high=105

---

### PredictionCache (services/cache.py)

Redis-backed with in-process LRU fallback (256 entries max).

**TTL logic:** During market hours (09:15-15:30 IST weekdays): 300s. Otherwise: seconds to next 9:15 IST open. Minimum always 300s.

**Cache key format:** `predict:{symbol}:{model_version}`

**Serialization:** `json.dumps(asdict(prediction), cls=_NumpyEncoder)` — handles np.ndarray, np.integer, np.floating, datetime.

---

### UIDataService (services/ui_data.py)

Pure data-access helpers for non-demo UI endpoints. All degrade gracefully to empty/honest values on failure.

Key functions:

| Function | Description |
|---|---|
| `read_model_metadata(name)` | Reads `{artifact_dir}/{name}/latest/metadata.json` |
| `read_xgb_feature_importance(limit=10)` | From `xgboost/feature_importance.csv`, sorted desc |
| `system_resources()` | CPU (os.getloadavg), memory (/proc/meminfo), disk (shutil.disk_usage) |
| `ohlcv_store_stats()` | Row count, symbol count, latest date, per-symbol coverage |
| `fetch_intraday_5m(symbol)` | yfinance 1d/5m, 2-min in-process cache |
| `fetch_corporate_events(symbols)` | yfinance .calendar per symbol, 6-hour in-process cache |
| `compute_sector_trends()` | Equal-weight normalized price series per sector, 10-min cache |

---

## Key Data Classes

### PredictionResponse (API output)
```python
symbol: str
direction: int           # -1 SELL, 0 HOLD, 1 BUY
direction_label: str     # "BUY" / "SELL" / "HOLD"
magnitude: float         # expected return, fractional (0.03 = 3%)
composite_score: float   # 0-100
meta_confidence: float   # 0.0-1.0
is_tradeable: bool
regime: str              # bull/bear/sideways/high_volatility/unknown
price_targets: PriceTargets   # {low, mid, high}
risk: RiskInfo                # {position_size_pct, model_disagreement}
trade_setup: TradeSetup       # {entry_price?, stop_loss_price?, take_profit_price?}
model_version: str
generated_at: datetime
warnings: list[str]
```

### StockPrediction (internal dataclass — PredictionEngine output)
All fields from PredictionResponse plus: `timestamp`, `regime_probabilities: np.ndarray[4]`, `model_disagreement: float`, `position_size_pct: float`, `entry_price`, `stop_loss_price`, `take_profit_price`.

### ScanStockItem (ui_support.py)
`symbol, name, price, change_pct, sector, cap, direction, confidence, regime, t7/t15/t30{low,high,pct}, sparkline, composite_score, meta_confidence, magnitude, price_targets, top_feature, ai_insight`

### DashboardSummary (paper_trading.py)
`total_predictions, correct_predictions, accuracy_7d/30d/all, total_return, sharpe_ratio, max_drawdown, days_tracked, round_trip_cost_pct, tracks[TrackStatsOut]`

### TrackStatsOut (paper_trading.py)
`name, n_selected, n_evaluated, n_wins_net, win_rate_net, avg_return_gross, avg_return_net, total_return_net, profit_factor_net, sharpe_net, max_drawdown_net`

---

## Scheduler Jobs (scheduler.py)

All times IST (Asia/Kolkata).

| Job | Time | What |
|---|---|---|
| run_daily_predictions | Daily 08:30 | Predict all tier symbols, persist as paper_trades |
| run_daily_evaluation | Daily 15:45 | Update paper_trade outcomes at 15-day horizon |
| run_quality_check | Daily 15:50 | QualityChecker.run_full_check(), email if critical |
| run_data_refresh | Daily 17:00 | Ingest last 5 days OHLCV |
| run_fii_dii_ingestion | Daily 18:30 | NSE FII/DII data (published ~17:30) |
| run_daily_xgboost_retrain | Daily 23:30 | XGBoost retrain (CX23-safe, 2 vCPU / 4 GB) |
| run_intraday_poll | Every 2 min (market hours) | LiveDataPoller.poll_once() for all symbols |
| run_bse_ingestion | Sunday 21:00 | BSE announcements last 7 days |
| run_trends_ingestion | Sunday 21:30 | Google Trends for 5 market sectors |
| run_drift_check | Saturday 20:00 | PSI + KS test on all features |
| run_monthly_retrain | 1st Saturday 22:00 | Full retrain if triggered by drift/performance |
| run_weekly_lstm_tft_retrain | Saturday 22:30 | Only if `ALPHAVEDHA_HEAVY_TRAINING=1` |
| run_rebalance_check | Monday 07:00 (March/Sep only) | Compare live Nifty 50 vs stocks.yaml |

Scheduler constants: `EVALUATION_HORIZON_TRADING_DAYS=15`, `EVALUATION_MIN_CALENDAR_DAYS=21`, `INITIAL_PORTFOLIO_VALUE=1_000_000.0`.

---

## CLI Commands (cli/main.py)

```bash
# Prediction
alphavedha predict SYMBOL [--demo] [--json]
alphavedha scan TIER [--top-n N] [--demo] [--json]
alphavedha serve [--host] [--port] [--demo] [--reload]

# Data
alphavedha data refresh|backfill|status|fii-refresh|derivatives-refresh
alphavedha data earnings-refresh|live-status|fetch-bse|quality-check|fetch-trends

# Training (dependency order)
alphavedha train xgboost|lstm|tft|regime|ensemble|meta|conformal|all

# Backtest
alphavedha backtest walk-forward

# Scheduler
alphavedha scheduler start
alphavedha scheduler run-now predictions|evaluation|drift|retrain|rebalance
alphavedha scheduler status

# Experiments
alphavedha experiment list [--model] [--limit]
alphavedha experiment compare RUN_A RUN_B

# Models
alphavedha model compare [--model-name]
```

---

## Custom Exceptions (exceptions.py)

```
AlphaVedhaError
├── DataProviderError
├── DataQualityError
├── FeatureComputationError
├── ModelTrainingError
├── ModelNotFoundError       → HTTP 503
├── PredictionError          → HTTP 500
├── ValidationError
├── ConfigError
├── SymbolNotFoundError      → HTTP 404
├── InsufficientDataError
└── CircuitBreakerTriggeredError
```

---

## Prometheus Metrics (/metrics)

| Metric | Type | Labels |
|---|---|---|
| alphavedha_prediction_seconds | Histogram | symbol, model |
| alphavedha_predictions_total | Counter | direction, tier |
| alphavedha_prediction_confidence | Histogram | — |
| alphavedha_model_load_seconds | Histogram | model_type |
| alphavedha_feature_compute_seconds | Histogram | symbol |
| alphavedha_scheduler_job_seconds | Histogram | job_name |
| alphavedha_scheduler_job_total | Counter | job_name, status |
| alphavedha_drift_psi | Gauge | feature_group |
| alphavedha_active_positions | Gauge | — |
| alphavedha_cache_hits_total | Counter | — |
| alphavedha_cache_misses_total | Counter | — |

Instrumented via `prometheus_fastapi_instrumentator`. Excludes `/health` and `/metrics`.
