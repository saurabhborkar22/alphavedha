# Architecture Decision Records

Key design decisions in AlphaVedha, with context on what was considered and why each choice was made.

---

## ADR-001: XGBoost + LSTM + TFT Ensemble

**Status:** Accepted  
**Date:** 2026-03

### Context

Stock prediction requires capturing both tabular cross-sectional patterns (company fundamentals, technicals) and temporal sequential patterns (price trends, momentum regimes). No single model architecture excels at both.

### Decision

Use a three-model ensemble:
- **XGBoost** for tabular features (159 features, all at once)
- **LSTM** for sequential patterns (top 30 features selected by XGBoost importance, 60-day windows)
- **Temporal Fusion Transformer (TFT)** for multi-horizon attention-based forecasting (7d/15d/30d)

A **RidgeClassifier stacking meta-learner** combines their out-of-fold predictions with HMM regime probabilities into a final signal.

### Alternatives Considered

1. **Single XGBoost** — fast and interpretable, but misses temporal dependencies entirely.
2. **Single Transformer** — handles sequences well, but struggles with the 159 heterogeneous tabular features and needs far more data.
3. **Simple averaging** — no learned combination weights, can't adapt to regime changes.

### Consequences

- Training is sequential: XGBoost first (provides feature selection for LSTM/TFT), then LSTM, TFT, regime detector, then ensemble.
- More complex deployment (3 model artifacts + meta-learner), but the stacking ensemble is cheap to retrain.
- RidgeClassifier was chosen over logistic regression because it's faster and handles multicollinearity in the 14 meta-features (3 probability vectors + 4 regime probs + 1 disagreement score).

---

## ADR-002: Triple Barrier Labeling

**Status:** Accepted  
**Date:** 2026-03

### Context

Traditional binary labels (price went up/down) ignore magnitude and holding period. A stock that rises 0.1% and one that rises 5% get the same label, and there's no concept of "no clear signal."

### Decision

Use the **Triple Barrier Method** (de Prado, *Advances in Financial Machine Learning*):
- Upper barrier: entry + 2.0x ATR (take profit)
- Lower barrier: entry - 1.5x ATR (stop loss)
- Time barrier: 15 trading days (max holding period)

Label = whichever barrier is hit first (+1 buy, -1 sell, 0 neutral).

### Alternatives Considered

1. **Fixed threshold labels** (e.g., +2% = buy) — ignores per-stock volatility. A 2% move means different things for Infosys vs a small-cap.
2. **Binary up/down** — no neutral class, forces the model to pick a side on noise.
3. **Regression only** (predict returns) — harder to translate into actionable signals.

### Consequences

- ATR-scaled barriers adapt to each stock's volatility automatically.
- Asymmetric multipliers (2.0x up vs 1.5x down) encode a risk-averse stance: require more evidence for a buy than a sell.
- The neutral class (0) is important — it lets the model say "I don't know," which the meta-labeling model can further filter.
- ATR must be computed using only past data to prevent look-ahead bias.

---

## ADR-003: Meta-Labeling as a Confidence Gate

**Status:** Accepted  
**Date:** 2026-04

### Context

Even with a good ensemble, many predictions are low-confidence noise. Trading on every signal destroys returns through transaction costs. We need a way to filter signals by reliability.

### Decision

Train a **Meta-Labeling Model** (XGBClassifier) that predicts P(ensemble prediction is correct) using original features + ensemble direction + ensemble confidence. Only trade when meta_confidence > 0.55.

### Alternatives Considered

1. **Confidence threshold on ensemble probabilities** — simple, but the ensemble's raw probabilities are poorly calibrated.
2. **Kelly criterion only** — sizes positions by confidence but doesn't filter bad signals out entirely.
3. **Conformal prediction sets** — good for uncertainty quantification but doesn't answer "is this signal tradeable?"

### Consequences

- Filters 30-40% of signals, keeping only the most reliable ones.
- Must be trained on a separate time period from the ensemble (no data leakage).
- The 0.55 threshold is conservative — barely above random, but on real data even this small edge compounds.
- Combined with Kelly position sizing, this creates a two-layer risk filter: meta-labeling decides whether to trade, Kelly decides how much.

---

## ADR-004: TimescaleDB for Time-Series Storage

**Status:** Accepted  
**Date:** 2026-05

### Context

AlphaVedha stores daily OHLCV data, features, predictions, and paper trades — all indexed by (symbol, timestamp). Query patterns are almost always time-range scans filtered by symbol.

### Decision

Use **PostgreSQL 16 + TimescaleDB** with hypertables:
- 8 tables converted to hypertables (monthly chunks)
- Composite natural primary keys (symbol + timestamp) replacing serial IDs
- Compression policies: daily_ohlcv after 6 months, features after 3 months

### Alternatives Considered

1. **Plain PostgreSQL** — works, but time-range queries degrade as data grows. No native chunk pruning or compression.
2. **InfluxDB** — purpose-built for time-series, but weak on relational queries (joins for features + labels + predictions).
3. **DuckDB / Parquet files** — good for analytics, poor for concurrent API reads + background writes.

### Consequences

- Chunk pruning makes time-range queries fast regardless of total data volume.
- Compression (6-month policy for OHLCV) keeps storage costs low on a VPS.
- Still fully PostgreSQL — Alembic migrations, SQLAlchemy async, asyncpg all work unchanged.
- Monthly chunks are a good default for daily data; weekly would be better if we had intraday.

---

## ADR-005: HMM Regime Detection (4 States)

**Status:** Accepted  
**Date:** 2026-04

### Context

Market behavior varies dramatically between bull, bear, and sideways conditions. A model trained on bull data makes poor predictions in a crash. We need regime awareness.

### Decision

Use a **4-state Gaussian HMM** (hmmlearn) trained on Nifty 50 log returns + India VIX:
- Bull (low vol, positive drift)
- Bear (high vol, negative drift)
- Sideways (low vol, near-zero drift)
- High-volatility (high vol, direction unclear)

Regime probabilities feed into the stacking ensemble as 4 additional meta-features.

### Alternatives Considered

1. **2-state model** (bull/bear) — too coarse, misses the sideways and high-vol regimes that dominate ~40% of trading days.
2. **Rule-based regime** (e.g., 200-day MA crossover) — fragile, produces many false transitions.
3. **Markov-switching GARCH** — theoretically better, but much harder to fit reliably and overkill for 4 states.

### Consequences

- Regime state probabilities give the ensemble context — it can learn to trust different base models in different regimes.
- HMM retrained monthly (regime dynamics are slow-moving).
- The high-volatility state is critical for Indian markets where circuit breakers trigger 5-10x per year on individual stocks.

---

## ADR-006: Conformal Prediction for Price Targets

**Status:** Accepted  
**Date:** 2026-04

### Context

Point predictions ("TCS will go up 2.5%") are overconfident. Users need calibrated uncertainty: "TCS is expected between 3850 and 4050 with 90% probability." This also helps with position sizing.

### Decision

Use **MAPIE** (jackknife+ method) to produce conformal prediction intervals:
- Coverage target: 90%
- Calibration set: most recent 60 trading days (rolling)
- Output: price_low, price_mid, price_high

### Alternatives Considered

1. **Bayesian neural network** — principled uncertainty, but much slower and harder to calibrate.
2. **Quantile regression** — direct, but requires separate models per quantile and doesn't guarantee coverage.
3. **Bootstrap intervals** — computationally expensive and coverage guarantees are asymptotic only.

### Consequences

- Conformal prediction provides finite-sample coverage guarantees — the intervals are valid regardless of the underlying model's assumptions.
- Intervals automatically widen during high-volatility regimes (because calibration residuals are larger).
- The 60-day rolling calibration window adapts to recent market conditions without being too noisy.

---

## ADR-007: Demo Mode for Development

**Status:** Accepted  
**Date:** 2026-04

### Context

The full prediction pipeline requires trained models, a populated database, and Redis. This makes local development, testing, and demos impossible without significant setup.

### Decision

Implement a **demo mode** (`--demo` flag) that returns deterministic fake predictions seeded by MD5(symbol + date). Demo mode:
- Requires no database, no Redis, no trained models
- Produces consistent output for the same symbol + date
- Covers all endpoints (predict, batch, scan, paper trading, track record)

### Alternatives Considered

1. **Fixtures / seed data** — requires DB setup, doesn't work for API demos.
2. **Mock objects in tests only** — doesn't help with live demos or local development.
3. **SQLite fallback** — partial solution, but still needs model artifacts.

### Consequences

- `create_app(demo=True)` boots instantly with zero external dependencies.
- All API tests run in demo mode (no DB needed), with a separate integration test suite for real DB tests.
- The MD5 seeding means demo output is reproducible — useful for consistent test assertions.
- Demo mode is clearly labeled in API responses (`model_version: "demo-v1"`) to prevent confusion.

---

## ADR-008: File-Based Experiment Tracking

**Status:** Accepted  
**Date:** 2026-05

### Context

Every training run needs to record hyperparameters, metrics, data ranges, and artifact paths for comparison and reproducibility. The standard solution is MLflow, but that adds infrastructure.

### Decision

Use a **JSON-based ExperimentTracker** that writes one JSON file per run to `models/artifacts/experiments/`. No external server, no database, no dependencies.

### Alternatives Considered

1. **MLflow** — industry standard, but requires a server process, adds 3+ Python dependencies, and is overkill for a single-user project.
2. **Weights & Biases** — cloud-hosted, requires API key, costs money at scale.
3. **DVC** — designed for data versioning, not experiment tracking. Better for large dataset management.

### Consequences

- Zero external dependencies — just JSON files on disk.
- CLI commands (`experiment list`, `experiment compare`) provide the needed functionality.
- Easy to migrate to MLflow later if the project scales (JSON records contain all the same fields).
- Trade-off: no web UI for browsing experiments, but `experiment compare` in the terminal is sufficient for one person.

---

## ADR-009: Async-First Architecture

**Status:** Accepted  
**Date:** 2026-03

### Context

The prediction pipeline involves I/O-heavy operations: database queries, feature loading, multiple model inferences, and cache lookups. Synchronous execution serializes all of this.

### Decision

Use **async throughout**: FastAPI (async routes), SQLAlchemy (async sessions with asyncpg), Redis (aioredis), and `asyncio.gather` for batch predictions with a semaphore (10 concurrent).

### Alternatives Considered

1. **Sync Flask + psycopg2** — simpler, but prediction latency would be 3-5x higher due to serial I/O.
2. **Celery for async** — adds Redis/RabbitMQ as a task broker, heavy for request-scoped work.
3. **ThreadPoolExecutor** — works for I/O but doesn't compose as cleanly with FastAPI's native async.

### Consequences

- Single-process uvicorn handles 100+ concurrent prediction requests efficiently.
- `asyncio.gather` with semaphore prevents model inference from consuming all memory during batch scans.
- All database fixtures in tests need `pytest-asyncio` and async event loops — adds testing complexity.
- PyTorch inference is CPU-bound, so it runs in the default executor (no benefit from async there), but the surrounding I/O is async.
