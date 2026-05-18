# Week 8: API + CLI Design Spec

## Goal

Wire up FastAPI REST endpoints and Typer CLI commands to the AlphaVedha prediction pipeline. Both interfaces share a common service layer. A `--demo` flag enables standalone operation with synthetic predictions (no trained models or database required).

## Architecture

```
CLI commands (Typer)  ─┐
                       ├──▶ PredictionService ──▶ PredictionEngine ──▶ StockPrediction
API routes (FastAPI)  ─┘         │                      │
                          ModelRegistry           CompositeScorer
                          (real / demo)           StockRanker
                          PredictionCache         RiskManager
                          (Redis)
```

A thin **service layer** (`alphavedha/services/`) sits between the transport layer (API routes, CLI commands) and the ML pipeline. The service handles model loading, feature fetching, caching, and the demo/real toggle. Routes and commands stay thin — validate input, call service, format output.

## Tech Stack

- **FastAPI** — REST API with async endpoints
- **Typer + Rich** — CLI with colored tables, panels, progress bars
- **Redis** — Response caching with market-hours-aware TTL
- **slowapi** — Rate limiting (built on `limits` library)
- **Pydantic v2** — Request/response validation

## File Structure

```
alphavedha/
├── api/
│   ├── app.py              # FastAPI app factory, lifespan, middleware, exception handlers
│   ├── deps.py             # Dependency injection (get_service, verify_api_key)
│   ├── schemas.py          # Pydantic response models (PredictionResponse, ScanResponse, etc.)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predictions.py  # GET /predict/{symbol}, POST /predict/batch, GET /scan/{tier}
│   │   └── health.py       # GET /health, GET /ready
│   └── __init__.py
├── cli/
│   ├── main.py             # Typer app with commands: predict, scan, serve, data
│   ├── formatters.py       # Rich output: prediction panels, ranking tables, progress bars
│   └── __init__.py
├── services/
│   ├── prediction_service.py  # PredictionService — central orchestrator
│   ├── model_registry.py      # ModelRegistry — load real or demo models
│   ├── cache.py               # PredictionCache — Redis with market-hours TTL
│   └── __init__.py
```

---

## Service Layer

### PredictionService

Central orchestrator shared by API and CLI.

```python
class PredictionService:
    def __init__(
        self,
        registry: ModelRegistry,
        cache: PredictionCache,
        config: AppConfig,
    ) -> None: ...

    async def predict_single(self, symbol: str, sector: str = "") -> StockPrediction:
        """Predict a single stock. Checks cache first."""

    async def scan_tier(self, tier: str, top_n: int = 10) -> RankingResult:
        """Get symbols for tier, predict all, rank and return top candidates."""

    async def predict_batch(self, symbols: list[str]) -> list[StockPrediction]:
        """Predict multiple symbols. Returns list (order preserved)."""
```

**`predict_single` flow:**
1. Compute cache key: `predict:{symbol}:{model_version}`
2. Check cache — return on hit
3. Load features for symbol (real: from feature store / synthetic: generated)
4. Get `PredictionEngine` from registry
5. Call `engine.predict(symbol, features, sector=sector)`
6. Cache result with market-hours TTL
7. Return `StockPrediction`

**`scan_tier` flow:**
1. Get symbol list for tier (real: `get_symbols_for_tier()` / demo: hardcoded NIFTY 50 sample)
2. Call `predict_single()` for each symbol (sequential — not parallel, to respect rate limits)
3. Collect results, pass to `StockRanker.rank(predictions, top_n=top_n)`
4. Return `RankingResult`

### ModelRegistry

Manages the demo/real toggle for model instantiation.

```python
class ModelRegistry:
    def __init__(self, demo: bool = False, artifact_dir: Path | None = None) -> None: ...

    def get_prediction_engine(self) -> PredictionEngine:
        """Return PredictionEngine with real or demo models."""

    @property
    def model_version(self) -> str:
        """'demo-v0.1.0' in demo mode, actual version from artifacts otherwise."""
```

**Demo mode:** Creates mock models that return plausible synthetic predictions:
- Base models: random direction (weighted 40% buy, 20% sell, 40% hold), magnitude 0.01-0.05
- Regime: cycles through bull/bear/sideways/high_volatility based on symbol hash (deterministic per symbol)
- Ensemble/meta/conformal: pass through with realistic confidence values
- Uses `CompositeScorer` and `RiskManager` as real (they're stateless, no artifacts needed)

**Real mode:** Loads trained model artifacts from `models/artifacts/` directory. Raises `ModelNotFoundError` if artifacts don't exist.

### PredictionCache

Redis-backed with market-hours awareness.

```python
class PredictionCache:
    def __init__(self, redis_url: str | None = None) -> None:
        """redis_url=None disables caching entirely."""

    async def get(self, key: str) -> StockPrediction | None: ...
    async def set(self, key: str, prediction: StockPrediction) -> None: ...
    async def health_check(self) -> bool: ...

    def _compute_ttl(self) -> int:
        """Market hours (9:15-15:30 IST): 300s. After hours: seconds until next 9:15 IST."""
```

Cache serialization: JSON via `dataclasses.asdict()` with numpy array → list conversion. Deserialization reconstructs `StockPrediction`.

If Redis is unavailable at startup, the cache logs a warning and operates as a no-op (all gets return None, all sets are silent). The API still functions — just without caching.

---

## API Design

### Endpoints

| Method | Path | Auth | Rate Limit | Description |
|--------|------|------|------------|-------------|
| `GET` | `/health` | No | None | Liveness check |
| `GET` | `/ready` | No | None | Readiness (models loaded, redis reachable) |
| `GET` | `/predict/{symbol}` | Yes | 100/min | Single stock prediction |
| `POST` | `/predict/batch` | Yes | 10/min | Batch predictions (max 20 symbols) |
| `GET` | `/scan/{tier}` | Yes | 10/min | Scan & rank a tier (large/mid/small) |

### Authentication

Simple API key via `X-API-Key` header.

- Key source: `ALPHAVEDHA_API_KEY` environment variable
- If env var is not set: auth is disabled (all requests allowed) — for local dev convenience
- `/health` and `/ready` are always public
- Missing key → 401 Unauthorized
- Invalid key → 403 Forbidden

Implemented as a FastAPI dependency (`verify_api_key` in `deps.py`).

### Response Models

**PredictionResponse** (for `/predict/{symbol}` and each item in batch/scan):

```json
{
  "symbol": "TCS",
  "direction": 1,
  "direction_label": "BUY",
  "magnitude": 0.03,
  "composite_score": 78.5,
  "meta_confidence": 0.72,
  "is_tradeable": true,
  "regime": "bull",
  "price_targets": {
    "low": 3850.0,
    "mid": 3920.0,
    "high": 3990.0
  },
  "risk": {
    "position_size_pct": 5.0,
    "model_disagreement": 0.05
  },
  "model_version": "v0.1.0",
  "generated_at": "2026-05-16T15:30:00+05:30",
  "warnings": []
}
```

**BatchResponse** (for `/predict/batch`):

```json
{
  "predictions": [PredictionResponse, ...],
  "total": 5,
  "successful": 4,
  "failed": [{"symbol": "INVALID", "error": "Symbol not found"}],
  "model_version": "v0.1.0",
  "generated_at": "2026-05-16T15:30:00+05:30"
}
```

**ScanResponse** (for `/scan/{tier}`):

```json
{
  "tier": "large",
  "buy_candidates": [PredictionResponse, ...],
  "sell_candidates": [PredictionResponse, ...],
  "excluded": [{"symbol": "HDFC", "reason": "hold signal"}],
  "total_scanned": 50,
  "model_version": "v0.1.0",
  "generated_at": "2026-05-16T15:30:00+05:30"
}
```

**ErrorResponse:**

```json
{
  "error": {
    "code": "SYMBOL_NOT_FOUND",
    "message": "Symbol 'INVALID' not found in universe",
    "details": {}
  }
}
```

Error codes: `SYMBOL_NOT_FOUND`, `PREDICTION_FAILED`, `RATE_LIMITED`, `INVALID_INPUT`, `MODELS_NOT_LOADED`, `INTERNAL_ERROR`.

### Rate Limiting

Using `slowapi` with `limits` library:
- Key function: client IP from request
- Default: 100 requests/minute
- Batch and scan: 10 requests/minute
- Health/ready: no limit
- Exceeded → 429 with `Retry-After` header

### Exception Handling

Global exception handlers in `app.py` map AlphaVedha exceptions to HTTP responses:
- `SymbolNotFoundError` → 404
- `PredictionError` → 500 with `PREDICTION_FAILED` code
- `ModelNotFoundError` → 503 with `MODELS_NOT_LOADED` code
- `ValueError` (bad input) → 400
- Unhandled → 500 with `INTERNAL_ERROR`

All errors logged via structlog before returning the response.

### App Factory & Lifespan

```python
def create_app(demo: bool = False) -> FastAPI:
    """Create FastAPI app. demo=True uses synthetic models."""
```

Uses FastAPI lifespan for startup/shutdown:
- **Startup:** Initialize ModelRegistry, PredictionCache (Redis connection), log config
- **Shutdown:** Close Redis connection, cleanup

The `demo` flag is passed via environment variable `ALPHAVEDHA_DEMO=1` or programmatically in tests.

---

## CLI Design

### Commands

```
alphavedha predict <SYMBOL>
    --json          Output as JSON instead of Rich panel
    --demo          Use synthetic predictions
    --config PATH   Custom config file

alphavedha scan <TIER>
    --top-n INT     Number of top candidates (default: 10)
    --json          Output as JSON
    --demo          Use synthetic predictions
    --config PATH   Custom config file

alphavedha serve
    --host TEXT     Bind host (default: 0.0.0.0)
    --port INT      Bind port (default: 8000)
    --demo          Start in demo mode
    --reload        Auto-reload on code changes (dev mode)

alphavedha data refresh      # Fetch latest market data
alphavedha data backfill     # Historical backfill
    --start DATE    Start date (default: 2005-01-01)
alphavedha data status       # Show data freshness
```

`predict` and `scan` run the pipeline directly (no server needed). `serve` starts the FastAPI server via uvicorn.

**Note on `data` subcommands:** These call existing data layer functions directly (`universe.refresh_universe()`, etc.) — they don't go through PredictionService. They remain thin CLI wrappers over the data pipeline and are out of scope for Week 8 implementation. They stay as stubs with a "not yet wired" message until the data pipeline has real DB integration.

### Rich Formatters

**`format_prediction(pred: StockPrediction) -> Panel`**
- Header: symbol + direction badge (green BUY / red SELL / yellow HOLD)
- Body: composite score (with bar), meta confidence, regime, price targets, risk section
- Footer: model version + timestamp + warnings

**`format_ranking(result: RankingResult) -> Table`**
- Columns: Rank, Symbol, Direction, Score, Position %, Regime
- Direction column colored (green/red/yellow)
- Excluded stocks listed below the table

**`format_scan_progress(current, total) -> Progress`**
- Rich progress bar: `Scanning [15/50] ████████░░░░ 30%`

### Global Options

Handled via Typer callback on the app:
- `--demo` sets a context variable used by all commands
- `--config` overrides the config path (clears `get_config` cache)

---

## Demo Mode

When `--demo` is active (CLI) or `ALPHAVEDHA_DEMO=1` (API):

1. **ModelRegistry** creates mock models with deterministic synthetic output
2. **PredictionService** generates synthetic features instead of querying the DB
3. **Cache** still works (Redis if available, otherwise no-op)
4. **Universe** uses a hardcoded list of 10-15 well-known Indian stocks (TCS, INFY, RELIANCE, HDFC, etc.) instead of querying the database

Demo predictions are deterministic per symbol (seeded by symbol hash) so repeated calls return consistent results. This makes testing and demos predictable.

---

## Dependencies

### New (add to pyproject.toml)

- `slowapi>=0.1.9` — rate limiting for FastAPI

### Already installed

- `fastapi>=0.111`
- `uvicorn[standard]>=0.30`
- `typer[all]>=0.12` (includes `rich`)
- `redis>=5.0`
- `httpx>=0.27`

---

## Testing Strategy

### Test Files

```
tests/unit/
├── api/
│   ├── test_schemas.py            # ~5 tests: response model validation
│   ├── test_predictions.py        # ~8 tests: route logic, auth, errors
│   └── test_health.py             # ~3 tests: health/ready endpoints
├── cli/
│   ├── test_commands.py           # ~8 tests: predict, scan, serve invocation
│   └── test_formatters.py         # ~4 tests: Rich output correctness
└── services/
    ├── test_prediction_service.py # ~6 tests: predict, scan, batch logic
    ├── test_model_registry.py     # ~4 tests: demo mode, real mode errors
    └── test_cache.py              # ~5 tests: TTL logic, market hours, Redis mock
```

**Estimated: ~43 tests across 8 test files.**

### Test Approach

- **API routes:** FastAPI `TestClient` with mocked `PredictionService` via dependency override. Tests: valid predictions, auth (401/403), bad symbols (404), rate limiting (429), batch validation, response schema.
- **CLI commands:** Typer `CliRunner` with mocked service. Tests: exit codes, output content, `--json` flag, `--demo` flag.
- **Services:** Unit tests with mocked dependencies. PredictionService gets mocked registry/cache. Cache gets mocked Redis (use `fakeredis`). ModelRegistry tested for demo vs real mode.
- **Auth:** Missing key → 401, invalid key → 403, valid key → 200, no env var → auth disabled, health → no auth.

### Test Dependencies

- `fakeredis[lua]` — for testing Redis cache without running Redis
- Already have: `pytest`, `pytest-asyncio`, `httpx` (for `TestClient`)
