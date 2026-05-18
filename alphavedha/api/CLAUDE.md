# API Layer — AlphaVedha

## Responsibility
FastAPI REST endpoints exposing predictions, scans, and system health.

## Architecture
- FastAPI with async endpoints, app factory pattern (`create_app(demo=False)`)
- Pydantic v2 for request/response validation (`schemas.py`)
- Redis for response caching with market-hours-aware TTL (`services/cache.py`)
- slowapi for rate limiting (100/min default, 10/min for batch/scan)
- API key auth via X-API-Key header (`deps.py`)
- Structured error responses with error codes

## Route Organization

```
routes/
├── health.py         # GET /health, GET /ready (no auth)
└── predictions.py    # GET /predict/{symbol}, POST /predict/batch, GET /scan/{tier} (auth required)
```

## Service Layer

```
services/
├── prediction_service.py  # PredictionService — orchestrates pipeline
├── model_registry.py      # ModelRegistry — loads real or demo models
└── cache.py               # PredictionCache — Redis with market-hours TTL
```

Both API routes and CLI commands share PredictionService. The service handles caching, model loading, and the demo/real toggle.

## Demo Mode
- `ALPHAVEDHA_DEMO=1` env var or `--demo` CLI flag
- ModelRegistry creates synthetic models (deterministic per symbol)
- No database or trained model artifacts required

## Response Standards
- Every response includes: `generated_at` (ISO 8601), `model_version`
- Prediction responses include: `direction_label`, `price_targets`, `risk` section
- Error responses: `{"error": {"code": "...", "message": "...", "details": {...}}}`
- HTTP status codes: 200 (success), 400 (bad input), 401 (missing key), 403 (invalid key), 404 (symbol not found), 422 (validation), 429 (rate limited), 500 (prediction error), 503 (models not loaded)

## Authentication
- `X-API-Key` header, key from `ALPHAVEDHA_API_KEY` env var
- If env var not set: auth disabled (local dev convenience)
- `/health` and `/ready` always public

## Rate Limiting
- Default: 100 requests/minute per client IP
- Batch/scan: 10 requests/minute
- Health/ready: no rate limit
- 429 response with Retry-After header

## Caching Strategy
- During market hours (9:15-15:30 IST): TTL = 300 seconds
- After market hours: TTL = seconds until next 9:15 AM IST
- Cache key: `predict:{symbol}:{model_version}`
- Redis unavailable: cache operates as no-op, API still functions
