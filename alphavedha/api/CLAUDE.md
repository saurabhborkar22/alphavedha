# API Layer — AlphaVedha

## Responsibility
FastAPI REST endpoints exposing predictions, scans, backtests, and system health.

## Architecture
- FastAPI with async endpoints
- Pydantic v2 for request/response validation
- Redis for response caching during market hours
- Structured error responses with error codes

## Route Organization

```
routes/
├── predictions.py    # /predict/{symbol}, /predict/batch, /scan/{tier}
├── universe.py       # /universe/{tier}, /constituents
├── backtest.py       # /backtest/{symbol}, /backtest/portfolio
└── health.py         # /health, /metrics, /drift
```

## Response Standards
- Every response includes: `generated_at` (ISO 8601, IST timezone), `model_version`
- Prediction responses include: `meta_confidence`, `regime`, `risk` section
- Error responses: `{"error": {"code": "...", "message": "...", "details": {...}}}`
- HTTP status codes: 200 (success), 400 (bad input), 404 (symbol not found), 429 (rate limited), 500 (server error)

## Caching Strategy
- During market hours (9:15-15:30 IST): cache TTL = 5 minutes
- After market hours: cache TTL = until next market open
- Cache key: `{endpoint}:{params_hash}:{model_version}`
- Cache invalidation: on new data ingestion or model promotion

## Rate Limiting
- Default: 100 requests/minute per client
- Batch endpoint: 10 requests/minute (heavy computation)
- Health/metrics: no rate limit

## Rules
- Never expose internal model details (weights, architecture) via API
- Log all prediction requests (for tracking accuracy later)
- API must work without Redis (fallback to in-memory cache)
- Include OpenAPI docs at /docs (FastAPI auto-generates this)
