# AlphaVedha API Guide

Base URL: `http://localhost:8000` (dev) or `https://your-domain.com` (prod)

## Authentication

All prediction endpoints require an API key via the `X-API-Key` header.
Health endpoints (`/health`, `/ready`, `/metrics`) are public.

```bash
# Set your key
export API_KEY="your-api-key-here"
```

## Endpoints

### Health

```bash
# Liveness check
curl http://localhost:8000/health
# → {"status": "ok", "version": "0.1.0"}

# Readiness check (verifies DB, Redis, models)
curl http://localhost:8000/ready
# → {"ready": true, "database_available": true, "cache_available": true,
#    "models_loaded": true, "model_version": "demo-v1"}
```

### Single Prediction

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/predict/TCS
```

Response:
```json
{
  "symbol": "TCS",
  "direction": 1,
  "direction_label": "BUY",
  "magnitude": 0.025,
  "composite_score": 0.72,
  "meta_confidence": 0.68,
  "is_tradeable": true,
  "regime": "bull",
  "price_targets": {"low": 3850.0, "mid": 3950.0, "high": 4050.0},
  "risk": {"position_size_pct": 5.2, "model_disagreement": 0.15},
  "model_version": "v1.0",
  "generated_at": "2026-05-21T09:30:00Z",
  "warnings": []
}
```

### Batch Prediction

```bash
curl -X POST -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["TCS", "INFY", "RELIANCE"]}' \
  http://localhost:8000/predict/batch
```

Response:
```json
{
  "predictions": [...],
  "total": 3,
  "successful": 3,
  "failed": [],
  "model_version": "v1.0",
  "generated_at": "2026-05-21T09:30:00Z"
}
```

Limits: 1-20 symbols per request.

### Scan Tier

```bash
# Scan Nifty 50, top 5
curl -H "X-API-Key: $API_KEY" "http://localhost:8000/scan/large?top_n=5"

# Scan Midcap 150
curl -H "X-API-Key: $API_KEY" "http://localhost:8000/scan/mid?top_n=10"
```

Valid tiers: `large`, `mid`, `small`, `all`
top_n range: 1-50

Response:
```json
{
  "tier": "large",
  "buy_candidates": [...],
  "sell_candidates": [...],
  "excluded": [{"symbol": "XYZ", "reason": "low liquidity"}],
  "total_scanned": 50,
  "model_version": "v1.0",
  "generated_at": "2026-05-21T09:30:00Z"
}
```

### Public Track Record

```bash
# Accuracy breakdown (no auth required)
curl http://localhost:8000/public/track-record

# Prediction history with filters
curl "http://localhost:8000/public/predictions?regime=bull&min_confidence=0.6&limit=50"

# Monthly returns
curl http://localhost:8000/public/monthly-returns

# Equity curve
curl http://localhost:8000/public/equity-curve
```

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description",
    "details": {}
  }
}
```

| Status | Code | Meaning |
|--------|------|---------|
| 400 | `INVALID_INPUT` | Bad symbol format or invalid tier |
| 401 | `MISSING_KEY` | No X-API-Key header |
| 403 | `INVALID_KEY` | Wrong API key |
| 404 | `SYMBOL_NOT_FOUND` | Symbol not in universe |
| 422 | (validation) | Request body validation failed |
| 429 | `RATE_LIMITED` | Too many requests (check Retry-After header) |
| 500 | `PREDICTION_FAILED` | Internal prediction error |
| 503 | `MODELS_NOT_LOADED` | Models not initialized yet |

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| Default | 100 requests/minute |
| Batch/Scan | 10 requests/minute |
| Health/Ready | No limit |

Rate-limited responses include a `Retry-After` header.

## Demo Mode

Start the server without trained models or database:

```bash
ALPHAVEDHA_DEMO=1 uvicorn alphavedha.api.app:create_app --factory
# or
alphavedha serve --demo
```

Demo mode returns deterministic synthetic predictions (seeded by symbol + date).
