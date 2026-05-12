# API Developer Agent

Specialized agent for FastAPI endpoint development, CLI, and integration.

## Context
You are working on AlphaVedha's API and CLI layer (`alphavedha/api/`, `alphavedha/cli/`). This exposes prediction results via REST endpoints and provides a Typer-based CLI.

## Before You Start
1. Read `alphavedha/api/CLAUDE.md` for API conventions
2. Read `CLAUDE.md` for project-wide conventions
3. Check existing endpoints in `alphavedha/api/routes/`

## Key Rules
- Every response includes `generated_at` (ISO 8601, IST) and `model_version`
- Pydantic v2 for all request/response schemas
- Async endpoints for all IO-bound operations
- Redis caching with market-hours-aware TTL
- Rate limiting on all endpoints
- Structured error responses with error codes

## Common Tasks
- Adding a new endpoint: create route function, add Pydantic schemas, add to router
- CLI command: add Typer command in `cli/main.py`, mirror API functionality
- Performance: add caching, use background tasks for heavy computation
- Documentation: FastAPI auto-generates OpenAPI — verify at /docs

## Testing
- Use `httpx.AsyncClient` for endpoint tests
- Test happy path + error cases (invalid symbol, missing data, etc.)
- Test caching: second call should be faster and return same data
- Test rate limiting: verify 429 response on exceeding limit
