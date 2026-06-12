"""AlphaVedha FastAPI application factory."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from alphavedha.api.deps import set_service
from alphavedha.api.routes import (
    dashboard,
    health,
    live,
    paper_trading,
    predictions,
    public,
    sectors,
    signals,
    ui_support,
    sentiment,
)
from alphavedha.config import get_config
from alphavedha.exceptions import (
    ModelNotFoundError,
    PredictionError,
    SymbolNotFoundError,
)
from alphavedha.services.cache import PredictionCache
from alphavedha.services.model_registry import ModelRegistry
from alphavedha.services.prediction_service import PredictionService

logger = structlog.get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a structured 429 response with Retry-After header."""
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "RATE_LIMITED",
                "message": f"Rate limit exceeded: {exc.detail}",
                "details": {},
            }
        },
        headers={"Retry-After": str(getattr(exc, "retry_after", 60))},
    )


def create_app(demo: bool | None = None) -> FastAPI:
    """Build and return a configured FastAPI application.

    Args:
        demo: If True, use synthetic mock models. If None, reads
              ALPHAVEDHA_DEMO env var.
    """
    if demo is None:
        demo = os.environ.get("ALPHAVEDHA_DEMO", "").lower() in ("1", "true", "yes")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = get_config()
        registry = ModelRegistry(demo=demo)

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        redis_client = None
        try:
            import redis.asyncio as aioredis

            redis_client = aioredis.from_url(redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info("redis_connected", url=redis_url)
        except Exception as e:
            logger.warning("redis_unavailable", error=str(e))
            redis_client = None

        cache = PredictionCache(redis_client=redis_client)
        service = PredictionService(registry=registry, cache=cache, config=config)
        set_service(service)
        if not demo:
            await service.warm_up()
        logger.info("app_started", demo=demo, model_version=registry.model_version)
        yield
        if redis_client is not None:
            await redis_client.aclose()

    app = FastAPI(
        title="AlphaVedha API",
        description="AI-powered Indian stock market prediction engine for NSE/BSE",
        version="0.1.0",
        lifespan=lifespan,
    )

    cors_origins = os.environ.get("ALPHAVEDHA_CORS_ORIGINS", "").strip()
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in cors_origins.split(",")],
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

    @app.exception_handler(SymbolNotFoundError)
    async def symbol_not_found_handler(request: Request, exc: SymbolNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "SYMBOL_NOT_FOUND",
                    "message": str(exc),
                    "details": {},
                }
            },
        )

    @app.exception_handler(PredictionError)
    async def prediction_error_handler(request: Request, exc: PredictionError) -> JSONResponse:
        logger.error("prediction_failed", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "PREDICTION_FAILED",
                    "message": str(exc),
                    "details": {},
                }
            },
        )

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "MODELS_NOT_LOADED",
                    "message": str(exc),
                    "details": {},
                }
            },
        )

    app.include_router(health.router)
    app.include_router(sectors.router)
    app.include_router(live.router)
    app.include_router(signals.router)
    app.include_router(sentiment.router)
    app.include_router(ui_support.router)  # registered first so demo scan/intraday take precedence
    app.include_router(predictions.router)
    app.include_router(paper_trading.router)
    app.include_router(dashboard.router)
    app.include_router(public.router)

    Instrumentator(
        should_group_status_codes=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


# Module-level instance for uvicorn: `uvicorn alphavedha.api.app:app`
app = create_app()
