"""Health and readiness endpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter

from alphavedha.api.deps import get_service

router = APIRouter(tags=["health"])

_GIT_SHA = os.environ.get("GIT_SHA", "unknown")


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check — always returns ok if the server is running."""
    return {"status": "ok", "version": "0.1.0", "sha": _GIT_SHA}


@router.get("/ready")
async def ready() -> dict[str, object]:
    """Readiness check — reports model, cache, and database status."""
    service = get_service()
    cache_ok = await service._cache.health_check()

    db_ok = False
    try:
        from alphavedha.data.database import check_health

        db_ok = await check_health()
    except Exception:
        pass

    models_loaded = service._registry.models_available()

    all_ok = cache_ok and db_ok and models_loaded
    return {
        "ready": all_ok,
        "models_loaded": models_loaded,
        "cache_available": cache_ok,
        "database_available": db_ok,
        "model_version": service._registry.model_version,
    }
