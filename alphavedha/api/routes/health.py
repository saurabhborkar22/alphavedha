"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from alphavedha.api.deps import get_service

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check — always returns ok if the server is running."""
    return {"status": "ok", "version": "0.1.0"}


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

    all_ok = cache_ok and db_ok
    return {
        "ready": all_ok,
        "models_loaded": True,
        "cache_available": cache_ok,
        "database_available": db_ok,
        "model_version": service._registry.model_version,
    }
