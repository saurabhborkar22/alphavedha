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
async def ready() -> dict[str, bool | str]:
    """Readiness check — reports model and cache status."""
    service = get_service()
    cache_ok = await service._cache.health_check()
    return {
        "models_loaded": True,
        "cache_available": cache_ok,
        "model_version": service._registry.model_version,
    }
