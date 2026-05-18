"""FastAPI dependency injection — service provider and API key auth."""

from __future__ import annotations

import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from alphavedha.services.prediction_service import PredictionService

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_service_instance: PredictionService | None = None


def set_service(service: PredictionService) -> None:
    """Set the global PredictionService instance (called during lifespan startup)."""
    global _service_instance
    _service_instance = service


def get_service() -> PredictionService:
    """Return the current PredictionService, or 503 if not initialized."""
    if _service_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _service_instance


def verify_api_key(api_key: str | None = Security(_api_key_header)) -> str | None:
    """Verify the X-API-Key header against ALPHAVEDHA_API_KEY env var.

    If ALPHAVEDHA_API_KEY is not set (or empty), all requests pass through.
    If set, requests must provide a matching key.
    """
    expected = os.environ.get("ALPHAVEDHA_API_KEY")
    if not expected:
        return None
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    if api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
