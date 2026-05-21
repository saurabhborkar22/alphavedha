"""FastAPI dependency injection — service provider and API key auth."""

from __future__ import annotations

import hashlib
import hmac
import os

import structlog
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from alphavedha.services.prediction_service import PredictionService

logger = structlog.get_logger(__name__)

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


def _get_valid_api_keys() -> list[str]:
    """Return list of valid API keys from environment.

    Supports key rotation: ALPHAVEDHA_API_KEY is the primary key,
    ALPHAVEDHA_API_KEY_SECONDARY is the rollover key (set the new key here
    first, then promote it to primary and remove secondary).
    """
    keys = []
    primary = os.environ.get("ALPHAVEDHA_API_KEY", "")
    if primary:
        keys.append(primary)
    secondary = os.environ.get("ALPHAVEDHA_API_KEY_SECONDARY", "")
    if secondary:
        keys.append(secondary)
    return keys


def verify_api_key(api_key: str | None = Security(_api_key_header)) -> str | None:
    """Verify the X-API-Key header against configured API keys.

    If no keys configured, all requests pass through (local dev).
    Supports dual keys for zero-downtime rotation.
    """
    valid_keys = _get_valid_api_keys()
    if not valid_keys:
        return None
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    if not any(hmac.compare_digest(api_key, k) for k in valid_keys):
        logger.warning("invalid_api_key_attempt", key_prefix=api_key[:4] + "..." if api_key else "")
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


def hash_api_key(key: str) -> str:
    """One-way hash an API key for safe logging/storage."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]
