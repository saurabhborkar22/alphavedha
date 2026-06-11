"""PredictionCache — Redis cache with market-hours-aware TTL."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import structlog

from alphavedha.prediction.engine import StockPrediction

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN = (9, 15)
_MARKET_CLOSE = (15, 30)
_MARKET_HOURS_TTL = 300
_MAX_LOCAL_ENTRIES = 256


def _now_ist() -> datetime:
    """Return current time in IST. Extracted for testability."""
    return datetime.now(IST)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and datetimes."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def _deserialize_prediction(data: dict[str, Any]) -> StockPrediction:
    """Reconstruct a StockPrediction from a JSON-decoded dict."""
    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
    data["regime_probabilities"] = np.array(data["regime_probabilities"])
    return StockPrediction(**data)


class PredictionCache:
    """Redis-backed prediction cache with market-hours-aware TTL.

    During Indian market hours (9:15-15:30 IST, Mon-Fri), predictions are
    cached for 300 seconds.  Outside market hours the TTL extends until the
    next market open so stale data is never served past a session boundary.

    Pass ``redis_client=None`` to disable caching entirely (no-op mode).
    When a Redis client is configured but unreachable, a bounded in-process
    cache takes over so repeated predictions aren't recomputed from scratch.
    """

    def __init__(self, redis_client: Any | None = None) -> None:
        self._redis = redis_client
        self._local: OrderedDict[str, tuple[float, str]] = OrderedDict()

    async def get(self, key: str) -> StockPrediction | None:
        """Fetch a cached prediction by key. Returns None on miss or error."""
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(key)
        except Exception as exc:
            logger.warning("cache_get_failed", key=key, error=str(exc))
            raw = self._local_get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return _deserialize_prediction(data)
        except Exception as exc:
            logger.warning("cache_deserialize_failed", key=key, error=str(exc))
            return None

    async def set(self, key: str, prediction: StockPrediction) -> None:
        """Store a prediction with a market-hours-aware TTL."""
        if self._redis is None:
            return
        try:
            data = asdict(prediction)
            raw = json.dumps(data, cls=_NumpyEncoder)
            ttl = self._compute_ttl()
        except Exception as exc:
            logger.warning("cache_serialize_failed", key=key, error=str(exc))
            return
        try:
            await self._redis.set(key, raw, ex=ttl)
        except Exception as exc:
            logger.warning("cache_set_failed", key=key, error=str(exc))
            self._local_set(key, raw, ttl)

    def _local_get(self, key: str) -> str | None:
        item = self._local.get(key)
        if item is None:
            return None
        expiry, raw = item
        if time.monotonic() > expiry:
            del self._local[key]
            return None
        return raw

    def _local_set(self, key: str, raw: str, ttl: int) -> None:
        self._local[key] = (time.monotonic() + ttl, raw)
        self._local.move_to_end(key)
        while len(self._local) > _MAX_LOCAL_ENTRIES:
            self._local.popitem(last=False)

    async def health_check(self) -> bool:
        """Return True if Redis is reachable."""
        if self._redis is None:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False

    @staticmethod
    def _compute_ttl() -> int:
        """Compute TTL in seconds based on current IST time.

        - During market hours (9:15-15:30 IST, weekdays): 300 s
        - Outside market hours: seconds until next market open
        - Minimum TTL is always ``_MARKET_HOURS_TTL`` (300 s)
        """
        now = _now_ist()
        weekday = now.weekday()

        # Weekend (Saturday=5, Sunday=6): until Monday 9:15 AM
        if weekday >= 5:
            days_until_monday = 7 - weekday
            next_open = now.replace(
                hour=_MARKET_OPEN[0],
                minute=_MARKET_OPEN[1],
                second=0,
                microsecond=0,
            ) + timedelta(days=days_until_monday)
            return max(int((next_open - now).total_seconds()), _MARKET_HOURS_TTL)

        market_open = now.replace(
            hour=_MARKET_OPEN[0],
            minute=_MARKET_OPEN[1],
            second=0,
            microsecond=0,
        )
        market_close = now.replace(
            hour=_MARKET_CLOSE[0],
            minute=_MARKET_CLOSE[1],
            second=0,
            microsecond=0,
        )

        # During market hours
        if market_open <= now <= market_close:
            return _MARKET_HOURS_TTL

        # After market close: next trading day 9:15 AM
        if now > market_close:
            if weekday == 4:  # Friday → Monday
                next_open = market_open + timedelta(days=3)
            else:
                next_open = market_open + timedelta(days=1)
        else:
            # Before market open same day
            next_open = market_open

        return max(int((next_open - now).total_seconds()), _MARKET_HOURS_TTL)
