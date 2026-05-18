"""Tests for PredictionCache — Redis caching with market-hours TTL."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from alphavedha.prediction.engine import StockPrediction
from alphavedha.services.cache import PredictionCache

IST = ZoneInfo("Asia/Kolkata")


def _make_prediction(symbol: str = "TCS") -> StockPrediction:
    return StockPrediction(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        direction=1,
        magnitude=0.03,
        composite_score=78.5,
        meta_confidence=0.72,
        is_tradeable=True,
        regime="bull",
        regime_probabilities=np.array([0.7, 0.1, 0.1, 0.1]),
        price_target_low=95.0,
        price_target_mid=100.0,
        price_target_high=105.0,
        model_disagreement=0.05,
        position_size_pct=5.0,
        model_version="v0.1.0",
        warnings=[],
    )


class TestPredictionCache:
    @pytest.fixture
    def cache(self) -> PredictionCache:
        import fakeredis.aioredis

        fake_redis = fakeredis.aioredis.FakeRedis()
        return PredictionCache(redis_client=fake_redis)

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self, cache: PredictionCache) -> None:
        result = await cache.get("predict:TCS:v0.1.0")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_roundtrip(self, cache: PredictionCache) -> None:
        pred = _make_prediction("TCS")
        await cache.set("predict:TCS:v0.1.0", pred)
        cached = await cache.get("predict:TCS:v0.1.0")
        assert cached is not None
        assert cached.symbol == "TCS"
        assert cached.direction == 1
        assert cached.composite_score == 78.5
        np.testing.assert_array_almost_equal(
            cached.regime_probabilities, [0.7, 0.1, 0.1, 0.1]
        )

    @pytest.mark.asyncio
    async def test_health_check_with_fake_redis(self, cache: PredictionCache) -> None:
        result = await cache.health_check()
        assert result is True

    def test_ttl_during_market_hours(self) -> None:
        market_time = datetime(2026, 5, 18, 11, 0, tzinfo=IST)  # Monday 11 AM IST
        with patch("alphavedha.services.cache._now_ist", return_value=market_time):
            ttl = PredictionCache._compute_ttl()
        assert ttl == 300

    def test_ttl_after_market_hours_same_day(self) -> None:
        after_hours = datetime(2026, 5, 18, 16, 0, tzinfo=IST)  # Monday 4 PM IST
        with patch("alphavedha.services.cache._now_ist", return_value=after_hours):
            ttl = PredictionCache._compute_ttl()
        # Should be seconds until next day 9:15 AM IST (Tuesday)
        assert ttl > 3600
        assert ttl < 86400

    def test_ttl_on_weekend(self) -> None:
        # Saturday 10:00 AM IST
        saturday = datetime(2026, 5, 23, 10, 0, tzinfo=IST)
        with patch("alphavedha.services.cache._now_ist", return_value=saturday):
            ttl = PredictionCache._compute_ttl()
        # Should be seconds until Monday 9:15 AM IST
        assert ttl > 86400  # more than 1 day

    def test_ttl_friday_after_close(self) -> None:
        # Friday 4 PM IST
        friday_evening = datetime(2026, 5, 22, 16, 0, tzinfo=IST)
        with patch("alphavedha.services.cache._now_ist", return_value=friday_evening):
            ttl = PredictionCache._compute_ttl()
        # Should be seconds until Monday 9:15 AM IST (~65+ hours)
        assert ttl > 2 * 86400

    def test_ttl_before_market_open(self) -> None:
        # Monday 8:00 AM IST (before market open)
        early_morning = datetime(2026, 5, 18, 8, 0, tzinfo=IST)
        with patch("alphavedha.services.cache._now_ist", return_value=early_morning):
            ttl = PredictionCache._compute_ttl()
        # Should be seconds until 9:15 AM IST (~75 min = 4500 sec)
        assert 4400 < ttl < 4600

    @pytest.mark.asyncio
    async def test_disabled_cache_returns_none(self) -> None:
        cache = PredictionCache(redis_client=None)
        result = await cache.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_cache_set_is_noop(self) -> None:
        cache = PredictionCache(redis_client=None)
        pred = _make_prediction("TCS")
        await cache.set("key", pred)  # should not raise

    @pytest.mark.asyncio
    async def test_disabled_cache_health_check_false(self) -> None:
        cache = PredictionCache(redis_client=None)
        result = await cache.health_check()
        assert result is False
