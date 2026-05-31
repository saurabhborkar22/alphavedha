"""Tests for LiveDataPoller and is_market_open."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from alphavedha.data.live_feed import (
    CACHE_INVALIDATE_EVERY_N_TICKS,
    POLL_INTERVAL_SECONDS,
    LiveDataPoller,
    PollResult,
    is_market_open,
)

IST = ZoneInfo("Asia/Kolkata")


class TestIsMarketOpen:
    def _ist(self, weekday: int, hour: int, minute: int) -> datetime:
        """Build an IST datetime on a specific weekday (0=Mon, 6=Sun)."""
        # Use a fixed date: Mon 2026-05-18 is weekday 0
        base = datetime(2026, 5, 18, tzinfo=IST)
        from datetime import timedelta

        return (base + timedelta(days=weekday)).replace(hour=hour, minute=minute, second=0)

    def test_open_during_market_hours(self) -> None:
        with patch("alphavedha.data.live_feed.datetime") as mock_dt:
            mock_dt.now.return_value = self._ist(0, 11, 0)
            assert is_market_open() is True

    def test_closed_before_market_open(self) -> None:
        with patch("alphavedha.data.live_feed.datetime") as mock_dt:
            mock_dt.now.return_value = self._ist(0, 9, 0)
            assert is_market_open() is False

    def test_closed_after_market_close(self) -> None:
        with patch("alphavedha.data.live_feed.datetime") as mock_dt:
            mock_dt.now.return_value = self._ist(0, 16, 0)
            assert is_market_open() is False

    def test_closed_on_saturday(self) -> None:
        with patch("alphavedha.data.live_feed.datetime") as mock_dt:
            mock_dt.now.return_value = self._ist(5, 11, 0)  # Saturday
            assert is_market_open() is False

    def test_closed_on_sunday(self) -> None:
        with patch("alphavedha.data.live_feed.datetime") as mock_dt:
            mock_dt.now.return_value = self._ist(6, 11, 0)  # Sunday
            assert is_market_open() is False

    def test_open_at_exact_open(self) -> None:
        with patch("alphavedha.data.live_feed.datetime") as mock_dt:
            mock_dt.now.return_value = self._ist(0, 9, 15)
            assert is_market_open() is True

    def test_open_at_exact_close(self) -> None:
        with patch("alphavedha.data.live_feed.datetime") as mock_dt:
            mock_dt.now.return_value = self._ist(0, 15, 30)
            assert is_market_open() is True


class TestLiveDataPoller:
    def _make_poller(self, symbols: list[str] | None = None) -> LiveDataPoller:
        session_factory = MagicMock()
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=MagicMock(execute=AsyncMock(), commit=AsyncMock()))
        cm.__aexit__ = AsyncMock(return_value=False)
        session_factory.return_value = cm
        return LiveDataPoller(
            symbols=symbols or ["TCS.NS"],
            session_factory=session_factory,
        )

    def test_constants(self) -> None:
        assert POLL_INTERVAL_SECONDS == 120.0
        assert CACHE_INVALIDATE_EVERY_N_TICKS == 5

    def test_initial_tick_count_zero(self) -> None:
        poller = self._make_poller()
        assert poller.tick_count == 0

    @pytest.mark.asyncio
    async def test_poll_once_increments_tick(self) -> None:
        poller = self._make_poller()
        fast_info = {
            "open": 3500.0,
            "high": 3550.0,
            "low": 3480.0,
            "last_price": 3520.0,
            "volume": 100000,
        }
        with patch("alphavedha.data.live_feed._fetch_fast_info", return_value=fast_info):
            await poller.poll_once()
        assert poller.tick_count == 1

    @pytest.mark.asyncio
    async def test_poll_once_returns_results(self) -> None:
        poller = self._make_poller(["TCS.NS", "INFY.NS"])
        fast_info = {
            "open": 3500.0,
            "high": 3550.0,
            "low": 3480.0,
            "last_price": 3520.0,
            "volume": 100000,
        }
        with patch("alphavedha.data.live_feed._fetch_fast_info", return_value=fast_info):
            results = await poller.poll_once()
        assert len(results) == 2
        assert all(isinstance(r, PollResult) for r in results)

    @pytest.mark.asyncio
    async def test_zero_last_price_is_failure(self) -> None:
        poller = self._make_poller()
        fast_info = {"open": 0, "high": 0, "low": 0, "last_price": 0, "volume": 0}
        with patch("alphavedha.data.live_feed._fetch_fast_info", return_value=fast_info):
            results = await poller.poll_once()
        assert results[0].success is False
        assert results[0].error is not None

    @pytest.mark.asyncio
    async def test_fetch_exception_is_failure(self) -> None:
        poller = self._make_poller()
        with patch(
            "alphavedha.data.live_feed._fetch_fast_info", side_effect=RuntimeError("timeout")
        ):
            results = await poller.poll_once()
        assert results[0].success is False
        assert "timeout" in results[0].error

    @pytest.mark.asyncio
    async def test_cache_not_invalidated_before_5_ticks(self) -> None:
        redis = AsyncMock()
        poller = LiveDataPoller(
            symbols=["TCS.NS"],
            session_factory=self._make_poller()._session_factory,
            redis_client=redis,
        )
        fast_info = {"open": 1.0, "high": 1.0, "low": 1.0, "last_price": 100.0, "volume": 1}
        with patch("alphavedha.data.live_feed._fetch_fast_info", return_value=fast_info):
            for _ in range(4):
                await poller.poll_once()
        redis.keys.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_invalidated_on_5th_tick(self) -> None:
        redis = AsyncMock()
        redis.keys = AsyncMock(return_value=[b"predict:TCS.NS:v1"])
        poller = LiveDataPoller(
            symbols=["TCS.NS"],
            session_factory=self._make_poller()._session_factory,
            redis_client=redis,
        )
        fast_info = {"open": 1.0, "high": 1.0, "low": 1.0, "last_price": 100.0, "volume": 1}
        with patch("alphavedha.data.live_feed._fetch_fast_info", return_value=fast_info):
            for _ in range(5):
                await poller.poll_once()
        redis.keys.assert_called_once_with("predict:TCS.NS:*")
        redis.delete.assert_called_once()
