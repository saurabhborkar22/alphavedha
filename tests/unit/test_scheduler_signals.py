"""Tests for signal strategies scheduler wiring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from alphavedha.scheduler import (
    SIGNAL_STRATEGIES_TIME,
    AlphaVedhaScheduler,
    _persist_signal_paper_trades,
)

IST = ZoneInfo("Asia/Kolkata")
_WEEKDAY = datetime(2026, 6, 11, 8, 35, tzinfo=IST)
_SATURDAY = datetime(2026, 6, 13, 8, 35, tzinfo=IST)


@dataclass
class _FakeBlowup:
    symbol: str
    total_score: int
    on_avoid_list: bool


@dataclass
class _FakeEventSignal:
    symbol: str
    direction: int
    confidence: float


class TestSignalStrategiesConfig:
    def test_time_between_prediction_and_hash(self) -> None:
        assert SIGNAL_STRATEGIES_TIME == "08:35"


class TestPersistSignalPaperTrades:
    async def test_persists_signals(self) -> None:
        signals = [
            {"symbol": "TCS.NS", "direction": 1, "confidence": 0.75},
            {"symbol": "INFY.NS", "direction": -1, "confidence": 0.60},
        ]
        with (
            patch(
                "alphavedha.scheduler._latest_close",
                new_callable=AsyncMock,
                return_value=3500.0,
            ),
            patch(
                "alphavedha.data.store.store_paper_trade",
                new_callable=AsyncMock,
                return_value=1,
            ) as mock_store,
        ):
            from datetime import date

            n = await _persist_signal_paper_trades(signals, "event_drift_v1", date(2026, 6, 11))

        assert n == 2
        assert mock_store.await_count == 2
        call_args = mock_store.await_args_list[0].args[0]
        assert call_args["strategy"] == "event_drift_v1"
        assert call_args["predicted_direction"] == 1
        assert call_args["entry_price"] == 3500.0

    async def test_continues_on_single_failure(self) -> None:
        signals = [
            {"symbol": "TCS.NS", "direction": 1, "confidence": 0.75},
            {"symbol": "INFY.NS", "direction": 1, "confidence": 0.60},
        ]
        with (
            patch(
                "alphavedha.scheduler._latest_close",
                new_callable=AsyncMock,
                return_value=3500.0,
            ),
            patch(
                "alphavedha.data.store.store_paper_trade",
                new_callable=AsyncMock,
                side_effect=[Exception("db error"), 1],
            ),
        ):
            from datetime import date

            n = await _persist_signal_paper_trades(signals, "event_drift_v1", date(2026, 6, 11))

        assert n == 1


class TestRunSignalStrategies:
    def test_skipped_on_weekend(self) -> None:
        sched = AlphaVedhaScheduler(demo=False)
        with patch("alphavedha.scheduler._now_ist", return_value=_SATURDAY):
            result = sched.run_signal_strategies()
        assert result.success is True
        assert "weekend" in (result.error or "")

    def test_skipped_in_demo(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        with patch("alphavedha.scheduler._now_ist", return_value=_WEEKDAY):
            result = sched.run_signal_strategies()
        assert result.success is True
        assert result.job_name == "signal_strategies"

    def test_runs_all_strategies(self) -> None:
        sched = AlphaVedhaScheduler(demo=False)

        fake_blowup = _FakeBlowup(symbol="YESBANK.NS", total_score=80, on_avoid_list=True)
        fake_event = _FakeEventSignal(symbol="TCS.NS", direction=1, confidence=0.7)
        fake_insider = _FakeEventSignal(symbol="INFY.NS", direction=1, confidence=0.65)
        fake_guidance = _FakeEventSignal(symbol="HDFCBANK.NS", direction=1, confidence=0.6)

        with (
            patch("alphavedha.scheduler._now_ist", return_value=_WEEKDAY),
            patch(
                "alphavedha.scheduler._run_async",
                side_effect=[
                    [fake_blowup],
                    1,
                    [fake_event],
                    1,
                    [fake_insider],
                    1,
                    [fake_guidance],
                    1,
                ],
            ),
            patch(
                "alphavedha.intel.signals.blowup_score.compute_avoid_list",
                return_value=[fake_blowup],
            ),
        ):
            result = sched.run_signal_strategies()

        assert result.success is True
        assert result.symbols_processed == 4
        assert sched.state.last_signal_strategies is not None

    def test_vetoes_avoid_listed_event_drift(self) -> None:
        sched = AlphaVedhaScheduler(demo=False)

        fake_blowup = _FakeBlowup(symbol="TCS.NS", total_score=75, on_avoid_list=True)
        fake_event = _FakeEventSignal(symbol="TCS.NS", direction=1, confidence=0.7)

        run_async_results = [
            [fake_blowup],
            1,
            [fake_event],
            0,
            [],
            [],
        ]
        call_idx = 0

        def _mock_run_async(coro: object) -> object:  # type: ignore[type-arg]
            nonlocal call_idx
            result = run_async_results[call_idx]
            call_idx += 1
            return result

        with (
            patch("alphavedha.scheduler._now_ist", return_value=_WEEKDAY),
            patch("alphavedha.scheduler._run_async", side_effect=_mock_run_async),
            patch(
                "alphavedha.intel.signals.blowup_score.compute_avoid_list",
                return_value=[fake_blowup],
            ),
        ):
            result = sched.run_signal_strategies()

        assert result.success is True

    def test_handles_exception_gracefully(self) -> None:
        sched = AlphaVedhaScheduler(demo=False)
        with (
            patch("alphavedha.scheduler._now_ist", return_value=_WEEKDAY),
            patch(
                "alphavedha.scheduler._run_async",
                side_effect=Exception("blowup module import error"),
            ),
        ):
            result = sched.run_signal_strategies()
        assert result.success is False
        assert "blowup module import error" in (result.error or "")
