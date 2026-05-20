"""Tests for background scheduler."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import schedule

from alphavedha.scheduler import (
    DRIFT_CHECK_DAY,
    EVALUATION_TIME,
    IST,
    PREDICTION_TIME,
    RETRAIN_DAY,
    AlphaVedhaScheduler,
    JobResult,
    SchedulerState,
)


class TestJobResult:
    def test_defaults(self) -> None:
        r = JobResult(job_name="test", started_at=datetime.now(IST))
        assert r.success is False
        assert r.error is None
        assert r.symbols_processed == 0


class TestSchedulerState:
    def test_defaults(self) -> None:
        s = SchedulerState()
        assert s.is_running is False
        assert s.job_history == []
        assert s.last_prediction_run is None


class TestSchedulerConfig:
    def test_times_are_strings(self) -> None:
        assert isinstance(PREDICTION_TIME, str)
        assert isinstance(EVALUATION_TIME, str)

    def test_ist_timezone(self) -> None:
        assert ZoneInfo("Asia/Kolkata") == IST

    def test_prediction_before_market_open(self) -> None:
        hour, minute = map(int, PREDICTION_TIME.split(":"))
        assert hour == 8
        assert minute == 30

    def test_evaluation_after_market_close(self) -> None:
        hour, minute = map(int, EVALUATION_TIME.split(":"))
        assert hour == 15
        assert minute == 45

    def test_drift_on_weekend(self) -> None:
        assert DRIFT_CHECK_DAY == "saturday"

    def test_retrain_on_weekend(self) -> None:
        assert RETRAIN_DAY == "saturday"


class TestAlphaVedhaScheduler:
    def setup_method(self) -> None:
        schedule.clear()

    def test_init_defaults(self) -> None:
        sched = AlphaVedhaScheduler()
        assert sched._tier == "large"
        assert sched._demo is False
        assert sched.state.is_running is False

    def test_init_custom(self) -> None:
        sched = AlphaVedhaScheduler(tier="mid", demo=True)
        assert sched._tier == "mid"
        assert sched._demo is True

    def test_run_daily_predictions_demo(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        result = sched.run_daily_predictions()
        assert result.job_name == "daily_predictions"
        assert result.success is True
        assert result.symbols_processed > 0
        assert result.finished_at is not None
        assert sched.state.last_prediction_run is not None

    def test_run_daily_evaluation(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        result = sched.run_daily_evaluation()
        assert result.job_name == "daily_evaluation"
        assert result.success is True
        assert sched.state.last_evaluation_run is not None

    def test_run_drift_check(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        result = sched.run_drift_check()
        assert result.job_name == "weekly_drift_check"
        assert result.success is True
        assert sched.state.last_drift_check is not None

    def test_run_monthly_retrain(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        result = sched.run_monthly_retrain()
        assert result.job_name == "monthly_retrain"
        assert result.success is True
        assert sched.state.last_retrain is not None

    def test_job_history_recorded(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        sched.run_daily_evaluation()
        sched.run_drift_check()
        assert len(sched.state.job_history) == 2
        assert sched.state.job_history[0].job_name == "daily_evaluation"
        assert sched.state.job_history[1].job_name == "weekly_drift_check"

    def test_job_history_capped(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        for _ in range(110):
            sched.run_daily_evaluation()
        assert len(sched.state.job_history) <= 100

    def test_setup_schedule_registers_jobs(self) -> None:
        schedule.clear()
        sched = AlphaVedhaScheduler(demo=True)
        sched.setup_schedule()
        assert len(schedule.get_jobs()) == 4

    def test_maybe_monthly_retrain_first_week(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        with patch("alphavedha.scheduler._now_ist") as mock_now:
            mock_now.return_value = datetime(2026, 5, 2, 22, 0, tzinfo=IST)
            result = sched._maybe_monthly_retrain()
            assert result is not None
            assert result.job_name == "monthly_retrain"

    def test_maybe_monthly_retrain_later_week_skips(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        with patch("alphavedha.scheduler._now_ist") as mock_now:
            mock_now.return_value = datetime(2026, 5, 15, 22, 0, tzinfo=IST)
            result = sched._maybe_monthly_retrain()
            assert result is None

    def test_stop(self) -> None:
        sched = AlphaVedhaScheduler()
        sched._state.is_running = True
        sched.stop()
        assert sched.state.is_running is False

    def test_failed_prediction_records_error(self) -> None:
        sched = AlphaVedhaScheduler(demo=False)
        with patch(
            "alphavedha.services.model_registry.ModelRegistry.get_prediction_engine",
            side_effect=Exception("model not found"),
        ):
            result = sched.run_daily_predictions()
            assert result.success is False
            assert result.error is not None
            assert "model not found" in result.error
