"""Tests for background scheduler."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, mock_open, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import schedule

from alphavedha.scheduler import (
    DRIFT_CHECK_DAY,
    EVALUATION_HORIZON_TRADING_DAYS,
    EVALUATION_MIN_CALENDAR_DAYS,
    EVALUATION_TIME,
    INITIAL_PORTFOLIO_VALUE,
    IST,
    PREDICTION_TIME,
    REBALANCE_CHECK_DAY,
    REBALANCE_MONTHS,
    RETRAIN_DAY,
    AlphaVedhaScheduler,
    EvaluationSummary,
    JobResult,
    SchedulerState,
    _evaluate_open_paper_trades,
    _persist_paper_trades,
    _store_pnl_summary,
)


def _make_prediction(symbol: str, direction: int = 1) -> SimpleNamespace:
    """Minimal stand-in for StockPrediction with the fields the scheduler
    persists and the ranker filters on."""
    return SimpleNamespace(
        symbol=symbol,
        direction=direction,
        magnitude=0.025,
        meta_confidence=0.71,
        model_version="v0.1.0",
        regime="bull",
        is_tradeable=True,
        position_size_pct=0.05,
        composite_score=70.0,
    )


def _make_ohlcv(start: date, n_days: int, start_price: float = 100.0) -> pd.DataFrame:
    """Business-day OHLCV frame with monotonically increasing closes."""
    idx = pd.bdate_range(start=start, periods=n_days)
    closes = start_price + np.arange(n_days, dtype=float)
    return pd.DataFrame({"close": closes}, index=idx)


# The prediction job skips weekends, so tests that exercise it pin the clock
# to a trading day (Thursday) to stay deterministic regardless of run date.
_WEEKDAY = datetime(2026, 6, 11, 8, 30, tzinfo=IST)
_SATURDAY = datetime(2026, 6, 13, 8, 30, tzinfo=IST)


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
        assert s.last_rebalance_check is None


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
        with patch("alphavedha.scheduler._now_ist", return_value=_WEEKDAY):
            result = sched.run_daily_predictions()
        assert result.job_name == "daily_predictions"
        assert result.success is True
        assert result.symbols_processed > 0
        assert result.finished_at is not None
        assert sched.state.last_prediction_run is not None

    def test_run_daily_predictions_skipped_on_weekend(self) -> None:
        sched = AlphaVedhaScheduler(demo=False)
        with (
            patch("alphavedha.scheduler._now_ist", return_value=_SATURDAY),
            patch(
                "alphavedha.scheduler._persist_paper_trades",
                new_callable=AsyncMock,
            ) as mock_persist,
        ):
            result = sched.run_daily_predictions()
        assert result.success is True
        assert "weekend" in (result.error or "")
        assert result.symbols_processed == 0
        mock_persist.assert_not_awaited()

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
        # The in-process LSTM/TFT retrain is only registered when the
        # heavy-training gate is set; otherwise train.yml owns it on the
        # autoscale deployment (see scheduler.setup_schedule).
        schedule.clear()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALPHAVEDHA_HEAVY_TRAINING", None)
            AlphaVedhaScheduler(demo=True).setup_schedule()
            assert len(schedule.get_jobs()) == 22  # heavy retrain gated off

        schedule.clear()
        with patch.dict(os.environ, {"ALPHAVEDHA_HEAVY_TRAINING": "1"}, clear=False):
            AlphaVedhaScheduler(demo=True).setup_schedule()
            assert len(schedule.get_jobs()) == 23  # incl. weekly LSTM/TFT retrain

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
        with (
            patch("alphavedha.scheduler._now_ist", return_value=_WEEKDAY),
            patch(
                "alphavedha.services.model_registry.ModelRegistry.get_prediction_engine",
                side_effect=Exception("model not found"),
            ),
        ):
            result = sched.run_daily_predictions()
            assert result.success is False
            assert result.error is not None
            assert "model not found" in result.error

    def test_run_rebalance_check_no_changes(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        mock_df = pd.DataFrame({"symbol": ["TCS", "INFY"]})
        with (
            patch("alphavedha.scheduler.Path.exists", return_value=True),
            patch(
                "alphavedha.scheduler.Path.open",
                mock_open(read_data="sectors:\n  it:\n    - TCS\n    - INFY\n"),
            ),
            patch("alphavedha.scheduler._run_async", return_value=mock_df),
        ):
            result = sched.run_rebalance_check()
            assert result.success is True
            assert result.job_name == "quarterly_rebalance_check"
            assert result.symbols_processed == 2
            assert sched.state.last_rebalance_check is not None

    def test_run_rebalance_check_detects_changes(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        mock_df = pd.DataFrame({"symbol": ["TCS", "INFY", "NEWSTOCK"]})
        with (
            patch("alphavedha.scheduler.Path.exists", return_value=True),
            patch(
                "alphavedha.scheduler.Path.open",
                mock_open(read_data="sectors:\n  it:\n    - TCS\n    - INFY\n    - OLDSTOCK\n"),
            ),
            patch("alphavedha.scheduler._run_async", return_value=mock_df),
        ):
            result = sched.run_rebalance_check()
            assert result.success is True
            assert result.symbols_processed == 3

    def test_run_rebalance_check_missing_config(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        with patch("alphavedha.scheduler.Path.exists", return_value=False):
            result = sched.run_rebalance_check()
            assert result.success is False
            assert "not found" in result.error

    def test_maybe_quarterly_rebalance_march(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        with (
            patch("alphavedha.scheduler._now_ist") as mock_now,
            patch.object(sched, "run_rebalance_check") as mock_run,
        ):
            mock_now.return_value = datetime(2026, 3, 2, 7, 0, tzinfo=IST)
            mock_run.return_value = JobResult(
                job_name="quarterly_rebalance_check",
                started_at=mock_now.return_value,
                success=True,
            )
            result = sched._maybe_quarterly_rebalance()
            assert result is not None
            mock_run.assert_called_once()

    def test_maybe_quarterly_rebalance_september(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        with (
            patch("alphavedha.scheduler._now_ist") as mock_now,
            patch.object(sched, "run_rebalance_check") as mock_run,
        ):
            mock_now.return_value = datetime(2026, 9, 7, 7, 0, tzinfo=IST)
            mock_run.return_value = JobResult(
                job_name="quarterly_rebalance_check",
                started_at=mock_now.return_value,
                success=True,
            )
            result = sched._maybe_quarterly_rebalance()
            assert result is not None

    def test_maybe_quarterly_rebalance_wrong_month_skips(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        with patch("alphavedha.scheduler._now_ist") as mock_now:
            mock_now.return_value = datetime(2026, 5, 5, 7, 0, tzinfo=IST)
            result = sched._maybe_quarterly_rebalance()
            assert result is None

    def test_rebalance_config_values(self) -> None:
        assert REBALANCE_CHECK_DAY == "monday"
        assert {3, 9} == REBALANCE_MONTHS

    def test_run_daily_predictions_real_mode_persists(self) -> None:
        """Non-demo predictions must ALL be persisted as paper trades —
        including ones the meta-labeling gate marks untradeable, so the
        track record measures model accuracy on no-signal days too."""
        sched = AlphaVedhaScheduler(demo=False)
        predictions = [_make_prediction("TCS", 1), _make_prediction("INFY", -1)]
        mock_service = SimpleNamespace(predict_tier=AsyncMock(return_value=predictions))

        with (
            patch("alphavedha.scheduler._now_ist", return_value=_WEEKDAY),
            patch(
                "alphavedha.services.prediction_service.PredictionService",
                return_value=mock_service,
            ),
            patch(
                "alphavedha.services.model_registry.ModelRegistry",
                return_value=SimpleNamespace(),
            ),
            patch(
                "alphavedha.scheduler._persist_paper_trades",
                new_callable=AsyncMock,
                return_value=2,
            ) as mock_persist,
        ):
            result = sched.run_daily_predictions()

        assert result.success is True
        assert result.symbols_processed == 2
        mock_persist.assert_awaited_once()
        persisted_predictions = mock_persist.await_args.args[0]
        assert [p.symbol for p in persisted_predictions] == ["TCS", "INFY"]

    def test_run_daily_predictions_demo_skips_persistence(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        with (
            patch("alphavedha.scheduler._now_ist", return_value=_WEEKDAY),
            patch(
                "alphavedha.scheduler._persist_paper_trades",
                new_callable=AsyncMock,
            ) as mock_persist,
        ):
            result = sched.run_daily_predictions()
        assert result.success is True
        mock_persist.assert_not_awaited()

    def test_run_daily_evaluation_real_mode(self) -> None:
        sched = AlphaVedhaScheduler(demo=False)
        summary = EvaluationSummary(
            n_evaluated=3, n_correct=2, directional_returns=[0.01, 0.02, -0.005]
        )

        with (
            patch(
                "alphavedha.scheduler._evaluate_open_paper_trades",
                new_callable=AsyncMock,
                return_value=summary,
            ) as mock_eval,
            patch(
                "alphavedha.scheduler._store_pnl_summary",
                new_callable=AsyncMock,
            ) as mock_pnl,
        ):
            result = sched.run_daily_evaluation()

        assert result.success is True
        assert result.symbols_processed == 3
        mock_eval.assert_awaited_once()
        mock_pnl.assert_awaited_once()

    def test_run_daily_evaluation_no_trades_skips_pnl(self) -> None:
        sched = AlphaVedhaScheduler(demo=False)
        with (
            patch(
                "alphavedha.scheduler._evaluate_open_paper_trades",
                new_callable=AsyncMock,
                return_value=EvaluationSummary(),
            ),
            patch(
                "alphavedha.scheduler._store_pnl_summary",
                new_callable=AsyncMock,
            ) as mock_pnl,
        ):
            result = sched.run_daily_evaluation()
        assert result.success is True
        assert result.symbols_processed == 0
        mock_pnl.assert_not_awaited()

    def test_run_daily_evaluation_error_recorded(self) -> None:
        sched = AlphaVedhaScheduler(demo=False)
        with patch(
            "alphavedha.scheduler._evaluate_open_paper_trades",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ):
            result = sched.run_daily_evaluation()
        assert result.success is False
        assert "db down" in (result.error or "")


class TestPersistPaperTrades:
    async def test_persists_all_predictions(self) -> None:
        predictions = [_make_prediction("TCS", 1), _make_prediction("INFY", -1)]
        prediction_date = date(2026, 6, 11)
        ohlcv = _make_ohlcv(date(2026, 6, 1), 8, start_price=100.0)

        with (
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=ohlcv,
            ),
            patch(
                "alphavedha.data.store.store_paper_trade",
                new_callable=AsyncMock,
                return_value=1,
            ) as mock_store,
        ):
            persisted = await _persist_paper_trades(predictions, prediction_date)

        assert persisted == 2
        assert mock_store.await_count == 2
        first_row = mock_store.await_args_list[0].args[0]
        assert first_row["symbol"] == "TCS"
        assert first_row["prediction_date"] == prediction_date
        assert first_row["predicted_direction"] == 1
        assert first_row["predicted_magnitude"] == 0.025
        assert first_row["confidence"] == 0.71
        assert first_row["model_version"] == "v0.1.0"
        assert first_row["regime"] == "bull"
        assert first_row["is_tradeable"] is True
        assert first_row["entry_price"] == float(ohlcv["close"].iloc[-1])

    async def test_continues_on_per_symbol_store_failure(self) -> None:
        predictions = [_make_prediction("BAD"), _make_prediction("GOOD")]
        ohlcv = _make_ohlcv(date(2026, 6, 1), 5)

        with (
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=ohlcv,
            ),
            patch(
                "alphavedha.data.store.store_paper_trade",
                new_callable=AsyncMock,
                side_effect=[RuntimeError("constraint violation"), 1],
            ) as mock_store,
        ):
            persisted = await _persist_paper_trades(predictions, date(2026, 6, 11))

        assert persisted == 1
        assert mock_store.await_count == 2

    async def test_stores_with_null_entry_price_when_no_ohlcv(self) -> None:
        with (
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=pd.DataFrame(),
            ),
            patch(
                "alphavedha.data.store.store_paper_trade",
                new_callable=AsyncMock,
                return_value=1,
            ) as mock_store,
        ):
            persisted = await _persist_paper_trades([_make_prediction("TCS")], date(2026, 6, 11))

        assert persisted == 1
        assert mock_store.await_args.args[0]["entry_price"] is None


class TestEvaluateOpenPaperTrades:
    def _open_trades_df(self, prediction_date: date) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": "TCS",
                    "prediction_date": prediction_date,
                    "predicted_direction": 1,
                    "predicted_magnitude": 0.02,
                    "confidence": 0.7,
                    "model_version": "v0.1.0",
                    "regime": "bull",
                    "entry_price": 100.0,
                    "exit_price": None,
                    "actual_return": None,
                    "is_correct": None,
                },
                {
                    "symbol": "DONE",
                    "prediction_date": prediction_date,
                    "predicted_direction": -1,
                    "predicted_magnitude": 0.01,
                    "confidence": 0.6,
                    "model_version": "v0.1.0",
                    "regime": "bear",
                    "entry_price": 50.0,
                    "exit_price": 49.0,
                    "actual_return": -0.02,
                    "is_correct": True,
                },
            ]
        )

    async def test_evaluates_open_trades_only(self) -> None:
        as_of = date(2026, 6, 11)
        prediction_date = as_of - timedelta(days=EVALUATION_MIN_CALENDAR_DAYS)
        trades = self._open_trades_df(prediction_date)
        ohlcv = _make_ohlcv(prediction_date, 20, start_price=100.0)

        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=ohlcv,
            ),
            patch(
                "alphavedha.data.store.update_paper_trade_outcome",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            summary = await _evaluate_open_paper_trades(as_of)

        # Only the open TCS trade is evaluated; the DONE trade already has an outcome.
        assert summary.n_evaluated == 1
        mock_update.assert_awaited_once()
        kwargs = mock_update.await_args.kwargs
        assert kwargs["symbol"] == "TCS"
        assert kwargs["prediction_date"] == prediction_date

        # Exit at the 15th trading day after entry (bars strictly after prediction_date).
        future = ohlcv[ohlcv.index.date > prediction_date]
        expected_exit = float(future["close"].iloc[EVALUATION_HORIZON_TRADING_DAYS - 1])
        assert kwargs["exit_price"] == expected_exit
        expected_return = (expected_exit - 100.0) / 100.0
        assert kwargs["actual_return"] == expected_return
        assert kwargs["is_correct"] is True  # direction=1, rising prices
        assert summary.n_correct == 1
        assert summary.directional_returns == [expected_return]

    async def test_skips_trade_with_missing_entry_price(self) -> None:
        as_of = date(2026, 6, 11)
        prediction_date = date(2026, 5, 15)
        trades = self._open_trades_df(prediction_date)
        trades.loc[trades["symbol"] == "TCS", "entry_price"] = None

        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
            ) as mock_load,
            patch(
                "alphavedha.data.store.update_paper_trade_outcome",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            summary = await _evaluate_open_paper_trades(as_of)

        assert summary.n_evaluated == 0
        mock_load.assert_not_awaited()
        mock_update.assert_not_awaited()

    async def test_continues_after_per_symbol_failure(self) -> None:
        as_of = date(2026, 6, 11)
        prediction_date = date(2026, 5, 15)
        trades = self._open_trades_df(prediction_date)
        # Make both trades open so both are attempted.
        trades["exit_price"] = None
        ohlcv = _make_ohlcv(prediction_date, 20)

        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                side_effect=[RuntimeError("provider down"), ohlcv],
            ),
            patch(
                "alphavedha.data.store.update_paper_trade_outcome",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            summary = await _evaluate_open_paper_trades(as_of)

        assert summary.n_evaluated == 1
        mock_update.assert_awaited_once()
        assert mock_update.await_args.kwargs["symbol"] == "DONE"

    async def test_no_matured_trades(self) -> None:
        with patch(
            "alphavedha.data.store.load_paper_trades",
            new_callable=AsyncMock,
            return_value=pd.DataFrame(),
        ):
            summary = await _evaluate_open_paper_trades(date(2026, 6, 11))
        assert summary.n_evaluated == 0


class TestStorePnlSummary:
    async def test_stores_first_pnl_row(self) -> None:
        as_of = date(2026, 6, 11)
        summary = EvaluationSummary(n_evaluated=2, n_correct=1, directional_returns=[0.02, -0.01])

        with (
            patch(
                "alphavedha.data.store.load_daily_pnl",
                new_callable=AsyncMock,
                return_value=pd.DataFrame(),
            ),
            patch(
                "alphavedha.data.store.store_daily_pnl",
                new_callable=AsyncMock,
                return_value=1,
            ) as mock_store,
        ):
            await _store_pnl_summary(as_of, summary)

        row = mock_store.await_args.args[0]
        expected_return = (0.02 - 0.01) / 2
        assert row["date"] == as_of
        assert row["daily_return"] == expected_return
        assert row["portfolio_value"] == INITIAL_PORTFOLIO_VALUE * (1 + expected_return)
        assert row["cumulative_return"] == row["portfolio_value"] / INITIAL_PORTFOLIO_VALUE - 1.0
        assert row["n_positions"] == 2
        assert row["n_correct"] == 1
        assert row["n_total_predictions"] == 2

    async def test_compounds_from_prior_portfolio_value(self) -> None:
        as_of = date(2026, 6, 11)
        prior = pd.DataFrame([{"date": date(2026, 6, 10), "portfolio_value": 1_050_000.0}])

        async def _load_daily_pnl(
            start: date | None = None, end: date | None = None
        ) -> pd.DataFrame:
            if start == as_of:
                return pd.DataFrame()  # no row for today yet
            return prior

        with (
            patch("alphavedha.data.store.load_daily_pnl", side_effect=_load_daily_pnl),
            patch(
                "alphavedha.data.store.store_daily_pnl",
                new_callable=AsyncMock,
                return_value=1,
            ) as mock_store,
        ):
            await _store_pnl_summary(
                as_of, EvaluationSummary(n_evaluated=1, n_correct=1, directional_returns=[0.01])
            )

        row = mock_store.await_args.args[0]
        assert row["portfolio_value"] == 1_050_000.0 * 1.01

    async def test_skips_when_row_already_exists(self) -> None:
        as_of = date(2026, 6, 11)
        existing = pd.DataFrame([{"date": as_of, "portfolio_value": 1_000_000.0}])

        with (
            patch(
                "alphavedha.data.store.load_daily_pnl",
                new_callable=AsyncMock,
                return_value=existing,
            ),
            patch(
                "alphavedha.data.store.store_daily_pnl",
                new_callable=AsyncMock,
            ) as mock_store,
        ):
            await _store_pnl_summary(as_of, EvaluationSummary(n_evaluated=1))

        mock_store.assert_not_awaited()


class TestDataRefreshJob:
    def test_run_data_refresh_success(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        ingestion = SimpleNamespace(symbols_succeeded=50, symbols_failed=0, total_rows_stored=250)
        with patch("alphavedha.scheduler._run_async", return_value=ingestion):
            result = sched.run_data_refresh()
        assert result.job_name == "daily_data_refresh"
        assert result.success is True
        assert result.symbols_processed == 50
        assert sched.state.last_data_refresh is not None

    def test_run_data_refresh_failure_recorded(self) -> None:
        sched = AlphaVedhaScheduler(demo=True)
        with patch("alphavedha.scheduler._run_async", side_effect=Exception("yfinance down")):
            result = sched.run_data_refresh()
        assert result.success is False
        assert result.error is not None
        assert "yfinance down" in result.error
