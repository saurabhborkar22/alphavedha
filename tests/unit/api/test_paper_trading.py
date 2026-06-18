"""Tests for paper trading and dashboard API schemas."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from alphavedha.api.routes.dashboard import (
    AccuracyByCategory,
    DailyPnLRecord,
    PublicTrackRecord,
)
from alphavedha.api.routes.paper_trading import (
    DashboardSummary,
    PaperTradeRequest,
    PredictionRecord,
    TrackStatsOut,
    TradeOutcomeRequest,
)


class TestPaperTradingSchemas:
    def test_paper_trade_request_valid(self) -> None:
        req = PaperTradeRequest(
            symbol="TCS.NS",
            predicted_direction=1,
            predicted_magnitude=0.02,
            confidence=0.75,
            model_version="v1.0",
        )
        assert req.symbol == "TCS.NS"
        assert req.predicted_direction == 1

    def test_paper_trade_request_direction_bounds(self) -> None:
        with pytest.raises(ValueError):
            PaperTradeRequest(
                symbol="TCS.NS",
                predicted_direction=2,
                predicted_magnitude=0.02,
                confidence=0.75,
                model_version="v1.0",
            )

    def test_paper_trade_request_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            PaperTradeRequest(
                symbol="TCS.NS",
                predicted_direction=1,
                predicted_magnitude=0.02,
                confidence=1.5,
                model_version="v1.0",
            )

    def test_trade_outcome_request(self) -> None:
        req = TradeOutcomeRequest(
            symbol="TCS.NS",
            prediction_date="2024-03-15",
            exit_price=3500.0,
            actual_return=0.015,
            is_correct=True,
        )
        assert req.is_correct is True

    def test_dashboard_summary_empty(self) -> None:
        summary = DashboardSummary(
            total_predictions=0,
            correct_predictions=0,
            accuracy_7d=None,
            accuracy_30d=None,
            accuracy_all=None,
            total_return=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            days_tracked=0,
        )
        assert summary.total_predictions == 0
        # New cost/track fields default to None for backward compatibility.
        assert summary.round_trip_cost_pct is None
        assert summary.tracks is None

    def test_paper_trade_request_accepts_gate_flag(self) -> None:
        req = PaperTradeRequest(
            symbol="TCS.NS",
            predicted_direction=1,
            predicted_magnitude=0.02,
            confidence=0.75,
            model_version="v1.0",
            is_tradeable=False,
        )
        assert req.is_tradeable is False

    def test_track_stats_out(self) -> None:
        stats = TrackStatsOut(
            name="gate_passed",
            n_selected=10,
            n_evaluated=8,
            n_wins_net=5,
            win_rate_net=0.625,
            avg_return_gross=0.012,
            avg_return_net=0.0073,
            total_return_net=0.0584,
            profit_factor_net=1.8,
            sharpe_net=1.1,
            max_drawdown_net=-0.02,
        )
        assert stats.win_rate_net == 0.625

    def test_dashboard_summary_with_tracks(self) -> None:
        track = TrackStatsOut(
            name="all",
            n_selected=50,
            n_evaluated=0,
            n_wins_net=0,
            win_rate_net=None,
            avg_return_gross=None,
            avg_return_net=None,
            total_return_net=0.0,
            profit_factor_net=None,
            sharpe_net=None,
            max_drawdown_net=0.0,
        )
        summary = DashboardSummary(
            total_predictions=50,
            correct_predictions=0,
            accuracy_7d=None,
            accuracy_30d=None,
            accuracy_all=None,
            total_return=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            days_tracked=1,
            round_trip_cost_pct=0.0047,
            tracks={"all": track},
        )
        assert summary.tracks is not None
        assert summary.tracks["all"].n_selected == 50

    def test_prediction_record(self) -> None:
        rec = PredictionRecord(
            symbol="TCS.NS",
            prediction_date="2024-03-15",
            predicted_direction=1,
            predicted_magnitude=0.02,
            confidence=0.75,
            model_version="v1.0",
            regime="bull",
            entry_price=3450.0,
            exit_price=3500.0,
            actual_return=0.015,
            is_correct=True,
        )
        assert rec.regime == "bull"

    def test_paper_trade_request_with_stop_levels(self) -> None:
        req = PaperTradeRequest(
            symbol="TCS.NS",
            predicted_direction=1,
            predicted_magnitude=0.02,
            confidence=0.75,
            model_version="v1.0",
            entry_price=3450.0,
            stop_loss_price=3380.0,
            take_profit_price=3550.0,
        )
        assert req.stop_loss_price == 3380.0
        assert req.take_profit_price == 3550.0

    def test_prediction_record_with_exit_reason(self) -> None:
        rec = PredictionRecord(
            symbol="TCS.NS",
            prediction_date="2024-03-15",
            predicted_direction=1,
            predicted_magnitude=0.02,
            confidence=0.75,
            model_version="v1.0",
            regime="bull",
            entry_price=3450.0,
            stop_loss_price=3380.0,
            take_profit_price=3550.0,
            exit_price=3380.0,
            exit_reason="stop_loss",
            actual_return=-0.0203,
            is_correct=False,
        )
        assert rec.exit_reason == "stop_loss"
        assert rec.stop_loss_price == 3380.0


class TestDashboardSchemas:
    def test_daily_pnl_record(self) -> None:
        rec = DailyPnLRecord(
            date="2024-03-15",
            portfolio_value=1015000.0,
            daily_return=0.015,
            cumulative_return=0.015,
            n_positions=5,
            n_correct=3,
            n_total_predictions=5,
            benchmark_return=0.005,
        )
        assert rec.n_correct == 3

    def test_accuracy_by_category(self) -> None:
        acc = AccuracyByCategory(
            category="bull",
            total=100,
            correct=60,
            accuracy=0.6,
        )
        assert acc.accuracy == 0.6

    def test_public_track_record_empty(self) -> None:
        tr = PublicTrackRecord(
            start_date=None,
            end_date=None,
            total_days=0,
            total_predictions=0,
            overall_accuracy=None,
            cumulative_return=0.0,
            benchmark_cumulative_return=0.0,
            alpha=0.0,
            accuracy_by_regime=[],
            accuracy_by_confidence=[],
            monthly_returns=[],
        )
        assert tr.total_predictions == 0
        assert tr.alpha == 0.0


def _trade_row(
    symbol: str,
    pred_date: date,
    direction: int,
    confidence: float,
    actual_return: float | None,
    is_tradeable: bool | None = None,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "prediction_date": pred_date,
        "predicted_direction": direction,
        "predicted_magnitude": 0.02,
        "confidence": confidence,
        "model_version": "v1",
        "regime": "bull",
        "is_tradeable": is_tradeable,
        "entry_price": 100.0,
        "exit_price": 102.0 if actual_return is not None else None,
        "actual_return": actual_return,
        "is_correct": actual_return is not None,
    }


class TestDashboardEndpoint:
    async def test_empty_table_reports_cost_and_no_tracks(self) -> None:
        from alphavedha.api.routes.paper_trading import get_dashboard

        with patch(
            "alphavedha.data.store.load_paper_trades",
            new_callable=AsyncMock,
            return_value=pd.DataFrame(),
        ):
            summary = await get_dashboard()

        assert summary.total_predictions == 0
        assert summary.round_trip_cost_pct is not None
        assert summary.round_trip_cost_pct > 0
        assert summary.tracks is None

    async def test_correct_short_increases_total_return(self) -> None:
        """Regression: returns must be directional — a correct short on a
        falling stock is a gain, not a loss."""
        from alphavedha.api.routes.paper_trading import get_dashboard

        trades = pd.DataFrame(
            [
                _trade_row("LONGWIN", date(2026, 5, 4), 1, 0.7, 0.02, is_tradeable=True),
                _trade_row("SHORTWIN", date(2026, 5, 4), -1, 0.6, -0.03, is_tradeable=False),
            ]
        )
        with patch(
            "alphavedha.data.store.load_paper_trades",
            new_callable=AsyncMock,
            return_value=trades,
        ):
            summary = await get_dashboard()

        # Gross directional: +0.02 (long) + 0.03 (correct short) = +0.05.
        assert summary.total_return == pytest.approx(0.05)
        assert summary.tracks is not None
        all_track = summary.tracks["all"]
        assert all_track.n_evaluated == 2
        assert all_track.avg_return_gross == pytest.approx(0.025)
        assert all_track.avg_return_net is not None
        assert all_track.avg_return_net == pytest.approx(0.025 - summary.round_trip_cost_pct)
        # Gate uses the persisted flag: only LONGWIN passed.
        assert summary.tracks["gate_passed"].n_selected == 1
        assert summary.tracks["top_k"].n_selected == 2

    async def test_flat_and_pending_trades_do_not_pollute_returns(self) -> None:
        from alphavedha.api.routes.paper_trading import get_dashboard

        trades = pd.DataFrame(
            [
                _trade_row("FLAT", date(2026, 5, 4), 0, 0.9, 0.10),
                _trade_row("PENDING", date(2026, 5, 4), 1, 0.8, None),
                _trade_row("WIN", date(2026, 5, 4), 1, 0.7, 0.01),
            ]
        )
        with patch(
            "alphavedha.data.store.load_paper_trades",
            new_callable=AsyncMock,
            return_value=trades,
        ):
            summary = await get_dashboard()

        assert summary.total_predictions == 3
        assert summary.total_return == pytest.approx(0.01)
        assert summary.tracks is not None
        assert summary.tracks["all"].n_selected == 3
        assert summary.tracks["all"].n_evaluated == 1
