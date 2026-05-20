"""Tests for paper trading and dashboard API schemas."""

from __future__ import annotations

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
