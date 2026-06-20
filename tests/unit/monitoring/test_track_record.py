"""Unit tests for the cost-adjusted paper trade track record."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from alphavedha.monitoring.track_record import (
    DEFAULT_GATE_CONFIDENCE,
    HORIZON_TRADING_DAYS,
    TrackRecord,
    TrackStats,
    compute_track_record,
    compute_track_records_by_strategy,
    compute_track_stats,
)


def _trade(
    symbol: str,
    pred_date: date,
    direction: int,
    confidence: float,
    actual_return: float | None = None,
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
        "exit_price": None,
        "actual_return": actual_return,
        "is_correct": None,
    }


D1 = date(2026, 5, 4)
D2 = date(2026, 5, 5)
D3 = date(2026, 5, 6)


class TestComputeTrackStats:
    def test_empty_frame_returns_zeroed_stats(self) -> None:
        stats = compute_track_stats("all", pd.DataFrame(), cost_pct=0.005)
        assert stats.n_selected == 0
        assert stats.n_evaluated == 0
        assert stats.avg_return_net is None
        assert stats.total_return_net == 0.0

    def test_directional_gross_and_net_math(self) -> None:
        # Long win +4%, long loss -2%, correct short on a -3% move → gross +3%.
        trades = pd.DataFrame(
            [
                _trade("A", D1, 1, 0.7, actual_return=0.04),
                _trade("B", D1, 1, 0.6, actual_return=-0.02),
                _trade("C", D1, -1, 0.5, actual_return=-0.03),
            ]
        )
        stats = compute_track_stats("all", trades, cost_pct=0.005)

        assert stats.n_selected == 3
        assert stats.n_evaluated == 3
        assert stats.avg_return_gross == pytest.approx((0.04 - 0.02 + 0.03) / 3)
        assert stats.avg_return_net == pytest.approx((0.04 - 0.02 + 0.03) / 3 - 0.005)
        assert stats.total_return_net == pytest.approx(0.05 - 3 * 0.005)
        # Net per-trade: +0.035, -0.025, +0.025 → 2 wins out of 3.
        assert stats.n_wins_net == 2
        assert stats.win_rate_net == pytest.approx(2 / 3)
        assert stats.profit_factor_net == pytest.approx((0.035 + 0.025) / 0.025)

    def test_correct_short_counts_as_gain(self) -> None:
        trades = pd.DataFrame([_trade("A", D1, -1, 0.7, actual_return=-0.05)])
        stats = compute_track_stats("all", trades, cost_pct=0.0)
        assert stats.avg_return_gross == pytest.approx(0.05)
        assert stats.n_wins_net == 1

    def test_flat_predictions_are_never_traded(self) -> None:
        trades = pd.DataFrame(
            [
                _trade("A", D1, 0, 0.9, actual_return=0.10),
                _trade("B", D1, 1, 0.6, actual_return=0.01),
            ]
        )
        stats = compute_track_stats("all", trades, cost_pct=0.0)
        assert stats.n_selected == 2
        assert stats.n_evaluated == 1
        assert stats.avg_return_gross == pytest.approx(0.01)

    def test_unevaluated_trades_excluded_from_returns(self) -> None:
        trades = pd.DataFrame(
            [
                _trade("A", D1, 1, 0.6, actual_return=None),
                _trade("B", D1, 1, 0.6, actual_return=0.02),
            ]
        )
        stats = compute_track_stats("all", trades, cost_pct=0.0)
        assert stats.n_selected == 2
        assert stats.n_evaluated == 1
        assert stats.total_return_net == pytest.approx(0.02)

    def test_costs_reduce_net_below_gross(self) -> None:
        trades = pd.DataFrame([_trade("A", D1, 1, 0.6, actual_return=0.03)])
        stats = compute_track_stats("all", trades, cost_pct=0.0047)
        assert stats.avg_return_gross is not None
        assert stats.avg_return_net is not None
        assert stats.avg_return_net == pytest.approx(stats.avg_return_gross - 0.0047)

    def test_sharpe_and_drawdown_over_cohorts(self) -> None:
        # Three one-trade cohorts: +2%, -1%, +1%.
        trades = pd.DataFrame(
            [
                _trade("A", D1, 1, 0.6, actual_return=0.02),
                _trade("A", D2, 1, 0.6, actual_return=-0.01),
                _trade("A", D3, 1, 0.6, actual_return=0.01),
            ]
        )
        stats = compute_track_stats("all", trades, cost_pct=0.0)

        cohorts = np.array([0.02, -0.01, 0.01])
        expected_sharpe = cohorts.mean() / cohorts.std(ddof=1) * np.sqrt(252 / HORIZON_TRADING_DAYS)
        assert stats.sharpe_net == pytest.approx(expected_sharpe)
        # Equity: 1.02 → 1.0098 → drawdown bottom is the -1% cohort.
        assert stats.max_drawdown_net == pytest.approx(1.0098 / 1.02 - 1.0)

    def test_single_cohort_has_no_sharpe_and_zero_drawdown(self) -> None:
        trades = pd.DataFrame([_trade("A", D1, 1, 0.6, actual_return=0.02)])
        stats = compute_track_stats("all", trades, cost_pct=0.0)
        assert stats.sharpe_net is None
        assert stats.max_drawdown_net == pytest.approx(0.0)

    def test_all_wins_has_no_profit_factor(self) -> None:
        trades = pd.DataFrame(
            [
                _trade("A", D1, 1, 0.6, actual_return=0.02),
                _trade("B", D1, 1, 0.6, actual_return=0.03),
            ]
        )
        stats = compute_track_stats("all", trades, cost_pct=0.0)
        assert stats.profit_factor_net is None


class TestTrackSelection:
    def test_gate_uses_persisted_is_tradeable_flag(self) -> None:
        trades = pd.DataFrame(
            [
                _trade("A", D1, 1, 0.9, actual_return=0.02, is_tradeable=True),
                _trade("B", D1, 1, 0.9, actual_return=0.02, is_tradeable=False),
                _trade("C", D1, 1, 0.9, actual_return=0.02, is_tradeable=None),
            ]
        )
        record = compute_track_record(trades, round_trip_cost_pct=0.0)
        assert record.gate_passed.n_selected == 1

    def test_gate_falls_back_to_confidence_when_flag_missing(self) -> None:
        trades = pd.DataFrame(
            [
                _trade("A", D1, 1, DEFAULT_GATE_CONFIDENCE + 0.1, actual_return=0.02),
                _trade("B", D1, 1, DEFAULT_GATE_CONFIDENCE - 0.2, actual_return=0.02),
            ]
        )
        record = compute_track_record(trades, round_trip_cost_pct=0.0)
        assert record.gate_passed.n_selected == 1

    def test_top_k_picks_highest_confidence_per_day(self) -> None:
        trades = pd.DataFrame(
            [
                _trade("A", D1, 1, 0.9, actual_return=0.01),
                _trade("B", D1, 1, 0.8, actual_return=0.01),
                _trade("C", D1, 1, 0.2, actual_return=0.01),
                _trade("A", D2, -1, 0.7, actual_return=-0.01),
                _trade("B", D2, 1, 0.6, actual_return=0.01),
                _trade("C", D2, 1, 0.1, actual_return=0.01),
            ]
        )
        record = compute_track_record(trades, round_trip_cost_pct=0.0, top_k=2)
        assert record.top_k.n_selected == 4
        assert record.top_k.name == "top_2"

    def test_top_k_excludes_flat_predictions(self) -> None:
        trades = pd.DataFrame(
            [
                _trade("A", D1, 0, 0.99, actual_return=0.05),
                _trade("B", D1, 1, 0.4, actual_return=0.01),
            ]
        )
        record = compute_track_record(trades, round_trip_cost_pct=0.0, top_k=2)
        assert record.top_k.n_selected == 1


class TestComputeTrackRecord:
    def test_empty_frame(self) -> None:
        record = compute_track_record(pd.DataFrame(), round_trip_cost_pct=0.0047)
        assert isinstance(record, TrackRecord)
        assert record.round_trip_cost_pct == pytest.approx(0.0047)
        assert record.all_predictions.n_selected == 0
        assert record.gate_passed.n_selected == 0
        assert record.top_k.n_selected == 0

    def test_tracks_share_cost_model(self) -> None:
        trades = pd.DataFrame([_trade("A", D1, 1, 0.9, actual_return=0.03, is_tradeable=True)])
        record = compute_track_record(trades, round_trip_cost_pct=0.005)
        for stats in (record.all_predictions, record.gate_passed, record.top_k):
            assert isinstance(stats, TrackStats)
            assert stats.avg_return_net == pytest.approx(0.03 - 0.005)


class TestComputeTrackRecordsByStrategy:
    def test_empty_frame_returns_empty_dict(self) -> None:
        result = compute_track_records_by_strategy(pd.DataFrame(), round_trip_cost_pct=0.005)
        assert result == {}

    def test_groups_by_strategy(self) -> None:
        rows = [
            {**_trade("A", D1, 1, 0.7, actual_return=0.03), "strategy": "ensemble_v1"},
            {**_trade("B", D1, 1, 0.6, actual_return=0.02), "strategy": "ensemble_v1"},
            {**_trade("C", D1, -1, 0.8, actual_return=-0.04), "strategy": "event_drift_v1"},
        ]
        trades = pd.DataFrame(rows)
        result = compute_track_records_by_strategy(trades, round_trip_cost_pct=0.0)
        assert set(result.keys()) == {"ensemble_v1", "event_drift_v1"}
        assert result["ensemble_v1"].all_predictions.n_selected == 2
        assert result["event_drift_v1"].all_predictions.n_selected == 1

    def test_missing_strategy_column_falls_back(self) -> None:
        trades = pd.DataFrame([_trade("A", D1, 1, 0.7, actual_return=0.03)])
        result = compute_track_records_by_strategy(trades, round_trip_cost_pct=0.0)
        assert "ensemble_v1" in result
        assert result["ensemble_v1"].all_predictions.n_selected == 1
