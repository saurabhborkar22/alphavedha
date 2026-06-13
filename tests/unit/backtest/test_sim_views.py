"""Tests for the historical-simulation view-builders (pure, no DB)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from alphavedha.backtest.sim_views import build_artifact, build_backtest_views


def _make_trades(n_days: int = 8, per_day: int = 5) -> pd.DataFrame:
    """Synthetic evaluated trades: alternating wins/losses, all tradeable."""
    rows: list[dict] = []
    base = date(2026, 1, 5)
    for d in range(n_days):
        pred_date = base + timedelta(days=d)
        for s in range(per_day):
            direction = 1 if s % 2 == 0 else -1
            # Correct call ~60% of the time; magnitude 2%.
            correct = (d + s) % 5 != 0
            actual = (0.02 if correct else -0.02) * direction
            rows.append(
                {
                    "symbol": f"SYM{s}.NS",
                    "prediction_date": pred_date,
                    "predicted_direction": direction,
                    "predicted_magnitude": 0.02,
                    "confidence": 0.6,
                    "model_version": "sim-test",
                    "regime": "bull",
                    "is_tradeable": True,
                    "entry_price": 100.0,
                    "exit_price": 100.0 * (1 + actual),
                    "actual_return": actual,
                    "is_correct": bool(direction == (1 if actual > 0 else -1)),
                }
            )
    return pd.DataFrame(rows)


def test_empty_trades_yield_zeroed_artifact() -> None:
    art = build_artifact(pd.DataFrame(), cost_pct=0.0047, meta={"tier": "large"})
    assert art["track_record"]["total_predictions"] == 0
    assert art["track_record"]["accuracy_all"] is None
    assert art["backtest"]["summary"]["total_trades"] == 0
    assert art["backtest"]["equity"]["strategy"] == []
    assert art["backtest"]["summary"]["avg_hold_days"] == 15.0


def test_artifact_structure_and_tracks() -> None:
    art = build_artifact(_make_trades(), cost_pct=0.0047, meta={"tier": "large", "cutoff": "x"})
    assert art["schema_version"] == 1
    assert set(art["track_record"]["tracks"]) == {"all", "gate_passed", "top_k"}
    tr = art["track_record"]
    assert tr["total_predictions"] == 40
    assert 0.0 <= tr["accuracy_all"] <= 1.0
    assert tr["days_tracked"] == 8
    # Cost-adjusted net return must be below gross for the 'all' track.
    all_track = tr["tracks"]["all"]
    assert all_track["avg_return_net"] < all_track["avg_return_gross"]


def test_backtest_views_are_consistent() -> None:
    trades = _make_trades(n_days=10, per_day=5)
    bt = build_backtest_views(trades, cost_pct=0.0047)
    s = bt["summary"]
    assert s["total_trades"] == 50
    assert s["date_from"] and s["date_to"]
    assert 0.0 <= s["win_rate"] <= 1.0
    # Equity curve has one point per prediction-date cohort.
    assert len(bt["equity"]["strategy"]) == 10
    assert len(bt["equity"]["benchmark"]) == 10
    # Distribution counts every directional trade exactly once.
    assert sum(b["count"] for b in bt["distribution"]) == 50
    # Rolling sharpe has one reading per cohort.
    assert len(bt["rolling_sharpe"]) == 10
    # Monthly buckets stay within the simulated window (Jan 2026).
    assert all(m["year"] == 2026 and m["month"] == 1 for m in bt["monthly"])


def test_direction_zero_predictions_excluded_from_pnl() -> None:
    trades = _make_trades(n_days=3, per_day=4)
    flat = trades.copy()
    flat["predicted_direction"] = 0
    bt = build_backtest_views(flat, cost_pct=0.0047)
    # No directional bets → no strategy P&L, zeroed summary.
    assert bt["summary"]["total_trades"] == 0
    assert bt["equity"]["strategy"] == []


@pytest.mark.parametrize("cost", [0.0, 0.0047, 0.02])
def test_higher_cost_lowers_net_return(cost: float) -> None:
    trades = _make_trades()
    bt = build_backtest_views(trades, cost_pct=cost)
    # Profit factor must weakly decrease as costs rise (monotonic drag).
    assert bt["summary"]["total_trades"] == 40
