"""Tests for the historical-simulation view-builders (pure, no DB)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from alphavedha.backtest.sim_views import (
    build_artifact,
    build_backtest_views,
    build_calibration,
    build_cost_sensitivity,
    build_diagnostics,
    build_range_view,
)


def test_build_range_view_slices_and_computes() -> None:
    # 4 dated equity points (INITIAL_VALUE=1e6). Returns: +10%, -5%, +5%, +5%.
    strat = [
        {"date": "2025-06-02", "y": 1_100_000.0},
        {"date": "2025-06-03", "y": 1_045_000.0},
        {"date": "2025-06-04", "y": 1_097_250.0},
        {"date": "2025-06-05", "y": 1_152_112.5},
    ]
    bench = [{"date": p["date"], "y": 1_000_000.0} for p in strat]  # flat benchmark

    full = build_range_view(strat, bench, None, None)
    assert full["summary"]["n_days"] == 4
    assert len(full["per_day"]) == 4
    assert full["per_day"][0]["date"] == "2025-06-02"
    assert full["per_day"][0]["return_pct"] == pytest.approx(10.0, abs=1e-6)
    assert full["summary"]["win_rate"] == pytest.approx(0.75)  # 3 of 4 positive
    assert full["available"]["date_from"] == "2025-06-02"

    sliced = build_range_view(strat, bench, "2025-06-03", "2025-06-04")
    assert [d["date"] for d in sliced["per_day"]] == ["2025-06-03", "2025-06-04"]
    assert sliced["summary"]["n_days"] == 2

    one = build_range_view(strat, bench, "2025-06-02", "2025-06-02")
    assert one["summary"]["n_days"] == 1
    assert one["summary"]["total_return"] == pytest.approx(0.10, abs=1e-4)

    assert build_range_view([], [], None, None)["summary"]["n_days"] == 0
    assert build_range_view(strat, bench, "2030-01-01", "2030-12-31")["per_day"] == []


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
    assert art["schema_version"] == 2
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


def _calibrated_trades(n: int = 200) -> pd.DataFrame:
    """Trades where higher confidence is genuinely more likely to be correct."""
    rng = np.random.default_rng(0)
    base = date(2026, 1, 5)
    rows: list[dict] = []
    for i in range(n):
        conf = float(rng.uniform(0.30, 0.90))
        direction = int(rng.choice([-1, 1]))
        correct = rng.random() < conf  # P(correct) rises with confidence
        actual = (0.02 if correct else -0.02) * direction
        rows.append(
            {
                "symbol": "SYM.NS",
                "prediction_date": base + timedelta(days=i % 20),
                "predicted_direction": direction,
                "predicted_magnitude": 0.02,
                "confidence": conf,
                "model_version": "calib-test",
                "regime": "bull",
                "is_tradeable": conf >= 0.5,
                "entry_price": 100.0,
                "exit_price": 100.0 * (1 + actual),
                "actual_return": actual,
                "is_correct": direction == (1 if actual > 0 else -1),
            }
        )
    return pd.DataFrame(rows)


def test_calibration_buckets_track_confidence() -> None:
    cal = build_calibration(_calibrated_trades(), cost_pct=0.0047, n_bins=10)
    assert len(cal) == 10
    # Positively-calibrated fixture: top decile wins more than the bottom.
    assert cal[-1]["conf_mean"] > cal[0]["conf_mean"]
    assert cal[-1]["win_rate"] > cal[0]["win_rate"]


def test_cost_sensitivity_monotonic() -> None:
    cs = build_cost_sensitivity(_make_trades(), base_cost=0.0047)
    assert [r["cost_mult"] for r in cs] == [0.0, 0.5, 1.0, 2.0]
    nets = [r["avg_net"] for r in cs]
    assert nets == sorted(nets, reverse=True)  # higher cost -> lower net
    assert cs[0]["cost_pct"] == 0.0


def test_diagnostics_in_artifact() -> None:
    art = build_artifact(_make_trades(), cost_pct=0.0047, meta={"tier": "large"})
    assert art["schema_version"] == 2
    diag = art["diagnostics"]
    assert set(diag) == {"calibration", "cost_sensitivity", "monthly_by_track"}
    assert set(diag["monthly_by_track"]) == {"all", "gate_passed", "top_k"}


def test_diagnostics_empty_safe() -> None:
    diag = build_diagnostics(pd.DataFrame(), cost_pct=0.0047)
    assert diag["calibration"] == []
    assert diag["cost_sensitivity"] == []
