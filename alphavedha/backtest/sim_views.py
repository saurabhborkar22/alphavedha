"""Pure view-builders for the historical-simulation artifact.

Given the evaluated simulated trades — the same shape as the ``paper_trades``
table after outcomes are filled — produce the JSON payload the UI consumes:

- ``track_record``: the three-track cost-adjusted record (paper page), reusing
  :func:`alphavedha.monitoring.track_record.compute_track_record` so the numbers
  match the live dashboard exactly.
- ``backtest``: walk-forward equity / monthly / distribution / rolling-Sharpe
  plus a summary block (backtest page), derived from the gate-passed cohort —
  i.e. the strategy as it would actually trade.

No I/O, no DB, no model code — pure pandas, so it is unit-testable with
synthetic frames and shared by both the runner script and the tests. The
strategy series is the per-prediction-date cohort mean: one equal-weight,
~15-trading-day bet placed each morning (overlapping cohorts, see
``track_record`` module docstring).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from alphavedha.monitoring.track_record import (
    DEFAULT_GATE_CONFIDENCE,
    HORIZON_TRADING_DAYS,
    _select_gate_passed,
    compute_track_record,
)

INITIAL_VALUE = 1_000_000.0
_ANNUALIZATION = float(np.sqrt(252 / HORIZON_TRADING_DAYS))
ROLLING_WINDOW = 21  # cohorts (~1 trading month of overlapping bets)

# Buckets for the net-return distribution histogram (matches the UI labels).
_DIST_BUCKETS: list[tuple[str, float, float]] = [
    ("< -5%", -np.inf, -0.05),
    ("-5 to -3%", -0.05, -0.03),
    ("-3 to -1%", -0.03, -0.01),
    ("-1 to 0%", -0.01, 0.0),
    ("0 to 1%", 0.0, 0.01),
    ("1 to 3%", 0.01, 0.03),
    ("3 to 5%", 0.03, 0.05),
    ("> 5%", 0.05, np.inf),
]


def _iso(value: Any) -> str:
    """Render a date/Timestamp/str as an ISO date string."""
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _directional_net(trades: pd.DataFrame, cost_pct: float) -> pd.DataFrame:
    """Matured directional bets with gross and net (of cost) returns."""
    mask = (trades["predicted_direction"] != 0) & trades["actual_return"].notna()
    out = trades[mask].copy()
    if out.empty:
        return out
    out["gross"] = out["predicted_direction"].astype(float) * out["actual_return"].astype(float)
    out["net"] = out["gross"] - cost_pct
    return out


def _cohort_mean(per_trade: pd.Series, prediction_date: pd.Series) -> pd.Series:
    """Equal-weight cohort series: mean per prediction date, sorted by date."""
    if per_trade.empty:
        return pd.Series(dtype=float)
    return per_trade.groupby(prediction_date).mean().sort_index()


def _equity_curve(cohort: pd.Series) -> pd.Series:
    """Growth of INITIAL_VALUE compounding the per-cohort returns."""
    if cohort.empty:
        return pd.Series(dtype=float)
    return INITIAL_VALUE * (1.0 + cohort).cumprod()


def _series_points(equity: pd.Series) -> list[dict[str, Any]]:
    return [{"date": _iso(idx), "y": round(float(val), 2)} for idx, val in equity.items()]


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _rolling_sharpe(cohort: pd.Series, window: int = ROLLING_WINDOW) -> list[dict[str, float]]:
    if len(cohort) < 2:
        return []
    out: list[dict[str, float]] = []
    for i in range(1, len(cohort) + 1):
        window_vals = cohort.iloc[max(0, i - window) : i]
        if len(window_vals) >= 2 and float(window_vals.std()) > 0:
            sharpe = float(window_vals.mean() / window_vals.std() * _ANNUALIZATION)
        else:
            sharpe = 0.0
        out.append({"y": round(sharpe, 3)})
    return out


def _monthly_returns(cohort: pd.Series) -> list[dict[str, Any]]:
    """Compounded strategy return within each calendar month of the window."""
    if cohort.empty:
        return []
    idx = pd.to_datetime(pd.Series(cohort.index)).dt.to_period("M")
    grouped = 1.0 + cohort.to_numpy()
    df = pd.DataFrame({"period": idx.to_numpy(), "growth": grouped})
    out: list[dict[str, Any]] = []
    for period, grp in df.groupby("period", sort=True):
        ret = float(grp["growth"].prod() - 1.0)
        out.append(
            {
                "year": int(period.year),
                "month": int(period.month),
                "return_pct": round(ret * 100, 2),
            }
        )
    return out


def _distribution(net_per_trade: pd.Series) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    vals = net_per_trade.to_numpy()
    for label, lo, hi in _DIST_BUCKETS:
        count = int(np.sum((vals >= lo) & (vals < hi)))
        out.append({"label": label, "count": count})
    return out


def _annualized_return(total_return: float, n_cohorts: int) -> float:
    """CAGR implied by total strategy return over n_cohorts ~15-day bets."""
    if n_cohorts <= 0:
        return 0.0
    years = (n_cohorts * HORIZON_TRADING_DAYS) / 252.0
    if years <= 0:
        return 0.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def build_backtest_views(
    trades: pd.DataFrame,
    cost_pct: float,
    gate_confidence: float = DEFAULT_GATE_CONFIDENCE,
) -> dict[str, Any]:
    """Backtest-page views from the gate-passed (tradeable) cohort."""
    empty = {
        "summary": {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "avg_hold_days": float(HORIZON_TRADING_DAYS),
            "profit_factor": 0.0,
            "calmar": 0.0,
            "date_from": "",
            "date_to": "",
        },
        "equity": {"strategy": [], "benchmark": []},
        "monthly": [],
        "distribution": [],
        "rolling_sharpe": [],
    }
    if trades.empty:
        return empty

    strat = _directional_net(_select_gate_passed(trades, gate_confidence), cost_pct)
    if strat.empty:
        return empty

    cohort = _cohort_mean(strat["net"], strat["prediction_date"])
    strat_equity = _equity_curve(cohort)

    # Benchmark: equal-weight long-the-basket forward return per cohort date.
    bench_eval = trades[trades["actual_return"].notna()]
    bench_cohort = _cohort_mean(
        bench_eval["actual_return"].astype(float), bench_eval["prediction_date"]
    )
    bench_equity = _equity_curve(bench_cohort.reindex(cohort.index).fillna(0.0))

    net = strat["net"]
    wins = net[net > 0]
    losses = net[net <= 0]
    total_return = float(strat_equity.iloc[-1] / INITIAL_VALUE - 1.0)
    max_dd = _max_drawdown(strat_equity)
    sharpe = (
        float(cohort.mean() / cohort.std() * _ANNUALIZATION)
        if len(cohort) >= 2 and float(cohort.std()) > 0
        else 0.0
    )
    cagr = _annualized_return(total_return, len(cohort))
    gross_loss = float(losses.abs().sum())
    profit_factor = float(wins.sum() / gross_loss) if gross_loss > 0 else 0.0

    return {
        "summary": {
            "cagr": round(cagr, 4),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(float(len(wins) / len(net)), 4),
            "total_trades": len(net),
            "avg_hold_days": float(HORIZON_TRADING_DAYS),
            "profit_factor": round(profit_factor, 3),
            "calmar": round(cagr / abs(max_dd), 3) if max_dd < 0 else 0.0,
            "date_from": _iso(cohort.index[0]),
            "date_to": _iso(cohort.index[-1]),
        },
        "equity": {
            "strategy": _series_points(strat_equity),
            "benchmark": _series_points(bench_equity),
        },
        "monthly": _monthly_returns(cohort),
        "distribution": _distribution(net),
        "rolling_sharpe": _rolling_sharpe(cohort),
    }


def _accuracy_window(evaluated: pd.DataFrame, days: int) -> float | None:
    if evaluated.empty:
        return None
    last = pd.to_datetime(evaluated["prediction_date"]).max()
    cutoff = last - pd.Timedelta(days=days)
    window = evaluated[pd.to_datetime(evaluated["prediction_date"]) >= cutoff]
    if window.empty:
        return None
    return round(float(window["is_correct"].mean()), 4)


def build_track_record_view(trades: pd.DataFrame, cost_pct: float) -> dict[str, Any]:
    """Paper-page view: the three-track cost-adjusted record + accuracy."""
    record = compute_track_record(trades, round_trip_cost_pct=cost_pct)
    tracks = {
        "all": asdict(record.all_predictions),
        "gate_passed": asdict(record.gate_passed),
        "top_k": asdict(record.top_k),
    }
    if trades.empty:
        return {
            "tracks": tracks,
            "round_trip_cost_pct": cost_pct,
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy_all": None,
            "accuracy_30d": None,
            "days_tracked": 0,
        }

    evaluated = trades[trades["is_correct"].notna()]
    return {
        "tracks": tracks,
        "round_trip_cost_pct": cost_pct,
        "total_predictions": len(trades),
        "correct_predictions": int(evaluated["is_correct"].sum()) if not evaluated.empty else 0,
        "accuracy_all": round(float(evaluated["is_correct"].mean()), 4)
        if not evaluated.empty
        else None,
        "accuracy_30d": _accuracy_window(evaluated, 30),
        "days_tracked": int(trades["prediction_date"].nunique()),
    }


def build_artifact(
    trades: pd.DataFrame,
    cost_pct: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full simulation artifact (track_record + backtest + meta)."""
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "meta": meta,
        "track_record": build_track_record_view(trades, cost_pct),
        "backtest": build_backtest_views(trades, cost_pct),
    }
