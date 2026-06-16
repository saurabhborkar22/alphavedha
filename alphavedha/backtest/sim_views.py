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
    DEFAULT_TOP_K,
    HORIZON_TRADING_DAYS,
    _select_gate_passed,
    _select_top_k,
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


def build_range_view(
    strategy_points: list[dict[str, Any]],
    benchmark_points: list[dict[str, Any]],
    start: str | None,
    end: str | None,
) -> dict[str, Any]:
    """Per-day + date-range performance, re-sliced from the dated equity curve.

    Reconstructs the per-cohort returns from the strategy equity curve (one
    equal-weight ~15-trading-day bet placed each trading day), filters to the
    inclusive [start, end] ISO-date window, and recomputes window metrics +
    a per-day breakdown. Pure function over the artifact's ``equity`` points —
    no trade-level data, DB, or model code required, so it works on the
    existing committed simulation artifact with zero recomputation.

    Returns fractions (total_return/cagr/max_drawdown/win_rate/benchmark) — the
    same unit convention as the backtest summary; the UI multiplies by 100.
    """
    avail = {
        "date_from": str(strategy_points[0]["date"]) if strategy_points else None,
        "date_to": str(strategy_points[-1]["date"]) if strategy_points else None,
    }
    empty: dict[str, Any] = {
        "summary": {
            "date_from": "",
            "date_to": "",
            "n_days": 0,
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "best_day": 0.0,
            "worst_day": 0.0,
            "benchmark_return": 0.0,
            "excess_return": 0.0,
        },
        "per_day": [],
        "equity": {"strategy": [], "benchmark": []},
        "available": avail,
    }
    if not strategy_points:
        return empty

    def _returns(points: list[dict[str, Any]]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        prev = INITIAL_VALUE
        for p in points:
            y = float(p["y"])
            if prev > 0:
                out.append((str(p["date"]), y / prev - 1.0))
            prev = y
        return out

    def _in_range(d: str) -> bool:
        return (start is None or d >= start) and (end is None or d <= end)

    window = [(d, r) for d, r in _returns(strategy_points) if _in_range(d)]
    if not window:
        return empty

    dates = [d for d, _ in window]
    rets = np.array([r for _, r in window], dtype=float)
    eq = INITIAL_VALUE * np.cumprod(1.0 + rets)
    eq_series = pd.Series(eq, index=pd.to_datetime(dates))

    total_return = float(eq[-1] / INITIAL_VALUE - 1.0)
    max_dd = _max_drawdown(eq_series)
    sharpe = (
        float(rets.mean() / rets.std() * _ANNUALIZATION)
        if len(rets) >= 2 and float(rets.std()) > 0
        else 0.0
    )
    win_rate = float((rets > 0).mean())

    bench_map = {d: r for d, r in _returns(benchmark_points)}
    bench_rets = np.array([bench_map.get(d, 0.0) for d in dates], dtype=float)
    bench_total = float(np.prod(1.0 + bench_rets) - 1.0)
    bench_eq = INITIAL_VALUE * np.cumprod(1.0 + bench_rets)

    return {
        "summary": {
            "date_from": dates[0],
            "date_to": dates[-1],
            "n_days": len(rets),
            "total_return": round(total_return, 4),
            "cagr": round(_annualized_return(total_return, len(rets)), 4),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "best_day": round(float(rets.max()), 4),
            "worst_day": round(float(rets.min()), 4),
            "benchmark_return": round(bench_total, 4),
            "excess_return": round(total_return - bench_total, 4),
        },
        "per_day": [
            {"date": d, "return_pct": round(r * 100, 3), "equity": round(float(e), 2)}
            for (d, r), e in zip(window, eq, strict=True)
        ],
        "equity": {
            "strategy": [
                {"date": d, "y": round(float(e), 2)} for d, e in zip(dates, eq, strict=True)
            ],
            "benchmark": [
                {"date": d, "y": round(float(e), 2)} for d, e in zip(dates, bench_eq, strict=True)
            ],
        },
        "available": avail,
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


def build_calibration(
    trades: pd.DataFrame, cost_pct: float, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Confidence-decile reliability curve over directional bets.

    For each confidence bin, the realized win-rate. If win-rate does NOT rise
    with ``conf_mean``, the model's confidence is mis-calibrated (flat) or
    inverted (declining) out-of-sample — i.e. selecting by confidence hurts.
    """
    d = _directional_net(trades, cost_pct)
    if len(d) < n_bins:
        return []
    # Rank-based bins → even sample sizes even when confidence has ties.
    d = d.assign(_rank=d["confidence"].rank(method="first"))
    d["_bin"] = pd.qcut(d["_rank"], n_bins, labels=False)
    out: list[dict[str, Any]] = []
    for b, grp in d.groupby("_bin"):
        out.append(
            {
                "bin": int(b),
                "n": len(grp),
                "conf_mean": round(float(grp["confidence"].mean()), 4),
                "conf_lo": round(float(grp["confidence"].min()), 4),
                "conf_hi": round(float(grp["confidence"].max()), 4),
                "win_rate": round(float((grp["gross"] > 0).mean()), 4),
                "avg_gross": round(float(grp["gross"].mean()), 5),
                "avg_net": round(float(grp["net"].mean()), 5),
            }
        )
    return out


def build_cost_sensitivity(
    trades: pd.DataFrame,
    base_cost: float,
    multipliers: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
) -> list[dict[str, Any]]:
    """Net P&L of the raw (all-prediction) signal at several cost levels.

    At 0x the gross edge is visible; at 1x the live cost; at 2x a stress case.
    Shows how much of the result is edge vs cost.
    """
    d = _directional_net(trades, cost_pct=0.0)
    if d.empty:
        return []
    gross = d["gross"]
    out: list[dict[str, Any]] = []
    for m in multipliers:
        net = gross - base_cost * m
        out.append(
            {
                "cost_mult": m,
                "cost_pct": round(base_cost * m, 5),
                "avg_net": round(float(net.mean()), 5),
                "total_net": round(float(net.sum()), 3),
                "win_rate_net": round(float((net > 0).mean()), 4),
            }
        )
    return out


def _track_monthly(selected: pd.DataFrame, cost_pct: float) -> list[dict[str, Any]]:
    d = _directional_net(selected, cost_pct)
    if d.empty:
        return []
    return _monthly_returns(_cohort_mean(d["net"], d["prediction_date"]))


def build_diagnostics(
    trades: pd.DataFrame,
    cost_pct: float,
    gate_confidence: float = DEFAULT_GATE_CONFIDENCE,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Calibration, cost-sensitivity, and per-track monthly breakdowns."""
    if trades.empty:
        return {
            "calibration": [],
            "cost_sensitivity": [],
            "monthly_by_track": {"all": [], "gate_passed": [], "top_k": []},
        }
    return {
        "calibration": build_calibration(trades, cost_pct),
        "cost_sensitivity": build_cost_sensitivity(trades, cost_pct),
        "monthly_by_track": {
            "all": _track_monthly(trades, cost_pct),
            "gate_passed": _track_monthly(_select_gate_passed(trades, gate_confidence), cost_pct),
            "top_k": _track_monthly(_select_top_k(trades, top_k), cost_pct),
        },
    }


def build_artifact(
    trades: pd.DataFrame,
    cost_pct: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full simulation artifact (track_record + backtest + diagnostics)."""
    return {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "meta": meta,
        "track_record": build_track_record_view(trades, cost_pct),
        "backtest": build_backtest_views(trades, cost_pct),
        "diagnostics": build_diagnostics(trades, cost_pct),
    }
