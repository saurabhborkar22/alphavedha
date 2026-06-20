"""Cost-adjusted live track record computed from paper trades.

Turns the raw ``paper_trades`` table into the numbers that decide whether the
system makes money: directional returns net of Indian transaction costs,
split into three tracks:

- ``all``:         every persisted prediction (measures raw model quality)
- ``gate_passed``: only predictions the meta-labeling gate marked tradeable —
                   the strategy as it would actually run
- ``top_k``:       the k highest-confidence directional calls per day — the
                   "best ideas" stream, measurable even on days the gate
                   keeps every position closed

Conventions:
- A trade's gross return is ``predicted_direction * actual_return`` so a
  correct short counts as a gain. Direction-0 (flat) predictions are never
  "traded" and contribute no P&L.
- Net return subtracts a flat round-trip cost fraction (see
  ``alphavedha.backtest.costs.compute_round_trip_cost_pct``). Shorts get the
  same charge — slightly conservative, since a futures short would pay lower
  STT than cash delivery.
- Sharpe and drawdown are computed on per-prediction-date cohort means. Each
  cohort is one ~15-trading-day bet, so Sharpe annualizes by
  sqrt(252 / 15). Cohorts from adjacent days hold overlapping positions;
  treat drawdown as indicative until the history is several horizons long.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

HORIZON_TRADING_DAYS = 15
_ANNUALIZATION = float(np.sqrt(252 / HORIZON_TRADING_DAYS))

# Fallback when is_tradeable was not persisted (rows predating the column).
# Mid-range of the regime-dependent thresholds (0.40-0.55).
DEFAULT_GATE_CONFIDENCE = 0.50
DEFAULT_TOP_K = 5


@dataclass
class TrackStats:
    """Performance of one selection rule over the paper trade history."""

    name: str
    n_selected: int = 0
    n_evaluated: int = 0
    n_wins_net: int = 0
    win_rate_net: float | None = None
    avg_return_gross: float | None = None
    avg_return_net: float | None = None
    total_return_net: float = 0.0
    profit_factor_net: float | None = None
    sharpe_net: float | None = None
    max_drawdown_net: float = 0.0


@dataclass
class TrackRecord:
    round_trip_cost_pct: float
    all_predictions: TrackStats
    gate_passed: TrackStats
    top_k: TrackStats


def _directional_evaluated(trades: pd.DataFrame) -> pd.DataFrame:
    """Matured trades with an actual directional bet (direction != 0)."""
    mask = (trades["predicted_direction"] != 0) & trades["actual_return"].notna()
    out = trades[mask].copy()
    out["gross"] = out["predicted_direction"].astype(float) * out["actual_return"].astype(float)
    return out


def compute_track_stats(name: str, selected: pd.DataFrame, cost_pct: float) -> TrackStats:
    """Compute win rate, expectancy, Sharpe, and drawdown for one track.

    ``cost_pct`` is the full round-trip cost as a fraction of notional;
    pass 0.0 for gross statistics.
    """
    stats = TrackStats(name=name, n_selected=len(selected))
    if selected.empty:
        return stats

    evaluated = _directional_evaluated(selected)
    stats.n_evaluated = len(evaluated)
    if evaluated.empty:
        return stats

    net = evaluated["gross"] - cost_pct
    wins = net[net > 0]
    losses = net[net <= 0]

    stats.n_wins_net = len(wins)
    stats.win_rate_net = float(len(wins) / len(net))
    stats.avg_return_gross = float(evaluated["gross"].mean())
    stats.avg_return_net = float(net.mean())
    stats.total_return_net = float(net.sum())

    gross_losses = float(losses.abs().sum())
    if gross_losses > 0:
        stats.profit_factor_net = float(wins.sum() / gross_losses)

    # One cohort per prediction date: the equal-weight 15-trading-day bet
    # placed that morning.
    cohort = net.groupby(evaluated["prediction_date"]).mean().sort_index()
    if len(cohort) >= 2 and float(cohort.std()) > 0:
        stats.sharpe_net = float(cohort.mean() / cohort.std() * _ANNUALIZATION)

    equity = (1.0 + cohort).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    stats.max_drawdown_net = float(drawdown.min())

    return stats


def _select_gate_passed(trades: pd.DataFrame, gate_confidence: float) -> pd.DataFrame:
    """Rows the meta-labeling gate approved at prediction time.

    Uses the persisted ``is_tradeable`` flag when available; legacy rows
    (null flag) fall back to a flat confidence threshold, which only
    approximates the regime-dependent gate.
    """
    if "is_tradeable" in trades.columns and trades["is_tradeable"].notna().any():
        mask = trades["is_tradeable"].astype("boolean").fillna(False).astype(bool)
        return trades[mask]
    return trades[trades["confidence"] >= gate_confidence]


def _select_top_k(trades: pd.DataFrame, k: int) -> pd.DataFrame:
    """The k highest-confidence directional predictions per prediction date."""
    candidates = trades[trades["predicted_direction"] != 0]
    if candidates.empty:
        return candidates
    return candidates.sort_values("confidence", ascending=False).groupby("prediction_date").head(k)


def compute_track_record(
    trades: pd.DataFrame,
    round_trip_cost_pct: float,
    gate_confidence: float = DEFAULT_GATE_CONFIDENCE,
    top_k: int = DEFAULT_TOP_K,
) -> TrackRecord:
    """Compute the three-track, cost-adjusted live track record.

    ``trades`` is the frame returned by ``load_paper_trades()``. An empty
    frame yields zeroed stats rather than raising.
    """
    if trades.empty:
        return TrackRecord(
            round_trip_cost_pct=round_trip_cost_pct,
            all_predictions=TrackStats(name="all"),
            gate_passed=TrackStats(name="gate_passed"),
            top_k=TrackStats(name=f"top_{top_k}"),
        )

    record = TrackRecord(
        round_trip_cost_pct=round_trip_cost_pct,
        all_predictions=compute_track_stats("all", trades, round_trip_cost_pct),
        gate_passed=compute_track_stats(
            "gate_passed",
            _select_gate_passed(trades, gate_confidence),
            round_trip_cost_pct,
        ),
        top_k=compute_track_stats(
            f"top_{top_k}",
            _select_top_k(trades, top_k),
            round_trip_cost_pct,
        ),
    )

    logger.info(
        "track_record_computed",
        n_trades=len(trades),
        n_evaluated=record.all_predictions.n_evaluated,
        gate_selected=record.gate_passed.n_selected,
        cost_pct=round_trip_cost_pct,
    )
    return record


def compute_track_records_by_strategy(
    trades: pd.DataFrame,
    round_trip_cost_pct: float,
    gate_confidence: float = DEFAULT_GATE_CONFIDENCE,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, TrackRecord]:
    """Compute per-strategy track records.

    Returns a dict keyed by strategy name. Falls back to a single
    'ensemble_v1' entry if the strategy column is missing.
    """
    if trades.empty:
        return {}

    if "strategy" not in trades.columns:
        return {
            "ensemble_v1": compute_track_record(trades, round_trip_cost_pct, gate_confidence, top_k)
        }

    results: dict[str, TrackRecord] = {}
    for strategy_name, group in trades.groupby("strategy"):
        results[str(strategy_name)] = compute_track_record(
            group.reset_index(drop=True), round_trip_cost_pct, gate_confidence, top_k
        )
    return results
