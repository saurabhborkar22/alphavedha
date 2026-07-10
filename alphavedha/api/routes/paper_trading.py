"""Paper trading API — record predictions, track P&L, verify track record.

Predictions are timestamped before market open (9:15 AM IST).
After market close, outcomes are recorded and P&L updated.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from alphavedha.monitoring.track_record import TrackStats

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/paper", tags=["paper-trading"])


class PaperTradeRequest(BaseModel):
    symbol: str
    predicted_direction: int = Field(..., ge=-1, le=1)
    predicted_magnitude: float
    confidence: float = Field(..., ge=0, le=1)
    model_version: str
    strategy: str = "ensemble_v1"
    regime: str | None = None
    is_tradeable: bool | None = None
    entry_price: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None


class PaperTradeResponse(BaseModel):
    symbol: str
    prediction_date: str
    strategy: str
    predicted_direction: int
    confidence: float
    model_version: str
    created_at: str


class TradeOutcomeRequest(BaseModel):
    symbol: str
    prediction_date: str
    strategy: str = "ensemble_v1"
    exit_price: float
    actual_return: float
    is_correct: bool
    exit_reason: str | None = None


class TrackStatsOut(BaseModel):
    """Cost-adjusted performance of one selection rule (see monitoring/track_record.py)."""

    name: str
    n_selected: int
    n_evaluated: int
    n_wins_net: int
    win_rate_net: float | None
    avg_return_gross: float | None
    avg_return_net: float | None
    total_return_net: float
    profit_factor_net: float | None
    sharpe_net: float | None
    max_drawdown_net: float


class DashboardSummary(BaseModel):
    total_predictions: int
    correct_predictions: int
    accuracy_7d: float | None
    accuracy_30d: float | None
    accuracy_all: float | None
    # Directional gross return summed over evaluated trades (a correct short
    # counts as a gain). See `tracks` for cost-adjusted numbers.
    total_return: float
    sharpe_ratio: float | None
    max_drawdown: float
    days_tracked: int
    round_trip_cost_pct: float | None = None
    tracks: dict[str, TrackStatsOut] | None = None


class PredictionRecord(BaseModel):
    symbol: str
    prediction_date: str
    strategy: str = "ensemble_v1"
    predicted_direction: int
    predicted_magnitude: float
    confidence: float
    model_version: str
    regime: str | None
    is_tradeable: bool | None = None
    entry_price: float | None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    exit_price: float | None
    exit_reason: str | None = None
    actual_return: float | None
    is_correct: bool | None


@router.post("/predict", response_model=PaperTradeResponse)
async def record_prediction(req: PaperTradeRequest) -> PaperTradeResponse:
    """Record a pre-market prediction for paper trading."""
    from alphavedha.data.store import store_paper_trade

    today = date.today()

    row = {
        "symbol": req.symbol,
        "prediction_date": today,
        "strategy": req.strategy,
        "predicted_direction": req.predicted_direction,
        "predicted_magnitude": req.predicted_magnitude,
        "confidence": req.confidence,
        "model_version": req.model_version,
        "regime": req.regime,
        "is_tradeable": req.is_tradeable,
        "entry_price": req.entry_price,
        "stop_loss_price": req.stop_loss_price,
        "take_profit_price": req.take_profit_price,
    }

    try:
        await store_paper_trade(row)
    except Exception as e:
        logger.error("paper_trade_store_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to store prediction") from None

    return PaperTradeResponse(
        symbol=req.symbol,
        prediction_date=today.isoformat(),
        strategy=req.strategy,
        predicted_direction=req.predicted_direction,
        confidence=req.confidence,
        model_version=req.model_version,
        created_at=datetime.now().isoformat(),
    )


@router.post("/outcome")
async def record_outcome(req: TradeOutcomeRequest) -> dict[str, str]:
    """Record the actual outcome for a paper trade after market close."""
    from alphavedha.data.store import update_paper_trade_outcome

    try:
        pred_date = date.fromisoformat(req.prediction_date)
        await update_paper_trade_outcome(
            symbol=req.symbol,
            prediction_date=pred_date,
            exit_price=req.exit_price,
            actual_return=req.actual_return,
            is_correct=req.is_correct,
            exit_reason=req.exit_reason,
            strategy=req.strategy,
        )
    except Exception as e:
        logger.error("paper_outcome_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update outcome") from None

    return {"status": "updated", "symbol": req.symbol, "date": req.prediction_date}


@router.post("/evaluate-stops")
async def evaluate_stops(evaluation_date: str | None = None) -> dict[str, Any]:
    """Check open paper trades for stop-loss or take-profit hits.

    Delegates to the shared stop-evaluation service — the scheduler runs the
    same check daily after the market-close data refresh; this endpoint
    exists for manual/backfill runs.
    """
    from alphavedha.services.stop_evaluation import evaluate_stop_hits

    eval_date = date.fromisoformat(evaluation_date) if evaluation_date else None
    result: dict[str, Any] = dict(await evaluate_stop_hits(eval_date))
    return result


def _track_stats_out(stats: TrackStats) -> TrackStatsOut:
    return TrackStatsOut(**asdict(stats))


@router.get("/dashboard", response_model=DashboardSummary)
async def get_dashboard(strategy: str | None = None) -> DashboardSummary:
    """Paper trading dashboard: accuracy plus the cost-adjusted track record.

    Returns are directional (predicted_direction * actual_return), so a
    correct short counts as a gain. `tracks` reports gross and net-of-cost
    stats for all predictions, gate-passed trades, and top-5 daily picks.
    """
    from alphavedha.backtest.costs import compute_long_short_cost_pct
    from alphavedha.config import get_config
    from alphavedha.data.store import load_paper_trades
    from alphavedha.monitoring.track_record import compute_track_record, compute_track_stats

    trades_df = await load_paper_trades(strategy=strategy)
    # Longs = cash delivery; shorts = stock futures (a swing short can't be held
    # overnight in the cash market). Each leg is charged its real round-trip.
    cost_pct, short_cost_pct = compute_long_short_cost_pct("large", get_config().backtest)

    if trades_df.empty:
        return DashboardSummary(
            total_predictions=0,
            correct_predictions=0,
            accuracy_7d=None,
            accuracy_30d=None,
            accuracy_all=None,
            total_return=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            days_tracked=0,
            round_trip_cost_pct=cost_pct,
            tracks=None,
        )

    total = len(trades_df)
    evaluated = trades_df[trades_df["is_correct"].notna()]
    correct = int(evaluated["is_correct"].sum()) if not evaluated.empty else 0

    today = date.today()
    trades_df["prediction_date"] = trades_df["prediction_date"].apply(
        lambda x: x if isinstance(x, date) else date.fromisoformat(str(x))
    )

    def _accuracy_window(days: int) -> float | None:
        from datetime import timedelta

        cutoff = today - timedelta(days=days)
        window = evaluated[evaluated["prediction_date"] >= cutoff]
        if window.empty:
            return None
        return float(window["is_correct"].mean())

    acc_7d = _accuracy_window(7)
    acc_30d = _accuracy_window(30)
    acc_all = float(evaluated["is_correct"].mean()) if not evaluated.empty else None

    # Legacy top-level fields stay gross; cost-adjusted numbers live in tracks.
    gross_all = compute_track_stats("all_gross", trades_df, cost_pct=0.0)
    record = compute_track_record(
        trades_df, round_trip_cost_pct=cost_pct, short_round_trip_cost_pct=short_cost_pct
    )

    return DashboardSummary(
        total_predictions=total,
        correct_predictions=correct,
        accuracy_7d=acc_7d,
        accuracy_30d=acc_30d,
        accuracy_all=acc_all,
        total_return=gross_all.total_return_net,
        sharpe_ratio=gross_all.sharpe_net,
        max_drawdown=gross_all.max_drawdown_net,
        days_tracked=int(trades_df["prediction_date"].nunique()),
        round_trip_cost_pct=cost_pct,
        tracks={
            "all": _track_stats_out(record.all_predictions),
            "gate_passed": _track_stats_out(record.gate_passed),
            "top_k": _track_stats_out(record.top_k),
        },
    )


@router.get("/trades", response_model=list[PredictionRecord])
async def list_trades(
    symbol: str | None = None,
    strategy: str | None = None,
    limit: int = 100,
) -> list[PredictionRecord]:
    """List paper trade predictions with outcomes."""
    from alphavedha.data.store import load_paper_trades

    trades_df = await load_paper_trades(symbol=symbol, strategy=strategy)

    if trades_df.empty:
        return []

    trades_df = trades_df.tail(limit)

    return [
        PredictionRecord(
            symbol=row["symbol"],
            prediction_date=str(row["prediction_date"]),
            strategy=row.get("strategy", "ensemble_v1"),
            predicted_direction=row["predicted_direction"],
            predicted_magnitude=row["predicted_magnitude"],
            confidence=row["confidence"],
            model_version=row["model_version"],
            regime=row.get("regime"),
            is_tradeable=row.get("is_tradeable"),
            entry_price=row.get("entry_price"),
            stop_loss_price=row.get("stop_loss_price"),
            take_profit_price=row.get("take_profit_price"),
            exit_price=row.get("exit_price"),
            exit_reason=row.get("exit_reason"),
            actual_return=row.get("actual_return"),
            is_correct=row.get("is_correct"),
        )
        for _, row in trades_df.iterrows()
    ]


@router.get("/simulation")
async def get_simulation() -> dict[str, Any]:
    """Out-of-sample historical-simulation track record (backfilled).

    Distinct from /paper/dashboard, which reports live forward trades. Returns
    the 3-track cost-adjusted record produced by the one-time sim job
    (scripts/sim_paper_trading.py), or ``available: false`` when no simulation
    artifact has been generated yet.
    """
    from alphavedha.api.sim_artifact import load_sim_artifact

    art = load_sim_artifact()
    if not art:
        return {
            "available": False,
            "track_record": None,
            "diagnostics": None,
            "meta": None,
            "generated_at": None,
        }
    return {
        "available": True,
        "track_record": art.get("track_record"),
        "diagnostics": art.get("diagnostics"),
        "meta": art.get("meta"),
        "generated_at": art.get("generated_at"),
    }


@router.get("/simulations")
async def list_simulations() -> dict[str, Any]:
    """All archived historical-simulation runs, newest first (for the run picker)."""
    from alphavedha.api.sim_artifact import list_sim_runs

    runs = list_sim_runs()
    return {"runs": runs, "count": len(runs)}


@router.get("/simulation/{slug}")
async def get_simulation_run(slug: str) -> dict[str, Any]:
    """One archived simulation run by slug — full artifact incl. backtest views."""
    from alphavedha.api.sim_artifact import load_sim_run

    art = load_sim_run(slug)
    if not art:
        return {
            "available": False,
            "slug": slug,
            "track_record": None,
            "diagnostics": None,
            "backtest": None,
            "meta": None,
            "generated_at": None,
        }
    return {
        "available": True,
        "slug": slug,
        "track_record": art.get("track_record"),
        "diagnostics": art.get("diagnostics"),
        "backtest": art.get("backtest"),
        "meta": art.get("meta"),
        "generated_at": art.get("generated_at"),
    }
