"""Public dashboard API — verifiable prediction track record.

All endpoints are public (no auth) to allow anyone to verify
the system's prediction accuracy and paper trading performance.
"""

from __future__ import annotations

from datetime import date

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/dashboard", tags=["public-dashboard"])


class DailyPnLRecord(BaseModel):
    date: str
    portfolio_value: float
    daily_return: float
    cumulative_return: float
    n_positions: int
    n_correct: int
    n_total_predictions: int
    benchmark_return: float


class AccuracyByCategory(BaseModel):
    category: str
    total: int
    correct: int
    accuracy: float


class PublicTrackRecord(BaseModel):
    start_date: str | None
    end_date: str | None
    total_days: int
    total_predictions: int
    overall_accuracy: float | None
    cumulative_return: float
    benchmark_cumulative_return: float
    alpha: float
    accuracy_by_regime: list[AccuracyByCategory]
    accuracy_by_confidence: list[AccuracyByCategory]
    monthly_returns: list[dict]


@router.get("/track-record", response_model=PublicTrackRecord)
async def get_track_record() -> PublicTrackRecord:
    """Get full public track record with accuracy breakdowns."""

    from alphavedha.data.store import load_daily_pnl, load_paper_trades

    trades_df = await load_paper_trades()
    pnl_df = await load_daily_pnl()

    if trades_df.empty:
        return PublicTrackRecord(
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

    evaluated = trades_df[trades_df["is_correct"].notna()]

    overall_acc = float(evaluated["is_correct"].mean()) if not evaluated.empty else None

    cum_ret = 0.0
    bench_cum_ret = 0.0
    if not pnl_df.empty:
        cum_ret = float(pnl_df["cumulative_return"].iloc[-1])
        bench_cum_ret = float(pnl_df["benchmark_return"].sum())

    regime_acc: list[AccuracyByCategory] = []
    if "regime" in evaluated.columns:
        for regime, grp in evaluated.groupby("regime"):
            if regime is None:
                continue
            regime_acc.append(
                AccuracyByCategory(
                    category=str(regime),
                    total=len(grp),
                    correct=int(grp["is_correct"].sum()),
                    accuracy=float(grp["is_correct"].mean()),
                )
            )

    conf_acc: list[AccuracyByCategory] = []
    if not evaluated.empty and "confidence" in evaluated.columns:
        bins = [(0.55, 0.65, "0.55-0.65"), (0.65, 0.75, "0.65-0.75"), (0.75, 1.0, "0.75+")]
        for lo, hi, label in bins:
            grp = evaluated[(evaluated["confidence"] >= lo) & (evaluated["confidence"] < hi)]
            if not grp.empty:
                conf_acc.append(
                    AccuracyByCategory(
                        category=label,
                        total=len(grp),
                        correct=int(grp["is_correct"].sum()),
                        accuracy=float(grp["is_correct"].mean()),
                    )
                )

    monthly_rets: list[dict] = []
    if not pnl_df.empty:
        pnl_df["date"] = pnl_df["date"].apply(
            lambda x: x if isinstance(x, date) else date.fromisoformat(str(x))
        )
        pnl_df["month"] = pnl_df["date"].apply(lambda d: d.strftime("%Y-%m"))
        for month, grp in pnl_df.groupby("month"):
            monthly_rets.append(
                {
                    "month": month,
                    "return": float(grp["daily_return"].sum()),
                    "predictions": int(grp["n_total_predictions"].sum()),
                }
            )

    dates = trades_df["prediction_date"]
    start = str(dates.min()) if not dates.empty else None
    end = str(dates.max()) if not dates.empty else None

    return PublicTrackRecord(
        start_date=start,
        end_date=end,
        total_days=int(dates.nunique()) if not dates.empty else 0,
        total_predictions=len(trades_df),
        overall_accuracy=overall_acc,
        cumulative_return=cum_ret,
        benchmark_cumulative_return=bench_cum_ret,
        alpha=cum_ret - bench_cum_ret,
        accuracy_by_regime=regime_acc,
        accuracy_by_confidence=conf_acc,
        monthly_returns=monthly_rets,
    )


@router.get("/equity-curve")
async def get_equity_curve() -> list[DailyPnLRecord]:
    """Get daily equity curve for the paper portfolio."""
    from alphavedha.data.store import load_daily_pnl

    pnl_df = await load_daily_pnl()
    if pnl_df.empty:
        return []

    return [
        DailyPnLRecord(
            date=str(row["date"]),
            portfolio_value=row["portfolio_value"],
            daily_return=row["daily_return"],
            cumulative_return=row["cumulative_return"],
            n_positions=row["n_positions"],
            n_correct=row["n_correct"],
            n_total_predictions=row["n_total_predictions"],
            benchmark_return=row["benchmark_return"],
        )
        for _, row in pnl_df.iterrows()
    ]
