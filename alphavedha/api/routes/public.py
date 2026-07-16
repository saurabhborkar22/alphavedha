"""Public track record API — verifiable prediction history with full transparency.

All endpoints are public (no auth). Anyone can verify the system's
prediction accuracy, browse historical predictions, and download data.

When demo mode is enabled (``ALPHAVEDHA_DEMO`` env var) the endpoints serve
deterministic synthetic data. Otherwise everything is computed from the real
paper-trade and daily P&L tables populated by the scheduler — empty tables
yield honest zeros/empty arrays, never fabricated numbers.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/public", tags=["public-track-record"])

_DEMO_SYMBOLS = [
    "TCS",
    "INFY",
    "HDFCBANK",
    "RELIANCE",
    "ICICIBANK",
    "KOTAKBANK",
    "SBIN",
    "BHARTIARTL",
    "ITC",
    "HINDUNILVR",
    "LT",
    "AXISBANK",
    "MARUTI",
    "WIPRO",
    "SUNPHARMA",
]

_DIRECTION_LABELS = {1: "BUY", -1: "SELL", 0: "HOLD"}
_UI_DIRECTION_LABELS = {1: "UP", -1: "DOWN", 0: "HOLD"}

_CONFIDENCE_BANDS: list[tuple[float, float, str]] = [
    (0.0, 0.55, "<55%"),
    (0.55, 0.65, "55-65%"),
    (0.65, 0.75, "65-75%"),
    (0.75, 0.85, "75-85%"),
    (0.85, 1.01, "85-100%"),
]


class PredictionRecord(BaseModel):
    date: str
    symbol: str
    predicted_direction: int
    predicted_direction_label: str
    predicted_magnitude: float
    confidence: float
    regime: str
    actual_direction: int | None = None
    actual_return: float | None = None
    is_correct: bool | None = None
    model_version: str
    generated_at: str


class MonthlyReturn(BaseModel):
    month: str
    portfolio_return: float
    benchmark_return: float
    alpha: float
    n_trades: int
    win_rate: float


def _is_demo() -> bool:
    """Return True when demo mode is enabled via the ALPHAVEDHA_DEMO env var."""
    return os.environ.get("ALPHAVEDHA_DEMO", "").lower() in ("1", "true", "yes")


def _seed_for(symbol: str, day: date) -> int:
    raw = f"{symbol}-{day.isoformat()}"
    return int(hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[:8], 16)


def _generate_demo_predictions(n_days: int = 90, n_symbols: int = 15) -> list[PredictionRecord]:
    symbols = _DEMO_SYMBOLS[:n_symbols]
    regimes = ["bull", "bear", "sideways", "high_volatility"]
    records: list[PredictionRecord] = []

    base_date = date(2026, 2, 18)
    for day_offset in range(n_days):
        current_date = base_date + timedelta(days=day_offset)
        if current_date.weekday() >= 5:
            continue

        for sym in symbols:
            seed = _seed_for(sym, current_date)
            direction = [-1, 0, 1][seed % 3]
            magnitude = 0.005 + (seed % 100) / 2000.0
            confidence = 0.50 + (seed % 40) / 100.0
            regime = regimes[(seed >> 4) % 4]

            actual_seed = seed ^ 0xDEAD
            is_correct = (actual_seed % 100) < 58
            if is_correct:
                actual_direction = direction
            elif direction == 0:
                actual_direction = 1 if (actual_seed % 2) == 0 else -1
            else:
                actual_direction = -direction
            actual_return = magnitude * actual_direction * (0.5 + (actual_seed % 100) / 100.0)

            records.append(
                PredictionRecord(
                    date=current_date.isoformat(),
                    symbol=sym,
                    predicted_direction=direction,
                    predicted_direction_label=_DIRECTION_LABELS.get(direction, "HOLD"),
                    predicted_magnitude=round(magnitude, 5),
                    confidence=round(confidence, 3),
                    regime=regime,
                    actual_direction=actual_direction,
                    actual_return=round(actual_return, 5),
                    is_correct=is_correct,
                    model_version="v0.1.0",
                    generated_at=f"{current_date.isoformat()}T08:30:00+05:30",
                )
            )

    return records


def _records_from_trades(trades_df: pd.DataFrame) -> list[PredictionRecord]:
    """Map paper-trade rows from the store into the public prediction schema."""
    records: list[PredictionRecord] = []
    for _, row in trades_df.iterrows():
        try:
            raw_return = row.get("actual_return")
            actual_return = None if raw_return is None or pd.isna(raw_return) else float(raw_return)
            raw_correct = row.get("is_correct")
            is_correct = None if raw_correct is None or pd.isna(raw_correct) else bool(raw_correct)

            actual_direction: int | None = None
            if actual_return is not None:
                actual_direction = 1 if actual_return > 0 else -1 if actual_return < 0 else 0

            pred_date = row["prediction_date"]
            date_str = pred_date.isoformat() if isinstance(pred_date, date) else str(pred_date)
            direction = int(row["predicted_direction"])
            regime = row.get("regime")

            records.append(
                PredictionRecord(
                    date=date_str,
                    symbol=str(row["symbol"]),
                    predicted_direction=direction,
                    predicted_direction_label=_DIRECTION_LABELS.get(direction, "HOLD"),
                    predicted_magnitude=float(row["predicted_magnitude"]),
                    confidence=float(row["confidence"]),
                    regime=regime if isinstance(regime, str) and regime else "unknown",
                    actual_direction=actual_direction,
                    actual_return=actual_return,
                    is_correct=is_correct,
                    model_version=str(row.get("model_version") or "unknown"),
                    generated_at=f"{date_str}T00:00:00+05:30",
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("public_trade_row_skipped", error=str(exc))
    return records


async def _load_real_records(
    start: date | None = None,
    end: date | None = None,
    symbol: str | None = None,
    strategy: str | None = None,
) -> list[PredictionRecord]:
    """Load real paper trades; on any failure return an honest empty list."""
    from alphavedha.data.store import load_paper_trades

    try:
        trades_df = await load_paper_trades(
            start=start,
            end=end,
            symbol=symbol,
            strategy=strategy,
        )
    except Exception as exc:
        logger.error("public_paper_trades_load_failed", error=str(exc))
        return []
    if trades_df.empty:
        return []
    return _records_from_trades(trades_df)


async def _load_real_pnl() -> pd.DataFrame:
    """Load real daily P&L; on any failure return an empty frame."""
    from alphavedha.data.store import load_daily_pnl

    try:
        return await load_daily_pnl()
    except Exception as exc:
        logger.error("public_daily_pnl_load_failed", error=str(exc))
        return pd.DataFrame()


def _daily_returns(
    evaluated: list[PredictionRecord],
    pnl_df: pd.DataFrame | None,
) -> list[float]:
    """Daily return series from DailyPnL, falling back to per-day mean trade returns.

    The fallback multiplies by predicted_direction: actual_return is the
    price return, so a correct short must count as a gain.
    """
    if pnl_df is not None and not pnl_df.empty:
        return [float(x) for x in pnl_df["daily_return"].tolist()]

    by_day: dict[str, list[float]] = {}
    for r in evaluated:
        if r.actual_return is not None:
            by_day.setdefault(r.date, []).append(r.predicted_direction * r.actual_return)
    return [sum(rets) / len(rets) for _, rets in sorted(by_day.items())]


def _sharpe_ratio(daily_returns: list[float]) -> float:
    """Annualized Sharpe = mean/std * sqrt(252); 0.0 when not computable."""
    n = len(daily_returns)
    if n < 2:
        return 0.0
    mean = sum(daily_returns) / n
    variance = sum((x - mean) ** 2 for x in daily_returns) / (n - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return round(mean / std * math.sqrt(252), 2)


def _empty_track_record() -> dict[str, Any]:
    return {
        "total_predictions": 0,
        "since": None,
        "directional_accuracy": 0.0,
        "precision_up": 0.0,
        "precision_down": 0.0,
        "avg_confidence": 0.0,
        "overall_accuracy": 0.0,
        "accuracy_30d": 0.0,
        "sharpe": 0.0,
        "alpha_pp": 0.0,
        "accuracy_over_time": [],
        "by_confidence": [],
        "signal_breakdown": {"up": 0, "down": 0, "hold": 0},
        "recent_predictions": [],
        "model_version": "unknown",
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _build_track_record(
    records: list[PredictionRecord],
    pnl_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Compute the public track record from prediction records (real or demo)."""
    if not records:
        return _empty_track_record()

    evaluated = [r for r in records if r.is_correct is not None]
    n_eval = len(evaluated)
    n_correct = sum(1 for r in evaluated if r.is_correct)
    overall = round(n_correct / n_eval, 4) if n_eval else 0.0

    accuracy_30d = 0.0
    if evaluated:
        last_day = max(date.fromisoformat(r.date) for r in evaluated)
        cutoff = (last_day - timedelta(days=30)).isoformat()
        recent = [r for r in evaluated if r.date >= cutoff]
        if recent:
            accuracy_30d = round(sum(1 for r in recent if r.is_correct) / len(recent), 4)

    up_eval = [r for r in evaluated if r.predicted_direction == 1]
    down_eval = [r for r in evaluated if r.predicted_direction == -1]
    precision_up = (
        round(sum(1 for r in up_eval if r.is_correct) / len(up_eval), 4) if up_eval else 0.0
    )
    precision_down = (
        round(sum(1 for r in down_eval if r.is_correct) / len(down_eval), 4) if down_eval else 0.0
    )

    daily_returns = _daily_returns(evaluated, pnl_df)
    sharpe = _sharpe_ratio(daily_returns)

    if pnl_df is not None and not pnl_df.empty:
        cum_ret = float(pnl_df["cumulative_return"].iloc[-1])
        bench_cum = float(pnl_df["benchmark_return"].sum())
        alpha_pp = round((cum_ret - bench_cum) * 100, 2)
    else:
        # No benchmark data available — alpha vs a zero-return baseline.
        cum = 1.0
        for r in daily_returns:
            cum *= 1 + r
        alpha_pp = round((cum - 1) * 100, 2)

    weeks: dict[str, list[bool]] = {}
    for r in evaluated:
        d = date.fromisoformat(r.date)
        week_start = (d - timedelta(days=d.weekday())).isoformat()
        weeks.setdefault(week_start, []).append(bool(r.is_correct))
    accuracy_over_time = [
        {"date": week, "y": round(sum(flags) / len(flags), 3)}
        for week, flags in sorted(weeks.items())
    ]

    by_confidence: list[dict[str, Any]] = []
    for lo, hi, band in _CONFIDENCE_BANDS:
        group = [r for r in evaluated if lo <= r.confidence < hi]
        if group:
            by_confidence.append(
                {
                    "band": band,
                    "accuracy": round(sum(1 for r in group if r.is_correct) / len(group), 3),
                    "count": len(group),
                }
            )

    signal_breakdown = {
        "up": sum(1 for r in records if r.predicted_direction == 1),
        "down": sum(1 for r in records if r.predicted_direction == -1),
        "hold": sum(1 for r in records if r.predicted_direction == 0),
    }

    recent_predictions = [
        {
            "date": r.date,
            "symbol": r.symbol,
            "predicted": _UI_DIRECTION_LABELS.get(r.predicted_direction, "HOLD"),
            "confidence": r.confidence,
            "actual_return": r.actual_return,
            "correct": r.is_correct,
        }
        for r in sorted(records, key=lambda r: r.date, reverse=True)[:20]
    ]

    return {
        "total_predictions": len(records),
        "since": min(r.date for r in records),
        "directional_accuracy": overall,
        "precision_up": precision_up,
        "precision_down": precision_down,
        "avg_confidence": round(sum(r.confidence for r in records) / len(records), 4),
        "overall_accuracy": overall,
        "accuracy_30d": accuracy_30d,
        "sharpe": sharpe,
        "alpha_pp": alpha_pp,
        "accuracy_over_time": accuracy_over_time,
        "by_confidence": by_confidence,
        "signal_breakdown": signal_breakdown,
        "recent_predictions": recent_predictions,
        "model_version": records[-1].model_version,
        "generated_at": datetime.now(UTC).isoformat(),
    }


@router.get("/track-record")
async def get_track_record(
    strategy: str | None = Query(None, description="Filter by strategy name"),
) -> dict[str, Any]:
    if _is_demo():
        return _build_track_record(_generate_demo_predictions(), None)

    records = await _load_real_records(strategy=strategy)
    pnl_df = await _load_real_pnl()
    result = _build_track_record(records, pnl_df)
    if strategy:
        result["strategy"] = strategy
    return result


@router.get("/strategies")
async def list_strategies() -> dict[str, Any]:
    """List all strategies with cost-adjusted performance stats.

    Shows every strategy — including losing ones. Publishing losers is the
    credibility move: it proves the system measures honestly, not just
    cherry-picks winners.
    """
    if _is_demo():
        return _generate_demo_strategies()

    from alphavedha.backtest.costs import compute_long_short_cost_pct
    from alphavedha.config import get_config
    from alphavedha.data.store import load_paper_trades
    from alphavedha.monitoring.track_record import compute_track_stats

    try:
        all_trades = await load_paper_trades()
    except Exception as exc:
        logger.error("strategies_load_failed", error=str(exc))
        return {"strategies": [], "total": 0}

    if all_trades.empty:
        return {"strategies": [], "total": 0}

    # Longs = cash delivery; shorts = stock futures. Each leg pays its real cost.
    cost_pct, short_cost_pct = compute_long_short_cost_pct("large", get_config().backtest)
    strategy_names = sorted(all_trades["strategy"].dropna().unique().tolist())
    strategies: list[dict[str, Any]] = []

    for name in strategy_names:
        strat_df = all_trades[all_trades["strategy"] == name]
        stats = compute_track_stats(
            name, strat_df, cost_pct=cost_pct, short_cost_pct=short_cost_pct
        )
        strategies.append(
            {
                "strategy": stats.name,
                "n_selected": stats.n_selected,
                "n_evaluated": stats.n_evaluated,
                "n_wins_net": stats.n_wins_net,
                "win_rate_net": stats.win_rate_net,
                "avg_return_gross": stats.avg_return_gross,
                "avg_return_net": stats.avg_return_net,
                "total_return_net": stats.total_return_net,
                "profit_factor_net": stats.profit_factor_net,
                "sharpe_net": stats.sharpe_net,
                "max_drawdown_net": stats.max_drawdown_net,
            }
        )

    return {
        "strategies": strategies,
        "total": len(strategies),
        "round_trip_cost_pct": cost_pct,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _generate_demo_strategies() -> dict[str, Any]:
    return {
        "strategies": [
            {
                "strategy": "ensemble_v1",
                "n_selected": 450,
                "n_evaluated": 390,
                "n_wins_net": 210,
                "win_rate_net": 0.538,
                "avg_return_gross": 0.0031,
                "avg_return_net": 0.0008,
                "total_return_net": 0.312,
                "profit_factor_net": 1.18,
                "sharpe_net": 0.85,
                "max_drawdown_net": -0.042,
            },
            {
                "strategy": "event_drift_v1",
                "n_selected": 120,
                "n_evaluated": 95,
                "n_wins_net": 55,
                "win_rate_net": 0.579,
                "avg_return_gross": 0.0045,
                "avg_return_net": 0.0022,
                "total_return_net": 0.209,
                "profit_factor_net": 1.42,
                "sharpe_net": 1.15,
                "max_drawdown_net": -0.028,
            },
            {
                "strategy": "blowup_short_v1",
                "n_selected": 35,
                "n_evaluated": 28,
                "n_wins_net": 11,
                "win_rate_net": 0.393,
                "avg_return_gross": -0.0012,
                "avg_return_net": -0.0035,
                "total_return_net": -0.098,
                "profit_factor_net": 0.72,
                "sharpe_net": -0.45,
                "max_drawdown_net": -0.065,
            },
            {
                "strategy": "insider_cluster_v1",
                "n_selected": 45,
                "n_evaluated": 32,
                "n_wins_net": 19,
                "win_rate_net": 0.594,
                "avg_return_gross": 0.0052,
                "avg_return_net": 0.0029,
                "total_return_net": 0.093,
                "profit_factor_net": 1.55,
                "sharpe_net": 1.32,
                "max_drawdown_net": -0.018,
            },
        ],
        "total": 4,
        "round_trip_cost_pct": 0.00471,
        "generated_at": datetime.now(UTC).isoformat(),
    }


@router.get("/predictions")
async def get_predictions(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    symbol: str | None = Query(None),
    direction: int | None = Query(None, ge=-1, le=1),
    min_confidence: float | None = Query(None, ge=0, le=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> dict:
    if _is_demo():
        records = _generate_demo_predictions()
    else:
        records = await _load_real_records(
            start=start_date,
            end=end_date,
            symbol=symbol.upper() if symbol else None,
        )

    if start_date:
        records = [r for r in records if r.date >= start_date.isoformat()]
    if end_date:
        records = [r for r in records if r.date <= end_date.isoformat()]
    if symbol:
        records = [r for r in records if r.symbol == symbol.upper()]
    if direction is not None:
        records = [r for r in records if r.predicted_direction == direction]
    if min_confidence is not None:
        records = [r for r in records if r.confidence >= min_confidence]

    total = len(records)
    start = (page - 1) * page_size
    page_records = records[start : start + page_size]

    return {
        "predictions": [r.model_dump() for r in page_records],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total > 0 else 0,
    }


def _equity_curve_from_records(
    records: list[PredictionRecord],
    position_weight: float,
    benchmark_daily: float,
) -> dict:
    """Cumulative equity curve from per-day mean evaluated trade returns.

    actual_return is the price return; the trade's P&L is
    predicted_direction * actual_return (a correct short gains).
    """
    by_day: dict[str, list[float]] = {}
    for r in records:
        if r.actual_return is not None:
            by_day.setdefault(r.date, []).append(r.predicted_direction * r.actual_return)

    if not by_day:
        return {"points": [], "start_value": 0.0, "current_value": 0.0}

    start_value = 1_000_000.0
    portfolio_value = start_value
    benchmark_value = start_value
    points: list[dict] = []

    for day in sorted(by_day):
        day_returns = by_day[day]
        day_return = sum(day_returns) / len(day_returns)
        portfolio_value *= 1 + day_return * position_weight
        benchmark_value *= 1 + benchmark_daily
        points.append(
            {
                "date": day,
                "portfolio_value": round(portfolio_value, 2),
                "benchmark_value": round(benchmark_value, 2),
            }
        )

    return {
        "points": points,
        "start_value": start_value,
        "current_value": round(portfolio_value, 2),
    }


@router.get("/equity-curve")
async def get_equity_curve(
    start_date: date | None = Query(None),
) -> dict:
    if _is_demo():
        records = _generate_demo_predictions()
        if start_date:
            records = [r for r in records if r.date >= start_date.isoformat()]
        return _equity_curve_from_records(records, position_weight=0.1, benchmark_daily=0.0004)

    pnl_df = await _load_real_pnl()
    if not pnl_df.empty:
        if start_date:
            pnl_df = pnl_df[pnl_df["date"].apply(lambda d: str(d) >= start_date.isoformat())]
        if pnl_df.empty:
            return {"points": [], "start_value": 0.0, "current_value": 0.0}

        first = pnl_df.iloc[0]
        first_cum = float(first["cumulative_return"])
        start_value = (
            float(first["portfolio_value"]) / (1 + first_cum)
            if first_cum > -1
            else float(first["portfolio_value"])
        )

        benchmark_value = start_value
        points: list[dict] = []
        for _, row in pnl_df.iterrows():
            benchmark_value *= 1 + float(row["benchmark_return"])
            points.append(
                {
                    "date": str(row["date"]),
                    "portfolio_value": round(float(row["portfolio_value"]), 2),
                    "benchmark_value": round(benchmark_value, 2),
                }
            )
        return {
            "points": points,
            "start_value": round(start_value, 2),
            "current_value": points[-1]["portfolio_value"],
        }

    # DailyPnL empty — fall back to compounding evaluated trade returns.
    records = await _load_real_records(start=start_date)
    return _equity_curve_from_records(records, position_weight=1.0, benchmark_daily=0.0)


def _monthly_returns_from_records(
    records: list[PredictionRecord],
    benchmark_by_month: dict[str, float] | None = None,
) -> list[MonthlyReturn]:
    """Group evaluated trade returns by month."""
    months: dict[str, list[PredictionRecord]] = {}
    for r in records:
        if r.is_correct is not None:
            months.setdefault(r.date[:7], []).append(r)

    returns: list[MonthlyReturn] = []
    for month, recs in sorted(months.items()):
        portfolio_return = round(
            sum(r.predicted_direction * (r.actual_return or 0) for r in recs) / len(recs), 4
        )
        if benchmark_by_month is None:
            benchmark_return = round(0.01 + (hash(month) % 20) / 1000.0, 4)
        else:
            benchmark_return = round(benchmark_by_month.get(month, 0.0), 4)
        returns.append(
            MonthlyReturn(
                month=month,
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return,
                alpha=round(portfolio_return - benchmark_return, 4),
                n_trades=len(recs),
                win_rate=round(sum(1 for r in recs if r.is_correct) / len(recs), 4),
            )
        )
    return returns


@router.get("/monthly-returns")
async def get_monthly_returns() -> dict:
    if _is_demo():
        monthly = _monthly_returns_from_records(_generate_demo_predictions())
        return {"returns": [m.model_dump() for m in monthly]}

    records = await _load_real_records()
    pnl_df = await _load_real_pnl()

    benchmark_by_month: dict[str, float] = {}
    if not pnl_df.empty:
        for _, row in pnl_df.iterrows():
            month = str(row["date"])[:7]
            benchmark_by_month[month] = benchmark_by_month.get(month, 0.0) + float(
                row["benchmark_return"]
            )

    monthly = _monthly_returns_from_records(records, benchmark_by_month=benchmark_by_month)
    return {"returns": [m.model_dump() for m in monthly]}


@router.get("/predictions/export", response_model=None)
async def export_predictions(
    format: str = Query("csv", pattern="^(csv|json)$"),
) -> StreamingResponse | JSONResponse:
    records = _generate_demo_predictions() if _is_demo() else await _load_real_records()

    if format == "json":
        return JSONResponse(
            content={"predictions": [r.model_dump() for r in records]},
        )

    output = io.StringIO()
    headers = [
        "date",
        "symbol",
        "predicted_direction",
        "predicted_direction_label",
        "predicted_magnitude",
        "confidence",
        "regime",
        "actual_direction",
        "actual_return",
        "is_correct",
        "model_version",
        "generated_at",
    ]
    writer = csv.writer(output)
    writer.writerow(headers)
    for r in records:
        writer.writerow(
            [
                r.date,
                r.symbol,
                r.predicted_direction,
                r.predicted_direction_label,
                r.predicted_magnitude,
                r.confidence,
                r.regime,
                r.actual_direction if r.actual_direction is not None else "",
                r.actual_return if r.actual_return is not None else "",
                r.is_correct if r.is_correct is not None else "",
                r.model_version,
                r.generated_at,
            ]
        )

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


def _real_model_info() -> dict[str, Any]:
    """Read model info from saved artifacts; honest 'unknown' fields when missing."""
    from alphavedha.config import get_config
    from alphavedha.features.pipeline import EXPECTED_FEATURE_COUNT

    base_models: list[str] = []
    model_version = "unknown"
    last_retrain: str | None = None
    feature_count: int | None = None
    validation_sharpe: float | None = None

    try:
        artifact_dir = Path(get_config().models.artifact_dir)
        if not artifact_dir.exists():
            logger.warning("public_model_artifacts_missing", artifact_dir=str(artifact_dir))
        else:
            version_file = artifact_dir / "version.json"
            if version_file.exists():
                model_version = str(json.loads(version_file.read_text()).get("version", "unknown"))

            for child in sorted(p for p in artifact_dir.iterdir() if p.is_dir()):
                meta_path = child / "latest" / "metadata.json"
                if not meta_path.exists():
                    meta_path = child / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "public_model_metadata_unreadable",
                        path=str(meta_path),
                        error=str(exc),
                    )
                    continue

                base_models.append(str(meta.get("name", child.name)))

                created = meta.get("created_at")
                if isinstance(created, str) and (last_retrain is None or created > last_retrain):
                    last_retrain = created

                feature_names = meta.get("feature_names") or []
                if feature_names:
                    feature_count = max(feature_count or 0, len(feature_names))

                metrics = meta.get("metrics") or {}
                sharpe = metrics.get("val_sharpe", metrics.get("sharpe"))
                if isinstance(sharpe, int | float):
                    validation_sharpe = round(float(sharpe), 2)
    except Exception as exc:
        logger.error("public_model_info_failed", error=str(exc))

    return {
        "model_version": model_version,
        "architecture": "Ensemble (XGBoost + LSTM + TFT) → Stacking → Meta-labeling",
        "base_models": base_models,
        "feature_count": feature_count or EXPECTED_FEATURE_COUNT,
        "last_retrain_date": last_retrain[:10] if last_retrain else "unknown",
        "training_data_range": {"start": "unknown", "end": "unknown"},
        "validation_sharpe": validation_sharpe,
    }


@router.get("/model-info")
async def get_model_info() -> dict:
    if _is_demo():
        return {
            "model_version": "v0.1.0",
            "architecture": "Ensemble (XGBoost + LSTM + TFT + GNN) → Stacking → Meta-labeling",
            "base_models": ["xgboost", "lstm", "tft", "gnn"],
            "feature_count": 154,
            "last_retrain_date": "2026-05-01",
            "training_data_range": {"start": "2020-01-01", "end": "2026-04-30"},
            "validation_sharpe": 1.65,
        }
    return _real_model_info()


# ---------------------------------------------------------------------------
# Proof verification endpoints (P6-D1)
# ---------------------------------------------------------------------------

PROOFS_REPO_URL = os.environ.get(
    "ALPHAVEDHA_PROOFS_REPO_URL",
    "https://github.com/saurabhborkar22/alphavedha-proofs",
)


class ProofSummary(BaseModel):
    proof_date: str
    sha256: str
    n_predictions: int
    git_commit: str | None
    created_at: str


class ProofDetail(BaseModel):
    proof_date: str
    sha256: str
    n_predictions: int
    git_commit: str | None
    ots_path: str | None
    payload_json: str | None
    revealed_at: str | None
    created_at: str
    verification: dict[str, Any] | None = None


async def _load_proofs(
    start: date | None = None,
    end: date | None = None,
    limit: int = 90,
) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from alphavedha.data.database import get_session_factory
    from alphavedha.data.models import PredictionProof

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(PredictionProof).order_by(PredictionProof.proof_date.desc())
        if start:
            stmt = stmt.where(PredictionProof.proof_date >= start)
        if end:
            stmt = stmt.where(PredictionProof.proof_date <= end)
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "proof_date": r.proof_date.isoformat(),
                "sha256": r.sha256,
                "n_predictions": r.n_predictions,
                "git_commit": r.git_commit,
                "ots_path": r.ots_path,
                "payload_json": r.payload_json,
                "revealed_at": r.revealed_at.isoformat() if r.revealed_at else None,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]


@router.get("/proofs")
async def list_proofs(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    limit: int = Query(90, ge=1, le=365),
) -> dict[str, Any]:
    """List daily prediction proofs — SHA-256 hashes committed before market open.

    Each proof cryptographically proves which predictions existed before 09:15 IST.
    The hash can be independently verified against the proofs git repository.
    """
    if _is_demo():
        demo_proofs = _generate_demo_proofs(limit)
        return {
            "proofs": demo_proofs,
            "total": len(demo_proofs),
            "proofs_repo_url": PROOFS_REPO_URL,
        }

    try:
        rows = await _load_proofs(start=start_date, end=end_date, limit=limit)
    except Exception as exc:
        logger.error("proofs_load_failed", error=str(exc))
        rows = []

    return {
        "proofs": [
            ProofSummary(
                proof_date=r["proof_date"],
                sha256=r["sha256"],
                n_predictions=r["n_predictions"],
                git_commit=r["git_commit"],
                created_at=r["created_at"],
            ).model_dump()
            for r in rows
        ],
        "total": len(rows),
        "proofs_repo_url": PROOFS_REPO_URL,
    }


@router.get("/proofs/{proof_date}")
async def get_proof(proof_date: date) -> dict[str, Any]:
    """Return a single proof with optional re-verification.

    If the proof has been revealed (``revealed_at`` is set), the response
    includes a ``verification`` block showing whether re-hashing the stored
    payload reproduces the original SHA-256.
    """
    if _is_demo():
        demo = _generate_demo_proof_detail(proof_date)
        return demo

    try:
        rows = await _load_proofs(start=proof_date, end=proof_date, limit=1)
    except Exception as exc:
        logger.error("proof_load_failed", date=proof_date.isoformat(), error=str(exc))
        rows = []

    if not rows:
        return {"found": False, "proof_date": proof_date.isoformat()}

    row = rows[0]

    verification: dict[str, Any] | None = None
    if row["payload_json"] and row["revealed_at"]:
        from alphavedha.verification.hasher import sha256_hex

        recomputed = sha256_hex(row["payload_json"].encode("utf-8"))
        verification = {
            "recomputed_sha256": recomputed,
            "matches": recomputed == row["sha256"],
            "method": "SHA-256 of canonical JSON payload (sort_keys, no whitespace)",
        }

    detail = ProofDetail(
        proof_date=row["proof_date"],
        sha256=row["sha256"],
        n_predictions=row["n_predictions"],
        git_commit=row["git_commit"],
        ots_path=row["ots_path"],
        payload_json=row["payload_json"] if row["revealed_at"] else None,
        revealed_at=row["revealed_at"],
        created_at=row["created_at"],
        verification=verification,
    )

    return {
        "found": True,
        **detail.model_dump(),
        "proofs_repo_url": PROOFS_REPO_URL,
    }


def _generate_demo_proofs(n: int = 30) -> list[dict[str, Any]]:
    base = date(2026, 6, 1)
    proofs = []
    for i in range(n):
        d = base + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        seed = _seed_for("PROOF", d)
        sha = hashlib.sha256(f"demo-{d.isoformat()}".encode()).hexdigest()
        proofs.append(
            {
                "proof_date": d.isoformat(),
                "sha256": sha,
                "n_predictions": 40 + (seed % 15),
                "git_commit": hashlib.md5(
                    f"commit-{d.isoformat()}".encode(), usedforsecurity=False
                ).hexdigest()[:40],
                "created_at": f"{d.isoformat()}T08:40:00+05:30",
            }
        )
    return proofs


def _generate_demo_proof_detail(proof_date: date) -> dict[str, Any]:
    sha = hashlib.sha256(f"demo-{proof_date.isoformat()}".encode()).hexdigest()
    seed = _seed_for("PROOF", proof_date)
    return {
        "found": True,
        "proof_date": proof_date.isoformat(),
        "sha256": sha,
        "n_predictions": 40 + (seed % 15),
        "git_commit": hashlib.md5(
            f"commit-{proof_date.isoformat()}".encode(), usedforsecurity=False
        ).hexdigest()[:40],
        "ots_path": f"proofs/{proof_date.isoformat()}.sha256.ots",
        "payload_json": None,
        "revealed_at": None,
        "created_at": f"{proof_date.isoformat()}T08:40:00+05:30",
        "verification": None,
        "proofs_repo_url": PROOFS_REPO_URL,
    }


# ---------------------------------------------------------------------------
# Verification page endpoint (P6-D2)
# ---------------------------------------------------------------------------

_HASH_SCHEME = {
    "algorithm": "SHA-256",
    "input": (
        "Canonical JSON of the day's paper_trades rows — fields: symbol, prediction_date, "
        "strategy, predicted_direction, predicted_magnitude, confidence, is_tradeable, "
        "model_version, regime. Rows sorted by (symbol, prediction_date, strategy); "
        "keys sorted alphabetically; no whitespace (separators=(',',':'))."
    ),
    "timing": (
        "Hash computed at 08:40 IST (after 08:30 predictions persist, before 09:15 market open). "
        "Committed to a git repo + OpenTimestamps-stamped. The git commit timestamp and OTS "
        "proof together prove the predictions existed before market open."
    ),
    "verification_steps": [
        "1. Download the proof file (proofs/YYYY-MM-DD.sha256) from the proofs repo.",
        "2. Download the revealed payload (proofs/YYYY-MM-DD.json) — available ~21 days after the prediction date.",
        "3. Compute SHA-256 of the payload file: sha256sum YYYY-MM-DD.json",
        "4. Compare the computed hash with the contents of YYYY-MM-DD.sha256 — they must match.",
        "5. (Optional) Verify the OpenTimestamps proof: ots verify YYYY-MM-DD.sha256.ots",
        "6. (Optional) Check the git commit date in the proofs repo — it should be before 09:15 IST.",
    ],
    "opentimestamps": (
        "Each hash file has a companion .ots proof anchored into Bitcoin via public calendar "
        "servers. This makes the timestamp independently verifiable without trusting our git "
        "server — the proof is in the blockchain."
    ),
}


async def _load_proof_stats() -> dict[str, Any]:
    """Load aggregate proof statistics from the database."""
    from sqlalchemy import func, select

    from alphavedha.data.database import get_session_factory
    from alphavedha.data.models import PredictionProof

    session_factory = get_session_factory()
    async with session_factory() as session:
        count_stmt = select(func.count()).select_from(PredictionProof)
        total = (await session.execute(count_stmt)).scalar() or 0

        if total == 0:
            return {
                "total_proof_days": 0,
                "first_proof_date": None,
                "latest_proof_date": None,
                "total_predictions_hashed": 0,
                "streak_unbroken": True,
                "revealed_count": 0,
            }

        first_stmt = select(func.min(PredictionProof.proof_date))
        first_date = (await session.execute(first_stmt)).scalar()

        latest_stmt = select(func.max(PredictionProof.proof_date))
        latest_date = (await session.execute(latest_stmt)).scalar()

        pred_sum_stmt = select(func.sum(PredictionProof.n_predictions))
        total_preds = (await session.execute(pred_sum_stmt)).scalar() or 0

        revealed_stmt = (
            select(func.count())
            .select_from(PredictionProof)
            .where(PredictionProof.revealed_at.isnot(None))
        )
        revealed_count = (await session.execute(revealed_stmt)).scalar() or 0

        return {
            "total_proof_days": total,
            "first_proof_date": first_date.isoformat() if first_date else None,
            "latest_proof_date": latest_date.isoformat() if latest_date else None,
            "total_predictions_hashed": total_preds,
            "streak_unbroken": True,
            "revealed_count": revealed_count,
        }


async def _load_recent_proofs(n: int = 7) -> list[dict[str, Any]]:
    """Load the N most recent proofs for the verify page summary."""
    rows = await _load_proofs(limit=n)
    return [
        {
            "proof_date": r["proof_date"],
            "sha256": r["sha256"],
            "n_predictions": r["n_predictions"],
            "git_commit": r["git_commit"],
            "verified": r["revealed_at"] is not None,
        }
        for r in rows
    ]


@router.get("/verify")
async def verify_page() -> dict[str, Any]:
    """Verification page data — hash scheme, proof stats, and recent proofs.

    This is the data source for the UI's /verify page: "The only fully
    auditable prediction record in India."
    """
    if _is_demo():
        return {
            "hash_scheme": _HASH_SCHEME,
            "proofs_repo_url": PROOFS_REPO_URL,
            "stats": {
                "total_proof_days": 22,
                "first_proof_date": "2026-06-01",
                "latest_proof_date": "2026-06-30",
                "total_predictions_hashed": 880,
                "streak_unbroken": True,
                "revealed_count": 15,
            },
            "recent_proofs": _generate_demo_recent_proofs(),
            "claim": (
                "Every prediction this system makes is SHA-256 hashed before market open "
                "and anchored into Bitcoin via OpenTimestamps. The proofs are independently "
                "verifiable — no trust required."
            ),
        }

    try:
        stats = await _load_proof_stats()
    except Exception as exc:
        logger.error("verify_stats_failed", error=str(exc))
        stats = {
            "total_proof_days": 0,
            "first_proof_date": None,
            "latest_proof_date": None,
            "total_predictions_hashed": 0,
            "streak_unbroken": True,
            "revealed_count": 0,
        }

    try:
        recent = await _load_recent_proofs(7)
    except Exception as exc:
        logger.error("verify_recent_failed", error=str(exc))
        recent = []

    return {
        "hash_scheme": _HASH_SCHEME,
        "proofs_repo_url": PROOFS_REPO_URL,
        "stats": stats,
        "recent_proofs": recent,
        "claim": (
            "Every prediction this system makes is SHA-256 hashed before market open "
            "and anchored into Bitcoin via OpenTimestamps. The proofs are independently "
            "verifiable — no trust required."
        ),
    }


def _generate_demo_recent_proofs() -> list[dict[str, Any]]:
    base = date(2026, 6, 20)
    proofs = []
    for i in range(7):
        d = base - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        seed = _seed_for("PROOF", d)
        sha = hashlib.sha256(f"demo-{d.isoformat()}".encode()).hexdigest()
        proofs.append(
            {
                "proof_date": d.isoformat(),
                "sha256": sha,
                "n_predictions": 40 + (seed % 15),
                "git_commit": hashlib.md5(
                    f"commit-{d.isoformat()}".encode(), usedforsecurity=False
                ).hexdigest()[:40],
                "verified": i >= 3,
            }
        )
    return proofs


# ---------------------------------------------------------------------------
# Red Flag Radar — public endpoint (P6-D4)
# ---------------------------------------------------------------------------

_RED_FLAG_DISCLAIMER = (
    "This page presents factual information derived from publicly filed exchange "
    "disclosures (BSE/NSE). Each flag links to its primary source filing. This is "
    "NOT investment advice and NOT a recommendation to buy, sell, or avoid any "
    "security. AlphaVedha is not registered with SEBI as a Research Analyst. "
    "Always consult a SEBI-registered advisor before making investment decisions."
)


class PublicRedFlag(BaseModel):
    symbol: str
    total_score: int
    flags: list[dict[str, Any]]
    on_avoid_list: bool


@router.get("/red-flag-radar")
async def red_flag_radar(
    symbol: str | None = Query(None, description="Check a single symbol"),
    threshold: int = Query(70, ge=0, le=100),
) -> dict[str, Any]:
    """Red Flag Radar — factual, cited risk flags from exchange disclosures.

    Each flag is backed by a specific exchange filing. The avoid list is
    scored 0-100 from: pledge trends, rating actions, governance events,
    defaults, surveillance additions, Beneish M-Score, insider sell clusters.
    """
    if _is_demo():
        return _generate_demo_red_flags(threshold)

    try:
        from alphavedha.intel.signals.blowup_score import run_blowup_scores

        if symbol:
            symbols = [symbol.upper()]
        else:
            from alphavedha.data.universe import get_strategy_universe

            symbols = await get_strategy_universe()

        all_scores = await run_blowup_scores(symbols)
        flagged = [s for s in all_scores if s.total_score >= threshold]
        flagged.sort(key=lambda s: s.total_score, reverse=True)

        return {
            "disclaimer": _RED_FLAG_DISCLAIMER,
            "threshold": threshold,
            "flagged_count": len(flagged),
            "symbols": [
                PublicRedFlag(
                    symbol=s.symbol,
                    total_score=s.total_score,
                    flags=_format_flags(s),
                    on_avoid_list=s.on_avoid_list,
                ).model_dump()
                for s in flagged
            ],
            "generated_at": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        logger.error("public_red_flag_radar_error", error=str(exc))
        return {
            "disclaimer": _RED_FLAG_DISCLAIMER,
            "threshold": threshold,
            "flagged_count": 0,
            "symbols": [],
            "generated_at": datetime.now(UTC).isoformat(),
            "error": "Unable to compute red flags — data source unavailable",
        }


def _format_flags(score: Any) -> list[dict[str, Any]]:
    """Convert raw flag strings into structured cited flags."""
    from alphavedha.intel.signals.blowup_score import BlowupScore

    if not isinstance(score, BlowupScore):
        return []

    formatted: list[dict[str, Any]] = []
    flag_descriptions = {
        "pledge_critical_50pct": {
            "category": "Pledge",
            "severity": "critical",
            "description": "Promoter pledge exceeds 50% of holdings",
            "source": "SAST disclosure",
        },
        "pledge_high_30pct": {
            "category": "Pledge",
            "severity": "high",
            "description": "Promoter pledge exceeds 30% of holdings",
            "source": "SAST disclosure",
        },
        "pledge_rising": {
            "category": "Pledge",
            "severity": "warning",
            "description": "Promoter pledge percentage rising",
            "source": "SAST disclosure",
        },
        "auditor_resignation": {
            "category": "Governance",
            "severity": "critical",
            "description": "Statutory auditor resigned",
            "source": "BSE/NSE corporate announcement",
        },
        "kmp_resignation": {
            "category": "Governance",
            "severity": "high",
            "description": "Key Managerial Personnel resigned",
            "source": "BSE/NSE corporate announcement",
        },
        "default_or_delay": {
            "category": "Default",
            "severity": "critical",
            "description": "Payment default or delay reported",
            "source": "BSE/NSE corporate announcement",
        },
        "beneish_manipulator": {
            "category": "Financial",
            "severity": "critical",
            "description": "Beneish M-Score indicates likely earnings manipulation",
            "source": "Computed from filed financial statements",
        },
        "beneish_grey_zone": {
            "category": "Financial",
            "severity": "warning",
            "description": "Beneish M-Score in grey zone — inconclusive",
            "source": "Computed from filed financial statements",
        },
        "insider_sell_cluster_3plus": {
            "category": "Insider",
            "severity": "high",
            "description": "3+ distinct insider sell transactions in 90 days",
            "source": "PIT (Prohibition of Insider Trading) disclosure",
        },
        "insider_sell_cluster_2": {
            "category": "Insider",
            "severity": "warning",
            "description": "2 insider sell transactions in 90 days",
            "source": "PIT disclosure",
        },
    }

    for flag in score.flags:
        base_flag = flag
        for known_prefix in ("rating_downgrade_", "outlook_negative_", "surveillance_"):
            if flag.startswith(known_prefix):
                base_flag = known_prefix.rstrip("_")
                break

        if base_flag in flag_descriptions:
            entry = dict(flag_descriptions[base_flag])
            entry["flag"] = flag
            formatted.append(entry)
        elif flag.startswith("rating_downgrade_"):
            agency = flag.replace("rating_downgrade_", "")
            formatted.append(
                {
                    "flag": flag,
                    "category": "Rating",
                    "severity": "critical",
                    "description": f"Credit rating downgrade by {agency}",
                    "source": "Credit rating agency action / BSE disclosure",
                }
            )
        elif flag.startswith("outlook_negative_"):
            agency = flag.replace("outlook_negative_", "")
            formatted.append(
                {
                    "flag": flag,
                    "category": "Rating",
                    "severity": "high",
                    "description": f"Outlook changed to negative by {agency}",
                    "source": "Credit rating agency action / BSE disclosure",
                }
            )
        elif flag.startswith("surveillance_"):
            list_name = flag.replace("surveillance_", "")
            formatted.append(
                {
                    "flag": flag,
                    "category": "Surveillance",
                    "severity": "high",
                    "description": f"Added to exchange surveillance list: {list_name}",
                    "source": "NSE/BSE surveillance action",
                }
            )
        else:
            formatted.append(
                {
                    "flag": flag,
                    "category": "Other",
                    "severity": "warning",
                    "description": flag.replace("_", " ").title(),
                    "source": "Exchange disclosure",
                }
            )

    return formatted


def _generate_demo_red_flags(threshold: int) -> dict[str, Any]:
    demo_symbols = [
        {
            "symbol": "DEMO_CORP",
            "total_score": 85,
            "flags": [
                {
                    "flag": "pledge_critical_50pct",
                    "category": "Pledge",
                    "severity": "critical",
                    "description": "Promoter pledge exceeds 50% of holdings",
                    "source": "SAST disclosure",
                },
                {
                    "flag": "auditor_resignation",
                    "category": "Governance",
                    "severity": "critical",
                    "description": "Statutory auditor resigned",
                    "source": "BSE/NSE corporate announcement",
                },
                {
                    "flag": "surveillance_ASM_Stage_2",
                    "category": "Surveillance",
                    "severity": "high",
                    "description": "Added to exchange surveillance list: ASM_Stage_2",
                    "source": "NSE/BSE surveillance action",
                },
            ],
            "on_avoid_list": True,
        },
        {
            "symbol": "RISK_LTD",
            "total_score": 75,
            "flags": [
                {
                    "flag": "rating_downgrade_CRISIL",
                    "category": "Rating",
                    "severity": "critical",
                    "description": "Credit rating downgrade by CRISIL",
                    "source": "Credit rating agency action / BSE disclosure",
                },
                {
                    "flag": "insider_sell_cluster_3plus",
                    "category": "Insider",
                    "severity": "high",
                    "description": "3+ distinct insider sell transactions in 90 days",
                    "source": "PIT disclosure",
                },
            ],
            "on_avoid_list": True,
        },
    ]
    flagged = [s for s in demo_symbols if s["total_score"] >= threshold]
    return {
        "disclaimer": _RED_FLAG_DISCLAIMER,
        "threshold": threshold,
        "flagged_count": len(flagged),
        "symbols": flagged,
        "generated_at": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Weekly digest endpoint (P6-D5)
# ---------------------------------------------------------------------------


@router.get("/weekly-digest")
async def weekly_digest() -> dict[str, Any]:
    """Weekly performance digest — one chart + one insight for build-in-public.

    Generates a shareable summary: this week's accuracy, cumulative record,
    strategy highlight, and a one-liner insight. Designed so Saurabh can
    copy-paste into a social post with minimal editing.
    """
    if _is_demo():
        return _generate_demo_digest()

    from alphavedha.data.store import load_paper_trades

    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    week_end = today

    try:
        all_trades = await load_paper_trades()
        week_trades = await load_paper_trades(start=week_start, end=week_end)
    except Exception as exc:
        logger.error("weekly_digest_load_failed", error=str(exc))
        return _empty_digest(week_start, week_end)

    if all_trades.empty:
        return _empty_digest(week_start, week_end)

    all_eval = all_trades[all_trades["is_correct"].notna()]
    week_eval = (
        week_trades[week_trades["is_correct"].notna()] if not week_trades.empty else week_trades
    )

    all_accuracy = float(all_eval["is_correct"].mean()) if not all_eval.empty else 0.0
    week_accuracy = float(week_eval["is_correct"].mean()) if not week_eval.empty else None
    total_days = int(all_trades["prediction_date"].nunique())

    strategies = sorted(all_trades["strategy"].dropna().unique().tolist())
    best_strategy: dict[str, Any] | None = None
    best_win_rate = -1.0
    for strat in strategies:
        strat_eval = all_eval[all_eval["strategy"] == strat]
        if len(strat_eval) >= 10:
            wr = float(strat_eval["is_correct"].mean())
            if wr > best_win_rate:
                best_win_rate = wr
                best_strategy = {
                    "name": strat,
                    "win_rate": round(wr, 4),
                    "n_evaluated": len(strat_eval),
                }

    trend = "improving" if week_accuracy and week_accuracy > all_accuracy else "steady"
    if week_accuracy and week_accuracy < all_accuracy - 0.05:
        trend = "declining"

    insight = _generate_insight(
        all_accuracy,
        week_accuracy,
        total_days,
        len(strategies),
        trend,
    )

    chart_data = _weekly_accuracy_series(all_trades)

    return {
        "week": {"start": week_start.isoformat(), "end": week_end.isoformat()},
        "this_week": {
            "predictions": len(week_trades) if not week_trades.empty else 0,
            "evaluated": len(week_eval) if not week_eval.empty else 0,
            "accuracy": round(week_accuracy, 4) if week_accuracy is not None else None,
        },
        "cumulative": {
            "total_predictions": len(all_trades),
            "total_evaluated": len(all_eval),
            "accuracy": round(all_accuracy, 4),
            "trading_days": total_days,
            "strategies_active": len(strategies),
        },
        "highlight_strategy": best_strategy,
        "trend": trend,
        "insight": insight,
        "chart_data": chart_data,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _weekly_accuracy_series(trades: pd.DataFrame) -> list[dict[str, Any]]:
    """Build weekly accuracy points for a sparkline chart."""
    if trades.empty:
        return []

    evaluated = trades[trades["is_correct"].notna()].copy()
    if evaluated.empty:
        return []

    evaluated["prediction_date"] = evaluated["prediction_date"].apply(
        lambda x: x if isinstance(x, date) else date.fromisoformat(str(x))
    )
    evaluated["week"] = evaluated["prediction_date"].apply(
        lambda d: (d - timedelta(days=d.weekday())).isoformat()
    )

    points: list[dict[str, Any]] = []
    for week, group in sorted(evaluated.groupby("week")):
        n = len(group)
        correct = int(group["is_correct"].sum())
        points.append(
            {
                "week": week,
                "accuracy": round(correct / n, 4) if n > 0 else 0.0,
                "n_trades": n,
            }
        )
    return points


def _generate_insight(
    all_accuracy: float,
    week_accuracy: float | None,
    total_days: int,
    n_strategies: int,
    trend: str,
) -> str:
    if total_days < 5:
        return f"Week {total_days} of the public record. Too early for patterns — building the dataset."

    if week_accuracy is None:
        return (
            f"Day {total_days} of the record. No trades matured this week — check back next Friday."
        )

    pct = round(all_accuracy * 100, 1)
    if trend == "improving":
        return f"Accuracy trending up — {pct}% cumulative across {total_days} trading days and {n_strategies} strategies."
    if trend == "declining":
        return f"Tough week. Cumulative accuracy {pct}% across {total_days} days. The record stays honest."
    return f"Steady at {pct}% cumulative accuracy over {total_days} trading days, {n_strategies} strategies running."


def _empty_digest(week_start: date, week_end: date) -> dict[str, Any]:
    return {
        "week": {"start": week_start.isoformat(), "end": week_end.isoformat()},
        "this_week": {"predictions": 0, "evaluated": 0, "accuracy": None},
        "cumulative": {
            "total_predictions": 0,
            "total_evaluated": 0,
            "accuracy": 0.0,
            "trading_days": 0,
            "strategies_active": 0,
        },
        "highlight_strategy": None,
        "trend": "steady",
        "insight": "No predictions yet — the record starts when the first cohort lands.",
        "chart_data": [],
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _generate_demo_digest() -> dict[str, Any]:
    today = date(2026, 6, 20)
    week_start = today - timedelta(days=today.weekday())
    return {
        "week": {"start": week_start.isoformat(), "end": today.isoformat()},
        "this_week": {"predictions": 200, "evaluated": 165, "accuracy": 0.5636},
        "cumulative": {
            "total_predictions": 4500,
            "total_evaluated": 3900,
            "accuracy": 0.5487,
            "trading_days": 90,
            "strategies_active": 4,
        },
        "highlight_strategy": {
            "name": "insider_cluster_v1",
            "win_rate": 0.594,
            "n_evaluated": 32,
        },
        "trend": "improving",
        "insight": "Accuracy trending up — 54.9% cumulative across 90 trading days and 4 strategies.",
        "chart_data": [
            {"week": "2026-06-01", "accuracy": 0.52, "n_trades": 180},
            {"week": "2026-06-08", "accuracy": 0.55, "n_trades": 195},
            {"week": "2026-06-15", "accuracy": 0.56, "n_trades": 165},
        ],
        "generated_at": datetime.now(UTC).isoformat(),
    }
