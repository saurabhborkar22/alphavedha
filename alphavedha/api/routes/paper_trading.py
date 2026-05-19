"""Paper trading API — record predictions, track P&L, verify track record.

Predictions are timestamped before market open (9:15 AM IST).
After market close, outcomes are recorded and P&L updated.
"""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/paper", tags=["paper-trading"])


class PaperTradeRequest(BaseModel):
    symbol: str
    predicted_direction: int = Field(..., ge=-1, le=1)
    predicted_magnitude: float
    confidence: float = Field(..., ge=0, le=1)
    model_version: str
    regime: str | None = None
    entry_price: float | None = None


class PaperTradeResponse(BaseModel):
    symbol: str
    prediction_date: str
    predicted_direction: int
    confidence: float
    model_version: str
    created_at: str


class TradeOutcomeRequest(BaseModel):
    symbol: str
    prediction_date: str
    exit_price: float
    actual_return: float
    is_correct: bool


class DashboardSummary(BaseModel):
    total_predictions: int
    correct_predictions: int
    accuracy_7d: float | None
    accuracy_30d: float | None
    accuracy_all: float | None
    total_return: float
    sharpe_ratio: float | None
    max_drawdown: float
    days_tracked: int


class PredictionRecord(BaseModel):
    symbol: str
    prediction_date: str
    predicted_direction: int
    predicted_magnitude: float
    confidence: float
    model_version: str
    regime: str | None
    entry_price: float | None
    exit_price: float | None
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
        "predicted_direction": req.predicted_direction,
        "predicted_magnitude": req.predicted_magnitude,
        "confidence": req.confidence,
        "model_version": req.model_version,
        "regime": req.regime,
        "entry_price": req.entry_price,
    }

    try:
        await store_paper_trade(row)
    except Exception as e:
        logger.error("paper_trade_store_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to store prediction")

    return PaperTradeResponse(
        symbol=req.symbol,
        prediction_date=today.isoformat(),
        predicted_direction=req.predicted_direction,
        confidence=req.confidence,
        model_version=req.model_version,
        created_at=datetime.now().isoformat(),
    )


@router.post("/outcome")
async def record_outcome(req: TradeOutcomeRequest) -> dict:
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
        )
    except Exception as e:
        logger.error("paper_outcome_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update outcome")

    return {"status": "updated", "symbol": req.symbol, "date": req.prediction_date}


@router.get("/dashboard", response_model=DashboardSummary)
async def get_dashboard() -> DashboardSummary:
    """Get paper trading dashboard summary."""
    from alphavedha.data.store import load_paper_trades

    import numpy as np

    trades_df = await load_paper_trades()

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

    returns = evaluated["actual_return"].dropna()
    total_ret = float(returns.sum()) if not returns.empty else 0.0

    if len(returns) >= 2 and returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    else:
        sharpe = None

    if not returns.empty:
        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    unique_dates = trades_df["prediction_date"].nunique()

    return DashboardSummary(
        total_predictions=total,
        correct_predictions=correct,
        accuracy_7d=acc_7d,
        accuracy_30d=acc_30d,
        accuracy_all=acc_all,
        total_return=total_ret,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        days_tracked=unique_dates,
    )


@router.get("/trades", response_model=list[PredictionRecord])
async def list_trades(
    symbol: str | None = None,
    limit: int = 100,
) -> list[PredictionRecord]:
    """List paper trade predictions with outcomes."""
    from alphavedha.data.store import load_paper_trades

    trades_df = await load_paper_trades(symbol=symbol)

    if trades_df.empty:
        return []

    trades_df = trades_df.tail(limit)

    return [
        PredictionRecord(
            symbol=row["symbol"],
            prediction_date=str(row["prediction_date"]),
            predicted_direction=row["predicted_direction"],
            predicted_magnitude=row["predicted_magnitude"],
            confidence=row["confidence"],
            model_version=row["model_version"],
            regime=row.get("regime"),
            entry_price=row.get("entry_price"),
            exit_price=row.get("exit_price"),
            actual_return=row.get("actual_return"),
            is_correct=row.get("is_correct"),
        )
        for _, row in trades_df.iterrows()
    ]
