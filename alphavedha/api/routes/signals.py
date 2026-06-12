"""Entry/exit signal endpoints — execution timing and buy/sell signal generation.

Combines ML prediction with the ExecutionEngine to answer:
  "Should I buy/sell this stock right now, and how do I execute the trade?"

Endpoints
---------
GET /signals/timing
    Current market timing quality and next optimal window.

GET /signals/execution/{symbol}
    Full execution plan (order type, tranches, slippage) for a symbol.

GET /signals/buy-sell/{symbol}
    Unified signal: ML direction + confidence + tradeable flag + execution plan.
    This is the primary endpoint for trading decision support.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from fastapi import APIRouter, Depends, Query

from alphavedha.api.deps import get_service, verify_api_key
from alphavedha.services.prediction_service import PredictionService
from alphavedha.signals.execution import ExecutionEngine

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/signals", tags=["signals"], dependencies=[Depends(verify_api_key)])

IST = ZoneInfo("Asia/Kolkata")
_engine = ExecutionEngine()

_DIRECTION_LABEL = {1: "BUY", -1: "SELL", 0: "HOLD"}
_CAP_TIER_BY_SYMBOL: dict[str, str] = {}


def _cap_tier(symbol: str) -> str:
    """Infer cap tier from symbol list (lazy lookup — defaults to 'large')."""
    s = symbol.upper()
    if s in _CAP_TIER_BY_SYMBOL:
        return _CAP_TIER_BY_SYMBOL[s]
    return "large"


@router.get("/timing")
async def market_timing() -> dict[str, Any]:
    """Return current market timing quality and the next optimal execution window."""
    now = datetime.now(IST)
    is_good, reason = _engine.is_good_time_to_trade(now)
    on_expiry = _engine.is_expiry_day(now)
    next_expiry = _engine.next_expiry(now)

    optimal = [
        {
            "start": str(w.start),
            "end": str(w.end),
            "quality": w.quality,
            "reason": w.reason,
        }
        for w in _engine.plan_execution(
            symbol="_dummy",
            cap_tier="large",
            avg_daily_volume=1_000_000,
            order_size_shares=100,
        ).recommended_windows
    ]

    return {
        "current_time_ist": now.isoformat(),
        "is_good_to_trade": is_good,
        "timing_quality_reason": reason,
        "is_expiry_day": on_expiry,
        "next_fo_expiry": next_expiry.date().isoformat() if next_expiry else None,
        "optimal_windows": optimal,
        "generated_at": now.isoformat(),
    }


@router.get("/execution/{symbol}")
async def execution_plan(
    symbol: str,
    cap_tier: str = Query(default="large", pattern="^(large|mid|small)$"),
    avg_daily_volume: float = Query(default=500_000.0, gt=0),
    order_size_shares: int = Query(default=100, gt=0),
    current_spread_pct: float = Query(default=0.001, gt=0),
) -> dict[str, Any]:
    """Generate an optimal execution plan (order type, tranches, slippage) for a trade."""
    symbol = symbol.upper().strip()
    now = datetime.now(IST)
    on_expiry = _engine.is_expiry_day(now)

    plan = _engine.plan_execution(
        symbol=symbol,
        cap_tier=cap_tier,
        avg_daily_volume=avg_daily_volume,
        order_size_shares=order_size_shares,
        current_spread_pct=current_spread_pct,
        is_expiry_day=on_expiry,
        current_time=now,
    )

    is_good_now, timing_reason = _engine.is_good_time_to_trade(now)

    return {
        "symbol": symbol,
        "cap_tier": plan.cap_tier,
        "order_type": plan.order_type,
        "n_tranches": plan.n_tranches,
        "tranche_interval_minutes": plan.tranche_interval_minutes,
        "estimated_slippage_pct": plan.estimated_slippage_pct,
        "is_good_time_now": is_good_now,
        "timing_reason": timing_reason,
        "is_expiry_day": on_expiry,
        "recommended_windows": [
            {
                "start": str(w.start),
                "end": str(w.end),
                "quality": w.quality,
                "reason": w.reason,
            }
            for w in plan.recommended_windows
        ],
        "warnings": plan.warnings,
        "generated_at": now.isoformat(),
    }


@router.get("/buy-sell/{symbol}")
async def buy_sell_signal(
    symbol: str,
    cap_tier: str = Query(default="large", pattern="^(large|mid|small)$"),
    avg_daily_volume: float = Query(default=500_000.0, gt=0),
    order_size_shares: int = Query(default=100, gt=0),
    service: PredictionService = Depends(get_service),
) -> dict[str, Any]:
    """Unified buy/sell signal combining ML prediction with execution timing.

    Returns a single, actionable object:
      - signal: BUY | SELL | HOLD
      - tradeable: whether the meta-labeling model considers this a valid trade
      - execute_now: whether the current time is within an optimal execution window
      - execution_plan: how to split and execute the order
    """
    symbol = symbol.upper().strip()
    now = datetime.now(IST)

    prediction = await service.predict_single(symbol)
    direction_label = _DIRECTION_LABEL.get(prediction.direction, "HOLD")

    on_expiry = _engine.is_expiry_day(now)
    plan = _engine.plan_execution(
        symbol=symbol,
        cap_tier=cap_tier,
        avg_daily_volume=avg_daily_volume,
        order_size_shares=order_size_shares,
        is_expiry_day=on_expiry,
        current_time=now,
    )
    is_good_now, timing_reason = _engine.is_good_time_to_trade(now)

    # Actionable only when: model says tradeable AND timing window is good
    execute_now = prediction.is_tradeable and is_good_now

    return {
        "symbol": symbol,
        "signal": direction_label,
        "direction": prediction.direction,
        "magnitude_pct": round(prediction.magnitude * 100, 2),
        "meta_confidence": round(prediction.meta_confidence, 4),
        "composite_score": round(prediction.composite_score, 2),
        "is_tradeable": prediction.is_tradeable,
        "execute_now": execute_now,
        "execute_now_reason": timing_reason
        if not is_good_now
        else "Within optimal execution window",
        "regime": prediction.regime,
        "price_targets": {
            "low": round(prediction.price_target_low, 2),
            "mid": round(prediction.price_target_mid, 2),
            "high": round(prediction.price_target_high, 2),
        },
        "execution_plan": {
            "order_type": plan.order_type,
            "n_tranches": plan.n_tranches,
            "tranche_interval_minutes": plan.tranche_interval_minutes,
            "estimated_slippage_pct": plan.estimated_slippage_pct,
            "is_expiry_day": on_expiry,
            "warnings": plan.warnings,
            "recommended_windows": [
                {"start": str(w.start), "end": str(w.end), "quality": w.quality}
                for w in plan.recommended_windows
            ],
        },
        "model_version": prediction.model_version,
        "generated_at": now.isoformat(),
    }
