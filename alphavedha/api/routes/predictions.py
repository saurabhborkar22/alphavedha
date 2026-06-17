"""Prediction, batch, and scan endpoints."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query

from alphavedha.api.deps import get_service, verify_api_key
from alphavedha.api.schemas import (
    BatchRequest,
    BatchResponse,
    ExcludedStock,
    PredictionResponse,
    ScanResponse,
)
from alphavedha.services.prediction_service import PredictionService

router = APIRouter(tags=["predictions"], dependencies=[Depends(verify_api_key)])

_SYMBOL_RE = re.compile(r"^[A-Z0-9&_.-]{1,20}$")
_VALID_TIERS = {"large", "mid", "small", "all"}


def _validate_symbol(symbol: str) -> str:
    s = symbol.upper().strip()
    if not _SYMBOL_RE.match(s):
        raise HTTPException(status_code=400, detail=f"Invalid symbol format: {symbol}")
    return s


@router.get("/predict/{symbol}")
async def predict_single(
    symbol: str,
    service: PredictionService = Depends(get_service),
) -> PredictionResponse:
    """Predict direction, magnitude, and price targets for a single stock."""
    symbol = _validate_symbol(symbol)
    prediction = await service.predict_single(symbol)
    return PredictionResponse.from_stock_prediction(prediction, is_demo=service._registry.is_demo)


@router.post("/predict/batch")
async def predict_batch(
    body: BatchRequest,
    service: PredictionService = Depends(get_service),
) -> BatchResponse:
    """Predict multiple stocks in one request (max 20)."""
    demo = service._registry.is_demo
    predictions: list[PredictionResponse] = []
    failed: list[dict[str, str]] = []
    for sym in body.symbols:
        try:
            validated = _validate_symbol(sym)
            pred = await service.predict_single(validated)
            predictions.append(PredictionResponse.from_stock_prediction(pred, is_demo=demo))
        except Exception as e:
            failed.append({"symbol": sym, "error": str(e)})

    return BatchResponse(
        predictions=predictions,
        total=len(body.symbols),
        successful=len(predictions),
        failed=failed,
        model_version=predictions[0].model_version if predictions else "unknown",
        is_demo=demo,
    )


@router.get("/scan/{tier}")
async def scan_tier(
    tier: str,
    top_n: int = Query(default=10, ge=1, le=50),
    service: PredictionService = Depends(get_service),
) -> ScanResponse:
    """Scan a universe tier and rank stocks into buy/sell candidates."""
    tier = tier.lower().strip()
    if tier not in _VALID_TIERS:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {tier}. Use: {_VALID_TIERS}")
    demo = service._registry.is_demo
    result = await service.scan_tier(tier, top_n=top_n)
    return ScanResponse(
        tier=tier,
        buy_candidates=[
            PredictionResponse.from_stock_prediction(p, is_demo=demo)
            for p in result.buy_candidates
        ],
        sell_candidates=[
            PredictionResponse.from_stock_prediction(p, is_demo=demo)
            for p in result.sell_candidates
        ],
        excluded=[ExcludedStock(symbol=sym, reason=reason) for sym, reason in result.excluded],
        total_scanned=len(result.buy_candidates)
        + len(result.sell_candidates)
        + len(result.excluded),
        model_version=service._registry.model_version,
        is_demo=demo,
    )
