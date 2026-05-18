"""Prediction, batch, and scan endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

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


@router.get("/predict/{symbol}")
async def predict_single(
    symbol: str,
    service: PredictionService = Depends(get_service),
) -> PredictionResponse:
    """Predict direction, magnitude, and price targets for a single stock."""
    prediction = await service.predict_single(symbol.upper())
    return PredictionResponse.from_stock_prediction(prediction)


@router.post("/predict/batch")
async def predict_batch(
    body: BatchRequest,
    service: PredictionService = Depends(get_service),
) -> BatchResponse:
    """Predict multiple stocks in one request (max 20)."""
    predictions: list[PredictionResponse] = []
    failed: list[dict[str, str]] = []
    for sym in body.symbols:
        try:
            pred = await service.predict_single(sym.upper())
            predictions.append(PredictionResponse.from_stock_prediction(pred))
        except Exception as e:
            failed.append({"symbol": sym, "error": str(e)})

    return BatchResponse(
        predictions=predictions,
        total=len(body.symbols),
        successful=len(predictions),
        failed=failed,
        model_version=predictions[0].model_version if predictions else "unknown",
    )


@router.get("/scan/{tier}")
async def scan_tier(
    tier: str,
    top_n: int = 10,
    service: PredictionService = Depends(get_service),
) -> ScanResponse:
    """Scan a universe tier and rank stocks into buy/sell candidates."""
    result = await service.scan_tier(tier, top_n=top_n)
    return ScanResponse(
        tier=tier,
        buy_candidates=[
            PredictionResponse.from_stock_prediction(p) for p in result.buy_candidates
        ],
        sell_candidates=[
            PredictionResponse.from_stock_prediction(p) for p in result.sell_candidates
        ],
        excluded=[
            ExcludedStock(symbol=sym, reason=reason) for sym, reason in result.excluded
        ],
        total_scanned=len(result.buy_candidates)
        + len(result.sell_candidates)
        + len(result.excluded),
        model_version=service._registry.model_version,
    )
