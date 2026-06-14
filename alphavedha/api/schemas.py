"""Pydantic response models for the AlphaVedha API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from alphavedha.prediction.engine import StockPrediction

_DIRECTION_LABELS = {1: "BUY", -1: "SELL", 0: "HOLD"}


class PriceTargets(BaseModel):
    low: float
    mid: float
    high: float


class RiskInfo(BaseModel):
    position_size_pct: float
    model_disagreement: float


class TradeSetup(BaseModel):
    entry_price: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None


class PredictionResponse(BaseModel):
    symbol: str
    direction: int
    direction_label: str
    magnitude: float
    composite_score: float
    meta_confidence: float
    is_tradeable: bool
    regime: str
    price_targets: PriceTargets
    risk: RiskInfo
    trade_setup: TradeSetup
    model_version: str
    generated_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_stock_prediction(cls, pred: StockPrediction) -> PredictionResponse:
        return cls(
            symbol=pred.symbol,
            direction=pred.direction,
            direction_label=_DIRECTION_LABELS.get(pred.direction, "UNKNOWN"),
            magnitude=pred.magnitude,
            composite_score=pred.composite_score,
            meta_confidence=pred.meta_confidence,
            is_tradeable=pred.is_tradeable,
            regime=pred.regime,
            price_targets=PriceTargets(
                low=pred.price_target_low,
                mid=pred.price_target_mid,
                high=pred.price_target_high,
            ),
            risk=RiskInfo(
                position_size_pct=pred.position_size_pct,
                model_disagreement=pred.model_disagreement,
            ),
            trade_setup=TradeSetup(
                entry_price=pred.entry_price,
                stop_loss_price=pred.stop_loss_price,
                take_profit_price=pred.take_profit_price,
            ),
            model_version=pred.model_version,
            generated_at=pred.timestamp,
            warnings=pred.warnings,
        )


class ExcludedStock(BaseModel):
    symbol: str
    reason: str


class ScanResponse(BaseModel):
    tier: str
    buy_candidates: list[PredictionResponse]
    sell_candidates: list[PredictionResponse]
    excluded: list[ExcludedStock]
    total_scanned: int
    model_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BatchRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, max_length=20)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    successful: int
    failed: list[dict[str, str]] = Field(default_factory=list)
    model_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail
