"""Tests for API response schema validation."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from alphavedha.api.schemas import (
    BatchResponse,
    ErrorDetail,
    ErrorResponse,
    PredictionResponse,
    ScanResponse,
)
from alphavedha.prediction.engine import StockPrediction


def _make_prediction(symbol: str = "TCS", direction: int = 1) -> StockPrediction:
    return StockPrediction(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        direction=direction,
        magnitude=0.03,
        composite_score=78.5,
        meta_confidence=0.72,
        is_tradeable=True,
        regime="bull",
        regime_probabilities=np.array([0.7, 0.1, 0.1, 0.1]),
        price_target_low=95.0,
        price_target_mid=100.0,
        price_target_high=105.0,
        model_disagreement=0.05,
        position_size_pct=5.0,
        model_version="v0.1.0",
        warnings=["test warning"],
    )


class TestPredictionResponse:
    def test_from_stock_prediction(self) -> None:
        pred = _make_prediction("TCS", direction=1)
        resp = PredictionResponse.from_stock_prediction(pred)
        assert resp.symbol == "TCS"
        assert resp.direction == 1
        assert resp.direction_label == "BUY"
        assert resp.composite_score == 78.5
        assert resp.price_targets.low == 95.0
        assert resp.risk.position_size_pct == 5.0
        assert resp.model_version == "v0.1.0"
        assert resp.warnings == ["test warning"]

    def test_direction_labels(self) -> None:
        buy = PredictionResponse.from_stock_prediction(_make_prediction(direction=1))
        sell = PredictionResponse.from_stock_prediction(_make_prediction(direction=-1))
        hold = PredictionResponse.from_stock_prediction(_make_prediction(direction=0))
        assert buy.direction_label == "BUY"
        assert sell.direction_label == "SELL"
        assert hold.direction_label == "HOLD"

    def test_json_serializable(self) -> None:
        pred = _make_prediction()
        resp = PredictionResponse.from_stock_prediction(pred)
        data = resp.model_dump(mode="json")
        assert isinstance(data["generated_at"], str)
        assert isinstance(data["price_targets"], dict)


class TestBatchResponse:
    def test_batch_response_structure(self) -> None:
        resp = BatchResponse(
            predictions=[
                PredictionResponse.from_stock_prediction(_make_prediction("TCS"))
            ],
            total=2,
            successful=1,
            failed=[{"symbol": "BAD", "error": "not found"}],
            model_version="v0.1.0",
        )
        assert resp.total == 2
        assert resp.successful == 1
        assert len(resp.failed) == 1


class TestScanResponse:
    def test_scan_response_structure(self) -> None:
        resp = ScanResponse(
            tier="large",
            buy_candidates=[
                PredictionResponse.from_stock_prediction(_make_prediction("TCS")),
            ],
            sell_candidates=[],
            excluded=[],
            total_scanned=1,
            model_version="v0.1.0",
        )
        assert resp.tier == "large"
        assert len(resp.buy_candidates) == 1
        assert resp.total_scanned == 1
        data = resp.model_dump(mode="json")
        assert isinstance(data["generated_at"], str)


class TestErrorResponse:
    def test_error_response_structure(self) -> None:
        resp = ErrorResponse(
            error=ErrorDetail(
                code="SYMBOL_NOT_FOUND",
                message="Symbol 'BAD' not found",
            )
        )
        data = resp.model_dump(mode="json")
        assert data["error"]["code"] == "SYMBOL_NOT_FOUND"
