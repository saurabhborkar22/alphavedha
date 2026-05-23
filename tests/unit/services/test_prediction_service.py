"""Tests for PredictionService — central prediction orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from alphavedha.prediction.engine import StockPrediction
from alphavedha.prediction.ranker import RankingResult
from alphavedha.services.prediction_service import PredictionService


def _make_mock_prediction(symbol: str = "TCS") -> StockPrediction:
    from datetime import UTC, datetime

    return StockPrediction(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        direction=1,
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
        model_version="demo-v0.1.0",
        warnings=[],
    )


@pytest.fixture
def service() -> PredictionService:
    from alphavedha.config import get_config
    from alphavedha.services.cache import PredictionCache
    from alphavedha.services.model_registry import ModelRegistry

    registry = ModelRegistry(demo=True)
    cache = PredictionCache(redis_client=None)
    return PredictionService(registry=registry, cache=cache, config=get_config())


class TestPredictionService:
    @pytest.mark.asyncio
    async def test_predict_single_returns_stock_prediction(
        self, service: PredictionService
    ) -> None:
        result = await service.predict_single("TCS")
        assert isinstance(result, StockPrediction)
        assert result.symbol == "TCS"
        assert result.direction in (-1, 0, 1)
        assert 0.0 <= result.composite_score <= 100.0

    @pytest.mark.asyncio
    async def test_predict_single_uses_cache(self) -> None:
        from alphavedha.config import get_config
        from alphavedha.services.cache import PredictionCache
        from alphavedha.services.model_registry import ModelRegistry

        registry = ModelRegistry(demo=True)
        cache = PredictionCache(redis_client=None)
        cache.get = AsyncMock(return_value=_make_mock_prediction("CACHED"))
        service = PredictionService(registry=registry, cache=cache, config=get_config())

        result = await service.predict_single("CACHED")
        assert result.symbol == "CACHED"
        cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_scan_tier_returns_ranking_result(self, service: PredictionService) -> None:
        result = await service.scan_tier("large", top_n=3)
        assert isinstance(result, RankingResult)
        assert len(result.buy_candidates) + len(result.sell_candidates) + len(result.excluded) > 0

    @pytest.mark.asyncio
    async def test_predict_batch_returns_list(self, service: PredictionService) -> None:
        results = await service.predict_batch(["TCS", "INFY"])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, StockPrediction) for r in results)

    @pytest.mark.asyncio
    async def test_predict_batch_preserves_order(self, service: PredictionService) -> None:
        results = await service.predict_batch(["INFY", "TCS"])
        assert results[0].symbol == "INFY"
        assert results[1].symbol == "TCS"


class TestWarmUp:
    @pytest.mark.asyncio
    async def test_warmup_runs_prediction(self, service: PredictionService) -> None:
        service.predict_single = AsyncMock(return_value=_make_mock_prediction())
        await service.warm_up()
        service.predict_single.assert_called_once()

    @pytest.mark.asyncio
    async def test_warmup_failure_does_not_raise(self, service: PredictionService) -> None:
        service.predict_single = AsyncMock(side_effect=RuntimeError("model not loaded"))
        await service.warm_up()


class TestBatchConcurrent:
    @pytest.mark.asyncio
    async def test_predict_batch_concurrent_all_symbols(self, service: PredictionService) -> None:
        service.predict_single = AsyncMock(return_value=_make_mock_prediction())
        results = await service.predict_batch(["TCS", "INFY", "RELIANCE"])
        assert len(results) == 3
        assert service.predict_single.call_count == 3

    @pytest.mark.asyncio
    async def test_predict_batch_concurrent_preserves_order(
        self, service: PredictionService
    ) -> None:
        async def _mock_predict(symbol: str, sector: str = "") -> StockPrediction:
            pred = _make_mock_prediction(symbol)
            return pred

        service.predict_single = AsyncMock(side_effect=_mock_predict)
        results = await service.predict_batch(["A", "B", "C"])
        assert [r.symbol for r in results] == ["A", "B", "C"]
