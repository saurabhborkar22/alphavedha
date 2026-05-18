"""PredictionService — central orchestrator shared by API and CLI."""

from __future__ import annotations

import structlog

from alphavedha.config import AppConfig
from alphavedha.prediction.engine import StockPrediction
from alphavedha.prediction.ranker import RankingResult, StockRanker
from alphavedha.services.cache import PredictionCache
from alphavedha.services.model_registry import ModelRegistry

logger = structlog.get_logger(__name__)


class PredictionService:
    """Orchestrates prediction pipeline for both API and CLI."""

    def __init__(
        self,
        registry: ModelRegistry,
        cache: PredictionCache,
        config: AppConfig,
    ) -> None:
        self._registry = registry
        self._cache = cache
        self._config = config
        self._engine = registry.get_prediction_engine()
        self._ranker = StockRanker()

    async def predict_single(self, symbol: str, sector: str = "") -> StockPrediction:
        """Predict a single stock, checking cache first.

        Args:
            symbol: Stock symbol (e.g. "TCS").
            sector: Optional sector name for risk constraints.

        Returns:
            StockPrediction with direction, score, price targets, etc.
        """
        cache_key = f"predict:{symbol}:{self._registry.model_version}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug("cache_hit", symbol=symbol)
            return cached

        features = self._registry.get_demo_features(symbol)
        prediction = self._engine.predict(symbol, features, sector=sector)

        await self._cache.set(cache_key, prediction)
        logger.info("prediction_generated", symbol=symbol, direction=prediction.direction)
        return prediction

    async def scan_tier(self, tier: str, top_n: int = 10) -> RankingResult:
        """Scan all demo symbols and rank them into buy/sell candidates.

        Args:
            tier: Universe tier name (e.g. "large", "mid").
            top_n: Maximum number of buy/sell candidates to return.

        Returns:
            RankingResult with buy_candidates, sell_candidates, and excluded.
        """
        symbols = self._registry.get_demo_symbols()
        logger.info("scan_started", tier=tier, symbols=len(symbols))

        predictions: list[StockPrediction] = []
        for symbol in symbols:
            pred = await self.predict_single(symbol)
            predictions.append(pred)

        return self._ranker.rank(predictions, top_n=top_n)

    async def predict_batch(self, symbols: list[str]) -> list[StockPrediction]:
        """Predict multiple symbols, preserving input order.

        Args:
            symbols: List of stock symbols.

        Returns:
            List of StockPrediction in the same order as input.
        """
        results: list[StockPrediction] = []
        for symbol in symbols:
            pred = await self.predict_single(symbol)
            results.append(pred)
        return results
