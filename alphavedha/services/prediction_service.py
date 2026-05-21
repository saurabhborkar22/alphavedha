"""PredictionService — central orchestrator shared by API and CLI."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
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

    def _get_features(self, symbol: str) -> pd.DataFrame:
        if self._registry.is_demo:
            return self._registry.get_demo_features(symbol)
        return self._load_real_features(symbol)

    def _load_real_features(self, symbol: str) -> pd.DataFrame:
        """Load the latest computed features from the feature store.

        Fetches the most recent row from the PostgreSQL feature store.
        Falls back to on-the-fly computation from OHLCV data if the
        feature store is empty for this symbol.
        """
        import asyncio

        from alphavedha.data.store import load_features

        today = date.today()
        start = today - timedelta(days=7)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    features_df = pool.submit(
                        asyncio.run, load_features(symbol, start, today)
                    ).result()
            else:
                features_df = asyncio.run(load_features(symbol, start, today))
        except Exception:
            logger.warning("feature_store_unavailable", symbol=symbol)
            features_df = pd.DataFrame()

        if features_df.empty:
            features_df = self._compute_features_on_the_fly(symbol)

        if features_df.empty:
            raise ValueError(
                f"No features available for {symbol}. "
                "Run `alphavedha data refresh` to populate the feature store."
            )

        return features_df.iloc[[-1]]

    def _compute_features_on_the_fly(self, symbol: str) -> pd.DataFrame:
        """Compute features from cached OHLCV data when feature store is empty."""
        import asyncio

        from alphavedha.data.store import load_ohlcv
        from alphavedha.features.pipeline import compute_all_features

        today = date.today()
        start = today - timedelta(days=300)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    ohlcv_df = pool.submit(asyncio.run, load_ohlcv(symbol, start, today)).result()
            else:
                ohlcv_df = asyncio.run(load_ohlcv(symbol, start, today))
        except Exception:
            logger.warning("ohlcv_store_unavailable", symbol=symbol)
            return pd.DataFrame()

        if ohlcv_df.empty or len(ohlcv_df) < 60:
            logger.warning(
                "insufficient_ohlcv_for_features",
                symbol=symbol,
                rows=len(ohlcv_df),
            )
            return pd.DataFrame()

        try:
            result = compute_all_features(symbol=symbol, ohlcv_df=ohlcv_df)
            return result.df
        except Exception as e:
            logger.warning("feature_computation_failed", symbol=symbol, error=str(e))
            return pd.DataFrame()

    def _get_symbols(self, tier: str) -> list[str]:
        if self._registry.is_demo:
            return self._registry.get_demo_symbols()
        from alphavedha.data.universe import get_symbols_for_tier

        return get_symbols_for_tier(tier)

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

        features = self._get_features(symbol)
        prediction = self._engine.predict(symbol, features, sector=sector)

        await self._cache.set(cache_key, prediction)
        logger.info("prediction_generated", symbol=symbol, direction=prediction.direction)
        return prediction

    async def scan_tier(self, tier: str, top_n: int = 10) -> RankingResult:
        """Scan symbols in a tier and rank them into buy/sell candidates.

        Args:
            tier: Universe tier name (e.g. "large", "mid").
            top_n: Maximum number of buy/sell candidates to return.

        Returns:
            RankingResult with buy_candidates, sell_candidates, and excluded.
        """
        symbols = self._get_symbols(tier)
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
