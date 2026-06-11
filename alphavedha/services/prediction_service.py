"""PredictionService — central orchestrator shared by API and CLI."""

from __future__ import annotations

import asyncio
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

    async def _get_features(self, symbol: str) -> pd.DataFrame:
        if self._registry.is_demo:
            return self._registry.get_demo_features(symbol)
        return await self._load_real_features(symbol)

    async def _load_real_features(self, symbol: str) -> pd.DataFrame:
        """Load the latest computed features from the feature store.

        Fetches the most recent row from the PostgreSQL feature store.
        Falls back to on-the-fly computation from OHLCV data if the
        feature store is empty for this symbol.
        """
        from alphavedha.data.store import load_features

        today = date.today()
        start = today - timedelta(days=7)

        try:
            features_df = await load_features(symbol, start, today)
        except Exception as e:
            logger.warning("feature_store_unavailable", symbol=symbol, error=str(e))
            features_df = pd.DataFrame()

        if features_df.empty:
            features_df = await self._compute_features_on_the_fly(symbol)

        if features_df.empty:
            raise ValueError(
                f"No features available for {symbol}. "
                "Run `alphavedha data refresh` to populate the feature store."
            )

        return features_df.iloc[[-1]]

    async def _compute_features_on_the_fly(self, symbol: str) -> pd.DataFrame:
        """Compute features from cached OHLCV data when feature store is empty."""
        from alphavedha.data.store import load_ohlcv
        from alphavedha.features.pipeline import compute_all_features

        today = date.today()
        start = today - timedelta(days=300)

        try:
            ohlcv_df = await load_ohlcv(symbol, start, today)
        except Exception as e:
            logger.warning("ohlcv_store_unavailable", symbol=symbol, error=str(e))
            return pd.DataFrame()

        if ohlcv_df.empty or len(ohlcv_df) < 60:
            logger.warning(
                "insufficient_ohlcv_for_features",
                symbol=symbol,
                rows=len(ohlcv_df),
            )
            return pd.DataFrame()

        try:
            # CPU-heavy (~2-4s) — run off the event loop
            result = await asyncio.to_thread(compute_all_features, symbol=symbol, ohlcv_df=ohlcv_df)
            return result.df
        except Exception as e:
            logger.warning("feature_computation_failed", symbol=symbol, error=str(e))
            return pd.DataFrame()

    async def _get_symbols(self, tier: str) -> list[str]:
        if self._registry.is_demo:
            return self._registry.get_demo_symbols()
        from alphavedha.data.universe import get_symbols_for_tier

        return await get_symbols_for_tier(tier)

    async def warm_up(self) -> None:
        """Run a single prediction to warm up the full inference path."""
        try:
            tiers = self._config.universe.default_tiers
            if not tiers:
                logger.warning("warmup_no_tiers")
                return
            symbols = await self._get_symbols(tiers[0])
            if not symbols:
                logger.warning("warmup_no_symbols")
                return
            await self.predict_single(symbols[0])
            logger.info("model_warmup_complete", symbol=symbols[0])
        except Exception as e:
            logger.warning("model_warmup_failed", error=str(e))

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

        features = await self._get_features(symbol)
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
        symbols = await self._get_symbols(tier)
        logger.info("scan_started", tier=tier, symbols=len(symbols))

        predictions = await self.predict_batch(symbols)
        return self._ranker.rank(predictions, top_n=top_n)

    async def predict_batch(self, symbols: list[str]) -> list[StockPrediction]:
        """Predict multiple symbols concurrently, preserving input order.

        Args:
            symbols: List of stock symbols.

        Returns:
            List of StockPrediction in the same order as input.
        """
        semaphore = asyncio.Semaphore(10)

        async def _predict_one(symbol: str) -> StockPrediction:
            async with semaphore:
                return await self.predict_single(symbol)

        return list(await asyncio.gather(*[_predict_one(s) for s in symbols]))
