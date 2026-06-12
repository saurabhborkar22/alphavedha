"""PredictionService — central orchestrator shared by API and CLI."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import numpy as np
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
        self._inflight_scans: dict[str, asyncio.Task[RankingResult]] = {}
        self._market_features_cache: tuple[date, pd.DataFrame | None] | None = None

    async def _get_features(self, symbol: str) -> pd.DataFrame:
        if self._registry.is_demo:
            return self._registry.get_demo_features(symbol)
        return await self._load_real_features(symbol)

    @property
    def _feature_window_rows(self) -> int:
        """Rows of feature history needed per prediction.

        LSTM/TFT need a full sequence window to emit a prediction for the
        latest row — anything shorter lands entirely in their warmup pad.
        """
        return max(
            self._config.models.lstm.sequence_length,
            self._config.models.tft.sequence_length,
        )

    async def _load_real_features(self, symbol: str) -> pd.DataFrame:
        """Load the latest window of computed features from the feature store.

        Fetches the most recent `_feature_window_rows` rows from the
        PostgreSQL feature store. Falls back to on-the-fly computation from
        OHLCV data if the feature store has too little history.
        """
        from alphavedha.data.store import load_features

        n_rows = self._feature_window_rows
        today = date.today()
        # Calendar buffer: ~250 trading days per 365 calendar days
        start = today - timedelta(days=max(2 * n_rows, 30))

        try:
            features_df = await load_features(symbol, start, today)
        except Exception as e:
            logger.warning("feature_store_unavailable", symbol=symbol, error=str(e))
            features_df = pd.DataFrame()

        if len(features_df) < n_rows:
            computed = await self._compute_features_on_the_fly(symbol)
            if len(computed) > len(features_df):
                features_df = computed

        if features_df.empty:
            raise ValueError(
                f"No features available for {symbol}. "
                "Run `alphavedha data refresh` to populate the feature store."
            )

        return features_df.tail(n_rows)

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
        except Exception as e:
            logger.warning("feature_computation_failed", symbol=symbol, error=str(e))
            return pd.DataFrame()

        await self._persist_features(symbol, result.df)
        return result.df

    async def _persist_features(self, symbol: str, features_df: pd.DataFrame) -> None:
        """Write computed features through to the feature store (best-effort)."""
        from alphavedha.data.store import store_features

        try:
            stored = await store_features(symbol, features_df)
            logger.debug("features_persisted", symbol=symbol, rows=stored)
        except Exception as e:
            logger.warning("feature_persist_failed", symbol=symbol, error=str(e))

    async def _get_last_close(self, symbol: str) -> float | None:
        """Latest close from the OHLCV store, for return→price conversion."""
        if self._registry.is_demo:
            return None
        from alphavedha.data.store import load_ohlcv

        today = date.today()
        try:
            ohlcv_df = await load_ohlcv(symbol, today - timedelta(days=10), today)
        except Exception as e:
            logger.warning("last_close_unavailable", symbol=symbol, error=str(e))
            return None
        if ohlcv_df.empty or "close" not in ohlcv_df.columns:
            return None
        return float(ohlcv_df["close"].iloc[-1])

    async def _get_market_features(self) -> pd.DataFrame | None:
        """Market-level returns/volatility for regime detection, cached per day.

        Reconstructs the same series the regime detector was trained on:
        equal-weight portfolio log returns of the default tier plus their
        20-day realized volatility (see training pipeline _get_regime_probs).
        """
        if self._registry.is_demo:
            return None
        today = date.today()
        if self._market_features_cache is not None and self._market_features_cache[0] == today:
            return self._market_features_cache[1]
        market_features = await self._build_market_features()
        self._market_features_cache = (today, market_features)
        return market_features

    async def _build_market_features(self) -> pd.DataFrame | None:
        from alphavedha.data.store import load_ohlcv

        tiers = self._config.universe.default_tiers
        if not tiers:
            return None
        try:
            symbols = await self._get_symbols(tiers[0])
        except Exception as e:
            logger.warning("market_features_symbols_failed", error=str(e))
            return None

        today = date.today()
        start = today - timedelta(days=400)
        all_returns: list[pd.Series] = []
        for symbol in symbols:
            try:
                ohlcv_df = await load_ohlcv(symbol, start, today)
            except Exception:
                continue
            if "close" in ohlcv_df.columns and len(ohlcv_df) > 50:
                returns = np.log(ohlcv_df["close"] / ohlcv_df["close"].shift(1)).dropna()
                all_returns.append(returns)

        if not all_returns:
            logger.warning("market_features_no_data")
            return None

        combined = pd.concat(all_returns, axis=1)
        portfolio_returns = combined.mean(axis=1).dropna()
        realized_vol = portfolio_returns.rolling(20).std().dropna()
        portfolio_returns = portfolio_returns.loc[realized_vol.index]
        if portfolio_returns.empty:
            return None

        logger.info("market_features_built", rows=len(portfolio_returns))
        return pd.DataFrame({"returns": portfolio_returns, "volatility": realized_vol})

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
        market_features = await self._get_market_features()
        last_close = await self._get_last_close(symbol)
        prediction = self._engine.predict(
            symbol,
            features,
            sector=sector,
            market_features=market_features,
            last_close=last_close,
        )

        await self._cache.set(cache_key, prediction)
        logger.info("prediction_generated", symbol=symbol, direction=prediction.direction)
        return prediction

    async def scan_tier(self, tier: str, top_n: int = 10) -> RankingResult:
        """Scan symbols in a tier and rank them into buy/sell candidates.

        Concurrent requests for the same tier share a single in-flight scan
        — a full-tier scan is expensive and UI retries would otherwise stack.

        Args:
            tier: Universe tier name (e.g. "large", "mid").
            top_n: Maximum number of buy/sell candidates to return.

        Returns:
            RankingResult with buy_candidates, sell_candidates, and excluded.
        """
        key = f"{tier}:{top_n}"
        task = self._inflight_scans.get(key)
        if task is None or task.done():
            task = asyncio.create_task(self._scan_tier_impl(tier, top_n))
            self._inflight_scans[key] = task
        return await task

    async def _scan_tier_impl(self, tier: str, top_n: int) -> RankingResult:
        predictions = await self.predict_tier(tier)
        return self._ranker.rank(predictions, top_n=top_n)

    async def predict_tier(self, tier: str) -> list[StockPrediction]:
        """Predict every symbol in a tier, without ranking or filtering.

        Args:
            tier: Universe tier name (e.g. "large", "mid").

        Returns:
            List of StockPrediction for all symbols in the tier.
        """
        symbols = await self._get_symbols(tier)
        logger.info("scan_started", tier=tier, symbols=len(symbols))
        return await self.predict_batch(symbols)

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
