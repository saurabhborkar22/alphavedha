"""Unit tests for universe manager — index composition fetching and tracking."""

from __future__ import annotations

import pytest

from alphavedha.data.universe import INDEX_URLS, fetch_index_constituents


class TestIndexURLs:
    def test_all_tiers_have_urls(self):
        assert "NIFTY 50" in INDEX_URLS
        assert "NIFTY MIDCAP 150" in INDEX_URLS
        assert "NIFTY SMALLCAP 250" in INDEX_URLS

    def test_urls_are_csv(self):
        for url in INDEX_URLS.values():
            assert url.endswith(".csv")


class TestFetchConstituents:
    @pytest.mark.asyncio
    async def test_unknown_index_raises(self):
        with pytest.raises(ValueError, match="Unknown index"):
            await fetch_index_constituents("INVALID_INDEX")


class TestConfig:
    def test_config_loads(self):
        from alphavedha.config import get_config

        cfg = get_config()
        assert cfg.universe.available_tiers["large"].index == "NIFTY 50"
        assert cfg.universe.available_tiers["large"].count == 50
        assert cfg.universe.available_tiers["mid"].count == 150

    def test_config_preprocessing(self):
        from alphavedha.config import get_config

        cfg = get_config()
        assert cfg.preprocessing.fractional_diff.max_lags == 100
        assert cfg.preprocessing.circuit.thresholds == [0.05, 0.10, 0.20]

    def test_config_models(self):
        from alphavedha.config import get_config

        cfg = get_config()
        assert cfg.models.xgboost.params.learning_rate == 0.05
        assert cfg.models.lstm.hidden_size == 64
        assert cfg.models.tft.horizons == [7, 15, 30]

    def test_config_risk(self):
        from alphavedha.config import get_config

        cfg = get_config()
        assert cfg.risk.position_sizing.max_single_stock_pct == 10.0
        assert cfg.risk.circuit_breaker.level_3_drawdown == 20.0


class TestGetStrategyUniverse:
    """Strategies must scan live index membership, not a stale hardcoded list."""

    @pytest.mark.asyncio
    async def test_uses_db_constituents(self) -> None:
        from unittest.mock import AsyncMock, patch

        from alphavedha.data.universe import get_strategy_universe

        with patch(
            "alphavedha.data.universe.get_symbols_for_tier",
            new_callable=AsyncMock,
            return_value=["tcs ", "TMPV", "RELIANCE"],
        ):
            symbols = await get_strategy_universe()

        assert symbols == ["TCS", "TMPV", "RELIANCE"]

    @pytest.mark.asyncio
    async def test_falls_back_to_hardcoded_on_empty(self) -> None:
        from unittest.mock import AsyncMock, patch

        from alphavedha.data.universe import get_strategy_universe

        with patch(
            "alphavedha.data.universe.get_symbols_for_tier",
            new_callable=AsyncMock,
            return_value=[],
        ):
            symbols = await get_strategy_universe()

        assert len(symbols) >= 20
        assert "TCS" in symbols

    @pytest.mark.asyncio
    async def test_falls_back_on_error(self) -> None:
        from unittest.mock import AsyncMock, patch

        from alphavedha.data.universe import get_strategy_universe

        with patch(
            "alphavedha.data.universe.get_symbols_for_tier",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ):
            symbols = await get_strategy_universe()

        assert len(symbols) >= 20
