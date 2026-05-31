from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from alphavedha.data.providers.trends_provider import (
    SECTOR_KEYWORDS,
    GoogleTrendsProvider,
)


def test_sector_keywords_has_5_sectors() -> None:
    assert set(SECTOR_KEYWORDS.keys()) == {"banking", "it", "pharma", "auto", "fmcg"}


def test_each_sector_has_keywords() -> None:
    for sector, keywords in SECTOR_KEYWORDS.items():
        assert len(keywords) >= 1, f"Sector {sector} has no keywords"


def test_symbol_to_sector_maps_known_symbols() -> None:
    provider = GoogleTrendsProvider()
    assert provider.symbol_to_sector("TCS.NS") == "it"
    assert provider.symbol_to_sector("SBIN.NS") == "banking"
    assert provider.symbol_to_sector("UNKNOWN.NS") is None


@pytest.mark.asyncio
async def test_fetch_sector_trends_returns_dataframe() -> None:
    provider = GoogleTrendsProvider()
    mock_df = pd.DataFrame({"TCS": [50, 60, 70]})

    with patch.object(provider, "_fetch_sync", return_value=mock_df):
        result = await provider.fetch_sector_trends("it")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_fetch_sector_trends_returns_empty_on_error() -> None:
    provider = GoogleTrendsProvider()

    with patch.object(provider, "_fetch_sync", side_effect=Exception("rate limit")):
        result = await provider.fetch_sector_trends("it")

    assert isinstance(result, pd.DataFrame)
    assert result.empty


@pytest.mark.asyncio
async def test_fetch_sector_trends_unknown_sector_returns_empty() -> None:
    provider = GoogleTrendsProvider()
    result = await provider.fetch_sector_trends("unknown_sector")
    assert result.empty


@pytest.mark.asyncio
async def test_fetch_all_sectors_returns_all_keys() -> None:
    provider = GoogleTrendsProvider()
    empty_df = pd.DataFrame()

    with patch.object(provider, "fetch_sector_trends", new=AsyncMock(return_value=empty_df)):
        result = await provider.fetch_all_sectors()

    assert set(result.keys()) == set(SECTOR_KEYWORDS.keys())
