"""Tests for data ingestion orchestrator."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from alphavedha.data.ingestion import (
    DerivativesResult,
    EarningsIngestionResult,
    FIIDIIResult,
    IngestionResult,
    ingest_symbol,
    ingest_universe,
)


@pytest.fixture
def mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.fetch_ohlcv = AsyncMock(
        return_value=pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [98.0],
                "close": [103.0],
                "volume": [1000000],
            }
        )
    )
    return provider


class TestIngestionResult:
    def test_defaults(self) -> None:
        r = IngestionResult()
        assert r.symbols_requested == 0
        assert r.symbols_succeeded == 0
        assert r.symbols_failed == 0
        assert r.total_rows_stored == 0
        assert r.failed_symbols == []
        assert r.errors == {}


class TestFIIDIIResult:
    def test_defaults(self) -> None:
        r = FIIDIIResult()
        assert r.rows_fetched == 0
        assert r.rows_stored == 0
        assert r.error is None


class TestDerivativesResult:
    def test_defaults(self) -> None:
        r = DerivativesResult()
        assert r.symbols_requested == 0
        assert r.symbols_succeeded == 0
        assert r.rows_stored == 0


class TestEarningsIngestionResult:
    def test_defaults(self) -> None:
        r = EarningsIngestionResult()
        assert r.symbols_requested == 0
        assert r.total_quarters == 0


class TestIngestSymbol:
    @pytest.mark.asyncio
    async def test_empty_data_returns_zero(self) -> None:
        provider = MagicMock()
        provider.fetch_ohlcv = AsyncMock(return_value=pd.DataFrame())
        result = await ingest_symbol("TCS", date(2024, 1, 1), date(2024, 1, 31), provider)
        assert result == 0

    @pytest.mark.asyncio
    async def test_successful_ingestion(self, mock_provider: MagicMock) -> None:
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.df = pd.DataFrame({"close": [103.0]})
        mock_pipeline_result.circuit_hits = 0

        with (
            patch("alphavedha.data.ingestion.run_pipeline", return_value=mock_pipeline_result),
            patch("alphavedha.data.ingestion.store_ohlcv", new_callable=AsyncMock, return_value=1),
        ):
            result = await ingest_symbol("TCS", date(2024, 1, 1), date(2024, 1, 31), mock_provider)
            assert result == 1


class TestIngestUniverse:
    @pytest.mark.asyncio
    async def test_empty_universe(self) -> None:
        with patch(
            "alphavedha.data.ingestion.get_symbols_for_tier",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await ingest_universe("large", date(2024, 1, 1), date(2024, 1, 31))
            assert isinstance(result, IngestionResult)
            assert result.symbols_requested == 0

    @pytest.mark.asyncio
    async def test_tracks_failures(self) -> None:
        with (
            patch(
                "alphavedha.data.ingestion.get_symbols_for_tier",
                new_callable=AsyncMock,
                return_value=["TCS", "INFY"],
            ),
            patch(
                "alphavedha.data.ingestion.ingest_symbol",
                new_callable=AsyncMock,
                side_effect=Exception("provider down"),
            ),
        ):
            result = await ingest_universe("large", date(2024, 1, 1), date(2024, 1, 31))
            assert result.symbols_failed == 2
            assert "TCS" in result.failed_symbols
            assert "INFY" in result.failed_symbols
