"""Unit tests for data providers — mocked API calls, validation logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphavedha.data.providers.base import (
    FetchResult,
    RateLimiter,
    validate_ohlcv,
)
from alphavedha.exceptions import DataProviderError


class TestValidateOHLCV:
    def test_valid_df_passes(self):
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [95.0],
                "Close": [102.0],
                "Adj Close": [102.0],
                "Volume": [1000000],
            }
        )
        result = validate_ohlcv(df, "TCS", "test")
        assert list(result.columns) == ["open", "high", "low", "close", "adj_close", "volume"]

    def test_missing_adj_close_uses_close(self):
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000000],
            }
        )
        result = validate_ohlcv(df, "TCS", "test")
        assert "adj_close" in result.columns
        assert result["adj_close"].iloc[0] == 102.0

    def test_negative_prices_dropped(self):
        df = pd.DataFrame(
            {
                "open": [100.0, -1.0],
                "high": [105.0, 105.0],
                "low": [95.0, 95.0],
                "close": [102.0, 102.0],
                "volume": [1000000, 1000000],
            }
        )
        result = validate_ohlcv(df, "TCS", "test")
        assert len(result) == 1

    def test_empty_df_returns_empty(self):
        result = validate_ohlcv(pd.DataFrame(), "TCS", "test")
        assert result.empty

    def test_missing_required_columns_raises(self):
        df = pd.DataFrame({"open": [100.0], "close": [102.0]})
        with pytest.raises(DataProviderError):
            validate_ohlcv(df, "TCS", "test")


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiter_spacing(self):
        limiter = RateLimiter(requests_per_second=100)
        import time

        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_no_limit(self):
        limiter = RateLimiter()
        await limiter.acquire()


class TestFetchResult:
    def test_rows_fetched_auto_computed(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = FetchResult(symbol="TCS", df=df, provider="test")
        assert result.rows_fetched == 3

    def test_empty_result(self):
        result = FetchResult(symbol="TCS", df=pd.DataFrame(), provider="test", had_errors=True)
        assert result.rows_fetched == 0
        assert result.had_errors


class TestYFinanceProvider:
    @pytest.mark.asyncio
    @patch("alphavedha.data.providers.yfinance_provider.yf")
    async def test_fetch_ohlcv_adds_ns_suffix(self, mock_yf: MagicMock):
        from datetime import date

        from alphavedha.data.providers.yfinance_provider import YFinanceProvider

        mock_ticker = MagicMock()
        dates = pd.bdate_range("2024-01-01", periods=5)
        mock_ticker.history.return_value = pd.DataFrame(
            {
                "Open": np.full(5, 100.0),
                "High": np.full(5, 105.0),
                "Low": np.full(5, 95.0),
                "Close": np.full(5, 102.0),
                "Adj Close": np.full(5, 102.0),
                "Volume": np.full(5, 1000000, dtype=int),
            },
            index=dates,
        )
        mock_yf.Ticker.return_value = mock_ticker

        provider = YFinanceProvider()
        df = await provider.fetch_ohlcv("TCS", date(2024, 1, 1), date(2024, 1, 10))

        mock_yf.Ticker.assert_called_once_with("TCS.NS")
        assert len(df) == 5

    @pytest.mark.asyncio
    @patch("alphavedha.data.providers.yfinance_provider.yf")
    async def test_already_has_ns_suffix(self, mock_yf: MagicMock):
        from datetime import date

        from alphavedha.data.providers.yfinance_provider import YFinanceProvider

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [95.0],
                "Close": [102.0],
                "Adj Close": [102.0],
                "Volume": [1000000],
            },
            index=pd.bdate_range("2024-01-01", periods=1),
        )
        mock_yf.Ticker.return_value = mock_ticker

        provider = YFinanceProvider()
        await provider.fetch_ohlcv("TCS.NS", date(2024, 1, 1), date(2024, 1, 2))

        mock_yf.Ticker.assert_called_once_with("TCS.NS")
