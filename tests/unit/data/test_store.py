"""Tests for feature/OHLCV store — mock DB session for unit testing."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from alphavedha.data import store


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.fixture
def mock_session_factory(mock_session):
    factory = MagicMock(return_value=mock_session)
    with patch.object(store, "get_session_factory", return_value=factory):
        yield mock_session


class TestStoreFeatures:
    async def test_empty_dataframe_returns_zero(self) -> None:
        result = await store.store_features("TCS", pd.DataFrame())
        assert result == 0

    async def test_stores_rows(self, mock_session_factory) -> None:
        df = pd.DataFrame(
            {"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        result = await store.store_features("TCS", df, feature_version="v1")
        assert result == 2
        assert mock_session_factory.execute.await_count == 2
        mock_session_factory.commit.assert_awaited_once()


class TestLoadFeatures:
    async def test_empty_result_returns_empty_df(self, mock_session_factory) -> None:
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session_factory.execute.return_value = mock_result

        df = await store.load_features("TCS", date(2024, 1, 1), date(2024, 1, 31))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    async def test_returns_dataframe_with_date_index(self, mock_session_factory) -> None:
        row = MagicMock()
        row.date = date(2024, 1, 1)
        row.feature_json = {"feat_a": 1.0, "feat_b": 2.0}
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [row]
        mock_session_factory.execute.return_value = mock_result

        df = await store.load_features("TCS", date(2024, 1, 1), date(2024, 1, 31))
        assert not df.empty
        assert "feat_a" in df.columns
        assert df.index.name == "date"


class TestStoreOHLCV:
    async def test_empty_dataframe_returns_zero(self) -> None:
        result = await store.store_ohlcv("TCS", pd.DataFrame())
        assert result == 0

    async def test_stores_rows(self, mock_session_factory) -> None:
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000, 1100],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        result = await store.store_ohlcv("TCS", df)
        assert result == 2
        mock_session_factory.commit.assert_awaited_once()

    async def test_handles_optional_columns(self, mock_session_factory) -> None:
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000],
                "delivery_pct": [45.5],
                "circuit_hit": ["upper"],
                "is_adjusted": [True],
                "is_filled": [False],
            },
            index=pd.to_datetime(["2024-01-01"]),
        )
        result = await store.store_ohlcv("TCS", df)
        assert result == 1


class TestLoadOHLCV:
    async def test_empty_result(self, mock_session_factory) -> None:
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session_factory.execute.return_value = mock_result

        df = await store.load_ohlcv("TCS", date(2024, 1, 1), date(2024, 1, 31))
        assert df.empty

    async def test_returns_correct_columns(self, mock_session_factory) -> None:
        row = MagicMock()
        row.date = date(2024, 1, 1)
        row.open = 100.0
        row.high = 102.0
        row.low = 99.0
        row.close = 101.0
        row.adj_close = 101.0
        row.volume = 1000
        row.delivery_pct = 45.0
        row.circuit_hit = None
        row.is_adjusted = True
        row.is_filled = False
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [row]
        mock_session_factory.execute.return_value = mock_result

        df = await store.load_ohlcv("TCS", date(2024, 1, 1), date(2024, 1, 31))
        assert not df.empty
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df.index.name == "date"


class TestDeleteOHLCV:
    async def test_returns_deleted_count(self, mock_session_factory) -> None:
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_session_factory.execute.return_value = mock_result

        deleted = await store.delete_ohlcv("TCS")
        assert deleted == 5
        mock_session_factory.commit.assert_awaited_once()


class TestStoreFiiDii:
    async def test_empty_rows_returns_zero(self) -> None:
        result = await store.store_fii_dii([])
        assert result == 0

    async def test_stores_records(self, mock_session_factory) -> None:
        rows = [
            {
                "date": date(2024, 1, 1),
                "category": "FII",
                "buy_value": 1000.0,
                "sell_value": 800.0,
                "net_value": 200.0,
            }
        ]
        result = await store.store_fii_dii(rows)
        assert result == 1


class TestStoreEarnings:
    async def test_empty_returns_zero(self) -> None:
        result = await store.store_earnings([])
        assert result == 0

    async def test_stores_earnings_record(self, mock_session_factory) -> None:
        rows = [
            {
                "symbol": "TCS",
                "quarter": 1,
                "year": 2024,
                "revenue_actual": 5000.0,
                "profit_actual": 1200.0,
            }
        ]
        result = await store.store_earnings(rows)
        assert result == 1


class TestStorePromoterHoldings:
    async def test_empty_returns_zero(self) -> None:
        result = await store.store_promoter_holdings([])
        assert result == 0

    async def test_stores_holding(self, mock_session_factory) -> None:
        rows = [{"symbol": "TCS", "quarter_end": date(2024, 3, 31), "promoter_pct": 72.3}]
        result = await store.store_promoter_holdings(rows)
        assert result == 1


class TestStoreInsiderTrades:
    async def test_empty_returns_zero(self) -> None:
        result = await store.store_insider_trades([])
        assert result == 0

    async def test_stores_trade(self, mock_session_factory) -> None:
        rows = [
            {
                "symbol": "TCS",
                "trade_date": date(2024, 1, 15),
                "trade_type": "Buy",
                "shares": 500,
            }
        ]
        result = await store.store_insider_trades(rows)
        assert result == 1


class TestStoreNewsArticles:
    async def test_empty_returns_zero(self) -> None:
        result = await store.store_news_articles([])
        assert result == 0

    async def test_stores_article(self, mock_session_factory) -> None:
        rows = [
            {
                "source": "finnhub",
                "title": "TCS Q1 results beat estimates",
                "published_date": date(2024, 7, 10),
                "content_hash": "abc123def456",
                "sentiment_score": 0.75,
            }
        ]
        result = await store.store_news_articles(rows)
        assert result == 1


class TestStoreAlternativeData:
    async def test_empty_returns_zero(self) -> None:
        result = await store.store_alternative_data([])
        assert result == 0

    async def test_stores_record(self, mock_session_factory) -> None:
        rows = [
            {
                "data_type": "auto_sales",
                "period_date": date(2024, 1, 1),
                "value": 35000.0,
                "yoy_change": 12.5,
                "sector": "auto",
            }
        ]
        result = await store.store_alternative_data(rows)
        assert result == 1


class TestStorePaperTrade:
    async def test_stores_and_returns_one(self, mock_session_factory) -> None:
        row = {
            "symbol": "TCS",
            "prediction_date": date(2024, 1, 15),
            "predicted_direction": 1,
            "predicted_magnitude": 0.025,
            "confidence": 0.72,
            "model_version": "v1.0",
        }
        result = await store.store_paper_trade(row)
        assert result == 1
