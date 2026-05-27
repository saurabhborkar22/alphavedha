from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphavedha.data.store import delete_ohlcv, load_ohlcv, store_ohlcv

pytestmark = pytest.mark.integration


def _make_ohlcv(symbol: str, n_days: int, start: str = "2024-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days, freq="B")
    rng = np.random.default_rng(42)
    base = 100.0
    closes = base * np.cumprod(1 + rng.normal(0.001, 0.015, n_days))
    return pd.DataFrame(
        {
            "open": closes * (1 + rng.normal(0, 0.005, n_days)),
            "high": closes * (1 + np.abs(rng.normal(0, 0.01, n_days))),
            "low": closes * (1 - np.abs(rng.normal(0, 0.01, n_days))),
            "close": closes,
            "adj_close": closes,
            "volume": rng.integers(1_000_000, 10_000_000, size=n_days),
        },
        index=dates,
    )


def _patch_sf(session_factory):
    return patch("alphavedha.data.store.get_session_factory", return_value=session_factory)


class TestOHLCVStoreLoad:
    @pytest.mark.asyncio()
    async def test_insert_and_query_ohlcv(self, session_factory) -> None:
        df = _make_ohlcv("TCS.NS", 20)
        with _patch_sf(session_factory):
            stored = await store_ohlcv("TCS.NS", df)
            assert stored == 20

            loaded = await load_ohlcv("TCS.NS", date(2024, 1, 1), date(2024, 12, 31))
        assert len(loaded) == 20
        np.testing.assert_allclose(loaded["close"].values, df["close"].values, rtol=1e-6)

    @pytest.mark.asyncio()
    async def test_upsert_idempotent(self, session_factory) -> None:
        df = _make_ohlcv("INFY.NS", 10)
        with _patch_sf(session_factory):
            await store_ohlcv("INFY.NS", df)
            await store_ohlcv("INFY.NS", df)
            loaded = await load_ohlcv("INFY.NS", date(2024, 1, 1), date(2024, 12, 31))
        assert len(loaded) == 10

    @pytest.mark.asyncio()
    async def test_date_range_filtering(self, session_factory) -> None:
        df = _make_ohlcv("RELIANCE.NS", 100)
        with _patch_sf(session_factory):
            await store_ohlcv("RELIANCE.NS", df)
            loaded = await load_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 31))
        assert len(loaded) < 100
        assert all(d.date() <= date(2024, 1, 31) for d in loaded.index)

    @pytest.mark.asyncio()
    async def test_multiple_symbols_no_cross_contamination(self, session_factory) -> None:
        df_tcs = _make_ohlcv("TCS.NS", 15)
        df_infy = _make_ohlcv("INFY.NS", 10)
        with _patch_sf(session_factory):
            await store_ohlcv("TCS.NS", df_tcs)
            await store_ohlcv("INFY.NS", df_infy)
            loaded_tcs = await load_ohlcv("TCS.NS", date(2024, 1, 1), date(2024, 12, 31))
            loaded_infy = await load_ohlcv("INFY.NS", date(2024, 1, 1), date(2024, 12, 31))
        assert len(loaded_tcs) == 15
        assert len(loaded_infy) == 10

    @pytest.mark.asyncio()
    async def test_delete_ohlcv(self, session_factory) -> None:
        df = _make_ohlcv("HDFC.NS", 5)
        with _patch_sf(session_factory):
            await store_ohlcv("HDFC.NS", df)
            deleted = await delete_ohlcv("HDFC.NS")
            assert deleted == 5
            loaded = await load_ohlcv("HDFC.NS", date(2024, 1, 1), date(2024, 12, 31))
        assert len(loaded) == 0
