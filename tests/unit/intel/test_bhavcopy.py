"""Tests for bhavcopy collector — parsing and normalisation."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

from alphavedha.intel.collectors.bhavcopy import EQUITY_SERIES, parse_bhavcopy

SAMPLE_CSV = """\
 SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER
 TCS, EQ, 16-Jun-2026, 2162.00, 2167.00, 2209.00, 2160.60, 2197.90, 2199.00, 2189.97, 3079898, 67448.90, 120376, 1314406, 42.68
 INFY, EQ, 16-Jun-2026, 1800.00, 1805.00, 1820.00, 1795.00, 1812.00, 1815.00, 1810.50, 2000000, 36210.00, 80000, 900000, 45.00
 RELIANCE, BE, 16-Jun-2026, 2500.00, 2510.00, 2530.00, 2490.00, 2520.00, 2525.00, 2515.00, 500000, 12575.00, 20000, 400000, 80.00
 GOLDBEES, IV, 16-Jun-2026, 55.00, 55.10, 55.20, 54.90, 55.15, 55.10, 55.08, 100000, 55.10, 500, 80000, 80.00
 1018GS2026, GS, 16-Jun-2026, 102.98, 103.12, 103.12, 103.11, 103.11, 103.11, 103.11, 18, 0.02, 3, 14, 77.78
"""


class TestParseBhavcopy:
    def test_filters_to_equity_series(self) -> None:
        df = parse_bhavcopy(SAMPLE_CSV)
        assert len(df) == 3
        assert set(df["symbol"]) == {"TCS.NS", "INFY.NS", "RELIANCE.NS"}

    def test_excludes_non_equity_series(self) -> None:
        df = parse_bhavcopy(SAMPLE_CSV)
        symbols = df["symbol"].tolist()
        assert "GOLDBEES.NS" not in symbols
        assert "1018GS2026.NS" not in symbols

    def test_column_names(self) -> None:
        df = parse_bhavcopy(SAMPLE_CSV)
        expected = {
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "delivery_pct",
            "adj_close",
        }
        assert set(df.columns) == expected

    def test_symbol_has_ns_suffix(self) -> None:
        df = parse_bhavcopy(SAMPLE_CSV)
        assert all(s.endswith(".NS") for s in df["symbol"])

    def test_price_values_correct(self) -> None:
        df = parse_bhavcopy(SAMPLE_CSV)
        tcs = df[df["symbol"] == "TCS.NS"].iloc[0]
        assert tcs["open"] == 2167.0
        assert tcs["high"] == 2209.0
        assert tcs["low"] == 2160.6
        assert tcs["close"] == 2199.0

    def test_volume_is_int(self) -> None:
        df = parse_bhavcopy(SAMPLE_CSV)
        assert df["volume"].dtype in ("int64", "int32")

    def test_delivery_pct_present(self) -> None:
        df = parse_bhavcopy(SAMPLE_CSV)
        tcs = df[df["symbol"] == "TCS.NS"].iloc[0]
        assert abs(tcs["delivery_pct"] - 42.68) < 0.01

    def test_adj_close_equals_close(self) -> None:
        df = parse_bhavcopy(SAMPLE_CSV)
        assert (df["adj_close"] == df["close"]).all()

    def test_date_parsed_correctly(self) -> None:
        df = parse_bhavcopy(SAMPLE_CSV)
        assert df["date"].iloc[0] == date(2026, 6, 16)

    def test_be_series_included(self) -> None:
        df = parse_bhavcopy(SAMPLE_CSV)
        assert "RELIANCE.NS" in df["symbol"].values

    def test_empty_csv(self) -> None:
        empty = " SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
        df = parse_bhavcopy(empty)
        assert df.empty

    def test_equity_series_constant(self) -> None:
        assert "EQ" in EQUITY_SERIES
        assert "BE" in EQUITY_SERIES


class TestIngestBhavcopy:
    async def test_ingest_calls_store_per_symbol(self) -> None:
        with (
            patch(
                "alphavedha.intel.collectors.bhavcopy.fetch_bhavcopy",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch(
                "alphavedha.intel.collectors.bhavcopy.store_ohlcv",
                new_callable=AsyncMock,
                return_value=1,
            ) as mock_store,
        ):
            mock_fetch.return_value = parse_bhavcopy(SAMPLE_CSV)
            from alphavedha.intel.collectors.bhavcopy import ingest_bhavcopy

            rows = await ingest_bhavcopy(date(2026, 6, 16))
            assert rows == 3
            assert mock_store.call_count == 3

    async def test_ingest_returns_zero_on_fetch_error(self) -> None:
        with patch(
            "alphavedha.intel.collectors.bhavcopy.fetch_bhavcopy",
            new_callable=AsyncMock,
            side_effect=Exception("NSE down"),
        ):
            from alphavedha.intel.collectors.bhavcopy import ingest_bhavcopy

            rows = await ingest_bhavcopy(date(2026, 6, 16))
            assert rows == 0
