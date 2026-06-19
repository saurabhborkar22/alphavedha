"""Tests for bulk/block deals collector."""

from __future__ import annotations

from datetime import date

from alphavedha.intel.collectors.bulk_block_deals import (
    _parse_date,
    _parse_deal_rows,
)


class TestParseDate:
    def test_parses_dashes(self) -> None:
        assert _parse_date("18-Jun-2026") == date(2026, 6, 18)

    def test_parses_slashes(self) -> None:
        assert _parse_date("18/06/2026") == date(2026, 6, 18)

    def test_strips_whitespace(self) -> None:
        assert _parse_date("  18-Jun-2026  ") == date(2026, 6, 18)

    def test_returns_none_for_garbage(self) -> None:
        assert _parse_date("not a date") is None

    def test_returns_none_for_empty(self) -> None:
        assert _parse_date("") is None


class TestParseDealRows:
    def _make_item(self, **overrides: str | int | None) -> dict[str, str | int | None]:
        base: dict[str, str | int | None] = {
            "symbol": "TCS",
            "date": "18-Jun-2026",
            "clientName": "ACME Fund",
            "buySell": "BUY",
            "qty": "100000",
            "watp": "3500.50",
        }
        base.update(overrides)
        return base

    def test_parses_valid_row(self) -> None:
        rows = _parse_deal_rows([self._make_item()], "BULK")
        assert len(rows) == 1
        r = rows[0]
        assert r["symbol"] == "TCS.NS"
        assert r["deal_date"] == date(2026, 6, 18)
        assert r["deal_type"] == "BULK"
        assert r["client_name"] == "ACME Fund"
        assert r["trade_type"] == "BUY"
        assert r["quantity"] == 100000
        assert r["price"] == 3500.50

    def test_sell_trade_type(self) -> None:
        rows = _parse_deal_rows([self._make_item(buySell="SELL")], "BLOCK")
        assert rows[0]["trade_type"] == "SELL"

    def test_unknown_trade_type(self) -> None:
        rows = _parse_deal_rows([self._make_item(buySell="")], "BULK")
        assert rows[0]["trade_type"] == "UNKNOWN"

    def test_skips_empty_symbol(self) -> None:
        rows = _parse_deal_rows([self._make_item(symbol="")], "BULK")
        assert len(rows) == 0

    def test_skips_zero_quantity(self) -> None:
        rows = _parse_deal_rows([self._make_item(qty="0")], "BULK")
        assert len(rows) == 0

    def test_handles_comma_in_qty(self) -> None:
        rows = _parse_deal_rows([self._make_item(qty="1,00,000")], "BULK")
        assert rows[0]["quantity"] == 100000

    def test_handles_comma_in_price(self) -> None:
        rows = _parse_deal_rows([self._make_item(watp="3,500.50")], "BULK")
        assert rows[0]["price"] == 3500.50

    def test_truncates_client_name(self) -> None:
        rows = _parse_deal_rows([self._make_item(clientName="x" * 300)], "BULK")
        assert len(rows[0]["client_name"]) == 200

    def test_truncates_deal_type(self) -> None:
        rows = _parse_deal_rows([self._make_item()], "VERYLONGTYPE")
        assert len(rows[0]["deal_type"]) == 10

    def test_unknown_client_fallback(self) -> None:
        rows = _parse_deal_rows([self._make_item(clientName="")], "BULK")
        assert rows[0]["client_name"] == "Unknown"

    def test_multiple_items(self) -> None:
        items = [
            self._make_item(symbol="TCS"),
            self._make_item(symbol="INFY"),
            self._make_item(symbol="RELIANCE"),
        ]
        rows = _parse_deal_rows(items, "BLOCK")
        assert len(rows) == 3
        symbols = {r["symbol"] for r in rows}
        assert symbols == {"TCS.NS", "INFY.NS", "RELIANCE.NS"}

    def test_bad_qty_skipped(self) -> None:
        rows = _parse_deal_rows([self._make_item(qty="abc")], "BULK")
        assert len(rows) == 0

    def test_bad_price_defaults_zero(self) -> None:
        rows = _parse_deal_rows([self._make_item(watp="abc")], "BULK")
        assert rows[0]["price"] == 0.0
