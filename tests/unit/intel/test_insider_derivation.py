"""Tests for insider-trade derivation from disclosure events.

Parser cases mirror real production summaries (disclosure_events ids
177-270 on the VPS) — the LLM output shapes this module must handle.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from alphavedha.intel.insider_derivation import (
    derive_insider_trades,
    parse_person,
    parse_shares,
    parse_trade_date,
    parse_value_lakhs,
)

IST = ZoneInfo("Asia/Kolkata")
FILED = datetime(2026, 6, 26, 14, 0, tzinfo=IST)


class TestParsePerson:
    def test_strips_role_prefix(self) -> None:
        s = "Director Ajay Pancholi purchased 77,000 equity shares worth INR 4.07 crores"
        assert parse_person(s, 1) == "Ajay Pancholi"

    def test_promoter_group_member(self) -> None:
        s = "Promoter group member Mrinaal Mittal sold 46.83 lakh shares (2.99%) on 25-Jun-2026"
        assert parse_person(s, 2) == "Mrinaal Mittal"

    def test_entity_name_kept_whole(self) -> None:
        s = "Sethi Funds Management Private Limited acquired 1,95,000 equity shares (0.17%)"
        assert parse_person(s, 3) == "Sethi Funds Management Private Limited"

    def test_trust_with_tendered_verb(self) -> None:
        s = "Azim Premji Trust tendered 3.83Cr shares (0.39%) in Wipro buyback"
        assert parse_person(s, 4) == "Azim Premji Trust"

    def test_fallback_synthetic_key(self) -> None:
        assert parse_person("Routine compliance filing", 42) == "filing-42"


class TestParseShares:
    def test_indian_comma_format(self) -> None:
        assert parse_shares("acquired 1,95,000 equity shares (0.17%)") == 195_000

    def test_lakh_suffix_word(self) -> None:
        assert parse_shares("sold 46.83 lakh shares (2.99%)") == 4_683_000

    def test_l_suffix(self) -> None:
        assert parse_shares("tendered 2.88L shares (0.003%)") == 288_000

    def test_crore_suffix(self) -> None:
        assert parse_shares("tendered 12.17Cr shares (1.23%)") == 121_700_000

    def test_absent(self) -> None:
        assert parse_shares("increased stake via market purchase") == 0


class TestParseValueLakhs:
    def test_inr_crores(self) -> None:
        s = "purchased 77,000 equity shares worth INR 4.07 crores"
        assert parse_value_lakhs(s) == pytest.approx(407.0)

    def test_rs_lakh(self) -> None:
        assert parse_value_lakhs("acquired shares worth Rs. 55 lakh") == pytest.approx(55.0)

    def test_plain_rupees(self) -> None:
        assert parse_value_lakhs("aggregating to ₹12,50,000") == pytest.approx(12.5)

    def test_absent(self) -> None:
        assert parse_value_lakhs("tendered 2.88L shares (0.003%) in buyback") == 0.0

    def test_percentage_not_mistaken_for_value(self) -> None:
        assert parse_value_lakhs("sold 46.83 lakh shares (2.99%)") == 0.0


class TestParseTradeDate:
    def test_summary_date_used(self) -> None:
        s = "sold 46.83 lakh shares (2.99%) on 25-Jun-2026"
        assert parse_trade_date(s, FILED) == date(2026, 6, 25)

    def test_iso_date(self) -> None:
        s = "acquired shares on 2026-06-20"
        assert parse_trade_date(s, FILED) == date(2026, 6, 20)

    def test_future_date_clamped_to_filing(self) -> None:
        """Point-in-time: a trade only becomes knowable when filed."""
        s = "will acquire shares on 30-Jun-2026"
        assert parse_trade_date(s, FILED) == FILED.date()

    def test_no_date_falls_back_to_filing(self) -> None:
        assert parse_trade_date("acquired shares", FILED) == FILED.date()


class _FakeEvent:
    def __init__(self, disclosure_id: int, symbol: str, event_type: str, summary: str) -> None:
        self.disclosure_id = disclosure_id
        self.symbol = symbol
        self.event_type = event_type
        self.summary = summary


class _FakeDisclosure:
    def __init__(self, filed_at: datetime) -> None:
        self.filed_at = filed_at


class TestDeriveInsiderTrades:
    @pytest.mark.asyncio
    async def test_derives_rows_from_events(self) -> None:
        pairs = [
            (
                _FakeEvent(
                    10,
                    "EBGNG.NS",
                    "insider_buy",
                    "Director Ajay Pancholi purchased 77,000 equity shares worth INR 4.07 crores",
                ),
                _FakeDisclosure(FILED),
            ),
            (
                _FakeEvent(
                    11,
                    "WIPRO",
                    "insider_sell",
                    "Azim Premji Trust tendered 3.83Cr shares (0.39%) in Wipro buyback",
                ),
                _FakeDisclosure(FILED),
            ),
        ]

        ohlcv = pd.DataFrame(
            {"close": [250.0]},
            index=pd.DatetimeIndex([pd.Timestamp(date(2026, 6, 25))]),
        )

        stored_rows: list[dict[str, Any]] = []

        async def fake_store(rows: list[dict[str, Any]]) -> int:
            stored_rows.extend(rows)
            return len(rows)

        with (
            patch(
                "alphavedha.intel.insider_derivation._load_recent_insider_events",
                new_callable=AsyncMock,
                return_value=pairs,
            ),
            patch(
                "alphavedha.data.store.store_insider_trades",
                side_effect=fake_store,
            ),
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=ohlcv,
            ),
        ):
            stored = await derive_insider_trades()

        assert stored == 2
        buy, sell = stored_rows

        assert buy["symbol"] == "EBGNG"  # .NS stripped
        assert buy["trade_type"] == "buy"
        assert buy["person_name"] == "Ajay Pancholi"
        assert buy["shares"] == 77_000
        assert buy["value_lakhs"] == pytest.approx(407.0)  # explicit value wins
        assert buy["trade_date"] == FILED.date()
        assert buy["person_category"] == "derived_from_disclosure"

        assert sell["symbol"] == "WIPRO"
        assert sell["trade_type"] == "sell"
        assert sell["person_name"] == "Azim Premji Trust"
        # no explicit value → estimated from close: 3.83Cr shares * 250 / 1e5
        assert sell["value_lakhs"] == pytest.approx(38_300_000 * 250.0 / 100_000)

    @pytest.mark.asyncio
    async def test_no_events_returns_zero(self) -> None:
        with patch(
            "alphavedha.intel.insider_derivation._load_recent_insider_events",
            new_callable=AsyncMock,
            return_value=[],
        ):
            assert await derive_insider_trades() == 0
