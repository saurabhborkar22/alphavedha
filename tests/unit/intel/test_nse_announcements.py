"""Tests for NSE announcements collector."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from alphavedha.intel.collectors.nse_announcements import (
    PIT_CATEGORIES,
    PLEDGE_CATEGORIES,
    _is_pit_filing,
    _is_pledge_filing,
    _parse_filed_at,
    _row_to_disclosure,
    _text_hash,
)

IST = ZoneInfo("Asia/Kolkata")


class TestParseFiledAt:
    def test_parses_with_seconds(self) -> None:
        result = _parse_filed_at("18-Jun-2026 22:48:04")
        assert result is not None
        assert result.year == 2026
        assert result.month == 6
        assert result.day == 18
        assert result.hour == 22
        assert result.minute == 48
        assert result.second == 4
        assert result.tzinfo == IST

    def test_parses_without_seconds(self) -> None:
        result = _parse_filed_at("18-Jun-2026 10:30")
        assert result is not None
        assert result.minute == 30
        assert result.tzinfo == IST

    def test_parses_date_only(self) -> None:
        result = _parse_filed_at("18-Jun-2026")
        assert result is not None
        assert result.hour == 0
        assert result.tzinfo == IST

    def test_returns_none_for_garbage(self) -> None:
        assert _parse_filed_at("not a date") is None

    def test_returns_none_for_empty(self) -> None:
        assert _parse_filed_at("") is None


class TestPitPledgeDetection:
    def test_pit_categories_defined(self) -> None:
        assert len(PIT_CATEGORIES) > 0

    def test_pledge_categories_defined(self) -> None:
        assert len(PLEDGE_CATEGORIES) > 0

    def test_detects_insider_trading(self) -> None:
        assert _is_pit_filing("Insider Trading", "Director bought shares")

    def test_detects_pit_in_headline(self) -> None:
        assert _is_pit_filing("General", "Disclosure under PIT regulations")

    def test_no_false_positive(self) -> None:
        assert not _is_pit_filing("Board Meeting", "Quarterly results")

    def test_detects_pledge(self) -> None:
        assert _is_pledge_filing("Pledge", "Promoter pledged shares")

    def test_detects_sast(self) -> None:
        assert _is_pledge_filing("SAST", "Regulation 30 disclosure")

    def test_detects_encumbrance(self) -> None:
        assert _is_pledge_filing("General", "Encumbrance of shares by promoter")

    def test_pledge_no_false_positive(self) -> None:
        assert not _is_pledge_filing("Board Meeting", "Quarterly results")


class TestRowToDisclosure:
    def _make_row(self, **overrides: str | None) -> dict[str, str | None]:
        base: dict[str, str | None] = {
            "an_dt": "18-Jun-2026 10:30:00",
            "symbol": "TCS",
            "attchmntText": "Board meeting outcome",
            "desc": "Board Meeting",
            "attchmntFile": "https://nsearchives.nseindia.com/corporate/TCS.pdf",
            "sort_date": None,
            "sm_name": None,
        }
        base.update(overrides)
        return base

    def test_converts_valid_row(self) -> None:
        disc = _row_to_disclosure(self._make_row())
        assert disc is not None
        assert disc["symbol"] == "TCS.NS"
        assert disc["source"] == "NSE"
        assert disc["category"] == "Board Meeting"
        assert disc["headline"] == "Board meeting outcome"
        assert isinstance(disc["filed_at"], datetime)
        assert disc["filed_at"].tzinfo == IST
        assert disc["url"] == "https://nsearchives.nseindia.com/corporate/TCS.pdf"

    def test_returns_none_missing_date(self) -> None:
        assert _row_to_disclosure(self._make_row(an_dt="bad")) is None

    def test_returns_none_missing_symbol(self) -> None:
        assert _row_to_disclosure(self._make_row(symbol="")) is None

    def test_returns_none_no_headline(self) -> None:
        assert _row_to_disclosure(self._make_row(attchmntText="", sm_name="")) is None

    def test_falls_back_to_sm_name(self) -> None:
        disc = _row_to_disclosure(self._make_row(attchmntText="", sm_name="Tata Consultancy"))
        assert disc is not None
        assert disc["headline"] == "Tata Consultancy"

    def test_empty_url_becomes_none(self) -> None:
        disc = _row_to_disclosure(self._make_row(attchmntFile=""))
        assert disc is not None
        assert disc["url"] is None

    def test_pit_flag_set(self) -> None:
        disc = _row_to_disclosure(self._make_row(desc="Insider Trading"))
        assert disc is not None
        assert disc["_is_pit"] is True

    def test_pledge_flag_set(self) -> None:
        disc = _row_to_disclosure(self._make_row(desc="Pledge"))
        assert disc is not None
        assert disc["_is_pledge"] is True

    def test_truncates_long_headline(self) -> None:
        disc = _row_to_disclosure(self._make_row(attchmntText="x" * 2000))
        assert disc is not None
        assert len(disc["headline"]) == 1000

    def test_truncates_long_category(self) -> None:
        disc = _row_to_disclosure(self._make_row(desc="y" * 200))
        assert disc is not None
        assert len(disc["category"]) == 100


class TestTextHash:
    def test_deterministic(self) -> None:
        assert _text_hash("hello") == _text_hash("hello")

    def test_different_for_different_text(self) -> None:
        assert _text_hash("hello") != _text_hash("world")
