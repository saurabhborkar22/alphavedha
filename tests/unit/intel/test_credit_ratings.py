"""Tests for credit rating actions collector."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from alphavedha.intel.collectors.credit_ratings import (
    AGENCY_PATTERNS,
    _is_credit_rating,
    _map_action,
    _parse_filed_at,
    detect_agency,
    row_to_rating_event,
)

IST = ZoneInfo("Asia/Kolkata")


class TestDetectAgency:
    def test_detects_crisil(self) -> None:
        assert detect_agency("Rating from CRISIL Ratings Limited") == "CRISIL"

    def test_detects_icra(self) -> None:
        assert detect_agency("ICRA has assigned rating") == "ICRA"

    def test_detects_care(self) -> None:
        assert detect_agency("Credit Rating from CARE Ratings Limited") == "CARE"

    def test_detects_india_ratings(self) -> None:
        assert detect_agency("India Ratings has affirmed") == "India Ratings"

    def test_detects_fitch(self) -> None:
        assert detect_agency("Fitch has upgraded") == "Fitch"

    def test_detects_brickwork(self) -> None:
        assert detect_agency("Brickwork Ratings assigned") == "Brickwork"

    def test_detects_acuite(self) -> None:
        assert detect_agency("Acuite Ratings has reaffirmed") == "Acuite"

    def test_detects_infomerics(self) -> None:
        assert detect_agency("Infomerics Valuation assigned") == "Infomerics"

    def test_returns_unknown_for_no_match(self) -> None:
        assert detect_agency("Company has informed about rating") == "Unknown"

    def test_case_insensitive(self) -> None:
        assert detect_agency("crisil ratings") == "CRISIL"

    def test_agency_patterns_count(self) -> None:
        assert len(AGENCY_PATTERNS) == 8


class TestIsCreditRating:
    def test_plain_credit_rating(self) -> None:
        assert _is_credit_rating("Credit Rating") is True

    def test_new(self) -> None:
        assert _is_credit_rating("Credit Rating- New") is True

    def test_revision(self) -> None:
        assert _is_credit_rating("Credit Rating- Revision") is True

    def test_others(self) -> None:
        assert _is_credit_rating("Credit Rating- Others") is True

    def test_non_rating(self) -> None:
        assert _is_credit_rating("Board Meeting") is False

    def test_empty(self) -> None:
        assert _is_credit_rating("") is False


class TestMapAction:
    def test_plain(self) -> None:
        assert _map_action("Credit Rating") == "reaffirmed"

    def test_new(self) -> None:
        assert _map_action("Credit Rating- New") == "assigned"

    def test_revision(self) -> None:
        assert _map_action("Credit Rating- Revision") == "revised"

    def test_others(self) -> None:
        assert _map_action("Credit Rating- Others") == "other"

    def test_unknown_defaults_reaffirmed(self) -> None:
        assert _map_action("Something Else") == "reaffirmed"


class TestParseFiledAt:
    def test_with_seconds(self) -> None:
        result = _parse_filed_at("18-Jun-2026 10:30:00")
        assert result is not None
        assert result.tzinfo == IST

    def test_without_seconds(self) -> None:
        result = _parse_filed_at("18-Jun-2026 10:30")
        assert result is not None

    def test_date_only(self) -> None:
        result = _parse_filed_at("18-Jun-2026")
        assert result is not None

    def test_garbage(self) -> None:
        assert _parse_filed_at("not a date") is None


class TestRowToRatingEvent:
    def _make_row(self, **overrides: str | None) -> dict[str, str | None]:
        base: dict[str, str | None] = {
            "desc": "Credit Rating",
            "an_dt": "18-Jun-2026 10:30:00",
            "symbol": "TCS",
            "attchmntText": "Credit Rating from CRISIL Ratings Limited",
            "attchmntFile": "https://nsearchives.nseindia.com/corporate/TCS_rating.pdf",
            "sort_date": None,
            "sm_name": None,
        }
        base.update(overrides)
        return base

    def test_converts_valid_row(self) -> None:
        event = row_to_rating_event(self._make_row())
        assert event is not None
        assert event["symbol"] == "TCS.NS"
        assert event["agency"] == "CRISIL"
        assert event["action"] == "reaffirmed"
        assert isinstance(event["filed_at"], datetime)
        assert event["filed_at"].tzinfo == IST
        assert event["_pdf_url"] is not None

    def test_new_action(self) -> None:
        event = row_to_rating_event(self._make_row(desc="Credit Rating- New"))
        assert event is not None
        assert event["action"] == "assigned"

    def test_revision_action(self) -> None:
        event = row_to_rating_event(self._make_row(desc="Credit Rating- Revision"))
        assert event is not None
        assert event["action"] == "revised"

    def test_returns_none_for_non_rating(self) -> None:
        assert row_to_rating_event(self._make_row(desc="Board Meeting")) is None

    def test_returns_none_for_bad_date(self) -> None:
        assert row_to_rating_event(self._make_row(an_dt="bad")) is None

    def test_returns_none_for_empty_symbol(self) -> None:
        assert row_to_rating_event(self._make_row(symbol="")) is None

    def test_unknown_agency_when_no_match(self) -> None:
        event = row_to_rating_event(self._make_row(attchmntText="Company informed about rating"))
        assert event is not None
        assert event["agency"] == "Unknown"

    def test_detects_icra(self) -> None:
        event = row_to_rating_event(self._make_row(attchmntText="Credit Rating from ICRA Limited"))
        assert event is not None
        assert event["agency"] == "ICRA"

    def test_no_pdf_url(self) -> None:
        event = row_to_rating_event(self._make_row(attchmntFile=""))
        assert event is not None
        assert event["_pdf_url"] is None

    def test_rationale_initially_none(self) -> None:
        event = row_to_rating_event(self._make_row())
        assert event is not None
        assert event["rationale_text"] is None
        assert event["rating_from"] is None
        assert event["rating_to"] is None
        assert event["outlook"] is None
