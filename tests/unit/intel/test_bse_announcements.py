"""Tests for BSE announcements collector — parsing, PDF extraction, and ingestion."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from alphavedha.intel.collectors.bse_announcements import (
    _parse_filed_at,
    _pdf_url,
    _row_to_disclosure,
    extract_pdf_text,
)

IST = ZoneInfo("Asia/Kolkata")


class TestParseFiledAt:
    def test_parses_fractional_seconds(self) -> None:
        result = _parse_filed_at("2026-06-17T11:35:19.91")
        assert result is not None
        assert result.year == 2026
        assert result.month == 6
        assert result.day == 17
        assert result.tzinfo == IST

    def test_parses_without_fractional(self) -> None:
        result = _parse_filed_at("2026-06-17T11:35:19")
        assert result is not None
        assert result.hour == 11

    def test_returns_none_for_garbage(self) -> None:
        assert _parse_filed_at("not-a-date") is None
        assert _parse_filed_at("") is None

    def test_timezone_aware(self) -> None:
        result = _parse_filed_at("2026-06-17T11:35:19.91")
        assert result is not None
        assert result.tzinfo is not None


class TestPdfUrl:
    def test_builds_url(self) -> None:
        url = _pdf_url("abc123.pdf")
        assert url is not None
        assert url.endswith("/abc123.pdf")
        assert "AttachLive" in url

    def test_strips_whitespace(self) -> None:
        url = _pdf_url("  abc.pdf  ")
        assert url is not None
        assert url.endswith("/abc.pdf")

    def test_none_for_empty(self) -> None:
        assert _pdf_url("") is None
        assert _pdf_url("   ") is None


class TestRowToDisclosure:
    def test_converts_valid_row(self) -> None:
        row = {
            "DT_TM": "2026-06-17T11:35:19.91",
            "NEWSSUB": "TCS wins deal",
            "CATEGORYNAME": "Company Update",
            "ATTACHMENTNAME": "test.pdf",
        }
        disc = _row_to_disclosure("TCS.NS", row)
        assert disc is not None
        assert disc["symbol"] == "TCS.NS"
        assert disc["source"] == "BSE"
        assert disc["category"] == "Company Update"
        assert disc["headline"] == "TCS wins deal"
        assert isinstance(disc["filed_at"], datetime)
        assert disc["url"] is not None

    def test_returns_none_for_missing_date(self) -> None:
        row = {"NEWSSUB": "headline", "DT_TM": ""}
        assert _row_to_disclosure("TCS.NS", row) is None

    def test_returns_none_for_empty_headline(self) -> None:
        row = {"DT_TM": "2026-06-17T11:35:19.91", "NEWSSUB": "", "HEADLINE": ""}
        assert _row_to_disclosure("TCS.NS", row) is None

    def test_replaces_double_quotes(self) -> None:
        row = {
            "DT_TM": "2026-06-17T11:35:19.91",
            "NEWSSUB": "Elopak''s operations",
            "CATEGORYNAME": "General",
        }
        disc = _row_to_disclosure("TCS.NS", row)
        assert disc is not None
        assert "''" not in disc["headline"]
        assert "'" in disc["headline"]

    def test_no_attachment(self) -> None:
        row = {
            "DT_TM": "2026-06-17T11:35:19.91",
            "NEWSSUB": "headline",
            "CATEGORYNAME": "General",
        }
        disc = _row_to_disclosure("TCS.NS", row)
        assert disc is not None
        assert disc["url"] is None


class TestExtractPdfText:
    def test_returns_none_for_invalid_bytes(self) -> None:
        result = extract_pdf_text(b"not a pdf")
        assert result is None

    def test_returns_none_for_empty(self) -> None:
        result = extract_pdf_text(b"")
        assert result is None

    def test_extracts_from_valid_pdf(self) -> None:
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Hello from AlphaVedha test")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = extract_pdf_text(pdf_bytes)
        assert result is not None
        assert "AlphaVedha" in result

    def test_respects_max_pages(self) -> None:
        import fitz

        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((50, 50), f"Page {i}")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = extract_pdf_text(pdf_bytes)
        assert result is not None
        assert "Page 0" in result
        assert "Page 2" in result
