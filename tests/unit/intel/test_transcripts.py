"""Tests for concall transcript collector."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from alphavedha.intel.collectors.transcripts import (
    TRANSCRIPT_DESC,
    infer_fiscal_quarter,
    is_transcript_announcement,
    row_to_transcript,
    split_sections,
)

IST = ZoneInfo("Asia/Kolkata")


class TestInferFiscalQuarter:
    def test_may_is_q4(self) -> None:
        dt = datetime(2026, 5, 15, tzinfo=IST)
        assert infer_fiscal_quarter(dt) == "Q4FY26"

    def test_june_is_q4(self) -> None:
        dt = datetime(2026, 6, 10, tzinfo=IST)
        assert infer_fiscal_quarter(dt) == "Q4FY26"

    def test_july_is_q4(self) -> None:
        dt = datetime(2026, 7, 20, tzinfo=IST)
        assert infer_fiscal_quarter(dt) == "Q4FY26"

    def test_august_is_q1(self) -> None:
        dt = datetime(2026, 8, 10, tzinfo=IST)
        assert infer_fiscal_quarter(dt) == "Q1FY27"

    def test_october_is_q1(self) -> None:
        dt = datetime(2026, 10, 5, tzinfo=IST)
        assert infer_fiscal_quarter(dt) == "Q1FY27"

    def test_november_is_q2(self) -> None:
        dt = datetime(2026, 11, 15, tzinfo=IST)
        assert infer_fiscal_quarter(dt) == "Q2FY27"

    def test_december_is_q2(self) -> None:
        dt = datetime(2026, 12, 20, tzinfo=IST)
        assert infer_fiscal_quarter(dt) == "Q2FY27"

    def test_january_is_q2(self) -> None:
        dt = datetime(2027, 1, 10, tzinfo=IST)
        assert infer_fiscal_quarter(dt) == "Q2FY27"

    def test_february_is_q3(self) -> None:
        dt = datetime(2027, 2, 15, tzinfo=IST)
        assert infer_fiscal_quarter(dt) == "Q3FY28"

    def test_march_is_q3(self) -> None:
        dt = datetime(2027, 3, 20, tzinfo=IST)
        assert infer_fiscal_quarter(dt) == "Q3FY28"

    def test_april_is_q3(self) -> None:
        dt = datetime(2027, 4, 10, tzinfo=IST)
        assert infer_fiscal_quarter(dt) == "Q3FY28"


class TestIsTranscriptAnnouncement:
    def test_exact_desc_match(self) -> None:
        assert is_transcript_announcement(TRANSCRIPT_DESC, "Some headline") is True

    def test_transcript_keyword_in_headline(self) -> None:
        assert is_transcript_announcement("General Updates", "Transcript of AGM") is True

    def test_concall_keyword(self) -> None:
        assert is_transcript_announcement("Updates", "Concall results") is True

    def test_non_transcript(self) -> None:
        assert is_transcript_announcement("Board Meeting", "Quarterly results") is False

    def test_schedule_filtered_out(self) -> None:
        assert is_transcript_announcement(TRANSCRIPT_DESC, "Schedule of analyst meet") is False

    def test_schedule_with_transcript_kept(self) -> None:
        assert is_transcript_announcement(TRANSCRIPT_DESC, "Revised schedule + transcript") is True

    def test_intimation_filtered(self) -> None:
        assert is_transcript_announcement(TRANSCRIPT_DESC, "Intimation of conference call") is False


class TestSplitSections:
    def test_splits_on_qa_marker(self) -> None:
        text = "Opening remarks by CEO.\n\nQ&A Session\nQuestion from analyst."
        result = split_sections(text)
        assert "management" in result or "qa" in result
        assert "qa" in result
        assert "Question from analyst" in result["qa"]

    def test_splits_on_moderator(self) -> None:
        text = "CEO speech here.\n\nModerator: We open for questions."
        result = split_sections(text)
        assert "qa" in result

    def test_management_only(self) -> None:
        text = "Management Remarks\nThe company performed well this quarter."
        result = split_sections(text)
        assert "management" in result

    def test_full_when_no_markers(self) -> None:
        text = "Some random text without any markers."
        result = split_sections(text)
        assert "full" in result
        assert result["full"] == text

    def test_question_and_answer_session(self) -> None:
        text = "Intro text.\n\nQuestion and Answer Session\nQ1: What about growth?"
        result = split_sections(text)
        assert "qa" in result

    def test_open_floor_for_questions(self) -> None:
        text = "CEO presentation.\n\nWe will now open the floor for questions."
        result = split_sections(text)
        assert "qa" in result

    def test_empty_text(self) -> None:
        result = split_sections("")
        assert "full" in result


class TestRowToTranscript:
    def _make_row(self, **overrides: str | None) -> dict[str, str | None]:
        base: dict[str, str | None] = {
            "desc": TRANSCRIPT_DESC,
            "an_dt": "18-Jun-2026 10:30:00",
            "symbol": "TCS",
            "attchmntText": "Transcript of earnings call",
            "attchmntFile": "https://nsearchives.nseindia.com/corporate/TCS_transcript.pdf",
            "sort_date": None,
            "sm_name": None,
        }
        base.update(overrides)
        return base

    def test_converts_valid_row(self) -> None:
        tr = row_to_transcript(self._make_row())
        assert tr is not None
        assert tr["symbol"] == "TCS.NS"
        assert tr["fiscal_quarter"] == "Q4FY26"
        assert isinstance(tr["filed_at"], datetime)
        assert tr["filed_at"].tzinfo == IST
        assert tr["_pdf_url"] is not None
        assert tr["text"] is None
        assert tr["sections"] is None

    def test_returns_none_for_non_transcript(self) -> None:
        assert (
            row_to_transcript(
                self._make_row(desc="Board Meeting", attchmntText="Quarterly results")
            )
            is None
        )

    def test_returns_none_for_bad_date(self) -> None:
        assert row_to_transcript(self._make_row(an_dt="bad")) is None

    def test_returns_none_for_empty_symbol(self) -> None:
        assert row_to_transcript(self._make_row(symbol="")) is None

    def test_no_pdf_url(self) -> None:
        tr = row_to_transcript(self._make_row(attchmntFile=""))
        assert tr is not None
        assert tr["_pdf_url"] is None

    def test_schedule_headline_filtered(self) -> None:
        assert row_to_transcript(self._make_row(attchmntText="Schedule of analyst meet")) is None

    def test_transcript_in_general_updates(self) -> None:
        tr = row_to_transcript(
            self._make_row(desc="General Updates", attchmntText="Transcript of AGM")
        )
        assert tr is not None
