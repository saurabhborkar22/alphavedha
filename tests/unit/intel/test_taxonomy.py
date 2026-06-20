"""Tests for event taxonomy and extraction schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alphavedha.intel.extraction.schemas import (
    DisclosureExtraction,
    ExtractedNumbers,
    TranscriptDelta,
    TriageResult,
)
from alphavedha.intel.extraction.taxonomy import (
    ALWAYS_EXTRACT_PATTERNS,
    BOILERPLATE_CATEGORIES,
    BOILERPLATE_HEADLINE_PATTERNS,
    EVENT_CATALOG,
    RED_FLAG_TYPES,
    EventMeta,
    EventType,
)


class TestEventType:
    def test_has_21_types(self) -> None:
        assert len(EventType) == 21

    def test_all_string_enums(self) -> None:
        for et in EventType:
            assert isinstance(et.value, str)

    def test_catalog_covers_all_types(self) -> None:
        for et in EventType:
            assert et in EVENT_CATALOG, f"{et} missing from EVENT_CATALOG"

    def test_catalog_has_no_extras(self) -> None:
        for et in EVENT_CATALOG:
            assert et in EventType, f"{et} in catalog but not in EventType"

    def test_red_flag_types(self) -> None:
        expected = {
            EventType.RATING_DOWNGRADE,
            EventType.PLEDGE_INCREASE,
            EventType.AUDITOR_RESIGNATION,
            EventType.KMP_RESIGNATION,
            EventType.DEFAULT_OR_DELAY,
            EventType.SURVEILLANCE_ACTION,
        }
        assert expected == RED_FLAG_TYPES

    def test_event_meta_fields(self) -> None:
        meta = EVENT_CATALOG[EventType.ORDER_WIN]
        assert isinstance(meta, EventMeta)
        assert meta.default_direction == 1
        assert meta.is_red_flag is False
        assert len(meta.description) > 0

    def test_boilerplate_categories_non_empty(self) -> None:
        assert len(BOILERPLATE_CATEGORIES) >= 25
        assert "Trading Window" in BOILERPLATE_CATEGORIES
        assert "ESOP/ESOS" in BOILERPLATE_CATEGORIES
        assert "Board Meeting - Intimation" in BOILERPLATE_CATEGORIES
        assert "Investor Presentation" in BOILERPLATE_CATEGORIES

    def test_boilerplate_headline_patterns_non_empty(self) -> None:
        assert len(BOILERPLATE_HEADLINE_PATTERNS) > 0

    def test_always_extract_patterns_non_empty(self) -> None:
        assert len(ALWAYS_EXTRACT_PATTERNS) > 0

    def test_enum_value_lowercase_snake(self) -> None:
        for et in EventType:
            assert et.value == et.value.lower()
            assert " " not in et.value


class TestDisclosureExtraction:
    """Validate extraction schema against 10 hand-written examples."""

    def test_order_win(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.ORDER_WIN,
            direction=1,
            materiality=7,
            confidence=0.95,
            summary="TCS wins $500M deal from UK-based financial services firm",
            red_flags=[],
            numbers=ExtractedNumbers(order_value_cr=4200),
        )
        assert e.event_type == EventType.ORDER_WIN
        assert e.direction == 1
        assert e.numbers.order_value_cr == 4200

    def test_rating_downgrade(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.RATING_DOWNGRADE,
            direction=-1,
            materiality=8,
            confidence=0.9,
            summary="ICRA downgrades long-term rating to BB+ from A- on weakening cash flows",
            red_flags=["two-notch downgrade", "cash flow deterioration"],
            numbers=ExtractedNumbers(rating_notches=-2),
        )
        assert e.event_type in RED_FLAG_TYPES
        assert len(e.red_flags) == 2

    def test_auditor_resignation(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.AUDITOR_RESIGNATION,
            direction=-1,
            materiality=9,
            confidence=0.98,
            summary="Statutory auditor resigns citing management non-cooperation",
            red_flags=["auditor resigned mid-term", "cited management non-cooperation"],
        )
        assert e.event_type in RED_FLAG_TYPES
        assert e.materiality == 9

    def test_pledge_increase(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.PLEDGE_INCREASE,
            direction=-1,
            materiality=6,
            confidence=0.85,
            summary="Promoter pledge rises to 62% of total holding from 48%",
            red_flags=["promoter pledge above 50%"],
            numbers=ExtractedNumbers(pledge_pct=62.0),
        )
        assert e.numbers.pledge_pct == 62.0

    def test_insider_buy(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.INSIDER_BUY,
            direction=1,
            materiality=5,
            confidence=0.9,
            summary="MD purchases 50,000 shares worth Rs 2.3 Cr from open market",
            red_flags=[],
            numbers=ExtractedNumbers(insider_value_cr=2.3),
        )
        assert e.direction == 1

    def test_capacity_expansion(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.CAPACITY_EXPANSION,
            direction=1,
            materiality=6,
            confidence=0.88,
            summary="Board approves 30% capacity expansion at Gujarat plant, Rs 450 Cr capex",
            red_flags=[],
            numbers=ExtractedNumbers(capacity_pct_change=30.0, deal_value_cr=450),
        )
        assert e.numbers.capacity_pct_change == 30.0

    def test_default_or_delay(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.DEFAULT_OR_DELAY,
            direction=-1,
            materiality=10,
            confidence=0.99,
            summary="Company defaults on Rs 120 Cr term loan interest payment due Jun 15",
            red_flags=["loan default", "interest payment missed"],
        )
        assert e.materiality == 10
        assert e.event_type in RED_FLAG_TYPES

    def test_results_guidance(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.RESULTS_GUIDANCE,
            direction=1,
            materiality=7,
            confidence=0.85,
            summary="Q4 revenue up 18% YoY, PAT up 24%, margin expansion to 22.5%",
            red_flags=[],
            numbers=ExtractedNumbers(revenue_cr=5800, profit_cr=1300, margin_pct=22.5),
        )
        assert e.numbers.margin_pct == 22.5

    def test_kmp_resignation(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.KMP_RESIGNATION,
            direction=-1,
            materiality=7,
            confidence=0.92,
            summary="CFO resigns effective immediately citing personal reasons",
            red_flags=["CFO resignation", "immediate effect"],
        )
        assert e.event_type in RED_FLAG_TYPES

    def test_dividend_buyback(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.DIVIDEND_BUYBACK,
            direction=1,
            materiality=4,
            confidence=0.95,
            summary="Board recommends final dividend of Rs 28 per share for FY26",
            red_flags=[],
            numbers=ExtractedNumbers(dividend_per_share=28.0),
        )
        assert e.numbers.dividend_per_share == 28.0


class TestExtractionValidation:
    def test_direction_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            DisclosureExtraction(
                event_type=EventType.OTHER,
                direction=2,
                materiality=5,
                confidence=0.5,
                summary="test",
            )

    def test_materiality_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            DisclosureExtraction(
                event_type=EventType.OTHER,
                direction=0,
                materiality=11,
                confidence=0.5,
                summary="test",
            )

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            DisclosureExtraction(
                event_type=EventType.OTHER,
                direction=0,
                materiality=5,
                confidence=1.5,
                summary="test",
            )

    def test_summary_max_length(self) -> None:
        with pytest.raises(ValidationError):
            DisclosureExtraction(
                event_type=EventType.OTHER,
                direction=0,
                materiality=5,
                confidence=0.5,
                summary="x" * 201,
            )

    def test_defaults_for_optional_fields(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.OTHER,
            direction=0,
            materiality=1,
            confidence=0.5,
            summary="Minor update",
        )
        assert e.red_flags == []
        assert e.numbers.order_value_cr is None

    def test_json_round_trip(self) -> None:
        e = DisclosureExtraction(
            event_type=EventType.M_AND_A,
            direction=1,
            materiality=8,
            confidence=0.88,
            summary="Acquires 51% stake in XYZ Pharma for Rs 800 Cr",
            red_flags=[],
            numbers=ExtractedNumbers(deal_value_cr=800),
        )
        json_str = e.model_dump_json()
        restored = DisclosureExtraction.model_validate_json(json_str)
        assert restored == e

    def test_invalid_event_type(self) -> None:
        with pytest.raises(ValidationError):
            DisclosureExtraction(
                event_type="nonexistent_type",  # type: ignore[arg-type]
                direction=0,
                materiality=5,
                confidence=0.5,
                summary="test",
            )


class TestTriageResult:
    def test_relevant(self) -> None:
        t = TriageResult(
            is_relevant=True,
            category=EventType.ORDER_WIN,
            reason="Order win announcement with disclosed value",
        )
        assert t.is_relevant is True
        assert t.category == EventType.ORDER_WIN

    def test_boilerplate(self) -> None:
        t = TriageResult(
            is_relevant=False,
            reason="Routine trading window closure notice",
        )
        assert t.is_relevant is False
        assert t.category is None

    def test_reason_max_length(self) -> None:
        with pytest.raises(ValidationError):
            TriageResult(is_relevant=True, reason="x" * 151)


class TestTranscriptDelta:
    def test_positive_delta(self) -> None:
        td = TranscriptDelta(
            guidance_delta=2,
            tone_delta=1,
            dropped_commitments=[],
            new_commitments=["Target 25% margin by Q2"],
            evasiveness_score=2,
            summary="Management raised guidance across segments",
        )
        assert td.guidance_delta == 2
        assert len(td.new_commitments) == 1

    def test_negative_delta(self) -> None:
        td = TranscriptDelta(
            guidance_delta=-1,
            tone_delta=-2,
            dropped_commitments=["Rs 5000 Cr revenue target", "20% EBITDA margin"],
            new_commitments=[],
            evasiveness_score=7,
            summary="Management dropped prior guidance, avoided margin questions",
        )
        assert len(td.dropped_commitments) == 2
        assert td.evasiveness_score == 7

    def test_guidance_delta_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptDelta(
                guidance_delta=3,
                tone_delta=0,
                evasiveness_score=5,
                summary="test",
            )

    def test_evasiveness_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptDelta(
                guidance_delta=0,
                tone_delta=0,
                evasiveness_score=11,
                summary="test",
            )
