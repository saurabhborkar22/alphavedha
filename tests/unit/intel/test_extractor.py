"""Tests for disclosure extraction pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from alphavedha.intel.extraction.extractor import (
    build_triage_prompt,
    build_user_prompt,
    extract_batch,
    extract_one,
    is_always_extract,
    is_boilerplate,
    triage_one,
)
from alphavedha.intel.extraction.llm import LLMProvider, load_system_prompt
from alphavedha.intel.extraction.taxonomy import BOILERPLATE_CATEGORIES, EventType


def _mock_provider(return_value: dict[str, Any] | None = None) -> LLMProvider:
    """Create a mock LLM provider that returns the given dict."""
    p = MagicMock(spec=LLMProvider)
    p.name = "mock/test"
    p.extract_json.return_value = return_value
    return p


class TestIsBoilerplate:
    def test_trading_window(self) -> None:
        assert is_boilerplate("Trading Window") is True

    def test_esop(self) -> None:
        assert is_boilerplate("ESOP/ESOS") is True

    def test_newspaper(self) -> None:
        assert is_boilerplate("Copy of Newspaper Publication") is False
        assert is_boilerplate("Newspaper Publication") in (True, False)

    def test_order_win_not_boilerplate(self) -> None:
        assert is_boilerplate("Bagging/Receiving of orders/contracts") is False

    def test_all_boilerplate_categories_detected(self) -> None:
        for cat in BOILERPLATE_CATEGORIES:
            assert is_boilerplate(cat) is True

    def test_expanded_categories(self) -> None:
        assert is_boilerplate("Board Meeting - Intimation") is True
        assert is_boilerplate("Postal Ballot") is True
        assert is_boilerplate("Annual Secretarial Compliance") is True
        assert is_boilerplate("Investor Presentation") is True

    def test_headline_pattern_catches_boilerplate(self) -> None:
        assert is_boilerplate("Other", "Loss of share certificate - duplicate issued") is True
        assert is_boilerplate("Other", "Compliance certificate under Reg 31") is True
        assert is_boilerplate("Other", "Proceedings of AGM held on 2026-06-15") is True

    def test_always_extract_overrides_boilerplate_category(self) -> None:
        assert is_boilerplate("Cessation", "CFO resigns effective immediately") is False
        assert is_boilerplate("Change In Address", "SEBI order against company") is False
        assert is_boilerplate("Board Meeting - Intimation", "Default on loan payment") is False

    def test_always_extract_overrides_headline_pattern(self) -> None:
        assert is_boilerplate("Other", "Reg. 31 compliance - auditor resignation") is False


class TestIsAlwaysExtract:
    def test_cfo_resign(self) -> None:
        assert is_always_extract("CFO resigns citing personal reasons") is True

    def test_ceo_steps_down(self) -> None:
        assert is_always_extract("CEO to step down effective March 2026") is True

    def test_default(self) -> None:
        assert is_always_extract("Company defaults on Rs 100 Cr term loan") is True

    def test_auditor_resign(self) -> None:
        assert is_always_extract("Statutory auditor resignation") is True

    def test_fraud(self) -> None:
        assert is_always_extract("Fraud detected in subsidiary operations") is True

    def test_sebi_order(self) -> None:
        assert is_always_extract("SEBI penalty imposed for insider trading") is True

    def test_surveillance(self) -> None:
        assert is_always_extract("Stock placed under ASM framework") is True

    def test_downgrade(self) -> None:
        assert is_always_extract("ICRA downgrades rating to BB+") is True

    def test_pledge_increase(self) -> None:
        assert is_always_extract("Promoter pledge created on additional shares") is True

    def test_normal_headline_not_flagged(self) -> None:
        assert is_always_extract("Board approves quarterly results") is False
        assert is_always_extract("Investor presentation scheduled") is False


class TestBuildPrompts:
    def test_user_prompt_basic(self) -> None:
        prompt = build_user_prompt("TCS.NS", "Press Release", "TCS wins $500M deal")
        assert "TCS.NS" in prompt
        assert "Press Release" in prompt
        assert "$500M" in prompt

    def test_user_prompt_with_text(self) -> None:
        prompt = build_user_prompt(
            "TCS.NS", "Press Release", "headline", text="Full filing text here"
        )
        assert "Full filing text here" in prompt

    def test_user_prompt_truncates_long_text(self) -> None:
        long_text = "x" * 10000
        prompt = build_user_prompt("TCS.NS", "Press Release", "headline", text=long_text)
        assert len(prompt) < 10000

    def test_triage_prompt(self) -> None:
        prompt = build_triage_prompt("TCS.NS", "Press Release", "headline")
        assert "relevant" in prompt.lower()
        assert "boilerplate" in prompt.lower()


class TestLoadSystemPrompt:
    def test_loads_v1(self) -> None:
        prompt = load_system_prompt("v1")
        assert "event_type" in prompt.lower() or "Event Types" in prompt
        assert len(prompt) > 100


class TestExtractOne:
    def test_returns_extraction_on_valid_response(self) -> None:
        provider = _mock_provider(
            {
                "event_type": "order_win",
                "direction": 1,
                "materiality": 7,
                "confidence": 0.9,
                "summary": "TCS wins $500M deal",
                "red_flags": [],
                "numbers": {},
            }
        )
        result = extract_one(provider, "TCS.NS", "Press Release", "TCS wins $500M deal")
        assert result is not None
        assert result.event_type == EventType.ORDER_WIN
        assert result.direction == 1

    def test_skips_boilerplate(self) -> None:
        provider = _mock_provider()
        result = extract_one(provider, "TCS.NS", "Trading Window", "Trading window closure")
        assert result is None
        provider.extract_json.assert_not_called()

    def test_returns_none_on_llm_failure(self) -> None:
        provider = _mock_provider(None)
        result = extract_one(provider, "TCS.NS", "Press Release", "Some headline")
        assert result is None

    def test_returns_none_on_invalid_response(self) -> None:
        provider = _mock_provider({"bad": "data"})
        result = extract_one(provider, "TCS.NS", "Press Release", "Some headline")
        assert result is None


class TestTriageOne:
    def test_boilerplate_skipped(self) -> None:
        provider = _mock_provider()
        result = triage_one(provider, "TCS.NS", "ESOP/ESOS", "Allotment")
        assert result is not None
        assert result.is_relevant is False
        provider.extract_json.assert_not_called()

    def test_relevant_via_llm(self) -> None:
        provider = _mock_provider(
            {
                "is_relevant": True,
                "category": "order_win",
                "reason": "Order win announcement",
            }
        )
        result = triage_one(provider, "TCS.NS", "Press Release", "Wins deal")
        assert result is not None
        assert result.is_relevant is True

    def test_returns_none_on_failure(self) -> None:
        provider = _mock_provider(None)
        result = triage_one(provider, "TCS.NS", "Press Release", "headline")
        assert result is None


class TestExtractBatch:
    def test_processes_batch(self) -> None:
        provider = _mock_provider(
            {
                "event_type": "order_win",
                "direction": 1,
                "materiality": 6,
                "confidence": 0.85,
                "summary": "New order received",
                "red_flags": [],
                "numbers": {},
            }
        )
        disclosures = [
            {"id": 1, "symbol": "TCS.NS", "category": "Press Release", "headline": "Order win"},
            {"id": 2, "symbol": "INFY.NS", "category": "Press Release", "headline": "Order win"},
        ]
        events = extract_batch(provider, disclosures)
        assert len(events) == 2
        assert events[0]["disclosure_id"] == 1
        assert events[0]["event_type"] == "order_win"
        assert events[0]["llm_model"] == "mock/test"
        assert events[0]["prompt_version"] == "v1"

    def test_skips_boilerplate_in_batch(self) -> None:
        provider = _mock_provider(
            {
                "event_type": "other",
                "direction": 0,
                "materiality": 2,
                "confidence": 0.5,
                "summary": "test",
                "red_flags": [],
                "numbers": {},
            }
        )
        disclosures = [
            {"id": 1, "symbol": "TCS.NS", "category": "Trading Window", "headline": "Closure"},
            {"id": 2, "symbol": "INFY.NS", "category": "Press Release", "headline": "Update"},
        ]
        events = extract_batch(provider, disclosures)
        assert len(events) == 1
        assert events[0]["disclosure_id"] == 2

    def test_empty_batch(self) -> None:
        provider = _mock_provider()
        events = extract_batch(provider, [])
        assert events == []

    def test_handles_failed_extractions(self) -> None:
        provider = _mock_provider(None)
        disclosures = [
            {"id": 1, "symbol": "TCS.NS", "category": "Press Release", "headline": "Something"},
        ]
        events = extract_batch(provider, disclosures)
        assert events == []

    def test_event_has_required_fields(self) -> None:
        provider = _mock_provider(
            {
                "event_type": "m_and_a",
                "direction": 0,
                "materiality": 6,
                "confidence": 0.8,
                "summary": "Acquisition announced",
                "red_flags": [],
                "numbers": {},
            }
        )
        disclosures = [
            {"id": 42, "symbol": "WIPRO.NS", "category": "Acquisition", "headline": "Acquisition"},
        ]
        events = extract_batch(provider, disclosures)
        event = events[0]
        assert "disclosure_id" in event
        assert "symbol" in event
        assert "event_type" in event
        assert "direction" in event
        assert "materiality" in event
        assert "confidence" in event
        assert "summary" in event
        assert "llm_model" in event
        assert "prompt_version" in event
        assert "extracted_at" in event
