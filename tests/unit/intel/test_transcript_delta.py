"""Tests for transcript delta comparison."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alphavedha.intel.extraction.llm import LLMProvider
from alphavedha.intel.extraction.transcript_delta import (
    _coerce_delta_output,
    _truncate_text,
    build_delta_prompt,
    compare_one,
    process_symbol_deltas,
    run_transcript_deltas,
)


def _mock_provider(return_value: dict[str, Any] | None = None) -> LLMProvider:
    p = MagicMock(spec=LLMProvider)
    p.name = "mock/test"
    p.extract_json.return_value = return_value
    return p


VALID_DELTA = {
    "guidance_delta": 1,
    "tone_delta": 0,
    "dropped_commitments": [],
    "new_commitments": ["Revenue target 20% YoY"],
    "evasiveness_score": 2,
    "summary": "Management raised guidance slightly",
}


class TestTruncateText:
    def test_none_returns_placeholder(self) -> None:
        assert "no transcript" in _truncate_text(None)

    def test_empty_returns_placeholder(self) -> None:
        assert "no transcript" in _truncate_text("")

    def test_short_text_unchanged(self) -> None:
        assert _truncate_text("Hello") == "Hello"

    def test_long_text_truncated(self) -> None:
        long = "x" * 20000
        result = _truncate_text(long, max_chars=100)
        assert len(result) == 100


class TestBuildDeltaPrompt:
    def test_contains_both_quarters(self) -> None:
        prompt = build_delta_prompt("TCS.NS", "Q1FY26", "Q4FY25", "current", "previous")
        assert "Q1FY26" in prompt
        assert "Q4FY25" in prompt
        assert "TCS.NS" in prompt

    def test_contains_comparison_instructions(self) -> None:
        prompt = build_delta_prompt("TCS.NS", "Q1", "Q4", "text1", "text2")
        assert "guidance" in prompt.lower()
        assert "evasive" in prompt.lower()


class TestCoerceDeltaOutput:
    def test_adds_missing_fields(self) -> None:
        raw: dict[str, Any] = {"guidance_delta": 0, "tone_delta": 0, "evasiveness_score": 0}
        result = _coerce_delta_output(raw)
        assert "summary" in result
        assert "dropped_commitments" in result
        assert "new_commitments" in result

    def test_coerces_string_ints(self) -> None:
        raw: dict[str, Any] = {
            "guidance_delta": "1",
            "tone_delta": "-1",
            "evasiveness_score": "5",
            "summary": "test",
        }
        result = _coerce_delta_output(raw)
        assert result["guidance_delta"] == 1
        assert result["tone_delta"] == -1
        assert result["evasiveness_score"] == 5


class TestCompareOne:
    def test_returns_delta_on_valid(self) -> None:
        provider = _mock_provider(VALID_DELTA)
        result = compare_one(provider, "TCS.NS", "Q1FY26", "Q4FY25", "text1", "text2")
        assert result is not None
        assert result.guidance_delta == 1
        assert result.tone_delta == 0
        assert "Revenue target" in result.new_commitments[0]

    def test_returns_none_on_failure(self) -> None:
        provider = _mock_provider(None)
        result = compare_one(provider, "TCS.NS", "Q1FY26", "Q4FY25", "text1", "text2")
        assert result is None

    def test_returns_none_on_invalid_response(self) -> None:
        provider = _mock_provider({"bad": "data"})
        result = compare_one(provider, "TCS.NS", "Q1FY26", "Q4FY25", "text1", "text2")
        assert result is None

    def test_handles_none_text(self) -> None:
        provider = _mock_provider(VALID_DELTA)
        result = compare_one(provider, "TCS.NS", "Q1FY26", "Q4FY25", None, None)
        assert result is not None


class TestProcessSymbolDeltas:
    @pytest.mark.asyncio
    async def test_no_pairs(self) -> None:
        with patch(
            "alphavedha.intel.extraction.transcript_delta.load_transcript_pairs",
            new_callable=AsyncMock,
            return_value=[],
        ):
            provider = _mock_provider()
            events = await process_symbol_deltas("TCS.NS", provider)
            assert events == []

    @pytest.mark.asyncio
    async def test_produces_events(self) -> None:
        pairs = [
            (
                {"id": 10, "symbol": "TCS.NS", "fiscal_quarter": "Q1FY26", "text": "current"},
                {"id": 9, "symbol": "TCS.NS", "fiscal_quarter": "Q4FY25", "text": "previous"},
            )
        ]
        provider = _mock_provider(VALID_DELTA)

        with patch(
            "alphavedha.intel.extraction.transcript_delta.load_transcript_pairs",
            new_callable=AsyncMock,
            return_value=pairs,
        ):
            events = await process_symbol_deltas("TCS.NS", provider)
            assert len(events) == 1
            assert events[0]["event_type"] == "results_guidance"
            assert events[0]["disclosure_id"] == 10
            assert events[0]["direction"] == 1

    @pytest.mark.asyncio
    async def test_bearish_delta(self) -> None:
        bearish = {
            "guidance_delta": -2,
            "tone_delta": -1,
            "dropped_commitments": ["Revenue growth target"],
            "new_commitments": [],
            "evasiveness_score": 8,
            "summary": "Guidance cut with evasive Q&A",
        }
        pairs = [
            (
                {"id": 20, "symbol": "INFY.NS", "fiscal_quarter": "Q2FY26", "text": "bad quarter"},
                {"id": 19, "symbol": "INFY.NS", "fiscal_quarter": "Q1FY26", "text": "good quarter"},
            )
        ]
        provider = _mock_provider(bearish)

        with patch(
            "alphavedha.intel.extraction.transcript_delta.load_transcript_pairs",
            new_callable=AsyncMock,
            return_value=pairs,
        ):
            events = await process_symbol_deltas("INFY.NS", provider)
            assert len(events) == 1
            assert events[0]["direction"] == -1
            assert events[0]["red_flags"] is not None
            assert "evasiveness" in events[0]["red_flags"][0]


class TestRunTranscriptDeltas:
    @pytest.mark.asyncio
    async def test_empty_transcripts(self) -> None:
        import pandas as pd

        with patch(
            "alphavedha.intel.store.load_transcripts",
            new_callable=AsyncMock,
            return_value=pd.DataFrame(),
        ):
            provider = _mock_provider()
            result = await run_transcript_deltas(provider=provider)
            assert result["status"] == "empty"

    @pytest.mark.asyncio
    async def test_processes_symbols(self) -> None:
        provider = _mock_provider(VALID_DELTA)
        pairs = [
            (
                {"id": 10, "symbol": "TCS.NS", "fiscal_quarter": "Q1FY26", "text": "current"},
                {"id": 9, "symbol": "TCS.NS", "fiscal_quarter": "Q4FY25", "text": "previous"},
            )
        ]

        with (
            patch(
                "alphavedha.intel.extraction.transcript_delta.load_transcript_pairs",
                new_callable=AsyncMock,
                return_value=pairs,
            ),
            patch(
                "alphavedha.intel.extraction.transcript_delta.store_disclosure_events",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            result = await run_transcript_deltas(symbols=["TCS.NS"], provider=provider)
            assert result["status"] == "ok"
            assert result["symbols_processed"] == 1
            assert result["events_stored"] == 1
