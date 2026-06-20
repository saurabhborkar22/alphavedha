"""Tests for the guidance delta signal generator."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from alphavedha.intel.signals.guidance_delta import (
    STRATEGY_NAME,
    generate_guidance_signals,
)

IST = ZoneInfo("Asia/Kolkata")


def _event(
    symbol: str = "TCS.NS",
    direction: int = 1,
    materiality: int = 4,
    confidence: float = 0.7,
    extracted_at: datetime | None = None,
    summary: str = "Guidance improved significantly this quarter",
) -> dict[str, object]:
    if extracted_at is None:
        extracted_at = datetime(2026, 6, 19, 14, 0, tzinfo=IST)
    return {
        "symbol": symbol,
        "event_type": "results_guidance",
        "direction": direction,
        "materiality": materiality,
        "confidence": confidence,
        "extracted_at": extracted_at,
        "summary": summary,
        "red_flags": None,
    }


class TestGenerateGuidanceSignals:
    def test_positive_guidance_generates_long_signal(self) -> None:
        events = [_event(direction=1, summary="Guidance improved")]
        signals = generate_guidance_signals(events, date(2026, 6, 20))
        assert len(signals) == 1
        assert signals[0].direction == 1
        assert signals[0].short_constrained is False

    def test_negative_guidance_generates_short_constrained(self) -> None:
        events = [_event(direction=-1, summary="Guidance deteriorated")]
        signals = generate_guidance_signals(events, date(2026, 6, 20))
        assert len(signals) == 1
        assert signals[0].direction == -1
        assert signals[0].short_constrained is True

    def test_look_ahead_prevention(self) -> None:
        future_event = _event(extracted_at=datetime(2026, 6, 20, 10, 0, tzinfo=IST))
        signals = generate_guidance_signals([future_event], date(2026, 6, 20))
        assert len(signals) == 0

    def test_stale_events_excluded(self) -> None:
        old = _event(extracted_at=datetime(2026, 5, 1, 10, 0, tzinfo=IST))
        signals = generate_guidance_signals([old], date(2026, 6, 20))
        assert len(signals) == 0

    def test_non_guidance_events_excluded(self) -> None:
        ev = _event()
        ev["event_type"] = "order_win"
        signals = generate_guidance_signals([ev], date(2026, 6, 20))
        assert len(signals) == 0

    def test_zero_direction_excluded(self) -> None:
        events = [_event(direction=0, summary="Unchanged guidance")]
        signals = generate_guidance_signals(events, date(2026, 6, 20))
        assert len(signals) == 0

    def test_low_materiality_filtered(self) -> None:
        events = [_event(materiality=1)]
        signals = generate_guidance_signals(events, date(2026, 6, 20))
        assert len(signals) == 0

    def test_multiple_events_same_symbol_picks_highest_materiality(self) -> None:
        events = [
            _event(materiality=3, summary="Small change"),
            _event(materiality=5, summary="Guidance improved significantly"),
        ]
        signals = generate_guidance_signals(events, date(2026, 6, 20))
        assert len(signals) == 1
        assert signals[0].materiality == 5

    def test_evasiveness_flag_affects_tone(self) -> None:
        ev = _event(direction=-1, summary="Deteriorated")
        ev["red_flags"] = ["evasiveness_score=8"]
        signals = generate_guidance_signals([ev], date(2026, 6, 20))
        assert len(signals) == 1
        assert signals[0].tone_delta == -1

    def test_empty_events_returns_empty(self) -> None:
        assert generate_guidance_signals([], date(2026, 6, 20)) == []


class TestConstants:
    def test_strategy_name(self) -> None:
        assert STRATEGY_NAME == "guidance_delta_v1"
