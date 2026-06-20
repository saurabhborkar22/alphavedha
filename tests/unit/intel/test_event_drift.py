"""Tests for the event drift signal generator."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from alphavedha.intel.signals.event_drift import (
    MAX_POSITIONS_PER_DAY,
    MIN_MATERIALITY,
    STRATEGY_NAME,
    EventDriftSignal,
    generate_signals,
)

IST = ZoneInfo("Asia/Kolkata")


def _event(
    symbol: str = "TCS.NS",
    event_type: str = "order_win",
    direction: int = 1,
    materiality: int = 4,
    confidence: float = 0.8,
    extracted_at: datetime | None = None,
    summary: str = "Major order win",
) -> dict[str, object]:
    if extracted_at is None:
        extracted_at = datetime(2026, 6, 19, 14, 0, tzinfo=IST)
    return {
        "symbol": symbol,
        "event_type": event_type,
        "direction": direction,
        "materiality": materiality,
        "confidence": confidence,
        "extracted_at": extracted_at,
        "summary": summary,
    }


class TestGenerateSignals:
    def test_basic_positive_event_generates_long_signal(self) -> None:
        events = [_event(direction=1, materiality=4, confidence=0.8)]
        signals = generate_signals(events, date(2026, 6, 20))
        assert len(signals) == 1
        assert signals[0].direction == 1
        assert signals[0].short_constrained is False
        assert signals[0].symbol == "TCS.NS"

    def test_negative_event_generates_short_constrained_signal(self) -> None:
        events = [_event(direction=-1, event_type="rating_downgrade", materiality=5)]
        signals = generate_signals(events, date(2026, 6, 20))
        assert len(signals) == 1
        assert signals[0].direction == -1
        assert signals[0].short_constrained is True

    def test_look_ahead_prevention_same_day_event_excluded(self) -> None:
        """Events filed on or after signal_date must NOT generate signals."""
        future_event = _event(
            extracted_at=datetime(2026, 6, 20, 10, 0, tzinfo=IST),
        )
        signals = generate_signals([future_event], date(2026, 6, 20))
        assert len(signals) == 0

    def test_stale_events_excluded(self) -> None:
        """Events older than 3 days are not eligible."""
        old_event = _event(
            extracted_at=datetime(2026, 6, 16, 10, 0, tzinfo=IST),
        )
        signals = generate_signals([old_event], date(2026, 6, 20))
        assert len(signals) == 0

    def test_low_materiality_filtered(self) -> None:
        events = [_event(materiality=MIN_MATERIALITY - 1)]
        signals = generate_signals(events, date(2026, 6, 20))
        assert len(signals) == 0

    def test_multiple_events_same_symbol_picks_highest_materiality(self) -> None:
        events = [
            _event(materiality=3, confidence=0.9, summary="Small order"),
            _event(materiality=5, confidence=0.7, summary="Big order"),
        ]
        signals = generate_signals(events, date(2026, 6, 20))
        assert len(signals) == 1
        assert signals[0].materiality == 5

    def test_max_positions_cap(self) -> None:
        events = [
            _event(symbol=f"SYM{i}.NS", materiality=4) for i in range(MAX_POSITIONS_PER_DAY + 5)
        ]
        signals = generate_signals(events, date(2026, 6, 20))
        assert len(signals) == MAX_POSITIONS_PER_DAY

    def test_zero_direction_uses_taxonomy_default(self) -> None:
        events = [_event(direction=0, event_type="order_win", materiality=4)]
        signals = generate_signals(events, date(2026, 6, 20))
        assert len(signals) == 1
        assert signals[0].direction == 1

    def test_zero_direction_neutral_type_excluded(self) -> None:
        """Event types with neutral default direction and direction=0 produce no signal."""
        events = [_event(direction=0, event_type="fund_raise", materiality=4)]
        signals = generate_signals(events, date(2026, 6, 20))
        assert len(signals) == 0

    def test_empty_events_returns_empty(self) -> None:
        assert generate_signals([], date(2026, 6, 20)) == []

    def test_strategy_name_constant(self) -> None:
        assert STRATEGY_NAME == "event_drift_v1"

    def test_signal_dataclass_fields(self) -> None:
        sig = EventDriftSignal(
            symbol="TCS.NS",
            direction=1,
            confidence=0.72,
            materiality=4,
            event_type="order_win",
            short_constrained=False,
            event_summary="Major order win",
        )
        assert sig.symbol == "TCS.NS"
        assert sig.confidence == 0.72
