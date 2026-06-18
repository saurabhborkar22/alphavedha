"""Tests for prediction hasher — deterministic SHA-256 of daily predictions."""

from __future__ import annotations

from datetime import date

from alphavedha.verification.hasher import (
    canonical_payload,
    hash_daily_trades,
    sha256_hex,
)


def _sample_trade(
    symbol: str = "TCS.NS",
    direction: int = 1,
    confidence: float = 0.75,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "prediction_date": date(2026, 6, 18),
        "predicted_direction": direction,
        "predicted_magnitude": 0.02,
        "confidence": confidence,
        "is_tradeable": True,
        "model_version": "v1.0",
        "regime": "bull",
        "entry_price": 3500.0,
        "exit_price": None,
    }


class TestCanonicalPayload:
    def test_deterministic_across_row_order(self) -> None:
        trades_a = [_sample_trade("TCS.NS"), _sample_trade("INFY.NS")]
        trades_b = [_sample_trade("INFY.NS"), _sample_trade("TCS.NS")]
        assert canonical_payload(trades_a) == canonical_payload(trades_b)

    def test_deterministic_across_dict_key_order(self) -> None:
        trade_a = _sample_trade()
        trade_b = dict(reversed(list(_sample_trade().items())))
        assert canonical_payload([trade_a]) == canonical_payload([trade_b])

    def test_changing_field_changes_payload(self) -> None:
        trade_a = _sample_trade(confidence=0.75)
        trade_b = _sample_trade(confidence=0.80)
        assert canonical_payload([trade_a]) != canonical_payload([trade_b])

    def test_empty_day_produces_valid_payload(self) -> None:
        payload = canonical_payload([])
        assert payload == b"[]"

    def test_extra_fields_ignored(self) -> None:
        trade = _sample_trade()
        trade["extra_field"] = "should_be_ignored"
        payload_with = canonical_payload([trade])

        trade_clean = _sample_trade()
        payload_without = canonical_payload([trade_clean])
        assert payload_with == payload_without

    def test_date_objects_serialized_as_iso(self) -> None:
        trade = _sample_trade()
        payload = canonical_payload([trade])
        assert b"2026-06-18" in payload

    def test_float_precision_normalized(self) -> None:
        trade_a = _sample_trade()
        trade_a["confidence"] = 0.750000001
        trade_b = _sample_trade()
        trade_b["confidence"] = 0.75
        assert canonical_payload([trade_a]) == canonical_payload([trade_b])


class TestSha256Hex:
    def test_known_hash(self) -> None:
        digest = sha256_hex(b"hello")
        assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_empty_payload(self) -> None:
        digest = sha256_hex(b"")
        assert len(digest) == 64

    def test_different_inputs_different_hashes(self) -> None:
        assert sha256_hex(b"a") != sha256_hex(b"b")


class TestHashDailyTrades:
    def test_returns_hex_and_payload(self) -> None:
        trades = [_sample_trade()]
        hex_digest, payload = hash_daily_trades(trades)
        assert len(hex_digest) == 64
        assert isinstance(payload, bytes)
        assert sha256_hex(payload) == hex_digest

    def test_empty_trades(self) -> None:
        hex_digest, payload = hash_daily_trades([])
        assert len(hex_digest) == 64
        assert payload == b"[]"
