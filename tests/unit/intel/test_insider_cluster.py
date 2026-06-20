"""Tests for the insider cluster signal generator."""

from __future__ import annotations

from datetime import date

from alphavedha.intel.signals.insider_cluster import (
    CLUSTER_WINDOW_DAYS,
    MIN_DISTINCT_INSIDERS,
    MIN_VALUE_LAKHS,
    STRATEGY_NAME,
    generate_insider_cluster_signals,
)


def _trade(
    person: str = "Insider A",
    trade_type: str = "Buy",
    value_lakhs: float = 30.0,
    trade_date: date | None = None,
) -> dict[str, object]:
    if trade_date is None:
        trade_date = date(2026, 6, 15)
    return {
        "person_name": person,
        "trade_type": trade_type,
        "value_lakhs": value_lakhs,
        "trade_date": trade_date,
    }


SIGNAL_DATE = date(2026, 6, 20)


class TestGenerateInsiderClusterSignals:
    def test_cluster_fires_with_two_distinct_buyers(self) -> None:
        trades = {
            "TCS.NS": [
                _trade("CEO", "Buy", 30.0, date(2026, 6, 15)),
                _trade("CFO", "Buy", 20.0, date(2026, 6, 16)),
            ]
        }
        signals = generate_insider_cluster_signals(trades, SIGNAL_DATE)
        assert len(signals) == 1
        assert signals[0].symbol == "TCS.NS"
        assert signals[0].direction == 1
        assert signals[0].distinct_insiders == 2
        assert signals[0].total_value_lakhs == 50.0

    def test_single_insider_does_not_fire(self) -> None:
        trades = {
            "TCS.NS": [
                _trade("CEO", "Buy", 100.0, date(2026, 6, 15)),
            ]
        }
        signals = generate_insider_cluster_signals(trades, SIGNAL_DATE)
        assert len(signals) == 0

    def test_below_value_threshold_does_not_fire(self) -> None:
        trades = {
            "TCS.NS": [
                _trade("CEO", "Buy", 10.0, date(2026, 6, 15)),
                _trade("CFO", "Buy", 5.0, date(2026, 6, 16)),
            ]
        }
        signals = generate_insider_cluster_signals(trades, SIGNAL_DATE)
        assert len(signals) == 0

    def test_net_seller_excluded(self) -> None:
        trades = {
            "TCS.NS": [
                _trade("CEO", "Buy", 30.0, date(2026, 6, 15)),
                _trade("CFO", "Buy", 50.0, date(2026, 6, 16)),
                _trade("CFO", "Sell", 60.0, date(2026, 6, 17)),
            ]
        }
        signals = generate_insider_cluster_signals(trades, SIGNAL_DATE)
        assert len(signals) == 0

    def test_stale_trades_outside_window_excluded(self) -> None:
        old_date = SIGNAL_DATE - __import__("datetime").timedelta(days=CLUSTER_WINDOW_DAYS + 5)
        trades = {
            "TCS.NS": [
                _trade("CEO", "Buy", 30.0, old_date),
                _trade("CFO", "Buy", 30.0, old_date),
            ]
        }
        signals = generate_insider_cluster_signals(trades, SIGNAL_DATE)
        assert len(signals) == 0

    def test_avoid_list_vetos_signal(self) -> None:
        trades = {
            "TCS.NS": [
                _trade("CEO", "Buy", 30.0, date(2026, 6, 15)),
                _trade("CFO", "Buy", 30.0, date(2026, 6, 16)),
            ]
        }
        signals = generate_insider_cluster_signals(
            trades, SIGNAL_DATE, avoid_symbols=frozenset({"TCS.NS"})
        )
        assert len(signals) == 0

    def test_multiple_symbols_sorted_by_confidence(self) -> None:
        trades = {
            "TCS.NS": [
                _trade("CEO", "Buy", 30.0, date(2026, 6, 15)),
                _trade("CFO", "Buy", 30.0, date(2026, 6, 16)),
            ],
            "INFY.NS": [
                _trade("A", "Buy", 100.0, date(2026, 6, 15)),
                _trade("B", "Buy", 100.0, date(2026, 6, 16)),
                _trade("C", "Buy", 100.0, date(2026, 6, 17)),
            ],
        }
        signals = generate_insider_cluster_signals(trades, SIGNAL_DATE)
        assert len(signals) == 2
        assert signals[0].symbol == "INFY.NS"

    def test_empty_trades_returns_empty(self) -> None:
        assert generate_insider_cluster_signals({}, SIGNAL_DATE) == []

    def test_date_string_parsed(self) -> None:
        trades = {
            "TCS.NS": [
                _trade("CEO", "Buy", 30.0),
                {
                    "person_name": "CFO",
                    "trade_type": "Buy",
                    "value_lakhs": 30.0,
                    "trade_date": "2026-06-16",
                },
            ]
        }
        signals = generate_insider_cluster_signals(trades, SIGNAL_DATE)
        assert len(signals) == 1


class TestConstants:
    def test_strategy_name(self) -> None:
        assert STRATEGY_NAME == "insider_cluster_v1"

    def test_thresholds(self) -> None:
        assert MIN_DISTINCT_INSIDERS == 2
        assert MIN_VALUE_LAKHS == 25.0
        assert CLUSTER_WINDOW_DAYS == 14
