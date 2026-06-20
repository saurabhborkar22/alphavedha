"""Tests for the per-strategy daily summary report."""

from __future__ import annotations

from datetime import date

import pandas as pd

from alphavedha.monitoring.strategy_summary import (
    StrategySummaryReport,
    build_strategy_summary,
)


def _trade(
    symbol: str,
    strategy: str = "ensemble_v1",
    direction: int = 1,
    confidence: float = 0.7,
    actual_return: float | None = None,
    exit_price: float | None = None,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "prediction_date": date(2026, 6, 10),
        "predicted_direction": direction,
        "predicted_magnitude": 0.02,
        "confidence": confidence,
        "model_version": "v1",
        "regime": "bull",
        "is_tradeable": True,
        "entry_price": 100.0,
        "exit_price": exit_price,
        "actual_return": actual_return,
        "is_correct": None,
        "strategy": strategy,
    }


D = date(2026, 6, 20)


class TestBuildStrategySummary:
    def test_empty_trades(self) -> None:
        report = build_strategy_summary(pd.DataFrame(), D)
        assert isinstance(report, StrategySummaryReport)
        assert report.strategy_sections == []
        assert report.total_open == 0
        assert report.total_matured == 0

    def test_single_strategy(self) -> None:
        trades = pd.DataFrame(
            [
                _trade("TCS.NS", "ensemble_v1", actual_return=0.02, exit_price=102.0),
                _trade("INFY.NS", "ensemble_v1", actual_return=-0.01, exit_price=99.0),
            ]
        )
        report = build_strategy_summary(trades, D)
        assert len(report.strategy_sections) == 1
        assert report.strategy_sections[0]["strategy"] == "ensemble_v1"
        assert report.total_matured == 2
        assert report.total_open == 0

    def test_multiple_strategies(self) -> None:
        trades = pd.DataFrame(
            [
                _trade("TCS.NS", "ensemble_v1", actual_return=0.02, exit_price=102.0),
                _trade("INFY.NS", "event_drift_v1", actual_return=0.03, exit_price=103.0),
                _trade("HDFC.NS", "insider_cluster_v1"),
            ]
        )
        report = build_strategy_summary(trades, D)
        assert len(report.strategy_sections) == 3
        strategies = [s["strategy"] for s in report.strategy_sections]
        assert "ensemble_v1" in strategies
        assert "event_drift_v1" in strategies
        assert "insider_cluster_v1" in strategies
        assert report.total_open == 1
        assert report.total_matured == 2

    def test_avoid_list_included(self) -> None:
        report = build_strategy_summary(
            pd.DataFrame(),
            D,
            avoid_list_symbols=["YESBANK.NS", "DHFL.NS"],
        )
        assert report.avoid_list_symbols == ["YESBANK.NS", "DHFL.NS"]

    def test_report_date(self) -> None:
        report = build_strategy_summary(pd.DataFrame(), D)
        assert report.report_date == D


class TestFormatText:
    def test_empty_report_has_header(self) -> None:
        report = build_strategy_summary(pd.DataFrame(), D)
        text = report.format_text()
        assert "2026-06-20" in text
        assert "No paper trades found" in text

    def test_report_with_data_contains_strategy_names(self) -> None:
        trades = pd.DataFrame(
            [
                _trade("TCS.NS", "ensemble_v1", actual_return=0.02, exit_price=102.0),
                _trade("INFY.NS", "event_drift_v1", actual_return=0.03, exit_price=103.0),
            ]
        )
        report = build_strategy_summary(trades, D)
        text = report.format_text()
        assert "ensemble_v1" in text
        assert "event_drift_v1" in text
        assert "Win rate" in text

    def test_avoid_list_in_text(self) -> None:
        report = build_strategy_summary(
            pd.DataFrame(),
            D,
            avoid_list_symbols=["YESBANK.NS"],
        )
        text = report.format_text()
        assert "YESBANK.NS" in text

    def test_empty_avoid_list(self) -> None:
        report = build_strategy_summary(pd.DataFrame(), D)
        text = report.format_text()
        assert "Avoid list: (empty)" in text
