"""Tests for CLI Rich formatters."""

from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO

import numpy as np
from rich.console import Console

from alphavedha.cli.formatters import format_prediction, format_ranking
from alphavedha.prediction.engine import StockPrediction
from alphavedha.prediction.ranker import RankingResult


def _make_prediction(
    symbol: str = "TCS", direction: int = 1, composite_score: float = 78.5
) -> StockPrediction:
    return StockPrediction(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        direction=direction,
        magnitude=0.03,
        composite_score=composite_score,
        meta_confidence=0.72,
        is_tradeable=True,
        regime="bull",
        regime_probabilities=np.array([0.7, 0.1, 0.1, 0.1]),
        price_target_low=95.0,
        price_target_mid=100.0,
        price_target_high=105.0,
        model_disagreement=0.05,
        position_size_pct=5.0,
        model_version="v0.1.0",
        warnings=[],
    )


class TestFormatPrediction:
    def test_panel_contains_symbol(self) -> None:
        panel = format_prediction(_make_prediction("TCS"))
        output = StringIO()
        Console(file=output, force_terminal=True, width=100).print(panel)
        text = output.getvalue()
        assert "TCS" in text

    def test_buy_direction_shown(self) -> None:
        panel = format_prediction(_make_prediction(direction=1))
        output = StringIO()
        Console(file=output, force_terminal=True, width=100).print(panel)
        text = output.getvalue()
        assert "BUY" in text

    def test_sell_direction_shown(self) -> None:
        panel = format_prediction(_make_prediction(direction=-1))
        output = StringIO()
        Console(file=output, force_terminal=True, width=100).print(panel)
        text = output.getvalue()
        assert "SELL" in text

    def test_panel_contains_score(self) -> None:
        panel = format_prediction(_make_prediction(composite_score=85.3))
        output = StringIO()
        Console(file=output, force_terminal=True, width=100).print(panel)
        text = output.getvalue()
        assert "85.3" in text


class TestFormatRanking:
    def test_table_contains_symbols(self) -> None:
        result = RankingResult(
            buy_candidates=[_make_prediction("TCS"), _make_prediction("INFY")],
            sell_candidates=[_make_prediction("RELIANCE", direction=-1)],
            excluded=[("HDFC", "hold signal")],
        )
        table = format_ranking(result)
        output = StringIO()
        Console(file=output, force_terminal=True, width=120).print(table)
        text = output.getvalue()
        assert "TCS" in text
        assert "INFY" in text
