"""Tests for StockRanker — filter and rank predictions."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from alphavedha.prediction.engine import StockPrediction
from alphavedha.prediction.ranker import RankingResult, StockRanker


def _make_prediction(
    symbol: str,
    direction: int = 1,
    composite_score: float = 75.0,
    is_tradeable: bool = True,
    position_size_pct: float = 5.0,
) -> StockPrediction:
    return StockPrediction(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        direction=direction,
        magnitude=0.03,
        composite_score=composite_score,
        meta_confidence=0.7,
        is_tradeable=is_tradeable,
        regime="bull",
        regime_probabilities=np.array([0.7, 0.1, 0.1, 0.1]),
        price_target_low=100.0,
        price_target_mid=105.0,
        price_target_high=110.0,
        model_disagreement=0.05,
        position_size_pct=position_size_pct,
        model_version="v0.1.0",
        warnings=[],
    )


class TestStockRanker:
    def test_filters_non_tradeable(self) -> None:
        ranker = StockRanker()
        preds = [
            _make_prediction("TCS", direction=1, is_tradeable=True),
            _make_prediction("INFY", direction=1, is_tradeable=False),
        ]
        result = ranker.rank(preds)
        assert isinstance(result, RankingResult)
        assert len(result.buy_candidates) == 1
        assert result.buy_candidates[0].symbol == "TCS"
        assert any(sym == "INFY" for sym, _ in result.excluded)

    def test_separates_buy_and_sell(self) -> None:
        ranker = StockRanker()
        preds = [
            _make_prediction("TCS", direction=1),
            _make_prediction("RELIANCE", direction=-1),
            _make_prediction("HDFC", direction=0, is_tradeable=True),
        ]
        result = ranker.rank(preds)
        assert len(result.buy_candidates) == 1
        assert len(result.sell_candidates) == 1
        assert result.buy_candidates[0].symbol == "TCS"
        assert result.sell_candidates[0].symbol == "RELIANCE"

    def test_sorts_by_composite_score_desc(self) -> None:
        ranker = StockRanker()
        preds = [
            _make_prediction("LOW", direction=1, composite_score=60.0),
            _make_prediction("HIGH", direction=1, composite_score=90.0),
            _make_prediction("MID", direction=1, composite_score=75.0),
        ]
        result = ranker.rank(preds)
        scores = [p.composite_score for p in result.buy_candidates]
        assert scores == sorted(scores, reverse=True)
        assert result.buy_candidates[0].symbol == "HIGH"

    def test_respects_top_n(self) -> None:
        ranker = StockRanker()
        preds = [
            _make_prediction(f"STOCK{i}", direction=1, composite_score=float(90 - i))
            for i in range(20)
        ]
        result = ranker.rank(preds, top_n=5)
        assert len(result.buy_candidates) == 5

    def test_circuit_hit_excluded(self) -> None:
        ranker = StockRanker()
        preds = [
            _make_prediction("TCS", direction=1),
            _make_prediction("INFY", direction=1),
        ]
        result = ranker.rank(preds, circuit_hit_symbols={"INFY"})
        assert len(result.buy_candidates) == 1
        assert result.buy_candidates[0].symbol == "TCS"
        assert any(sym == "INFY" for sym, _ in result.excluded)
