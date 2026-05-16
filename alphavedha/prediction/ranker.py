"""StockRanker — filter and rank stock predictions into buy/sell candidate lists."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from alphavedha.prediction.engine import StockPrediction

logger = structlog.get_logger(__name__)


@dataclass
class RankingResult:
    buy_candidates: list[StockPrediction]
    sell_candidates: list[StockPrediction]
    excluded: list[tuple[str, str]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class StockRanker:
    def rank(
        self,
        predictions: list[StockPrediction],
        top_n: int = 10,
        circuit_hit_symbols: set[str] | None = None,
    ) -> RankingResult:
        circuit_hits = circuit_hit_symbols or set()
        excluded: list[tuple[str, str]] = []
        candidates: list[StockPrediction] = []

        for pred in predictions:
            if pred.symbol in circuit_hits:
                excluded.append((pred.symbol, "circuit hit"))
                continue
            if not pred.is_tradeable:
                excluded.append((pred.symbol, "not tradeable"))
                continue
            if pred.position_size_pct <= 0:
                excluded.append((pred.symbol, "zero position size"))
                continue
            candidates.append(pred)

        buy = sorted(
            [p for p in candidates if p.direction == 1],
            key=lambda p: p.composite_score,
            reverse=True,
        )[:top_n]

        sell = sorted(
            [p for p in candidates if p.direction == -1],
            key=lambda p: p.composite_score,
            reverse=True,
        )[:top_n]

        logger.info(
            "ranking_completed",
            total=len(predictions),
            buy=len(buy),
            sell=len(sell),
            excluded=len(excluded),
        )

        return RankingResult(
            buy_candidates=buy,
            sell_candidates=sell,
            excluded=excluded,
        )
