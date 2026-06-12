"""Social sentiment API endpoint."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from alphavedha.api.deps import verify_api_key
from alphavedha.sentiment.aggregator import SentimentAggregator

logger = structlog.get_logger(__name__)
router = APIRouter(
    prefix="/sentiment",
    tags=["sentiment"],
    dependencies=[Depends(verify_api_key)],
)

_aggregator = SentimentAggregator()


@router.get("/{symbol}")
async def social_sentiment(
    symbol: str,
    lookback_days: int = Query(default=7, ge=1, le=30, description="Days of posts to aggregate"),
) -> dict[str, Any]:
    """Return social media sentiment score for an NSE symbol.

    Aggregates RSS feeds (Moneycontrol, ET Markets, Business Standard) and,
    when REDDIT_CLIENT_ID/SECRET are configured, Indian investing subreddits.
    Articles are scored with FinBERT and combined into a single composite score.

    Score range: -1.0 (very bearish) to +1.0 (very bullish). 0.0 = neutral / no data.

    Verdict thresholds
    ------------------
    score >= 0.15   → bullish
    score >= 0.05   → cautiously_bullish
    -0.05 to 0.05   → neutral
    score <= -0.05  → cautiously_bearish
    score <= -0.15  → bearish
    """
    symbol = symbol.upper().strip()
    if not symbol or len(symbol) > 20:
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")

    result = await _aggregator.aggregate(symbol, lookback_days=lookback_days)

    return {
        "symbol": result.symbol,
        "score": result.score,
        "momentum": result.momentum,
        "verdict": result.verdict,
        "post_count": result.post_count,
        "source_counts": result.source_counts,
        "data_quality": result.data_quality,
        "lookback_days": lookback_days,
        "generated_at": result.generated_at,
    }
