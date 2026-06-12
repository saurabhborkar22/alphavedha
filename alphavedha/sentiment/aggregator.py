"""SentimentAggregator — combines social sources, scores with FinBERT, returns result.

Usage
-----
    from alphavedha.sentiment.aggregator import SentimentAggregator, SocialSentimentResult

    agg = SentimentAggregator()
    result = await agg.aggregate("TCS", lookback_days=7)
    print(result.score, result.verdict)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import structlog

from alphavedha.features.sentiment import score_articles
from alphavedha.sentiment.sources import RedditSource, RSSSource, SentimentPost

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class SocialSentimentResult:
    symbol: str
    # Composite score: weighted average of FinBERT net scores (positive - negative)
    # Range: [-1.0, 1.0]; 0.0 = neutral / no data
    score: float
    # 7d momentum: current score minus prior 3d mean
    momentum: float
    # Total number of posts/articles scored
    post_count: int
    # Per-source breakdown
    source_counts: dict[str, int] = field(default_factory=dict)
    # Verdict based on score + post_count thresholds
    verdict: str = "neutral"  # "bullish" | "cautiously_bullish" | "neutral" | "cautiously_bearish" | "bearish"
    data_quality: float = 1.0  # 0.0 when no posts found
    generated_at: str = ""


_BULLISH_THRESHOLD = 0.15
_CAUTIOUS_BULL_THRESHOLD = 0.05
_CAUTIOUS_BEAR_THRESHOLD = -0.05
_BEARISH_THRESHOLD = -0.15


def _verdict(score: float, post_count: int) -> str:
    if post_count == 0:
        return "neutral"
    if score >= _BULLISH_THRESHOLD:
        return "bullish"
    if score >= _CAUTIOUS_BULL_THRESHOLD:
        return "cautiously_bullish"
    if score <= _BEARISH_THRESHOLD:
        return "bearish"
    if score <= _CAUTIOUS_BEAR_THRESHOLD:
        return "cautiously_bearish"
    return "neutral"


def _score_posts(posts: list[SentimentPost]) -> list[float]:
    texts = [p.text for p in posts]
    if not texts:
        return []
    scored = score_articles(texts)
    return [s["positive"] - s["negative"] for s in scored]


class SentimentAggregator:
    """Aggregates sentiment from RSS feeds and Reddit into a single score."""

    def __init__(self) -> None:
        self._rss = RSSSource()
        self._reddit = RedditSource()

    async def aggregate(
        self,
        symbol: str,
        lookback_days: int = 7,
    ) -> SocialSentimentResult:
        """Fetch posts from all sources, score with FinBERT, return aggregated result."""
        now_ist = datetime.now(IST).isoformat()

        # Fetch concurrently
        rss_posts, reddit_posts = await asyncio.gather(
            self._rss.fetch(symbol, lookback_days),
            self._reddit.fetch(symbol, lookback_days),
        )

        all_posts = rss_posts + reddit_posts

        if not all_posts:
            logger.info("sentiment_no_posts", symbol=symbol)
            return SocialSentimentResult(
                symbol=symbol,
                score=0.0,
                momentum=0.0,
                post_count=0,
                source_counts={},
                verdict="neutral",
                data_quality=0.0,
                generated_at=now_ist,
            )

        # FinBERT scoring
        net_scores = _score_posts(all_posts)
        overall_score = float(sum(net_scores) / len(net_scores)) if net_scores else 0.0

        # Momentum: compare recent half vs earlier half
        half = max(1, len(all_posts) // 2)
        recent_scores = net_scores[:half]
        earlier_scores = net_scores[half:] or [0.0]
        recent_mean = sum(recent_scores) / len(recent_scores)
        earlier_mean = sum(earlier_scores) / len(earlier_scores)
        momentum = round(recent_mean - earlier_mean, 4)

        # Source breakdown
        source_counts: dict[str, int] = {}
        for p in all_posts:
            source_counts[p.source] = source_counts.get(p.source, 0) + 1

        quality = min(1.0, len(all_posts) / 10)  # 10+ posts = full quality

        logger.info(
            "sentiment_aggregated",
            symbol=symbol,
            n_posts=len(all_posts),
            score=round(overall_score, 4),
            momentum=momentum,
        )

        return SocialSentimentResult(
            symbol=symbol,
            score=round(overall_score, 4),
            momentum=momentum,
            post_count=len(all_posts),
            source_counts=source_counts,
            verdict=_verdict(overall_score, len(all_posts)),
            data_quality=round(quality, 2),
            generated_at=now_ist,
        )
