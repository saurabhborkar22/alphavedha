"""Unit tests for SentimentAggregator."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from alphavedha.sentiment.aggregator import SentimentAggregator, SocialSentimentResult, _verdict
from alphavedha.sentiment.sources import SentimentPost


def _post(text: str, source: str = "rss") -> SentimentPost:
    return SentimentPost(text=text, source=source, published_at=datetime.now(timezone.utc))


class TestVerdict:
    def test_no_posts_neutral(self) -> None:
        assert _verdict(0.5, 0) == "neutral"

    def test_high_positive_bullish(self) -> None:
        assert _verdict(0.20, 5) == "bullish"

    def test_mild_positive_cautiously_bullish(self) -> None:
        assert _verdict(0.08, 5) == "cautiously_bullish"

    def test_near_zero_neutral(self) -> None:
        assert _verdict(0.01, 5) == "neutral"

    def test_mild_negative_cautiously_bearish(self) -> None:
        assert _verdict(-0.08, 5) == "cautiously_bearish"

    def test_strong_negative_bearish(self) -> None:
        assert _verdict(-0.20, 5) == "bearish"


class TestSentimentAggregator:
    @pytest.mark.asyncio
    async def test_no_posts_returns_neutral_result(self) -> None:
        agg = SentimentAggregator()
        with (
            patch.object(agg._rss, "fetch", new=AsyncMock(return_value=[])),
            patch.object(agg._reddit, "fetch", new=AsyncMock(return_value=[])),
        ):
            result = await agg.aggregate("TCS", lookback_days=7)

        assert isinstance(result, SocialSentimentResult)
        assert result.score == 0.0
        assert result.post_count == 0
        assert result.verdict == "neutral"
        assert result.data_quality == 0.0
        assert result.symbol == "TCS"

    @pytest.mark.asyncio
    async def test_positive_posts_produce_bullish(self) -> None:
        posts = [_post("TCS strong earnings beat expectations") for _ in range(5)]
        mock_scores = [{"positive": 0.8, "negative": 0.05, "neutral": 0.15}] * 5

        agg = SentimentAggregator()
        with (
            patch.object(agg._rss, "fetch", new=AsyncMock(return_value=posts)),
            patch.object(agg._reddit, "fetch", new=AsyncMock(return_value=[])),
            patch("alphavedha.sentiment.aggregator.score_articles", return_value=mock_scores),
        ):
            result = await agg.aggregate("TCS", lookback_days=7)

        assert result.post_count == 5
        assert result.score > 0.0
        assert result.verdict in ("bullish", "cautiously_bullish")

    @pytest.mark.asyncio
    async def test_negative_posts_produce_bearish(self) -> None:
        posts = [_post("TCS misses earnings, stock tanks") for _ in range(5)]
        mock_scores = [{"positive": 0.05, "negative": 0.85, "neutral": 0.10}] * 5

        agg = SentimentAggregator()
        with (
            patch.object(agg._rss, "fetch", new=AsyncMock(return_value=posts)),
            patch.object(agg._reddit, "fetch", new=AsyncMock(return_value=[])),
            patch("alphavedha.sentiment.aggregator.score_articles", return_value=mock_scores),
        ):
            result = await agg.aggregate("TCS", lookback_days=7)

        assert result.score < 0.0
        assert result.verdict in ("bearish", "cautiously_bearish")

    @pytest.mark.asyncio
    async def test_source_counts_populated(self) -> None:
        rss_posts = [_post("TCS RSS news", source="moneycontrol_rss")]
        reddit_posts = [_post("TCS reddit post", source="reddit_indiainvestments")]
        mock_scores = [{"positive": 0.5, "negative": 0.2, "neutral": 0.3}] * 2

        agg = SentimentAggregator()
        with (
            patch.object(agg._rss, "fetch", new=AsyncMock(return_value=rss_posts)),
            patch.object(agg._reddit, "fetch", new=AsyncMock(return_value=reddit_posts)),
            patch("alphavedha.sentiment.aggregator.score_articles", return_value=mock_scores),
        ):
            result = await agg.aggregate("TCS", lookback_days=7)

        assert "moneycontrol_rss" in result.source_counts
        assert "reddit_indiainvestments" in result.source_counts
        assert result.post_count == 2

    @pytest.mark.asyncio
    async def test_data_quality_scales_with_post_count(self) -> None:
        posts = [_post("TCS news") for _ in range(10)]
        mock_scores = [{"positive": 0.5, "negative": 0.2, "neutral": 0.3}] * 10

        agg = SentimentAggregator()
        with (
            patch.object(agg._rss, "fetch", new=AsyncMock(return_value=posts)),
            patch.object(agg._reddit, "fetch", new=AsyncMock(return_value=[])),
            patch("alphavedha.sentiment.aggregator.score_articles", return_value=mock_scores),
        ):
            result = await agg.aggregate("TCS", lookback_days=7)

        assert result.data_quality == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_generated_at_is_populated(self) -> None:
        agg = SentimentAggregator()
        with (
            patch.object(agg._rss, "fetch", new=AsyncMock(return_value=[])),
            patch.object(agg._reddit, "fetch", new=AsyncMock(return_value=[])),
        ):
            result = await agg.aggregate("TCS")

        assert result.generated_at != ""
