"""Unit tests for social sentiment sources."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alphavedha.sentiment.sources import (
    RSSSource,
    RedditSource,
    SentimentPost,
    _item_matches,
    _symbol_variants,
)


class TestSymbolVariants:
    def test_plain_symbol(self) -> None:
        variants = _symbol_variants("TCS")
        assert "TCS" in variants

    def test_strips_ns_suffix(self) -> None:
        variants = _symbol_variants("INFY.NS")
        assert "INFY" in variants

    def test_strips_bo_suffix(self) -> None:
        variants = _symbol_variants("RELIANCE.BO")
        assert "RELIANCE" in variants


class TestItemMatches:
    def test_exact_symbol_match(self) -> None:
        assert _item_matches("TCS announces buyback", "", ["TCS"])

    def test_case_insensitive(self) -> None:
        assert _item_matches("tcs q4 results", "", ["TCS"])

    def test_no_match(self) -> None:
        assert not _item_matches("Wipro surges on results", "", ["TCS"])

    def test_match_in_summary(self) -> None:
        assert _item_matches("Market update", "TCS Q4 earnings beat", ["TCS"])


class TestRSSSource:
    @pytest.mark.asyncio
    async def test_rss_no_posts_when_network_fails(self) -> None:
        with patch("alphavedha.sentiment.sources._fetch_rss", new=AsyncMock(return_value=[])):
            src = RSSSource()
            posts = await src.fetch("TCS", lookback_days=7)
        assert posts == []

    @pytest.mark.asyncio
    async def test_rss_filters_by_symbol(self) -> None:
        items = [
            ("TCS Q4 profit rises", "Tata Consultancy profits up", "Mon, 09 Jun 2026 10:00:00 +0000"),
            ("Wipro acquires company", "Wipro deal closed", "Mon, 09 Jun 2026 10:00:00 +0000"),
        ]
        with patch("alphavedha.sentiment.sources._fetch_rss", new=AsyncMock(return_value=items)):
            src = RSSSource()
            posts = await src.fetch("TCS", lookback_days=7)
        # Each of the 3 feeds returns both items; only TCS matches the filter
        assert all("TCS" in p.text.upper() for p in posts)
        assert all("WIPRO" not in p.text.upper() for p in posts)
        assert len(posts) >= 1

    @pytest.mark.asyncio
    async def test_rss_filters_by_lookback(self) -> None:
        # Old article should be filtered out
        items = [
            ("TCS old news", "Old story", "Mon, 01 Jan 2020 10:00:00 +0000"),
        ]
        with patch("alphavedha.sentiment.sources._fetch_rss", new=AsyncMock(return_value=items)):
            src = RSSSource()
            posts = await src.fetch("TCS", lookback_days=7)
        assert posts == []

    @pytest.mark.asyncio
    async def test_rss_post_has_correct_source(self) -> None:
        items = [
            ("TCS results", "Quarterly earnings", "Mon, 09 Jun 2026 10:00:00 +0000"),
        ]
        # Patch only one feed returning data
        async def mock_fetch(source_name: str, url: str):
            if "moneycontrol" in url:
                return items
            return []

        with patch("alphavedha.sentiment.sources._fetch_rss", side_effect=mock_fetch):
            src = RSSSource()
            posts = await src.fetch("TCS", lookback_days=7)
        assert any(p.source == "moneycontrol_rss" for p in posts)


class TestRedditSource:
    @pytest.mark.asyncio
    async def test_reddit_returns_empty_without_credentials(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("REDDIT_CLIENT_ID", None)
            os.environ.pop("REDDIT_CLIENT_SECRET", None)
            src = RedditSource()
            posts = await src.fetch("TCS", lookback_days=7)
        assert posts == []

    @pytest.mark.asyncio
    async def test_reddit_returns_empty_when_praw_missing(self) -> None:
        # _get_reddit returns None → fetch returns []
        with patch.dict("os.environ", {"REDDIT_CLIENT_ID": "x", "REDDIT_CLIENT_SECRET": "y"}):
            src = RedditSource()
            with patch.object(src, "_get_reddit", return_value=None):
                posts = await src.fetch("TCS", lookback_days=7)
        assert isinstance(posts, list)
        assert posts == []
