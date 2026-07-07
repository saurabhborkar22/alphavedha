"""Unit tests for social sentiment sources."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from alphavedha.sentiment.sources import (
    RedditSource,
    RSSSource,
    TelegramSource,
    _item_matches,
    _symbol_variants,
)


def _recent_rfc822(days_ago: int = 2) -> str:
    """Recent RFC-822 date so lookback-window tests don't expire as time passes."""
    return (datetime.now(UTC) - timedelta(days=days_ago)).strftime("%a, %d %b %Y %H:%M:%S +0000")


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
            (
                "TCS Q4 profit rises",
                "Tata Consultancy profits up",
                _recent_rfc822(),
            ),
            ("Wipro acquires company", "Wipro deal closed", _recent_rfc822()),
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
            ("TCS results", "Quarterly earnings", _recent_rfc822()),
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


class TestTelegramSource:
    def _disclosures_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": "TCS",
                    "source": "TELEGRAM",
                    "headline": "TCS wins mega deal",
                    "text": "TCS wins mega deal from European client",
                    "filed_at": datetime.now(UTC) - timedelta(days=1),
                    "url": "https://t.me/moneycontrolcom/1",
                },
                {
                    "symbol": "TCS",
                    "source": "BSE",  # exchange filing — must be excluded
                    "headline": "Board meeting intimation",
                    "text": "Board meeting on ...",
                    "filed_at": datetime.now(UTC) - timedelta(days=2),
                    "url": None,
                },
            ]
        )

    @pytest.mark.asyncio
    async def test_returns_only_telegram_rows(self) -> None:
        with patch(
            "alphavedha.intel.store.load_disclosures",
            new=AsyncMock(return_value=self._disclosures_df()),
        ):
            src = TelegramSource()
            posts = await src.fetch("TCS", lookback_days=7)
        assert len(posts) == 1
        assert posts[0].source == "telegram"
        assert "mega deal" in posts[0].text
        assert posts[0].published_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_empty_dataframe(self) -> None:
        with patch(
            "alphavedha.intel.store.load_disclosures",
            new=AsyncMock(return_value=pd.DataFrame()),
        ):
            posts = await TelegramSource().fetch("TCS", lookback_days=7)
        assert posts == []

    @pytest.mark.asyncio
    async def test_degrades_on_db_error(self) -> None:
        with patch(
            "alphavedha.intel.store.load_disclosures",
            new=AsyncMock(side_effect=RuntimeError("db down")),
        ):
            posts = await TelegramSource().fetch("TCS", lookback_days=7)
        assert posts == []
