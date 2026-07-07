"""Social sentiment data sources for Indian equities.

Supported sources
-----------------
RSSSource  — Moneycontrol, ET Markets, Business Standard RSS feeds (no auth)
RedditSource — r/IndiaInvestments, r/niftyoptions, r/DalalStreetTalks (requires PRAW + credentials)

Both sources degrade gracefully to empty lists when credentials are missing or
network calls fail, so the aggregator always gets a valid (possibly empty) response.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Protocol

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Common data model
# ---------------------------------------------------------------------------


@dataclass
class SentimentPost:
    text: str  # title + snippet suitable for FinBERT scoring
    source: str  # e.g. "moneycontrol_rss", "reddit_india_investments"
    published_at: datetime
    url: str = ""


# ---------------------------------------------------------------------------
# Source protocol
# ---------------------------------------------------------------------------


class SocialSource(Protocol):
    async def fetch(self, symbol: str, lookback_days: int) -> list[SentimentPost]: ...


# ---------------------------------------------------------------------------
# RSS source — Moneycontrol / ET Markets / Business Standard
# ---------------------------------------------------------------------------

# Each entry: (source_name, url_template — {symbol} replaced with NSE ticker)
_RSS_FEEDS: list[tuple[str, str]] = [
    (
        "moneycontrol_rss",
        "https://www.moneycontrol.com/rss/buzzingstocks.xml",
    ),
    (
        "etmarkets_rss",
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    ),
    (
        "bsnews_rss",
        "https://www.business-standard.com/rss/markets-106.rss",
    ),
]

_HTTP_TIMEOUT = 8.0  # seconds


def _parse_rfc2822(date_str: str) -> datetime:
    """Parse RFC-2822 date (RSS pubDate) to UTC-aware datetime."""
    try:
        return parsedate_to_datetime(date_str).astimezone(UTC)
    except Exception:
        return datetime.now(UTC)


def _symbol_variants(symbol: str) -> list[str]:
    """Return search terms for a given NSE symbol (e.g. TCS → ['TCS', 'Tata Consultancy'])."""
    base = symbol.upper().replace(".NS", "").replace(".BO", "")
    variants = [base]
    # Strip numeric suffixes common in BSE codes
    alpha_only = re.sub(r"\d+$", "", base)
    if alpha_only != base:
        variants.append(alpha_only)
    return variants


def _item_matches(title: str, summary: str, terms: list[str]) -> bool:
    haystack = (title + " " + summary).upper()
    return any(t in haystack for t in terms)


async def _fetch_rss(source_name: str, url: str) -> list[tuple[str, str, str]]:
    """Fetch RSS and return list of (title, summary, published_at_str)."""
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "AlphaVedha/1.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            items = []
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                summary = (item.findtext("description") or item.findtext("summary") or "").strip()
                pubdate = item.findtext("pubDate") or ""
                items.append((title, summary, pubdate))
            return items
    except Exception as exc:
        logger.debug("rss_fetch_failed", source=source_name, url=url, error=str(exc))
        return []


class RSSSource:
    """Fetches news from Indian financial RSS feeds and filters by symbol keywords."""

    async def fetch(self, symbol: str, lookback_days: int = 7) -> list[SentimentPost]:
        terms = _symbol_variants(symbol)
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        posts: list[SentimentPost] = []

        for source_name, url in _RSS_FEEDS:
            items = await _fetch_rss(source_name, url)
            for title, summary, pubdate_str in items:
                if not _item_matches(title, summary, terms):
                    continue
                published_at = _parse_rfc2822(pubdate_str) if pubdate_str else datetime.now(UTC)
                if published_at < cutoff:
                    continue
                text = f"{title}. {summary}"[:512]
                posts.append(
                    SentimentPost(text=text, source=source_name, published_at=published_at)
                )

        logger.debug("rss_posts_fetched", symbol=symbol, n=len(posts))
        return posts


# ---------------------------------------------------------------------------
# Reddit source — r/IndiaInvestments, r/niftyoptions, r/DalalStreetTalks
# ---------------------------------------------------------------------------

_REDDIT_SUBREDDITS = [
    "IndiaInvestments",
    "niftyoptions",
    "DalalStreetTalks",
    "IndianStockMarket",
]

_REDDIT_POST_LIMIT = 25  # posts per subreddit


class RedditSource:
    """Fetches posts from Indian investing subreddits.

    Requires PRAW and environment variables:
        REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

    Degrades gracefully to empty list when credentials are missing.
    """

    def _get_reddit(self) -> object | None:
        import os

        client_id = os.environ.get("REDDIT_CLIENT_ID", "")
        secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        user_agent = os.environ.get("REDDIT_USER_AGENT", "AlphaVedha/1.0")
        if not client_id or not secret:
            return None
        try:
            import praw

            return praw.Reddit(
                client_id=client_id,
                client_secret=secret,
                user_agent=user_agent,
                read_only=True,
            )
        except ImportError:
            logger.debug("praw_not_installed")
            return None
        except Exception as exc:
            logger.warning("reddit_init_failed", error=str(exc))
            return None

    async def fetch(self, symbol: str, lookback_days: int = 7) -> list[SentimentPost]:
        import asyncio

        reddit = self._get_reddit()
        if reddit is None:
            return []

        terms = _symbol_variants(symbol)
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        def _fetch_sync() -> list[SentimentPost]:
            posts: list[SentimentPost] = []
            for sub_name in _REDDIT_SUBREDDITS:
                try:
                    sub = reddit.subreddit(sub_name)  # type: ignore[attr-defined]
                    for post in sub.search(symbol, limit=_REDDIT_POST_LIMIT, sort="new"):
                        published_at = datetime.fromtimestamp(post.created_utc, tz=UTC)
                        if published_at < cutoff:
                            continue
                        title = post.title or ""
                        selftext = (post.selftext or "")[:300]
                        if not _item_matches(title, selftext, terms):
                            continue
                        text = f"{title}. {selftext}"[:512]
                        posts.append(
                            SentimentPost(
                                text=text,
                                source=f"reddit_{sub_name.lower()}",
                                published_at=published_at,
                                url=f"https://reddit.com{post.permalink}",
                            )
                        )
                except Exception as exc:
                    logger.debug("reddit_sub_failed", sub=sub_name, error=str(exc))
            return posts

        try:
            return await asyncio.to_thread(_fetch_sync)
        except Exception as exc:
            logger.warning("reddit_fetch_failed", symbol=symbol, error=str(exc))
            return []


# ---------------------------------------------------------------------------
# Telegram source — news rows stored by the intel collectors
# ---------------------------------------------------------------------------


class TelegramSource:
    """Surfaces Telegram channel news already stored in ``disclosures``.

    No network calls — the polling collector / live watcher persist messages
    with source="TELEGRAM"; this reads them back per symbol for FinBERT
    scoring. Degrades gracefully to an empty list on any DB error.
    """

    async def fetch(self, symbol: str, lookback_days: int = 7) -> list[SentimentPost]:
        since = datetime.now(UTC) - timedelta(days=lookback_days)
        try:
            from alphavedha.intel.store import load_disclosures

            df = await load_disclosures(symbol=symbol, since=since, limit=100)
        except Exception as exc:
            logger.debug("telegram_source_failed", symbol=symbol, error=str(exc))
            return []

        if df.empty or "source" not in df.columns:
            return []

        posts: list[SentimentPost] = []
        for _, row in df[df["source"] == "TELEGRAM"].iterrows():
            text = str(row.get("text") or row.get("headline") or "").strip()[:512]
            if not text:
                continue
            published_at = row["filed_at"]
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=UTC)
            posts.append(
                SentimentPost(
                    text=text,
                    source="telegram",
                    published_at=published_at.astimezone(UTC),
                    url=str(row.get("url") or ""),
                )
            )

        logger.debug("telegram_posts_fetched", symbol=symbol, n=len(posts))
        return posts
