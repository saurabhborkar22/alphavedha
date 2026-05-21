"""Indian financial news provider — RSS feeds from major financial news sources.

Sources: Moneycontrol, Economic Times, Livemint, Business Standard.
Uses RSS/Atom feeds (no API key required). Parses headlines and maps
to stock symbols via keyword matching.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from datetime import date

import httpx
import structlog

logger = structlog.get_logger(__name__)

RSS_FEEDS: dict[str, str] = {
    "moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_news": "https://www.moneycontrol.com/rss/latestnews.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_companies": "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",
    "livemint": "https://www.livemint.com/rss/markets",
    "business_std": "https://www.business-standard.com/rss/markets-106.rss",
}

_SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "RELIANCE.NS": ["reliance", "ril", "jio"],
    "TCS.NS": ["tcs", "tata consultancy"],
    "HDFCBANK.NS": ["hdfc bank"],
    "INFY.NS": ["infosys", "infy"],
    "ICICIBANK.NS": ["icici bank"],
    "HINDUNILVR.NS": ["hindustan unilever", "hul"],
    "SBIN.NS": ["sbi", "state bank"],
    "BHARTIARTL.NS": ["bharti airtel", "airtel"],
    "ITC.NS": ["itc ltd", "itc "],
    "KOTAKBANK.NS": ["kotak mahindra", "kotak bank"],
    "LT.NS": ["larsen", "l&t"],
    "AXISBANK.NS": ["axis bank"],
    "ASIANPAINT.NS": ["asian paints"],
    "MARUTI.NS": ["maruti suzuki", "maruti"],
    "TITAN.NS": ["titan company", "titan "],
    "SUNPHARMA.NS": ["sun pharma"],
    "BAJFINANCE.NS": ["bajaj finance"],
    "WIPRO.NS": ["wipro"],
    "ULTRACEMCO.NS": ["ultratech"],
    "HCLTECH.NS": ["hcl tech"],
    "TATAMOTORS.NS": ["tata motors"],
    "TATASTEEL.NS": ["tata steel"],
}

_MARKET_KEYWORDS = [
    "nifty",
    "sensex",
    "market",
    "rbi",
    "sebi",
    "inflation",
    "gdp",
    "rupee",
    "fii",
    "dii",
    "ipo",
    "mutual fund",
    "interest rate",
    "repo rate",
    "crude oil",
    "gold price",
]


@dataclass
class NewsArticleRecord:
    source: str
    title: str
    url: str | None
    published_date: date
    symbol: str | None
    content_hash: str


class NewsProvider:
    """Fetches financial news from Indian RSS feeds."""

    def __init__(self, timeout: float = 20.0) -> None:
        self._timeout = timeout

    async def fetch_all_feeds(self) -> list[NewsArticleRecord]:
        """Fetch articles from all configured RSS feeds."""
        all_articles: list[NewsArticleRecord] = []

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for source, url in RSS_FEEDS.items():
                try:
                    articles = await self._fetch_feed(client, source, url)
                    all_articles.extend(articles)
                except Exception:
                    logger.warning("news_feed_failed", source=source)
                await asyncio.sleep(0.5)

        logger.info("news_fetch_complete", total_articles=len(all_articles))
        return all_articles

    async def fetch_for_symbol(self, symbol: str) -> list[NewsArticleRecord]:
        """Fetch all recent articles and filter for a specific symbol."""
        all_articles = await self.fetch_all_feeds()
        return [a for a in all_articles if a.symbol == symbol]

    async def _fetch_feed(
        self,
        client: httpx.AsyncClient,
        source: str,
        url: str,
    ) -> list[NewsArticleRecord]:
        """Fetch and parse a single RSS feed."""
        resp = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (AlphaVedha News Aggregator)"},
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return []

        return self._parse_rss_xml(resp.text, source)

    def _parse_rss_xml(self, xml_text: str, source: str) -> list[NewsArticleRecord]:
        """Parse RSS XML without feedparser dependency — lightweight XML extraction."""
        articles: list[NewsArticleRecord] = []

        items = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL)
        if not items:
            items = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)

        for item_xml in items:
            title_match = re.search(r"<title[^>]*>(.*?)</title>", item_xml, re.DOTALL)
            if not title_match:
                continue
            title = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", title_match.group(1)).strip()
            title = re.sub(r"<[^>]+>", "", title).strip()

            link_match = re.search(r"<link[^>]*>(.*?)</link>", item_xml, re.DOTALL)
            if not link_match:
                link_match = re.search(r'<link[^>]+href="([^"]+)"', item_xml)
            url = link_match.group(1).strip() if link_match else None
            if url:
                url = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", url).strip()

            pub_date = self._extract_date(item_xml)
            symbol = self._match_symbol(title)
            content_hash = hashlib.sha256(f"{source}:{title}".encode()).hexdigest()

            articles.append(
                NewsArticleRecord(
                    source=source,
                    title=title[:500],
                    url=url[:1000] if url else None,
                    published_date=pub_date,
                    symbol=symbol,
                    content_hash=content_hash,
                )
            )

        return articles

    def _extract_date(self, item_xml: str) -> date:
        """Extract publication date from RSS item."""
        for tag in ("pubDate", "published", "dc:date", "updated"):
            match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", item_xml, re.DOTALL)
            if match:
                date_str = match.group(1).strip()
                try:
                    from dateutil.parser import parse as parse_date

                    return parse_date(date_str).date()
                except (ValueError, ImportError):
                    pass
        return date.today()

    def _match_symbol(self, title: str) -> str | None:
        """Match article title to a stock symbol via keyword matching."""
        title_lower = title.lower()

        for symbol, keywords in _SYMBOL_KEYWORDS.items():
            for kw in keywords:
                if kw in title_lower:
                    return symbol

        for kw in _MARKET_KEYWORDS:
            if kw in title_lower:
                return None

        return None

    def group_by_date_and_symbol(
        self,
        articles: list[NewsArticleRecord],
    ) -> dict[str, dict[str, list[str]]]:
        """Group article titles by date string and symbol.

        Returns: {date_str: {symbol: [titles...]}}
        """
        grouped: dict[str, dict[str, list[str]]] = {}
        for a in articles:
            date_str = a.published_date.isoformat()
            if date_str not in grouped:
                grouped[date_str] = {}
            sym = a.symbol or "_market_"
            if sym not in grouped[date_str]:
                grouped[date_str][sym] = []
            grouped[date_str][sym].append(a.title)
        return grouped

    def to_daily_articles(
        self,
        articles: list[NewsArticleRecord],
        symbol: str | None = None,
    ) -> dict[str, list[str]]:
        """Convert articles to daily_articles format for sentiment.py.

        Args:
            articles: List of news article records.
            symbol: If set, only include articles for this symbol + market-wide.

        Returns:
            Dict mapping date strings to lists of article titles.
        """
        daily: dict[str, list[str]] = {}
        for a in articles:
            if symbol and a.symbol and a.symbol != symbol:
                continue
            date_str = a.published_date.isoformat()
            if date_str not in daily:
                daily[date_str] = []
            daily[date_str].append(a.title)
        return daily
