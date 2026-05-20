"""Tests for news provider RSS parsing and symbol matching."""

from __future__ import annotations

from alphavedha.data.providers.news_provider import NewsProvider

_SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Market News</title>
<item>
<title>TCS reports strong Q4 results, beats estimates</title>
<link>https://example.com/tcs-q4</link>
<pubDate>Mon, 15 Apr 2024 10:30:00 +0530</pubDate>
</item>
<item>
<title>Nifty 50 hits all-time high on FII inflows</title>
<link>https://example.com/nifty-high</link>
<pubDate>Tue, 16 Apr 2024 09:00:00 +0530</pubDate>
</item>
<item>
<title><![CDATA[Reliance Jio adds 5 million subscribers]]></title>
<link>https://example.com/jio-subs</link>
<pubDate>Wed, 17 Apr 2024 14:00:00 +0530</pubDate>
</item>
<item>
<title>Global markets rally on dovish Fed stance</title>
<link>https://example.com/fed</link>
<pubDate>Thu, 18 Apr 2024 08:00:00 +0530</pubDate>
</item>
</channel>
</rss>"""


class TestNewsProviderParsing:
    def test_parse_rss_extracts_articles(self) -> None:
        provider = NewsProvider()
        articles = provider._parse_rss_xml(_SAMPLE_RSS, "test")
        assert len(articles) == 4

    def test_title_extracted_correctly(self) -> None:
        provider = NewsProvider()
        articles = provider._parse_rss_xml(_SAMPLE_RSS, "test")
        titles = [a.title for a in articles]
        assert "TCS reports strong Q4 results, beats estimates" in titles

    def test_cdata_stripped(self) -> None:
        provider = NewsProvider()
        articles = provider._parse_rss_xml(_SAMPLE_RSS, "test")
        jio_article = [a for a in articles if "Jio" in a.title]
        assert len(jio_article) == 1
        assert "<![CDATA[" not in jio_article[0].title

    def test_symbol_matching_tcs(self) -> None:
        provider = NewsProvider()
        articles = provider._parse_rss_xml(_SAMPLE_RSS, "test")
        tcs_articles = [a for a in articles if a.symbol == "TCS.NS"]
        assert len(tcs_articles) == 1

    def test_symbol_matching_reliance(self) -> None:
        provider = NewsProvider()
        articles = provider._parse_rss_xml(_SAMPLE_RSS, "test")
        rel_articles = [a for a in articles if a.symbol == "RELIANCE.NS"]
        assert len(rel_articles) == 1

    def test_market_wide_news_no_symbol(self) -> None:
        provider = NewsProvider()
        articles = provider._parse_rss_xml(_SAMPLE_RSS, "test")
        nifty_articles = [a for a in articles if "Nifty" in a.title]
        assert len(nifty_articles) == 1
        assert nifty_articles[0].symbol is None

    def test_content_hash_unique(self) -> None:
        provider = NewsProvider()
        articles = provider._parse_rss_xml(_SAMPLE_RSS, "test")
        hashes = [a.content_hash for a in articles]
        assert len(set(hashes)) == len(hashes)

    def test_to_daily_articles_format(self) -> None:
        provider = NewsProvider()
        articles = provider._parse_rss_xml(_SAMPLE_RSS, "test")
        daily = provider.to_daily_articles(articles)
        assert isinstance(daily, dict)
        assert all(isinstance(v, list) for v in daily.values())

    def test_to_daily_articles_filters_by_symbol(self) -> None:
        provider = NewsProvider()
        articles = provider._parse_rss_xml(_SAMPLE_RSS, "test")
        daily = provider.to_daily_articles(articles, symbol="TCS.NS")
        all_titles = [t for titles in daily.values() for t in titles]
        assert any("TCS" in t for t in all_titles)
        assert not any("Jio" in t for t in all_titles)
