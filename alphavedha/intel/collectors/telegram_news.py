"""Telegram public-channel news collector — polls t.me/s/ preview pages.

Telegram exposes the last ~20 messages of any public channel at
``https://t.me/s/<channel>`` with no account or API key. This collector
fetches the preview pages of configured financial-news channels, maps each
message to universe symbols via ``intel.symbols``, and upserts matched
messages into ``disclosures`` (source="TELEGRAM", category="news") where the
nightly LLM extraction batch picks them up like any exchange filing.

The (symbol, source, filed_at, headline) unique constraint makes repeated
polls idempotent. Messages mentioning no universe symbol are dropped.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

import httpx
import structlog

from alphavedha.intel.symbols import match_symbols

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_CHANNELS: tuple[str, ...] = ("moneycontrolcom", "ndtvprofitnews", "cnbc_tv18")

_PREVIEW_URL = "https://t.me/s/{channel}"
_HTTP_TIMEOUT = 15.0
_INTER_CHANNEL_DELAY_SECONDS = 1.0
_MAX_TEXT_CHARS = 8000
_MAX_HEADLINE_CHARS = 500

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
}


def configured_channels() -> list[str]:
    """Channel handles to poll — TELEGRAM_NEWS_CHANNELS env (csv) or defaults."""
    raw = os.environ.get("TELEGRAM_NEWS_CHANNELS", "")
    channels = [c.strip().lstrip("@") for c in raw.split(",") if c.strip()]
    return channels or list(DEFAULT_CHANNELS)


@dataclass(frozen=True)
class TelegramMessage:
    channel: str
    msg_id: int
    text: str
    posted_at: datetime  # tz-aware
    url: str


@dataclass
class _ParsedMessage:
    data_post: str = ""
    datetime_attr: str = ""
    text_parts: list[str] = field(default_factory=list)


class _PreviewParser(HTMLParser):
    """Extracts (data-post, datetime, text) triples from a t.me/s/ page.

    Message blocks are ``div[data-post="channel/id"]``; the body lives in a
    nested ``div.tgme_widget_message_text`` and the timestamp in a ``time``
    element with a ``datetime`` attribute.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.messages: list[_ParsedMessage] = []
        self._current: _ParsedMessage | None = None
        self._msg_div_depth = 0
        self._text_div_depth = 0

    @property
    def _capturing_text(self) -> bool:
        return self._text_div_depth > 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: v or "" for k, v in attrs}

        if tag == "br" and self._capturing_text and self._current is not None:
            self._current.text_parts.append("\n")
            return

        if tag == "time" and self._current is not None and attr_map.get("datetime"):
            if not self._current.datetime_attr:
                self._current.datetime_attr = attr_map["datetime"]
            return

        if tag != "div":
            return

        if self._current is None:
            if attr_map.get("data-post"):
                self._current = _ParsedMessage(data_post=attr_map["data-post"])
                self._msg_div_depth = 1
            return

        self._msg_div_depth += 1
        classes = attr_map.get("class", "")
        if self._capturing_text:
            self._text_div_depth += 1
        elif "tgme_widget_message_text" in classes:
            self._text_div_depth = 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br" and self._capturing_text and self._current is not None:
            self._current.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag != "div" or self._current is None:
            return

        if self._capturing_text:
            self._text_div_depth -= 1

        self._msg_div_depth -= 1
        if self._msg_div_depth == 0:
            self.messages.append(self._current)
            self._current = None
            self._text_div_depth = 0

    def handle_data(self, data: str) -> None:
        if self._capturing_text and self._current is not None:
            self._current.text_parts.append(data)


def _parse_posted_at(datetime_attr: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(datetime_attr)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def parse_channel_messages(html: str, channel: str) -> list[TelegramMessage]:
    """Parse a t.me/s/ preview page into messages. Malformed blocks are skipped."""
    parser = _PreviewParser()
    parser.feed(html)

    messages: list[TelegramMessage] = []
    for parsed in parser.messages:
        text = "".join(parsed.text_parts).strip()
        if not text or not parsed.datetime_attr:
            continue

        posted_at = _parse_posted_at(parsed.datetime_attr)
        if posted_at is None:
            logger.debug("telegram_bad_timestamp", channel=channel, raw=parsed.datetime_attr)
            continue

        _, _, id_part = parsed.data_post.rpartition("/")
        if not id_part.isdigit():
            continue

        messages.append(
            TelegramMessage(
                channel=channel,
                msg_id=int(id_part),
                text=text[:_MAX_TEXT_CHARS],
                posted_at=posted_at,
                url=f"https://t.me/{parsed.data_post}",
            )
        )
    return messages


def message_rows(messages: list[TelegramMessage]) -> list[dict[str, object]]:
    """Convert messages to disclosure rows — one row per (message, matched symbol)."""
    rows: list[dict[str, object]] = []
    for msg in messages:
        symbols = match_symbols(msg.text)
        if not symbols:
            continue

        headline = msg.text.split("\n", 1)[0].strip()[:_MAX_HEADLINE_CHARS]
        if not headline:
            continue

        text_hash = hashlib.sha256(msg.text.encode()).hexdigest()
        for symbol in symbols:
            rows.append(
                {
                    "symbol": symbol,
                    "source": "TELEGRAM",
                    "category": "news",
                    "headline": headline,
                    "filed_at": msg.posted_at.astimezone(IST),
                    "url": msg.url,
                    "text": msg.text,
                    "text_hash": text_hash,
                }
            )
    return rows


async def fetch_channel_html(client: httpx.AsyncClient, channel: str) -> str | None:
    """Fetch a channel's preview page; None on any failure (logged, not raised)."""
    url = _PREVIEW_URL.format(channel=channel)
    try:
        resp = await client.get(url, headers=_HEADERS, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("telegram_fetch_failed", channel=channel, error=str(exc))
        return None


async def ingest_telegram_news(channels: list[str] | None = None) -> dict[str, int]:
    """Poll configured channels and upsert symbol-matched messages as disclosures."""
    from alphavedha.intel.store import store_disclosures

    channels = channels or configured_channels()
    all_messages: list[TelegramMessage] = []

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        for i, channel in enumerate(channels):
            if i > 0:
                await asyncio.sleep(_INTER_CHANNEL_DELAY_SECONDS)
            html = await fetch_channel_html(client, channel)
            if html is None:
                continue
            parsed = parse_channel_messages(html, channel)
            logger.debug("telegram_channel_parsed", channel=channel, messages=len(parsed))
            all_messages.extend(parsed)

    rows = message_rows(all_messages)
    stored = await store_disclosures(rows)

    counts = {"channels": len(channels), "messages": len(all_messages), "rows": stored}
    logger.info("telegram_news_ingested", **counts)
    return counts
