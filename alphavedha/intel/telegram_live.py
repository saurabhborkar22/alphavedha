"""Live Telegram news watcher — real-time red-flag alerts on universe symbols.

Connects to Telegram as a user account (MTProto via Telethon), listens to the
configured public news channels, and on every message:

1. maps the text to universe symbols (``intel.symbols``),
2. stores matched messages as disclosures (same rows the polling collector
   writes — the unique constraint dedupes whichever lands first), and
3. pushes an alert through the execution-engine Telegram bot when the message
   trips a red-flag pattern: *critical* severity always alerts, *notable*
   alerts only for symbols with an open paper trade.

Requires TELEGRAM_API_ID / TELEGRAM_API_HASH (from https://my.telegram.org)
and a one-time interactive login to create the session file:

    python -m alphavedha.intel.telegram_live login
    python -m alphavedha.intel.telegram_live watch

Telethon is an optional dependency (``pip install telethon``); the module
imports it lazily so the rest of the package never needs it.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import structlog

from alphavedha.exceptions import ConfigError
from alphavedha.intel.collectors.telegram_news import (
    _MAX_TEXT_CHARS,
    TelegramMessage,
    configured_channels,
    message_rows,
)

logger = structlog.get_logger(__name__)

_WATCHLIST_TTL_SECONDS = 300.0

# ---------------------------------------------------------------------------
# Red-flag classification
# ---------------------------------------------------------------------------

_CRITICAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bfraud\b",
        r"\bscam\b",
        r"\braid(?:s|ed)?\b",
        r"\benforcement directorate\b",
        r"SEBI\s+(?:probe|investigat\w*|order|bar(?:s|red)?|ban(?:s|ned)?)",
        r"\binsolvency\b",
        r"\bbankruptcy\b",
        r"\bNCLT\b",
        r"\bdefault(?:s|ed)?\b",
        r"auditor\s+resign\w*",
        r"pledge[sd]?\s+invoked|invocation\s+of\s+pledge",
        r"\bfire\b|\bblast\b|\bexplosion\b",
        r"(?:plant|production|operations?)\s+(?:shut|halt|closure|suspended)\w*",
        r"(?:workers?|labour|employees?)\s+strike",
        r"\brecalls?\b",
        r"\barrest(?:s|ed)?\b",
        r"show[-\s]cause",
        r"(?:GST|tax)\s+(?:notice|raid|survey|demand)",
        r"\bpenalt(?:y|ies)\b|\bfined\b",
        r"warning\s+letter|import\s+alert|Form\s+483",
        r"data\s+breach|cyber\s?attack|ransomware",
        r"whistle-?blower",
    )
)

# Agency acronyms only count in exact uppercase — "ed"/"cbi" appear in words.
_CRITICAL_CASE_SENSITIVE: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bED\b"),
    re.compile(r"\bCBI\b"),
)

_NOTABLE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"resign(?:s|ed|ation)?\b",
        r"\bdowngrade[sd]?\b",
        r"profit\s+warning|guidance\s+cut|cuts?\s+guidance",
        r"stake\s+sale|block\s+deal|bulk\s+deal",
        r"\bpledge[sd]?\b",
        r"open\s+offer",
        r"insider\s+trading",
        r"credit\s?watch|rating\s+watch",
    )
)


def classify_severity(text: str) -> tuple[str, list[str]] | None:
    """Return ("critical"|"notable", matched terms) or None for routine news."""
    if not text:
        return None

    critical_hits = [m.group(0) for p in _CRITICAL_PATTERNS if (m := p.search(text))]
    critical_hits += [m.group(0) for p in _CRITICAL_CASE_SENSITIVE if (m := p.search(text))]
    if critical_hits:
        return ("critical", critical_hits)

    notable_hits = [m.group(0) for p in _NOTABLE_PATTERNS if (m := p.search(text))]
    if notable_hits:
        return ("notable", notable_hits)
    return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchConfig:
    api_id: int
    api_hash: str
    session_path: str
    channels: list[str]

    @classmethod
    def from_env(cls) -> WatchConfig:
        api_id_raw = os.environ.get("TELEGRAM_API_ID", "")
        api_hash = os.environ.get("TELEGRAM_API_HASH", "")
        if not api_id_raw.isdigit() or not api_hash:
            raise ConfigError(
                "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set "
                "(create them at https://my.telegram.org)"
            )
        return cls(
            api_id=int(api_id_raw),
            api_hash=api_hash,
            session_path=os.environ.get("TELEGRAM_SESSION", "telegram_news"),
            channels=configured_channels(),
        )


def _build_client(config: WatchConfig) -> Any:
    try:
        from telethon import TelegramClient
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConfigError("telethon is not installed — pip install 'alphavedha[telegram]'") from exc
    return TelegramClient(config.session_path, config.api_id, config.api_hash)


# ---------------------------------------------------------------------------
# Watchlist (open paper trades) — refreshed lazily with a TTL
# ---------------------------------------------------------------------------


class _Watchlist:
    def __init__(self) -> None:
        self._symbols: set[str] = set()
        self._loaded_at = 0.0

    async def symbols(self) -> set[str]:
        if time.monotonic() - self._loaded_at < _WATCHLIST_TTL_SECONDS:
            return self._symbols
        try:
            from alphavedha.data.store import load_paper_trades

            df = await load_paper_trades()
            if not df.empty and "exit_price" in df.columns:
                open_trades = df[df["exit_price"].isna()]
                self._symbols = set(open_trades["symbol"].astype(str))
            else:
                self._symbols = set()
        except Exception as exc:
            logger.warning("watchlist_load_failed", error=str(exc))
        self._loaded_at = time.monotonic()
        return self._symbols


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------


def format_alert(
    symbols: list[str],
    severity: str,
    terms: list[str],
    headline: str,
    channel: str,
    url: str,
    open_positions: set[str],
) -> str:
    """Build the HTML alert message for the execution bot."""
    icon = "🚨" if severity == "critical" else "⚠️"
    tagged = ", ".join(f"<b>{s}</b>" + (" [OPEN]" if s in open_positions else "") for s in symbols)
    return (
        f"{icon} {tagged} — {severity} news\n"
        f"<i>{headline[:300]}</i>\n"
        f"matched: {', '.join(terms[:5])}\n"
        f"via @{channel} — {url}"
    )


async def _maybe_alert(
    bot: Any | None,
    watchlist: _Watchlist,
    symbols: list[str],
    severity_result: tuple[str, list[str]] | None,
    headline: str,
    channel: str,
    url: str,
) -> bool:
    if bot is None or severity_result is None:
        return False

    severity, terms = severity_result
    open_positions = await watchlist.symbols()
    if severity != "critical" and not any(s in open_positions for s in symbols):
        return False

    text = format_alert(symbols, severity, terms, headline, channel, url, open_positions)
    await bot.send_message(text)
    return True


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------


async def handle_event_text(
    channel: str,
    msg_id: int,
    text: str,
    posted_at: Any,
    bot: Any | None,
    watchlist: _Watchlist,
) -> dict[str, Any]:
    """Process one channel message: store matched rows, alert on red flags."""
    msg = TelegramMessage(
        channel=channel,
        msg_id=msg_id,
        text=text[:_MAX_TEXT_CHARS],
        posted_at=posted_at,
        url=f"https://t.me/{channel}/{msg_id}",
    )
    rows = message_rows([msg])
    if not rows:
        return {"symbols": [], "stored": 0, "alerted": False}

    symbols = sorted({str(r["symbol"]) for r in rows})

    from alphavedha.intel.store import store_disclosures

    stored = await store_disclosures(rows)

    severity_result = classify_severity(text)
    headline = text.split("\n", 1)[0].strip()
    alerted = await _maybe_alert(
        bot, watchlist, symbols, severity_result, headline, channel, msg.url
    )

    logger.info(
        "telegram_live_message",
        channel=channel,
        symbols=symbols,
        severity=severity_result[0] if severity_result else None,
        alerted=alerted,
    )
    return {"symbols": symbols, "stored": stored, "alerted": alerted}


def _build_alert_bot() -> Any | None:
    from alphavedha.execution.telegram import TelegramBot, TelegramConfig

    config = TelegramConfig.from_env()
    if config is None:
        logger.warning("alert_bot_disabled", reason="TELEGRAM_BOT_TOKEN/CHAT_ID not set")
        return None
    return TelegramBot(config)


async def run_news_watch() -> None:
    """Connect and listen until disconnected. Raises ConfigError when unusable."""
    config = WatchConfig.from_env()
    client = _build_client(config)  # raises ConfigError if telethon missing

    from telethon import events

    bot = _build_alert_bot()
    watchlist = _Watchlist()

    await client.connect()
    if not await client.is_user_authorized():
        raise ConfigError(
            "Telegram session not authorized — run "
            "`python -m alphavedha.intel.telegram_live login` once first"
        )

    @client.on(events.NewMessage(chats=config.channels))
    async def _on_message(event: Any) -> None:
        try:
            text = (event.raw_text or "").strip()
            if not text:
                return
            chat = await event.get_chat()
            channel = getattr(chat, "username", None) or str(event.chat_id)
            await handle_event_text(
                channel=channel,
                msg_id=event.id,
                text=text,
                posted_at=event.message.date,
                bot=bot,
                watchlist=watchlist,
            )
        except Exception as exc:
            logger.error("telegram_live_handler_failed", error=str(exc))

    logger.info(
        "telegram_live_started",
        channels=config.channels,
        alerts_enabled=bot is not None,
    )
    await client.run_until_disconnected()


async def login() -> None:
    """One-time interactive login — creates the Telethon session file."""
    config = WatchConfig.from_env()
    client = _build_client(config)
    await client.start()  # prompts for phone + code on the terminal
    me = await client.get_me()
    print(f"Logged in as {getattr(me, 'username', None) or getattr(me, 'phone', '?')}")
    print(f"Session saved to {config.session_path}.session")
    await client.disconnect()


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "watch"
    if command == "login":
        asyncio.run(login())
    elif command == "watch":
        asyncio.run(run_news_watch())
    else:
        print(f"Unknown command: {command}. Use 'login' or 'watch'.")
        sys.exit(2)


if __name__ == "__main__":
    main()
