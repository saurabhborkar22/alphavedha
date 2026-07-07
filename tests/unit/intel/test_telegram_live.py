"""Tests for the live Telegram watcher — severity classification and alerting."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from alphavedha.exceptions import ConfigError
from alphavedha.intel.telegram_live import (
    WatchConfig,
    _maybe_alert,
    classify_severity,
    format_alert,
    handle_event_text,
)


class TestClassifySeverity:
    @pytest.mark.parametrize(
        "text",
        [
            "SEBI bars promoter from securities market",
            "ED raids company premises in money laundering probe",
            "Auditor resigns citing governance concerns",
            "Company defaults on NCD interest payment",
            "USFDA issues warning letter for Baddi plant",
            "Workers strike halts production at Pune facility",
            "Promoter pledge invoked by lenders",
        ],
    )
    def test_critical(self, text: str) -> None:
        result = classify_severity(text)
        assert result is not None
        assert result[0] == "critical"

    @pytest.mark.parametrize(
        "text",
        [
            "CFO resigns to pursue other opportunities",
            "Crisil downgrades long-term rating outlook",
            "Promoters plan stake sale via block deal",
        ],
    )
    def test_notable(self, text: str) -> None:
        result = classify_severity(text)
        assert result is not None
        assert result[0] == "notable"

    @pytest.mark.parametrize(
        "text",
        [
            "Q1 net profit rises 12% YoY, beats estimates",
            "Fed raises rates by 25 bps",  # 'Fed' must not trip the ED acronym
            "Nifty ends flat in rangebound trade",
            "",
        ],
    )
    def test_routine(self, text: str) -> None:
        assert classify_severity(text) is None

    def test_acronyms_require_uppercase(self) -> None:
        assert classify_severity("The board edited its charter") is None
        result = classify_severity("CBI registers case against vendor")
        assert result is not None
        assert result[0] == "critical"


class TestFormatAlert:
    def test_marks_open_positions(self) -> None:
        text = format_alert(
            symbols=["RELIANCE", "TCS"],
            severity="critical",
            terms=["fraud"],
            headline="Fraud alleged at unit",
            channel="moneycontrolcom",
            url="https://t.me/moneycontrolcom/9",
            open_positions={"TCS"},
        )
        assert "🚨" in text
        assert "<b>RELIANCE</b>" in text
        assert "<b>TCS</b> [OPEN]" in text
        assert "@moneycontrolcom" in text


class TestMaybeAlert:
    async def test_critical_always_alerts(self) -> None:
        bot = AsyncMock()
        watchlist = AsyncMock()
        watchlist.symbols.return_value = set()
        alerted = await _maybe_alert(
            bot, watchlist, ["INFY"], ("critical", ["fraud"]), "h", "c", "u"
        )
        assert alerted is True
        bot.send_message.assert_awaited_once()

    async def test_notable_needs_open_position(self) -> None:
        bot = AsyncMock()
        watchlist = AsyncMock()
        watchlist.symbols.return_value = {"TCS"}
        assert (
            await _maybe_alert(bot, watchlist, ["INFY"], ("notable", ["resigns"]), "h", "c", "u")
            is False
        )
        assert (
            await _maybe_alert(bot, watchlist, ["TCS"], ("notable", ["resigns"]), "h", "c", "u")
            is True
        )

    async def test_no_bot_no_alert(self) -> None:
        watchlist = AsyncMock()
        assert (
            await _maybe_alert(None, watchlist, ["TCS"], ("critical", ["fraud"]), "h", "c", "u")
            is False
        )


class TestHandleEventText:
    async def test_stores_and_alerts_on_critical(self) -> None:
        bot = AsyncMock()
        watchlist = AsyncMock()
        watchlist.symbols.return_value = set()
        with patch(
            "alphavedha.intel.store.store_disclosures", new=AsyncMock(return_value=1)
        ) as store:
            result = await handle_event_text(
                channel="moneycontrolcom",
                msg_id=7,
                text="SEBI bars Infosys unit executive from markets",
                posted_at=datetime(2026, 7, 7, 9, 0, tzinfo=UTC),
                bot=bot,
                watchlist=watchlist,
            )
        assert result["symbols"] == ["INFY"]
        assert result["alerted"] is True
        store.assert_awaited_once()
        bot.send_message.assert_awaited_once()

    async def test_unmatched_message_is_noop(self) -> None:
        result = await handle_event_text(
            channel="c",
            msg_id=1,
            text="Rupee weakens past 84 per dollar",
            posted_at=datetime(2026, 7, 7, 9, 0, tzinfo=UTC),
            bot=None,
            watchlist=AsyncMock(),
        )
        assert result == {"symbols": [], "stored": 0, "alerted": False}


class TestWatchConfig:
    def test_missing_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
        monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)
        with pytest.raises(ConfigError):
            WatchConfig.from_env()

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_API_ID", "12345")
        monkeypatch.setenv("TELEGRAM_API_HASH", "abcdef")
        monkeypatch.setenv("TELEGRAM_SESSION", "/tmp/x")
        monkeypatch.setenv("TELEGRAM_NEWS_CHANNELS", "foo,bar")
        config = WatchConfig.from_env()
        assert config.api_id == 12345
        assert config.session_path == "/tmp/x"
        assert config.channels == ["foo", "bar"]
