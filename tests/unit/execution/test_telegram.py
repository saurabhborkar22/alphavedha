"""Tests for the Telegram control plane."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from alphavedha.execution.kill_switch import KillSwitch, KillSwitchConfig
from alphavedha.execution.telegram import TelegramBot, TelegramConfig


@pytest.fixture
def config() -> TelegramConfig:
    return TelegramConfig(bot_token="test-token-123", allowed_chat_ids=[12345])


@pytest.fixture
def bot(config: TelegramConfig) -> TelegramBot:
    return TelegramBot(config)


class TestTelegramConfig:
    def test_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "tok-123", "TELEGRAM_CHAT_ID": "111,222"},
        ):
            cfg = TelegramConfig.from_env()
            assert cfg is not None
            assert cfg.bot_token == "tok-123"
            assert cfg.allowed_chat_ids == [111, 222]

    def test_from_env_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = TelegramConfig.from_env()
            assert cfg is None

    def test_from_env_empty_token(self) -> None:
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "111"},
        ):
            cfg = TelegramConfig.from_env()
            assert cfg is None


class TestAuthorization:
    def test_authorized_chat(self, bot: TelegramBot) -> None:
        assert bot._is_authorized(12345) is True

    def test_unauthorized_chat(self, bot: TelegramBot) -> None:
        assert bot._is_authorized(99999) is False


class TestParseCommand:
    def test_valid_command(self, bot: TelegramBot) -> None:
        update = {"message": {"text": "/status", "chat": {"id": 12345}}}
        result = bot.parse_command(update)
        assert result == ("/status", 12345)

    def test_command_with_bot_mention(self, bot: TelegramBot) -> None:
        update = {"message": {"text": "/status@AlphaVedhaBot", "chat": {"id": 12345}}}
        result = bot.parse_command(update)
        assert result == ("/status", 12345)

    def test_unauthorized_command(self, bot: TelegramBot) -> None:
        update = {"message": {"text": "/status", "chat": {"id": 99999}}}
        result = bot.parse_command(update)
        assert result is None

    def test_non_command(self, bot: TelegramBot) -> None:
        update = {"message": {"text": "hello", "chat": {"id": 12345}}}
        result = bot.parse_command(update)
        assert result is None

    def test_empty_message(self, bot: TelegramBot) -> None:
        update: dict[str, Any] = {"message": {}}
        result = bot.parse_command(update)
        assert result is None


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send_message_calls_api(self, bot: TelegramBot) -> None:
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await bot.send_message("test message")
            assert result == {"ok": True}
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_handles_error(self, bot: TelegramBot) -> None:
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("network error")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await bot.send_message("test")
            assert result is None


class TestSignalSummary:
    @pytest.mark.asyncio
    async def test_empty_signals(self, bot: TelegramBot) -> None:
        with patch.object(bot, "send_message", new_callable=AsyncMock) as mock_send:
            await bot.send_signal_summary([], "2026-06-21")
            mock_send.assert_called_once()
            msg = mock_send.call_args[0][0]
            assert "No gate-passed signals" in msg

    @pytest.mark.asyncio
    async def test_with_signals(self, bot: TelegramBot) -> None:
        signals = [
            {
                "symbol": "TCS.NS",
                "direction": 1,
                "magnitude": 0.03,
                "entry_price": 3500,
                "strategy": "ensemble_v1",
            },
            {
                "symbol": "INFY.NS",
                "direction": -1,
                "magnitude": 0.02,
                "entry_price": 1500,
                "strategy": "blowup_short_v1",
            },
        ]
        with patch.object(bot, "send_message", new_callable=AsyncMock) as mock_send:
            await bot.send_signal_summary(signals, "2026-06-21")
            msg = mock_send.call_args[0][0]
            assert "TCS.NS" in msg
            assert "LONG" in msg
            assert "SHORT" in msg
            assert "2 signals" in msg


class TestCommands:
    @pytest.mark.asyncio
    async def test_help_command(self, bot: TelegramBot) -> None:
        with patch.object(bot, "send_message", new_callable=AsyncMock) as mock_send:
            await bot.handle_command("/help", 12345)
            msg = mock_send.call_args[0][0]
            assert "/status" in msg
            assert "/panic" in msg

    @pytest.mark.asyncio
    async def test_unknown_command(self, bot: TelegramBot) -> None:
        with patch.object(bot, "send_message", new_callable=AsyncMock) as mock_send:
            await bot.handle_command("/unknown", 12345)
            msg = mock_send.call_args[0][0]
            assert "Unknown command" in msg

    @pytest.mark.asyncio
    async def test_halt_command(self, bot: TelegramBot) -> None:
        ks = KillSwitch(KillSwitchConfig())
        with patch.object(bot, "send_message", new_callable=AsyncMock) as mock_send:
            await bot.handle_command("/halt", 12345, kill_switch=ks)
            msg = mock_send.call_args[0][0]
            assert "HALT" in msg
            assert ks.state.halted is True

    @pytest.mark.asyncio
    async def test_resume_command(self, bot: TelegramBot) -> None:
        ks = KillSwitch(KillSwitchConfig())
        ks.halt()
        with patch.object(bot, "send_message", new_callable=AsyncMock) as mock_send:
            await bot.handle_command("/resume", 12345, kill_switch=ks)
            msg = mock_send.call_args[0][0]
            assert "CLEARED" in msg

    @pytest.mark.asyncio
    async def test_status_command(self, bot: TelegramBot) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig())
            ks.check(
                open_positions=3,
                daily_new_exposure_pct=15,
                daily_pnl_pct=-0.5,
                current_equity=990_000,
                peak_equity=1_000_000,
            )
        with patch.object(bot, "send_message", new_callable=AsyncMock) as mock_send:
            await bot.handle_command("/status", 12345, kill_switch=ks)
            msg = mock_send.call_args[0][0]
            assert "ACTIVE" in msg
            assert "Positions: 3" in msg

    @pytest.mark.asyncio
    async def test_status_no_kill_switch(self, bot: TelegramBot) -> None:
        with patch.object(bot, "send_message", new_callable=AsyncMock) as mock_send:
            await bot.handle_command("/status", 12345, kill_switch=None)
            msg = mock_send.call_args[0][0]
            assert "not available" in msg

    @pytest.mark.asyncio
    async def test_panic_command(self, bot: TelegramBot) -> None:
        from alphavedha.execution.broker import OrderSide, PaperBroker
        from alphavedha.execution.oms import OrderManager

        ks = KillSwitch(KillSwitchConfig())
        broker = PaperBroker()
        oms = OrderManager(broker=broker, kill_switch=ks)

        o1 = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(o1.order_id, 3500.0)

        with patch.object(bot, "send_message", new_callable=AsyncMock) as mock_send:
            await bot.handle_command("/panic", 12345, kill_switch=ks, oms=oms)
            msg = mock_send.call_args[0][0]
            assert "PANIC" in msg
            assert "1 flatten" in msg
            assert ks.state.halted is True

    @pytest.mark.asyncio
    async def test_positions_command_empty(self, bot: TelegramBot) -> None:
        from alphavedha.execution.broker import PaperBroker
        from alphavedha.execution.oms import OrderManager

        ks = KillSwitch()
        broker = PaperBroker()
        oms = OrderManager(broker=broker, kill_switch=ks)

        with patch.object(bot, "send_message", new_callable=AsyncMock) as mock_send:
            await bot.handle_command("/positions", 12345, oms=oms)
            msg = mock_send.call_args[0][0]
            assert "No open positions" in msg

    @pytest.mark.asyncio
    async def test_positions_command_with_holdings(self, bot: TelegramBot) -> None:
        from alphavedha.execution.broker import OrderSide, PaperBroker
        from alphavedha.execution.oms import OrderManager

        ks = KillSwitch()
        broker = PaperBroker()
        oms = OrderManager(broker=broker, kill_switch=ks)
        o1 = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(o1.order_id, 3500.0)

        with patch.object(bot, "send_message", new_callable=AsyncMock) as mock_send:
            await bot.handle_command("/positions", 12345, oms=oms)
            msg = mock_send.call_args[0][0]
            assert "TCS.NS" in msg
            assert "10" in msg
