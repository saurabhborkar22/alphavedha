"""Telegram control plane for the execution engine.

Sends daily signal summaries with approve/deny buttons, handles
operator commands (/status, /positions, /halt, /resume, /panic),
and enforces a chat-ID allowlist for auth.

Uses the Telegram Bot API directly via httpx — no heavy deps.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.telegram.org/bot{token}"


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    allowed_chat_ids: list[int]

    @classmethod
    def from_env(cls) -> TelegramConfig | None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_ids_raw = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_ids_raw:
            return None
        chat_ids = [int(cid.strip()) for cid in chat_ids_raw.split(",") if cid.strip()]
        return cls(bot_token=token, allowed_chat_ids=chat_ids)


class TelegramBot:
    """Telegram bot for execution engine control.

    Usage:
        bot = TelegramBot(config)
        await bot.send_signal_summary(signals)
        await bot.poll_and_handle(kill_switch, oms)
    """

    def __init__(self, config: TelegramConfig) -> None:
        self._config = config
        self._base_url = _BASE_URL.format(token=config.bot_token)
        self._last_update_id: int = 0

    def _is_authorized(self, chat_id: int) -> bool:
        return chat_id in self._config.allowed_chat_ids

    async def send_message(
        self,
        text: str,
        chat_id: int | None = None,
        parse_mode: str = "HTML",
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Send a message to a chat. Uses first allowed chat ID if none specified."""
        target = chat_id or self._config.allowed_chat_ids[0]
        payload: dict[str, Any] = {
            "chat_id": target,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{self._base_url}/sendMessage", json=payload)
                resp.raise_for_status()
                result: dict[str, Any] = resp.json()
                return result
        except Exception as e:
            logger.error("telegram_send_failed", error=str(e))
            return None

    async def send_signal_summary(
        self,
        signals: list[dict[str, Any]],
        run_date: str,
    ) -> None:
        """Send today's gate-passed signals with approve/deny summary."""
        if not signals:
            await self.send_message(f"<b>AlphaVedha {run_date}</b>\nNo gate-passed signals today.")
            return

        lines = [f"<b>AlphaVedha Signals — {run_date}</b>", ""]
        for sig in signals:
            direction = "LONG" if sig.get("direction", 0) >= 0 else "SHORT"
            symbol = sig.get("symbol", "?")
            magnitude = sig.get("magnitude", 0)
            strategy = sig.get("strategy", "?")
            entry = sig.get("entry_price", 0)
            lines.append(
                f"{'🟢' if direction == 'LONG' else '🔴'} <b>{symbol}</b> {direction} "
                f"| mag: {magnitude:.2%} | entry: ₹{entry:,.0f} | {strategy}"
            )

        lines.append(f"\n<i>{len(signals)} signals. Shadow mode — no real orders.</i>")
        await self.send_message("\n".join(lines))

    async def get_updates(self) -> list[dict[str, Any]]:
        """Long-poll for new messages/callbacks."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{self._base_url}/getUpdates",
                    params={
                        "offset": self._last_update_id + 1,
                        "timeout": 20,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                updates: list[dict[str, Any]] = data.get("result", [])
                if updates:
                    self._last_update_id = updates[-1]["update_id"]
                return updates
        except Exception as e:
            logger.error("telegram_poll_failed", error=str(e))
            return []

    def parse_command(self, update: dict[str, Any]) -> tuple[str, int] | None:
        """Extract command text and chat_id from an update.

        Returns (command, chat_id) or None if not a command.
        """
        message = update.get("message", {})
        text = message.get("text", "")
        chat_id = message.get("chat", {}).get("id", 0)

        if not text.startswith("/"):
            return None

        if not self._is_authorized(chat_id):
            logger.warning("telegram_unauthorized", chat_id=chat_id)
            return None

        command = text.split()[0].lower().split("@")[0]
        return (command, chat_id)

    async def handle_command(
        self,
        command: str,
        chat_id: int,
        kill_switch: Any | None = None,
        oms: Any | None = None,
    ) -> None:
        """Dispatch a command to the appropriate handler."""
        handlers: dict[str, str] = {
            "/status": "status",
            "/positions": "positions",
            "/halt": "halt",
            "/resume": "resume",
            "/panic": "panic",
            "/help": "help",
        }

        handler_name = handlers.get(command)
        if handler_name is None:
            await self.send_message(f"Unknown command: {command}. Try /help", chat_id)
            return

        if handler_name == "help":
            await self._cmd_help(chat_id)
        elif handler_name == "status":
            await self._cmd_status(chat_id, kill_switch)
        elif handler_name == "positions":
            await self._cmd_positions(chat_id, oms)
        elif handler_name == "halt":
            await self._cmd_halt(chat_id, kill_switch)
        elif handler_name == "resume":
            await self._cmd_resume(chat_id, kill_switch)
        elif handler_name == "panic":
            await self._cmd_panic(chat_id, kill_switch, oms)

    async def _cmd_help(self, chat_id: int) -> None:
        text = (
            "<b>AlphaVedha Commands</b>\n"
            "/status — Kill switch + system state\n"
            "/positions — Open positions\n"
            "/halt — Manual halt (block all orders)\n"
            "/resume — Clear manual halt\n"
            "/panic — Flatten all positions + halt"
        )
        await self.send_message(text, chat_id)

    async def _cmd_status(self, chat_id: int, kill_switch: Any | None) -> None:
        if kill_switch is None:
            await self.send_message("Kill switch not available.", chat_id)
            return
        state = kill_switch.state
        status = "HALTED" if state.halted else "ACTIVE"
        reasons = ", ".join(r.value for r in state.halt_reasons) if state.halt_reasons else "none"
        text = (
            f"<b>Status: {status}</b>\n"
            f"Halt reasons: {reasons}\n"
            f"Positions: {state.open_positions}\n"
            f"Daily P&L: {state.daily_pnl_pct:+.2f}%\n"
            f"Drawdown: {state.drawdown_pct:.2f}%\n"
            f"Enabled: {kill_switch.is_enabled}"
        )
        await self.send_message(text, chat_id)

    async def _cmd_positions(self, chat_id: int, oms: Any | None) -> None:
        if oms is None:
            await self.send_message("OMS not available.", chat_id)
            return
        positions = await oms._broker.get_positions()
        if not positions:
            await self.send_message("No open positions.", chat_id)
            return
        lines = ["<b>Open Positions</b>"]
        for pos in positions:
            lines.append(f"  {pos.symbol}: {pos.quantity} @ ₹{pos.avg_price:,.2f}")
        await self.send_message("\n".join(lines), chat_id)

    async def _cmd_halt(self, chat_id: int, kill_switch: Any | None) -> None:
        if kill_switch is None:
            await self.send_message("Kill switch not available.", chat_id)
            return
        kill_switch.halt()
        await self.send_message("Manual HALT activated. All new orders blocked.", chat_id)

    async def _cmd_resume(self, chat_id: int, kill_switch: Any | None) -> None:
        if kill_switch is None:
            await self.send_message("Kill switch not available.", chat_id)
            return
        kill_switch.resume()
        await self.send_message("Manual halt CLEARED. Orders will pass other checks.", chat_id)

    async def _cmd_panic(
        self,
        chat_id: int,
        kill_switch: Any | None,
        oms: Any | None,
    ) -> None:
        if kill_switch is None or oms is None:
            await self.send_message("Kill switch or OMS not available.", chat_id)
            return
        kill_switch.halt()
        orders = await oms.flatten_all()
        count = len(orders)
        await self.send_message(
            f"PANIC executed. {count} flatten orders placed. Engine HALTED.",
            chat_id,
        )
        logger.warning("telegram_panic_executed", flatten_orders=count)
