"""Tests for semi-auto live mode — Telegram-approved execution."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest

from alphavedha.execution.broker import OrderSide, PaperBroker
from alphavedha.execution.kill_switch import KillSwitch, KillSwitchConfig
from alphavedha.execution.oms import OrderManager, OrderPlan
from alphavedha.execution.semi_auto import (
    ApprovalStatus,
    PendingApproval,
    SemiAutoConfig,
    SemiAutoRunner,
)


@pytest.fixture
def oms() -> OrderManager:
    ks = KillSwitch(KillSwitchConfig())
    broker = PaperBroker(initial_capital=50_000.0)
    return OrderManager(broker=broker, kill_switch=ks, equity=50_000.0)


@pytest.fixture
def bot() -> AsyncMock:
    mock_bot = AsyncMock()
    mock_bot.send_message.return_value = {
        "ok": True,
        "result": {"message_id": 100},
    }
    return mock_bot


@pytest.fixture
def runner(oms: OrderManager, bot: AsyncMock) -> SemiAutoRunner:
    return SemiAutoRunner(oms=oms, telegram_bot=bot)


def _sample_plan(
    symbol: str = "TCS.NS",
    position_value: float = 5_000.0,
) -> OrderPlan:
    return OrderPlan(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=2,
        entry_price=3500.0,
        target_price=3700.0,
        stop_price=3300.0,
        position_value=position_value,
        position_pct=10.0,
        strategy="ensemble_v1",
        prediction_date=date.today(),
    )


class TestValidation:
    def test_valid_plan(self, runner: SemiAutoRunner) -> None:
        plan = _sample_plan(position_value=8_000.0)
        assert runner.validate_plan(plan) is None

    def test_exceeds_position_cap(self, runner: SemiAutoRunner) -> None:
        plan = _sample_plan(position_value=15_000.0)
        error = runner.validate_plan(plan)
        assert error is not None
        assert "exceeds cap" in error

    def test_custom_cap(self, oms: OrderManager, bot: AsyncMock) -> None:
        runner = SemiAutoRunner(
            oms=oms,
            telegram_bot=bot,
            config=SemiAutoConfig(position_cap=5_000.0),
        )
        plan = _sample_plan(position_value=6_000.0)
        assert runner.validate_plan(plan) is not None


class TestSendForApproval:
    @pytest.mark.asyncio
    async def test_sends_telegram_messages(self, runner: SemiAutoRunner, bot: AsyncMock) -> None:
        plans = [_sample_plan("TCS.NS"), _sample_plan("INFY.NS")]
        pending = await runner.send_for_approval(plans)
        assert len(pending) == 2
        assert bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_pending_status(self, runner: SemiAutoRunner) -> None:
        plans = [_sample_plan()]
        pending = await runner.send_for_approval(plans)
        assert pending[0].status == ApprovalStatus.PENDING
        assert pending[0].callback_id == "sa_1"

    @pytest.mark.asyncio
    async def test_captures_message_id(self, runner: SemiAutoRunner) -> None:
        plans = [_sample_plan()]
        pending = await runner.send_for_approval(plans)
        assert pending[0].telegram_message_id == 100

    @pytest.mark.asyncio
    async def test_rejects_over_cap(self, runner: SemiAutoRunner, bot: AsyncMock) -> None:
        plans = [_sample_plan(position_value=15_000.0)]
        pending = await runner.send_for_approval(plans)
        assert len(pending) == 0
        bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_caps_signals_per_day(self, oms: OrderManager, bot: AsyncMock) -> None:
        runner = SemiAutoRunner(
            oms=oms,
            telegram_bot=bot,
            config=SemiAutoConfig(max_signals_per_day=2),
        )
        plans = [_sample_plan(f"SYM{i}.NS") for i in range(5)]
        pending = await runner.send_for_approval(plans)
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_inline_buttons(self, runner: SemiAutoRunner, bot: AsyncMock) -> None:
        plans = [_sample_plan()]
        await runner.send_for_approval(plans)
        call_kwargs = bot.send_message.call_args[1]
        markup = call_kwargs["reply_markup"]
        buttons = markup["inline_keyboard"][0]
        assert buttons[0]["text"] == "Approve"
        assert buttons[1]["text"] == "Deny"
        assert "approve:sa_1" in buttons[0]["callback_data"]
        assert "deny:sa_1" in buttons[1]["callback_data"]


class TestProcessCallback:
    def test_approve(self, runner: SemiAutoRunner) -> None:
        pending = [
            PendingApproval(
                plan=_sample_plan(),
                callback_id="sa_1",
            )
        ]
        result = runner.process_callback("approve:sa_1", pending)
        assert result is not None
        assert result.status == ApprovalStatus.APPROVED

    def test_deny(self, runner: SemiAutoRunner) -> None:
        pending = [
            PendingApproval(
                plan=_sample_plan(),
                callback_id="sa_1",
            )
        ]
        result = runner.process_callback("deny:sa_1", pending)
        assert result is not None
        assert result.status == ApprovalStatus.DENIED

    def test_unknown_callback(self, runner: SemiAutoRunner) -> None:
        pending = [
            PendingApproval(
                plan=_sample_plan(),
                callback_id="sa_1",
            )
        ]
        result = runner.process_callback("approve:sa_99", pending)
        assert result is None

    def test_invalid_format(self, runner: SemiAutoRunner) -> None:
        result = runner.process_callback("garbage", [])
        assert result is None


class TestExecuteApproved:
    @pytest.mark.asyncio
    async def test_executes_approved(self, runner: SemiAutoRunner, bot: AsyncMock) -> None:
        pending = [
            PendingApproval(
                plan=_sample_plan(),
                callback_id="sa_1",
                status=ApprovalStatus.APPROVED,
            )
        ]
        result = await runner.execute_approved(pending)
        assert result.approved == 1
        assert result.denied == 0

    @pytest.mark.asyncio
    async def test_skips_denied(self, runner: SemiAutoRunner, bot: AsyncMock) -> None:
        pending = [
            PendingApproval(
                plan=_sample_plan(),
                callback_id="sa_1",
                status=ApprovalStatus.DENIED,
            )
        ]
        result = await runner.execute_approved(pending)
        assert result.approved == 0
        assert result.denied == 1
        assert len(result.executed) == 0

    @pytest.mark.asyncio
    async def test_expires_pending(self, runner: SemiAutoRunner, bot: AsyncMock) -> None:
        pending = [
            PendingApproval(
                plan=_sample_plan(),
                callback_id="sa_1",
                status=ApprovalStatus.PENDING,
            )
        ]
        result = await runner.execute_approved(pending)
        assert result.expired == 1
        assert pending[0].status == ApprovalStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_sends_summary(self, runner: SemiAutoRunner, bot: AsyncMock) -> None:
        pending = [
            PendingApproval(
                plan=_sample_plan(),
                callback_id="sa_1",
                status=ApprovalStatus.APPROVED,
            )
        ]
        await runner.execute_approved(pending)
        last_call = bot.send_message.call_args_list[-1]
        msg = last_call[0][0]
        assert "Semi-Auto Summary" in msg

    @pytest.mark.asyncio
    async def test_mixed_statuses(self, runner: SemiAutoRunner, bot: AsyncMock) -> None:
        pending = [
            PendingApproval(
                plan=_sample_plan("TCS.NS"),
                callback_id="sa_1",
                status=ApprovalStatus.APPROVED,
            ),
            PendingApproval(
                plan=_sample_plan("INFY.NS"),
                callback_id="sa_2",
                status=ApprovalStatus.DENIED,
            ),
            PendingApproval(
                plan=_sample_plan("HDFC.NS"),
                callback_id="sa_3",
                status=ApprovalStatus.PENDING,
            ),
        ]
        result = await runner.execute_approved(pending)
        assert result.approved == 1
        assert result.denied == 1
        assert result.expired == 1
        assert result.signals_sent == 3
