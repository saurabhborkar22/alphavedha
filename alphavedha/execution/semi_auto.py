"""Semi-auto live mode — Telegram-approved execution.

In semi-auto mode, every order requires explicit human approval via
Telegram inline buttons before the OMS sends it to the broker. This is
the first live tier (₹50,000 capital, position cap ₹10k).

Flow:
  1. SemiAutoRunner receives gate-passed signals
  2. Sends each as a Telegram message with Approve/Deny buttons
  3. Polls for callback responses (button taps)
  4. Approved signals → OMS execute_plan() → real broker
  5. Denied signals → logged, skipped
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import structlog

from alphavedha.execution.oms import OmsResult, OrderManager, OrderPlan

logger = structlog.get_logger(__name__)

_POSITION_CAP_LIVE_SMALL: float = 10_000.0


class ApprovalStatus:
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class PendingApproval:
    """An order plan waiting for Telegram approval."""

    plan: OrderPlan
    callback_id: str
    status: str = ApprovalStatus.PENDING
    telegram_message_id: int | None = None


@dataclass
class SemiAutoResult:
    """Result of a semi-auto execution run."""

    run_date: date
    signals_sent: int = 0
    approved: int = 0
    denied: int = 0
    expired: int = 0
    executed: list[OmsResult] = field(default_factory=list)
    blocked: list[OmsResult] = field(default_factory=list)


@dataclass(frozen=True)
class SemiAutoConfig:
    """Configuration for semi-auto live mode."""

    capital: float = 50_000.0
    position_cap: float = _POSITION_CAP_LIVE_SMALL
    approval_timeout_seconds: int = 300
    max_signals_per_day: int = 5


class SemiAutoRunner:
    """Orchestrates Telegram-approved order execution.

    Usage:
        runner = SemiAutoRunner(oms, bot, config)
        plans = [oms.compute_plan(...) for signal in signals]
        approvals = await runner.send_for_approval(plans)
        # ... user taps approve/deny on Telegram ...
        result = await runner.execute_approved(approvals)
    """

    def __init__(
        self,
        oms: OrderManager,
        telegram_bot: Any,
        config: SemiAutoConfig | None = None,
    ) -> None:
        self._oms = oms
        self._bot = telegram_bot
        self._config = config or SemiAutoConfig()
        self._counter = 0

    def _next_callback_id(self) -> str:
        self._counter += 1
        return f"sa_{self._counter}"

    def validate_plan(self, plan: OrderPlan) -> str | None:
        """Check plan against semi-auto constraints.

        Returns None if valid, or an error message.
        """
        if plan.position_value > self._config.position_cap:
            return (
                f"Position value ₹{plan.position_value:,.0f} exceeds "
                f"cap ₹{self._config.position_cap:,.0f}"
            )
        return None

    async def send_for_approval(
        self,
        plans: list[OrderPlan],
        run_date: date | None = None,
    ) -> list[PendingApproval]:
        """Send order plans to Telegram for approval.

        Returns list of PendingApproval objects to track responses.
        """
        pending: list[PendingApproval] = []

        capped = plans[: self._config.max_signals_per_day]
        if len(plans) > self._config.max_signals_per_day:
            logger.warning(
                "semi_auto_signals_capped",
                total=len(plans),
                cap=self._config.max_signals_per_day,
            )

        for plan in capped:
            validation_error = self.validate_plan(plan)
            if validation_error:
                logger.warning(
                    "semi_auto_plan_rejected",
                    symbol=plan.symbol,
                    reason=validation_error,
                )
                continue

            callback_id = self._next_callback_id()
            text = (
                f"<b>Order Approval</b>\n"
                f"{'🟢' if plan.side.value == 'BUY' else '🔴'} <b>{plan.symbol}</b> "
                f"{plan.side.value}\n"
                f"Qty: {plan.quantity} @ ₹{plan.entry_price:,.2f}\n"
                f"Target: ₹{plan.target_price:,.2f} | Stop: ₹{plan.stop_price:,.2f}\n"
                f"Position: ₹{plan.position_value:,.0f} ({plan.position_pct:.1f}%)\n"
                f"Strategy: {plan.strategy}\n"
                f"\n<i>Tap to approve or deny.</i>"
            )

            reply_markup = {
                "inline_keyboard": [
                    [
                        {
                            "text": "Approve",
                            "callback_data": f"approve:{callback_id}",
                        },
                        {
                            "text": "Deny",
                            "callback_data": f"deny:{callback_id}",
                        },
                    ]
                ]
            }

            result = await self._bot.send_message(text, reply_markup=reply_markup)
            msg_id = None
            if result and "result" in result:
                msg_id = result["result"].get("message_id")

            approval = PendingApproval(
                plan=plan,
                callback_id=callback_id,
                telegram_message_id=msg_id,
            )
            pending.append(approval)

            logger.info(
                "semi_auto_sent_for_approval",
                symbol=plan.symbol,
                callback_id=callback_id,
                position_value=plan.position_value,
            )

        return pending

    def process_callback(
        self,
        callback_data: str,
        pending: list[PendingApproval],
    ) -> PendingApproval | None:
        """Process a Telegram callback button press.

        Returns the matching PendingApproval (with updated status) or None.
        """
        parts = callback_data.split(":", 1)
        if len(parts) != 2:
            return None

        action, callback_id = parts
        for approval in pending:
            if approval.callback_id == callback_id:
                if action == "approve":
                    approval.status = ApprovalStatus.APPROVED
                elif action == "deny":
                    approval.status = ApprovalStatus.DENIED
                logger.info(
                    "semi_auto_callback",
                    callback_id=callback_id,
                    action=action,
                    symbol=approval.plan.symbol,
                )
                return approval
        return None

    async def execute_approved(
        self,
        pending: list[PendingApproval],
    ) -> SemiAutoResult:
        """Execute all approved plans, skip denied/expired ones."""
        result = SemiAutoResult(run_date=date.today())
        result.signals_sent = len(pending)

        for approval in pending:
            if approval.status == ApprovalStatus.APPROVED:
                result.approved += 1
                oms_result = await self._oms.execute_plan(approval.plan)
                if oms_result.blocked:
                    result.blocked.append(oms_result)
                    await self._bot.send_message(
                        f"Blocked: {approval.plan.symbol} — {oms_result.block_reason}"
                    )
                else:
                    result.executed.append(oms_result)
                    await self._bot.send_message(
                        f"Executed: {approval.plan.symbol} "
                        f"{approval.plan.side.value} {approval.plan.quantity} "
                        f"@ ₹{approval.plan.entry_price:,.2f}"
                    )
            elif approval.status == ApprovalStatus.DENIED:
                result.denied += 1
            else:
                result.expired += 1
                approval.status = ApprovalStatus.EXPIRED

        summary = (
            f"<b>Semi-Auto Summary</b>\n"
            f"Sent: {result.signals_sent} | "
            f"Approved: {result.approved} | "
            f"Denied: {result.denied} | "
            f"Expired: {result.expired}\n"
            f"Executed: {len(result.executed)} | "
            f"Blocked: {len(result.blocked)}"
        )
        await self._bot.send_message(summary)

        logger.info(
            "semi_auto_run_complete",
            signals=result.signals_sent,
            approved=result.approved,
            executed=len(result.executed),
            blocked=len(result.blocked),
        )

        return result
