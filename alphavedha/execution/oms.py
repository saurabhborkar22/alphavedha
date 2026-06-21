"""Order Management System — converts signals into order plans.

Takes a gate-passed prediction (StockPrediction with is_tradeable=True),
computes position size, entry/exit prices, and produces an OrderPlan
that the broker adapter can execute.

All orders pass through the kill switch before being sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import structlog

from alphavedha.execution.broker import (
    BrokerAdapter,
    Fill,
    Order,
    OrderSide,
    OrderType,
)
from alphavedha.execution.kill_switch import KillSwitch, KillSwitchState

logger = structlog.get_logger(__name__)

_MAX_POSITION_PCT: float = 5.0


@dataclass(frozen=True)
class OrderPlan:
    """Computed order parameters ready for execution."""

    symbol: str
    side: OrderSide
    quantity: int
    entry_price: float
    target_price: float
    stop_price: float
    position_value: float
    position_pct: float
    strategy: str
    prediction_date: date


@dataclass
class OmsResult:
    """Outcome of an OMS execution attempt."""

    planned: OrderPlan | None = None
    order: Order | None = None
    fill: Fill | None = None
    blocked: bool = False
    block_reason: str = ""
    kill_switch_state: KillSwitchState | None = None


@dataclass
class OmsState:
    """Tracks daily OMS state for kill switch calculations."""

    daily_orders: list[OrderPlan] = field(default_factory=list)
    daily_fills: list[Fill] = field(default_factory=list)
    current_date: date | None = None


class OrderManager:
    """Converts signals to orders, enforcing kill switch at every step.

    Usage:
        oms = OrderManager(broker, kill_switch, equity=1_000_000)
        result = await oms.execute_signal(prediction, strategy="event_drift_v1")
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        kill_switch: KillSwitch,
        equity: float = 1_000_000.0,
    ) -> None:
        self._broker = broker
        self._kill_switch = kill_switch
        self._equity = equity
        self._peak_equity = equity
        self._state = OmsState()

    @property
    def equity(self) -> float:
        return self._equity

    @equity.setter
    def equity(self, value: float) -> None:
        self._equity = value
        self._peak_equity = max(self._peak_equity, value)

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if self._state.current_date != today:
            self._state = OmsState(current_date=today)

    def compute_plan(
        self,
        symbol: str,
        direction: int,
        magnitude: float,
        position_size_pct: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        strategy: str = "ensemble_v1",
    ) -> OrderPlan | None:
        """Compute an order plan from a prediction's parameters.

        Returns None if the signal produces no actionable order
        (e.g., zero quantity, zero position size).
        """
        if entry_price <= 0 or position_size_pct <= 0:
            return None

        capped_pct = min(position_size_pct, _MAX_POSITION_PCT)
        position_value = self._equity * (capped_pct / 100.0)
        quantity = max(1, int(position_value / entry_price))

        side = OrderSide.BUY if direction >= 0 else OrderSide.SELL

        target = take_profit_price if take_profit_price > 0 else entry_price * (1 + abs(magnitude))
        stop = stop_loss_price if stop_loss_price > 0 else entry_price * (1 - abs(magnitude) * 0.75)

        if side == OrderSide.SELL:
            target, stop = stop, target

        return OrderPlan(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=round(entry_price, 2),
            target_price=round(target, 2),
            stop_price=round(stop, 2),
            position_value=round(position_value, 2),
            position_pct=round(capped_pct, 2),
            strategy=strategy,
            prediction_date=date.today(),
        )

    async def execute_plan(self, plan: OrderPlan) -> OmsResult:
        """Execute an order plan through the broker, checking kill switch first."""
        self._reset_daily_if_needed()

        positions = await self._broker.get_positions()
        open_count = len(positions)

        daily_exposure = sum(p.position_pct for p in self._state.daily_orders)
        daily_pnl_pct = (
            (self._equity - self._peak_equity) / self._peak_equity * 100.0
            if self._peak_equity > 0
            else 0.0
        )

        ks_state = self._kill_switch.check(
            open_positions=open_count,
            daily_new_exposure_pct=daily_exposure + plan.position_pct,
            daily_pnl_pct=daily_pnl_pct,
            current_equity=self._equity,
            peak_equity=self._peak_equity,
        )

        if ks_state.halted:
            reasons = ", ".join(r.value for r in ks_state.halt_reasons)
            logger.warning(
                "oms_order_blocked",
                symbol=plan.symbol,
                reasons=reasons,
            )
            return OmsResult(
                planned=plan,
                blocked=True,
                block_reason=f"Kill switch: {reasons}",
                kill_switch_state=ks_state,
            )

        logger.info(
            "oms_placing_order",
            symbol=plan.symbol,
            side=plan.side.value,
            qty=plan.quantity,
            price=plan.entry_price,
            strategy=plan.strategy,
        )

        order = await self._broker.place_order(
            symbol=plan.symbol,
            side=plan.side,
            quantity=plan.quantity,
            order_type=OrderType.LIMIT,
            price=plan.entry_price,
            tag=plan.strategy,
        )

        self._state.daily_orders.append(plan)

        if plan.target_price > 0 and plan.stop_price > 0:
            await self._broker.create_gtt_oco(
                symbol=plan.symbol,
                quantity=plan.quantity,
                target_price=plan.target_price,
                stop_price=plan.stop_price,
            )

        return OmsResult(
            planned=plan,
            order=order,
            kill_switch_state=ks_state,
        )

    async def flatten_all(self) -> list[Order]:
        """Close all open positions — used when kill switch trips daily loss / drawdown."""
        positions = await self._broker.get_positions()
        orders: list[Order] = []
        for pos in positions:
            if pos.quantity > 0:
                order = await self._broker.place_order(
                    symbol=pos.symbol,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                    order_type=OrderType.MARKET,
                    tag="kill_switch_flatten",
                )
                orders.append(order)
                logger.warning(
                    "oms_flatten",
                    symbol=pos.symbol,
                    qty=pos.quantity,
                )
        return orders
