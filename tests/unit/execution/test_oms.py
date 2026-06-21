"""Tests for the Order Management System."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from alphavedha.execution.broker import (
    OrderSide,
    OrderStatus,
    PaperBroker,
)
from alphavedha.execution.kill_switch import (
    HaltReason,
    KillSwitch,
    KillSwitchConfig,
)
from alphavedha.execution.oms import (
    OrderManager,
    OrderPlan,
)


@pytest.fixture
def broker() -> PaperBroker:
    return PaperBroker(initial_capital=1_000_000.0)


@pytest.fixture
def kill_switch() -> KillSwitch:
    return KillSwitch(KillSwitchConfig())


@pytest.fixture
def oms(broker: PaperBroker, kill_switch: KillSwitch) -> OrderManager:
    return OrderManager(broker=broker, kill_switch=kill_switch, equity=1_000_000.0)


class TestComputePlan:
    def test_buy_plan(self, oms: OrderManager) -> None:
        plan = oms.compute_plan(
            symbol="TCS.NS",
            direction=1,
            magnitude=0.03,
            position_size_pct=5.0,
            entry_price=3500.0,
            stop_loss_price=3350.0,
            take_profit_price=3700.0,
            strategy="ensemble_v1",
        )
        assert plan is not None
        assert plan.side == OrderSide.BUY
        assert plan.symbol == "TCS.NS"
        assert plan.entry_price == 3500.0
        assert plan.target_price == 3700.0
        assert plan.stop_price == 3350.0
        assert plan.quantity > 0
        assert plan.position_pct == 5.0

    def test_sell_plan(self, oms: OrderManager) -> None:
        plan = oms.compute_plan(
            symbol="TCS.NS",
            direction=-1,
            magnitude=0.03,
            position_size_pct=3.0,
            entry_price=3500.0,
            stop_loss_price=3650.0,
            take_profit_price=3350.0,
            strategy="blowup_short_v1",
        )
        assert plan is not None
        assert plan.side == OrderSide.SELL

    def test_caps_at_5_pct(self, oms: OrderManager) -> None:
        plan = oms.compute_plan(
            symbol="TCS.NS",
            direction=1,
            magnitude=0.03,
            position_size_pct=10.0,
            entry_price=3500.0,
            stop_loss_price=3350.0,
            take_profit_price=3700.0,
        )
        assert plan is not None
        assert plan.position_pct == 5.0

    def test_returns_none_for_zero_price(self, oms: OrderManager) -> None:
        plan = oms.compute_plan(
            symbol="TCS.NS",
            direction=1,
            magnitude=0.03,
            position_size_pct=5.0,
            entry_price=0.0,
            stop_loss_price=0.0,
            take_profit_price=0.0,
        )
        assert plan is None

    def test_returns_none_for_zero_size(self, oms: OrderManager) -> None:
        plan = oms.compute_plan(
            symbol="TCS.NS",
            direction=1,
            magnitude=0.03,
            position_size_pct=0.0,
            entry_price=3500.0,
            stop_loss_price=3350.0,
            take_profit_price=3700.0,
        )
        assert plan is None

    def test_quantity_calculation(self, oms: OrderManager) -> None:
        plan = oms.compute_plan(
            symbol="TCS.NS",
            direction=1,
            magnitude=0.03,
            position_size_pct=5.0,
            entry_price=3500.0,
            stop_loss_price=3350.0,
            take_profit_price=3700.0,
        )
        assert plan is not None
        expected_value = 1_000_000 * 0.05
        expected_qty = int(expected_value / 3500.0)
        assert plan.quantity == expected_qty

    def test_fallback_prices_when_zero(self, oms: OrderManager) -> None:
        plan = oms.compute_plan(
            symbol="TCS.NS",
            direction=1,
            magnitude=0.03,
            position_size_pct=5.0,
            entry_price=3500.0,
            stop_loss_price=0.0,
            take_profit_price=0.0,
        )
        assert plan is not None
        assert plan.target_price > plan.entry_price
        assert plan.stop_price < plan.entry_price


class TestExecutePlan:
    @pytest.mark.asyncio
    async def test_blocked_when_disabled(self, oms: OrderManager) -> None:
        plan = OrderPlan(
            symbol="TCS.NS",
            side=OrderSide.BUY,
            quantity=14,
            entry_price=3500.0,
            target_price=3700.0,
            stop_price=3350.0,
            position_value=50000.0,
            position_pct=5.0,
            strategy="ensemble_v1",
            prediction_date=__import__("datetime").date.today(),
        )
        result = await oms.execute_plan(plan)
        assert result.blocked is True
        assert "MASTER_DISABLED" in result.block_reason
        assert result.order is None

    @pytest.mark.asyncio
    async def test_places_order_when_enabled(self, oms: OrderManager) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            plan = OrderPlan(
                symbol="TCS.NS",
                side=OrderSide.BUY,
                quantity=14,
                entry_price=3500.0,
                target_price=3700.0,
                stop_price=3350.0,
                position_value=50000.0,
                position_pct=5.0,
                strategy="ensemble_v1",
                prediction_date=__import__("datetime").date.today(),
            )
            result = await oms.execute_plan(plan)
            assert result.blocked is False
            assert result.order is not None
            assert result.order.status == OrderStatus.PENDING
            assert result.order.symbol == "TCS.NS"

    @pytest.mark.asyncio
    async def test_creates_gtt_oco(self, broker: PaperBroker, kill_switch: KillSwitch) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            oms = OrderManager(broker=broker, kill_switch=kill_switch)
            plan = OrderPlan(
                symbol="TCS.NS",
                side=OrderSide.BUY,
                quantity=14,
                entry_price=3500.0,
                target_price=3700.0,
                stop_price=3350.0,
                position_value=50000.0,
                position_pct=5.0,
                strategy="ensemble_v1",
                prediction_date=__import__("datetime").date.today(),
            )
            await oms.execute_plan(plan)
            assert len(broker._gtt_triggers) == 1

    @pytest.mark.asyncio
    async def test_blocked_at_max_positions(
        self, broker: PaperBroker, kill_switch: KillSwitch
    ) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(max_positions=2))
            oms = OrderManager(broker=broker, kill_switch=ks)
            for i, sym in enumerate(["TCS.NS", "INFY.NS"]):
                order = await broker.place_order(sym, OrderSide.BUY, 10)
                await broker.simulate_fill(order.order_id, 3500.0 + i * 100)

            plan = OrderPlan(
                symbol="HDFCBANK.NS",
                side=OrderSide.BUY,
                quantity=10,
                entry_price=1600.0,
                target_price=1700.0,
                stop_price=1500.0,
                position_value=50000.0,
                position_pct=5.0,
                strategy="ensemble_v1",
                prediction_date=__import__("datetime").date.today(),
            )
            result = await oms.execute_plan(plan)
            assert result.blocked is True
            assert HaltReason.MAX_POSITIONS.value in result.block_reason


class TestFlattenAll:
    @pytest.mark.asyncio
    async def test_flatten_closes_positions(
        self, broker: PaperBroker, kill_switch: KillSwitch
    ) -> None:
        oms = OrderManager(broker=broker, kill_switch=kill_switch)
        o1 = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(o1.order_id, 3500.0)
        o2 = await broker.place_order("INFY.NS", OrderSide.BUY, 5)
        await broker.simulate_fill(o2.order_id, 1500.0)

        orders = await oms.flatten_all()
        assert len(orders) == 2
        for order in orders:
            assert order.side == OrderSide.SELL
            assert order.tag == "kill_switch_flatten"

    @pytest.mark.asyncio
    async def test_flatten_empty_portfolio(
        self, broker: PaperBroker, kill_switch: KillSwitch
    ) -> None:
        oms = OrderManager(broker=broker, kill_switch=kill_switch)
        orders = await oms.flatten_all()
        assert len(orders) == 0


class TestEquityTracking:
    def test_equity_setter_updates_peak(self, oms: OrderManager) -> None:
        oms.equity = 1_100_000
        assert oms._peak_equity == 1_100_000
        oms.equity = 1_050_000
        assert oms._peak_equity == 1_100_000

    def test_initial_equity(self, oms: OrderManager) -> None:
        assert oms.equity == 1_000_000.0
