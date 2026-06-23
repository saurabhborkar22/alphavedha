"""Tests for broker adapter protocol and PaperBroker."""

from __future__ import annotations

import pytest

from alphavedha.config import SlippageConfig
from alphavedha.execution.broker import (
    BrokerAdapter,
    Fill,
    GttOcoParams,
    MarginInfo,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperBroker,
    Position,
)


@pytest.fixture
def broker() -> PaperBroker:
    return PaperBroker(initial_capital=1_000_000.0)


class TestBrokerAdapterProtocol:
    def test_paper_broker_is_adapter(self) -> None:
        assert isinstance(PaperBroker(), BrokerAdapter)

    def test_order_dataclass_fields(self) -> None:
        order = Order(
            order_id="test-1",
            symbol="TCS.NS",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            price=None,
        )
        assert order.order_id == "test-1"
        assert order.symbol == "TCS.NS"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING

    def test_fill_dataclass_frozen(self) -> None:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        fill = Fill(
            order_id="test-1",
            symbol="TCS.NS",
            side=OrderSide.BUY,
            quantity=10,
            fill_price=3500.0,
            decision_price=3497.0,
            slippage_bps=8.58,
            filled_at=datetime.now(ZoneInfo("Asia/Kolkata")),
        )
        assert fill.fill_price == 3500.0
        with pytest.raises(AttributeError):
            fill.fill_price = 3600.0  # type: ignore[misc]

    def test_position_dataclass(self) -> None:
        pos = Position(
            symbol="TCS.NS",
            quantity=10,
            avg_price=3500.0,
            current_price=3550.0,
            pnl=500.0,
            pnl_pct=1.43,
        )
        assert pos.pnl == 500.0

    def test_margin_info_frozen(self) -> None:
        info = MarginInfo(available_cash=900_000.0, used_margin=100_000.0, total_equity=1_000_000.0)
        assert info.total_equity == 1_000_000.0

    def test_gtt_oco_params(self) -> None:
        gtt = GttOcoParams(target_price=3700.0, stop_price=3300.0, quantity=10)
        assert gtt.target_price == 3700.0
        assert gtt.stop_price == 3300.0

    def test_order_side_enum_values(self) -> None:
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_order_type_enum_values(self) -> None:
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"

    def test_order_status_enum_values(self) -> None:
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"


class TestPaperBrokerAuth:
    @pytest.mark.asyncio
    async def test_authenticate(self, broker: PaperBroker) -> None:
        result = await broker.authenticate()
        assert result is True
        assert broker._authenticated is True


class TestPaperBrokerOrders:
    @pytest.mark.asyncio
    async def test_place_order_returns_pending(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        assert order.status == OrderStatus.PENDING
        assert order.symbol == "TCS.NS"
        assert order.side == OrderSide.BUY
        assert order.quantity == 10
        assert order.order_id.startswith("paper-")

    @pytest.mark.asyncio
    async def test_place_limit_order(self, broker: PaperBroker) -> None:
        order = await broker.place_order("INFY.NS", OrderSide.BUY, 5, OrderType.LIMIT, price=1500.0)
        assert order.order_type == OrderType.LIMIT
        assert order.price == 1500.0

    @pytest.mark.asyncio
    async def test_place_order_with_tag(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10, tag="event_drift_v1")
        assert order.tag == "event_drift_v1"

    @pytest.mark.asyncio
    async def test_modify_pending_order(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        modified = await broker.modify_order(order.order_id, quantity=20, price=3500.0)
        assert modified.quantity == 20
        assert modified.price == 3500.0

    @pytest.mark.asyncio
    async def test_modify_nonexistent_raises(self, broker: PaperBroker) -> None:
        with pytest.raises(ValueError, match="not found"):
            await broker.modify_order("bad-id")

    @pytest.mark.asyncio
    async def test_modify_filled_raises(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(order.order_id, 3500.0)
        with pytest.raises(ValueError, match="Cannot modify"):
            await broker.modify_order(order.order_id, quantity=20)

    @pytest.mark.asyncio
    async def test_cancel_order(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        cancelled = await broker.cancel_order(order.order_id)
        assert cancelled.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_raises(self, broker: PaperBroker) -> None:
        with pytest.raises(ValueError, match="not found"):
            await broker.cancel_order("bad-id")

    @pytest.mark.asyncio
    async def test_cancel_filled_raises(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(order.order_id, 3500.0)
        with pytest.raises(ValueError, match="Cannot cancel"):
            await broker.cancel_order(order.order_id)

    @pytest.mark.asyncio
    async def test_get_order(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        retrieved = await broker.get_order(order.order_id)
        assert retrieved.order_id == order.order_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_order_raises(self, broker: PaperBroker) -> None:
        with pytest.raises(ValueError, match="not found"):
            await broker.get_order("bad-id")


class TestPaperBrokerFills:
    @pytest.mark.asyncio
    async def test_fill_buy_with_slippage(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        fill = await broker.simulate_fill(order.order_id, 3500.0, "large")
        assert fill.fill_price > 3500.0
        assert fill.slippage_bps > 0
        assert fill.decision_price == 3500.0

    @pytest.mark.asyncio
    async def test_fill_sell_with_slippage(self, broker: PaperBroker) -> None:
        buy = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(buy.order_id, 3500.0)
        sell = await broker.place_order("TCS.NS", OrderSide.SELL, 10)
        fill = await broker.simulate_fill(sell.order_id, 3600.0, "large")
        assert fill.fill_price < 3600.0
        assert fill.slippage_bps > 0

    @pytest.mark.asyncio
    async def test_fill_large_cap_slippage_rate(self) -> None:
        broker = PaperBroker(slippage_config=SlippageConfig(large_cap=0.001))
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 100)
        fill = await broker.simulate_fill(order.order_id, 3500.0, "large")
        expected = 3500.0 * 1.001
        assert fill.fill_price == pytest.approx(expected, abs=0.01)

    @pytest.mark.asyncio
    async def test_fill_midcap_higher_slippage(self) -> None:
        config = SlippageConfig(large_cap=0.001, mid_cap=0.003)
        broker = PaperBroker(slippage_config=config)
        order = await broker.place_order("AUROPHARMA.NS", OrderSide.BUY, 100)
        fill = await broker.simulate_fill(order.order_id, 1200.0, "mid")
        expected = 1200.0 * 1.003
        assert fill.fill_price == pytest.approx(expected, abs=0.01)

    @pytest.mark.asyncio
    async def test_fill_already_filled_raises(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(order.order_id, 3500.0)
        with pytest.raises(ValueError, match="already in status"):
            await broker.simulate_fill(order.order_id, 3500.0)

    @pytest.mark.asyncio
    async def test_fill_nonexistent_raises(self, broker: PaperBroker) -> None:
        with pytest.raises(ValueError, match="not found"):
            await broker.simulate_fill("bad-id", 3500.0)

    @pytest.mark.asyncio
    async def test_fills_list(self, broker: PaperBroker) -> None:
        o1 = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        o2 = await broker.place_order("INFY.NS", OrderSide.BUY, 5)
        await broker.simulate_fill(o1.order_id, 3500.0)
        await broker.simulate_fill(o2.order_id, 1500.0)
        assert len(broker.fills) == 2

    @pytest.mark.asyncio
    async def test_fill_tag_propagated(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10, tag="shadow_v1")
        fill = await broker.simulate_fill(order.order_id, 3500.0)
        assert fill.tag == "shadow_v1"


class TestPaperBrokerPositions:
    @pytest.mark.asyncio
    async def test_buy_creates_position(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(order.order_id, 3500.0)
        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "TCS.NS"
        assert positions[0].quantity == 10

    @pytest.mark.asyncio
    async def test_sell_closes_position(self, broker: PaperBroker) -> None:
        buy = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(buy.order_id, 3500.0)
        sell = await broker.place_order("TCS.NS", OrderSide.SELL, 10)
        await broker.simulate_fill(sell.order_id, 3600.0)
        positions = await broker.get_positions()
        assert len(positions) == 0

    @pytest.mark.asyncio
    async def test_partial_sell(self, broker: PaperBroker) -> None:
        buy = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(buy.order_id, 3500.0)
        sell = await broker.place_order("TCS.NS", OrderSide.SELL, 5)
        await broker.simulate_fill(sell.order_id, 3600.0)
        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == 5

    @pytest.mark.asyncio
    async def test_capital_decreases_on_buy(self, broker: PaperBroker) -> None:
        initial = broker.capital
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        fill = await broker.simulate_fill(order.order_id, 3500.0)
        assert broker.capital < initial
        expected = initial - fill.fill_price * 10
        assert broker.capital == pytest.approx(expected, abs=0.01)

    @pytest.mark.asyncio
    async def test_capital_increases_on_sell(self, broker: PaperBroker) -> None:
        buy = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(buy.order_id, 3500.0)
        mid_capital = broker.capital
        sell = await broker.place_order("TCS.NS", OrderSide.SELL, 10)
        fill = await broker.simulate_fill(sell.order_id, 3600.0)
        assert broker.capital > mid_capital
        expected = mid_capital + fill.fill_price * 10
        assert broker.capital == pytest.approx(expected, abs=0.01)

    @pytest.mark.asyncio
    async def test_add_to_existing_position(self, broker: PaperBroker) -> None:
        o1 = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(o1.order_id, 3500.0)
        o2 = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(o2.order_id, 3600.0)
        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == 20


class TestPaperBrokerMargins:
    @pytest.mark.asyncio
    async def test_initial_margins(self, broker: PaperBroker) -> None:
        margins = await broker.get_margins()
        assert margins.available_cash == 1_000_000.0
        assert margins.used_margin == 0.0
        assert margins.total_equity == 1_000_000.0

    @pytest.mark.asyncio
    async def test_margins_after_buy(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(order.order_id, 3500.0)
        margins = await broker.get_margins()
        assert margins.available_cash < 1_000_000.0
        assert margins.used_margin > 0


class TestPaperBrokerGtt:
    @pytest.mark.asyncio
    async def test_create_gtt(self, broker: PaperBroker) -> None:
        gtt_id = await broker.create_gtt_oco("TCS.NS", 10, 3700.0, 3300.0)
        assert gtt_id.startswith("gtt-")

    @pytest.mark.asyncio
    async def test_cancel_gtt(self, broker: PaperBroker) -> None:
        gtt_id = await broker.create_gtt_oco("TCS.NS", 10, 3700.0, 3300.0)
        result = await broker.cancel_gtt(gtt_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_gtt(self, broker: PaperBroker) -> None:
        result = await broker.cancel_gtt("bad-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_gtt_target_triggers(self, broker: PaperBroker) -> None:
        await broker.create_gtt_oco("TCS.NS", 10, 3700.0, 3300.0)
        triggered = broker.check_gtt_triggers("TCS.NS", high=3750.0, low=3400.0)
        assert len(triggered) == 1

    @pytest.mark.asyncio
    async def test_gtt_stop_triggers(self, broker: PaperBroker) -> None:
        await broker.create_gtt_oco("TCS.NS", 10, 3700.0, 3300.0)
        triggered = broker.check_gtt_triggers("TCS.NS", high=3500.0, low=3250.0)
        assert len(triggered) == 1

    @pytest.mark.asyncio
    async def test_gtt_no_trigger(self, broker: PaperBroker) -> None:
        await broker.create_gtt_oco("TCS.NS", 10, 3700.0, 3300.0)
        triggered = broker.check_gtt_triggers("TCS.NS", high=3600.0, low=3400.0)
        assert len(triggered) == 0


class TestPaperBrokerReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(order.order_id, 3500.0)
        await broker.create_gtt_oco("TCS.NS", 10, 3700.0, 3300.0)
        broker.reset()
        assert len(broker.fills) == 0
        assert len(await broker.get_positions()) == 0
        assert broker.capital == 1_000_000.0


class TestPaperBrokerHoldings:
    @pytest.mark.asyncio
    async def test_holdings_same_as_positions(self, broker: PaperBroker) -> None:
        order = await broker.place_order("TCS.NS", OrderSide.BUY, 10)
        await broker.simulate_fill(order.order_id, 3500.0)
        holdings = await broker.get_holdings()
        positions = await broker.get_positions()
        assert len(holdings) == len(positions)
