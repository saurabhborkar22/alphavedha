"""Broker adapter protocol and PaperBroker implementation.

BrokerAdapter is the protocol every broker backend must implement.
PaperBroker simulates fills at next open ± slippage — used by shadow mode
and by tests. Real broker adapters (Kite, Dhan) will implement the same
protocol when P4-D5 lands.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

import structlog

from alphavedha.config import SlippageConfig

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class GttOcoParams:
    """Good Till Triggered — One Cancels Other exit bracket."""

    target_price: float
    stop_price: float
    quantity: int


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float | None
    status: OrderStatus = OrderStatus.PENDING
    gtt_oco: GttOcoParams | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(IST))
    updated_at: datetime = field(default_factory=lambda: datetime.now(IST))
    tag: str = ""


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    fill_price: float
    decision_price: float
    slippage_bps: float
    filled_at: datetime
    tag: str = ""


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    pnl: float
    pnl_pct: float


@dataclass(frozen=True)
class MarginInfo:
    available_cash: float
    used_margin: float
    total_equity: float


@runtime_checkable
class BrokerAdapter(Protocol):
    """Protocol that every broker backend must implement."""

    async def authenticate(self) -> bool:
        """Authenticate with the broker. Returns True on success."""
        ...

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        tag: str = "",
    ) -> Order:
        """Place an order. Returns the created Order with an ID."""
        ...

    async def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
    ) -> Order:
        """Modify a pending/open order."""
        ...

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending/open order."""
        ...

    async def create_gtt_oco(
        self,
        symbol: str,
        quantity: int,
        target_price: float,
        stop_price: float,
    ) -> str:
        """Create a GTT OCO exit bracket. Returns the GTT trigger ID."""
        ...

    async def cancel_gtt(self, gtt_id: str) -> bool:
        """Cancel a GTT trigger."""
        ...

    async def get_order(self, order_id: str) -> Order:
        """Get current state of an order."""
        ...

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    async def get_margins(self) -> MarginInfo:
        """Get available margin / cash info."""
        ...

    async def get_holdings(self) -> list[Position]:
        """Get delivery holdings (T+1 settled)."""
        ...


def _slippage_rate(market_cap_tier: str, config: SlippageConfig) -> float:
    rates = {
        "large": config.large_cap,
        "mid": config.mid_cap,
        "small": config.small_cap,
    }
    return rates.get(market_cap_tier, config.mid_cap)


class PaperBroker:
    """Simulated broker that fills at next open ± slippage.

    Used by shadow mode (P4-D3) to generate realistic fills without
    touching any real broker. Slippage is drawn from the existing
    cost model (large: 0.1%, mid: 0.3%, small: 0.5%).
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        slippage_config: SlippageConfig | None = None,
        default_tier: str = "large",
    ) -> None:
        self._capital = initial_capital
        self._slippage_config = slippage_config or SlippageConfig()
        self._default_tier = default_tier
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._positions: dict[str, Position] = {}
        self._gtt_triggers: dict[str, GttOcoParams] = {}
        self._authenticated = False

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def capital(self) -> float:
        return self._capital

    async def authenticate(self) -> bool:
        self._authenticated = True
        logger.info("paper_broker_authenticated")
        return True

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        tag: str = "",
    ) -> Order:
        order_id = f"paper-{uuid.uuid4().hex[:12]}"
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            tag=tag,
        )
        self._orders[order_id] = order
        logger.info(
            "paper_order_placed",
            order_id=order_id,
            symbol=symbol,
            side=side.value,
            qty=quantity,
            price=price,
            tag=tag,
        )
        return order

    async def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
    ) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            msg = f"Order {order_id} not found"
            raise ValueError(msg)
        if order.status not in (OrderStatus.PENDING, OrderStatus.OPEN):
            msg = f"Cannot modify order in status {order.status.value}"
            raise ValueError(msg)
        if quantity is not None:
            order.quantity = quantity
        if price is not None:
            order.price = price
        order.updated_at = datetime.now(IST)
        return order

    async def cancel_order(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            msg = f"Order {order_id} not found"
            raise ValueError(msg)
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            msg = f"Cannot cancel order in status {order.status.value}"
            raise ValueError(msg)
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now(IST)
        logger.info("paper_order_cancelled", order_id=order_id)
        return order

    async def create_gtt_oco(
        self,
        symbol: str,
        quantity: int,
        target_price: float,
        stop_price: float,
    ) -> str:
        gtt_id = f"gtt-{uuid.uuid4().hex[:12]}"
        self._gtt_triggers[gtt_id] = GttOcoParams(
            target_price=target_price,
            stop_price=stop_price,
            quantity=quantity,
        )
        logger.info(
            "paper_gtt_created",
            gtt_id=gtt_id,
            symbol=symbol,
            target=target_price,
            stop=stop_price,
        )
        return gtt_id

    async def cancel_gtt(self, gtt_id: str) -> bool:
        if gtt_id not in self._gtt_triggers:
            return False
        del self._gtt_triggers[gtt_id]
        logger.info("paper_gtt_cancelled", gtt_id=gtt_id)
        return True

    async def get_order(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            msg = f"Order {order_id} not found"
            raise ValueError(msg)
        return order

    async def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def get_margins(self) -> MarginInfo:
        used = sum(p.avg_price * p.quantity for p in self._positions.values())
        return MarginInfo(
            available_cash=self._capital,
            used_margin=used,
            total_equity=self._capital + sum(p.pnl for p in self._positions.values()),
        )

    async def get_holdings(self) -> list[Position]:
        return list(self._positions.values())

    async def simulate_fill(
        self,
        order_id: str,
        market_open_price: float,
        market_cap_tier: str | None = None,
    ) -> Fill:
        """Simulate a fill at market open price ± slippage.

        Called by the shadow mode job after market opens to convert
        pending orders into fills with realistic slippage.
        """
        order = self._orders.get(order_id)
        if order is None:
            msg = f"Order {order_id} not found"
            raise ValueError(msg)
        if order.status != OrderStatus.PENDING:
            msg = f"Order {order_id} already in status {order.status.value}"
            raise ValueError(msg)

        tier = market_cap_tier or self._default_tier
        slip_rate = _slippage_rate(tier, self._slippage_config)

        if order.side == OrderSide.BUY:
            fill_price = market_open_price * (1.0 + slip_rate)
        else:
            fill_price = market_open_price * (1.0 - slip_rate)

        slippage_bps = abs(fill_price - market_open_price) / market_open_price * 10_000

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=round(fill_price, 2),
            decision_price=market_open_price,
            slippage_bps=round(slippage_bps, 2),
            filled_at=datetime.now(IST),
            tag=order.tag,
        )
        self._fills.append(fill)

        order.status = OrderStatus.FILLED
        order.updated_at = datetime.now(IST)

        self._update_position(fill)

        logger.info(
            "paper_fill",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            qty=order.quantity,
            decision_price=market_open_price,
            fill_price=fill.fill_price,
            slippage_bps=fill.slippage_bps,
        )
        return fill

    def _update_position(self, fill: Fill) -> None:
        """Update internal position state after a fill."""
        pos = self._positions.get(fill.symbol)
        trade_value = fill.fill_price * fill.quantity

        if fill.side == OrderSide.BUY:
            if pos is None:
                self._positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    quantity=fill.quantity,
                    avg_price=fill.fill_price,
                    current_price=fill.fill_price,
                    pnl=0.0,
                    pnl_pct=0.0,
                )
            else:
                total_qty = pos.quantity + fill.quantity
                pos.avg_price = (pos.avg_price * pos.quantity + trade_value) / total_qty
                pos.quantity = total_qty
            self._capital -= trade_value
        else:
            if pos is not None:
                close_qty = min(fill.quantity, pos.quantity)
                realized = (fill.fill_price - pos.avg_price) * close_qty
                pos.quantity -= close_qty
                pos.pnl += realized
                self._capital += trade_value
                if pos.quantity <= 0:
                    del self._positions[fill.symbol]
            else:
                self._capital += trade_value

    def check_gtt_triggers(self, symbol: str, high: float, low: float) -> list[str]:
        """Check if any GTT triggers fire for given price range.

        Returns list of triggered GTT IDs (target or stop hit).
        """
        triggered: list[str] = []
        for gtt_id, params in list(self._gtt_triggers.items()):
            if high >= params.target_price or low <= params.stop_price:
                triggered.append(gtt_id)
                del self._gtt_triggers[gtt_id]
                hit = "target" if high >= params.target_price else "stop"
                logger.info(
                    "paper_gtt_triggered",
                    gtt_id=gtt_id,
                    symbol=symbol,
                    hit=hit,
                    target=params.target_price,
                    stop=params.stop_price,
                )
        return triggered

    def reset(self) -> None:
        """Reset all state — for testing."""
        self._orders.clear()
        self._fills.clear()
        self._positions.clear()
        self._gtt_triggers.clear()
        self._capital = 1_000_000.0
        self._authenticated = False
