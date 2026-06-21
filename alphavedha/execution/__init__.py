"""Execution engine — broker adapters, order management, and shadow mode."""

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
from alphavedha.execution.kill_switch import (
    HaltReason,
    KillSwitch,
    KillSwitchConfig,
    KillSwitchState,
)
from alphavedha.execution.oms import (
    OmsResult,
    OmsState,
    OrderManager,
    OrderPlan,
)

__all__ = [
    "BrokerAdapter",
    "Fill",
    "GttOcoParams",
    "HaltReason",
    "KillSwitch",
    "KillSwitchConfig",
    "KillSwitchState",
    "MarginInfo",
    "OmsResult",
    "OmsState",
    "Order",
    "OrderManager",
    "OrderPlan",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperBroker",
    "Position",
]
