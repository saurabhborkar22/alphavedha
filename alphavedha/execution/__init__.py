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
from alphavedha.execution.shadow import (
    ShadowResult,
    ShadowRunner,
    ShadowSignal,
    shadow_fills_to_rows,
)
from alphavedha.execution.telegram import (
    TelegramBot,
    TelegramConfig,
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
    "ShadowResult",
    "ShadowRunner",
    "ShadowSignal",
    "TelegramBot",
    "TelegramConfig",
    "shadow_fills_to_rows",
]

# KiteAdapter and KiteConfig are NOT re-exported here because importing
# them would fail if the kiteconnect SDK is not installed. Import
# directly: from alphavedha.execution.kite_adapter import KiteAdapter
