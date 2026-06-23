"""Kite Connect broker adapter — real broker behind BrokerAdapter protocol.

Built but NOT armed: `EXECUTION_ENABLED` stays 0 until all Phase 5 gates
pass. This adapter wraps the Kite Connect SDK (kiteconnect) and maps its
responses to the shared BrokerAdapter types.

Requires:
  - pip install kiteconnect
  - KITE_API_KEY and KITE_API_SECRET env vars
  - Daily login flow to get request_token → access_token

The kiteconnect SDK is imported lazily so the module doesn't crash when
the package isn't installed (shadow mode uses PaperBroker, not this).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from alphavedha.execution.broker import (
    MarginInfo,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

_ORDER_STATUS_MAP: dict[str, OrderStatus] = {
    "OPEN": OrderStatus.OPEN,
    "COMPLETE": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "TRIGGER PENDING": OrderStatus.PENDING,
}

_EXCHANGE = "NSE"
_PRODUCT = "CNC"


@dataclass(frozen=True)
class KiteConfig:
    api_key: str
    api_secret: str
    access_token: str = ""

    @classmethod
    def from_env(cls) -> KiteConfig | None:
        api_key = os.environ.get("KITE_API_KEY", "")
        api_secret = os.environ.get("KITE_API_SECRET", "")
        access_token = os.environ.get("KITE_ACCESS_TOKEN", "")
        if not api_key or not api_secret:
            return None
        return cls(api_key=api_key, api_secret=api_secret, access_token=access_token)


@dataclass
class KiteSession:
    access_token: str = ""
    public_token: str = ""
    user_id: str = ""
    authenticated: bool = False


class KiteAdapter:
    """Real broker adapter for Zerodha Kite Connect.

    Wraps the kiteconnect SDK and implements BrokerAdapter protocol.
    The SDK is imported lazily — this module stays importable even
    without kiteconnect installed.
    """

    def __init__(self, config: KiteConfig) -> None:
        self._config = config
        self._session = KiteSession()
        self._kite: Any = None
        self._orders: dict[str, Order] = {}

    def _ensure_sdk(self) -> Any:
        if self._kite is not None:
            return self._kite
        try:
            from kiteconnect import KiteConnect  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("kiteconnect SDK not installed. Run: pip install kiteconnect") from e
        self._kite = KiteConnect(api_key=self._config.api_key)
        if self._config.access_token:
            self._kite.set_access_token(self._config.access_token)
            self._session.access_token = self._config.access_token
            self._session.authenticated = True
        return self._kite

    async def authenticate(self) -> bool:
        kite = self._ensure_sdk()
        if self._session.authenticated:
            return True
        if not self._config.access_token:
            logger.error(
                "kite_auth_failed",
                reason="No access_token. Complete login flow first.",
            )
            return False
        try:
            kite.set_access_token(self._config.access_token)
            profile = kite.profile()
            self._session = KiteSession(
                access_token=self._config.access_token,
                user_id=profile.get("user_id", ""),
                authenticated=True,
            )
            logger.info("kite_authenticated", user_id=self._session.user_id)
            return True
        except Exception as e:
            logger.error("kite_auth_failed", error=str(e))
            return False

    def generate_login_url(self) -> str:
        kite = self._ensure_sdk()
        return str(kite.login_url())

    async def set_access_token_from_request_token(self, request_token: str) -> bool:
        kite = self._ensure_sdk()
        try:
            data = kite.generate_session(request_token, api_secret=self._config.api_secret)
            token: str = data["access_token"]
            kite.set_access_token(token)
            self._session = KiteSession(
                access_token=token,
                user_id=data.get("user_id", ""),
                authenticated=True,
            )
            logger.info("kite_session_created", user_id=self._session.user_id)
            return True
        except Exception as e:
            logger.error("kite_session_failed", error=str(e))
            return False

    def _map_order_type(self, order_type: OrderType) -> str:
        return "MARKET" if order_type == OrderType.MARKET else "LIMIT"

    def _map_transaction(self, side: OrderSide) -> str:
        return "BUY" if side == OrderSide.BUY else "SELL"

    def _map_status(self, kite_status: str) -> OrderStatus:
        return _ORDER_STATUS_MAP.get(kite_status, OrderStatus.PENDING)

    def _to_trading_symbol(self, symbol: str) -> str:
        return symbol.replace(".NS", "")

    def _from_trading_symbol(self, trading_symbol: str) -> str:
        return f"{trading_symbol}.NS"

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        tag: str = "",
    ) -> Order:
        kite = self._ensure_sdk()
        trading_symbol = self._to_trading_symbol(symbol)

        params: dict[str, Any] = {
            "variety": "regular",
            "exchange": _EXCHANGE,
            "tradingsymbol": trading_symbol,
            "transaction_type": self._map_transaction(side),
            "quantity": quantity,
            "product": _PRODUCT,
            "order_type": self._map_order_type(order_type),
        }
        if price is not None and order_type == OrderType.LIMIT:
            params["price"] = price
        if tag:
            params["tag"] = tag[:20]

        try:
            order_id = str(kite.place_order(**params))
            logger.info(
                "kite_order_placed",
                order_id=order_id,
                symbol=symbol,
                side=side.value,
                qty=quantity,
            )
        except Exception as e:
            logger.error("kite_order_failed", symbol=symbol, error=str(e))
            raise

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
        return order

    async def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
    ) -> Order:
        kite = self._ensure_sdk()
        params: dict[str, Any] = {"variety": "regular", "order_id": order_id}
        if quantity is not None:
            params["quantity"] = quantity
        if price is not None:
            params["price"] = price

        kite.modify_order(**params)
        return await self.get_order(order_id)

    async def cancel_order(self, order_id: str) -> Order:
        kite = self._ensure_sdk()
        kite.cancel_order(variety="regular", order_id=order_id)
        return await self.get_order(order_id)

    async def create_gtt_oco(
        self,
        symbol: str,
        quantity: int,
        target_price: float,
        stop_price: float,
    ) -> str:
        kite = self._ensure_sdk()
        trading_symbol = self._to_trading_symbol(symbol)

        trigger_values = [stop_price, target_price]
        orders = [
            {
                "exchange": _EXCHANGE,
                "tradingsymbol": trading_symbol,
                "transaction_type": "SELL",
                "quantity": quantity,
                "order_type": "LIMIT",
                "product": _PRODUCT,
                "price": stop_price,
            },
            {
                "exchange": _EXCHANGE,
                "tradingsymbol": trading_symbol,
                "transaction_type": "SELL",
                "quantity": quantity,
                "order_type": "LIMIT",
                "product": _PRODUCT,
                "price": target_price,
            },
        ]

        try:
            gtt_id = str(
                kite.place_gtt(
                    trigger_type=kite.GTT_TYPE_OCO,
                    tradingsymbol=trading_symbol,
                    exchange=_EXCHANGE,
                    trigger_values=trigger_values,
                    last_price=stop_price,
                    orders=orders,
                )
            )
            logger.info(
                "kite_gtt_created",
                gtt_id=gtt_id,
                symbol=symbol,
                target=target_price,
                stop=stop_price,
            )
            return gtt_id
        except Exception as e:
            logger.error("kite_gtt_failed", symbol=symbol, error=str(e))
            raise

    async def cancel_gtt(self, gtt_id: str) -> bool:
        kite = self._ensure_sdk()
        try:
            kite.delete_gtt(int(gtt_id))
            return True
        except Exception as e:
            logger.error("kite_gtt_cancel_failed", gtt_id=gtt_id, error=str(e))
            return False

    async def get_order(self, order_id: str) -> Order:
        kite = self._ensure_sdk()
        history = kite.order_history(order_id)
        latest = history[-1]

        order = self._orders.get(order_id)
        if order is None:
            order = Order(
                order_id=order_id,
                symbol=self._from_trading_symbol(latest.get("tradingsymbol", "")),
                side=OrderSide.BUY if latest.get("transaction_type") == "BUY" else OrderSide.SELL,
                order_type=OrderType.MARKET
                if latest.get("order_type") == "MARKET"
                else OrderType.LIMIT,
                quantity=latest.get("quantity", 0),
                price=latest.get("price"),
            )
            self._orders[order_id] = order

        order.status = self._map_status(latest.get("status", ""))
        order.updated_at = datetime.now(IST)
        return order

    async def get_positions(self) -> list[Position]:
        kite = self._ensure_sdk()
        data = kite.positions()
        net = data.get("net", [])
        positions: list[Position] = []
        for pos in net:
            qty = pos.get("quantity", 0)
            if qty == 0:
                continue
            avg = pos.get("average_price", 0.0)
            ltp = pos.get("last_price", avg)
            pnl = pos.get("pnl", 0.0)
            pnl_pct = (pnl / (avg * abs(qty)) * 100.0) if avg > 0 and qty != 0 else 0.0
            positions.append(
                Position(
                    symbol=self._from_trading_symbol(pos.get("tradingsymbol", "")),
                    quantity=qty,
                    avg_price=avg,
                    current_price=ltp,
                    pnl=pnl,
                    pnl_pct=round(pnl_pct, 2),
                )
            )
        return positions

    async def get_margins(self) -> MarginInfo:
        kite = self._ensure_sdk()
        margins = kite.margins(segment="equity")
        available = margins.get("available", {})
        utilised = margins.get("utilised", {})
        cash: float = available.get("cash", 0.0)
        used: float = utilised.get("debits", 0.0)
        return MarginInfo(
            available_cash=cash,
            used_margin=used,
            total_equity=cash + used,
        )

    async def get_holdings(self) -> list[Position]:
        kite = self._ensure_sdk()
        holdings = kite.holdings()
        result: list[Position] = []
        for h in holdings:
            qty = h.get("quantity", 0)
            if qty == 0:
                continue
            avg = h.get("average_price", 0.0)
            ltp = h.get("last_price", avg)
            pnl = h.get("pnl", 0.0)
            pnl_pct = (pnl / (avg * qty) * 100.0) if avg > 0 and qty > 0 else 0.0
            result.append(
                Position(
                    symbol=self._from_trading_symbol(h.get("tradingsymbol", "")),
                    quantity=qty,
                    avg_price=avg,
                    current_price=ltp,
                    pnl=pnl,
                    pnl_pct=round(pnl_pct, 2),
                )
            )
        return result
