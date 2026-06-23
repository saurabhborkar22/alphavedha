"""Tests for KiteAdapter — real broker adapter behind BrokerAdapter protocol.

All tests mock the kiteconnect SDK since it's not installed in the test env.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alphavedha.execution.broker import (
    OrderSide,
    OrderStatus,
    OrderType,
)


def _make_mock_kiteconnect() -> ModuleType:
    """Create a mock kiteconnect module with KiteConnect class."""
    mock_kc_module = ModuleType("kiteconnect")
    mock_kc_class = MagicMock()
    mock_kc_instance = MagicMock()
    mock_kc_instance.GTT_TYPE_OCO = "two-leg"
    mock_kc_instance.login_url.return_value = "https://kite.zerodha.com/connect/login?api_key=test"
    mock_kc_class.return_value = mock_kc_instance
    mock_kc_module.KiteConnect = mock_kc_class  # type: ignore[attr-defined]
    return mock_kc_module


@pytest.fixture(autouse=True)
def _mock_kiteconnect() -> Any:
    """Inject mock kiteconnect module for all tests."""
    mock_module = _make_mock_kiteconnect()
    with patch.dict(sys.modules, {"kiteconnect": mock_module}):
        yield mock_module


@pytest.fixture
def config() -> Any:
    from alphavedha.execution.kite_adapter import KiteConfig

    return KiteConfig(api_key="test_key", api_secret="test_secret", access_token="test_token")


@pytest.fixture
def adapter(config: Any) -> Any:
    from alphavedha.execution.kite_adapter import KiteAdapter

    return KiteAdapter(config)


class TestKiteConfig:
    def test_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "KITE_API_KEY": "key123",
                "KITE_API_SECRET": "secret456",
                "KITE_ACCESS_TOKEN": "tok789",
            },
        ):
            from alphavedha.execution.kite_adapter import KiteConfig

            cfg = KiteConfig.from_env()
            assert cfg is not None
            assert cfg.api_key == "key123"
            assert cfg.api_secret == "secret456"
            assert cfg.access_token == "tok789"

    def test_from_env_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            from alphavedha.execution.kite_adapter import KiteConfig

            cfg = KiteConfig.from_env()
            assert cfg is None


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_authenticate_with_token(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.profile.return_value = {"user_id": "AB1234"}
        adapter._session.authenticated = False

        result = await adapter.authenticate()
        assert result is True
        assert adapter._session.user_id == "AB1234"

    @pytest.mark.asyncio
    async def test_authenticate_no_token(self) -> None:
        from alphavedha.execution.kite_adapter import KiteAdapter, KiteConfig

        cfg = KiteConfig(api_key="key", api_secret="secret", access_token="")
        a = KiteAdapter(cfg)

        result = await a.authenticate()
        assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_sdk_error(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.profile.side_effect = Exception("Invalid token")

        adapter._session.authenticated = False
        result = await adapter.authenticate()
        assert result is False

    def test_login_url(self, adapter: Any) -> None:
        url = adapter.generate_login_url()
        assert "kite.zerodha.com" in url

    @pytest.mark.asyncio
    async def test_set_access_token_from_request(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.generate_session.return_value = {
            "access_token": "new_tok",
            "user_id": "AB1234",
        }

        result = await adapter.set_access_token_from_request_token("req_tok")
        assert result is True
        assert adapter._session.access_token == "new_tok"


class TestOrders:
    @pytest.mark.asyncio
    async def test_place_market_order(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.place_order.return_value = "ORD001"

        order = await adapter.place_order("TCS.NS", OrderSide.BUY, 10)
        assert order.order_id == "ORD001"
        assert order.symbol == "TCS.NS"
        assert order.side == OrderSide.BUY
        assert order.quantity == 10

        call_kwargs = kite.place_order.call_args[1]
        assert call_kwargs["tradingsymbol"] == "TCS"
        assert call_kwargs["transaction_type"] == "BUY"

    @pytest.mark.asyncio
    async def test_place_limit_order(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.place_order.return_value = "ORD002"

        order = await adapter.place_order(
            "INFY.NS", OrderSide.SELL, 5, OrderType.LIMIT, price=1500.0
        )
        assert order.order_id == "ORD002"
        assert order.price == 1500.0

        call_kwargs = kite.place_order.call_args[1]
        assert call_kwargs["price"] == 1500.0
        assert call_kwargs["order_type"] == "LIMIT"

    @pytest.mark.asyncio
    async def test_place_order_with_tag(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.place_order.return_value = "ORD003"

        await adapter.place_order("TCS.NS", OrderSide.BUY, 10, tag="ensemble_v1")
        call_kwargs = kite.place_order.call_args[1]
        assert call_kwargs["tag"] == "ensemble_v1"

    @pytest.mark.asyncio
    async def test_modify_order(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.order_history.return_value = [
            {
                "tradingsymbol": "TCS",
                "transaction_type": "BUY",
                "order_type": "LIMIT",
                "quantity": 15,
                "price": 3600,
                "status": "OPEN",
            }
        ]

        order = await adapter.modify_order("ORD001", quantity=15, price=3600)
        assert order.quantity == 15

    @pytest.mark.asyncio
    async def test_cancel_order(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.order_history.return_value = [
            {
                "tradingsymbol": "TCS",
                "transaction_type": "BUY",
                "order_type": "LIMIT",
                "quantity": 10,
                "price": 3500,
                "status": "CANCELLED",
            }
        ]

        order = await adapter.cancel_order("ORD001")
        assert order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_get_order_status(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.order_history.return_value = [
            {
                "tradingsymbol": "TCS",
                "transaction_type": "BUY",
                "order_type": "MARKET",
                "quantity": 10,
                "price": None,
                "status": "COMPLETE",
            }
        ]

        order = await adapter.get_order("ORD001")
        assert order.status == OrderStatus.FILLED


class TestGtt:
    @pytest.mark.asyncio
    async def test_create_gtt_oco(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.place_gtt.return_value = 12345

        gtt_id = await adapter.create_gtt_oco("TCS.NS", 10, 3800.0, 3200.0)
        assert gtt_id == "12345"
        kite.place_gtt.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_gtt(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.delete_gtt.return_value = None

        result = await adapter.cancel_gtt("12345")
        assert result is True
        kite.delete_gtt.assert_called_once_with(12345)

    @pytest.mark.asyncio
    async def test_cancel_gtt_failure(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.delete_gtt.side_effect = Exception("Not found")

        result = await adapter.cancel_gtt("99999")
        assert result is False


class TestPositionsAndMargins:
    @pytest.mark.asyncio
    async def test_get_positions(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.positions.return_value = {
            "net": [
                {
                    "tradingsymbol": "TCS",
                    "quantity": 10,
                    "average_price": 3500.0,
                    "last_price": 3600.0,
                    "pnl": 1000.0,
                },
                {
                    "tradingsymbol": "INFY",
                    "quantity": 0,
                    "average_price": 1500.0,
                    "last_price": 1500.0,
                    "pnl": 0.0,
                },
            ]
        }

        positions = await adapter.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "TCS.NS"
        assert positions[0].quantity == 10
        assert positions[0].pnl == 1000.0

    @pytest.mark.asyncio
    async def test_get_margins(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.margins.return_value = {
            "available": {"cash": 500_000.0},
            "utilised": {"debits": 200_000.0},
        }

        margins = await adapter.get_margins()
        assert margins.available_cash == 500_000.0
        assert margins.used_margin == 200_000.0
        assert margins.total_equity == 700_000.0

    @pytest.mark.asyncio
    async def test_get_holdings(self, adapter: Any) -> None:
        kite = adapter._ensure_sdk()
        kite.holdings.return_value = [
            {
                "tradingsymbol": "RELIANCE",
                "quantity": 5,
                "average_price": 2500.0,
                "last_price": 2600.0,
                "pnl": 500.0,
            }
        ]

        holdings = await adapter.get_holdings()
        assert len(holdings) == 1
        assert holdings[0].symbol == "RELIANCE.NS"
        assert holdings[0].pnl_pct == pytest.approx(4.0, abs=0.01)


class TestSymbolMapping:
    def test_to_trading_symbol(self, adapter: Any) -> None:
        assert adapter._to_trading_symbol("TCS.NS") == "TCS"
        assert adapter._to_trading_symbol("RELIANCE.NS") == "RELIANCE"

    def test_from_trading_symbol(self, adapter: Any) -> None:
        assert adapter._from_trading_symbol("TCS") == "TCS.NS"

    def test_status_mapping(self, adapter: Any) -> None:
        assert adapter._map_status("OPEN") == OrderStatus.OPEN
        assert adapter._map_status("COMPLETE") == OrderStatus.FILLED
        assert adapter._map_status("CANCELLED") == OrderStatus.CANCELLED
        assert adapter._map_status("REJECTED") == OrderStatus.REJECTED
        assert adapter._map_status("UNKNOWN") == OrderStatus.PENDING


class TestSdkImport:
    def test_missing_sdk_raises(self) -> None:
        with patch.dict(sys.modules, {"kiteconnect": None}):
            from alphavedha.execution.kite_adapter import KiteAdapter, KiteConfig

            cfg = KiteConfig(api_key="k", api_secret="s")
            a = KiteAdapter(cfg)
            a._kite = None
            with pytest.raises(RuntimeError, match="kiteconnect SDK not installed"):
                a._ensure_sdk()
