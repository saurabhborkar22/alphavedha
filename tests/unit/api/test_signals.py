"""Tests for /signals/* endpoints."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


def _make_app() -> object:
    from alphavedha.api.app import create_app

    return create_app(demo=True)


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = _make_app()
    with TestClient(app) as c:
        yield c


class TestMarketTiming:
    def test_timing_endpoint_returns_expected_keys(self, client: TestClient) -> None:
        resp = client.get("/signals/timing")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_good_to_trade" in data
        assert "is_expiry_day" in data
        assert "optimal_windows" in data
        assert isinstance(data["optimal_windows"], list)
        assert "generated_at" in data


class TestExecutionPlan:
    def test_execution_returns_plan(self, client: TestClient) -> None:
        resp = client.get(
            "/signals/execution/TCS?cap_tier=large&avg_daily_volume=1000000&order_size_shares=200"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "TCS"
        assert data["order_type"] in ("market", "limit", "vwap")
        assert isinstance(data["n_tranches"], int)
        assert data["n_tranches"] >= 1
        assert "estimated_slippage_pct" in data
        assert "recommended_windows" in data


class TestBuySellSignal:
    def test_buy_sell_returns_full_signal(self, client: TestClient) -> None:
        resp = client.get("/signals/buy-sell/TCS?cap_tier=large&order_size_shares=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "TCS"
        assert data["signal"] in ("BUY", "SELL", "HOLD")
        assert "is_tradeable" in data
        assert "execute_now" in data
        assert "execution_plan" in data
        assert "price_targets" in data
        ep = data["execution_plan"]
        assert "order_type" in ep
        assert "n_tranches" in ep
        assert "estimated_slippage_pct" in ep
