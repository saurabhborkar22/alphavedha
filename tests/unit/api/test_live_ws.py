"""Tests for live WebSocket endpoints."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from alphavedha.api.routes.live import _fetch_intraday_snapshot, _fetch_tick, _is_market_open


def _make_app() -> object:
    from alphavedha.api.app import create_app

    return create_app(demo=True)


class TestMarketOpen:
    def test_returns_bool(self) -> None:
        result = _is_market_open()
        assert isinstance(result, bool)


class TestFetchHelpers:
    def test_fetch_snapshot_returns_list_on_error(self) -> None:
        with patch("yfinance.download", side_effect=Exception("network")):
            result = _fetch_intraday_snapshot("TCS")
        assert isinstance(result, list)
        assert result == []

    def test_fetch_tick_returns_dict_on_error(self) -> None:
        with patch("yfinance.Ticker", side_effect=Exception("network")):
            result = _fetch_tick("TCS")
        assert isinstance(result, dict)
        assert result == {}


class TestWebSocketLive:
    @pytest.fixture(autouse=True)
    def _client(self) -> None:
        app = _make_app()
        self.client = TestClient(app)

    def test_ws_live_connects_and_sends_snapshot(self) -> None:
        mock_candles = [
            {"time": "09:15", "timestamp": 0, "open": 100.0, "high": 102.0,
             "low": 99.0, "close": 101.0, "volume": 50000}
        ]
        mock_tick = {"price": 101.0, "open": 100.0, "high": 102.0, "low": 99.0,
                     "prev_close": 99.5, "change_pct": 1.51, "volume": 50000}

        with (
            patch(
                "alphavedha.api.routes.live._fetch_intraday_snapshot",
                return_value=mock_candles,
            ),
            patch(
                "alphavedha.api.routes.live._fetch_tick",
                return_value=mock_tick,
            ),
            patch(
                "alphavedha.api.routes.live._is_market_open",
                return_value=False,
            ),
        ):
            with self.client.websocket_connect("/ws/live/TCS") as ws:
                snapshot = json.loads(ws.receive_text())
                assert snapshot["type"] == "snapshot"
                assert snapshot["symbol"] == "TCS"
                assert snapshot["candles"] == mock_candles
                assert snapshot["tick"] == mock_tick
                assert "generated_at" in snapshot

                closed_msg = json.loads(ws.receive_text())
                assert closed_msg["type"] == "market_closed"

    def test_ws_market_connects_and_sends_summary(self) -> None:
        mock_tick = {"price": 22500.0, "open": 22400.0, "high": 22600.0, "low": 22300.0,
                     "prev_close": 22400.0, "change_pct": 0.45, "volume": 0}

        with (
            patch(
                "alphavedha.api.routes.live._fetch_tick",
                return_value=mock_tick,
            ),
            patch(
                "alphavedha.api.routes.live._is_market_open",
                return_value=False,
            ),
        ):
            with self.client.websocket_connect("/ws/market") as ws:
                summary = json.loads(ws.receive_text())
                assert summary["type"] == "market_summary"
                assert "indices" in summary
                assert len(summary["indices"]) == 3
                assert "generated_at" in summary
