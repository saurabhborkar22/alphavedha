"""Tests for prediction API routes — auth, predictions, batch, scan."""

from __future__ import annotations

import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from alphavedha.api.app import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(demo=True)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def authed_client() -> Iterator[TestClient]:
    with patch.dict(os.environ, {"ALPHAVEDHA_API_KEY": "test-key-123"}):
        app = create_app(demo=True)
        with TestClient(app) as c:
            yield c


class TestAuth:
    def test_no_env_key_means_open_access(self, client: TestClient) -> None:
        with patch.dict(os.environ, {"ALPHAVEDHA_API_KEY": ""}, clear=False):
            resp = client.get("/predict/TCS")
        assert resp.status_code == 200

    def test_missing_key_returns_401(self, authed_client: TestClient) -> None:
        resp = authed_client.get("/predict/TCS")
        assert resp.status_code == 401

    def test_invalid_key_returns_403(self, authed_client: TestClient) -> None:
        resp = authed_client.get("/predict/TCS", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403

    def test_valid_key_returns_200(self, authed_client: TestClient) -> None:
        resp = authed_client.get("/predict/TCS", headers={"X-API-Key": "test-key-123"})
        assert resp.status_code == 200


class TestPredictEndpoint:
    def test_predict_returns_prediction_response(self, client: TestClient) -> None:
        resp = client.get("/predict/TCS")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "TCS"
        assert data["direction"] in (-1, 0, 1)
        assert data["direction_label"] in ("BUY", "SELL", "HOLD")
        assert "price_targets" in data
        assert "risk" in data
        assert "model_version" in data
        assert "generated_at" in data


class TestBatchEndpoint:
    def test_batch_returns_predictions(self, client: TestClient) -> None:
        resp = client.post("/predict/batch", json={"symbols": ["TCS", "INFY"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["successful"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_rejects_empty_list(self, client: TestClient) -> None:
        resp = client.post("/predict/batch", json={"symbols": []})
        assert resp.status_code == 422

    def test_batch_rejects_over_20(self, client: TestClient) -> None:
        resp = client.post("/predict/batch", json={"symbols": [f"S{i}" for i in range(21)]})
        assert resp.status_code == 422


class TestScanEndpoint:
    def test_scan_returns_ranking(self, client: TestClient) -> None:
        resp = client.get("/scan/large?top_n=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "large"
        assert "buy_candidates" in data
        assert "sell_candidates" in data
        assert "excluded" in data
        assert data["total_scanned"] > 0
