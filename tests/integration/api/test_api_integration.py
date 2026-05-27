from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from alphavedha.api.app import create_app

pytestmark = pytest.mark.integration


@pytest.fixture()
def client() -> Iterator[TestClient]:
    app = create_app(demo=True)
    with TestClient(app) as c:
        yield c


class TestHealthWithDB:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_ready_reports_db_status(self, client: TestClient) -> None:
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert "database_available" in data
        assert "models_loaded" in data
        assert data["models_loaded"] is True


class TestPredictEndpoints:
    def test_predict_returns_valid_response(self, client: TestClient) -> None:
        resp = client.get("/predict/TCS.NS")
        assert resp.status_code == 200
        data = resp.json()
        assert "symbol" in data
        assert "direction" in data
        assert "direction_label" in data
        assert "meta_confidence" in data
        assert "generated_at" in data
        assert "model_version" in data

    def test_predict_batch(self, client: TestClient) -> None:
        resp = client.post(
            "/predict/batch",
            json={"symbols": ["TCS.NS", "INFY.NS", "RELIANCE.NS"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3

    def test_predict_invalid_symbol_format(self, client: TestClient) -> None:
        resp = client.get("/predict/INVALID!!!")
        assert resp.status_code == 400

    def test_scan_tier(self, client: TestClient) -> None:
        resp = client.get("/scan/large?top_n=3")
        assert resp.status_code == 200
        data = resp.json()
        assert "buy_candidates" in data
        assert "sell_candidates" in data
        assert "total_scanned" in data

    def test_scan_invalid_tier(self, client: TestClient) -> None:
        resp = client.get("/scan/nonexistent")
        assert resp.status_code == 400
