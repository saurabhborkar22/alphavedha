"""Tests for public track record API — all endpoints, no auth required."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from alphavedha.api.app import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(demo=True)
    with TestClient(app) as c:
        yield c


class TestTrackRecord:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/public/track-record")
        assert resp.status_code == 200

    def test_has_summary(self, client: TestClient) -> None:
        data = client.get("/public/track-record").json()
        assert "total_predictions" in data
        assert "directional_accuracy" in data
        assert data["total_predictions"] > 0

    def test_has_breakdowns(self, client: TestClient) -> None:
        data = client.get("/public/track-record").json()
        assert len(data["by_confidence"]) > 0
        assert "band" in data["by_confidence"][0]
        assert "accuracy" in data["by_confidence"][0]

    def test_has_monthly_returns(self, client: TestClient) -> None:
        data = client.get("/public/track-record").json()
        assert len(data["recent_predictions"]) > 0
        first = data["recent_predictions"][0]
        assert "date" in first
        assert "symbol" in first
        assert "predicted" in first


class TestPredictions:
    def test_list_returns_200(self, client: TestClient) -> None:
        resp = client.get("/public/predictions")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert "total" in data

    def test_pagination(self, client: TestClient) -> None:
        resp = client.get("/public/predictions?page=1&page_size=10")
        data = resp.json()
        assert len(data["predictions"]) <= 10
        assert data["page"] == 1

    def test_filter_by_symbol(self, client: TestClient) -> None:
        resp = client.get("/public/predictions?symbol=TCS")
        data = resp.json()
        for pred in data["predictions"]:
            assert pred["symbol"] == "TCS"


class TestEquityCurve:
    def test_returns_points(self, client: TestClient) -> None:
        resp = client.get("/public/equity-curve")
        assert resp.status_code == 200
        data = resp.json()
        assert "points" in data
        assert len(data["points"]) > 0
        assert "start_value" in data


class TestMonthlyReturns:
    def test_structure(self, client: TestClient) -> None:
        resp = client.get("/public/monthly-returns")
        assert resp.status_code == 200
        data = resp.json()
        assert "returns" in data


class TestExport:
    def test_csv_content_type(self, client: TestClient) -> None:
        resp = client.get("/public/predictions/export?format=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "date,symbol" in resp.text

    def test_json_content_type(self, client: TestClient) -> None:
        resp = client.get("/public/predictions/export?format=json")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data


class TestModelInfo:
    def test_returns_version(self, client: TestClient) -> None:
        resp = client.get("/public/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_version" in data
        assert "base_models" in data
        assert "feature_count" in data
