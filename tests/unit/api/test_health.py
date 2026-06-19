"""Tests for health and readiness endpoints."""

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


class TestHealthEndpoints:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_no_auth_required(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_ready_returns_status(self, client: TestClient) -> None:
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert "models_loaded" in data
        assert "cache_available" in data
        assert "database_available" in data
        assert "ready" in data

    def test_metrics_endpoint(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "http_request" in resp.text or "HELP" in resp.text

    def test_is_demo_middleware_returns_full_body(self, client: TestClient) -> None:
        # Regression: _IsDemoMiddleware was passing the original Content-Length header
        # through unchanged after enlarging the body with "is_demo", causing uvicorn to
        # raise RuntimeError("Response content longer than Content-Length") and close
        # every connection — all JSON endpoints returned 0 bytes.
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "is_demo" in data
        # Content-Length must match actual body — was stale (too small) before the fix
        cl = int(resp.headers["content-length"])
        assert cl == len(resp.content)
