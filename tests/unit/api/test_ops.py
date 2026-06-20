"""Tests for ops endpoints."""

from __future__ import annotations

import os
from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from alphavedha.api.app import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with patch.dict(os.environ, {"ALPHAVEDHA_API_KEY": "test-key-123"}):
        app = create_app(demo=True)
        with TestClient(app) as c:
            yield c


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test-key-123"}


class TestOpsHealth:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/api/ops/health")
        assert resp.status_code == 401

    def test_rejects_bad_key(self, client: TestClient) -> None:
        resp = client.get("/api/ops/health", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 403

    def test_returns_health_status(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/ops/health", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "critical")
        assert "infrastructure" in data
        assert "tables" in data
        assert "problems" in data
        assert "checked_at" in data

    def test_infrastructure_fields(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/ops/health", headers=auth_headers)
        data = resp.json()
        infra = data["infrastructure"]
        assert "database" in infra
        assert "redis" in infra
        assert "models_loaded" in infra
        assert "disk" in infra
        assert "used_pct" in infra["disk"]


class TestOpsTableCounts:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/api/ops/tables/counts")
        assert resp.status_code == 401

    def test_returns_counts(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/ops/tables/counts", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "counts" in data
        assert "checked_at" in data


class TestOpsTrigger:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.post("/api/ops/trigger/predictions")
        assert resp.status_code == 401

    def test_rejects_unknown_job(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.post("/api/ops/trigger/nonexistent_job", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "allowed_jobs" in data

    def test_lists_allowed_jobs(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.post("/api/ops/trigger/nonexistent_job", headers=auth_headers)
        data = resp.json()
        assert "predictions" in data["allowed_jobs"]
        assert "data_refresh" in data["allowed_jobs"]
        assert "surveillance" in data["allowed_jobs"]


class TestOpsIntelPush:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.post("/api/ops/intel/push", json={"table": "disclosures", "rows": []})
        assert resp.status_code == 401

    def test_rejects_missing_fields(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.post("/api/ops/intel/push", json={}, headers=auth_headers)
        data = resp.json()
        assert data["success"] is False
        assert "Missing" in data["error"]

    def test_rejects_unknown_table(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.post(
            "/api/ops/intel/push",
            json={"table": "fake_table", "rows": [{"a": 1}]},
            headers=auth_headers,
        )
        data = resp.json()
        assert data["success"] is False
        assert "allowed_tables" in data

    def test_accepts_valid_push(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        with patch(
            "alphavedha.intel.store.store_disclosures",
            new_callable=AsyncMock,
            return_value=2,
        ):
            resp = client.post(
                "/api/ops/intel/push",
                json={
                    "table": "disclosures",
                    "rows": [
                        {"symbol": "TCS.NS", "text": "test"},
                        {"symbol": "INFY.NS", "text": "test2"},
                    ],
                },
                headers=auth_headers,
            )
        data = resp.json()
        assert data["success"] is True
        assert data["rows_received"] == 2
        assert data["rows_stored"] == 2


class TestOpsSchedulerStatus:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/api/ops/scheduler/status")
        assert resp.status_code == 401

    def test_returns_status(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/ops/scheduler/status", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "scheduler_running" in data
