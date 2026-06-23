"""Tests for ops endpoints."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import date
from unittest.mock import AsyncMock, patch

import pandas as pd
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


class TestOpsIntelPending:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/api/ops/intel/pending")
        assert resp.status_code == 401

    def test_returns_pending_disclosures(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        mock_disclosures = [
            {
                "id": 1,
                "symbol": "TCS.NS",
                "category": "Results",
                "headline": "Q1 results declared",
                "text": "Revenue up 10%",
                "text_hash": "abc123",
                "filed_at": None,
            },
        ]
        with patch(
            "alphavedha.api.routes.ops.get_unprocessed_disclosures",
            new_callable=AsyncMock,
            return_value=mock_disclosures,
        ):
            resp = client.get("/api/ops/intel/pending", headers=auth_headers)
        data = resp.json()
        assert data["count"] == 1
        assert data["pending"][0]["symbol"] == "TCS.NS"

    def test_filters_boilerplate(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        mock_disclosures = [
            {
                "id": 1,
                "symbol": "TCS.NS",
                "category": "Trading Window",
                "headline": "Closure of trading window",
                "text": None,
                "text_hash": None,
                "filed_at": None,
            },
        ]
        with (
            patch(
                "alphavedha.api.routes.ops.get_unprocessed_disclosures",
                new_callable=AsyncMock,
                return_value=mock_disclosures,
            ),
            patch(
                "alphavedha.api.routes.ops.mark_disclosures_processed",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            resp = client.get("/api/ops/intel/pending", headers=auth_headers)
        data = resp.json()
        assert data["count"] == 0
        assert data["boilerplate_skipped"] == 1

    def test_empty_when_none_pending(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        with patch(
            "alphavedha.api.routes.ops.get_unprocessed_disclosures",
            new_callable=AsyncMock,
            return_value=[],
        ):
            resp = client.get("/api/ops/intel/pending", headers=auth_headers)
        data = resp.json()
        assert data["count"] == 0
        assert data["pending"] == []


class TestOpsIntelEvents:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.post("/api/ops/intel/events", json={"events": [], "processed_ids": []})
        assert resp.status_code == 401

    def test_rejects_empty_payload(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.post(
            "/api/ops/intel/events",
            json={"events": [], "processed_ids": []},
            headers=auth_headers,
        )
        data = resp.json()
        assert data["success"] is False

    def test_accepts_events(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        with (
            patch(
                "alphavedha.api.routes.ops.store_disclosure_events",
                new_callable=AsyncMock,
                return_value=1,
            ),
            patch(
                "alphavedha.api.routes.ops.mark_disclosures_processed",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            resp = client.post(
                "/api/ops/intel/events",
                json={
                    "events": [
                        {
                            "disclosure_id": 1,
                            "symbol": "TCS.NS",
                            "event_type": "order_win",
                            "direction": 1,
                            "materiality": 7,
                            "confidence": 0.9,
                            "summary": "Won Rs 100 Cr order",
                            "red_flags": [],
                        }
                    ],
                    "processed_ids": [1],
                },
                headers=auth_headers,
            )
        data = resp.json()
        assert data["success"] is True
        assert data["events_stored"] == 1
        assert data["ids_marked_processed"] == 1

    def test_marks_processed_only(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        with patch(
            "alphavedha.api.routes.ops.mark_disclosures_processed",
            new_callable=AsyncMock,
            return_value=3,
        ):
            resp = client.post(
                "/api/ops/intel/events",
                json={"events": [], "processed_ids": [1, 2, 3]},
                headers=auth_headers,
            )
        data = resp.json()
        assert data["success"] is True
        assert data["events_stored"] == 0
        assert data["ids_marked_processed"] == 3


class TestOpsSchedulerStatus:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/api/ops/scheduler/status")
        assert resp.status_code == 401

    def test_returns_status(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/ops/scheduler/status", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "scheduler_running" in data


class TestOpsPredictionsSummary:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/api/ops/predictions/summary")
        assert resp.status_code == 401

    def test_returns_summary_with_predictions(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        trades_df = pd.DataFrame(
            [
                {
                    "symbol": "TCS.NS",
                    "prediction_date": date.today(),
                    "strategy": "ensemble_v1",
                    "predicted_direction": 1,
                    "predicted_magnitude": 0.02,
                    "confidence": 0.85,
                    "model_version": "v1.0",
                    "regime": "bull",
                    "is_tradeable": True,
                    "entry_price": 100.0,
                    "stop_loss_price": 95.0,
                    "take_profit_price": 110.0,
                    "exit_price": None,
                    "exit_reason": None,
                    "actual_return": None,
                    "is_correct": None,
                },
                {
                    "symbol": "INFY.NS",
                    "prediction_date": date.today(),
                    "strategy": "ensemble_v1",
                    "predicted_direction": -1,
                    "predicted_magnitude": 0.01,
                    "confidence": 0.7,
                    "model_version": "v1.0",
                    "regime": "bear",
                    "is_tradeable": False,
                    "entry_price": 200.0,
                    "stop_loss_price": 210.0,
                    "take_profit_price": 185.0,
                    "exit_price": None,
                    "exit_reason": None,
                    "actual_return": None,
                    "is_correct": None,
                },
            ]
        )
        pnl_df = pd.DataFrame()
        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades_df,
            ),
            patch(
                "alphavedha.data.store.load_daily_pnl",
                new_callable=AsyncMock,
                return_value=pnl_df,
            ),
        ):
            resp = client.get("/api/ops/predictions/summary", headers=auth_headers)
        data = resp.json()
        assert data["predictions"] == 2
        assert "TCS.NS" in data["symbols"]
        assert data["n_tradeable"] == 1
        assert data["directions"]["bullish"] == 1
        assert data["directions"]["bearish"] == 1

    def test_returns_empty_when_no_predictions(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=pd.DataFrame(),
            ),
            patch(
                "alphavedha.data.store.load_daily_pnl",
                new_callable=AsyncMock,
                return_value=pd.DataFrame(),
            ),
        ):
            resp = client.get("/api/ops/predictions/summary", headers=auth_headers)
        data = resp.json()
        assert data["predictions"] == 0
        assert data["symbols"] == []


class TestOpsModelsStatus:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/api/ops/models/status")
        assert resp.status_code == 401

    def test_returns_model_status(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        with patch.dict(os.environ, {"ALPHAVEDHA_MODEL_DIR": "/tmp/nonexistent_models"}):
            resp = client.get("/api/ops/models/status", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "summary" in data
        assert "xgboost" in data["models"]
        assert data["models"]["xgboost"]["status"] == "missing"


class TestOpsTableDeltas:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/api/ops/tables/deltas")
        assert resp.status_code == 401

    def test_returns_deltas(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        resp = client.get("/api/ops/tables/deltas", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "deltas" in data
        assert "date" in data


class TestOpsWeeklyReport:
    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/api/ops/weekly/report")
        assert resp.status_code == 401

    def test_returns_report(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        with (
            patch(
                "alphavedha.data.store.load_daily_pnl",
                new_callable=AsyncMock,
                return_value=pd.DataFrame(),
            ),
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=pd.DataFrame(),
            ),
        ):
            resp = client.get("/api/ops/weekly/report", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "period" in data
        assert "predictions" in data
        assert "performance" in data
        assert data["performance"]["trading_days"] == 0
