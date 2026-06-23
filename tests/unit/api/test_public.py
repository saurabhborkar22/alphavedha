"""Tests for public track record API — all endpoints, no auth required."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from alphavedha.api.app import create_app
from alphavedha.api.routes import public


def _public_only_app() -> FastAPI:
    """App with only the public router mounted.

    The full app registers ui_support first, which shadows some /public paths;
    these tests target the public.py implementation directly.
    """
    app = FastAPI()
    app.include_router(public.router)
    return app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("ALPHAVEDHA_DEMO", "1")
    app = create_app(demo=True)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def demo_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("ALPHAVEDHA_DEMO", "1")
    with TestClient(_public_only_app()) as c:
        yield c


def _sample_trades_df() -> pd.DataFrame:
    rows = [
        {
            "symbol": "TCS",
            "prediction_date": date(2026, 5, 4),
            "predicted_direction": 1,
            "predicted_magnitude": 0.012,
            "confidence": 0.72,
            "model_version": "v1.2.0",
            "regime": "bull",
            "entry_price": 4100.0,
            "exit_price": 4150.0,
            "actual_return": 0.0122,
            "is_correct": True,
        },
        {
            "symbol": "INFY",
            "prediction_date": date(2026, 5, 4),
            "predicted_direction": -1,
            "predicted_magnitude": 0.008,
            "confidence": 0.61,
            "model_version": "v1.2.0",
            "regime": "bull",
            "entry_price": 1500.0,
            "exit_price": 1512.0,
            "actual_return": 0.008,
            "is_correct": False,
        },
        {
            "symbol": "TCS",
            "prediction_date": date(2026, 5, 5),
            "predicted_direction": 1,
            "predicted_magnitude": 0.01,
            "confidence": 0.81,
            "model_version": "v1.2.0",
            "regime": "sideways",
            "entry_price": 4150.0,
            "exit_price": 4180.0,
            "actual_return": 0.0072,
            "is_correct": True,
        },
        {
            # Not yet evaluated — must be excluded from accuracy metrics.
            "symbol": "SBIN",
            "prediction_date": date(2026, 5, 6),
            "predicted_direction": 1,
            "predicted_magnitude": 0.009,
            "confidence": 0.66,
            "model_version": "v1.2.0",
            "regime": "bull",
            "entry_price": 820.0,
            "exit_price": None,
            "actual_return": None,
            "is_correct": None,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def real_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Demo off — public routes read mocked store functions (real-data path)."""
    monkeypatch.delenv("ALPHAVEDHA_DEMO", raising=False)

    async def fake_load_paper_trades(
        start: date | None = None,
        end: date | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        df = _sample_trades_df()
        if symbol:
            df = df[df["symbol"] == symbol]
        if start:
            df = df[df["prediction_date"] >= start]
        if end:
            df = df[df["prediction_date"] <= end]
        return df.reset_index(drop=True)

    async def fake_load_daily_pnl(
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr("alphavedha.data.store.load_paper_trades", fake_load_paper_trades)
    monkeypatch.setattr("alphavedha.data.store.load_daily_pnl", fake_load_daily_pnl)

    with TestClient(_public_only_app()) as c:
        yield c


@pytest.fixture
def empty_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Demo off with empty tables — endpoints must return honest zeros."""
    monkeypatch.delenv("ALPHAVEDHA_DEMO", raising=False)

    async def empty_df(*args: object, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr("alphavedha.data.store.load_paper_trades", empty_df)
    monkeypatch.setattr("alphavedha.data.store.load_daily_pnl", empty_df)

    with TestClient(_public_only_app()) as c:
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

    def test_public_router_demo_synthetic(self, demo_client: TestClient) -> None:
        """public.py's own demo path serves the synthetic dataset."""
        data = demo_client.get("/public/track-record").json()
        assert data["total_predictions"] > 0
        assert data["since"] == "2026-02-18"
        assert len(data["by_confidence"]) > 0
        assert len(data["accuracy_over_time"]) > 0


class TestTrackRecordReal:
    def test_computed_from_real_trades(self, real_client: TestClient) -> None:
        data = real_client.get("/public/track-record").json()
        assert data["total_predictions"] == 4
        # 2 of 3 evaluated trades correct.
        assert data["overall_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
        assert data["directional_accuracy"] == data["overall_accuracy"]
        assert data["accuracy_30d"] == pytest.approx(2 / 3, abs=1e-3)
        assert data["since"] == "2026-05-04"
        assert data["signal_breakdown"] == {"up": 3, "down": 1, "hold": 0}
        assert len(data["accuracy_over_time"]) > 0
        bands = {b["band"]: b for b in data["by_confidence"]}
        assert bands["55-65%"]["count"] == 1
        assert bands["65-75%"]["count"] == 1
        assert bands["75-85%"]["count"] == 1

    def test_empty_tables_return_zeros(self, empty_client: TestClient) -> None:
        data = empty_client.get("/public/track-record").json()
        assert data["total_predictions"] == 0
        assert data["overall_accuracy"] == 0.0
        assert data["sharpe"] == 0.0
        assert data["accuracy_over_time"] == []
        assert data["by_confidence"] == []
        assert data["recent_predictions"] == []
        assert data["since"] is None


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

    def test_real_rows_mapped(self, real_client: TestClient) -> None:
        data = real_client.get("/public/predictions").json()
        assert data["total"] == 4
        first = data["predictions"][0]
        assert first["symbol"] == "TCS"
        assert first["date"] == "2026-05-04"
        assert first["predicted_direction_label"] == "BUY"
        assert first["model_version"] == "v1.2.0"
        assert first["is_correct"] is True

    def test_real_filter_by_symbol(self, real_client: TestClient) -> None:
        data = real_client.get("/public/predictions?symbol=SBIN").json()
        assert data["total"] == 1
        assert data["predictions"][0]["is_correct"] is None

    def test_real_empty(self, empty_client: TestClient) -> None:
        data = empty_client.get("/public/predictions").json()
        assert data["total"] == 0
        assert data["predictions"] == []


class TestEquityCurve:
    def test_returns_points(self, client: TestClient) -> None:
        resp = client.get("/public/equity-curve")
        assert resp.status_code == 200
        data = resp.json()
        assert "points" in data
        assert len(data["points"]) > 0
        assert "start_value" in data

    def test_real_fallback_from_trades(self, real_client: TestClient) -> None:
        # DailyPnL mocked empty — curve compounds evaluated trade returns.
        data = real_client.get("/public/equity-curve").json()
        assert len(data["points"]) == 2  # two evaluated trade days
        assert data["points"][0]["date"] == "2026-05-04"
        assert data["current_value"] != data["start_value"]

    def test_real_empty(self, empty_client: TestClient) -> None:
        data = empty_client.get("/public/equity-curve").json()
        assert data["points"] == []


class TestMonthlyReturns:
    def test_structure(self, client: TestClient) -> None:
        resp = client.get("/public/monthly-returns")
        assert resp.status_code == 200
        data = resp.json()
        assert "returns" in data

    def test_real_grouped_by_month(self, real_client: TestClient) -> None:
        data = real_client.get("/public/monthly-returns").json()
        assert len(data["returns"]) == 1
        month = data["returns"][0]
        assert month["month"] == "2026-05"
        assert month["n_trades"] == 3  # evaluated trades only
        assert month["win_rate"] == pytest.approx(2 / 3, abs=1e-3)
        assert month["benchmark_return"] == 0.0  # no DailyPnL — honest zero


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

    def test_real_export_json(self, real_client: TestClient) -> None:
        data = real_client.get("/public/predictions/export?format=json").json()
        assert len(data["predictions"]) == 4
        assert data["predictions"][0]["symbol"] == "TCS"

    def test_real_export_csv(self, real_client: TestClient) -> None:
        resp = real_client.get("/public/predictions/export?format=csv")
        assert "text/csv" in resp.headers["content-type"]
        assert "TCS" in resp.text


class TestModelInfo:
    def test_returns_version(self, client: TestClient) -> None:
        resp = client.get("/public/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_version" in data
        assert "base_models" in data
        assert "feature_count" in data

    def test_real_missing_artifacts_honest(self, empty_client: TestClient) -> None:
        info = empty_client.get("/public/model-info").json()
        assert "model_version" in info
        assert "feature_count" in info
        assert isinstance(info["base_models"], list)


class TestProofs:
    def test_demo_list_proofs(self, demo_client: TestClient) -> None:
        resp = demo_client.get("/public/proofs")
        assert resp.status_code == 200
        data = resp.json()
        assert "proofs" in data
        assert data["total"] > 0
        assert "proofs_repo_url" in data
        proof = data["proofs"][0]
        assert "proof_date" in proof
        assert "sha256" in proof
        assert "n_predictions" in proof

    def test_demo_single_proof(self, demo_client: TestClient) -> None:
        resp = demo_client.get("/public/proofs/2026-06-02")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["proof_date"] == "2026-06-02"
        assert len(data["sha256"]) == 64
        assert data["proofs_repo_url"]

    def test_demo_limit(self, demo_client: TestClient) -> None:
        resp = demo_client.get("/public/proofs?limit=5")
        data = resp.json()
        assert data["total"] <= 5

    def test_full_app_proofs(self, client: TestClient) -> None:
        resp = client.get("/public/proofs")
        assert resp.status_code == 200


class TestProofVerification:
    def test_verify_proof_helper(self) -> None:
        from alphavedha.verification.hasher import sha256_hex, verify_proof

        payload = '{"test": "data"}'
        digest = sha256_hex(payload.encode("utf-8"))
        assert verify_proof(digest, payload) is True
        assert verify_proof("wrong_hash", payload) is False


class TestVerifyPage:
    def test_demo_verify_page(self, demo_client: TestClient) -> None:
        resp = demo_client.get("/public/verify")
        assert resp.status_code == 200
        data = resp.json()
        assert "hash_scheme" in data
        assert data["hash_scheme"]["algorithm"] == "SHA-256"
        assert "verification_steps" in data["hash_scheme"]
        assert len(data["hash_scheme"]["verification_steps"]) >= 4
        assert "proofs_repo_url" in data
        assert "stats" in data
        assert data["stats"]["total_proof_days"] > 0
        assert "recent_proofs" in data
        assert len(data["recent_proofs"]) > 0
        assert "claim" in data

    def test_demo_verify_recent_proofs_structure(self, demo_client: TestClient) -> None:
        data = demo_client.get("/public/verify").json()
        proof = data["recent_proofs"][0]
        assert "proof_date" in proof
        assert "sha256" in proof
        assert "n_predictions" in proof
        assert "verified" in proof

    def test_full_app_verify(self, client: TestClient) -> None:
        resp = client.get("/public/verify")
        assert resp.status_code == 200
        data = resp.json()
        assert "hash_scheme" in data


class TestIsDemo:
    def test_env_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for value in ("1", "true", "YES"):
            monkeypatch.setenv("ALPHAVEDHA_DEMO", value)
            assert public._is_demo() is True

    def test_env_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALPHAVEDHA_DEMO", raising=False)
        assert public._is_demo() is False
        monkeypatch.setenv("ALPHAVEDHA_DEMO", "0")
        assert public._is_demo() is False
