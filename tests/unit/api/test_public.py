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
            "strategy": "ensemble_v1",
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
            "strategy": "ensemble_v1",
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
            "strategy": "ensemble_v1",
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
            "strategy": "ensemble_v1",
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
        {
            "symbol": "TCS",
            "prediction_date": date(2026, 5, 4),
            "strategy": "event_drift_v1",
            "predicted_direction": 1,
            "predicted_magnitude": 0.015,
            "confidence": 0.68,
            "model_version": "v1.2.0",
            "regime": "bull",
            "entry_price": 4100.0,
            "exit_price": 4130.0,
            "actual_return": 0.0073,
            "is_correct": True,
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
        strategy: str | None = None,
    ) -> pd.DataFrame:
        df = _sample_trades_df()
        if symbol:
            df = df[df["symbol"] == symbol]
        if strategy:
            df = df[df["strategy"] == strategy]
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
        assert data["total_predictions"] == 5
        # 3 of 4 evaluated trades correct.
        assert data["overall_accuracy"] == pytest.approx(3 / 4, abs=1e-3)
        assert data["directional_accuracy"] == data["overall_accuracy"]
        assert data["since"] == "2026-05-04"
        assert data["signal_breakdown"]["up"] == 4
        assert data["signal_breakdown"]["down"] == 1
        assert len(data["accuracy_over_time"]) > 0

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
        assert data["total"] == 5
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

    def test_wrong_short_counts_as_loss(self, real_client: TestClient) -> None:
        """Regression: actual_return is a price return — the INFY short that
        saw the price rise +0.8% must drag the curve down, not lift it."""
        data = real_client.get("/public/equity-curve").json()
        # 2026-05-04 mean trade return = (+0.0122 - 0.008 + 0.0073) / 3
        assert data["points"][0]["portfolio_value"] == pytest.approx(1_003_833.33, abs=0.01)

    def test_real_empty(self, empty_client: TestClient) -> None:
        data = empty_client.get("/public/equity-curve").json()
        assert data["points"] == []


def test_daily_returns_fallback_is_directional() -> None:
    """Regression: the Sharpe fallback counts a correct short as a gain."""
    rec = public.PredictionRecord(
        date="2026-05-04",
        symbol="INFY",
        predicted_direction=-1,
        predicted_direction_label="SELL",
        predicted_magnitude=0.01,
        confidence=0.6,
        regime="bear",
        actual_direction=-1,
        actual_return=-0.02,
        is_correct=True,
        model_version="v1.2.0",
        generated_at="2026-05-04T00:00:00+05:30",
    )
    assert public._daily_returns([rec], None) == [pytest.approx(0.02)]


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
        assert month["n_trades"] == 4  # evaluated trades only
        assert month["win_rate"] == pytest.approx(3 / 4, abs=1e-3)
        assert month["benchmark_return"] == 0.0  # no DailyPnL — honest zero

    def test_portfolio_return_is_directional(self, real_client: TestClient) -> None:
        """Regression: monthly P&L is predicted_direction * actual_return,
        so the wrong-way INFY short subtracts its price move."""
        data = real_client.get("/public/monthly-returns").json()
        month = data["returns"][0]
        # (+0.0122 - 0.008 + 0.0072 + 0.0073) / 4 = 0.004675 → rounds to 0.0047
        assert month["portfolio_return"] == pytest.approx(0.0047, abs=1e-4)


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
        assert len(data["predictions"]) == 5
        assert data["predictions"][0]["symbol"] == "TCS"

    def test_real_export_csv(self, real_client: TestClient) -> None:
        resp = real_client.get("/public/predictions/export?format=csv")
        assert "text/csv" in resp.headers["content-type"]
        assert "TCS" in resp.text


class TestStrategies:
    def test_demo_strategies(self, demo_client: TestClient) -> None:
        resp = demo_client.get("/public/strategies")
        assert resp.status_code == 200
        data = resp.json()
        assert "strategies" in data
        assert data["total"] >= 3
        strat = data["strategies"][0]
        assert "strategy" in strat
        assert "win_rate_net" in strat
        assert "sharpe_net" in strat
        assert "total_return_net" in strat

    def test_demo_includes_losers(self, demo_client: TestClient) -> None:
        data = demo_client.get("/public/strategies").json()
        losers = [s for s in data["strategies"] if (s["total_return_net"] or 0) < 0]
        assert len(losers) >= 1

    def test_full_app_strategies(self, client: TestClient) -> None:
        resp = client.get("/public/strategies")
        assert resp.status_code == 200

    def test_empty_strategies(self, empty_client: TestClient) -> None:
        data = empty_client.get("/public/strategies").json()
        assert data["strategies"] == []
        assert data["total"] == 0


class TestTrackRecordStrategyFilter:
    def test_filter_by_strategy(self, real_client: TestClient) -> None:
        data = real_client.get("/public/track-record?strategy=ensemble_v1").json()
        assert data["total_predictions"] == 4
        assert data["strategy"] == "ensemble_v1"

    def test_filter_event_drift(self, real_client: TestClient) -> None:
        data = real_client.get("/public/track-record?strategy=event_drift_v1").json()
        assert data["total_predictions"] == 1
        assert data["strategy"] == "event_drift_v1"

    def test_no_filter_returns_all(self, real_client: TestClient) -> None:
        data = real_client.get("/public/track-record").json()
        assert data["total_predictions"] == 5
        assert "strategy" not in data


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


class TestRedFlagRadar:
    def test_demo_radar(self, demo_client: TestClient) -> None:
        resp = demo_client.get("/public/red-flag-radar")
        assert resp.status_code == 200
        data = resp.json()
        assert "disclaimer" in data
        assert len(data["disclaimer"]) > 100
        assert "flagged_count" in data
        assert data["flagged_count"] >= 1
        assert "symbols" in data
        assert "generated_at" in data

    def test_demo_flags_are_cited(self, demo_client: TestClient) -> None:
        data = demo_client.get("/public/red-flag-radar").json()
        sym = data["symbols"][0]
        assert "flags" in sym
        flag = sym["flags"][0]
        assert "category" in flag
        assert "severity" in flag
        assert "description" in flag
        assert "source" in flag

    def test_demo_threshold_filter(self, demo_client: TestClient) -> None:
        data = demo_client.get("/public/red-flag-radar?threshold=90").json()
        assert data["threshold"] == 90
        for sym in data["symbols"]:
            assert sym["total_score"] >= 90

    def test_full_app_radar(self, client: TestClient) -> None:
        resp = client.get("/public/red-flag-radar")
        assert resp.status_code == 200
        data = resp.json()
        assert "disclaimer" in data


class TestFormatFlags:
    def test_known_flags(self) -> None:
        from alphavedha.intel.signals.blowup_score import BlowupScore

        score = BlowupScore(
            symbol="TEST",
            total_score=80,
            flags=["pledge_critical_50pct", "auditor_resignation", "rating_downgrade_CRISIL"],
            on_avoid_list=True,
        )
        formatted = public._format_flags(score)
        assert len(formatted) == 3
        assert formatted[0]["category"] == "Pledge"
        assert formatted[0]["severity"] == "critical"
        assert formatted[1]["category"] == "Governance"
        assert formatted[2]["category"] == "Rating"
        assert "CRISIL" in formatted[2]["description"]

    def test_surveillance_flag(self) -> None:
        from alphavedha.intel.signals.blowup_score import BlowupScore

        score = BlowupScore(
            symbol="TEST",
            total_score=15,
            flags=["surveillance_ASM_Stage_2"],
        )
        formatted = public._format_flags(score)
        assert len(formatted) == 1
        assert formatted[0]["category"] == "Surveillance"
        assert "ASM_Stage_2" in formatted[0]["description"]


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


class TestWeeklyDigest:
    def test_demo_digest(self, demo_client: TestClient) -> None:
        resp = demo_client.get("/public/weekly-digest")
        assert resp.status_code == 200
        data = resp.json()
        assert "week" in data
        assert "start" in data["week"]
        assert "this_week" in data
        assert "cumulative" in data
        assert data["cumulative"]["total_predictions"] > 0
        assert "highlight_strategy" in data
        assert "trend" in data
        assert "insight" in data
        assert len(data["insight"]) > 20
        assert "chart_data" in data
        assert len(data["chart_data"]) > 0

    def test_demo_chart_structure(self, demo_client: TestClient) -> None:
        data = demo_client.get("/public/weekly-digest").json()
        point = data["chart_data"][0]
        assert "week" in point
        assert "accuracy" in point
        assert "n_trades" in point

    def test_full_app_digest(self, client: TestClient) -> None:
        resp = client.get("/public/weekly-digest")
        assert resp.status_code == 200

    def test_empty_digest(self, empty_client: TestClient) -> None:
        data = empty_client.get("/public/weekly-digest").json()
        assert data["cumulative"]["total_predictions"] == 0
        assert data["this_week"]["predictions"] == 0
        assert data["chart_data"] == []
        assert data["highlight_strategy"] is None


class TestInsightGenerator:
    def test_early_days(self) -> None:
        result = public._generate_insight(0.5, 0.5, 3, 2, "steady")
        assert "Too early" in result

    def test_no_week_accuracy(self) -> None:
        result = public._generate_insight(0.55, None, 20, 3, "steady")
        assert "No trades matured" in result

    def test_improving(self) -> None:
        result = public._generate_insight(0.55, 0.60, 30, 4, "improving")
        assert "trending up" in result

    def test_declining(self) -> None:
        result = public._generate_insight(0.55, 0.45, 30, 4, "declining")
        assert "Tough week" in result
