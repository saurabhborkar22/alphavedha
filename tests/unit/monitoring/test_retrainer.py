"""Tests for auto retraining manager — retrain decisions, version lifecycle, and cleanup."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from alphavedha.config import RetrainingConfig
from alphavedha.monitoring.drift import DriftReport, DriftResult
from alphavedha.monitoring.performance import PerformanceReport, PerformanceSnapshot
from alphavedha.monitoring.retrainer import RetrainingManager


@pytest.fixture
def manager(tmp_path: Path) -> RetrainingManager:
    return RetrainingManager(artifact_dir=tmp_path / "artifacts")


@pytest.fixture
def drift_report_alert() -> DriftReport:
    return DriftReport(
        timestamp=datetime.now(UTC),
        features_checked=10,
        warnings=[],
        alerts=[
            DriftResult(
                feature_name="feat_a",
                psi_value=0.3,
                ks_statistic=0.5,
                ks_pvalue=0.001,
                is_warning=True,
                is_alert=True,
            )
        ],
        overall_psi=0.15,
        requires_retrain=True,
    )


@pytest.fixture
def drift_report_clean() -> DriftReport:
    return DriftReport(
        timestamp=datetime.now(UTC),
        features_checked=10,
        warnings=[],
        alerts=[],
        overall_psi=0.02,
        requires_retrain=False,
    )


@pytest.fixture
def perf_report_bad() -> PerformanceReport:
    return PerformanceReport(
        timestamp=datetime.now(UTC),
        model_version="v1.0.0",
        snapshots={
            30: PerformanceSnapshot(
                window_days=30,
                accuracy=0.48,
                precision_buy=0.45,
                precision_sell=0.50,
                magnitude_mae=0.02,
                n_predictions=100,
                profitable_pct=0.45,
            )
        },
        alpha_vs_benchmark=-0.005,
        requires_retrain=True,
    )


@pytest.fixture
def perf_report_good() -> PerformanceReport:
    return PerformanceReport(
        timestamp=datetime.now(UTC),
        model_version="v1.0.0",
        snapshots={
            30: PerformanceSnapshot(
                window_days=30,
                accuracy=0.65,
                precision_buy=0.68,
                precision_sell=0.62,
                magnitude_mae=0.01,
                n_predictions=100,
                profitable_pct=0.62,
            )
        },
        alpha_vs_benchmark=0.01,
        requires_retrain=False,
    )


class TestShouldRetrain:
    def test_should_retrain_on_schedule(self, manager: RetrainingManager) -> None:
        old_date = date.today() - timedelta(days=30)
        decision = manager.should_retrain(last_train_date=old_date)
        assert decision.should_retrain is True
        assert decision.reason == "scheduled"

    def test_should_retrain_on_drift(
        self, manager: RetrainingManager, drift_report_alert: DriftReport
    ) -> None:
        decision = manager.should_retrain(drift_report=drift_report_alert)
        assert decision.should_retrain is True
        assert decision.reason == "drift_detected"
        assert decision.drift_report is drift_report_alert

    def test_should_retrain_on_performance(
        self, manager: RetrainingManager, perf_report_bad: PerformanceReport
    ) -> None:
        decision = manager.should_retrain(performance_report=perf_report_bad)
        assert decision.should_retrain is True
        assert decision.reason == "performance_degraded"

    def test_should_not_retrain_when_recent(
        self,
        manager: RetrainingManager,
        drift_report_clean: DriftReport,
        perf_report_good: PerformanceReport,
    ) -> None:
        recent_date = date.today() - timedelta(days=1)
        decision = manager.should_retrain(
            drift_report=drift_report_clean,
            performance_report=perf_report_good,
            last_train_date=recent_date,
        )
        assert decision.should_retrain is False
        assert decision.reason == "no_trigger"

    def test_drift_takes_priority_over_schedule(
        self, manager: RetrainingManager, drift_report_alert: DriftReport
    ) -> None:
        recent_date = date.today() - timedelta(days=1)
        decision = manager.should_retrain(
            drift_report=drift_report_alert,
            last_train_date=recent_date,
        )
        assert decision.reason == "drift_detected"


class TestVersionLifecycle:
    def test_register_version_as_shadow(
        self, manager: RetrainingManager, tmp_path: Path
    ) -> None:
        artifact_path = tmp_path / "v1"
        artifact_path.mkdir()
        version = manager.register_version(
            version="v1.0.0",
            metrics={"accuracy": 0.65},
            artifact_path=artifact_path,
            data_range=("2023-01-01", "2024-01-01"),
        )
        assert version.status == "shadow"
        assert version.version == "v1.0.0"
        assert version.metrics["accuracy"] == 0.65

    def test_promote_version(
        self, manager: RetrainingManager, tmp_path: Path
    ) -> None:
        artifact_path = tmp_path / "v1"
        artifact_path.mkdir()
        manager.register_version(
            version="v1.0.0",
            metrics={"accuracy": 0.65},
            artifact_path=artifact_path,
            data_range=("2023-01-01", "2024-01-01"),
        )
        promoted = manager.promote_version("v1.0.0")
        assert promoted.status == "active"

    def test_promote_retires_previous_active(
        self, manager: RetrainingManager, tmp_path: Path
    ) -> None:
        p1 = tmp_path / "v1"
        p1.mkdir()
        p2 = tmp_path / "v2"
        p2.mkdir()

        manager.register_version("v1.0.0", {"accuracy": 0.60}, p1, ("2023-01-01", "2023-06-01"))
        manager.promote_version("v1.0.0")

        manager.register_version("v2.0.0", {"accuracy": 0.65}, p2, ("2023-01-01", "2024-01-01"))
        manager.promote_version("v2.0.0")

        active = manager.get_active_version()
        assert active is not None
        assert active.version == "v2.0.0"

        history = manager.get_version_history()
        v1 = next(v for v in history if v.version == "v1.0.0")
        assert v1.status == "retired"

    def test_get_active_version(
        self, manager: RetrainingManager, tmp_path: Path
    ) -> None:
        assert manager.get_active_version() is None

        p1 = tmp_path / "v1"
        p1.mkdir()
        manager.register_version("v1.0.0", {}, p1, ("2023-01-01", "2024-01-01"))
        assert manager.get_active_version() is None

        manager.promote_version("v1.0.0")
        active = manager.get_active_version()
        assert active is not None
        assert active.version == "v1.0.0"

    def test_promote_nonexistent_raises(self, manager: RetrainingManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            manager.promote_version("v99.0.0")

    def test_promote_non_shadow_raises(
        self, manager: RetrainingManager, tmp_path: Path
    ) -> None:
        p1 = tmp_path / "v1"
        p1.mkdir()
        manager.register_version("v1.0.0", {}, p1, ("2023-01-01", "2024-01-01"))
        manager.promote_version("v1.0.0")

        with pytest.raises(ValueError, match="can only promote shadow"):
            manager.promote_version("v1.0.0")


class TestCleanup:
    def test_cleanup_old_versions(self, tmp_path: Path) -> None:
        config = RetrainingConfig(keep_versions=2)
        manager = RetrainingManager(config=config, artifact_dir=tmp_path)

        paths = []
        for i in range(5):
            p = tmp_path / f"v{i}"
            p.mkdir()
            paths.append(p)
            manager.register_version(
                f"v{i}.0.0", {"accuracy": 0.5 + i * 0.02}, p, ("2023-01-01", "2024-01-01")
            )

        manager.promote_version("v2.0.0")

        removed = manager.cleanup_old_versions()
        assert len(removed) == 2

        remaining = manager.get_version_history()
        assert any(v.version == "v2.0.0" and v.status == "active" for v in remaining)

    def test_version_history_sorted(
        self, manager: RetrainingManager, tmp_path: Path
    ) -> None:
        for i in range(3):
            p = tmp_path / f"v{i}"
            p.mkdir()
            manager.register_version(
                f"v{i}.0.0", {}, p, ("2023-01-01", "2024-01-01")
            )

        history = manager.get_version_history()
        timestamps = [v.created_at for v in history]
        assert timestamps == sorted(timestamps)
