"""Tests for feature drift detection — PSI computation, KS tests, and drift reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import DriftConfig
from alphavedha.monitoring.drift import DriftDetector, DriftReport


@pytest.fixture
def detector() -> DriftDetector:
    return DriftDetector()


@pytest.fixture
def strict_detector() -> DriftDetector:
    return DriftDetector(config=DriftConfig(psi_warning=0.05, psi_alert=0.1))


class TestComputePSI:
    def test_psi_identical_distributions(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=1000)
        psi = detector.compute_psi(data, data.copy())
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_psi_shifted_distribution(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, size=1000)
        current = rng.normal(3, 1, size=1000)
        psi = detector.compute_psi(reference, current)
        assert psi > 0.5

    def test_psi_with_zeros_handled(self, detector: DriftDetector) -> None:
        reference = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        current = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0])
        psi = detector.compute_psi(reference, current)
        assert np.isfinite(psi)
        assert psi >= 0.0

    def test_psi_empty_arrays(self, detector: DriftDetector) -> None:
        psi = detector.compute_psi(np.array([]), np.array([1.0, 2.0]))
        assert psi == 0.0


class TestComputeKS:
    def test_ks_identical_distributions(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=500)
        stat, pvalue = detector.compute_ks(data, data.copy())
        assert stat == pytest.approx(0.0, abs=0.01)
        assert pvalue > 0.9

    def test_ks_different_distributions(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, size=500)
        current = rng.normal(5, 1, size=500)
        stat, pvalue = detector.compute_ks(reference, current)
        assert stat > 0.5
        assert pvalue < 0.001


class TestCheckDrift:
    def test_check_drift_no_alerts(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(42)
        n = 500
        reference = pd.DataFrame(
            {
                "feat_a": rng.normal(0, 1, n),
                "feat_b": rng.normal(10, 2, n),
            }
        )
        current = pd.DataFrame(
            {
                "feat_a": rng.normal(0, 1, n),
                "feat_b": rng.normal(10, 2, n),
            }
        )
        report = detector.check_drift(reference, current)
        assert isinstance(report, DriftReport)
        assert report.features_checked == 2
        assert len(report.alerts) == 0
        assert report.requires_retrain is False

    def test_check_drift_with_alerts(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(42)
        n = 500
        reference = pd.DataFrame(
            {
                "feat_a": rng.normal(0, 1, n),
                "feat_b": rng.normal(0, 1, n),
            }
        )
        current = pd.DataFrame(
            {
                "feat_a": rng.normal(5, 1, n),
                "feat_b": rng.normal(0, 1, n),
            }
        )
        report = detector.check_drift(reference, current)
        assert len(report.alerts) >= 1
        alert_names = [a.feature_name for a in report.alerts]
        assert "feat_a" in alert_names

    def test_check_drift_requires_retrain(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(42)
        n = 500
        reference = pd.DataFrame({"x": rng.normal(0, 1, n)})
        current = pd.DataFrame({"x": rng.normal(10, 1, n)})
        report = detector.check_drift(reference, current)
        assert report.requires_retrain is True

    def test_check_drift_missing_columns_handled(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(42)
        n = 100
        reference = pd.DataFrame(
            {
                "feat_a": rng.normal(0, 1, n),
                "feat_b": rng.normal(0, 1, n),
            }
        )
        current = pd.DataFrame(
            {
                "feat_a": rng.normal(0, 1, n),
                "feat_c": rng.normal(0, 1, n),
            }
        )
        report = detector.check_drift(reference, current)
        assert report.features_checked == 1

    def test_check_drift_empty_dataframes(self, detector: DriftDetector) -> None:
        reference = pd.DataFrame()
        current = pd.DataFrame()
        report = detector.check_drift(reference, current)
        assert report.features_checked == 0
        assert report.requires_retrain is False

    def test_check_drift_overall_psi(self, strict_detector: DriftDetector) -> None:
        rng = np.random.default_rng(42)
        n = 500
        reference = pd.DataFrame(
            {
                "a": rng.normal(0, 1, n),
                "b": rng.normal(0, 1, n),
            }
        )
        current = pd.DataFrame(
            {
                "a": rng.normal(0, 1, n),
                "b": rng.normal(0, 1, n),
            }
        )
        report = strict_detector.check_drift(reference, current)
        assert report.overall_psi >= 0.0
        assert isinstance(report.overall_psi, float)
