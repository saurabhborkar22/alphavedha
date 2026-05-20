"""Tests for prediction performance tracking — accuracy, precision, alpha, and retrain triggers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import PerformanceMonitorConfig
from alphavedha.monitoring.performance import PerformanceMonitor, PerformanceReport


@pytest.fixture
def monitor() -> PerformanceMonitor:
    return PerformanceMonitor()


def _make_predictions_and_actuals(
    n: int,
    accuracy: float,
    rng: np.random.Generator,
    start_date: str = "2024-01-02",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Helper to generate prediction/actual DataFrames with controlled accuracy."""
    dates = pd.bdate_range(start_date, periods=n, freq="B")
    directions = rng.choice([-1, 1], size=n)
    correct_mask = rng.random(n) < accuracy

    actual_directions = np.where(correct_mask, directions, -directions)
    actual_returns = actual_directions * rng.uniform(0.005, 0.03, n)
    predicted_magnitudes = actual_returns + rng.normal(0, 0.005, n)

    predictions = pd.DataFrame({
        "date": dates,
        "symbol": "TCS",
        "predicted_direction": directions,
        "predicted_magnitude": predicted_magnitudes,
        "confidence": rng.uniform(0.5, 0.9, n),
    })
    actuals = pd.DataFrame({
        "date": dates,
        "symbol": "TCS",
        "actual_direction": actual_directions,
        "actual_return": actual_returns,
    })
    return predictions, actuals


class TestEvaluate:
    def test_evaluate_perfect_predictions(self, monitor: PerformanceMonitor) -> None:
        rng = np.random.default_rng(42)
        n = 100
        dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
        directions = rng.choice([-1, 1], size=n)
        returns = directions * rng.uniform(0.01, 0.03, n)

        predictions = pd.DataFrame({
            "date": dates,
            "symbol": "TCS",
            "predicted_direction": directions,
            "predicted_magnitude": returns,
            "confidence": np.full(n, 0.9),
        })
        actuals = pd.DataFrame({
            "date": dates,
            "symbol": "TCS",
            "actual_direction": directions,
            "actual_return": returns,
        })

        report = monitor.evaluate(predictions, actuals, model_version="v1.0.0")
        assert isinstance(report, PerformanceReport)
        assert report.model_version == "v1.0.0"
        assert report.requires_retrain is False

        for snap in report.snapshots.values():
            assert snap.accuracy == pytest.approx(1.0)
            assert snap.magnitude_mae == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_random_predictions(self, monitor: PerformanceMonitor) -> None:
        rng = np.random.default_rng(42)
        predictions, actuals = _make_predictions_and_actuals(200, 0.5, rng)

        report = monitor.evaluate(predictions, actuals)
        snap_90 = report.snapshots[90]
        assert 0.3 <= snap_90.accuracy <= 0.7

    def test_evaluate_multiple_windows(self) -> None:
        config = PerformanceMonitorConfig(rolling_windows=[7, 30, 90])
        monitor = PerformanceMonitor(config=config)
        rng = np.random.default_rng(42)
        predictions, actuals = _make_predictions_and_actuals(200, 0.7, rng)

        report = monitor.evaluate(predictions, actuals)
        assert len(report.snapshots) == 3
        assert set(report.snapshots.keys()) == {7, 30, 90}

    def test_evaluate_requires_retrain_low_accuracy(self) -> None:
        config = PerformanceMonitorConfig(min_accuracy=0.90)
        monitor = PerformanceMonitor(config=config)
        rng = np.random.default_rng(42)
        predictions, actuals = _make_predictions_and_actuals(200, 0.5, rng)

        report = monitor.evaluate(predictions, actuals)
        assert report.requires_retrain is True

    def test_evaluate_no_retrain_good_accuracy(self, monitor: PerformanceMonitor) -> None:
        rng = np.random.default_rng(42)
        predictions, actuals = _make_predictions_and_actuals(200, 0.85, rng)

        report = monitor.evaluate(predictions, actuals)
        assert report.requires_retrain is False

    def test_evaluate_alpha_computation(self, monitor: PerformanceMonitor) -> None:
        rng = np.random.default_rng(42)
        predictions, actuals = _make_predictions_and_actuals(200, 0.8, rng)

        report = monitor.evaluate(predictions, actuals)
        assert isinstance(report.alpha_vs_benchmark, float)
        assert np.isfinite(report.alpha_vs_benchmark)

    def test_evaluate_empty_predictions(self, monitor: PerformanceMonitor) -> None:
        predictions = pd.DataFrame(
            columns=["date", "symbol", "predicted_direction", "predicted_magnitude", "confidence"]
        )
        actuals = pd.DataFrame(
            columns=["date", "symbol", "actual_direction", "actual_return"]
        )
        report = monitor.evaluate(predictions, actuals)
        assert len(report.snapshots) == 0
        assert report.requires_retrain is False

    def test_evaluate_single_window(self) -> None:
        config = PerformanceMonitorConfig(rolling_windows=[30])
        monitor = PerformanceMonitor(config=config)
        rng = np.random.default_rng(42)
        predictions, actuals = _make_predictions_and_actuals(100, 0.7, rng)

        report = monitor.evaluate(predictions, actuals)
        assert len(report.snapshots) == 1
        assert 30 in report.snapshots
        snap = report.snapshots[30]
        assert snap.window_days == 30
        assert snap.n_predictions > 0
