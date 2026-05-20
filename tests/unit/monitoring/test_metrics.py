"""Tests for Prometheus metrics definitions."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from alphavedha.monitoring.metrics import (
    ACTIVE_POSITIONS,
    CACHE_HITS,
    CACHE_MISSES,
    DRIFT_PSI,
    FEATURE_COMPUTE_LATENCY,
    MODEL_LOAD_LATENCY,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    SCHEDULER_JOB_DURATION,
    SCHEDULER_JOB_STATUS,
)


class TestMetricTypes:
    def test_prediction_latency_is_histogram(self) -> None:
        assert isinstance(PREDICTION_LATENCY, Histogram)

    def test_prediction_count_is_counter(self) -> None:
        assert isinstance(PREDICTION_COUNT, Counter)

    def test_prediction_confidence_is_histogram(self) -> None:
        assert isinstance(PREDICTION_CONFIDENCE, Histogram)

    def test_model_load_latency_is_histogram(self) -> None:
        assert isinstance(MODEL_LOAD_LATENCY, Histogram)

    def test_feature_compute_latency_is_histogram(self) -> None:
        assert isinstance(FEATURE_COMPUTE_LATENCY, Histogram)

    def test_scheduler_job_duration_is_histogram(self) -> None:
        assert isinstance(SCHEDULER_JOB_DURATION, Histogram)

    def test_scheduler_job_status_is_counter(self) -> None:
        assert isinstance(SCHEDULER_JOB_STATUS, Counter)

    def test_drift_psi_is_gauge(self) -> None:
        assert isinstance(DRIFT_PSI, Gauge)

    def test_active_positions_is_gauge(self) -> None:
        assert isinstance(ACTIVE_POSITIONS, Gauge)

    def test_cache_hits_is_counter(self) -> None:
        assert isinstance(CACHE_HITS, Counter)

    def test_cache_misses_is_counter(self) -> None:
        assert isinstance(CACHE_MISSES, Counter)


class TestMetricLabels:
    def test_prediction_latency_labels(self) -> None:
        assert PREDICTION_LATENCY._labelnames == ("symbol", "model")

    def test_prediction_count_labels(self) -> None:
        assert PREDICTION_COUNT._labelnames == ("direction", "tier")

    def test_scheduler_job_status_labels(self) -> None:
        assert SCHEDULER_JOB_STATUS._labelnames == ("job_name", "status")

    def test_drift_psi_labels(self) -> None:
        assert DRIFT_PSI._labelnames == ("feature_group",)
