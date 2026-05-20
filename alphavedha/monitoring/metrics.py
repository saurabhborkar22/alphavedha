"""Prometheus metrics for API, predictions, and model inference."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTION_LATENCY = Histogram(
    "alphavedha_prediction_seconds",
    "Time to generate a single prediction",
    ["symbol", "model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PREDICTION_COUNT = Counter(
    "alphavedha_predictions_total",
    "Total predictions generated",
    ["direction", "tier"],
)

PREDICTION_CONFIDENCE = Histogram(
    "alphavedha_prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
)

MODEL_LOAD_LATENCY = Histogram(
    "alphavedha_model_load_seconds",
    "Time to load a model from disk",
    ["model_type"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

FEATURE_COMPUTE_LATENCY = Histogram(
    "alphavedha_feature_compute_seconds",
    "Time to compute features for a symbol",
    ["symbol"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

SCHEDULER_JOB_DURATION = Histogram(
    "alphavedha_scheduler_job_seconds",
    "Duration of scheduler jobs",
    ["job_name"],
    buckets=(1.0, 5.0, 30.0, 60.0, 300.0, 600.0),
)

SCHEDULER_JOB_STATUS = Counter(
    "alphavedha_scheduler_job_total",
    "Scheduler job completions by status",
    ["job_name", "status"],
)

DRIFT_PSI = Gauge(
    "alphavedha_drift_psi",
    "Current PSI value per feature group",
    ["feature_group"],
)

ACTIVE_POSITIONS = Gauge(
    "alphavedha_active_positions",
    "Number of active paper trading positions",
)

CACHE_HITS = Counter(
    "alphavedha_cache_hits_total",
    "Prediction cache hit count",
)

CACHE_MISSES = Counter(
    "alphavedha_cache_misses_total",
    "Prediction cache miss count",
)
