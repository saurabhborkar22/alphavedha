"""Unit tests for ExperimentTracker — JSON-based ML run logging and comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from alphavedha.monitoring.experiment_tracker import ExperimentTracker, RunRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}_\d{6}_.+$")


def _make_tracker(tmp_path: Path) -> ExperimentTracker:
    return ExperimentTracker(base_dir=tmp_path / "models" / "artifacts")


def _log_sample_run(
    tracker: ExperimentTracker,
    model_name: str = "xgboost",
    val_acc: float = 0.65,
    extra: dict | None = None,
) -> RunRecord:
    return tracker.log_run(
        model_name=model_name,
        hyperparams={"n_estimators": 100, "max_depth": 6},
        train_metrics={"accuracy": 0.72, "f1": 0.70},
        val_metrics={"accuracy": val_acc, "f1": 0.60},
        data_range=("2020-01-01", "2023-12-31"),
        n_train_rows=50_000,
        n_val_rows=10_000,
        n_symbols=50,
        feature_count=141,
        artifact_path="models/artifacts/xgboost_v1.pkl",
        duration_seconds=42.5,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_id_format(tmp_path: Path) -> None:
    """run_id must match YYYYMMDD_HHMMSS_<model_name>."""
    tracker = _make_tracker(tmp_path)
    record = _log_sample_run(tracker, model_name="xgboost")
    assert _RUN_ID_PATTERN.match(record.run_id), f"Bad run_id format: {record.run_id}"
    assert record.run_id.endswith("_xgboost")


def test_log_run_creates_json(tmp_path: Path) -> None:
    """log_run must write a JSON file containing all expected fields."""
    tracker = _make_tracker(tmp_path)
    record = _log_sample_run(tracker)

    run_files = list((tmp_path / "models" / "artifacts" / "runs").glob("*.json"))
    assert len(run_files) == 1

    data = json.loads(run_files[0].read_text())
    assert data["run_id"] == record.run_id
    assert data["model_name"] == "xgboost"
    assert data["n_train_rows"] == 50_000
    assert data["feature_count"] == 141
    assert "val_metrics" in data
    assert "hyperparams" in data


def test_log_run_with_extra(tmp_path: Path) -> None:
    """Extra metadata must be persisted in the JSON file."""
    tracker = _make_tracker(tmp_path)
    extra = {"git_sha": "abc123", "notes": "baseline run"}
    record = _log_sample_run(tracker, extra=extra)

    run_files = list((tmp_path / "models" / "artifacts" / "runs").glob("*.json"))
    data = json.loads(run_files[0].read_text())
    assert data["extra"] == extra
    assert record.extra == extra


def test_list_runs_returns_recent(tmp_path: Path) -> None:
    """list_runs with limit=3 must return exactly 3 most-recent records from 5 logged."""
    tracker = _make_tracker(tmp_path)
    for _ in range(5):
        _log_sample_run(tracker)

    runs = tracker.list_runs(limit=3)
    assert len(runs) == 3


def test_list_runs_filter_by_model(tmp_path: Path) -> None:
    """list_runs filtered by model_name must exclude records from other models."""
    tracker = _make_tracker(tmp_path)
    for _ in range(3):
        _log_sample_run(tracker, model_name="xgboost")
    for _ in range(2):
        _log_sample_run(tracker, model_name="lstm")

    xgb_runs = tracker.list_runs(model_name="xgboost")
    assert len(xgb_runs) == 3
    assert all(r.model_name == "xgboost" for r in xgb_runs)

    lstm_runs = tracker.list_runs(model_name="lstm")
    assert len(lstm_runs) == 2
    assert all(r.model_name == "lstm" for r in lstm_runs)


def test_get_run_exists(tmp_path: Path) -> None:
    """get_run must return the exact record that was logged."""
    tracker = _make_tracker(tmp_path)
    original = _log_sample_run(tracker)
    retrieved = tracker.get_run(original.run_id)

    assert retrieved is not None
    assert retrieved.run_id == original.run_id
    assert retrieved.model_name == original.model_name
    assert retrieved.feature_count == original.feature_count


def test_get_run_not_found(tmp_path: Path) -> None:
    """get_run must return None when the run_id does not exist."""
    tracker = _make_tracker(tmp_path)
    result = tracker.get_run("20990101_120000_nonexistent")
    assert result is None


def test_compare_runs(tmp_path: Path) -> None:
    """compare_runs must return per-metric delta (b - a) with correct values."""
    tracker = _make_tracker(tmp_path)
    run_a = _log_sample_run(tracker, val_acc=0.60)
    run_b = _log_sample_run(tracker, val_acc=0.70)

    comparison = tracker.compare_runs(run_a.run_id, run_b.run_id)

    assert "accuracy" in comparison
    assert comparison["accuracy"]["a"] == pytest.approx(0.60)
    assert comparison["accuracy"]["b"] == pytest.approx(0.70)
    assert comparison["accuracy"]["delta"] == pytest.approx(0.10)

    assert "f1" in comparison
    assert comparison["f1"]["delta"] == pytest.approx(0.0)


def test_compare_runs_missing(tmp_path: Path) -> None:
    """compare_runs must raise ValueError if either run_id does not exist."""
    tracker = _make_tracker(tmp_path)
    real_run = _log_sample_run(tracker)
    fake_id = "20000101_000000_ghost"

    with pytest.raises(ValueError, match=fake_id):
        tracker.compare_runs(real_run.run_id, fake_id)

    with pytest.raises(ValueError, match=fake_id):
        tracker.compare_runs(fake_id, real_run.run_id)
