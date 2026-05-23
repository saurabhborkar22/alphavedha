"""Tests for experiment tracking and model comparison CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from alphavedha.cli.main import app

runner = CliRunner()


def _create_run_file(
    runs_dir: Path,
    run_id: str,
    model: str,
    accuracy: float,
    f1: float,
) -> None:
    data = {
        "run_id": run_id,
        "model_name": model,
        "started_at": "2024-01-01T00:00:00+00:00",
        "duration_seconds": 100.0,
        "hyperparams": {"lr": 0.05},
        "train_metrics": {"accuracy": accuracy + 0.05, "f1": f1 + 0.05},
        "val_metrics": {"accuracy": accuracy, "f1": f1},
        "data_range": ["2024-01-01", "2024-12-31"],
        "n_train_rows": 1000,
        "n_val_rows": 200,
        "n_symbols": 50,
        "feature_count": 141,
        "artifact_path": f"models/artifacts/{model}/v1",
        "extra": {},
    }
    (runs_dir / f"{run_id}.json").write_text(json.dumps(data))


class TestExperimentList:
    def test_list_runs(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        _create_run_file(runs_dir, "20240101_120000_xgboost", "xgboost", 0.75, 0.73)
        _create_run_file(runs_dir, "20240102_120000_lstm", "lstm", 0.72, 0.70)

        with patch("alphavedha.training.pipeline.ARTIFACTS_DIR", tmp_path):
            result = runner.invoke(app, ["experiment", "list"])
        assert result.exit_code == 0
        assert "xgboost" in result.output
        assert "lstm" in result.output

    def test_list_runs_filter(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        _create_run_file(runs_dir, "20240101_120000_xgboost", "xgboost", 0.75, 0.73)
        _create_run_file(runs_dir, "20240102_120000_lstm", "lstm", 0.72, 0.70)

        with patch("alphavedha.training.pipeline.ARTIFACTS_DIR", tmp_path):
            result = runner.invoke(app, ["experiment", "list", "--model", "lstm"])
        assert result.exit_code == 0
        assert "lstm" in result.output

    def test_list_runs_empty(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        with patch("alphavedha.training.pipeline.ARTIFACTS_DIR", tmp_path):
            result = runner.invoke(app, ["experiment", "list"])
        assert result.exit_code == 0
        assert "No experiment runs" in result.output


class TestExperimentCompare:
    def test_compare_runs(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        _create_run_file(runs_dir, "20240101_120000_xgboost", "xgboost", 0.75, 0.73)
        _create_run_file(runs_dir, "20240102_120000_xgboost", "xgboost", 0.78, 0.76)

        with patch("alphavedha.training.pipeline.ARTIFACTS_DIR", tmp_path):
            result = runner.invoke(
                app,
                ["experiment", "compare", "20240101_120000_xgboost", "20240102_120000_xgboost"],
            )
        assert result.exit_code == 0
        assert "accuracy" in result.output

    def test_compare_missing_run(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        _create_run_file(runs_dir, "20240101_120000_xgboost", "xgboost", 0.75, 0.73)

        with patch("alphavedha.training.pipeline.ARTIFACTS_DIR", tmp_path):
            result = runner.invoke(
                app,
                ["experiment", "compare", "20240101_120000_xgboost", "nonexistent"],
            )
        assert result.exit_code == 1
        assert "not found" in result.output
