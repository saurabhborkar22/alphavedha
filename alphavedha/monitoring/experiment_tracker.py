"""Experiment tracker — JSON-based ML run logging and comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class RunRecord:
    run_id: str
    model_name: str
    started_at: str
    duration_seconds: float
    hyperparams: dict[str, Any]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    data_range: tuple[str, str]
    n_train_rows: int
    n_val_rows: int
    n_symbols: int
    feature_count: int
    artifact_path: str
    extra: dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path("models/artifacts")
        self._runs_dir = self.base_dir / "runs"
        self._runs_dir.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        model_name: str,
        hyperparams: dict[str, Any],
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        data_range: tuple[str, str],
        n_train_rows: int,
        n_val_rows: int,
        n_symbols: int,
        feature_count: int,
        artifact_path: str,
        duration_seconds: float,
        extra: dict[str, Any] | None = None,
    ) -> RunRecord:
        now = datetime.now(tz=UTC)
        run_id = (
            f"{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}_{now.strftime('%f')}_{model_name}"
        )

        record = RunRecord(
            run_id=run_id,
            model_name=model_name,
            started_at=now.isoformat(),
            duration_seconds=duration_seconds,
            hyperparams=hyperparams,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            data_range=data_range,
            n_train_rows=n_train_rows,
            n_val_rows=n_val_rows,
            n_symbols=n_symbols,
            feature_count=feature_count,
            artifact_path=artifact_path,
            extra=extra or {},
        )

        run_path = self._runs_dir / f"{run_id}.json"
        run_path.write_text(json.dumps(asdict(record), indent=2, default=str))
        logger.info("experiment_run_logged", run_id=run_id, model=model_name)
        return record

    def list_runs(self, model_name: str | None = None, limit: int = 20) -> list[RunRecord]:
        run_files = sorted(self._runs_dir.glob("*.json"), reverse=True)
        records: list[RunRecord] = []
        for path in run_files:
            record = self._load_record(path)
            if record is None:
                continue
            if model_name and record.model_name != model_name:
                continue
            records.append(record)
            if len(records) >= limit:
                break
        return records

    def get_run(self, run_id: str) -> RunRecord | None:
        path = self._runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return self._load_record(path)

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict[str, dict[str, float]]:
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)
        if run_a is None:
            raise ValueError(f"Run {run_id_a} not found")
        if run_b is None:
            raise ValueError(f"Run {run_id_b} not found")

        all_metrics = set(run_a.val_metrics.keys()) | set(run_b.val_metrics.keys())
        result: dict[str, dict[str, float]] = {}
        for metric in sorted(all_metrics):
            val_a = run_a.val_metrics.get(metric, 0.0)
            val_b = run_b.val_metrics.get(metric, 0.0)
            result[metric] = {"a": val_a, "b": val_b, "delta": val_b - val_a}
        return result

    def _load_record(self, path: Path) -> RunRecord | None:
        try:
            data = json.loads(path.read_text())
            data["data_range"] = tuple(data["data_range"])
            return RunRecord(**data)
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("experiment_run_load_failed", path=str(path))
            return None
