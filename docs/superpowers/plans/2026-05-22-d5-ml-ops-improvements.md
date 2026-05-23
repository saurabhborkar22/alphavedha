# D5: ML Operations Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add experiment tracking, model serving improvements, automated model comparison, and RL pipeline integration — all lightweight, file-based, zero new dependencies.

**Architecture:** Four independent modules: (1) JSON-based experiment tracker logging runs alongside model artifacts, (2) eager warm-up + async batch prediction in PredictionService, (3) automated shadow vs active model comparison in RetrainingManager, (4) RL training wired into `train_all()` with walk-forward validation.

**Tech Stack:** Python 3.12, asyncio, dataclasses, structlog, Typer, Rich, pytest

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `alphavedha/monitoring/experiment_tracker.py` | `ExperimentTracker` + `RunRecord` |
| Modify | `alphavedha/training/pipeline.py` | Log runs after each model trains + RL Step 10 |
| Modify | `alphavedha/services/prediction_service.py` | `warm_up()` + async `predict_batch()` |
| Modify | `alphavedha/api/app.py` | Call `warm_up()` in lifespan |
| Modify | `alphavedha/monitoring/retrainer.py` | `ComparisonResult` + `compare_models()` |
| Modify | `alphavedha/training/rl_pipeline.py` | `WalkForwardResult` + `walk_forward_rl()` |
| Modify | `alphavedha/cli/main.py` | `experiment` + `model` subcommands |
| Create | `tests/unit/monitoring/test_experiment_tracker.py` | Tracker tests |
| Extend | `tests/unit/monitoring/test_retrainer.py` | Comparison tests |
| Extend | `tests/unit/services/test_prediction_service.py` | Warm-up + batch tests |
| Extend | `tests/unit/training/test_pipeline.py` | RL integration test |
| Extend | `tests/unit/training/test_rl_pipeline.py` | Walk-forward tests |

---

### Task 1: Experiment Tracker — Core Class + Tests

**Files:**
- Create: `alphavedha/monitoring/experiment_tracker.py`
- Create: `tests/unit/monitoring/test_experiment_tracker.py`
- Modify: `alphavedha/monitoring/__init__.py`

- [ ] **Step 1: Write failing tests for ExperimentTracker**

```python
# tests/unit/monitoring/test_experiment_tracker.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphavedha.monitoring.experiment_tracker import ExperimentTracker, RunRecord


@pytest.fixture()
def tracker(tmp_path: Path) -> ExperimentTracker:
    return ExperimentTracker(base_dir=tmp_path)


class TestRunRecord:
    def test_run_id_format(self, tracker: ExperimentTracker) -> None:
        record = tracker.log_run(
            model_name="xgboost",
            hyperparams={"lr": 0.05},
            train_metrics={"accuracy": 0.80},
            val_metrics={"accuracy": 0.75},
            data_range=("2024-01-01", "2024-12-31"),
            n_train_rows=1000,
            n_val_rows=200,
            n_symbols=50,
            feature_count=141,
            artifact_path="models/artifacts/xgboost/v1",
            duration_seconds=120.5,
        )
        # Format: YYYYMMDD_HHMMSS_modelname
        parts = record.run_id.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # date
        assert len(parts[1]) == 6  # time
        assert parts[2] == "xgboost"


class TestLogRun:
    def test_log_run_creates_json(self, tracker: ExperimentTracker) -> None:
        record = tracker.log_run(
            model_name="xgboost",
            hyperparams={"lr": 0.05, "max_depth": 6},
            train_metrics={"accuracy": 0.80, "f1": 0.78},
            val_metrics={"accuracy": 0.75, "f1": 0.73},
            data_range=("2024-01-01", "2024-12-31"),
            n_train_rows=1000,
            n_val_rows=200,
            n_symbols=50,
            feature_count=141,
            artifact_path="models/artifacts/xgboost/v1",
            duration_seconds=120.5,
        )
        json_path = tracker.base_dir / "runs" / f"{record.run_id}.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["model_name"] == "xgboost"
        assert data["hyperparams"]["lr"] == 0.05
        assert data["val_metrics"]["accuracy"] == 0.75
        assert data["n_train_rows"] == 1000
        assert data["duration_seconds"] == 120.5

    def test_log_run_with_extra(self, tracker: ExperimentTracker) -> None:
        record = tracker.log_run(
            model_name="regime",
            hyperparams={"n_states": 4},
            train_metrics={"log_likelihood": -500.0},
            val_metrics={"log_likelihood": -520.0},
            data_range=("2024-01-01", "2024-12-31"),
            n_train_rows=1000,
            n_val_rows=200,
            n_symbols=1,
            feature_count=2,
            artifact_path="models/artifacts/regime/v1",
            duration_seconds=30.0,
            extra={"regime_counts": {"bull": 100, "bear": 50}},
        )
        json_path = tracker.base_dir / "runs" / f"{record.run_id}.json"
        data = json.loads(json_path.read_text())
        assert data["extra"]["regime_counts"]["bull"] == 100


class TestListRuns:
    def test_list_runs_returns_recent(self, tracker: ExperimentTracker) -> None:
        for i in range(5):
            tracker.log_run(
                model_name="xgboost",
                hyperparams={"lr": 0.05},
                train_metrics={"accuracy": 0.70 + i * 0.02},
                val_metrics={"accuracy": 0.65 + i * 0.02},
                data_range=("2024-01-01", "2024-12-31"),
                n_train_rows=1000,
                n_val_rows=200,
                n_symbols=50,
                feature_count=141,
                artifact_path=f"models/artifacts/xgboost/v{i}",
                duration_seconds=100.0 + i,
            )
        runs = tracker.list_runs(limit=3)
        assert len(runs) == 3
        # Most recent first
        assert runs[0].duration_seconds >= runs[-1].duration_seconds

    def test_list_runs_filter_by_model(self, tracker: ExperimentTracker) -> None:
        for model in ["xgboost", "lstm", "xgboost"]:
            tracker.log_run(
                model_name=model,
                hyperparams={},
                train_metrics={"accuracy": 0.75},
                val_metrics={"accuracy": 0.70},
                data_range=("2024-01-01", "2024-12-31"),
                n_train_rows=1000,
                n_val_rows=200,
                n_symbols=50,
                feature_count=141,
                artifact_path=f"models/artifacts/{model}/v1",
                duration_seconds=100.0,
            )
        runs = tracker.list_runs(model_name="lstm")
        assert len(runs) == 1
        assert runs[0].model_name == "lstm"


class TestGetRun:
    def test_get_run_exists(self, tracker: ExperimentTracker) -> None:
        record = tracker.log_run(
            model_name="tft",
            hyperparams={"d_model": 64},
            train_metrics={"accuracy": 0.80},
            val_metrics={"accuracy": 0.76},
            data_range=("2024-01-01", "2024-12-31"),
            n_train_rows=1000,
            n_val_rows=200,
            n_symbols=50,
            feature_count=141,
            artifact_path="models/artifacts/tft/v1",
            duration_seconds=300.0,
        )
        retrieved = tracker.get_run(record.run_id)
        assert retrieved is not None
        assert retrieved.model_name == "tft"
        assert retrieved.val_metrics["accuracy"] == 0.76

    def test_get_run_not_found(self, tracker: ExperimentTracker) -> None:
        result = tracker.get_run("nonexistent_run_id")
        assert result is None


class TestCompareRuns:
    def test_compare_runs(self, tracker: ExperimentTracker) -> None:
        run_a = tracker.log_run(
            model_name="xgboost",
            hyperparams={"lr": 0.05},
            train_metrics={"accuracy": 0.80},
            val_metrics={"accuracy": 0.75, "f1": 0.73},
            data_range=("2024-01-01", "2024-12-31"),
            n_train_rows=1000,
            n_val_rows=200,
            n_symbols=50,
            feature_count=141,
            artifact_path="models/artifacts/xgboost/v1",
            duration_seconds=100.0,
        )
        run_b = tracker.log_run(
            model_name="xgboost",
            hyperparams={"lr": 0.03},
            train_metrics={"accuracy": 0.82},
            val_metrics={"accuracy": 0.78, "f1": 0.76},
            data_range=("2024-01-01", "2024-12-31"),
            n_train_rows=1000,
            n_val_rows=200,
            n_symbols=50,
            feature_count=141,
            artifact_path="models/artifacts/xgboost/v2",
            duration_seconds=110.0,
        )
        comparison = tracker.compare_runs(run_a.run_id, run_b.run_id)
        assert comparison["accuracy"]["a"] == 0.75
        assert comparison["accuracy"]["b"] == 0.78
        assert comparison["accuracy"]["delta"] == pytest.approx(0.03)
        assert comparison["f1"]["delta"] == pytest.approx(0.03)

    def test_compare_runs_missing(self, tracker: ExperimentTracker) -> None:
        record = tracker.log_run(
            model_name="xgboost",
            hyperparams={},
            train_metrics={"accuracy": 0.80},
            val_metrics={"accuracy": 0.75},
            data_range=("2024-01-01", "2024-12-31"),
            n_train_rows=1000,
            n_val_rows=200,
            n_symbols=50,
            feature_count=141,
            artifact_path="models/artifacts/xgboost/v1",
            duration_seconds=100.0,
        )
        with pytest.raises(ValueError, match="not found"):
            tracker.compare_runs(record.run_id, "nonexistent")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/monitoring/test_experiment_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'alphavedha.monitoring.experiment_tracker'`

- [ ] **Step 3: Implement ExperimentTracker**

```python
# alphavedha/monitoring/experiment_tracker.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
        now = datetime.now(tz=timezone.utc)
        run_id = f"{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}_{model_name}"

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

    def list_runs(
        self, model_name: str | None = None, limit: int = 20
    ) -> list[RunRecord]:
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

    def compare_runs(
        self, run_id_a: str, run_id_b: str
    ) -> dict[str, dict[str, float]]:
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
```

- [ ] **Step 4: Export from `__init__.py`**

Add to `alphavedha/monitoring/__init__.py`:

```python
from alphavedha.monitoring.experiment_tracker import ExperimentTracker, RunRecord
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/monitoring/test_experiment_tracker.py -v`
Expected: All 9 tests PASS

- [ ] **Step 6: Lint check**

Run: `ruff check alphavedha/monitoring/experiment_tracker.py tests/unit/monitoring/test_experiment_tracker.py`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add alphavedha/monitoring/experiment_tracker.py alphavedha/monitoring/__init__.py tests/unit/monitoring/test_experiment_tracker.py
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat(d5): add ExperimentTracker with JSON run logging and comparison"
```

---

### Task 2: Training Pipeline — Experiment Logging Integration

**Files:**
- Modify: `alphavedha/training/pipeline.py`
- Extend: `tests/unit/training/test_pipeline.py`

- [ ] **Step 1: Write failing test for experiment logging in train_all**

Add to `tests/unit/training/test_pipeline.py`:

```python
class TestExperimentLogging:
    def test_train_all_logs_experiments(
        self, tmp_path: Path, tier_data: TierData
    ) -> None:
        """train_all should log experiment runs for each successfully trained model."""
        runs_dir = tmp_path / "runs"
        with (
            patch("alphavedha.training.pipeline.ARTIFACTS_DIR", tmp_path),
            patch("alphavedha.training.pipeline._train_xgboost_on_data") as mock_xgb,
            patch("alphavedha.training.pipeline._train_lstm_on_data") as mock_lstm,
            patch("alphavedha.training.pipeline._select_top_features") as mock_feat,
            patch("alphavedha.training.pipeline._train_tft_on_data") as mock_tft,
            patch("alphavedha.training.pipeline._train_regime_on_data") as mock_regime,
            patch("alphavedha.training.pipeline._train_ensemble_on_data") as mock_ens,
            patch("alphavedha.training.pipeline._train_meta_labeling_on_data") as mock_meta,
            patch("alphavedha.training.pipeline._train_conformal_on_data") as mock_conf,
        ):
            mock_result = TrainingPipelineResult(model_name="xgboost")
            mock_result.metrics = {"accuracy": 0.75, "f1": 0.73}
            mock_xgb.return_value = mock_result
            mock_feat.return_value = list(range(30))
            for m in [mock_lstm, mock_tft, mock_regime, mock_ens, mock_meta, mock_conf]:
                r = TrainingPipelineResult(model_name="test")
                r.metrics = {"accuracy": 0.70}
                m.return_value = r

            train_all(tier_data)

        assert runs_dir.exists()
        run_files = list(runs_dir.glob("*.json"))
        assert len(run_files) >= 1  # At least xgboost logged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/training/test_pipeline.py::TestExperimentLogging -v`
Expected: FAIL

- [ ] **Step 3: Add experiment logging to train_all()**

In `alphavedha/training/pipeline.py`, add an import at the top:

```python
from alphavedha.monitoring.experiment_tracker import ExperimentTracker
```

Then, at the start of `train_all()`, create the tracker:

```python
tracker = ExperimentTracker(base_dir=ARTIFACTS_DIR)
```

After each `_train_*_on_data()` call that succeeds, add a one-liner:

```python
# After xgboost training succeeds:
tracker.log_run(
    model_name="xgboost",
    hyperparams=xgb_result.hyperparams if hasattr(xgb_result, 'hyperparams') else {},
    train_metrics=xgb_result.metrics,
    val_metrics=xgb_result.metrics,
    data_range=(str(data.train_start), str(data.train_end)),
    n_train_rows=len(data.X_train),
    n_val_rows=len(data.X_val),
    n_symbols=len(data.symbols),
    feature_count=data.X_train.shape[1],
    artifact_path=str(xgb_result.artifact_path) if xgb_result.artifact_path else "",
    duration_seconds=xgb_result.duration_seconds if hasattr(xgb_result, 'duration_seconds') else 0.0,
    extra=xgb_result.extra_metrics if hasattr(xgb_result, 'extra_metrics') else {},
)
```

Add a helper to reduce repetition:

```python
def _log_experiment(
    tracker: ExperimentTracker,
    result: TrainingPipelineResult,
    data: TierData,
) -> None:
    if result.errors:
        return
    tracker.log_run(
        model_name=result.model_name,
        hyperparams=getattr(result, "hyperparams", {}),
        train_metrics=result.metrics,
        val_metrics=result.metrics,
        data_range=(str(data.train_start), str(data.train_end)),
        n_train_rows=len(data.X_train),
        n_val_rows=len(data.X_val),
        n_symbols=len(data.symbols),
        feature_count=data.X_train.shape[1],
        artifact_path=str(result.artifact_path) if result.artifact_path else "",
        duration_seconds=getattr(result, "duration_seconds", 0.0),
        extra=getattr(result, "extra_metrics", {}),
    )
```

Call `_log_experiment(tracker, result, data)` after each successful training step (xgboost, lstm, tft, regime, ensemble, meta_labeling, conformal).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/training/test_pipeline.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add alphavedha/training/pipeline.py tests/unit/training/test_pipeline.py
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat(d5): integrate experiment logging into training pipeline"
```

---

### Task 3: Model Serving — Warm-Up + Batch Optimization

**Files:**
- Modify: `alphavedha/services/prediction_service.py`
- Modify: `alphavedha/api/app.py`
- Extend: `tests/unit/services/test_prediction_service.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/services/test_prediction_service.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch


class TestWarmUp:
    @pytest.mark.asyncio()
    async def test_warmup_runs_prediction(self, service: PredictionService) -> None:
        """warm_up should call predict_single once to exercise the full path."""
        service.predict_single = AsyncMock(return_value=_make_mock_prediction())
        await service.warm_up()
        service.predict_single.assert_called_once()

    @pytest.mark.asyncio()
    async def test_warmup_failure_does_not_raise(
        self, service: PredictionService
    ) -> None:
        """warm_up should log warning and not raise if prediction fails."""
        service.predict_single = AsyncMock(side_effect=RuntimeError("model not loaded"))
        await service.warm_up()  # Should not raise


class TestBatchConcurrent:
    @pytest.mark.asyncio()
    async def test_predict_batch_concurrent_all_symbols(
        self, service: PredictionService
    ) -> None:
        """predict_batch should return predictions for all symbols."""
        service.predict_single = AsyncMock(return_value=_make_mock_prediction())
        symbols = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]
        results = await service.predict_batch(symbols)
        assert len(results) == 3
        assert service.predict_single.call_count == 3

    @pytest.mark.asyncio()
    async def test_predict_batch_preserves_order(
        self, service: PredictionService
    ) -> None:
        """predict_batch should return results in the same order as input symbols."""
        call_count = 0

        async def _mock_predict(symbol: str) -> Any:
            nonlocal call_count
            call_count += 1
            pred = _make_mock_prediction()
            pred.symbol = symbol
            return pred

        service.predict_single = AsyncMock(side_effect=_mock_predict)
        symbols = ["A.NS", "B.NS", "C.NS"]
        results = await service.predict_batch(symbols)
        assert [r.symbol for r in results] == symbols

    @pytest.mark.asyncio()
    async def test_scan_tier_concurrent(self, service: PredictionService) -> None:
        """scan_tier should use concurrent prediction."""
        service.predict_single = AsyncMock(return_value=_make_mock_prediction())
        with patch.object(service, "_get_symbols", return_value=["TCS.NS", "INFY.NS"]):
            results = await service.scan_tier("large")
        assert len(results) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/services/test_prediction_service.py::TestWarmUp -v`
Expected: FAIL — `AttributeError: 'PredictionService' object has no attribute 'warm_up'`

- [ ] **Step 3: Add warm_up() to PredictionService**

In `alphavedha/services/prediction_service.py`, add:

```python
async def warm_up(self) -> None:
    """Run a single prediction to warm up the full inference path."""
    try:
        tiers = self._config.default_tiers
        if not tiers:
            logger.warning("warmup_no_tiers")
            return
        symbols = self._get_symbols(tiers[0])
        if not symbols:
            logger.warning("warmup_no_symbols")
            return
        await self.predict_single(symbols[0])
        logger.info("model_warmup_complete", symbol=symbols[0])
    except Exception as e:
        logger.warning("model_warmup_failed", error=str(e))
```

- [ ] **Step 4: Replace sequential predict_batch with concurrent version**

In `alphavedha/services/prediction_service.py`, replace `predict_batch()`:

```python
async def predict_batch(self, symbols: list[str]) -> list[StockPrediction]:
    semaphore = asyncio.Semaphore(10)

    async def _predict_one(symbol: str) -> StockPrediction:
        async with semaphore:
            return await self.predict_single(symbol)

    return list(await asyncio.gather(*[_predict_one(s) for s in symbols]))
```

Add `import asyncio` at the top if not already present.

Apply the same pattern to `scan_tier()`:

```python
async def scan_tier(self, tier: str) -> list[StockPrediction]:
    symbols = self._get_symbols(tier)
    return await self.predict_batch(symbols)
```

- [ ] **Step 5: Add warm_up call to API lifespan**

In `alphavedha/api/app.py`, inside the `lifespan` function, after `set_service(service)`:

```python
if not demo:
    await service.warm_up()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/unit/services/test_prediction_service.py -v`
Expected: All tests PASS

- [ ] **Step 7: Lint check**

Run: `ruff check alphavedha/services/prediction_service.py alphavedha/api/app.py`
Expected: No errors

- [ ] **Step 8: Commit**

```bash
git add alphavedha/services/prediction_service.py alphavedha/api/app.py tests/unit/services/test_prediction_service.py
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat(d5): add model warm-up and async batch prediction"
```

---

### Task 4: Automated Model Comparison

**Files:**
- Modify: `alphavedha/monitoring/retrainer.py`
- Extend: `tests/unit/monitoring/test_retrainer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/monitoring/test_retrainer.py`:

```python
from alphavedha.monitoring.retrainer import ComparisonResult


class TestCompareModels:
    def test_compare_models_promote(self, manager: RetrainingManager) -> None:
        """Shadow clearly better on both accuracy and F1 -> promote."""
        manager.register_version(
            "xgboost",
            ModelVersion(
                version="v1.0.0",
                model_name="xgboost",
                status="active",
                trained_at="2024-01-01",
                metrics={"accuracy": 0.70, "f1": 0.68},
                artifact_path="models/artifacts/xgboost/v1",
            ),
        )
        manager.register_version(
            "xgboost",
            ModelVersion(
                version="v1.1.0",
                model_name="xgboost",
                status="shadow",
                trained_at="2024-02-01",
                metrics={"accuracy": 0.75, "f1": 0.73},
                artifact_path="models/artifacts/xgboost/v2",
            ),
        )
        result = manager.compare_models("xgboost")
        assert isinstance(result, ComparisonResult)
        assert result.recommendation == "promote"
        assert result.metric_deltas["accuracy"] == pytest.approx(0.05)

    def test_compare_models_discard(self, manager: RetrainingManager) -> None:
        """Shadow clearly worse -> discard."""
        manager.register_version(
            "xgboost",
            ModelVersion(
                version="v1.0.0",
                model_name="xgboost",
                status="active",
                trained_at="2024-01-01",
                metrics={"accuracy": 0.75, "f1": 0.73},
                artifact_path="models/artifacts/xgboost/v1",
            ),
        )
        manager.register_version(
            "xgboost",
            ModelVersion(
                version="v1.1.0",
                model_name="xgboost",
                status="shadow",
                trained_at="2024-02-01",
                metrics={"accuracy": 0.70, "f1": 0.68},
                artifact_path="models/artifacts/xgboost/v2",
            ),
        )
        result = manager.compare_models("xgboost")
        assert result.recommendation == "discard"

    def test_compare_models_extend(self, manager: RetrainingManager) -> None:
        """Marginal difference -> extend_shadow."""
        manager.register_version(
            "xgboost",
            ModelVersion(
                version="v1.0.0",
                model_name="xgboost",
                status="active",
                trained_at="2024-01-01",
                metrics={"accuracy": 0.75, "f1": 0.73},
                artifact_path="models/artifacts/xgboost/v1",
            ),
        )
        manager.register_version(
            "xgboost",
            ModelVersion(
                version="v1.1.0",
                model_name="xgboost",
                status="shadow",
                trained_at="2024-02-01",
                metrics={"accuracy": 0.755, "f1": 0.735},
                artifact_path="models/artifacts/xgboost/v2",
            ),
        )
        result = manager.compare_models("xgboost")
        assert result.recommendation == "extend_shadow"

    def test_compare_models_no_shadow(self, manager: RetrainingManager) -> None:
        """No shadow version -> ValueError."""
        manager.register_version(
            "xgboost",
            ModelVersion(
                version="v1.0.0",
                model_name="xgboost",
                status="active",
                trained_at="2024-01-01",
                metrics={"accuracy": 0.75, "f1": 0.73},
                artifact_path="models/artifacts/xgboost/v1",
            ),
        )
        with pytest.raises(ValueError, match="No shadow"):
            manager.compare_models("xgboost")

    def test_compare_models_no_active(self, manager: RetrainingManager) -> None:
        """No active version -> ValueError."""
        manager.register_version(
            "xgboost",
            ModelVersion(
                version="v1.1.0",
                model_name="xgboost",
                status="shadow",
                trained_at="2024-02-01",
                metrics={"accuracy": 0.75, "f1": 0.73},
                artifact_path="models/artifacts/xgboost/v2",
            ),
        )
        with pytest.raises(ValueError, match="No active"):
            manager.compare_models("xgboost")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/monitoring/test_retrainer.py::TestCompareModels -v`
Expected: FAIL — `ImportError: cannot import name 'ComparisonResult'`

- [ ] **Step 3: Implement ComparisonResult and compare_models**

In `alphavedha/monitoring/retrainer.py`, add the dataclass:

```python
@dataclass
class ComparisonResult:
    active_version: str
    shadow_version: str
    active_metrics: dict[str, float]
    shadow_metrics: dict[str, float]
    metric_deltas: dict[str, float]
    recommendation: str  # "promote" | "discard" | "extend_shadow"
    reason: str
```

Add to `RetrainingManager`:

```python
def compare_models(self, model_name: str) -> ComparisonResult:
    versions = self._versions.get(model_name, [])
    active = next((v for v in versions if v.status == "active"), None)
    shadow = next((v for v in versions if v.status == "shadow"), None)

    if active is None:
        raise ValueError(f"No active version for {model_name}")
    if shadow is None:
        raise ValueError(f"No shadow version for {model_name}")

    deltas = {
        k: shadow.metrics.get(k, 0.0) - active.metrics.get(k, 0.0)
        for k in set(active.metrics) | set(shadow.metrics)
    }

    acc_delta = deltas.get("accuracy", 0.0)
    f1_delta = deltas.get("f1", 0.0)

    if acc_delta >= 0.01 and f1_delta >= 0.01:
        recommendation = "promote"
        reason = f"Shadow beats active: accuracy +{acc_delta:.3f}, f1 +{f1_delta:.3f}"
    elif acc_delta <= -0.02 or f1_delta <= -0.02:
        recommendation = "discard"
        reason = f"Shadow worse: accuracy {acc_delta:+.3f}, f1 {f1_delta:+.3f}"
    else:
        recommendation = "extend_shadow"
        reason = f"Marginal: accuracy {acc_delta:+.3f}, f1 {f1_delta:+.3f}"

    return ComparisonResult(
        active_version=active.version,
        shadow_version=shadow.version,
        active_metrics=active.metrics,
        shadow_metrics=shadow.metrics,
        metric_deltas=deltas,
        recommendation=recommendation,
        reason=reason,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/monitoring/test_retrainer.py -v`
Expected: All tests PASS (existing + 5 new)

- [ ] **Step 5: Commit**

```bash
git add alphavedha/monitoring/retrainer.py tests/unit/monitoring/test_retrainer.py
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat(d5): add automated model comparison with promote/discard/extend logic"
```

---

### Task 5: RL Pipeline Integration + Walk-Forward Validation

**Files:**
- Modify: `alphavedha/training/pipeline.py`
- Modify: `alphavedha/training/rl_pipeline.py`
- Extend: `tests/unit/training/test_pipeline.py`
- Extend: `tests/unit/training/test_rl_pipeline.py`

- [ ] **Step 1: Write failing test for walk_forward_rl**

Add to `tests/unit/training/test_rl_pipeline.py`:

```python
from alphavedha.training.rl_pipeline import walk_forward_rl, WalkForwardResult


class TestWalkForwardRL:
    def test_walk_forward_basic(self) -> None:
        """walk_forward_rl with 2 windows on synthetic data should return results."""
        feature_df, price_df = _make_training_data(n_days=200, n_symbols=3)
        symbols = [c for c in price_df.columns if c != "date"]

        with patch("alphavedha.training.rl_pipeline.PPOAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.select_action.return_value = (0, 0.5)
            mock_agent.evaluate_actions.return_value = (
                torch.tensor([0.5]),
                torch.tensor([0.5]),
                torch.tensor([0.1]),
            )
            MockAgent.return_value = mock_agent

            result = walk_forward_rl(
                feature_df=feature_df,
                price_df=price_df,
                symbols=symbols,
                n_windows=2,
                train_frac=0.6,
                n_episodes=2,
            )

        assert isinstance(result, WalkForwardResult)
        assert result.n_windows == 2
        assert len(result.window_results) == 2

    def test_walk_forward_metrics(self) -> None:
        """Averaged metrics should be computed correctly from window results."""
        feature_df, price_df = _make_training_data(n_days=200, n_symbols=3)
        symbols = [c for c in price_df.columns if c != "date"]

        with patch("alphavedha.training.rl_pipeline.PPOAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.select_action.return_value = (0, 0.5)
            mock_agent.evaluate_actions.return_value = (
                torch.tensor([0.5]),
                torch.tensor([0.5]),
                torch.tensor([0.1]),
            )
            MockAgent.return_value = mock_agent

            result = walk_forward_rl(
                feature_df=feature_df,
                price_df=price_df,
                symbols=symbols,
                n_windows=2,
                train_frac=0.6,
                n_episodes=2,
            )

        # avg_sharpe should be mean of window sharpe ratios
        expected_avg = sum(w.sharpe_ratio for w in result.window_results) / len(
            result.window_results
        )
        assert result.avg_sharpe == pytest.approx(expected_avg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/training/test_rl_pipeline.py::TestWalkForwardRL -v`
Expected: FAIL — `ImportError: cannot import name 'walk_forward_rl'`

- [ ] **Step 3: Implement WalkForwardResult and walk_forward_rl**

In `alphavedha/training/rl_pipeline.py`, add:

```python
@dataclass
class WalkForwardResult:
    n_windows: int
    window_results: list[RLTrainingResult]
    avg_sharpe: float
    avg_return: float
    avg_max_dd: float


def walk_forward_rl(
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    symbols: list[str],
    n_windows: int = 3,
    train_frac: float = 0.7,
    n_episodes: int = 50,
) -> WalkForwardResult:
    n_rows = len(feature_df)
    window_size = n_rows // n_windows
    window_results: list[RLTrainingResult] = []

    for i in range(n_windows):
        train_end = window_size * (i + 1)
        train_start = 0
        val_start = train_end
        val_end = min(train_end + int(window_size * (1 - train_frac) / train_frac), n_rows)

        if val_end <= val_start:
            continue

        train_features = feature_df.iloc[train_start:train_end]
        train_prices = price_df.iloc[train_start:train_end]
        val_features = feature_df.iloc[val_start:val_end]
        val_prices = price_df.iloc[val_start:val_end]

        result = train_rl_agent(
            feature_df=train_features,
            price_df=train_prices,
            symbols=symbols,
            val_feature_df=val_features,
            val_price_df=val_prices,
            n_episodes=n_episodes,
        )
        window_results.append(result)

    avg_sharpe = sum(r.sharpe_ratio for r in window_results) / len(window_results) if window_results else 0.0
    avg_return = sum(r.total_return for r in window_results) / len(window_results) if window_results else 0.0
    avg_max_dd = sum(r.max_drawdown for r in window_results) / len(window_results) if window_results else 0.0

    return WalkForwardResult(
        n_windows=len(window_results),
        window_results=window_results,
        avg_sharpe=avg_sharpe,
        avg_return=avg_return,
        avg_max_dd=avg_max_dd,
    )
```

- [ ] **Step 4: Run walk-forward tests**

Run: `python -m pytest tests/unit/training/test_rl_pipeline.py -v`
Expected: All tests PASS

- [ ] **Step 5: Write failing test for RL integration in train_all**

Add to `tests/unit/training/test_pipeline.py`:

```python
class TestRLIntegration:
    def test_train_all_includes_rl(
        self, tmp_path: Path, tier_data: TierData
    ) -> None:
        """train_all should include rl_agent in results."""
        with (
            patch("alphavedha.training.pipeline.ARTIFACTS_DIR", tmp_path),
            patch("alphavedha.training.pipeline._train_xgboost_on_data") as mock_xgb,
            patch("alphavedha.training.pipeline._select_top_features") as mock_feat,
            patch("alphavedha.training.pipeline._train_lstm_on_data") as mock_lstm,
            patch("alphavedha.training.pipeline._train_tft_on_data") as mock_tft,
            patch("alphavedha.training.pipeline._train_regime_on_data") as mock_regime,
            patch("alphavedha.training.pipeline._train_ensemble_on_data") as mock_ens,
            patch("alphavedha.training.pipeline._train_meta_labeling_on_data") as mock_meta,
            patch("alphavedha.training.pipeline._train_conformal_on_data") as mock_conf,
            patch("alphavedha.training.pipeline._train_rl_on_data") as mock_rl,
        ):
            for m in [mock_xgb, mock_lstm, mock_tft, mock_regime, mock_ens, mock_meta, mock_conf]:
                r = TrainingPipelineResult(model_name="test")
                r.metrics = {"accuracy": 0.70}
                m.return_value = r
            mock_feat.return_value = list(range(30))

            rl_result = TrainingPipelineResult(model_name="rl_agent")
            rl_result.extra_metrics = {"val_return": 0.15, "val_sharpe": 1.2}
            mock_rl.return_value = rl_result

            results = train_all(tier_data)

        assert "rl_agent" in results
```

- [ ] **Step 6: Add _train_rl_on_data helper and Step 10 to train_all**

In `alphavedha/training/pipeline.py`, add the bridge function:

```python
def _train_rl_on_data(data: TierData) -> TrainingPipelineResult:
    from alphavedha.training.rl_pipeline import train_rl_agent

    result = TrainingPipelineResult(model_name="rl_agent")
    try:
        price_dfs = []
        for symbol, ohlcv in data.ohlcv_by_symbol.items():
            price_dfs.append(ohlcv[["close"]].rename(columns={"close": symbol}))

        if not price_dfs:
            result.errors["rl_agent"] = "No price data available"
            return result

        import pandas as pd
        price_df = pd.concat(price_dfs, axis=1).dropna()
        symbols = list(data.ohlcv_by_symbol.keys())

        rl_result = train_rl_agent(
            feature_df=data.X_train,
            price_df=price_df.iloc[: len(data.X_train)],
            symbols=symbols,
            val_feature_df=data.X_val,
            val_price_df=price_df.iloc[len(data.X_train) : len(data.X_train) + len(data.X_val)],
            n_episodes=50,
        )

        result.extra_metrics = {
            "val_return": rl_result.total_return,
            "val_sharpe": rl_result.sharpe_ratio,
            "val_max_dd": rl_result.max_drawdown,
        }
        if rl_result.artifact_path:
            result.artifact_path = Path(rl_result.artifact_path)
    except Exception as e:
        result.errors["rl_agent"] = str(e)
        logger.error("train_rl_failed", error=str(e))

    return result
```

In `train_all()`, add Step 10 after conformal (Step 9):

```python
# Step 10: RL Agent
logger.info("train_all_step", step="rl_agent")
rl_result = _train_rl_on_data(data)
_log_experiment(tracker, rl_result, data)
results["rl_agent"] = rl_result
```

- [ ] **Step 7: Run all pipeline tests**

Run: `python -m pytest tests/unit/training/test_pipeline.py tests/unit/training/test_rl_pipeline.py -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add alphavedha/training/pipeline.py alphavedha/training/rl_pipeline.py tests/unit/training/test_pipeline.py tests/unit/training/test_rl_pipeline.py
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat(d5): add RL pipeline integration and walk-forward validation"
```

---

### Task 6: CLI Commands — Experiment + Model Comparison

**Files:**
- Modify: `alphavedha/cli/main.py`
- Create: `tests/unit/cli/test_experiment_cli.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/cli/test_experiment_cli.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from alphavedha.cli.main import app

runner = CliRunner()


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def _create_run_file(runs_dir: Path, run_id: str, model: str, accuracy: float, f1: float) -> None:
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
    def test_list_runs(self, runs_dir: Path) -> None:
        _create_run_file(runs_dir, "20240101_120000_xgboost", "xgboost", 0.75, 0.73)
        _create_run_file(runs_dir, "20240102_120000_lstm", "lstm", 0.72, 0.70)

        with patch(
            "alphavedha.cli.main.ARTIFACTS_DIR",
            runs_dir.parent,
        ):
            result = runner.invoke(app, ["experiment", "list"])
        assert result.exit_code == 0
        assert "xgboost" in result.output
        assert "lstm" in result.output

    def test_list_runs_filter(self, runs_dir: Path) -> None:
        _create_run_file(runs_dir, "20240101_120000_xgboost", "xgboost", 0.75, 0.73)
        _create_run_file(runs_dir, "20240102_120000_lstm", "lstm", 0.72, 0.70)

        with patch(
            "alphavedha.cli.main.ARTIFACTS_DIR",
            runs_dir.parent,
        ):
            result = runner.invoke(app, ["experiment", "list", "--model", "lstm"])
        assert result.exit_code == 0
        assert "lstm" in result.output


class TestExperimentCompare:
    def test_compare_runs(self, runs_dir: Path) -> None:
        _create_run_file(runs_dir, "20240101_120000_xgboost", "xgboost", 0.75, 0.73)
        _create_run_file(runs_dir, "20240102_120000_xgboost", "xgboost", 0.78, 0.76)

        with patch(
            "alphavedha.cli.main.ARTIFACTS_DIR",
            runs_dir.parent,
        ):
            result = runner.invoke(
                app,
                ["experiment", "compare", "20240101_120000_xgboost", "20240102_120000_xgboost"],
            )
        assert result.exit_code == 0
        assert "accuracy" in result.output


class TestModelCompare:
    def test_model_compare_no_versions(self) -> None:
        result = runner.invoke(app, ["model", "compare", "--model-name", "xgboost"])
        assert result.exit_code == 0 or result.exit_code == 1
        # Should output error about missing versions
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/cli/test_experiment_cli.py -v`
Expected: FAIL — `No such command 'experiment'`

- [ ] **Step 3: Add experiment and model CLI commands**

In `alphavedha/cli/main.py`, add:

```python
from alphavedha.monitoring.experiment_tracker import ExperimentTracker

ARTIFACTS_DIR = Path("models/artifacts")

experiment_app = typer.Typer(help="Experiment tracking commands.")
app.add_typer(experiment_app, name="experiment")

model_app = typer.Typer(help="Model management commands.")
app.add_typer(model_app, name="model")


@experiment_app.command("list")
def experiment_list(
    model: str | None = typer.Option(None, "--model", help="Filter by model name"),
    limit: int = typer.Option(20, "--limit", help="Max runs to show"),
) -> None:
    """List recent experiment runs."""
    tracker = ExperimentTracker(base_dir=ARTIFACTS_DIR)
    runs = tracker.list_runs(model_name=model, limit=limit)

    if not runs:
        console.print("[yellow]No experiment runs found.[/yellow]")
        return

    table = Table(title="Experiment Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Val Accuracy", justify="right")
    table.add_column("Val F1", justify="right")
    table.add_column("Duration (s)", justify="right")
    table.add_column("Date")

    for run in runs:
        table.add_row(
            run.run_id,
            run.model_name,
            f"{run.val_metrics.get('accuracy', 0):.4f}",
            f"{run.val_metrics.get('f1', 0):.4f}",
            f"{run.duration_seconds:.1f}",
            run.started_at[:10],
        )

    console.print(table)


@experiment_app.command("compare")
def experiment_compare(
    run_a: str = typer.Argument(help="First run ID"),
    run_b: str = typer.Argument(help="Second run ID"),
) -> None:
    """Compare two experiment runs side by side."""
    tracker = ExperimentTracker(base_dir=ARTIFACTS_DIR)
    try:
        comparison = tracker.compare_runs(run_a, run_b)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    table = Table(title=f"Comparison: {run_a} vs {run_b}")
    table.add_column("Metric", style="cyan")
    table.add_column("Run A", justify="right")
    table.add_column("Run B", justify="right")
    table.add_column("Delta", justify="right")

    for metric, values in comparison.items():
        delta = values["delta"]
        delta_style = "green" if delta > 0 else "red" if delta < 0 else "white"
        table.add_row(
            metric,
            f"{values['a']:.4f}",
            f"{values['b']:.4f}",
            f"[{delta_style}]{delta:+.4f}[/{delta_style}]",
        )

    console.print(table)


@model_app.command("compare")
def model_compare(
    model_name: str = typer.Option("xgboost", "--model-name", help="Model to compare"),
) -> None:
    """Compare active vs shadow model versions."""
    from alphavedha.monitoring.retrainer import RetrainingManager

    manager = RetrainingManager(base_dir=ARTIFACTS_DIR)
    try:
        result = manager.compare_models(model_name)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    table = Table(title=f"Model Comparison: {model_name}")
    table.add_column("", style="bold")
    table.add_column("Active", justify="right")
    table.add_column("Shadow", justify="right")
    table.add_column("Delta", justify="right")

    table.add_row("Version", result.active_version, result.shadow_version, "")

    for metric in sorted(result.metric_deltas.keys()):
        delta = result.metric_deltas[metric]
        delta_style = "green" if delta > 0 else "red" if delta < 0 else "white"
        table.add_row(
            metric,
            f"{result.active_metrics.get(metric, 0):.4f}",
            f"{result.shadow_metrics.get(metric, 0):.4f}",
            f"[{delta_style}]{delta:+.4f}[/{delta_style}]",
        )

    rec_style = {"promote": "green", "discard": "red", "extend_shadow": "yellow"}
    style = rec_style.get(result.recommendation, "white")
    console.print(f"\n[{style}]Recommendation: {result.recommendation}[/{style}]")
    console.print(f"Reason: {result.reason}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/cli/test_experiment_cli.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/unit/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 6: Lint check**

Run: `ruff check alphavedha/cli/main.py`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add alphavedha/cli/main.py tests/unit/cli/test_experiment_cli.py
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat(d5): add experiment list/compare and model compare CLI commands"
```

---

## Post-Implementation

After all 6 tasks are complete:

1. Run full test suite: `python -m pytest tests/unit/ -v --cov=alphavedha --cov-report=term-missing`
2. Run linter: `ruff check alphavedha/ tests/`
3. Update `docs/PROGRESS.md` with D5 status and new test counts
