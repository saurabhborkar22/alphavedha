# D5: ML Operations Improvements — Design Spec

## Goal

Add lightweight experiment tracking, model serving improvements (warm-up, batch optimization), automated model comparison for shadow promotion, and RL pipeline integration into the training workflow. All scoped for a solo developer with $0–15/month budget — no external services, no traffic splitting.

## Architecture

Four independent modules that integrate with existing code through small, targeted changes:

1. **Experiment Tracker** — JSON-based run logging alongside model artifacts
2. **Model Serving** — eager warm-up at startup + async batch prediction
3. **Model Comparison** — automated evaluation of shadow vs active model on validation data
4. **RL Pipeline Integration** — wire existing PPO training into `train_all()`

## D5.1: Experiment Tracker

### New file: `alphavedha/monitoring/experiment_tracker.py`

**Core class:** `ExperimentTracker`

```python
@dataclass
class RunRecord:
    run_id: str              # "20260522_143000_xgboost"
    model_name: str
    started_at: str          # ISO timestamp
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
    extra: dict[str, Any]    # model-specific (regime metrics, etc.)
```

**Methods:**
- `log_run(model_name, hyperparams, train_metrics, val_metrics, data_range, n_train_rows, n_val_rows, n_symbols, feature_count, artifact_path, duration_seconds, extra=None) -> RunRecord` — saves JSON to `models/artifacts/runs/{run_id}.json`
- `list_runs(model_name=None, limit=20) -> list[RunRecord]` — list recent runs, optionally filtered
- `get_run(run_id) -> RunRecord | None` — load a specific run
- `compare_runs(run_id_a, run_id_b) -> dict[str, dict[str, float]]` — returns `{"metric_name": {"a": val, "b": val, "delta": val}}` per metric

**Storage:**
```
models/artifacts/
  runs/
    20260522_143000_xgboost.json
    20260522_150000_lstm.json
```

Flat directory, filename = `{run_id}.json`. The `run_id` is `{YYYYMMDD}_{HHMMSS}_{model_name}`.

### Integration with training pipeline

Add `ExperimentTracker.log_run()` calls at the end of each model's training step in `training/pipeline.py`. Each `_train_*_on_data()` helper already returns metrics — the tracker call is a one-liner after save.

The tracker instance is created once at the top of `train_all()` and passed through, or each helper creates its own (stateless — just writes files).

### CLI commands

Add to `cli/main.py`:
- `alphavedha experiment list [--model MODEL] [--limit N]` — Rich table of recent runs (run_id, model, val_accuracy, val_f1, duration, date)
- `alphavedha experiment compare RUN_A RUN_B` — side-by-side Rich table showing metric diffs with color (green = improved, red = degraded)

### Test file: `tests/unit/monitoring/test_experiment_tracker.py`

Tests:
- `test_log_run_creates_json` — log a run, verify JSON file exists with correct fields
- `test_list_runs_returns_recent` — log 5 runs, list with limit=3, verify order
- `test_list_runs_filter_by_model` — log runs for 2 models, filter returns only one
- `test_get_run_exists` — log and retrieve by run_id
- `test_get_run_not_found` — returns None for missing run_id
- `test_compare_runs` — log 2 runs with different metrics, verify diffs computed correctly
- `test_run_id_format` — verify run_id matches expected pattern

---

## D5.2: Model Serving Improvements

### D5.2.1: Model Warm-Up

**Modify:** `alphavedha/api/app.py` (lifespan function)

Currently, `PredictionService.__init__` calls `registry.get_prediction_engine()` which loads models. This already happens at startup. However, the first actual prediction is slow because:
1. Feature computation paths aren't exercised
2. NumPy/torch lazy initialization hasn't happened

Add a warm-up step in the lifespan after creating the service:

```python
# In lifespan, after set_service(service):
if not demo:
    await service.warm_up()
```

**Add method to `PredictionService`:**

```python
async def warm_up(self) -> None:
    """Run a single prediction to warm up the full inference path."""
    try:
        warmup_symbol = self._get_symbols(self._config.default_tier)[0]
        await self.predict_single(warmup_symbol)
        logger.info("model_warmup_complete", symbol=warmup_symbol)
    except Exception as e:
        logger.warning("model_warmup_failed", error=str(e))
```

Warm-up at the `PredictionService` level exercises the full path (feature loading, model inference, scoring, caching) with a single call. For demo mode, skip warm-up (demo models are already instant).

### D5.2.2: Batch Prediction Optimization

**Modify:** `alphavedha/services/prediction_service.py`

Current `scan_tier()` and `predict_batch()` run sequentially:
```python
for symbol in symbols:
    pred = await self.predict_single(symbol)
```

Change to concurrent execution with bounded parallelism:

```python
async def predict_batch(self, symbols: list[str]) -> list[StockPrediction]:
    semaphore = asyncio.Semaphore(10)
    async def _predict_one(symbol: str) -> StockPrediction:
        async with semaphore:
            return await self.predict_single(symbol)
    return await asyncio.gather(*[_predict_one(s) for s in symbols])
```

Semaphore limits concurrent predictions to 10 to avoid overwhelming DB/Redis connections.

Update `scan_tier()` to use the same pattern.

### Tests

Add to existing `tests/unit/services/` or create `tests/unit/services/test_prediction_service.py`:
- `test_warmup_demo_skipped` — verify warm_up doesn't crash in demo mode
- `test_warmup_real_runs_prediction` — mock predict_single, verify it's called once
- `test_predict_batch_concurrent` — verify all symbols get predictions (order preserved)
- `test_scan_tier_concurrent` — verify scan uses concurrent path

---

## D5.3: Automated Model Comparison

### Modify: `alphavedha/monitoring/retrainer.py`

Add a `compare_models()` method to `RetrainingManager`:

```python
@dataclass
class ComparisonResult:
    active_version: str
    shadow_version: str
    active_metrics: dict[str, float]   # accuracy, f1, sharpe
    shadow_metrics: dict[str, float]
    metric_deltas: dict[str, float]    # shadow - active
    recommendation: str                 # "promote" | "discard" | "extend_shadow"
    reason: str
```

**Logic:**
- Load active and shadow model artifacts
- Run both on the same validation dataset (most recent data not seen during training)
- Compare key metrics: accuracy, F1 (weighted), Sharpe ratio from backtest
- Recommendation rules:
  - `promote`: shadow beats active on accuracy AND F1 by >= 1% each
  - `discard`: shadow is worse on accuracy OR F1 by >= 2%
  - `extend_shadow`: differences are marginal (within 1%), keep collecting data

### CLI command

Add to `cli/main.py`:
- `alphavedha model compare` — runs comparison if shadow exists, prints Rich table with recommendation

### Tests: `tests/unit/monitoring/test_retrainer.py` (extend existing)

- `test_compare_models_promote` — shadow clearly better, recommends promote
- `test_compare_models_discard` — shadow clearly worse, recommends discard
- `test_compare_models_extend` — marginal difference, recommends extend
- `test_compare_models_no_shadow` — no shadow version exists, returns error

---

## D5.4: RL Pipeline Integration

### Modify: `alphavedha/training/pipeline.py`

Add Step 10 to `train_all()` after conformal (Step 9):

```python
# Step 10: Train RL Agent (optional, after all other models)
logger.info("train_all_step", step="rl_agent")
rl_result = TrainingPipelineResult(model_name="rl_agent")
try:
    rl_training_result = _train_rl_on_data(data)
    rl_result.extra_metrics = {
        "val_return": rl_training_result.total_return,
        "val_sharpe": rl_training_result.sharpe_ratio,
        "val_max_dd": rl_training_result.max_drawdown,
    }
    rl_result.artifact_path = Path(rl_training_result.artifact_path) if rl_training_result.artifact_path else None
except Exception as e:
    rl_result.errors["rl_agent"] = str(e)
    logger.error("train_all_rl_failed", error=str(e))
results["rl_agent"] = rl_result
```

### New helper: `_train_rl_on_data(data: TierData) -> RLTrainingResult`

Bridges `TierData` to the existing `train_rl_agent()` function:
- Extracts price data from `data.ohlcv_by_symbol` into a multi-column price DataFrame
- Uses `data.X_val` features and corresponding price data for validation
- Gets regime labels from the trained regime detector (if available)
- Calls `train_rl_agent()` with appropriate parameters

### Walk-Forward Validation

Add `walk_forward_rl()` function to `training/rl_pipeline.py`:

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
```

Splits data into `n_windows` expanding windows (each window trains on all prior data, validates on the next chunk). Returns aggregate metrics across windows.

### Tests: extend `tests/unit/training/test_pipeline.py`

- `test_train_rl_on_data` — mock train_rl_agent, verify bridge function works
- `test_train_all_includes_rl` — verify rl_agent appears in train_all results

### Tests: `tests/unit/training/test_rl_pipeline.py` (extend existing)

- `test_walk_forward_rl_basic` — 2 windows on synthetic data
- `test_walk_forward_rl_metrics` — verify avg metrics computed correctly

---

## File Summary

| Action | File | Purpose |
|--------|------|---------|
| Create | `alphavedha/monitoring/experiment_tracker.py` | Run logging and comparison |
| Modify | `alphavedha/training/pipeline.py` | Log runs + RL integration |
| Modify | `alphavedha/services/prediction_service.py` | Warm-up + batch concurrency |
| Modify | `alphavedha/api/app.py` | Call warm_up in lifespan |
| Modify | `alphavedha/monitoring/retrainer.py` | Add compare_models() |
| Modify | `alphavedha/training/rl_pipeline.py` | Add walk_forward_rl() |
| Modify | `alphavedha/cli/main.py` | Add experiment + model CLI commands |
| Create | `tests/unit/monitoring/test_experiment_tracker.py` | Tracker tests |
| Extend | `tests/unit/monitoring/test_retrainer.py` | Comparison tests |
| Extend | `tests/unit/services/test_prediction_service.py` | Serving tests |
| Extend | `tests/unit/training/test_rl_pipeline.py` | Walk-forward tests |

## Out of Scope

- MLflow / Weights & Biases — overkill for solo use
- A/B traffic splitting — not needed for single user
- Live traffic routing to shadow models — skip
- RL as 5th ensemble input — skip for now
- Redis inference caching changes — existing `PredictionCache` already handles this correctly with `predict:{symbol}:{model_version}` keys and market-hours TTL

## Dependencies

No new dependencies. Everything uses existing libraries (structlog, dataclasses, asyncio, json, pathlib).
