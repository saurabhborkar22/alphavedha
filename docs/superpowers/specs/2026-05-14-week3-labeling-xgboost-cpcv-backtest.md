# Week 3 Design Spec: Labeling + XGBoost + CPCV + Backtesting

**Date:** 2026-05-14
**Status:** Approved
**Author:** Saurabh Borkar + Claude Opus 4.6

---

## 1. Overview

Week 3 builds the core ML pipeline: label generation from price data, the first base model (XGBoost), rigorous cross-validation (CPCV), and backtesting with real Indian market costs. This connects the feature engineering layer (Week 2) to trainable, validatable models.

**Modules delivered:**
- `labels/` — Triple barrier labeling + sample weighting
- `models/` — BaseModel ABC + XGBoost (classification + regression)
- `backtest/` — Indian cost model + CPCV validation + VectorBT engine

## 2. Dependencies

### Consumes from Week 1-2
- `data/providers/` — OHLCV data with DatetimeIndex
- `data/preprocessing/` — Corporate-action-adjusted, circuit-flagged data
- `features/pipeline.py` — `compute_all_features()` → 142-column DataFrame
- `config.py` — `LabelsConfig`, `ModelsConfig`, `ValidationConfig`, `BacktestConfig`
- `exceptions.py` — `ModelTrainingError`, `ValidationError`, `InsufficientDataError`

### New dependencies (pyproject.toml)
- `xgboost >= 2.0` — gradient-boosted trees
- `vectorbt >= 0.26` — backtesting engine
- `joblib >= 1.3` — model serialization
- `scikit-learn >= 1.4` — Ridge meta-learner, metrics, train/test utilities

## 3. Triple Barrier Labeling (`labels/triple_barrier.py`)

### Algorithm

For each trading day t in the OHLCV series:

1. Compute ATR(14) at time t using only data up to t (no look-ahead).
2. Skip if ATR / close < `min_atr_threshold` (default 0.5%) — too low volatility for meaningful labeling.
3. Set barriers:
   - Upper barrier = close[t] + `multiplier_up` x ATR[t] (default 2.0x)
   - Lower barrier = close[t] - `multiplier_down` x ATR[t] (default 1.5x)
   - Time barrier = t + `max_holding_period` trading days (default 15)
4. Walk forward from t+1 through the window:
   - If high[t+i] >= upper barrier before low[t+i] <= lower barrier → label = +1
   - If low[t+i] <= lower barrier first → label = -1
   - If neither touched by time barrier → label = 0
5. Record: return_pct = exit_price / entry_price - 1, days_to_hit, barrier_hit type.

### Asymmetric barriers

The upper multiplier (2.0) is larger than the lower (1.5), reflecting the asymmetry in trading: we require a bigger upside to call a buy signal than the downside for a sell signal. This is standard in de Prado's framework and builds in risk-awareness at the labeling stage.

### Interface

```python
@dataclass
class LabelResult:
    df: pd.DataFrame           # Columns: label, return_pct, barrier_hit, days_to_hit,
                                #          entry_price, exit_price, atr_at_entry
    symbol: str
    label_counts: dict[int, int]   # {1: N, -1: N, 0: N}
    skipped_low_atr: int
    avg_days_to_hit: float

def compute_triple_barrier_labels(
    ohlcv_df: pd.DataFrame,
    config: TripleBarrierConfig,
) -> LabelResult:
```

### Edge cases
- Last `max_holding_period` rows: cannot compute labels (insufficient forward data). Return NaN for these rows.
- Circuit-hit days: included in barrier walk (their prices are real, just constrained). The `circuit_hit` feature flag already captures this for the model.
- Same-bar touch: if both barriers are touched on the same day, the barrier closer to the open price is considered hit first. If ambiguous, use the lower barrier (conservative — favor the negative label).

### Config (from `configs/default.yaml`)
```yaml
labels:
  triple_barrier:
    multiplier_up: 2.0
    multiplier_down: 1.5
    max_holding_period: 15
    min_atr_threshold: 0.005
    atr_period: 14
```

## 4. Sample Weights (`labels/sample_weights.py`)

Triple barrier labels overlap in time (15-day windows), creating dependency between samples. Sample weights address this.

### Uniqueness weighting

For each sample t, count how many other active labels overlap with its barrier window [t, t + days_to_hit]. Weight = 1 / overlap_count. This reduces the contribution of crowded periods where many overlapping labels carry redundant information.

Implementation: build a concurrency array where concurrency[t] = number of active barrier windows at time t. Each sample's uniqueness weight = mean(1/concurrency) over its active window.

### Recency weighting

Exponential decay: weight[t] = exp(-ln(2) * (T - t) / halflife), where T = last timestamp, halflife = 252 trading days (1 year). Recent data gets more weight because market dynamics evolve.

### Combined weight

final_weight = uniqueness_weight * recency_weight, then normalized so weights sum to N (number of samples). This preserves the effective sample size while adjusting relative importance.

### Interface

```python
def compute_sample_weights(
    labels_df: pd.DataFrame,
    config: SampleWeightsConfig,
) -> pd.Series:
```

Returns a Series aligned to the labels DataFrame index.

## 5. BaseModel ABC (`models/base.py`)

All ML models (XGBoost, LSTM, TFT) implement this interface. ABC chosen over Protocol because models share substantial lifecycle behavior (serialization, versioning, metrics logging).

### Interface

```python
class BaseModel(ABC):
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self._name = name
        self._config = config
        self._version = "0.0.0"
        self._is_fitted = False
        self._train_metrics: dict[str, float] = {}
        self._feature_names: list[str] = []

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def is_fitted(self) -> bool: ...

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> TrainResult: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> PredictionResult: ...

    @abstractmethod
    def get_feature_importance(self) -> pd.Series | None: ...

    # Shared implementations (not abstract)
    def save(self, directory: Path) -> ModelArtifact: ...
    @classmethod
    def load(cls, directory: Path) -> BaseModel: ...
    def get_metrics(self) -> dict[str, float]: ...
```

### Result types

```python
@dataclass
class TrainResult:
    train_metrics: dict[str, float]    # accuracy, f1, logloss, rmse, etc.
    val_metrics: dict[str, float]
    feature_importances: pd.Series | None
    training_time_seconds: float
    n_train_samples: int
    n_val_samples: int
    hyperparams: dict[str, Any]

@dataclass
class PredictionResult:
    direction: np.ndarray          # int array: +1, 0, -1
    magnitude: np.ndarray          # float array: predicted % return
    probabilities: np.ndarray | None   # shape (n, 3) for 3-class
    confidence: np.ndarray         # float array: max class probability

@dataclass
class ModelArtifact:
    path: Path
    name: str
    version: str
    created_at: str                # ISO 8601
    feature_names: list[str]
    metrics: dict[str, float]
    config: dict[str, Any]
```

### Serialization convention

Each model saves to a directory:
```
models/artifacts/{name}_v{version}/
├── model.joblib          # or model.safetensors for PyTorch
├── metadata.json         # ModelArtifact fields
└── feature_importance.csv
```

Version auto-increments on each `save()` call (patch bump: 0.0.0 → 0.0.1 → 0.0.2).

## 6. XGBoost Model (`models/xgboost_model.py`)

### Architecture

`XGBoostModel` wraps two internal XGBoost models:
1. **Classifier** (`XGBClassifier`) — predicts direction as 3-class: {+1, 0, -1}
2. **Regressor** (`XGBRegressor`) — predicts magnitude (return_pct)

Both share the same feature set and hyperparameters (except objective/eval_metric).

### fit()

1. Map labels: {-1, 0, +1} → {0, 1, 2} for XGBClassifier (multi:softprob).
2. Train classifier with `sample_weight`, early stopping on validation logloss (50 rounds).
3. Train regressor on `return_pct` with `sample_weight`, early stopping on validation RMSE.
4. Extract feature importances (gain-based) from classifier.
5. Return `TrainResult` with combined metrics from both models.

### predict()

1. Classifier → class probabilities (n, 3) → argmax → direction.
2. Regressor → magnitude estimate.
3. Confidence = max class probability.
4. Return `PredictionResult`.

### Hyperparameters (from config)
```yaml
models:
  xgboost:
    task: classification
    params:
      learning_rate: 0.05
      max_depth: 6
      n_estimators: 500
      subsample: 0.8
      colsample_bytree: 0.8
      reg_alpha: 0.1
      reg_lambda: 1.0
      eval_metric: logloss
      early_stopping_rounds: 50
```

### Serialization

Uses `joblib` for both classifier and regressor. Saved as:
```
models/artifacts/xgboost_v0.0.1/
├── classifier.joblib
├── regressor.joblib
├── metadata.json
└── feature_importance.csv
```

## 7. CPCV Validation (`backtest/cpcv.py`)

### Combinatorial Purged Cross-Validation

Standard time-series CV (walk-forward) wastes data — only the most recent fold tests on the latest data. CPCV generates all combinatorial paths through the data, maximizing test coverage while preventing leakage.

### Algorithm

1. Sort the dataset by time index.
2. Split into N=6 equal contiguous segments: S1, S2, ..., S6.
3. Generate all C(6,2)=15 combinations of k=2 test segments.
4. For each combination (e.g., test = {S2, S5}):
   a. Training set = {S1, S3, S4, S6} (remaining segments).
   b. **Purge:** Remove the last 20 trading days of each training segment that immediately precedes a test segment. (S1's last 20 days are purged because S2 is test.)
   c. **Embargo:** Remove the first 20 trading days of each training segment that immediately follows a test segment. (S3's first 20 days are embargoed because S2 is test; S6's first 20 days are embargoed because S5 is test.)
   d. Train model on purged+embargoed training set.
   e. Predict on test segments, compute metrics.

### Why purge + embargo

- **Purge:** Labels at the end of S1 have barrier windows that leak into S2 (the test set). Removing them prevents train-test contamination.
- **Embargo:** The model might learn patterns at the start of S3 that are autocorrelated with the end of S2 (test). Embargo adds a buffer.
- Combined 40-day gap (20+20) is conservative but necessary for 15-day barrier windows.

### Interface

```python
@dataclass
class PathResult:
    path_id: int
    test_segments: tuple[int, ...]
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    sharpe_ratio: float
    total_return: float
    n_test_samples: int
    confusion_matrix: np.ndarray

@dataclass
class CPCVResult:
    path_results: list[PathResult]
    n_paths: int
    median_sharpe: float
    worst_sharpe: float
    best_sharpe: float
    mean_accuracy: float
    std_accuracy: float
    passed: bool                   # meets acceptance criteria

def run_cpcv(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    model_factory: Callable[[], BaseModel],
    config: CPCVConfig,
    acceptance: AcceptanceConfig,
) -> CPCVResult:
```

`model_factory` is a callable that returns a fresh, untrained model instance. This ensures each path gets an independent model.

### Acceptance criteria (from config)
```yaml
validation:
  acceptance:
    min_median_sharpe: 0.8
    min_worst_sharpe: 0.3
```

### Sharpe calculation per path

Sharpe ratio = mean(daily_returns) / std(daily_returns) * sqrt(252), computed on the test-set predictions converted to a daily return series using the triple barrier exit prices.

## 8. Indian Market Cost Model (`backtest/costs.py`)

### Cost components

| Cost | Rate | Applied to |
|------|------|-----------|
| STT (delivery) | 0.1% | Buy + sell value |
| Brokerage | Rs 20 flat | Per order |
| Exchange transaction | 0.00345% | Turnover |
| GST | 18% | On (brokerage + exchange txn) |
| SEBI turnover | 0.0001% | Turnover |
| Stamp duty | 0.015% | Buy value only |
| Slippage (large cap) | 0.1% | Per side |
| Slippage (mid cap) | 0.3% | Per side |
| Slippage (small cap) | 0.5% | Per side |

### Interface

```python
@dataclass
class TradeCost:
    stt: float
    brokerage: float
    exchange_txn: float
    gst: float
    sebi_turnover: float
    stamp_duty: float
    slippage: float
    total: float

def compute_trade_cost(
    trade_value: float,
    side: str,                    # "buy" or "sell"
    market_cap_tier: str,         # "large", "mid", "small"
    config: CostsConfig,
    slippage_config: SlippageConfig,
) -> TradeCost:

def compute_round_trip_cost_pct(
    market_cap_tier: str,
    config: BacktestConfig,
) -> float:
```

`compute_round_trip_cost_pct` returns the total cost as a percentage of trade value for a buy+sell round trip. This is what VectorBT uses as its fee parameter.

## 9. VectorBT Backtesting Engine (`backtest/engine.py`)

### Strategy

The backtester converts model predictions into trading signals and runs them through VectorBT:

1. **Entry signal:** direction == +1 AND confidence >= min_confidence (default 0.55)
2. **Exit signal:** direction == -1, OR stop-loss hit, OR time barrier expiry (max_holding_period days)
3. **Position sizing:** equal weight for Week 3 (half-Kelly comes in Week 6 with risk module)
4. **Costs:** round-trip cost percentage from the cost model, applied as VectorBT fixed fee

### Interface

```python
@dataclass
class BacktestResult:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    alpha_vs_benchmark: float
    beta: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_holding_days: float
    avg_return_per_trade: float
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trade_log: pd.DataFrame
    benchmark_return: float

def run_backtest(
    predictions_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    config: BacktestConfig,
    market_cap_tier: str = "large",
    min_confidence: float = 0.55,
) -> BacktestResult:
```

`predictions_df` must have columns: `direction`, `magnitude`, `confidence`, aligned to the same DatetimeIndex as `ohlcv_df`.

### Benchmark

Alpha computed against Nifty 50 (^NSEI) buy-and-hold over the same period. The benchmark data is fetched via yfinance and cached.

## 10. Module Structure

```
labels/
├── __init__.py                  # Exports: LabelResult, compute_triple_barrier_labels,
|                                #          compute_sample_weights
├── triple_barrier.py            # Label generation
└── sample_weights.py            # Uniqueness + recency weighting

models/
├── __init__.py                  # Exports: BaseModel, XGBoostModel, TrainResult,
|                                #          PredictionResult, ModelArtifact
├── base.py                      # BaseModel ABC + dataclasses
└── xgboost_model.py             # XGBoostModel implementation

backtest/
├── __init__.py                  # Exports: run_backtest, run_cpcv, compute_trade_cost,
|                                #          BacktestResult, CPCVResult
├── costs.py                     # Indian market cost calculator
├── cpcv.py                      # Combinatorial Purged Cross-Validation
└── engine.py                    # VectorBT backtesting engine
```

## 11. Testing Strategy

### Unit tests — labels (`tests/unit/labels/`)

| Test | What it verifies |
|------|-----------------|
| `test_known_price_path` | Hand-computed labels on a synthetic price series match algorithm output |
| `test_upper_barrier_hit` | Price rising steadily → label = +1, correct days_to_hit |
| `test_lower_barrier_hit` | Price dropping → label = -1 |
| `test_time_barrier_expiry` | Flat price → label = 0, days_to_hit = max_holding_period |
| `test_asymmetric_barriers` | Upper barrier is farther than lower (2.0 vs 1.5 ATR) |
| `test_low_atr_skip` | Samples with ATR/close < 0.5% are skipped (NaN label) |
| `test_last_rows_nan` | Last max_holding_period rows have NaN labels |
| `test_no_lookahead` | Labels at time T use only OHLCV data from T+1 onward (not T-1 for barriers) |
| `test_same_bar_touch` | Both barriers touched same day → lower wins (conservative) |
| `test_sample_weight_uniqueness` | Non-overlapping labels → weight = 1.0 |
| `test_sample_weight_overlap` | Overlapping labels → reduced weights |
| `test_sample_weight_recency` | Most recent sample has highest weight |
| `test_weights_sum_to_n` | Combined weights sum to number of samples |

### Unit tests — models (`tests/unit/models/`)

| Test | What it verifies |
|------|-----------------|
| `test_xgboost_fit_predict` | Train on synthetic data, predict returns correct shapes |
| `test_direction_output` | Predictions are in {-1, 0, +1} |
| `test_magnitude_output` | Magnitude is a float array, same length as input |
| `test_probabilities_shape` | Probabilities are (n, 3), rows sum to 1.0 |
| `test_confidence_range` | Confidence values in [0, 1] |
| `test_feature_importance` | Returns a Series with feature names, sums to ~1.0 |
| `test_save_load_roundtrip` | Save model, load it, predictions are identical |
| `test_metadata_json` | metadata.json has all required fields |
| `test_unfitted_predict_raises` | Calling predict() before fit() raises ModelTrainingError |
| `test_config_hyperparams` | Hyperparameters from config are applied to XGBoost |
| `test_early_stopping` | Training stops early when validation loss plateaus |
| `test_sample_weight_used` | Weighted training produces different results than unweighted |

### Unit tests — backtest (`tests/unit/backtest/`)

| Test | What it verifies |
|------|-----------------|
| `test_cost_round_trip_large` | Round-trip cost for large cap matches hand calculation |
| `test_cost_round_trip_mid` | Mid cap includes higher slippage |
| `test_cost_components` | Each cost component (STT, brokerage, etc.) is correct individually |
| `test_stamp_duty_buy_only` | Stamp duty only on buy side |
| `test_cpcv_15_paths` | C(6,2)=15 paths generated |
| `test_cpcv_no_overlap` | No training sample appears in its own test set |
| `test_cpcv_purge` | Training data near test boundary is removed (20 days) |
| `test_cpcv_embargo` | Post-test training data has embargo gap (20 days) |
| `test_cpcv_acceptance_pass` | Good model passes acceptance criteria |
| `test_cpcv_acceptance_fail` | Weak model fails acceptance criteria |
| `test_backtest_metrics` | Sharpe, drawdown, alpha computed correctly on known equity curve |
| `test_backtest_no_trades` | Zero trades when model always predicts 0 |
| `test_backtest_costs_applied` | Returns with costs < returns without costs |

### Backtest validation tests (`tests/backtest/`)

| Test | What it verifies |
|------|-----------------|
| `test_no_lookahead` | Labels and features at time T don't use data after T |
| `test_costs_complete` | All 7 cost components are included in round-trip cost |
| `test_cpcv_leakage` | No data leakage across purge+embargo boundaries |

## 12. Test Fixtures (additions to `tests/conftest.py`)

```python
# 500-day OHLCV for labeling + CPCV (needs enough data for 6 segments + purge)
sample_ohlcv_500: pd.DataFrame

# Synthetic features (142 columns) aligned to sample_ohlcv_500
sample_features_500: pd.DataFrame

# Known price path for deterministic label testing
# [100, 102, 104, 106, 95, 93, 100, 100, 100, ...] — clear up, then down, then flat
sample_known_path: pd.DataFrame

# Pre-computed labels for the known path (hand-verified)
expected_labels_known_path: pd.Series
```

## 13. Error Handling

| Error | Exception | Behavior |
|-------|-----------|----------|
| Insufficient data for labeling (< 50 rows) | `InsufficientDataError` | Raised before computation |
| All samples filtered by ATR threshold | `InsufficientDataError` | Raised with count of filtered samples |
| XGBoost training fails | `ModelTrainingError` | Logged + raised with XGBoost error details |
| Model file not found on load | `ModelNotFoundError` | Raised with expected path |
| CPCV segment too small after purge | `ValidationError` | Raised if any training fold < 50 samples |
| Backtest with no trades | No error | Returns BacktestResult with 0 trades, NaN metrics |

## 14. Performance Considerations

- **Label computation:** O(n * max_holding_period) — linear scan, no vectorization needed for correctness. For 5000 rows x 15 days = 75K iterations, completes in < 1 second.
- **XGBoost training:** ~500 trees on 142 features, ~4000 samples → < 30 seconds on CPU.
- **CPCV:** 15 paths x 30 seconds = ~7.5 minutes total. Parallelizable but sequential for Week 3 (add joblib Parallel in Week 6 if needed).
- **VectorBT backtest:** Vectorized operations, < 5 seconds for single-stock 5-year backtest.

## 15. Out of Scope (Week 3)

- Meta-labeling (Week 5 — needs primary model trained first)
- LSTM and TFT models (Week 4-5)
- HMM regime detection (Week 4)
- Stacking ensemble and Ridge meta-learner (Week 5)
- Half-Kelly position sizing (Week 6)
- Multi-stock portfolio backtesting (Week 6)
- FastAPI endpoints (Week 6)
