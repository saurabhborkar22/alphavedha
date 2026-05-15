# Week 5: HMM Regime Detector + Conformal Prediction — Design Spec

> **Date:** 2026-05-15 | **Branch:** `feature/week5-regime-conformal` | **Depends on:** Week 4 (LSTM, TFT merged to main)

## Goal

Add two standalone utility models that augment the base predictors:

1. **HMM Regime Detector** — classifies the current market regime (bull/bear/sideways/high-volatility) using hidden Markov model on index returns + volatility
2. **Conformal Predictor** — wraps any sklearn-compatible regressor with MAPIE to produce prediction intervals with coverage guarantees

Neither model predicts stock direction/magnitude directly. They are consumed by the prediction engine (Week 7) and ensemble (Week 6) as auxiliary signals.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| BaseModel ABC | Neither model extends it | They don't output direction/magnitude — forcing BaseModel would require dummy fields |
| HMM interface | Standalone `RegimeDetector` with own `RegimeResult` | Clean semantic fit — regime detection is a classifier over market states, not a stock predictor |
| Conformal interface | Standalone `ConformalPredictor` with own `ConformalResult` | It's a calibration wrapper, not a base predictor |
| Conformal base regressor | Accepts any sklearn-compatible regressor | Flexible for ensemble work — can wrap XGBoost, Ridge, or any future regressor |
| Serialization | `joblib` for both | Matches CLAUDE.md spec for HMM and sklearn-based models |
| Test data | Synthetic fixtures only | No external data dependencies, consistent with Weeks 3-4 |
| MAPIE method | `jackknife+` (`method="plus"`) | Finite-sample coverage guarantees without separate calibration holdout |

## Architecture

### HMM Regime Detector

**File:** `alphavedha/models/regime.py`

#### RegimeResult

```python
@dataclass
class RegimeResult:
    current_regime: str          # "bull", "bear", "sideways", "high_volatility"
    regime_id: int               # 0-3 (mapped state ID)
    state_probabilities: np.ndarray  # shape (4,) — P(each state) for last observation
    regime_history: np.ndarray   # shape (n_samples,) — decoded state sequence (semantic IDs, not HMM internal IDs)
    transition_matrix: np.ndarray  # shape (4, 4) — state transition probabilities
```

#### RegimeDetector

**Constructor:** `RegimeDetector(config: RegimeConfig | None = None)`

Uses existing `RegimeConfig` (n_states=4, covariance_type="full", n_iter=100, state_names=["bull", "bear", "sideways", "high_volatility"]).

**`fit(returns: pd.Series, volatility: pd.Series) -> dict[str, float]`**

1. Stack inputs into 2D array: `[log_returns, volatility]` shape (n_samples, 2)
2. Fit `hmmlearn.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)`
3. Auto-label states by mapping HMM's arbitrary state IDs to semantic names:
   - Highest mean return → "bull"
   - Lowest mean return → "bear"
   - Highest variance (of remaining 2) → "high_volatility"
   - Remaining → "sideways"
   - Tiebreaker: lower HMM state index wins the earlier label in the list above
4. Save the state mapping (HMM state ID → semantic name) for reuse in predict
5. Return training metrics: log-likelihood, AIC, BIC

**`predict(returns: pd.Series, volatility: pd.Series) -> RegimeResult`**

1. Stack inputs into 2D array
2. Run Viterbi decoding (`model.predict()`) to get most likely state sequence
3. Get state probabilities for last observation (`model.predict_proba()`)
4. Map HMM state IDs to semantic names using saved mapping
5. Return `RegimeResult` with current regime, probabilities, history, transition matrix

**`get_regime_features() -> pd.DataFrame`**

Returns state probabilities for all observations as a DataFrame with 4 columns (`p_bull`, `p_bear`, `p_sideways`, `p_high_volatility`). Useful as input features for the ensemble model.

**`save(directory: Path) -> None`** / **`load(cls, directory: Path) -> RegimeDetector`**

- `joblib.dump()` for the HMM model object
- `metadata.json` with: state mapping, config, training metrics, timestamp

### Conformal Predictor

**File:** `alphavedha/models/conformal.py`

#### ConformalResult

```python
@dataclass
class ConformalResult:
    price_low: np.ndarray       # lower bound per sample
    price_mid: np.ndarray       # point estimate per sample
    price_high: np.ndarray      # upper bound per sample
    interval_width: np.ndarray  # high - low per sample
    coverage: float             # target coverage (e.g. 0.90)
```

Note: "price" is a misnomer carried from the CLAUDE.md spec. Values are in return-space (predicted % return), not price-space. The prediction engine (Week 7) converts to price targets.

#### ConformalPredictor

**Constructor:** `ConformalPredictor(config: ConformalConfig | None = None, base_regressor: Any | None = None)`

- `config`: uses existing `ConformalConfig` (coverage=0.90, calibration_window=60) + new `method` field
- `base_regressor`: any sklearn-compatible regressor. Defaults to `XGBRegressor(n_estimators=100)` if None

**`fit(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]`**

1. Wrap the base regressor with `MapieRegressor(estimator=base_regressor, method="plus")`
2. Call `mapie_regressor.fit(X_train, y_train)` — internally cross-validates to build conformity scores
3. Return training metrics: base regressor's R², RMSE

**`predict(X: pd.DataFrame) -> ConformalResult`**

1. Call `mapie_regressor.predict(X, alpha=1-coverage)` → returns `(y_pred, y_pis)`
2. `y_pis` shape is `(n_samples, 2, 1)` — lower and upper bounds
3. Return `ConformalResult(price_low=y_pis[:, 0, 0], price_mid=y_pred, price_high=y_pis[:, 1, 0], ...)`

**`calibrate(X_cal: pd.DataFrame, y_cal: pd.Series) -> None`**

Recalibrate conformity scores on recent data (rolling 60-day window) without retraining the base regressor. Implementation: create a new `MapieRegressor` wrapping the same (frozen, pre-fitted) base regressor with `cv="prefit"`, then call `.fit(X_cal, y_cal)` to recompute conformity scores on the calibration window. This replaces the internal MAPIE state while keeping the base model unchanged.

**`save(directory: Path) -> None`** / **`load(cls, directory: Path) -> ConformalPredictor`**

- `joblib.dump()` for the `MapieRegressor` object (includes base regressor)
- `metadata.json` with: config, coverage, method, training metrics, timestamp

### Config Changes

**`config.py`** — add 1 field to `ConformalConfig`:

```python
class ConformalConfig(BaseModel):
    coverage: float = 0.90
    calibration_window: int = 60
    method: str = "plus"          # NEW — MAPIE method
```

### Module Exports

**`models/__init__.py`** — add to `__all__`:

```python
from alphavedha.models.regime import RegimeDetector, RegimeResult
from alphavedha.models.conformal import ConformalPredictor, ConformalResult
```

## Test Strategy

All tests use synthetic data. No external dependencies.

### test_regime.py (~14 tests)

**Synthetic fixture:** 1000 samples with 2 baked-in regimes — first 500 samples have high mean return + low variance (bull-like), last 500 have low mean return + high variance (bear-like). This gives the HMM clear structure to find.

| Test | Validates |
|------|-----------|
| `test_fit_returns_metrics` | fit() returns dict with log_likelihood, aic, bic |
| `test_predict_returns_regime_result` | predict() returns RegimeResult instance |
| `test_current_regime_is_valid_name` | current_regime in {"bull", "bear", "sideways", "high_volatility"} |
| `test_regime_id_in_range` | regime_id in 0..3 |
| `test_state_probabilities_shape_and_sum` | shape (4,), sums to 1.0 |
| `test_regime_history_shape` | shape matches input length |
| `test_regime_history_values` | all values in 0..3 |
| `test_transition_matrix_shape_and_rows` | shape (4,4), rows sum to 1.0 |
| `test_state_labeling_bull_has_highest_mean` | bull state has highest mean return |
| `test_state_labeling_bear_has_lowest_mean` | bear state has lowest mean return |
| `test_get_regime_features_shape` | returns DataFrame with 4 columns, n_samples rows |
| `test_save_load_roundtrip` | loaded model produces same predictions |
| `test_predict_before_fit_raises` | raises ModelTrainingError |
| `test_insufficient_data_raises` | raises InsufficientDataError with < 10 samples |

### test_conformal.py (~12 tests)

**Synthetic fixture:** 500 samples, 10 features, target = linear combination + noise. Clean regression setup.

| Test | Validates |
|------|-----------|
| `test_fit_returns_metrics` | fit() returns dict with r2, rmse |
| `test_predict_returns_conformal_result` | predict() returns ConformalResult instance |
| `test_prediction_shapes` | low/mid/high/width all shape (n_samples,) |
| `test_low_less_than_mid_less_than_high` | low <= mid <= high for all samples |
| `test_interval_width_positive` | all interval widths > 0 |
| `test_empirical_coverage` | actual coverage on test set >= 0.85 (target 0.90 minus tolerance) |
| `test_intervals_expand_for_noisy_data` | mean interval width on high-noise data > low-noise data |
| `test_save_load_roundtrip` | loaded model produces same predictions |
| `test_predict_before_fit_raises` | raises ModelTrainingError |
| `test_works_with_ridge_regressor` | fit/predict works with Ridge base regressor |
| `test_works_with_default_regressor` | fit/predict works when no base_regressor passed |
| `test_calibrate_updates_intervals` | calibrate() changes intervals without error |

## File Summary

| Action | File | Description |
|--------|------|-------------|
| Modify | `alphavedha/config.py` | Add `method` field to ConformalConfig |
| Create | `alphavedha/models/regime.py` | RegimeDetector + RegimeResult |
| Create | `alphavedha/models/conformal.py` | ConformalPredictor + ConformalResult |
| Modify | `alphavedha/models/__init__.py` | Add 4 new exports |
| Create | `tests/unit/models/test_regime.py` | ~14 tests |
| Create | `tests/unit/models/test_conformal.py` | ~12 tests |

**Estimated impact:** ~26 new tests (total ~254), 2 new source files, 2 modified files.
