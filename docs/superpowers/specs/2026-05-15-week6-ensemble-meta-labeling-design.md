# Week 6: Ensemble Stacking + Meta-Labeling — Design Spec

## Goal

Build the stacking ensemble meta-learner that combines XGBoost, LSTM, and TFT base model outputs with HMM regime probabilities into a final prediction, plus a meta-labeling model that gates low-confidence signals before they reach the risk layer.

## Architecture

Two new modules in `alphavedha/models/`, each with a single focused class. Neither subclasses `BaseModel` — they have custom interfaces (like `RegimeDetector` and `ConformalPredictor`) since their inputs are predictions from other models, not raw features.

**Tech stack:** scikit-learn `RidgeClassifier` (ensemble), XGBoost `XGBClassifier` (meta-labeling), joblib (serialization), structlog (logging).

---

## File Structure

| File | Class | Responsibility |
|------|-------|----------------|
| `alphavedha/models/ensemble.py` | `StackingEnsemble` | Combines 3 base model OOF outputs + regime probs → final prediction |
| `alphavedha/models/meta_model.py` | `MetaLabelingModel` | Binary gate: P(ensemble prediction is correct) |
| `tests/unit/models/test_ensemble.py` | — | ~14 tests for StackingEnsemble |
| `tests/unit/models/test_meta_model.py` | — | ~10 tests for MetaLabelingModel |
| `alphavedha/models/__init__.py` | — | Export new classes |
| `alphavedha/models/CLAUDE.md` | — | Update with implementation details |

---

## 1. StackingEnsemble (`ensemble.py`)

### Meta-Features (14 dimensions)

The meta-learner input is a matrix built from base model outputs:

| Source | Features | Count |
|--------|----------|-------|
| XGBoost | Class probabilities: P(down), P(neutral), P(up) | 3 |
| LSTM | Class probabilities: P(down), P(neutral), P(up) | 3 |
| TFT | Class probabilities: P(down), P(neutral), P(up) — primary horizon (7d) | 3 |
| Regime | State probabilities: P(bull), P(bear), P(sideways), P(high_vol) | 4 |
| Disagreement | std of the 3 base models' probability for the predicted class | 1 |
| **Total** | | **14** |

### Meta-Learner

`sklearn.linear_model.RidgeClassifier(alpha=config.alpha)` (default alpha=1.0).

- Fast to train, strong regularization prevents overfitting on small OOF datasets (~100-200 samples).
- Multi-class classification (3 classes: down/neutral/up).
- Probabilities derived via softmax of the decision function values (RidgeClassifier doesn't produce probabilities natively).

### Output Dataclass

```python
@dataclass
class EnsembleResult:
    direction: np.ndarray          # final direction per sample: -1, 0, or 1
    magnitude: np.ndarray          # confidence-weighted average of base magnitudes
    probabilities: np.ndarray      # softmax of decision function (n_samples, 3)
    confidence: np.ndarray         # max class probability per sample
    model_disagreement: np.ndarray # std across base model probabilities
```

### Interface

```python
class StackingEnsemble:
    EXPECTED_MODELS: ClassVar[list[str]] = ["xgboost", "lstm", "tft"]

    def __init__(self, config: EnsembleConfig | None = None) -> None: ...

    def fit(
        self,
        base_oof_predictions: dict[str, PredictionResult],
        regime_probs: np.ndarray,
        y_true: pd.Series,
    ) -> dict[str, float]: ...

    def predict(
        self,
        base_predictions: dict[str, PredictionResult],
        regime_probs: np.ndarray,
    ) -> EnsembleResult: ...

    def save(self, directory: Path) -> None: ...

    @classmethod
    def load(cls, directory: Path) -> StackingEnsemble: ...
```

### Meta-Feature Construction

The `_build_meta_features` private method constructs the 14-column matrix:

1. Extract `probabilities` array (n_samples, 3) from each base model's `PredictionResult`.
2. Append regime_probs (n_samples, 4).
3. Compute disagreement: for each sample, take the argmax class across the 3 models, then compute std of the probabilities for that class.
4. Column-stack all into (n_samples, 14).

### Magnitude Aggregation

Final magnitude = confidence-weighted average of base model magnitudes:

```
weights = [conf_xgb, conf_lstm, conf_tft] / sum(confidences)
magnitude = sum(weights[i] * base_magnitude[i])
```

### Direction Mapping

RidgeClassifier predicts class indices {0, 1, 2}. Map back via `{0: -1, 1: 0, 2: 1}`.

### Training Rule

The ensemble ONLY trains on out-of-fold predictions from base models (produced via CPCV). Never train on in-sample predictions — this would inflate meta-learner accuracy.

### Training Metrics

- `accuracy`: Classification accuracy on OOF set
- `f1_weighted`: Weighted F1 score

---

## 2. MetaLabelingModel (`meta_model.py`)

### Purpose

Answers "Should I bet on this signal?" — not "What direction?" The meta-labeling model takes the ensemble's output plus the original feature matrix and predicts the probability that the ensemble's direction prediction is correct.

### Input Features (N_features + 2)

| Feature | Description |
|---------|-------------|
| Original features | All features from the pipeline (30-141 columns depending on feature selection) |
| `ensemble_direction` | int: -1, 0, or 1 — what the ensemble predicted |
| `ensemble_confidence` | float in [0, 1] — how confident the ensemble is |

### Target

Binary: 1 if the ensemble's predicted direction matches the actual outcome, 0 otherwise.

### Model

`xgboost.XGBClassifier` with binary:logistic objective. Uses the same XGBoost hyperparameters from `config.models.xgboost.params` (learning_rate, max_depth, n_estimators, etc.) with early stopping.

### Output Dataclass

```python
@dataclass
class MetaLabelResult:
    meta_confidence: np.ndarray  # P(ensemble prediction is correct) per sample
    is_tradeable: np.ndarray     # boolean mask: meta_confidence > threshold
```

### Interface

```python
class MetaLabelingModel:
    def __init__(self, config: MetaLabelingConfig | None = None) -> None: ...

    def fit(
        self,
        X_features: pd.DataFrame,
        ensemble_direction: np.ndarray,
        ensemble_confidence: np.ndarray,
        y_correct: pd.Series,
        X_val: pd.DataFrame | None = None,
        ensemble_direction_val: np.ndarray | None = None,
        ensemble_confidence_val: np.ndarray | None = None,
        y_correct_val: pd.Series | None = None,
    ) -> dict[str, float]: ...

    def predict(
        self,
        X_features: pd.DataFrame,
        ensemble_direction: np.ndarray,
        ensemble_confidence: np.ndarray,
    ) -> MetaLabelResult: ...

    def save(self, directory: Path) -> None: ...

    @classmethod
    def load(cls, directory: Path) -> MetaLabelingModel: ...
```

### Feature Assembly

The `_build_features` private method:
1. Takes the raw feature DataFrame.
2. Appends `ensemble_direction` and `ensemble_confidence` as two new columns.
3. Returns the augmented DataFrame.

### Threshold

`config.min_confidence` (default 0.55). Predictions with `meta_confidence > threshold` are marked `is_tradeable = True`. Below threshold → signal is suppressed (not traded).

### Training Data

The meta-labeling model trains on the ensemble's OOF predictions (or a held-out calibration set). The binary labels are derived by comparing ensemble direction predictions to actual outcomes.

### Training Metrics

- `accuracy`: Binary classification accuracy
- `precision`: Precision of "tradeable" class
- `recall`: Recall of "tradeable" class
- `f1`: F1 score

---

## 3. Training Pipeline Data Flow

```
Step 1: Base models produce OOF predictions via CPCV
  XGBoost.fit(fold_train) → predict(fold_test) → OOF predictions
  LSTM.fit(fold_train) → predict(fold_test) → OOF predictions
  TFT.fit(fold_train) → predict(fold_test) → OOF predictions

Step 2: Regime detector runs on full market history
  RegimeDetector.predict(returns, volatility) → regime_probs per date

Step 3: Ensemble trains on OOF predictions
  StackingEnsemble.fit(oof_preds, regime_probs, y_true) → fitted meta-learner

Step 4: Ensemble produces its own OOF predictions
  (Use same CPCV folds or held-out calibration set)

Step 5: Meta-labeling trains on ensemble OOF + original features
  y_correct = (ensemble_direction == actual_direction).astype(int)
  MetaLabelingModel.fit(X_features, ens_dir, ens_conf, y_correct)
```

Each step follows the OOF principle — no data leakage between training and evaluation.

---

## 4. Configuration

Already defined in `alphavedha/config.py` — no changes needed:

```python
class EnsembleConfig(BaseModel):
    meta_learner: str = "ridge"    # only "ridge" supported
    alpha: float = 1.0             # Ridge regularization strength

class MetaLabelingConfig(BaseModel):
    min_confidence: float = 0.55   # gate threshold
    model: str = "xgboost"         # only "xgboost" supported
```

---

## 5. Serialization

Both follow the established pattern (same as RegimeDetector/ConformalPredictor):

| Model | Artifact | Format |
|-------|----------|--------|
| StackingEnsemble | `ridge_model.joblib` | joblib |
| StackingEnsemble | `metadata.json` | JSON (config, metrics, expected_models, created_at) |
| MetaLabelingModel | `meta_classifier.joblib` | joblib |
| MetaLabelingModel | `metadata.json` | JSON (config, metrics, feature_names, created_at) |

---

## 6. Error Handling

| Exception | When |
|-----------|------|
| `DataQualityError` | NaN or Inf values in any input array |
| `InsufficientDataError` | Fewer than 10 OOF samples for ensemble, or empty input |
| `ModelTrainingError` | Calling predict/save on unfitted model |
| `ModelNotFoundError` | Missing artifact files during load |
| `ValueError` | `base_oof_predictions` dict doesn't contain exactly the 3 expected model names |

All errors are logged with structlog before raising.

---

## 7. Exports

Update `alphavedha/models/__init__.py` to export:
- `StackingEnsemble`, `EnsembleResult`
- `MetaLabelingModel`, `MetaLabelResult`

---

## 8. Test Plan

### test_ensemble.py (~14 tests)

1. `test_fit_returns_metrics` — fit with synthetic 3-model OOF predictions + regime probs, verify dict with accuracy and f1_weighted
2. `test_predict_returns_ensemble_result` — predict returns EnsembleResult dataclass
3. `test_direction_values_valid` — all directions in {-1, 0, 1}
4. `test_probabilities_sum_to_one` — each row of probabilities sums to ~1.0
5. `test_confidence_range` — all confidence values in [0, 1]
6. `test_model_disagreement_is_std` — disagreement matches manual std computation
7. `test_magnitude_is_weighted_average` — magnitude matches confidence-weighted average of base magnitudes
8. `test_save_load_roundtrip` — save, load, predict produces same results
9. `test_predict_before_fit_raises` — ModelTrainingError
10. `test_nan_input_raises` — DataQualityError
11. `test_inf_input_raises` — DataQualityError
12. `test_missing_model_name_raises` — ValueError when dict is missing a model
13. `test_extra_model_name_raises` — ValueError when dict has unexpected model
14. `test_empty_predictions_raises` — InsufficientDataError

### test_meta_model.py (~10 tests)

1. `test_fit_returns_metrics` — fit with synthetic data, verify dict with accuracy, precision, recall, f1
2. `test_predict_returns_meta_label_result` — predict returns MetaLabelResult
3. `test_meta_confidence_range` — all values in [0, 1]
4. `test_is_tradeable_respects_threshold` — mask matches meta_confidence > 0.55
5. `test_custom_threshold` — changing min_confidence changes the mask
6. `test_save_load_roundtrip` — save, load, predict produces same results
7. `test_predict_before_fit_raises` — ModelTrainingError
8. `test_nan_input_raises` — DataQualityError
9. `test_empty_input_raises` — InsufficientDataError
10. `test_fit_with_validation_set` — fit with val data, early stopping works
