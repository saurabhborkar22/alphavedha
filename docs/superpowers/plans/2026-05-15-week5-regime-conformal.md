# Week 5: HMM Regime Detector + Conformal Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two standalone utility models — an HMM regime detector (4 market states) and a MAPIE conformal predictor (prediction intervals with coverage guarantees) — that feed into the ensemble and prediction engine in later weeks.

**Architecture:** Two independent files under `alphavedha/models/`, each with its own dataclass result type. `RegimeDetector` wraps `hmmlearn.GaussianHMM` with auto-labeling of states by return statistics. `ConformalPredictor` wraps any sklearn-compatible regressor with `mapie.regression.MapieRegressor` for jackknife+ intervals. Both use `joblib` serialization and `metadata.json` for persistence.

**Tech Stack:** hmmlearn 0.3, MAPIE 1.4, joblib, scikit-learn (Ridge for tests), xgboost (default base regressor), numpy, pandas, structlog, existing Pydantic configs (`RegimeConfig`, `ConformalConfig`)

---

## File Structure

```
alphavedha/
├── config.py                          # Modify: add `method` field to ConformalConfig (line 169-171)
├── models/
│   ├── __init__.py                    # Modify: add 4 new exports
│   ├── regime.py                      # Create: RegimeResult, RegimeDetector
│   └── conformal.py                   # Create: ConformalResult, ConformalPredictor
tests/unit/models/
├── test_regime.py                     # Create: ~14 tests
└── test_conformal.py                  # Create: ~12 tests
```

---

### Task 1: Config Update

**Files:**
- Modify: `alphavedha/config.py:169-171`

- [ ] **Step 1: Add `method` field to ConformalConfig**

In `alphavedha/config.py`, replace the `ConformalConfig` class (lines 169-171) with:

```python
class ConformalConfig(BaseModel):
    coverage: float = 0.90
    calibration_window: int = 60
    method: str = "plus"
```

One new field: `method` defaults to `"plus"` (jackknife+). Backwards compatible.

- [ ] **Step 2: Verify config loads**

Run: `/home/lenovo/alphavedha/.venv/bin/pytest tests/ -x -q --tb=short 2>&1 | tail -5`

Expected: All 228 tests still pass (no regressions from adding a default field).

- [ ] **Step 3: Commit**

```bash
git add alphavedha/config.py
git commit -m "feat: add method field to ConformalConfig for MAPIE method selection"
```

---

### Task 2: HMM Regime Detector — Tests

**Files:**
- Create: `tests/unit/models/test_regime.py`

- [ ] **Step 1: Write all failing tests**

Create `tests/unit/models/test_regime.py`:

```python
"""Tests for RegimeDetector — HMM-based market regime classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import RegimeConfig
from alphavedha.exceptions import InsufficientDataError, ModelTrainingError
from alphavedha.models.regime import RegimeDetector, RegimeResult


@pytest.fixture
def regime_config() -> RegimeConfig:
    return RegimeConfig(n_states=4, covariance_type="full", n_iter=200)


@pytest.fixture
def synthetic_regime_data() -> tuple[pd.Series, pd.Series]:
    """1000 samples with 2 distinct regimes baked in.

    First 500: high mean return (+0.05%), low volatility (0.5%) — bull-like.
    Last 500: low mean return (-0.05%), high volatility (2.0%) — bear-like.
    """
    rng = np.random.default_rng(42)
    n_half = 500

    returns_bull = rng.normal(0.0005, 0.005, size=n_half)
    returns_bear = rng.normal(-0.0005, 0.020, size=n_half)
    returns = pd.Series(
        np.concatenate([returns_bull, returns_bear]), name="log_returns"
    )

    vol_bull = rng.normal(0.005, 0.001, size=n_half).clip(0.001)
    vol_bear = rng.normal(0.020, 0.005, size=n_half).clip(0.001)
    volatility = pd.Series(
        np.concatenate([vol_bull, vol_bear]), name="volatility"
    )

    return returns, volatility


class TestRegimeDetectorFit:
    def test_fit_returns_metrics(
        self,
        synthetic_regime_data: tuple[pd.Series, pd.Series],
        regime_config: RegimeConfig,
    ) -> None:
        returns, volatility = synthetic_regime_data
        detector = RegimeDetector(config=regime_config)
        metrics = detector.fit(returns, volatility)
        assert isinstance(metrics, dict)
        assert "log_likelihood" in metrics
        assert "aic" in metrics
        assert "bic" in metrics

    def test_predict_before_fit_raises(self, regime_config: RegimeConfig) -> None:
        detector = RegimeDetector(config=regime_config)
        returns = pd.Series([0.01, -0.01, 0.0])
        volatility = pd.Series([0.02, 0.03, 0.01])
        with pytest.raises(ModelTrainingError):
            detector.predict(returns, volatility)

    def test_insufficient_data_raises(self, regime_config: RegimeConfig) -> None:
        detector = RegimeDetector(config=regime_config)
        returns = pd.Series([0.01] * 5)
        volatility = pd.Series([0.02] * 5)
        with pytest.raises(InsufficientDataError):
            detector.fit(returns, volatility)


class TestRegimeDetectorPredict:
    @pytest.fixture(autouse=True)
    def _fitted_detector(
        self,
        synthetic_regime_data: tuple[pd.Series, pd.Series],
        regime_config: RegimeConfig,
    ) -> None:
        returns, volatility = synthetic_regime_data
        self.detector = RegimeDetector(config=regime_config)
        self.detector.fit(returns, volatility)
        self.returns = returns
        self.volatility = volatility
        self.result = self.detector.predict(returns, volatility)

    def test_predict_returns_regime_result(self) -> None:
        assert isinstance(self.result, RegimeResult)

    def test_current_regime_is_valid_name(self) -> None:
        valid = {"bull", "bear", "sideways", "high_volatility"}
        assert self.result.current_regime in valid

    def test_regime_id_in_range(self) -> None:
        assert 0 <= self.result.regime_id <= 3

    def test_state_probabilities_shape_and_sum(self) -> None:
        assert self.result.state_probabilities.shape == (4,)
        assert self.result.state_probabilities.sum() == pytest.approx(1.0, abs=1e-5)

    def test_regime_history_shape(self) -> None:
        assert self.result.regime_history.shape == (1000,)

    def test_regime_history_values(self) -> None:
        unique_vals = set(np.unique(self.result.regime_history))
        assert unique_vals.issubset({0, 1, 2, 3})

    def test_transition_matrix_shape_and_rows(self) -> None:
        tm = self.result.transition_matrix
        assert tm.shape == (4, 4)
        row_sums = tm.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestRegimeDetectorLabeling:
    def test_state_labeling_bull_has_highest_mean(
        self,
        synthetic_regime_data: tuple[pd.Series, pd.Series],
        regime_config: RegimeConfig,
    ) -> None:
        returns, volatility = synthetic_regime_data
        detector = RegimeDetector(config=regime_config)
        detector.fit(returns, volatility)
        state_mapping = detector.state_mapping
        hmm_model = detector.hmm_model
        means = hmm_model.means_[:, 0]
        bull_hmm_id = state_mapping["bull"]
        assert means[bull_hmm_id] == max(means)

    def test_state_labeling_bear_has_lowest_mean(
        self,
        synthetic_regime_data: tuple[pd.Series, pd.Series],
        regime_config: RegimeConfig,
    ) -> None:
        returns, volatility = synthetic_regime_data
        detector = RegimeDetector(config=regime_config)
        detector.fit(returns, volatility)
        state_mapping = detector.state_mapping
        hmm_model = detector.hmm_model
        means = hmm_model.means_[:, 0]
        bear_hmm_id = state_mapping["bear"]
        assert means[bear_hmm_id] == min(means)


class TestRegimeDetectorFeatures:
    def test_get_regime_features_shape(
        self,
        synthetic_regime_data: tuple[pd.Series, pd.Series],
        regime_config: RegimeConfig,
    ) -> None:
        returns, volatility = synthetic_regime_data
        detector = RegimeDetector(config=regime_config)
        detector.fit(returns, volatility)
        detector.predict(returns, volatility)
        features = detector.get_regime_features()
        assert isinstance(features, pd.DataFrame)
        assert features.shape == (1000, 4)
        expected_cols = {"p_bull", "p_bear", "p_sideways", "p_high_volatility"}
        assert set(features.columns) == expected_cols


class TestRegimeDetectorPersistence:
    def test_save_load_roundtrip(
        self,
        synthetic_regime_data: tuple[pd.Series, pd.Series],
        regime_config: RegimeConfig,
        tmp_path: Path,
    ) -> None:
        returns, volatility = synthetic_regime_data
        detector = RegimeDetector(config=regime_config)
        detector.fit(returns, volatility)
        result_before = detector.predict(returns, volatility)

        save_dir = tmp_path / "regime_test"
        detector.save(save_dir)

        loaded = RegimeDetector.load(save_dir)
        result_after = loaded.predict(returns, volatility)

        assert result_before.current_regime == result_after.current_regime
        np.testing.assert_array_equal(
            result_before.regime_history, result_after.regime_history
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/lenovo/alphavedha/.venv/bin/pytest tests/unit/models/test_regime.py -v 2>&1 | tail -5`

Expected: FAIL — `ModuleNotFoundError: No module named 'alphavedha.models.regime'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/unit/models/test_regime.py
git commit -m "test: add failing tests for HMM RegimeDetector (14 tests)"
```

---

### Task 3: HMM Regime Detector — Implementation

**Files:**
- Create: `alphavedha/models/regime.py`

- [ ] **Step 1: Write implementation**

Create `alphavedha/models/regime.py`:

```python
"""HMM Regime Detector — classifies market regimes using Gaussian HMM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
from hmmlearn.hmm import GaussianHMM

from alphavedha.config import RegimeConfig
from alphavedha.exceptions import (
    InsufficientDataError,
    ModelNotFoundError,
    ModelTrainingError,
)

logger = structlog.get_logger(__name__)

_MIN_SAMPLES = 10


@dataclass
class RegimeResult:
    current_regime: str
    regime_id: int
    state_probabilities: np.ndarray
    regime_history: np.ndarray
    transition_matrix: np.ndarray


class RegimeDetector:
    """Detects market regimes (bull/bear/sideways/high-volatility) via Gaussian HMM."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self._config = config or RegimeConfig()
        self._hmm: GaussianHMM | None = None
        self._is_fitted = False
        self._state_mapping: dict[str, int] = {}
        self._reverse_mapping: dict[int, str] = {}
        self._training_metrics: dict[str, float] = {}
        self._last_posteriors: np.ndarray | None = None

    @property
    def state_mapping(self) -> dict[str, int]:
        return dict(self._state_mapping)

    @property
    def hmm_model(self) -> GaussianHMM:
        if self._hmm is None:
            raise ModelTrainingError("RegimeDetector is not fitted.")
        return self._hmm

    def fit(self, returns: pd.Series, volatility: pd.Series) -> dict[str, float]:
        X = self._prepare_input(returns, volatility)
        n_samples = X.shape[0]
        if n_samples < _MIN_SAMPLES:
            raise InsufficientDataError(
                f"Need at least {_MIN_SAMPLES} samples, got {n_samples}"
            )

        self._hmm = GaussianHMM(
            n_components=self._config.n_states,
            covariance_type=self._config.covariance_type,
            n_iter=self._config.n_iter,
            random_state=42,
        )
        self._hmm.fit(X)
        self._label_states()
        self._is_fitted = True

        n_params = self._count_params()
        log_likelihood = float(self._hmm.score(X))
        self._training_metrics = {
            "log_likelihood": log_likelihood,
            "aic": -2.0 * log_likelihood + 2.0 * n_params,
            "bic": -2.0 * log_likelihood + n_params * np.log(n_samples),
        }

        logger.info(
            "regime_detector_fitted",
            n_samples=n_samples,
            metrics=self._training_metrics,
        )
        return dict(self._training_metrics)

    def predict(self, returns: pd.Series, volatility: pd.Series) -> RegimeResult:
        if not self._is_fitted or self._hmm is None:
            raise ModelTrainingError("RegimeDetector is not fitted. Call fit() first.")

        X = self._prepare_input(returns, volatility)

        raw_states = self._hmm.predict(X)
        posteriors = self._hmm.predict_proba(X)
        self._last_posteriors = posteriors

        mapped_history = np.array(
            [self._hmm_id_to_semantic_id(s) for s in raw_states]
        )

        last_posteriors_reordered = self._reorder_probabilities(posteriors[-1])
        last_semantic_id = mapped_history[-1]
        current_name = self._config.state_names[last_semantic_id]

        raw_transmat = self._hmm.transmat_
        mapped_transmat = self._reorder_transition_matrix(raw_transmat)

        return RegimeResult(
            current_regime=current_name,
            regime_id=int(last_semantic_id),
            state_probabilities=last_posteriors_reordered,
            regime_history=mapped_history,
            transition_matrix=mapped_transmat,
        )

    def get_regime_features(self) -> pd.DataFrame:
        if self._last_posteriors is None:
            raise ModelTrainingError("No predictions available. Call predict() first.")

        n_states = self._config.n_states
        columns: list[str] = []
        reordered = np.zeros_like(self._last_posteriors)

        for semantic_id in range(n_states):
            name = self._config.state_names[semantic_id]
            columns.append(f"p_{name}")
            hmm_id = self._semantic_id_to_hmm_id(semantic_id)
            reordered[:, semantic_id] = self._last_posteriors[:, hmm_id]

        return pd.DataFrame(reordered, columns=columns)

    def save(self, directory: Path) -> None:
        if not self._is_fitted or self._hmm is None:
            raise ModelTrainingError("Cannot save unfitted RegimeDetector.")

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._hmm, directory / "hmm_model.joblib")

        metadata: dict[str, Any] = {
            "name": "regime_detector",
            "created_at": datetime.now(UTC).isoformat(),
            "state_mapping": self._state_mapping,
            "config": self._config.model_dump(),
            "metrics": self._training_metrics,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("regime_detector_saved", path=str(directory))

    @classmethod
    def load(cls, directory: Path) -> RegimeDetector:
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotFoundError(f"No metadata.json at {directory}")

        metadata = json.loads(metadata_path.read_text())

        hmm_path = directory / "hmm_model.joblib"
        if not hmm_path.exists():
            raise ModelNotFoundError(f"No hmm_model.joblib at {directory}")

        config = RegimeConfig.model_validate(metadata["config"])
        detector = cls(config=config)
        detector._hmm = joblib.load(hmm_path)
        detector._state_mapping = {
            k: int(v) for k, v in metadata["state_mapping"].items()
        }
        detector._reverse_mapping = {v: k for k, v in detector._state_mapping.items()}
        detector._training_metrics = metadata.get("metrics", {})
        detector._is_fitted = True

        logger.info("regime_detector_loaded", path=str(directory))
        return detector

    def _prepare_input(
        self, returns: pd.Series, volatility: pd.Series
    ) -> np.ndarray:
        return np.column_stack(
            [returns.values.astype(np.float64), volatility.values.astype(np.float64)]
        )

    def _label_states(self) -> None:
        if self._hmm is None:
            return

        means = self._hmm.means_[:, 0]
        n_states = self._config.n_states
        covariances = np.array(
            [self._hmm.covars_[i][0, 0] for i in range(n_states)]
        )

        assigned: set[int] = set()
        self._state_mapping = {}

        bull_id = int(np.argmax(means))
        self._state_mapping["bull"] = bull_id
        assigned.add(bull_id)

        remaining_for_bear = [i for i in range(n_states) if i not in assigned]
        bear_id = min(remaining_for_bear, key=lambda i: means[i])
        self._state_mapping["bear"] = bear_id
        assigned.add(bear_id)

        remaining = [i for i in range(n_states) if i not in assigned]
        highvol_id = max(remaining, key=lambda i: covariances[i])
        self._state_mapping["high_volatility"] = highvol_id
        assigned.add(highvol_id)

        sideways_id = [i for i in range(n_states) if i not in assigned][0]
        self._state_mapping["sideways"] = sideways_id

        self._reverse_mapping = {v: k for k, v in self._state_mapping.items()}

    def _hmm_id_to_semantic_id(self, hmm_id: int) -> int:
        name = self._reverse_mapping[hmm_id]
        return self._config.state_names.index(name)

    def _semantic_id_to_hmm_id(self, semantic_id: int) -> int:
        name = self._config.state_names[semantic_id]
        return self._state_mapping[name]

    def _reorder_probabilities(self, raw_probs: np.ndarray) -> np.ndarray:
        n_states = self._config.n_states
        reordered = np.zeros(n_states)
        for semantic_id in range(n_states):
            hmm_id = self._semantic_id_to_hmm_id(semantic_id)
            reordered[semantic_id] = raw_probs[hmm_id]
        return reordered

    def _reorder_transition_matrix(self, raw_transmat: np.ndarray) -> np.ndarray:
        n = self._config.n_states
        reordered = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                hmm_i = self._semantic_id_to_hmm_id(i)
                hmm_j = self._semantic_id_to_hmm_id(j)
                reordered[i, j] = raw_transmat[hmm_i, hmm_j]
        return reordered

    def _count_params(self) -> int:
        n = self._config.n_states
        n_features = 2
        start_probs = n - 1
        transition = n * (n - 1)
        means = n * n_features
        if self._config.covariance_type == "full":
            covars = n * n_features * (n_features + 1) // 2
        elif self._config.covariance_type == "diag":
            covars = n * n_features
        elif self._config.covariance_type == "spherical":
            covars = n
        else:
            covars = n * n_features
        return start_probs + transition + means + covars
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `/home/lenovo/alphavedha/.venv/bin/pytest tests/unit/models/test_regime.py -v 2>&1 | tail -20`

Expected: 14 passed

- [ ] **Step 3: Run lint**

Run: `/home/lenovo/alphavedha/.venv/bin/ruff check alphavedha/models/regime.py && /home/lenovo/alphavedha/.venv/bin/ruff format --check alphavedha/models/regime.py`

Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add alphavedha/models/regime.py
git commit -m "feat: add HMM RegimeDetector with auto-labeling and joblib persistence"
```

---

### Task 4: Conformal Predictor — Tests

**Files:**
- Create: `tests/unit/models/test_conformal.py`

- [ ] **Step 1: Write all failing tests**

Create `tests/unit/models/test_conformal.py`:

```python
"""Tests for ConformalPredictor — MAPIE-based prediction intervals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from alphavedha.config import ConformalConfig
from alphavedha.exceptions import ModelTrainingError
from alphavedha.models.conformal import ConformalPredictor, ConformalResult


@pytest.fixture
def conformal_config() -> ConformalConfig:
    return ConformalConfig(coverage=0.90, calibration_window=60, method="plus")


@pytest.fixture
def synthetic_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """500 samples, 10 features, target = linear combination + noise."""
    rng = np.random.default_rng(42)
    n, f = 500, 10
    X = pd.DataFrame(
        rng.standard_normal((n, f)), columns=[f"f{i}" for i in range(f)]
    )
    coeffs = rng.standard_normal(f)
    noise = rng.normal(0, 0.1, size=n)
    y = pd.Series(X.values @ coeffs + noise, name="target")
    return X, y


class TestConformalPredictorFit:
    def test_fit_returns_metrics(
        self,
        synthetic_regression_data: tuple[pd.DataFrame, pd.Series],
        conformal_config: ConformalConfig,
    ) -> None:
        X, y = synthetic_regression_data
        predictor = ConformalPredictor(config=conformal_config)
        metrics = predictor.fit(X[:400], y[:400])
        assert isinstance(metrics, dict)
        assert "r2" in metrics
        assert "rmse" in metrics

    def test_predict_before_fit_raises(self, conformal_config: ConformalConfig) -> None:
        predictor = ConformalPredictor(config=conformal_config)
        X = pd.DataFrame({"a": range(10), "b": range(10)})
        with pytest.raises(ModelTrainingError):
            predictor.predict(X)

    def test_works_with_default_regressor(
        self,
        synthetic_regression_data: tuple[pd.DataFrame, pd.Series],
        conformal_config: ConformalConfig,
    ) -> None:
        X, y = synthetic_regression_data
        predictor = ConformalPredictor(config=conformal_config)
        predictor.fit(X[:400], y[:400])
        result = predictor.predict(X[400:])
        assert isinstance(result, ConformalResult)

    def test_works_with_ridge_regressor(
        self,
        synthetic_regression_data: tuple[pd.DataFrame, pd.Series],
        conformal_config: ConformalConfig,
    ) -> None:
        X, y = synthetic_regression_data
        predictor = ConformalPredictor(
            config=conformal_config, base_regressor=Ridge(alpha=1.0)
        )
        predictor.fit(X[:400], y[:400])
        result = predictor.predict(X[400:])
        assert isinstance(result, ConformalResult)


class TestConformalPredictorPredict:
    @pytest.fixture(autouse=True)
    def _fitted_predictor(
        self,
        synthetic_regression_data: tuple[pd.DataFrame, pd.Series],
        conformal_config: ConformalConfig,
    ) -> None:
        X, y = synthetic_regression_data
        self.predictor = ConformalPredictor(config=conformal_config)
        self.predictor.fit(X[:400], y[:400])
        self.X_test = X[400:]
        self.y_test = y[400:]
        self.result = self.predictor.predict(self.X_test)

    def test_predict_returns_conformal_result(self) -> None:
        assert isinstance(self.result, ConformalResult)

    def test_prediction_shapes(self) -> None:
        n = len(self.X_test)
        assert self.result.price_low.shape == (n,)
        assert self.result.price_mid.shape == (n,)
        assert self.result.price_high.shape == (n,)
        assert self.result.interval_width.shape == (n,)

    def test_low_less_than_mid_less_than_high(self) -> None:
        assert np.all(self.result.price_low <= self.result.price_mid)
        assert np.all(self.result.price_mid <= self.result.price_high)

    def test_interval_width_positive(self) -> None:
        assert np.all(self.result.interval_width > 0)

    def test_empirical_coverage(self) -> None:
        in_interval = (self.y_test.values >= self.result.price_low) & (
            self.y_test.values <= self.result.price_high
        )
        actual_coverage = in_interval.mean()
        assert actual_coverage >= 0.85


class TestConformalPredictorVolatility:
    def test_intervals_expand_for_noisy_data(
        self, conformal_config: ConformalConfig
    ) -> None:
        rng = np.random.default_rng(123)
        n, f = 300, 5
        X = pd.DataFrame(
            rng.standard_normal((n, f)), columns=[f"f{i}" for i in range(f)]
        )
        coeffs = rng.standard_normal(f)

        y_low_noise = pd.Series(X.values @ coeffs + rng.normal(0, 0.01, size=n))
        y_high_noise = pd.Series(X.values @ coeffs + rng.normal(0, 1.0, size=n))

        pred_low = ConformalPredictor(config=conformal_config)
        pred_low.fit(X[:200], y_low_noise[:200])
        result_low = pred_low.predict(X[200:])

        pred_high = ConformalPredictor(config=conformal_config)
        pred_high.fit(X[:200], y_high_noise[:200])
        result_high = pred_high.predict(X[200:])

        assert result_high.interval_width.mean() > result_low.interval_width.mean()


class TestConformalPredictorCalibrate:
    def test_calibrate_updates_intervals(
        self,
        synthetic_regression_data: tuple[pd.DataFrame, pd.Series],
        conformal_config: ConformalConfig,
    ) -> None:
        X, y = synthetic_regression_data
        predictor = ConformalPredictor(config=conformal_config)
        predictor.fit(X[:300], y[:300])
        result_before = predictor.predict(X[400:])

        predictor.calibrate(X[300:400], y[300:400])
        result_after = predictor.predict(X[400:])

        assert not np.array_equal(
            result_before.interval_width, result_after.interval_width
        )


class TestConformalPredictorPersistence:
    def test_save_load_roundtrip(
        self,
        synthetic_regression_data: tuple[pd.DataFrame, pd.Series],
        conformal_config: ConformalConfig,
        tmp_path: Path,
    ) -> None:
        X, y = synthetic_regression_data
        predictor = ConformalPredictor(config=conformal_config)
        predictor.fit(X[:400], y[:400])
        result_before = predictor.predict(X[400:])

        save_dir = tmp_path / "conformal_test"
        predictor.save(save_dir)

        loaded = ConformalPredictor.load(save_dir)
        result_after = loaded.predict(X[400:])

        np.testing.assert_allclose(
            result_before.price_mid, result_after.price_mid, atol=1e-5
        )
        np.testing.assert_allclose(
            result_before.price_low, result_after.price_low, atol=1e-5
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/lenovo/alphavedha/.venv/bin/pytest tests/unit/models/test_conformal.py -v 2>&1 | tail -5`

Expected: FAIL — `ModuleNotFoundError: No module named 'alphavedha.models.conformal'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/unit/models/test_conformal.py
git commit -m "test: add failing tests for ConformalPredictor (12 tests)"
```

---

### Task 5: Conformal Predictor — Implementation

**Files:**
- Create: `alphavedha/models/conformal.py`

- [ ] **Step 1: Write implementation**

Create `alphavedha/models/conformal.py`:

```python
"""Conformal Predictor — MAPIE-based prediction intervals with coverage guarantees."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
from mapie.regression import MapieRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from alphavedha.config import ConformalConfig
from alphavedha.exceptions import ModelNotFoundError, ModelTrainingError

logger = structlog.get_logger(__name__)


@dataclass
class ConformalResult:
    price_low: np.ndarray
    price_mid: np.ndarray
    price_high: np.ndarray
    interval_width: np.ndarray
    coverage: float


class ConformalPredictor:
    """Wraps any sklearn-compatible regressor with MAPIE for prediction intervals."""

    def __init__(
        self,
        config: ConformalConfig | None = None,
        base_regressor: Any | None = None,
    ) -> None:
        self._config = config or ConformalConfig()
        self._base_regressor = base_regressor or XGBRegressor(
            n_estimators=100, random_state=42
        )
        self._mapie: MapieRegressor | None = None
        self._is_fitted = False
        self._training_metrics: dict[str, float] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
        self._mapie = MapieRegressor(
            estimator=self._base_regressor,
            method=self._config.method,
            random_state=42,
        )
        self._mapie.fit(X_train.values, y_train.values)
        self._is_fitted = True

        y_pred = self._mapie.predict(X_train.values)[0]
        self._training_metrics = {
            "r2": float(r2_score(y_train.values, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_train.values, y_pred))),
        }

        logger.info(
            "conformal_predictor_fitted",
            n_samples=len(X_train),
            metrics=self._training_metrics,
        )
        return dict(self._training_metrics)

    def predict(self, X: pd.DataFrame) -> ConformalResult:
        if not self._is_fitted or self._mapie is None:
            raise ModelTrainingError(
                "ConformalPredictor is not fitted. Call fit() first."
            )

        alpha = 1.0 - self._config.coverage
        y_pred, y_pis = self._mapie.predict(X.values, alpha=alpha)

        price_low = y_pis[:, 0, 0]
        price_high = y_pis[:, 1, 0]

        return ConformalResult(
            price_low=price_low,
            price_mid=y_pred,
            price_high=price_high,
            interval_width=price_high - price_low,
            coverage=self._config.coverage,
        )

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        if not self._is_fitted or self._mapie is None:
            raise ModelTrainingError(
                "ConformalPredictor is not fitted. Call fit() first."
            )

        fitted_base = self._mapie.estimator_.single_estimator_
        self._mapie = MapieRegressor(
            estimator=fitted_base, cv="prefit", random_state=42
        )
        self._mapie.fit(X_cal.values, y_cal.values)

        logger.info("conformal_predictor_calibrated", n_cal_samples=len(X_cal))

    def save(self, directory: Path) -> None:
        if not self._is_fitted or self._mapie is None:
            raise ModelTrainingError("Cannot save unfitted ConformalPredictor.")

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._mapie, directory / "mapie_model.joblib")

        metadata: dict[str, Any] = {
            "name": "conformal_predictor",
            "created_at": datetime.now(UTC).isoformat(),
            "config": self._config.model_dump(),
            "metrics": self._training_metrics,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("conformal_predictor_saved", path=str(directory))

    @classmethod
    def load(cls, directory: Path) -> ConformalPredictor:
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotFoundError(f"No metadata.json at {directory}")

        metadata = json.loads(metadata_path.read_text())

        mapie_path = directory / "mapie_model.joblib"
        if not mapie_path.exists():
            raise ModelNotFoundError(f"No mapie_model.joblib at {directory}")

        config = ConformalConfig.model_validate(metadata["config"])
        predictor = cls(config=config)
        predictor._mapie = joblib.load(mapie_path)
        predictor._training_metrics = metadata.get("metrics", {})
        predictor._is_fitted = True

        logger.info("conformal_predictor_loaded", path=str(directory))
        return predictor
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `/home/lenovo/alphavedha/.venv/bin/pytest tests/unit/models/test_conformal.py -v 2>&1 | tail -20`

Expected: 12 passed

- [ ] **Step 3: Run lint**

Run: `/home/lenovo/alphavedha/.venv/bin/ruff check alphavedha/models/conformal.py && /home/lenovo/alphavedha/.venv/bin/ruff format --check alphavedha/models/conformal.py`

Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add alphavedha/models/conformal.py
git commit -m "feat: add ConformalPredictor with MAPIE jackknife+ intervals"
```

---

### Task 6: Module Exports

**Files:**
- Modify: `alphavedha/models/__init__.py`

- [ ] **Step 1: Update __init__.py**

Replace the full content of `alphavedha/models/__init__.py` with:

```python
"""Models — BaseModel ABC, model implementations, and utility models."""

from alphavedha.models.base import (
    BaseModel,
    ModelArtifact,
    PredictionResult,
    TrainResult,
)
from alphavedha.models.conformal import ConformalPredictor, ConformalResult
from alphavedha.models.lstm_model import LSTMModel
from alphavedha.models.regime import RegimeDetector, RegimeResult
from alphavedha.models.temporal_attention import TemporalAttentionModel
from alphavedha.models.xgboost_model import XGBoostModel

__all__ = [
    "BaseModel",
    "ConformalPredictor",
    "ConformalResult",
    "LSTMModel",
    "ModelArtifact",
    "PredictionResult",
    "RegimeDetector",
    "RegimeResult",
    "TemporalAttentionModel",
    "TrainResult",
    "XGBoostModel",
]
```

- [ ] **Step 2: Verify imports work**

Run: `/home/lenovo/alphavedha/.venv/bin/pytest tests/unit/models/ -v --tb=short 2>&1 | tail -5`

Expected: All model tests pass (existing + new)

- [ ] **Step 3: Commit**

```bash
git add alphavedha/models/__init__.py
git commit -m "feat: export RegimeDetector, ConformalPredictor from models package"
```

---

### Task 7: Full Test Suite Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all new Week 5 model tests**

Run: `/home/lenovo/alphavedha/.venv/bin/pytest tests/unit/models/test_regime.py tests/unit/models/test_conformal.py -v 2>&1 | tail -30`

Expected: ~26 tests passed (14 + 12)

- [ ] **Step 2: Run full project test suite**

Run: `/home/lenovo/alphavedha/.venv/bin/pytest tests/ -v --tb=short 2>&1 | tail -10`

Expected: All 254 tests pass (228 existing + 26 new). No regressions.

- [ ] **Step 3: Run linter on all new files**

Run: `/home/lenovo/alphavedha/.venv/bin/ruff check alphavedha/models/regime.py alphavedha/models/conformal.py && /home/lenovo/alphavedha/.venv/bin/ruff format --check alphavedha/models/regime.py alphavedha/models/conformal.py`

Expected: No errors

- [ ] **Step 4: Run mypy on new files**

Run: `/home/lenovo/alphavedha/.venv/bin/mypy alphavedha/models/regime.py alphavedha/models/conformal.py --ignore-missing-imports 2>&1 | tail -5`

Expected: No errors (or only hmmlearn/mapie stub warnings which are in mypy overrides)

- [ ] **Step 5: Final commit if any lint/type fixes needed**

```bash
# Only if fixes were needed
git add alphavedha/models/regime.py alphavedha/models/conformal.py
git commit -m "fix: resolve lint and mypy issues in Week 5 model files"
```
