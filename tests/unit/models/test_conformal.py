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
