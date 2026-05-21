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
    returns = pd.Series(np.concatenate([returns_bull, returns_bear]), name="log_returns")

    vol_bull = rng.normal(0.005, 0.001, size=n_half).clip(0.001)
    vol_bear = rng.normal(0.020, 0.005, size=n_half).clip(0.001)
    volatility = pd.Series(np.concatenate([vol_bull, vol_bear]), name="volatility")

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
        np.testing.assert_array_equal(result_before.regime_history, result_after.regime_history)
