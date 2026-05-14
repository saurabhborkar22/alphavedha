"""Tests for Combinatorial Purged Cross-Validation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from alphavedha.backtest.cpcv import CPCVResult, PathResult, generate_cpcv_splits, run_cpcv
from alphavedha.config import AcceptanceConfig, CPCVConfig, XGBoostConfig
from alphavedha.models.xgboost_model import XGBoostModel


@pytest.fixture
def default_cpcv_config() -> CPCVConfig:
    return CPCVConfig()


@pytest.fixture
def default_acceptance() -> AcceptanceConfig:
    return AcceptanceConfig()


class TestGenerateSplits:
    def test_generates_15_paths(self, default_cpcv_config: CPCVConfig) -> None:
        n_samples = 500
        splits = generate_cpcv_splits(n_samples, default_cpcv_config)
        assert len(splits) == 15

    def test_each_split_has_train_and_test(self, default_cpcv_config: CPCVConfig) -> None:
        splits = generate_cpcv_splits(500, default_cpcv_config)
        for train_idx, test_idx, test_segs in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(test_segs) == 2

    def test_no_train_test_overlap(self, default_cpcv_config: CPCVConfig) -> None:
        splits = generate_cpcv_splits(500, default_cpcv_config)
        for train_idx, test_idx, _ in splits:
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_purge_gap_exists(self, default_cpcv_config: CPCVConfig) -> None:
        """Training indices near test boundaries should be removed."""
        splits = generate_cpcv_splits(500, default_cpcv_config)
        for train_idx, test_idx, _ in splits:
            train_set = set(train_idx)
            test_min = min(test_idx)
            purge = default_cpcv_config.purge_days
            for i in range(max(0, test_min - purge), test_min):
                assert i not in train_set

    def test_embargo_gap_exists(self, default_cpcv_config: CPCVConfig) -> None:
        """Training indices just after test segments should be removed."""
        splits = generate_cpcv_splits(600, default_cpcv_config)
        seg_size = 600 // default_cpcv_config.n_segments
        for train_idx, _test_idx, test_segs in splits:
            train_set = set(train_idx)
            embargo = default_cpcv_config.embargo_days
            for seg in test_segs:
                seg_end = min((seg + 1) * seg_size, 600)
                for i in range(seg_end, min(seg_end + embargo, 600)):
                    assert i not in train_set

    def test_custom_config(self) -> None:
        config = CPCVConfig(n_segments=4, k_test_segments=1)
        splits = generate_cpcv_splits(400, config)
        expected_paths = len(list(combinations(range(4), 1)))
        assert len(splits) == expected_paths


class TestRunCPCV:
    def test_returns_cpcv_result(
        self,
        sample_ohlcv_500: pd.DataFrame,
        sample_features_500: pd.DataFrame,
        default_cpcv_config: CPCVConfig,
        default_acceptance: AcceptanceConfig,
    ) -> None:
        rng = np.random.default_rng(42)
        n = len(sample_features_500)
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=sample_features_500.index)
        returns = pd.Series(rng.normal(0, 0.02, n), index=sample_features_500.index)

        def model_factory() -> XGBoostModel:
            config = XGBoostConfig()
            config.params.n_estimators = 10
            config.params.early_stopping_rounds = 5
            return XGBoostModel(config=config)

        result = run_cpcv(
            X=sample_features_500,
            y=y,
            returns=returns,
            sample_weight=None,
            model_factory=model_factory,
            config=default_cpcv_config,
            acceptance=default_acceptance,
        )

        assert isinstance(result, CPCVResult)
        assert result.n_paths == 15
        assert len(result.path_results) == 15

    def test_path_result_has_metrics(
        self,
        sample_ohlcv_500: pd.DataFrame,
        sample_features_500: pd.DataFrame,
        default_cpcv_config: CPCVConfig,
        default_acceptance: AcceptanceConfig,
    ) -> None:
        rng = np.random.default_rng(42)
        n = len(sample_features_500)
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=sample_features_500.index)
        returns = pd.Series(rng.normal(0, 0.02, n), index=sample_features_500.index)

        def model_factory() -> XGBoostModel:
            config = XGBoostConfig()
            config.params.n_estimators = 10
            config.params.early_stopping_rounds = 5
            return XGBoostModel(config=config)

        result = run_cpcv(
            X=sample_features_500, y=y, returns=returns,
            sample_weight=None, model_factory=model_factory,
            config=default_cpcv_config, acceptance=default_acceptance,
        )

        for pr in result.path_results:
            assert isinstance(pr, PathResult)
            assert 0 <= pr.accuracy <= 1
            assert pr.n_test_samples > 0

    def test_passed_flag(
        self,
        sample_features_500: pd.DataFrame,
        default_cpcv_config: CPCVConfig,
    ) -> None:
        rng = np.random.default_rng(42)
        n = len(sample_features_500)
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=sample_features_500.index)
        returns = pd.Series(rng.normal(0, 0.02, n), index=sample_features_500.index)

        easy_accept = AcceptanceConfig(min_median_sharpe=-999, min_worst_sharpe=-999)

        def model_factory() -> XGBoostModel:
            config = XGBoostConfig()
            config.params.n_estimators = 10
            config.params.early_stopping_rounds = 5
            return XGBoostModel(config=config)

        result = run_cpcv(
            X=sample_features_500, y=y, returns=returns,
            sample_weight=None, model_factory=model_factory,
            config=default_cpcv_config, acceptance=easy_accept,
        )
        assert result.passed is True
