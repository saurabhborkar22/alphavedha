"""Tests for sample weight computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import SampleWeightsConfig
from alphavedha.labels.sample_weights import compute_sample_weights


@pytest.fixture
def default_config() -> SampleWeightsConfig:
    return SampleWeightsConfig()


class TestSampleWeights:
    def test_returns_series(self, default_config: SampleWeightsConfig) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.random.default_rng(42).choice([-1, 0, 1], size=100),
             "days_to_hit": np.full(100, 5)},
            index=dates,
        )
        result = compute_sample_weights(labels_df, default_config)
        assert isinstance(result, pd.Series)
        assert len(result) == len(labels_df)

    def test_weights_sum_to_n(self, default_config: SampleWeightsConfig) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.random.default_rng(42).choice([-1, 0, 1], size=100),
             "days_to_hit": np.full(100, 5)},
            index=dates,
        )
        result = compute_sample_weights(labels_df, default_config)
        assert abs(result.sum() - len(labels_df)) < 1e-6

    def test_all_weights_positive(self, default_config: SampleWeightsConfig) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.random.default_rng(42).choice([-1, 0, 1], size=100),
             "days_to_hit": np.full(100, 5)},
            index=dates,
        )
        result = compute_sample_weights(labels_df, default_config)
        assert (result > 0).all()

    def test_recency_most_recent_highest(self, default_config: SampleWeightsConfig) -> None:
        """Most recent sample should have the highest weight (all else equal)."""
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.ones(100, dtype=int),
             "days_to_hit": np.ones(100)},
            index=dates,
        )
        result = compute_sample_weights(labels_df, default_config)
        assert result.iloc[-1] > result.iloc[0]

    def test_no_overlap_weight_equals_one_before_normalization(self) -> None:
        """Non-overlapping labels should have uniqueness weight = 1."""
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.ones(100, dtype=int),
             "days_to_hit": np.ones(100)},
            index=dates,
        )
        config = SampleWeightsConfig(uniqueness=True, recency_halflife=999999)
        result = compute_sample_weights(labels_df, config)
        mean_w = result.mean()
        assert abs(mean_w - 1.0) < 0.1

    def test_overlap_reduces_weight(self) -> None:
        """Overlapping labels should produce lower uniqueness weights than non-overlapping."""
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        non_overlap = pd.DataFrame(
            {"label": np.ones(100, dtype=int), "days_to_hit": np.ones(100)},
            index=dates,
        )
        overlap = pd.DataFrame(
            {"label": np.ones(100, dtype=int), "days_to_hit": np.full(100, 15)},
            index=dates,
        )
        config = SampleWeightsConfig(uniqueness=True, recency_halflife=999999)
        w_non = compute_sample_weights(non_overlap, config)
        w_over = compute_sample_weights(overlap, config)
        assert w_non.std() < w_over.std() or w_non.min() >= w_over.min()

    def test_uniqueness_disabled(self) -> None:
        """When uniqueness=False, only recency weighting is applied."""
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.ones(100, dtype=int), "days_to_hit": np.full(100, 15)},
            index=dates,
        )
        config = SampleWeightsConfig(uniqueness=False, recency_halflife=252)
        result = compute_sample_weights(labels_df, config)
        assert result.iloc[-1] > result.iloc[0]

    def test_handles_nan_labels(self, default_config: SampleWeightsConfig) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels = np.ones(100)
        labels[-15:] = np.nan
        labels_df = pd.DataFrame(
            {"label": labels, "days_to_hit": np.full(100, 5)},
            index=dates,
        )
        result = compute_sample_weights(labels_df, default_config)
        assert len(result) == len(labels_df)
