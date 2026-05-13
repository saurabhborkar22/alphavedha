"""Tests for return-derived feature computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphavedha.features.returns import RETURN_FEATURE_COUNT, compute_return_features


class TestReturnFeatures:
    def test_returns_correct_count(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_return_features(sample_ohlcv_long)
        assert len(result.columns) == RETURN_FEATURE_COUNT

    def test_log_return_1d_matches_manual(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_return_features(sample_ohlcv)
        close = sample_ohlcv["close"]
        expected = np.log(close / close.shift(1))
        pd.testing.assert_series_equal(
            result["ret_log_1d"],
            expected,
            check_names=False,
            atol=1e-10,
        )

    def test_log_return_5d(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_return_features(sample_ohlcv_long)
        close = sample_ohlcv_long["close"]
        expected_5d = np.log(close / close.shift(5))
        pd.testing.assert_series_equal(
            result["ret_log_5d"],
            expected_5d,
            check_names=False,
            atol=1e-10,
        )

    def test_rolling_std_positive(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_return_features(sample_ohlcv_long)
        std_20 = result["ret_std_20d"].dropna()
        assert (std_20 >= 0).all()

    def test_up_ratio_bounded(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_return_features(sample_ohlcv_long)
        ratio = result["ret_up_ratio_20d"].dropna()
        assert ratio.between(0, 1).all()

    def test_frac_diff_default_zero(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_return_features(sample_ohlcv)
        assert (result["ret_frac_diff"] == 0.0).all()

    def test_frac_diff_uses_column(self, sample_ohlcv: pd.DataFrame) -> None:
        df = sample_ohlcv.copy()
        df["close_frac_diff"] = np.arange(len(df), dtype=float)
        result = compute_return_features(df, frac_diff_col="close_frac_diff")
        pd.testing.assert_series_equal(
            result["ret_frac_diff"],
            df["close_frac_diff"],
            check_names=False,
        )

    def test_regime_default_sideways(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_return_features(sample_ohlcv)
        assert (result["ret_regime"] == 1).all()

    def test_52w_distances(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_return_features(sample_ohlcv_long)
        high_dist = result["ret_52w_high_dist"].dropna()
        low_dist = result["ret_52w_low_dist"].dropna()
        assert (high_dist <= 0).all(), "Distance to 52w high should be <= 0"
        assert (low_dist >= 0).all(), "Distance to 52w low should be >= 0"
