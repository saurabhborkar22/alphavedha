"""Tests for feature pipeline orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphavedha.features.pipeline import EXPECTED_FEATURE_COUNT, compute_all_features


class TestFeaturePipeline:
    def test_produces_expected_columns(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_all_features("TCS.NS", sample_ohlcv_long)
        assert result.feature_count == EXPECTED_FEATURE_COUNT

    def test_no_nan_after_fill(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_all_features("TCS.NS", sample_ohlcv_long)
        assert result.df.isna().sum().sum() == 0

    def test_no_inf(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_all_features("TCS.NS", sample_ohlcv_long)
        assert not np.isinf(result.df.values).any()

    def test_timing_recorded(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_all_features("TCS.NS", sample_ohlcv_long)
        assert result.computation_time_ms > 0

    def test_symbol_preserved(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_all_features("INFY.NS", sample_ohlcv_long)
        assert result.symbol == "INFY.NS"

    def test_short_data(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_all_features("TCS.NS", sample_ohlcv)
        assert result.df.isna().sum().sum() == 0
        assert len(result.df) == len(sample_ohlcv)
