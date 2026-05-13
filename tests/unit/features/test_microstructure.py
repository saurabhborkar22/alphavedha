"""Tests for microstructure feature computation."""

from __future__ import annotations

import pandas as pd

from alphavedha.features.microstructure import (
    MICROSTRUCTURE_FEATURE_COUNT,
    compute_microstructure_features,
)


class TestMicrostructureFeatures:
    def test_returns_correct_count(
        self,
        sample_ohlcv_with_delivery: pd.DataFrame,
    ) -> None:
        result = compute_microstructure_features(sample_ohlcv_with_delivery)
        assert len(result.columns) == MICROSTRUCTURE_FEATURE_COUNT

    def test_delivery_zscore_reasonable(
        self,
        sample_ohlcv_with_delivery: pd.DataFrame,
    ) -> None:
        result = compute_microstructure_features(sample_ohlcv_with_delivery)
        zscore = result["micro_delivery_zscore"].dropna()
        assert zscore.between(-5, 5).all(), "Z-score outside reasonable range"

    def test_binary_flags(
        self,
        sample_ohlcv_with_delivery: pd.DataFrame,
    ) -> None:
        result = compute_microstructure_features(sample_ohlcv_with_delivery)
        for col in ("micro_hd_up", "micro_hd_down", "micro_ld_up"):
            assert result[col].isin([0, 1]).all(), f"{col} not binary"

    def test_graceful_without_delivery(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_microstructure_features(sample_ohlcv_long)
        assert len(result.columns) == MICROSTRUCTURE_FEATURE_COUNT
        assert (result["micro_delivery_pct"] == 0.0).all()

    def test_vol_anomaly_positive(
        self,
        sample_ohlcv_with_delivery: pd.DataFrame,
    ) -> None:
        result = compute_microstructure_features(sample_ohlcv_with_delivery)
        anomaly = result["micro_vol_anomaly"].dropna()
        assert (anomaly > 0).all()

    def test_delivery_rolling_10d(
        self,
        sample_ohlcv_with_delivery: pd.DataFrame,
    ) -> None:
        result = compute_microstructure_features(sample_ohlcv_with_delivery)
        rolling = result["micro_delivery_rolling_10d"].dropna()
        assert rolling.between(0, 1).all()
