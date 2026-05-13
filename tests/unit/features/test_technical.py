"""Tests for technical feature computation."""

from __future__ import annotations

import pandas as pd

from alphavedha.features.technical import TECHNICAL_FEATURE_COUNT, compute_technical_features


class TestTechnicalFeatures:
    def test_returns_correct_count(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_technical_features(sample_ohlcv_long)
        assert len(result.columns) == TECHNICAL_FEATURE_COUNT

    def test_rsi_bounded(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_technical_features(sample_ohlcv_long)
        for col in ("rsi_7", "rsi_14", "rsi_21"):
            values = result[col].dropna()
            assert values.between(0, 100).all(), f"{col} out of bounds"

    def test_bollinger_upper_above_lower(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_technical_features(sample_ohlcv_long)
        valid = result[["bb_upper_20", "bb_lower_20"]].dropna()
        assert (valid["bb_upper_20"] >= valid["bb_lower_20"]).all()

    def test_atr_positive(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_technical_features(sample_ohlcv_long)
        atr = result["atr_14"].iloc[14:]
        assert (atr > 0).all()

    def test_volume_ratio_positive(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_technical_features(sample_ohlcv_long)
        ratio = result["vol_ratio_20"].dropna()
        assert (ratio > 0).all()

    def test_mfi_bounded(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_technical_features(sample_ohlcv_long)
        mfi = result["mfi_14"].dropna()
        assert mfi.between(0, 100).all()

    def test_adx_bounded(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_technical_features(sample_ohlcv_long)
        adx = result["adx_14"].dropna()
        assert adx.between(0, 100).all()

    def test_no_lookahead(self, sample_ohlcv_long: pd.DataFrame) -> None:
        """Features at row t should not change when future rows are removed."""
        full = compute_technical_features(sample_ohlcv_long)
        partial = compute_technical_features(sample_ohlcv_long.iloc[:100])
        shared_cols = full.columns
        for col in shared_cols:
            full_val = full[col].iloc[99]
            partial_val = partial[col].iloc[99]
            if pd.notna(full_val) and pd.notna(partial_val):
                assert abs(full_val - partial_val) < 1e-10, f"Look-ahead in {col}"

    def test_handles_short_data(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_technical_features(sample_ohlcv)
        assert len(result.columns) == TECHNICAL_FEATURE_COUNT
        assert len(result) == len(sample_ohlcv)

    def test_stochastic_bounded(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_technical_features(sample_ohlcv_long)
        for col in ("stoch_k_14", "stoch_d_14"):
            values = result[col].dropna()
            assert values.between(0, 100).all(), f"{col} out of bounds"

    def test_hvol_positive(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_technical_features(sample_ohlcv_long)
        for col in ("hvol_20", "hvol_60"):
            values = result[col].dropna()
            assert (values >= 0).all(), f"{col} has negative values"

    def test_price_to_sma_ratio(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_technical_features(sample_ohlcv_long)
        ratio = result["price_to_sma_20"].dropna()
        assert ratio.between(0.5, 2.0).all(), "price_to_sma_20 ratio unreasonable"
