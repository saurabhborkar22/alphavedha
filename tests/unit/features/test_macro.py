"""Tests for macro feature computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphavedha.features.macro import MACRO_FEATURE_COUNT, compute_macro_features


def _make_macro_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create mock macro data aligned to given index."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "vix": rng.uniform(12, 25, len(index)),
            "nifty": 22000 + rng.normal(0, 100, len(index)).cumsum(),
            "usdinr": 83 + rng.normal(0, 0.2, len(index)).cumsum(),
            "brent": 80 + rng.normal(0, 1, len(index)).cumsum(),
            "gold": 2300 + rng.normal(0, 10, len(index)).cumsum(),
            "us10y": 4.3 + rng.normal(0, 0.01, len(index)).cumsum(),
        },
        index=index,
    )


def _make_fii_dii_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "fii_net": rng.normal(500, 2000, len(index)),
            "dii_net": rng.normal(300, 1500, len(index)),
        },
        index=index,
    )


class TestMacroFeatures:
    def test_returns_correct_count_with_data(
        self,
        sample_ohlcv_long: pd.DataFrame,
    ) -> None:
        macro_df = _make_macro_df(sample_ohlcv_long.index)
        fii_df = _make_fii_dii_df(sample_ohlcv_long.index)
        result = compute_macro_features(sample_ohlcv_long, macro_df, fii_df)
        assert len(result.columns) == MACRO_FEATURE_COUNT

    def test_returns_correct_count_no_data(
        self,
        sample_ohlcv_long: pd.DataFrame,
    ) -> None:
        result = compute_macro_features(sample_ohlcv_long)
        assert len(result.columns) == MACRO_FEATURE_COUNT

    def test_vix_present(self, sample_ohlcv_long: pd.DataFrame) -> None:
        macro_df = _make_macro_df(sample_ohlcv_long.index)
        result = compute_macro_features(sample_ohlcv_long, macro_df)
        assert result["macro_vix"].notna().any()

    def test_fii_cumulative(self, sample_ohlcv_long: pd.DataFrame) -> None:
        fii_df = _make_fii_dii_df(sample_ohlcv_long.index)
        result = compute_macro_features(sample_ohlcv_long, fii_dii_df=fii_df)
        assert result["macro_fii_cum_5d"].notna().any()

    def test_sector_relative(self, sample_ohlcv_long: pd.DataFrame) -> None:
        sector_df = sample_ohlcv_long[["close"]].copy()
        sector_df["close"] = sector_df["close"] * 1.1
        result = compute_macro_features(
            sample_ohlcv_long,
            sector_df=sector_df,
        )
        assert result["macro_sector_rel_ret_1d"].notna().any()

    def test_graceful_empty_data(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_macro_features(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
