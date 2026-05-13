"""Tests for derivatives feature computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphavedha.features.derivatives import (
    DERIVATIVES_FEATURE_COUNT,
    compute_derivatives_features,
    implied_volatility,
)


def _make_deriv_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create mock derivatives data."""
    rng = np.random.default_rng(42)
    n = len(index)
    return pd.DataFrame(
        {
            "futures_oi": rng.integers(1_000_000, 5_000_000, n),
            "futures_price": 3800 + rng.normal(0, 50, n).cumsum(),
            "options_data_json": [
                {
                    "chain": [
                        {
                            "strike": 3800 + i * 50,
                            "call_oi": int(rng.integers(1000, 50000)),
                            "put_oi": int(rng.integers(1000, 50000)),
                            "call_vol": int(rng.integers(100, 5000)),
                            "put_vol": int(rng.integers(100, 5000)),
                            "call_iv": float(rng.uniform(0.15, 0.35)),
                        }
                        for i in range(-3, 4)
                    ]
                }
                for _ in range(n)
            ],
        },
        index=index,
    )


class TestImpliedVolatility:
    def test_known_value(self) -> None:
        iv = implied_volatility(market_price=100, s=3800, k=3800, t=30 / 365)
        assert 0.05 < iv < 2.0

    def test_returns_nan_zero_price(self) -> None:
        iv = implied_volatility(market_price=0, s=3800, k=3800, t=30 / 365)
        assert np.isnan(iv)

    def test_returns_nan_zero_time(self) -> None:
        iv = implied_volatility(market_price=100, s=3800, k=3800, t=0)
        assert np.isnan(iv)


class TestDerivativesFeatures:
    def test_returns_correct_count(self, sample_ohlcv_long: pd.DataFrame) -> None:
        deriv_df = _make_deriv_df(sample_ohlcv_long.index)
        result = compute_derivatives_features(sample_ohlcv_long, deriv_df)
        assert len(result.columns) == DERIVATIVES_FEATURE_COUNT

    def test_graceful_no_data(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_derivatives_features(sample_ohlcv_long)
        assert len(result.columns) == DERIVATIVES_FEATURE_COUNT

    def test_oi_buildup_binary(self, sample_ohlcv_long: pd.DataFrame) -> None:
        deriv_df = _make_deriv_df(sample_ohlcv_long.index)
        result = compute_derivatives_features(sample_ohlcv_long, deriv_df)
        for col in (
            "deriv_oi_buildup",
            "deriv_oi_unwind",
            "deriv_short_cover",
            "deriv_short_build",
        ):
            assert result[col].dropna().isin([0, 1]).all(), f"{col} not binary"

    def test_pcr_positive(self, sample_ohlcv_long: pd.DataFrame) -> None:
        deriv_df = _make_deriv_df(sample_ohlcv_long.index)
        result = compute_derivatives_features(sample_ohlcv_long, deriv_df)
        pcr = result["deriv_pcr_oi"].dropna()
        assert (pcr > 0).all()

    def test_iv_rank_bounded(self, sample_ohlcv_long: pd.DataFrame) -> None:
        deriv_df = _make_deriv_df(sample_ohlcv_long.index)
        result = compute_derivatives_features(sample_ohlcv_long, deriv_df)
        iv_rank = result["deriv_iv_rank"].dropna()
        assert iv_rank.between(0, 1).all()
