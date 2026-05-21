"""Tests for fundamental (earnings) feature computation."""

from __future__ import annotations

from datetime import date

import pandas as pd

from alphavedha.features.fundamental_features import (
    FUNDAMENTAL_FEATURE_COUNT,
    compute_fundamental_features,
)


def _make_earnings_df() -> pd.DataFrame:
    """Create sample earnings data for testing."""
    return pd.DataFrame(
        [
            {
                "symbol": "TCS",
                "quarter": 4,
                "year": 2023,
                "revenue_actual": 59162.0,
                "profit_actual": 11392.0,
                "expenses": 47770.0,
                "announced_date": date(2024, 4, 12),
                "revenue_estimate": None,
                "profit_estimate": None,
            },
            {
                "symbol": "TCS",
                "quarter": 1,
                "year": 2024,
                "revenue_actual": 62613.0,
                "profit_actual": 12040.0,
                "expenses": 50573.0,
                "announced_date": date(2024, 7, 11),
                "revenue_estimate": None,
                "profit_estimate": None,
            },
            {
                "symbol": "TCS",
                "quarter": 2,
                "year": 2024,
                "revenue_actual": 64259.0,
                "profit_actual": 11909.0,
                "expenses": 52350.0,
                "announced_date": date(2024, 10, 10),
                "revenue_estimate": None,
                "profit_estimate": None,
            },
            {
                "symbol": "TCS",
                "quarter": 3,
                "year": 2024,
                "revenue_actual": 63973.0,
                "profit_actual": 12380.0,
                "expenses": 51593.0,
                "announced_date": date(2025, 1, 9),
                "revenue_estimate": None,
                "profit_estimate": None,
            },
            {
                "symbol": "TCS",
                "quarter": 4,
                "year": 2024,
                "revenue_actual": 64479.0,
                "profit_actual": 12224.0,
                "expenses": 52255.0,
                "announced_date": date(2025, 4, 10),
                "revenue_estimate": 63500.0,
                "profit_estimate": 11800.0,
            },
        ]
    )


class TestFundamentalFeatures:
    def test_returns_correct_count(
        self,
        sample_ohlcv_long: pd.DataFrame,
    ) -> None:
        earnings = _make_earnings_df()
        result = compute_fundamental_features(sample_ohlcv_long, earnings)
        assert len(result.columns) == FUNDAMENTAL_FEATURE_COUNT

    def test_graceful_without_earnings(
        self,
        sample_ohlcv_long: pd.DataFrame,
    ) -> None:
        result = compute_fundamental_features(sample_ohlcv_long, None)
        assert len(result.columns) == FUNDAMENTAL_FEATURE_COUNT
        assert result["fund_earnings_surprise_pct"].isna().all()

    def test_graceful_empty_earnings(
        self,
        sample_ohlcv_long: pd.DataFrame,
    ) -> None:
        result = compute_fundamental_features(sample_ohlcv_long, pd.DataFrame())
        assert len(result.columns) == FUNDAMENTAL_FEATURE_COUNT

    def test_days_since_earnings_nonneg(
        self,
        sample_ohlcv_long: pd.DataFrame,
    ) -> None:
        earnings = _make_earnings_df()
        result = compute_fundamental_features(sample_ohlcv_long, earnings)
        days = result["fund_days_since_earnings"].dropna()
        assert (days >= 0).all(), "Days since earnings should be non-negative"

    def test_surprise_streak_integer(
        self,
        sample_ohlcv_long: pd.DataFrame,
    ) -> None:
        earnings = _make_earnings_df()
        result = compute_fundamental_features(sample_ohlcv_long, earnings)
        streak = result["fund_earnings_surprise_streak"].dropna()
        if not streak.empty:
            assert (streak == streak.astype(int)).all()

    def test_no_lookahead(
        self,
        sample_ohlcv_long: pd.DataFrame,
    ) -> None:
        earnings = _make_earnings_df()
        result = compute_fundamental_features(sample_ohlcv_long, earnings)
        first_announce = pd.Timestamp(date(2024, 4, 12))
        before_any = result.loc[result.index < first_announce, "fund_earnings_surprise_pct"]
        assert before_any.isna().all(), "No earnings data should leak before first announcement"

    def test_with_estimates_uses_surprise(
        self,
        sample_ohlcv_long: pd.DataFrame,
    ) -> None:
        earnings = _make_earnings_df()
        result = compute_fundamental_features(sample_ohlcv_long, earnings)
        last_announce = pd.Timestamp(date(2025, 4, 10))
        after_last = result.loc[result.index >= last_announce, "fund_earnings_surprise_pct"]
        valid = after_last.dropna()
        if not valid.empty:
            assert valid.iloc[0] != 0.0, "Surprise should reflect estimate vs actual"
