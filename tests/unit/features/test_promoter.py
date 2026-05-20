"""Tests for promoter/insider features in fundamental_features.py."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from alphavedha.features.fundamental_features import (
    FUNDAMENTAL_FEATURE_COUNT,
    compute_fundamental_features,
)


def _make_ohlcv(n_days: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))
    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": rng.integers(100000, 1000000, n_days),
        },
        index=dates,
    )


def _make_promoter_df() -> pd.DataFrame:
    return pd.DataFrame({
        "quarter_end": [
            date(2023, 9, 30),
            date(2023, 12, 31),
            date(2024, 3, 31),
        ],
        "promoter_pct": [52.0, 51.5, 51.0],
        "pledge_pct": [5.0, 8.0, 12.0],
        "public_pct": [20.0, 20.5, 21.0],
        "fii_pct": [18.0, 18.0, 18.0],
        "dii_pct": [10.0, 10.0, 10.0],
    })


def _make_insider_df() -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": [
            date(2024, 2, 15),
            date(2024, 2, 20),
            date(2024, 3, 5),
        ],
        "trade_type": ["buy", "sell", "buy"],
        "value_lakhs": [50.0, 20.0, 30.0],
    })


class TestPromoterFeatures:
    def test_feature_count_is_nine(self) -> None:
        assert FUNDAMENTAL_FEATURE_COUNT == 9

    def test_promoter_features_computed(self) -> None:
        ohlcv = _make_ohlcv()
        promoter = _make_promoter_df()
        result = compute_fundamental_features(ohlcv, promoter_df=promoter)
        assert "fund_promoter_pledge_pct" in result.columns
        assert "fund_pledge_change_qoq" in result.columns
        assert len(result) == len(ohlcv)

    def test_pledge_pct_matches_latest_quarter(self) -> None:
        ohlcv = _make_ohlcv()
        promoter = _make_promoter_df()
        result = compute_fundamental_features(ohlcv, promoter_df=promoter)
        late_rows = result.loc[result.index >= "2024-04-01"]
        if not late_rows.empty:
            assert late_rows["fund_promoter_pledge_pct"].iloc[0] == 12.0

    def test_pledge_change_calculated(self) -> None:
        ohlcv = _make_ohlcv()
        promoter = _make_promoter_df()
        result = compute_fundamental_features(ohlcv, promoter_df=promoter)
        late = result.loc[result.index >= "2024-04-01"]
        if not late.empty:
            change = late["fund_pledge_change_qoq"].iloc[0]
            assert change == pytest.approx(4.0, abs=0.01)

    def test_insider_buy_sell_ratio(self) -> None:
        ohlcv = _make_ohlcv()
        promoter = _make_promoter_df()
        insider = _make_insider_df()
        result = compute_fundamental_features(
            ohlcv, promoter_df=promoter, insider_df=insider,
        )
        march_rows = result.loc["2024-03-10":"2024-03-20"]
        if not march_rows.empty:
            ratio = march_rows["fund_insider_buy_sell_ratio"].iloc[0]
            assert 0 <= ratio <= 1

    def test_promoter_buying_30d_binary(self) -> None:
        ohlcv = _make_ohlcv()
        promoter = _make_promoter_df()
        insider = _make_insider_df()
        result = compute_fundamental_features(
            ohlcv, promoter_df=promoter, insider_df=insider,
        )
        valid = result["fund_promoter_buying_30d"].dropna()
        if not valid.empty:
            assert set(valid.unique()).issubset({0, 1})

    def test_no_lookahead_promoter(self) -> None:
        """Promoter data from Q3 2024 should not appear before that quarter ends."""
        ohlcv = _make_ohlcv()
        promoter = pd.DataFrame({
            "quarter_end": [date(2024, 9, 30)],
            "promoter_pct": [50.0],
            "pledge_pct": [10.0],
            "public_pct": [25.0],
            "fii_pct": [15.0],
            "dii_pct": [10.0],
        })
        result = compute_fundamental_features(ohlcv, promoter_df=promoter)
        before_q3 = result.loc[result.index < "2024-09-30"]
        assert before_q3["fund_promoter_pledge_pct"].isna().all()

    def test_graceful_without_promoter_data(self) -> None:
        ohlcv = _make_ohlcv()
        result = compute_fundamental_features(ohlcv)
        assert "fund_promoter_pledge_pct" in result.columns
        assert result["fund_promoter_pledge_pct"].isna().all()
