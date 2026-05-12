"""Unit tests for preprocessing modules — corporate actions, circuit, missing data, outliers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from alphavedha.data.preprocessing.circuit_handler import detect_circuit_hits
from alphavedha.data.preprocessing.corporate_actions import (
    CorporateActionRecord,
    adjust_ohlcv,
    compute_adjustment_factors,
    detect_potential_splits,
)
from alphavedha.data.preprocessing.fractional_diff import (
    find_min_d,
    frac_diff_ffd,
)
from alphavedha.data.preprocessing.missing_data import (
    detect_suspensions,
    handle_missing_data,
)
from alphavedha.data.preprocessing.outlier_treatment import (
    winsorize_features,
    winsorize_series,
)


class TestCorporateActions:
    def test_split_adjustment_halves_pre_split_prices(self, sample_ohlcv: pd.DataFrame):
        actions = [
            CorporateActionRecord(
                symbol="TCS",
                ex_date=date(2024, 1, 12),
                action_type="split",
                ratio=2.0,
            )
        ]
        result = adjust_ohlcv(sample_ohlcv, actions)

        split_date = pd.Timestamp("2024-01-12")
        pre_split = result.index < split_date
        post_split = result.index >= split_date

        assert result.loc[pre_split, "close"].mean() < sample_ohlcv.loc[pre_split, "close"].mean()
        np.testing.assert_array_almost_equal(
            result.loc[post_split, "close"].values,
            sample_ohlcv.loc[post_split, "close"].values,
        )

    def test_no_actions_returns_copy(self, sample_ohlcv: pd.DataFrame):
        result = adjust_ohlcv(sample_ohlcv, [])
        pd.testing.assert_frame_equal(result, sample_ohlcv)

    def test_adjustment_preserves_raw_columns(self, sample_ohlcv: pd.DataFrame):
        actions = [
            CorporateActionRecord(
                symbol="TCS", ex_date=date(2024, 1, 15), action_type="split", ratio=2.0
            )
        ]
        result = adjust_ohlcv(sample_ohlcv, actions)
        assert "raw_close" in result.columns
        assert "raw_open" in result.columns
        assert result["is_adjusted"].all()

    def test_volume_inverse_adjusted(self, sample_ohlcv: pd.DataFrame):
        actions = [
            CorporateActionRecord(
                symbol="TCS", ex_date=date(2024, 1, 15), action_type="split", ratio=2.0
            )
        ]
        result = adjust_ohlcv(sample_ohlcv, actions)
        pre_split = result.index < pd.Timestamp("2024-01-15")
        assert result.loc[pre_split, "volume"].mean() > sample_ohlcv.loc[pre_split, "volume"].mean()

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame()
        result = adjust_ohlcv(empty, [])
        assert result.empty

    def test_detect_potential_splits(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        df.iloc[10, df.columns.get_loc("close")] = df["close"].iloc[9] * 0.5
        splits = detect_potential_splits(df)
        assert len(splits) >= 1

    def test_compute_adjustment_factors_shape(self, sample_ohlcv: pd.DataFrame):
        actions = [
            CorporateActionRecord(
                symbol="TCS", ex_date=date(2024, 1, 15), action_type="split", ratio=2.0
            )
        ]
        factors = compute_adjustment_factors(actions, sample_ohlcv.index)
        assert len(factors) == len(sample_ohlcv)
        assert (factors > 0).all()


class TestCircuitHandler:
    def test_detects_upper_circuit(self, sample_ohlcv_with_circuit: pd.DataFrame):
        result = detect_circuit_hits(sample_ohlcv_with_circuit, thresholds=[0.05, 0.10, 0.20])
        assert "circuit_hit" in result.columns
        upper_hits = result[result["circuit_hit"] == "upper"]
        assert len(upper_hits) >= 1

    def test_no_circuit_on_normal_data(self, sample_ohlcv: pd.DataFrame):
        result = detect_circuit_hits(sample_ohlcv, thresholds=[0.05, 0.10, 0.20])
        n_hits = result["circuit_hit"].notna().sum()
        assert n_hits < len(result)

    def test_empty_df(self):
        result = detect_circuit_hits(pd.DataFrame(), thresholds=[0.05])
        assert "circuit_hit" in result.columns

    def test_circuit_band_column_added(self, sample_ohlcv_with_circuit: pd.DataFrame):
        result = detect_circuit_hits(sample_ohlcv_with_circuit, thresholds=[0.05])
        assert "circuit_band" in result.columns


class TestMissingData:
    def test_fills_gaps_with_forward_fill(self, sample_ohlcv_with_gaps: pd.DataFrame):
        original_len = len(sample_ohlcv_with_gaps)
        result = handle_missing_data(sample_ohlcv_with_gaps, max_gap_days=10)
        assert len(result) >= original_len
        assert result["close"].notna().all()

    def test_filled_rows_flagged(self, sample_ohlcv_with_gaps: pd.DataFrame):
        result = handle_missing_data(sample_ohlcv_with_gaps, max_gap_days=10)
        assert "is_filled" in result.columns
        assert result["is_filled"].any()

    def test_filled_volume_is_zero(self, sample_ohlcv_with_gaps: pd.DataFrame):
        result = handle_missing_data(sample_ohlcv_with_gaps, max_gap_days=10)
        filled_rows = result[result["is_filled"]]
        if not filled_rows.empty:
            assert (filled_rows["volume"] == 0).all()

    def test_no_gaps_returns_same(self, sample_ohlcv: pd.DataFrame):
        result = handle_missing_data(sample_ohlcv, max_gap_days=10)
        assert len(result) >= len(sample_ohlcv)

    def test_detect_suspensions(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        df.iloc[5:12, df.columns.get_loc("volume")] = 0
        suspensions = detect_suspensions(df, min_zero_volume_days=5)
        assert suspensions.any()

    def test_empty_df(self):
        result = handle_missing_data(pd.DataFrame())
        assert result.empty


class TestFractionalDiff:
    def test_frac_diff_produces_values(self, sample_ohlcv_long: pd.DataFrame):
        result = frac_diff_ffd(sample_ohlcv_long["close"], d=0.4, max_lags=50)
        valid = result.dropna()
        assert len(valid) > 0

    def test_d_zero_returns_original(self, sample_ohlcv_long: pd.DataFrame):
        result = frac_diff_ffd(sample_ohlcv_long["close"], d=0.0, max_lags=50)
        valid = result.dropna()
        np.testing.assert_array_almost_equal(
            valid.values,
            sample_ohlcv_long["close"].loc[valid.index].values,
            decimal=5,
        )

    def test_d_one_approximates_diff(self, sample_ohlcv_long: pd.DataFrame):
        result = frac_diff_ffd(sample_ohlcv_long["close"], d=1.0, max_lags=50)
        regular_diff = sample_ohlcv_long["close"].diff()
        valid_idx = result.dropna().index.intersection(regular_diff.dropna().index)
        if len(valid_idx) > 10:
            corr = result.loc[valid_idx].corr(regular_diff.loc[valid_idx])
            assert corr > 0.9

    def test_find_min_d_returns_valid_range(self, sample_ohlcv_long: pd.DataFrame):
        d = find_min_d(
            sample_ohlcv_long["close"],
            max_lags=50,
            adf_pvalue=0.05,
            d_range=(0.1, 0.8),
        )
        assert 0.1 <= d <= 0.8


class TestOutlierTreatment:
    def test_winsorize_clips_extremes(self):
        s = pd.Series([1, 2, 3, 4, 5, 100])
        result, n_outliers = winsorize_series(s, lower_pct=0.05, upper_pct=0.95)
        assert result.max() < 100
        assert n_outliers > 0

    def test_winsorize_features_skips_prices(self, sample_ohlcv: pd.DataFrame):
        df = sample_ohlcv.copy()
        df["some_feature"] = np.random.default_rng(42).normal(0, 10, len(df))
        df.iloc[0, df.columns.get_loc("some_feature")] = 1000  # extreme outlier

        result, _counts = winsorize_features(df, lower_pct=0.01, upper_pct=0.99)
        pd.testing.assert_series_equal(result["close"], df["close"])
        assert result["some_feature"].max() < 1000

    def test_empty_df(self):
        result, _counts = winsorize_features(pd.DataFrame(), lower_pct=0.01, upper_pct=0.99)
        assert result.empty
        assert not _counts

    def test_empty_series(self):
        result, count = winsorize_series(pd.Series(dtype=float))
        assert result.empty
        assert count == 0
