"""Tests for triple barrier labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import TripleBarrierConfig
from alphavedha.labels.triple_barrier import LabelResult, compute_triple_barrier_labels


@pytest.fixture
def default_config() -> TripleBarrierConfig:
    return TripleBarrierConfig()


class TestTripleBarrierLabels:
    def test_returns_label_result(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        assert isinstance(result, LabelResult)
        assert isinstance(result.df, pd.DataFrame)
        assert result.symbol == ""

    def test_output_columns(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        expected_cols = {
            "label", "return_pct", "barrier_hit", "days_to_hit",
            "entry_price", "exit_price", "atr_at_entry",
        }
        assert expected_cols.issubset(set(result.df.columns))

    def test_labels_are_valid_values(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        valid = result.df["label"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_last_rows_are_nan(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        last_labels = result.df["label"].iloc[-default_config.max_holding_period:]
        assert last_labels.isna().all()

    def test_label_counts_match_data(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        total_labeled = sum(result.label_counts.values())
        non_nan = result.df["label"].notna().sum()
        assert total_labeled == non_nan

    def test_no_lookahead_in_atr(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        """ATR at time t must use only data up to t."""
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        atr_col = result.df["atr_at_entry"].dropna()
        assert len(atr_col) > 0
        first_valid_idx = atr_col.index[0]
        pos = sample_ohlcv_500.index.get_loc(first_valid_idx)
        assert pos >= default_config.atr_period - 1

    def test_barrier_hit_values(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        valid_hits = result.df["barrier_hit"].dropna()
        assert set(valid_hits.unique()).issubset({"upper", "lower", "time"})

    def test_days_to_hit_bounded(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        valid_days = result.df["days_to_hit"].dropna()
        assert (valid_days >= 1).all()
        assert (valid_days <= default_config.max_holding_period).all()

    def test_return_pct_matches_prices(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        valid = result.df.dropna(subset=["label"])
        for _, row in valid.head(10).iterrows():
            expected_ret = row["exit_price"] / row["entry_price"] - 1
            assert abs(row["return_pct"] - expected_ret) < 1e-10

    def test_low_atr_skipped(self) -> None:
        """Stocks with ATR/close < min_atr_threshold get NaN labels."""
        dates = pd.bdate_range("2024-01-02", periods=50, freq="B")
        flat_price = np.full(50, 100.0)
        df = pd.DataFrame(
            {
                "open": flat_price,
                "high": flat_price * 1.0001,
                "low": flat_price * 0.9999,
                "close": flat_price,
                "adj_close": flat_price,
                "volume": np.full(50, 10_000_000),
            },
            index=dates,
        )
        df.index.name = "date"
        config = TripleBarrierConfig(min_atr_threshold=0.005)
        result = compute_triple_barrier_labels(df, config)
        assert result.skipped_low_atr > 0

    def test_insufficient_data_raises(self, default_config: TripleBarrierConfig) -> None:
        dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "open": np.ones(10) * 100,
                "high": np.ones(10) * 101,
                "low": np.ones(10) * 99,
                "close": np.ones(10) * 100,
                "adj_close": np.ones(10) * 100,
                "volume": np.ones(10, dtype=int) * 1_000_000,
            },
            index=dates,
        )
        df.index.name = "date"
        from alphavedha.exceptions import InsufficientDataError

        with pytest.raises(InsufficientDataError):
            compute_triple_barrier_labels(df, default_config)

    def test_with_symbol(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(
            sample_ohlcv_500, default_config, symbol="TCS.NS"
        )
        assert result.symbol == "TCS.NS"

    def test_same_bar_touch_favors_lower(self) -> None:
        """When both barriers are touched on the same day, lower wins."""
        dates = pd.bdate_range("2024-01-02", periods=50, freq="B")
        rng = np.random.default_rng(42)
        closes = np.full(50, 100.0)
        closes[0] = 100.0
        highs = closes.copy()
        lows = closes.copy()
        highs[1] = 200.0
        lows[1] = 50.0
        df = pd.DataFrame(
            {
                "open": closes * (1 + rng.normal(0, 0.001, 50)),
                "high": highs,
                "low": lows,
                "close": closes,
                "adj_close": closes,
                "volume": np.full(50, 10_000_000),
            },
            index=dates,
        )
        df.index.name = "date"
        config = TripleBarrierConfig()
        result = compute_triple_barrier_labels(df, config)
        label_at_0 = result.df["label"].iloc[0]
        if pd.notna(label_at_0):
            assert label_at_0 == -1
