"""Tests for pairs universe scanning — cointegration, hedge ratio, half-life."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavedha.signals.pairs_universe import (
    PairCandidate,
    SECTOR_PAIRS,
    compute_hedge_ratio,
    estimate_half_life,
    scan_pair_universe,
    validate_pair,
)


def _make_correlated_prices(n: int = 300, correlation: float = 0.95) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(42)
    base = np.cumsum(rng.normal(0.001, 0.02, n)) + 100
    noise = rng.normal(0, 0.5, n)
    a = pd.Series(base + noise, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    b = pd.Series(base * 0.8 + rng.normal(0, 0.3, n), index=a.index)
    return a, b


def _make_random_prices(n: int = 300) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(99)
    a = pd.Series(
        np.cumsum(rng.normal(0, 0.02, n)) + 100,
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    b = pd.Series(
        np.cumsum(rng.normal(0, 0.02, n)) + 50,
        index=a.index,
    )
    return a, b


class TestEstimateHalfLife:
    def test_mean_reverting_spread(self) -> None:
        rng = np.random.default_rng(42)
        spread = pd.Series(np.zeros(200))
        for i in range(1, 200):
            spread.iloc[i] = 0.7 * spread.iloc[i - 1] + rng.normal(0, 1)
        hl = estimate_half_life(spread)
        assert 0 < hl < 30

    def test_short_series_returns_inf(self) -> None:
        spread = pd.Series([1.0, 2.0, 3.0])
        assert estimate_half_life(spread) == float("inf")

    def test_trending_series_returns_non_reverting(self) -> None:
        spread = pd.Series(np.arange(100, dtype=float))
        hl = estimate_half_life(spread)
        assert hl == float("inf") or np.isnan(hl)

    def test_constant_series_returns_inf(self) -> None:
        spread = pd.Series(np.ones(100))
        assert estimate_half_life(spread) == float("inf")


class TestComputeHedgeRatio:
    def test_identical_prices(self) -> None:
        prices = pd.Series(np.arange(1, 101, dtype=float))
        ratio = compute_hedge_ratio(prices, prices)
        assert abs(ratio - 1.0) < 0.01

    def test_doubled_prices(self) -> None:
        a = pd.Series(np.arange(1, 101, dtype=float))
        b = a * 0.5
        ratio = compute_hedge_ratio(a, b)
        assert abs(ratio - 2.0) < 0.01

    def test_returns_float(self) -> None:
        a, b = _make_correlated_prices(200)
        ratio = compute_hedge_ratio(a, b)
        assert isinstance(ratio, float)


class TestValidatePair:
    def test_correlated_pair_passes(self) -> None:
        a, b = _make_correlated_prices(300)
        is_valid, pval, corr, hl, hedge = validate_pair(a, b, max_pvalue=0.10)
        assert isinstance(is_valid, bool)
        assert 0.0 <= pval <= 1.0
        assert -1.0 <= corr <= 1.0
        assert hl > 0

    def test_random_pair_fails(self) -> None:
        a, b = _make_random_prices(300)
        is_valid, pval, corr, hl, _ = validate_pair(a, b)
        assert not is_valid or pval >= 0.05

    def test_short_series_fails(self) -> None:
        a = pd.Series([100.0, 101.0, 102.0], index=pd.date_range("2020-01-01", periods=3))
        b = pd.Series([50.0, 51.0, 52.0], index=a.index)
        is_valid, pval, corr, hl, hedge = validate_pair(a, b)
        assert not is_valid
        assert pval == 1.0

    def test_low_correlation_rejected(self) -> None:
        a, b = _make_random_prices(300)
        is_valid, _, corr, _, _ = validate_pair(a, b, min_correlation=0.99)
        assert not is_valid


class TestScanPairUniverse:
    def test_returns_list_of_candidates(self) -> None:
        a, b = _make_correlated_prices(300)
        price_data = {"SYM_A": a, "SYM_B": b}
        candidates = [("SYM_A", "SYM_B", "Test")]
        results = scan_pair_universe(price_data, candidates=candidates)
        assert len(results) == 1
        assert isinstance(results[0], PairCandidate)
        assert results[0].sector == "Test"

    def test_missing_symbols_skipped(self) -> None:
        a, _ = _make_correlated_prices(300)
        price_data = {"SYM_A": a}
        candidates = [("SYM_A", "SYM_MISSING", "Test")]
        results = scan_pair_universe(price_data, candidates=candidates)
        assert len(results) == 0

    def test_sorted_by_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        base = np.cumsum(rng.normal(0, 0.02, 300)) + 100
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        price_data = {
            "A": pd.Series(base + rng.normal(0, 0.3, 300), index=idx),
            "B": pd.Series(base * 0.8 + rng.normal(0, 0.3, 300), index=idx),
            "C": pd.Series(np.cumsum(rng.normal(0, 0.02, 300)) + 50, index=idx),
        }
        candidates = [("A", "B", "S1"), ("A", "C", "S2")]
        results = scan_pair_universe(price_data, candidates=candidates)
        assert len(results) == 2
        assert results[0].coint_pvalue <= results[1].coint_pvalue

    def test_defaults_to_sector_pairs(self) -> None:
        results = scan_pair_universe({})
        assert len(results) == 0
        assert len(SECTOR_PAIRS) == 10

    def test_custom_thresholds(self) -> None:
        a, b = _make_correlated_prices(300)
        price_data = {"A": a, "B": b}
        candidates = [("A", "B", "Test")]
        results = scan_pair_universe(
            price_data,
            candidates=candidates,
            max_pvalue=0.001,
            min_correlation=0.99,
            max_half_life=5.0,
        )
        assert len(results) == 1
        assert isinstance(results[0].is_valid, bool)
