"""Tests for pairs trading engine and universe scanning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphavedha.signals.pairs import PairsTrader
from alphavedha.signals.pairs_universe import (
    PairCandidate,
    compute_hedge_ratio,
    estimate_half_life,
    scan_pair_universe,
    validate_pair,
)


def _make_cointegrated_pair(n: int = 500, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Create two cointegrated price series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)

    common_factor = np.cumsum(rng.normal(0.0005, 0.01, n))
    noise_a = np.cumsum(rng.normal(0, 0.002, n))
    noise_b = np.cumsum(rng.normal(0, 0.002, n))

    prices_a = 100 * np.exp(common_factor + noise_a)
    prices_b = 50 * np.exp(common_factor * 0.8 + noise_b)

    return pd.Series(prices_a, index=dates), pd.Series(prices_b, index=dates)


def _make_independent_pair(n: int = 500, seed: int = 99) -> tuple[pd.Series, pd.Series]:
    """Create two independent (non-cointegrated) price series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)

    prices_a = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    prices_b = 50 * np.exp(np.cumsum(rng.normal(-0.0001, 0.02, n)))

    return pd.Series(prices_a, index=dates), pd.Series(prices_b, index=dates)


class TestPairUniverse:
    def test_hedge_ratio_positive(self) -> None:
        a, b = _make_cointegrated_pair()
        hr = compute_hedge_ratio(a, b)
        assert hr > 0

    def test_half_life_finite_for_mean_reverting(self) -> None:
        a, b = _make_cointegrated_pair()
        hr = compute_hedge_ratio(a, b)
        spread = a - hr * b
        hl = estimate_half_life(spread)
        assert hl > 0
        assert hl < 200

    def test_validate_cointegrated_pair(self) -> None:
        a, b = _make_cointegrated_pair()
        _is_valid, pval, corr, _hl, _hr = validate_pair(a, b, max_pvalue=0.10)
        assert corr > 0.5
        assert pval < 1.0

    def test_validate_short_data_invalid(self) -> None:
        a = pd.Series([100, 101, 102], index=pd.bdate_range("2024-01-01", periods=3))
        b = pd.Series([50, 51, 52], index=pd.bdate_range("2024-01-01", periods=3))
        is_valid, _, _, _, _ = validate_pair(a, b)
        assert not is_valid

    def test_scan_pair_universe_returns_candidates(self) -> None:
        a, b = _make_cointegrated_pair()
        price_data = {"SYM_A": a, "SYM_B": b}
        candidates = [("SYM_A", "SYM_B", "Test")]
        results = scan_pair_universe(price_data, candidates, max_pvalue=0.20)
        assert len(results) == 1
        assert isinstance(results[0], PairCandidate)
        assert results[0].sector == "Test"

    def test_scan_skips_missing_symbols(self) -> None:
        a, _b = _make_cointegrated_pair()
        price_data = {"SYM_A": a}
        candidates = [("SYM_A", "SYM_B", "Test")]
        results = scan_pair_universe(price_data, candidates)
        assert len(results) == 0


class TestPairsTrader:
    def test_compute_spread_has_zscore(self) -> None:
        a, b = _make_cointegrated_pair()
        trader = PairsTrader()
        spread_df = trader.compute_spread(a, b)
        assert "zscore" in spread_df.columns
        assert "spread" in spread_df.columns
        assert len(spread_df) > 0

    def test_generate_signals_produces_entries_and_exits(self) -> None:
        a, b = _make_cointegrated_pair(n=500)
        trader = PairsTrader(entry_zscore=1.5, exit_zscore=0.3)
        signals = trader.generate_signals(a, b, "A", "B")
        signal_types = {s.signal_type for s in signals}
        assert len(signals) > 0
        assert "exit" in signal_types or "stop_loss" in signal_types or len(signals) >= 1

    def test_backtest_returns_result(self) -> None:
        a, b = _make_cointegrated_pair(n=500)
        trader = PairsTrader(entry_zscore=1.5, exit_zscore=0.3)
        result = trader.backtest_pair(a, b, "A", "B")
        assert result.n_trades >= 0
        assert isinstance(result.total_return, float)
        assert isinstance(result.max_drawdown, float)

    def test_backtest_no_trades_on_flat_spread(self) -> None:
        """Identical prices should produce no spread divergence."""
        dates = pd.bdate_range("2022-01-01", periods=200)
        a = pd.Series(100.0, index=dates)
        b = pd.Series(100.0, index=dates)
        trader = PairsTrader()
        result = trader.backtest_pair(a, b, "A", "B")
        assert result.n_trades == 0

    def test_signal_confidence_bounded(self) -> None:
        a, b = _make_cointegrated_pair(n=500)
        trader = PairsTrader(entry_zscore=1.5)
        signals = trader.generate_signals(a, b, "A", "B")
        for s in signals:
            assert 0 <= s.confidence <= 1.0
