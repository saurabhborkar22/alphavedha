"""Shared test fixtures — sample OHLCV data, mock providers, config overrides."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Realistic TCS OHLCV data (20 trading days, Jan 2024)."""
    dates = pd.bdate_range("2024-01-02", periods=20, freq="B")
    rng = np.random.default_rng(42)

    base_price = 3800.0
    returns = rng.normal(0.001, 0.015, size=20)
    closes = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "open": closes * (1 + rng.normal(0, 0.005, 20)),
            "high": closes * (1 + np.abs(rng.normal(0, 0.01, 20))),
            "low": closes * (1 - np.abs(rng.normal(0, 0.01, 20))),
            "close": closes,
            "adj_close": closes,
            "volume": rng.integers(5_000_000, 15_000_000, size=20),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def sample_ohlcv_with_gaps(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV data with missing trading days (simulates holidays/gaps)."""
    return sample_ohlcv.drop(sample_ohlcv.index[[5, 6, 12]])


@pytest.fixture
def sample_ohlcv_with_circuit(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV data with a simulated upper circuit hit on day 10."""
    df = sample_ohlcv.copy()
    prev_close = df["close"].iloc[9]
    circuit_close = prev_close * 1.05
    df.iloc[10, df.columns.get_loc("close")] = circuit_close
    df.iloc[10, df.columns.get_loc("high")] = circuit_close
    df.iloc[10, df.columns.get_loc("adj_close")] = circuit_close
    return df


@pytest.fixture
def sample_ohlcv_long() -> pd.DataFrame:
    """Longer OHLCV data (252 trading days) for fractional diff tests."""
    dates = pd.bdate_range("2023-01-02", periods=252, freq="B")
    rng = np.random.default_rng(42)

    base_price = 3800.0
    returns = rng.normal(0.0005, 0.02, size=252)
    closes = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "open": closes * (1 + rng.normal(0, 0.005, 252)),
            "high": closes * (1 + np.abs(rng.normal(0, 0.01, 252))),
            "low": closes * (1 - np.abs(rng.normal(0, 0.01, 252))),
            "close": closes,
            "adj_close": closes,
            "volume": rng.integers(5_000_000, 15_000_000, size=252),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def sample_corporate_actions() -> list[dict]:
    """Sample corporate actions for testing adjustments."""
    return [
        {
            "symbol": "TCS",
            "ex_date": date(2024, 1, 12),
            "action_type": "split",
            "ratio": 2.0,
            "details": "1:2 stock split",
        },
        {
            "symbol": "TCS",
            "ex_date": date(2024, 1, 18),
            "action_type": "bonus",
            "ratio": 1.5,
            "details": "1:2 bonus (3 shares for 2)",
        },
    ]
