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
def sample_ohlcv_with_delivery(sample_ohlcv_long: pd.DataFrame) -> pd.DataFrame:
    """252-day OHLCV with delivery_pct column for microstructure tests."""
    df = sample_ohlcv_long.copy()
    rng = np.random.default_rng(42)
    df["delivery_pct"] = rng.uniform(0.3, 0.8, size=len(df))
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


@pytest.fixture
def sample_ohlcv_500() -> pd.DataFrame:
    """500 trading days of OHLCV for labeling and CPCV tests."""
    dates = pd.bdate_range("2022-01-03", periods=500, freq="B")
    rng = np.random.default_rng(42)

    base_price = 3800.0
    returns = rng.normal(0.0005, 0.018, size=500)
    closes = base_price * np.cumprod(1 + returns)

    highs = closes * (1 + np.abs(rng.normal(0, 0.012, 500)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.012, 500)))
    opens = closes * (1 + rng.normal(0, 0.005, 500))

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "adj_close": closes,
            "volume": rng.integers(5_000_000, 15_000_000, size=500),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def sample_features_500(sample_ohlcv_500: pd.DataFrame) -> pd.DataFrame:
    """142 synthetic feature columns aligned to sample_ohlcv_500."""
    rng = np.random.default_rng(99)
    n = len(sample_ohlcv_500)
    data = rng.standard_normal((n, 142))
    columns = [f"feat_{i:03d}" for i in range(142)]
    return pd.DataFrame(data, index=sample_ohlcv_500.index, columns=columns)


@pytest.fixture
def sample_known_path() -> pd.DataFrame:
    """Deterministic price path for label verification.

    Days 0-4:   steady rise (100 -> 110) — should trigger upper barrier
    Days 5-9:   sharp drop (110 -> 95)  — should trigger lower barrier
    Days 10-24: flat (100)              — should trigger time barrier
    """
    dates = pd.bdate_range("2024-01-02", periods=25, freq="B")
    closes = np.array(
        [
            100,
            102,
            104,
            106,
            110,
            108,
            104,
            100,
            97,
            95,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
        ],
        dtype=float,
    )
    highs = closes * 1.005
    lows = closes * 0.995
    opens = closes * 1.001

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "adj_close": closes,
            "volume": np.full(25, 10_000_000),
        },
        index=dates,
    )
    df.index.name = "date"
    return df
