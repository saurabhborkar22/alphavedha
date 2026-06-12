"""Unit tests for sector rotation strategy."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from alphavedha.sectors.rotation import (
    SectorRotationReport,
    _compute_rs_momentum,
    _compute_rs_ratio,
    _pct_ret,
    _phase,
    compute_sector_rotation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _price_series(values: list[float], start: str = "2026-01-01") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# Unit tests for pure functions
# ---------------------------------------------------------------------------


class TestPhase:
    def test_leading(self) -> None:
        assert _phase(105.0, 101.0) == "leading"

    def test_weakening(self) -> None:
        assert _phase(105.0, 99.0) == "weakening"

    def test_lagging(self) -> None:
        assert _phase(95.0, 98.0) == "lagging"

    def test_improving(self) -> None:
        assert _phase(95.0, 102.0) == "improving"

    def test_boundary_at_100(self) -> None:
        assert _phase(100.0, 100.0) == "leading"


class TestPctRet:
    def test_positive_return(self) -> None:
        prices = _price_series([100.0, 105.0, 110.0])
        ret = _pct_ret(prices, 2)
        assert abs(ret - 10.0) < 0.01

    def test_zero_start_safe(self) -> None:
        prices = _price_series([0.0, 100.0, 110.0])
        ret = _pct_ret(prices, 2)
        assert ret == 0.0

    def test_single_point_returns_zero(self) -> None:
        prices = _price_series([100.0])
        assert _pct_ret(prices, 5) == 0.0


class TestComputeRsRatio:
    def test_outperforming_sector_above_100(self) -> None:
        # Sector grows faster than benchmark
        bench = _price_series([100.0] * 20 + [120.0] * 5)
        sector = _price_series([100.0] * 20 + [140.0] * 5)
        rs = _compute_rs_ratio(sector, bench)
        assert rs.iloc[-1] > 100.0

    def test_underperforming_sector_below_100(self) -> None:
        bench = _price_series([100.0] * 20 + [120.0] * 5)
        sector = _price_series([100.0] * 20 + [105.0] * 5)
        rs = _compute_rs_ratio(sector, bench)
        assert rs.iloc[-1] < 100.0

    def test_parity_sector_near_100(self) -> None:
        prices = _price_series(list(range(100, 130)))
        rs = _compute_rs_ratio(prices, prices)
        # Same price → RS-Ratio should be near 100
        assert abs(rs.iloc[-1] - 100.0) < 2.0

    def test_output_length_matches_benchmark(self) -> None:
        bench = _price_series([100.0 + i for i in range(30)])
        sector = _price_series([200.0 + i * 1.1 for i in range(30)])
        rs = _compute_rs_ratio(sector, bench)
        assert len(rs) == len(bench)


class TestComputeRsMomentum:
    def test_accelerating_above_100(self) -> None:
        # RS-Ratio steadily rising → momentum > 100
        rs = pd.Series([95.0, 97.0, 99.0, 101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0])
        mom = _compute_rs_momentum(rs)
        assert mom.iloc[-1] > 100.0

    def test_decelerating_below_100(self) -> None:
        rs = pd.Series([110.0, 108.0, 106.0, 104.0, 102.0, 100.0, 98.0, 96.0, 94.0, 92.0])
        mom = _compute_rs_momentum(rs)
        assert mom.iloc[-1] < 100.0


# ---------------------------------------------------------------------------
# Integration tests for compute_sector_rotation
# ---------------------------------------------------------------------------


def _make_prices(n: int = 30, start: float = 100.0, growth: float = 1.005) -> pd.Series:
    vals = [start * (growth**i) for i in range(n)]
    return _price_series(vals)


@pytest.mark.asyncio
async def test_compute_sector_rotation_no_benchmark_returns_empty() -> None:
    with patch("alphavedha.sectors.rotation._fetch_prices", new=AsyncMock(return_value={})):
        report = await compute_sector_rotation()

    assert isinstance(report, SectorRotationReport)
    assert report.sectors == []
    assert report.data_quality == 0.0


@pytest.mark.asyncio
async def test_compute_sector_rotation_with_data() -> None:
    from alphavedha.sectors.rotation import BENCHMARK_TICKER

    bench = _make_prices(30, start=18000.0, growth=1.002)
    it_sector = _make_prices(30, start=40000.0, growth=1.004)  # outperforming → leading

    prices = {
        BENCHMARK_TICKER: bench,
        "^CNXIT": it_sector,
    }
    with patch("alphavedha.sectors.rotation._fetch_prices", new=AsyncMock(return_value=prices)):
        report = await compute_sector_rotation()

    assert isinstance(report, SectorRotationReport)
    assert report.data_quality > 0.0
    assert len(report.sectors) >= 1
    it_sig = next((s for s in report.sectors if s.sector == "NIFTY_IT"), None)
    assert it_sig is not None
    assert it_sig.phase in ("leading", "improving", "weakening", "lagging")
    assert it_sig.rank >= 1
    assert isinstance(it_sig.interpretation, str)


@pytest.mark.asyncio
async def test_top_sectors_are_leading_or_improving() -> None:
    from alphavedha.sectors.rotation import BENCHMARK_TICKER

    bench = _make_prices(30, start=18000.0, growth=1.002)
    it_sector = _make_prices(30, start=40000.0, growth=1.006)  # strong outperformer

    prices = {BENCHMARK_TICKER: bench, "^CNXIT": it_sector}
    with patch("alphavedha.sectors.rotation._fetch_prices", new=AsyncMock(return_value=prices)):
        report = await compute_sector_rotation()

    for sector_name in report.top_sectors:
        sig = next(s for s in report.sectors if s.sector == sector_name)
        assert sig.phase in ("leading", "improving")


@pytest.mark.asyncio
async def test_avoid_sectors_are_lagging() -> None:
    from alphavedha.sectors.rotation import BENCHMARK_TICKER

    bench = _make_prices(30, start=18000.0, growth=1.005)
    # Lagging: growing slower, RS-Ratio will drop
    bad_sector = _make_prices(30, start=5000.0, growth=1.001)

    prices = {BENCHMARK_TICKER: bench, "^CNXMEDIA": bad_sector}
    with patch("alphavedha.sectors.rotation._fetch_prices", new=AsyncMock(return_value=prices)):
        report = await compute_sector_rotation()

    for sector_name in report.avoid_sectors:
        sig = next(s for s in report.sectors if s.sector == sector_name)
        assert sig.phase == "lagging"
