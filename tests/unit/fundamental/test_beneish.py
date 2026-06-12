"""Unit tests for Beneish M-Score computation."""

from __future__ import annotations

import pytest

from alphavedha.fundamental.beneish import BeneishResult, compute_beneish
from alphavedha.fundamental.fetcher import FinancialStatements


def _make_fs(**overrides: object) -> FinancialStatements:
    """Return a base FinancialStatements with realistic healthy numbers, optionally overridden."""
    defaults: dict[str, object] = {
        "symbol": "TCS",
        "revenue": (1_200_000.0, 1_000_000.0),
        "gross_profit": (480_000.0, 400_000.0),
        "cogs": (720_000.0, 600_000.0),
        "operating_income": (240_000.0, 200_000.0),
        "ebit": (240_000.0, 200_000.0),
        "net_income": (180_000.0, 150_000.0),
        "depreciation": (40_000.0, 35_000.0),
        "sga_expense": (60_000.0, 50_000.0),
        "total_assets": (2_000_000.0, 1_800_000.0),
        "current_assets": (800_000.0, 700_000.0),
        "cash": (300_000.0, 250_000.0),
        "receivables": (120_000.0, 100_000.0),
        "ppe": (500_000.0, 450_000.0),
        "total_liabilities": (700_000.0, 650_000.0),
        "current_liabilities": (300_000.0, 270_000.0),
        "long_term_debt": (200_000.0, 200_000.0),
        "retained_earnings": (900_000.0, 750_000.0),
        "stockholders_equity": (1_300_000.0, 1_150_000.0),
        "operating_cash_flow": (200_000.0, 160_000.0),
        "capex": (60_000.0, 50_000.0),
        "currency": "INR",
        "data_quality": 1.0,
    }
    defaults.update(overrides)
    return FinancialStatements(**defaults)  # type: ignore[arg-type]


class TestComputeBeneish:
    def test_returns_beneish_result(self) -> None:
        fs = _make_fs()
        result = compute_beneish(fs)
        assert isinstance(result, BeneishResult)
        assert result.symbol == "TCS"

    def test_healthy_company_non_manipulator(self) -> None:
        fs = _make_fs()
        result = compute_beneish(fs)
        # A stable, growing, profitable company should score well below -2.22
        assert result.verdict == "non_manipulator"
        assert result.m_score < -2.22

    def test_manipulator_verdict_when_high_accruals(self) -> None:
        # Inflate NI relative to OCF (high accruals = manipulation signal)
        fs = _make_fs(
            net_income=(500_000.0, 150_000.0),  # massive jump in net income
            operating_cash_flow=(10_000.0, 160_000.0),  # OCF not following NI
        )
        result = compute_beneish(fs)
        assert result.verdict == "manipulator"
        assert result.m_score > -1.78

    def test_all_components_populated(self) -> None:
        fs = _make_fs()
        result = compute_beneish(fs)
        for attr in ("dsri", "gmi", "aqi", "sgi", "depi", "sgai", "tata", "lvgi"):
            val = getattr(result, attr)
            assert isinstance(val, float), f"{attr} should be float"
            assert val == val, f"{attr} should not be NaN"  # NaN check

    def test_indices_capped_to_range(self) -> None:
        # Even extreme values should be capped at 10.0
        fs = _make_fs(
            receivables=(10_000_000.0, 100.0),  # extreme DSRI
            revenue=(1_000_000.0, 1_000_000.0),
        )
        result = compute_beneish(fs)
        assert result.dsri <= 10.0

    def test_zero_division_safe(self) -> None:
        # Zero prior-year revenue should not crash
        fs = _make_fs(revenue=(1_000_000.0, 0.0))
        result = compute_beneish(fs)
        assert isinstance(result.m_score, float)

    def test_data_quality_propagated(self) -> None:
        fs = _make_fs(data_quality=0.65)
        result = compute_beneish(fs)
        assert result.data_quality == pytest.approx(0.65)

    def test_interpretation_contains_score(self) -> None:
        fs = _make_fs()
        result = compute_beneish(fs)
        assert str(round(result.m_score, 2)) in result.interpretation
