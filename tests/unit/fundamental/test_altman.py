"""Unit tests for Altman Z'-Score computation."""

from __future__ import annotations

import pytest

from alphavedha.fundamental.altman import AltmanResult, compute_altman
from alphavedha.fundamental.fetcher import FinancialStatements


def _make_fs(**overrides: object) -> FinancialStatements:
    defaults: dict[str, object] = {
        "symbol": "INFY",
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
        "total_liabilities": (400_000.0, 380_000.0),
        "current_liabilities": (200_000.0, 180_000.0),
        "long_term_debt": (100_000.0, 100_000.0),
        "retained_earnings": (900_000.0, 750_000.0),
        "stockholders_equity": (1_600_000.0, 1_420_000.0),
        "operating_cash_flow": (200_000.0, 160_000.0),
        "capex": (60_000.0, 50_000.0),
        "currency": "INR",
        "data_quality": 1.0,
    }
    defaults.update(overrides)
    return FinancialStatements(**defaults)  # type: ignore[arg-type]


class TestComputeAltman:
    def test_returns_altman_result(self) -> None:
        fs = _make_fs()
        result = compute_altman(fs)
        assert isinstance(result, AltmanResult)
        assert result.symbol == "INFY"

    def test_healthy_company_safe_zone(self) -> None:
        fs = _make_fs()
        result = compute_altman(fs)
        # A company with strong equity, high EBIT, good WC should be in safe zone
        assert result.verdict == "safe"
        assert result.z_score > 2.60

    def test_distress_zone_when_insolvent(self) -> None:
        # Company with negative equity and tiny assets
        fs = _make_fs(
            total_assets=(500_000.0, 450_000.0),
            current_assets=(80_000.0, 70_000.0),
            current_liabilities=(400_000.0, 350_000.0),  # WC deeply negative
            retained_earnings=(-200_000.0, -150_000.0),
            ebit=(5_000.0, 4_000.0),
            stockholders_equity=(50_000.0, 60_000.0),
            total_liabilities=(450_000.0, 390_000.0),
        )
        result = compute_altman(fs)
        assert result.verdict == "distress"
        assert result.z_score < 1.10

    def test_grey_zone(self) -> None:
        # Z' = 6.56*0.04 + 3.26*0.20 + 6.72*0.08 + 1.05*0.667 ≈ 2.15 → grey_zone
        fs = _make_fs(
            total_assets=(1_000_000.0, 900_000.0),
            current_assets=(200_000.0, 180_000.0),
            current_liabilities=(160_000.0, 150_000.0),  # WC = 40_000 → X1=0.04
            retained_earnings=(200_000.0, 180_000.0),  # X2=0.20
            ebit=(80_000.0, 70_000.0),  # X3=0.08
            stockholders_equity=(400_000.0, 360_000.0),
            total_liabilities=(600_000.0, 540_000.0),  # X4=0.667
        )
        result = compute_altman(fs)
        assert result.verdict == "grey_zone"
        assert 1.10 < result.z_score < 2.60

    def test_all_components_populated(self) -> None:
        fs = _make_fs()
        result = compute_altman(fs)
        for attr in (
            "x1_working_capital_ratio",
            "x2_retained_earnings_ratio",
            "x3_ebit_ratio",
            "x4_equity_to_liabilities",
        ):
            val = getattr(result, attr)
            assert isinstance(val, float), f"{attr} should be float"
            assert val == val, f"{attr} should not be NaN"

    def test_zero_total_assets_no_crash(self) -> None:
        # X1/X2/X3 default to 0 when total_assets=0; X4 still computes from equity/liabilities
        fs = _make_fs(total_assets=(0.0, 0.0))
        result = compute_altman(fs)
        assert isinstance(result.z_score, float)
        assert result.x1_working_capital_ratio == pytest.approx(0.0)
        assert result.x2_retained_earnings_ratio == pytest.approx(0.0)
        assert result.x3_ebit_ratio == pytest.approx(0.0)

    def test_data_quality_propagated(self) -> None:
        fs = _make_fs(data_quality=0.78)
        result = compute_altman(fs)
        assert result.data_quality == pytest.approx(0.78)

    def test_interpretation_contains_score(self) -> None:
        fs = _make_fs()
        result = compute_altman(fs)
        assert str(round(result.z_score, 2)) in result.interpretation
