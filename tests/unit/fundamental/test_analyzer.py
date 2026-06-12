"""Unit tests for FundamentalReport orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from alphavedha.fundamental.altman import AltmanResult
from alphavedha.fundamental.analyzer import FundamentalReport, _overall_verdict, analyze
from alphavedha.fundamental.beneish import BeneishResult
from alphavedha.fundamental.fetcher import FinancialStatements


def _beneish(verdict: str, m_score: float) -> BeneishResult:
    return BeneishResult(
        symbol="X",
        m_score=m_score,
        verdict=verdict,
        dsri=1.0,
        gmi=1.0,
        aqi=1.0,
        sgi=1.0,
        depi=1.0,
        sgai=1.0,
        tata=0.01,
        lvgi=1.0,
        data_quality=1.0,
        interpretation=f"M={m_score}",
    )


def _altman(verdict: str, z_score: float) -> AltmanResult:
    return AltmanResult(
        symbol="X",
        z_score=z_score,
        verdict=verdict,
        x1_working_capital_ratio=0.2,
        x2_retained_earnings_ratio=0.3,
        x3_ebit_ratio=0.1,
        x4_equity_to_liabilities=2.0,
        data_quality=1.0,
        interpretation=f"Z={z_score}",
    )


def _make_fs() -> FinancialStatements:
    return FinancialStatements(
        symbol="TCS",
        revenue=(1_200_000.0, 1_000_000.0),
        gross_profit=(480_000.0, 400_000.0),
        cogs=(720_000.0, 600_000.0),
        operating_income=(240_000.0, 200_000.0),
        ebit=(240_000.0, 200_000.0),
        net_income=(180_000.0, 150_000.0),
        depreciation=(40_000.0, 35_000.0),
        sga_expense=(60_000.0, 50_000.0),
        total_assets=(2_000_000.0, 1_800_000.0),
        current_assets=(800_000.0, 700_000.0),
        cash=(300_000.0, 250_000.0),
        receivables=(120_000.0, 100_000.0),
        ppe=(500_000.0, 450_000.0),
        total_liabilities=(400_000.0, 380_000.0),
        current_liabilities=(200_000.0, 180_000.0),
        long_term_debt=(100_000.0, 100_000.0),
        retained_earnings=(900_000.0, 750_000.0),
        stockholders_equity=(1_600_000.0, 1_420_000.0),
        operating_cash_flow=(200_000.0, 160_000.0),
        capex=(60_000.0, 50_000.0),
        currency="INR",
        data_quality=1.0,
    )


class TestOverallVerdict:
    def test_both_none_returns_insufficient(self) -> None:
        verdict, _summary = _overall_verdict(None, None)
        assert verdict == "insufficient_data"

    def test_both_healthy_returns_healthy(self) -> None:
        b = _beneish("non_manipulator", -2.80)
        a = _altman("safe", 3.20)
        verdict, summary = _overall_verdict(b, a)
        assert verdict == "healthy"
        assert "HEALTHY" in summary

    def test_manipulator_returns_red_flag(self) -> None:
        b = _beneish("manipulator", -1.50)
        a = _altman("safe", 3.20)
        verdict, summary = _overall_verdict(b, a)
        assert verdict == "red_flag"
        assert "RED FLAG" in summary

    def test_distress_returns_red_flag(self) -> None:
        b = _beneish("non_manipulator", -2.80)
        a = _altman("distress", 0.80)
        verdict, _summary = _overall_verdict(b, a)
        assert verdict == "red_flag"

    def test_grey_zone_returns_caution(self) -> None:
        b = _beneish("grey_zone", -2.00)
        a = _altman("safe", 3.20)
        verdict, summary = _overall_verdict(b, a)
        assert verdict == "caution"
        assert "CAUTION" in summary

    def test_only_altman_none(self) -> None:
        b = _beneish("non_manipulator", -2.80)
        verdict, _ = _overall_verdict(b, None)
        assert verdict == "healthy"

    def test_only_beneish_none(self) -> None:
        a = _altman("safe", 3.20)
        verdict, _ = _overall_verdict(None, a)
        assert verdict == "healthy"


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_analyze_healthy_company(self) -> None:
        fs = _make_fs()
        with patch(
            "alphavedha.fundamental.analyzer.fetch_financials",
            new=AsyncMock(return_value=fs),
        ):
            report = await analyze("TCS")

        assert isinstance(report, FundamentalReport)
        assert report.symbol == "TCS"
        assert report.overall_verdict == "healthy"
        assert report.beneish is not None
        assert report.altman is not None
        assert report.data_quality == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_analyze_no_data_returns_insufficient(self) -> None:
        with patch(
            "alphavedha.fundamental.analyzer.fetch_financials",
            new=AsyncMock(return_value=None),
        ):
            report = await analyze("UNKNOWN")

        assert report.overall_verdict == "insufficient_data"
        assert report.beneish is None
        assert report.altman is None
        assert report.data_quality == 0.0
