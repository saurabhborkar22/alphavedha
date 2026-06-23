"""Tests for live-vs-paper tracking report."""

from __future__ import annotations

from datetime import date

from alphavedha.gates.live_vs_paper import (
    FillRecord,
    LiveVsPaperReport,
    compare_fills,
    format_report,
)


def _make_fill(
    symbol: str = "TCS.NS",
    fill_price: float = 3510.0,
    decision_price: float = 3500.0,
    source: str = "live",
    side: str = "BUY",
) -> FillRecord:
    return FillRecord(
        symbol=symbol,
        fill_date=date(2026, 8, 1),
        side=side,
        quantity=10,
        fill_price=fill_price,
        decision_price=decision_price,
        strategy="ensemble_v1",
        source=source,
    )


class TestCompareFills:
    def test_within_budget(self) -> None:
        live = [_make_fill(fill_price=3502.0, decision_price=3500.0, source="live")]
        paper = [_make_fill(fill_price=3501.0, decision_price=3500.0, source="paper")]
        report = compare_fills(
            live,
            paper,
            "ensemble_v1",
            date(2026, 8, 1),
            date(2026, 8, 31),
            slippage_budget=0.001,
        )
        assert report.within_budget is True
        assert "PASS" in report.verdict

    def test_over_budget(self) -> None:
        live = [_make_fill(fill_price=3600.0, decision_price=3500.0, source="live")]
        paper = [_make_fill(fill_price=3500.0, decision_price=3500.0, source="paper")]
        report = compare_fills(
            live,
            paper,
            "ensemble_v1",
            date(2026, 8, 1),
            date(2026, 8, 31),
            slippage_budget=0.001,
        )
        assert report.within_budget is False
        assert "FAIL" in report.verdict
        assert "shadow" in report.verdict.lower()

    def test_empty_fills(self) -> None:
        report = compare_fills(
            [],
            [],
            "ensemble_v1",
            date(2026, 8, 1),
            date(2026, 8, 31),
        )
        assert report.within_budget is True
        assert report.total_live_fills == 0
        assert report.total_paper_fills == 0

    def test_multiple_symbols(self) -> None:
        live = [
            _make_fill("TCS.NS", 3510, 3500, "live"),
            _make_fill("INFY.NS", 1510, 1500, "live"),
        ]
        paper = [
            _make_fill("TCS.NS", 3505, 3500, "paper"),
            _make_fill("INFY.NS", 1505, 1500, "paper"),
        ]
        report = compare_fills(
            live,
            paper,
            "ensemble_v1",
            date(2026, 8, 1),
            date(2026, 8, 31),
        )
        assert len(report.symbol_comparisons) == 2
        symbols = [sc.symbol for sc in report.symbol_comparisons]
        assert "INFY.NS" in symbols
        assert "TCS.NS" in symbols

    def test_symbol_only_in_live(self) -> None:
        live = [_make_fill("TCS.NS", 3510, 3500, "live")]
        paper = [_make_fill("INFY.NS", 1505, 1500, "paper")]
        report = compare_fills(
            live,
            paper,
            "ensemble_v1",
            date(2026, 8, 1),
            date(2026, 8, 31),
        )
        assert len(report.symbol_comparisons) == 2


class TestSymbolComparison:
    def test_price_divergence(self) -> None:
        live = [_make_fill("TCS.NS", 3510, 3500, "live")]
        paper = [_make_fill("TCS.NS", 3500, 3500, "paper")]
        report = compare_fills(
            live,
            paper,
            "ensemble_v1",
            date(2026, 8, 1),
            date(2026, 8, 31),
        )
        sc = report.symbol_comparisons[0]
        assert sc.live_avg_price == 3510.0
        assert sc.paper_avg_price == 3500.0
        assert sc.price_divergence_pct > 0

    def test_slippage_divergence(self) -> None:
        live = [_make_fill("TCS.NS", 3520, 3500, "live")]
        paper = [_make_fill("TCS.NS", 3505, 3500, "paper")]
        report = compare_fills(
            live,
            paper,
            "ensemble_v1",
            date(2026, 8, 1),
            date(2026, 8, 31),
        )
        sc = report.symbol_comparisons[0]
        assert sc.live_slippage_bps > sc.paper_slippage_bps
        assert sc.slippage_divergence_bps > 0


class TestBudgetMultiple:
    def test_exact_at_budget(self) -> None:
        live = [_make_fill(fill_price=3503.5, decision_price=3500.0, source="live")]
        paper = [_make_fill(fill_price=3500.0, decision_price=3500.0, source="paper")]
        report = compare_fills(
            live,
            paper,
            "ensemble_v1",
            date(2026, 8, 1),
            date(2026, 8, 31),
            slippage_budget=0.001,
        )
        assert report.budget_multiple == 1.0

    def test_zero_budget(self) -> None:
        live = [_make_fill(fill_price=3510, decision_price=3500, source="live")]
        paper = [_make_fill(fill_price=3505, decision_price=3500, source="paper")]
        report = compare_fills(
            live,
            paper,
            "ensemble_v1",
            date(2026, 8, 1),
            date(2026, 8, 31),
            slippage_budget=0.0,
        )
        assert report.budget_multiple == 0.0


class TestSellFills:
    def test_sell_return_inverted(self) -> None:
        live = [_make_fill(fill_price=3490, decision_price=3500, source="live", side="SELL")]
        paper = [_make_fill(fill_price=3495, decision_price=3500, source="paper", side="SELL")]
        report = compare_fills(
            live,
            paper,
            "ensemble_v1",
            date(2026, 8, 1),
            date(2026, 8, 31),
        )
        assert report.live_net_return > 0
        assert report.paper_net_return > 0


class TestFormatReport:
    def test_format_basic(self) -> None:
        report = LiveVsPaperReport(
            strategy="ensemble_v1",
            period_start=date(2026, 8, 1),
            period_end=date(2026, 8, 31),
            live_net_return=0.008,
            paper_net_return=0.009,
            return_divergence=0.001,
            slippage_budget=0.001,
            within_budget=True,
            budget_multiple=1.0,
            total_live_fills=20,
            total_paper_fills=20,
            verdict="PASS",
        )
        text = format_report(report)
        assert "ensemble_v1" in text
        assert "Live return" in text
        assert "Paper return" in text
        assert "Verdict" in text

    def test_format_with_symbols(self) -> None:
        from alphavedha.gates.live_vs_paper import SymbolComparison

        report = LiveVsPaperReport(
            strategy="test",
            period_start=date(2026, 8, 1),
            period_end=date(2026, 8, 31),
            verdict="PASS",
            symbol_comparisons=[
                SymbolComparison(
                    symbol="TCS.NS",
                    live_avg_price=3510,
                    paper_avg_price=3505,
                    price_divergence_pct=0.14,
                    live_slippage_bps=28.57,
                    paper_slippage_bps=14.29,
                    slippage_divergence_bps=14.28,
                    live_fills=5,
                    paper_fills=5,
                ),
            ],
        )
        text = format_report(report)
        assert "TCS.NS" in text
        assert "Per-symbol" in text
