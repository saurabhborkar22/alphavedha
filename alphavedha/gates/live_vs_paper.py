"""Live-vs-paper tracking report — G2 gate prerequisite.

Compares live execution fills against paper/shadow fills for the same
signals to measure execution quality. If divergence exceeds the
slippage budget x 1.5, the strategy goes back to shadow.

The report feeds into G2 criteria: "live net return within 1.5x
slippage budget of paper counterpart."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FillRecord:
    """A single fill from either live or paper execution."""

    symbol: str
    fill_date: date
    side: str
    quantity: int
    fill_price: float
    decision_price: float
    strategy: str
    source: str  # "live" or "paper"


@dataclass(frozen=True)
class SymbolComparison:
    """Per-symbol live vs paper comparison."""

    symbol: str
    live_avg_price: float
    paper_avg_price: float
    price_divergence_pct: float
    live_slippage_bps: float
    paper_slippage_bps: float
    slippage_divergence_bps: float
    live_fills: int
    paper_fills: int


@dataclass
class LiveVsPaperReport:
    """Aggregate comparison between live and paper execution."""

    strategy: str
    period_start: date
    period_end: date
    live_net_return: float = 0.0
    paper_net_return: float = 0.0
    return_divergence: float = 0.0
    slippage_budget: float = 0.0
    within_budget: bool = True
    budget_multiple: float = 0.0
    total_live_fills: int = 0
    total_paper_fills: int = 0
    symbol_comparisons: list[SymbolComparison] = field(default_factory=list)
    verdict: str = ""


def _compute_slippage_bps(fill_price: float, decision_price: float) -> float:
    if decision_price <= 0:
        return 0.0
    return abs(fill_price - decision_price) / decision_price * 10_000


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare_fills(
    live_fills: list[FillRecord],
    paper_fills: list[FillRecord],
    strategy: str,
    period_start: date,
    period_end: date,
    slippage_budget: float = 0.001,
    max_budget_multiple: float = 1.5,
) -> LiveVsPaperReport:
    """Compare live and paper fills, produce a divergence report.

    Args:
        live_fills: Fills from real broker execution
        paper_fills: Fills from shadow/paper execution
        strategy: Strategy name
        period_start: Start of comparison period
        period_end: End of comparison period
        slippage_budget: Expected slippage as a fraction (0.001 = 0.1%)
        max_budget_multiple: Max acceptable divergence as multiple of budget

    Returns:
        LiveVsPaperReport with per-symbol and aggregate comparisons
    """
    live_by_symbol: dict[str, list[FillRecord]] = {}
    paper_by_symbol: dict[str, list[FillRecord]] = {}

    for f in live_fills:
        live_by_symbol.setdefault(f.symbol, []).append(f)
    for f in paper_fills:
        paper_by_symbol.setdefault(f.symbol, []).append(f)

    all_symbols = sorted(set(live_by_symbol.keys()) | set(paper_by_symbol.keys()))

    comparisons: list[SymbolComparison] = []
    all_live_slippages: list[float] = []
    all_paper_slippages: list[float] = []

    for symbol in all_symbols:
        live_sym = live_by_symbol.get(symbol, [])
        paper_sym = paper_by_symbol.get(symbol, [])

        live_prices = [f.fill_price for f in live_sym]
        paper_prices = [f.fill_price for f in paper_sym]

        live_avg = _avg(live_prices)
        paper_avg = _avg(paper_prices)

        price_div = 0.0
        if paper_avg > 0:
            price_div = (live_avg - paper_avg) / paper_avg * 100

        live_slip = [_compute_slippage_bps(f.fill_price, f.decision_price) for f in live_sym]
        paper_slip = [_compute_slippage_bps(f.fill_price, f.decision_price) for f in paper_sym]

        all_live_slippages.extend(live_slip)
        all_paper_slippages.extend(paper_slip)

        comparisons.append(
            SymbolComparison(
                symbol=symbol,
                live_avg_price=round(live_avg, 2),
                paper_avg_price=round(paper_avg, 2),
                price_divergence_pct=round(price_div, 4),
                live_slippage_bps=round(_avg(live_slip), 2),
                paper_slippage_bps=round(_avg(paper_slip), 2),
                slippage_divergence_bps=round(_avg(live_slip) - _avg(paper_slip), 2),
                live_fills=len(live_sym),
                paper_fills=len(paper_sym),
            )
        )

    live_return = _compute_portfolio_return(live_fills)
    paper_return = _compute_portfolio_return(paper_fills)
    divergence = abs(live_return - paper_return)

    threshold = slippage_budget * max_budget_multiple
    within_budget = divergence <= threshold
    budget_mult = divergence / slippage_budget if slippage_budget > 0 else 0.0

    if within_budget:
        verdict = (
            f"PASS — divergence {divergence:.4%} within "
            f"{max_budget_multiple}x slippage budget ({threshold:.4%})"
        )
    else:
        verdict = (
            f"FAIL — divergence {divergence:.4%} exceeds "
            f"{max_budget_multiple}x slippage budget ({threshold:.4%}). "
            f"Back to shadow."
        )

    report = LiveVsPaperReport(
        strategy=strategy,
        period_start=period_start,
        period_end=period_end,
        live_net_return=round(live_return, 6),
        paper_net_return=round(paper_return, 6),
        return_divergence=round(divergence, 6),
        slippage_budget=slippage_budget,
        within_budget=within_budget,
        budget_multiple=round(budget_mult, 2),
        total_live_fills=len(live_fills),
        total_paper_fills=len(paper_fills),
        symbol_comparisons=comparisons,
        verdict=verdict,
    )

    logger.info(
        "live_vs_paper_report",
        strategy=strategy,
        within_budget=within_budget,
        divergence=round(divergence, 6),
        budget_multiple=round(budget_mult, 2),
        live_fills=len(live_fills),
        paper_fills=len(paper_fills),
    )

    return report


def _compute_portfolio_return(fills: list[FillRecord]) -> float:
    """Compute simple return from a list of fills.

    For each symbol: avg fill price vs decision price gives the
    per-trade return, then average across all trades.
    """
    if not fills:
        return 0.0

    returns: list[float] = []
    for f in fills:
        if f.decision_price > 0:
            if f.side == "BUY":
                ret = (f.fill_price - f.decision_price) / f.decision_price
            else:
                ret = (f.decision_price - f.fill_price) / f.decision_price
            returns.append(ret)

    return _avg(returns)


def format_report(report: LiveVsPaperReport) -> str:
    """Format report as readable text for Telegram or CLI."""
    lines = [
        f"Live-vs-Paper Report: {report.strategy}",
        f"Period: {report.period_start} to {report.period_end}",
        "",
        f"Live return:  {report.live_net_return:+.4%}",
        f"Paper return: {report.paper_net_return:+.4%}",
        f"Divergence:   {report.return_divergence:.4%} ({report.budget_multiple:.1f}x budget)",
        "",
        f"Live fills:  {report.total_live_fills}",
        f"Paper fills: {report.total_paper_fills}",
        "",
    ]

    if report.symbol_comparisons:
        lines.append("Per-symbol:")
        for sc in report.symbol_comparisons:
            lines.append(
                f"  {sc.symbol}: live={sc.live_avg_price:.2f} "
                f"paper={sc.paper_avg_price:.2f} "
                f"div={sc.price_divergence_pct:+.2f}% "
                f"slip_div={sc.slippage_divergence_bps:+.1f}bps"
            )
        lines.append("")

    lines.append(f"Verdict: {report.verdict}")
    return "\n".join(lines)
