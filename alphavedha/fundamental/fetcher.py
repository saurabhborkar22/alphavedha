"""Financial statement fetcher — pulls balance sheet, income statement, and cash flow from yfinance.

For Indian companies, yfinance provides annual financial statements under
the .financials, .balance_sheet, and .cashflow attributes. Data is returned
as a dict of {field: [current_year_value, prior_year_value]} so callers
can compute year-over-year ratios without knowing which column is which.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog
import yfinance as yf

logger = structlog.get_logger(__name__)

_INDEX_TICKERS: dict[str, str] = {
    "NIFTY50": "^NSEI",
    "NIFTY": "^NSEI",
    "SENSEX": "^BSESN",
}


def _yf_ticker(symbol: str) -> str:
    return _INDEX_TICKERS.get(symbol.upper(), f"{symbol}.NS")


@dataclass
class FinancialStatements:
    symbol: str
    # income statement (current year, prior year)
    revenue: tuple[float, float]
    gross_profit: tuple[float, float]
    cogs: tuple[float, float]
    operating_income: tuple[float, float]
    ebit: tuple[float, float]
    net_income: tuple[float, float]
    depreciation: tuple[float, float]
    sga_expense: tuple[float, float]
    # balance sheet (current year, prior year)
    total_assets: tuple[float, float]
    current_assets: tuple[float, float]
    cash: tuple[float, float]
    receivables: tuple[float, float]
    ppe: tuple[float, float]
    total_liabilities: tuple[float, float]
    current_liabilities: tuple[float, float]
    long_term_debt: tuple[float, float]
    retained_earnings: tuple[float, float]
    stockholders_equity: tuple[float, float]
    # cash flow
    operating_cash_flow: tuple[float, float]
    capex: tuple[float, float]
    # metadata
    currency: str = "INR"
    data_quality: float = 1.0  # fraction of fields successfully populated


def _safe_col(df: Any, *candidate_keys: str) -> tuple[float, float]:
    """Try each candidate row label; return (current, prior) or (0, 0) if not found."""
    if df is None or df.empty:
        return (0.0, 0.0)
    for key in candidate_keys:
        # Exact match
        if key in df.index:
            row = df.loc[key]
            vals = [float(v) if v is not None and str(v) != "nan" else 0.0 for v in row.values]
            cur = vals[0] if len(vals) > 0 else 0.0
            pri = vals[1] if len(vals) > 1 else 0.0
            return (cur, pri)
        # Case-insensitive partial match
        for idx in df.index:
            if isinstance(idx, str) and key.lower() in idx.lower():
                row = df.loc[idx]
                vals = [float(v) if v is not None and str(v) != "nan" else 0.0 for v in row.values]
                cur = vals[0] if len(vals) > 0 else 0.0
                pri = vals[1] if len(vals) > 1 else 0.0
                return (cur, pri)
    return (0.0, 0.0)


def _fetch_financials_sync(symbol: str) -> FinancialStatements | None:
    """Fetch all financial statements via yfinance (synchronous — run in thread)."""
    ticker_sym = _yf_ticker(symbol)
    try:
        t = yf.Ticker(ticker_sym)
        inc = t.financials  # rows = line items, cols = fiscal years
        bal = t.balance_sheet
        cf = t.cashflow

        if inc is None or inc.empty:
            logger.warning("fundamental_no_income", symbol=symbol)
            return None
        if bal is None or bal.empty:
            logger.warning("fundamental_no_balance", symbol=symbol)
            return None

        # Income statement
        revenue = _safe_col(inc, "Total Revenue", "Revenue", "Net Revenue")
        gross_profit = _safe_col(inc, "Gross Profit")
        cogs = _safe_col(inc, "Cost Of Revenue", "COGS", "Cost of Goods Sold")
        if cogs == (0.0, 0.0) and revenue != (0.0, 0.0) and gross_profit != (0.0, 0.0):
            cogs = (revenue[0] - gross_profit[0], revenue[1] - gross_profit[1])
        op_income = _safe_col(inc, "Operating Income", "EBIT", "Operating Profit")
        ebit = _safe_col(inc, "EBIT", "Operating Income", "Earnings Before Interest And Tax")
        if ebit == (0.0, 0.0):
            ebit = op_income
        net_income = _safe_col(inc, "Net Income", "Net Income Common Stockholders")
        dep = _safe_col(
            cf, "Depreciation", "Depreciation And Amortization", "Depreciation Amortization"
        )
        if dep == (0.0, 0.0):
            dep = _safe_col(inc, "Depreciation", "Depreciation And Amortization")
        sga = _safe_col(
            inc,
            "Selling General Administrative",
            "SG&A",
            "Selling General And Administration",
            "Operating Expense",
        )

        # Balance sheet
        total_assets = _safe_col(bal, "Total Assets")
        cur_assets = _safe_col(bal, "Current Assets", "Total Current Assets")
        cash = _safe_col(
            bal, "Cash And Cash Equivalents", "Cash", "Cash And Short Term Investments"
        )
        recv = _safe_col(
            bal,
            "Accounts Receivable",
            "Net Receivables",
            "Receivables",
            "Trade And Other Receivables",
        )
        ppe = _safe_col(
            bal,
            "Net PPE",
            "Property Plant Equipment Net",
            "Property Plant And Equipment Net",
            "Net Property Plant And Equipment",
        )
        total_liab = _safe_col(bal, "Total Liabilities Net Minority Interest", "Total Liabilities")
        cur_liab = _safe_col(
            bal, "Current Liabilities", "Total Current Liabilities", "Current Liabilities Net"
        )
        ltd = _safe_col(bal, "Long Term Debt", "Long Term Debt And Capital Lease Obligation")
        ret_earn = _safe_col(
            bal, "Retained Earnings", "Retained Earnings Deficit", "Retained Earnings Loss"
        )
        equity = _safe_col(
            bal,
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Total Stockholder Equity",
        )

        # Cash flow
        ocf = _safe_col(
            cf, "Operating Cash Flow", "Cash From Operations", "Net Cash From Operations"
        )
        capex = _safe_col(
            cf,
            "Capital Expenditure",
            "Purchase Of Property Plant And Equipment",
            "Investments In Property Plant And Equipment",
        )

        # Count populated fields
        all_fields = [
            revenue,
            gross_profit,
            cogs,
            op_income,
            ebit,
            net_income,
            dep,
            total_assets,
            cur_assets,
            cash,
            recv,
            ppe,
            total_liab,
            cur_liab,
            ltd,
            ret_earn,
            equity,
            ocf,
            capex,
        ]
        n_populated = sum(1 for f in all_fields if f != (0.0, 0.0))
        quality = n_populated / len(all_fields)

        currency = getattr(t.fast_info, "currency", "INR") or "INR"

        return FinancialStatements(
            symbol=symbol,
            revenue=revenue,
            gross_profit=gross_profit,
            cogs=cogs,
            operating_income=op_income,
            ebit=ebit,
            net_income=net_income,
            depreciation=dep,
            sga_expense=sga,
            total_assets=total_assets,
            current_assets=cur_assets,
            cash=cash,
            receivables=recv,
            ppe=ppe,
            total_liabilities=total_liab,
            current_liabilities=cur_liab,
            long_term_debt=ltd,
            retained_earnings=ret_earn,
            stockholders_equity=equity,
            operating_cash_flow=ocf,
            capex=capex,
            currency=currency,
            data_quality=round(quality, 2),
        )

    except Exception as e:
        logger.error("fundamental_fetch_failed", symbol=symbol, error=str(e))
        return None


async def fetch_financials(symbol: str) -> FinancialStatements | None:
    """Async wrapper — fetches in a thread pool to avoid blocking the event loop."""
    return await asyncio.to_thread(_fetch_financials_sync, symbol)
