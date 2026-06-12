"""Fundamental analysis — Beneish M-Score and Altman Z'-Score for Indian equities."""

from alphavedha.fundamental.altman import AltmanResult, compute_altman
from alphavedha.fundamental.analyzer import FundamentalReport, analyze
from alphavedha.fundamental.beneish import BeneishResult, compute_beneish
from alphavedha.fundamental.fetcher import FinancialStatements, fetch_financials

__all__ = [
    "AltmanResult",
    "BeneishResult",
    "FinancialStatements",
    "FundamentalReport",
    "analyze",
    "compute_altman",
    "compute_beneish",
    "fetch_financials",
]
