"""Altman Z-Score — bankruptcy / financial distress prediction.

Uses the Z'-Score model (Altman 2000) designed for non-manufacturing
and emerging-market firms. The original Z-Score (1968) was calibrated
for US manufacturing; this variant is more appropriate for Indian stocks.

Z'-Score = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

Components
----------
X1  Working Capital / Total Assets
X2  Retained Earnings / Total Assets
X3  EBIT / Total Assets
X4  Book Value of Equity / Book Value of Total Liabilities

Thresholds (Z' model)
---------------------
Z' > 2.60   →  Safe Zone    (GREEN)
1.10 < Z' < 2.60  →  Grey Zone   (YELLOW)
Z' < 1.10   →  Distress Zone (RED)

References
----------
Altman, E.I. (2000). "Predicting Financial Distress of Companies:
Revisiting the Z-Score and Zeta Models." Working Paper, NYU Stern.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphavedha.fundamental.fetcher import FinancialStatements


@dataclass
class AltmanResult:
    symbol: str
    z_score: float
    verdict: str  # "safe" | "grey_zone" | "distress"
    # components
    x1_working_capital_ratio: float
    x2_retained_earnings_ratio: float
    x3_ebit_ratio: float
    x4_equity_to_liabilities: float
    data_quality: float
    interpretation: str


_SAFE_THRESHOLD = 2.60
_DISTRESS_THRESHOLD = 1.10

_WEIGHTS = {
    "x1": 6.56,
    "x2": 3.26,
    "x3": 6.72,
    "x4": 1.05,
}


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def compute_altman(fs: FinancialStatements) -> AltmanResult:
    """Compute Altman Z'-Score from FinancialStatements."""
    ta_c, _ = fs.total_assets
    ca_c, _ = fs.current_assets
    cl_c, _ = fs.current_liabilities
    re_c, _ = fs.retained_earnings
    ebit_c, _ = fs.ebit
    equity_c, _ = fs.stockholders_equity
    total_liab_c, _ = fs.total_liabilities

    # X1 = Working Capital / Total Assets
    working_capital = ca_c - cl_c
    x1 = _safe_div(working_capital, ta_c)

    # X2 = Retained Earnings / Total Assets
    x2 = _safe_div(re_c, ta_c)

    # X3 = EBIT / Total Assets
    x3 = _safe_div(ebit_c, ta_c)

    # X4 = Book Value of Equity / Book Value of Total Liabilities
    x4 = _safe_div(equity_c, total_liab_c if total_liab_c != 0 else 1.0)

    z_score = (
        _WEIGHTS["x1"] * x1
        + _WEIGHTS["x2"] * x2
        + _WEIGHTS["x3"] * x3
        + _WEIGHTS["x4"] * x4
    )
    z_score = round(z_score, 4)

    if z_score > _SAFE_THRESHOLD:
        verdict = "safe"
        interpretation = (
            f"Z'-Score {z_score:.2f} > {_SAFE_THRESHOLD}: SAFE ZONE. "
            "Low probability of financial distress within 2 years."
        )
    elif z_score > _DISTRESS_THRESHOLD:
        verdict = "grey_zone"
        interpretation = (
            f"Z'-Score {z_score:.2f} in grey zone ({_DISTRESS_THRESHOLD}–{_SAFE_THRESHOLD}). "
            "Some financial vulnerability — monitor debt levels and cash flow."
        )
    else:
        verdict = "distress"
        interpretation = (
            f"Z'-Score {z_score:.2f} < {_DISTRESS_THRESHOLD}: DISTRESS ZONE. "
            "High probability of financial distress. Exercise significant caution."
        )

    return AltmanResult(
        symbol=fs.symbol,
        z_score=z_score,
        verdict=verdict,
        x1_working_capital_ratio=round(x1, 4),
        x2_retained_earnings_ratio=round(x2, 4),
        x3_ebit_ratio=round(x3, 4),
        x4_equity_to_liabilities=round(x4, 4),
        data_quality=fs.data_quality,
        interpretation=interpretation,
    )
