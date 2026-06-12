"""Beneish M-Score — earnings manipulation detection.

The M-Score (Beneish 1999) is an 8-variable model that identifies
companies likely to be manipulating earnings. Widely used in forensic
accounting and investment screening.

Thresholds
----------
M-Score > -1.78  →  likely manipulator  (RED flag)
-2.22 < M < -1.78 →  grey zone
M-Score < -2.22  →  likely non-manipulator  (SAFE)

References
----------
Beneish, M.D. (1999). "The Detection of Earnings Manipulation."
Financial Analysts Journal, 55(5), 24-36.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphavedha.fundamental.fetcher import FinancialStatements


@dataclass
class BeneishResult:
    symbol: str
    m_score: float
    verdict: str  # "manipulator" | "grey_zone" | "non_manipulator"
    # individual indices
    dsri: float   # Days Sales Receivable Index
    gmi: float    # Gross Margin Index
    aqi: float    # Asset Quality Index
    sgi: float    # Sales Growth Index
    depi: float   # Depreciation Index
    sgai: float   # SG&A Expense Index
    tata: float   # Total Accruals to Total Assets
    lvgi: float   # Leverage Index
    data_quality: float
    interpretation: str


_MANIPULATOR_THRESHOLD = -1.78
_SAFE_THRESHOLD = -2.22

# M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
#       + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
_INTERCEPT = -4.84
_WEIGHTS = {
    "dsri": 0.920,
    "gmi": 0.528,
    "aqi": 0.404,
    "sgi": 0.892,
    "depi": 0.115,
    "sgai": -0.172,
    "tata": 4.679,
    "lvgi": -0.327,
}


def _safe_div(a: float, b: float, default: float = 1.0) -> float:
    return a / b if b != 0 else default


def compute_beneish(fs: FinancialStatements) -> BeneishResult:
    """Compute Beneish M-Score from FinancialStatements."""
    rev_c, rev_p = fs.revenue
    gp_c, gp_p = fs.gross_profit
    cogs_c, cogs_p = fs.cogs
    recv_c, recv_p = fs.receivables
    ta_c, ta_p = fs.total_assets
    ca_c, ca_p = fs.current_assets
    cash_c, cash_p = fs.cash
    ppe_c, ppe_p = fs.ppe
    dep_c, dep_p = fs.depreciation
    sga_c, sga_p = fs.sga_expense
    cl_c, cl_p = fs.current_liabilities
    ltd_c, ltd_p = fs.long_term_debt
    ocf_c, _ = fs.operating_cash_flow
    ni_c, _ = fs.net_income

    # DSRI = (Receivables_t / Sales_t) / (Receivables_{t-1} / Sales_{t-1})
    dsri = _safe_div(
        _safe_div(recv_c, rev_c),
        _safe_div(recv_p, rev_p),
    )

    # GMI = [(Sales_{t-1} - COGS_{t-1}) / Sales_{t-1}] / [(Sales_t - COGS_t) / Sales_t]
    gm_c = _safe_div(gp_c if gp_c != 0 else rev_c - cogs_c, rev_c, 0.0)
    gm_p = _safe_div(gp_p if gp_p != 0 else rev_p - cogs_p, rev_p, 0.0)
    gmi = _safe_div(gm_p, gm_c)

    # AQI = [1 - (CA_t + PPE_t) / TA_t] / [1 - (CA_{t-1} + PPE_{t-1}) / TA_{t-1}]
    aq_c = 1 - _safe_div(ca_c + ppe_c, ta_c, 0.0)
    aq_p = 1 - _safe_div(ca_p + ppe_p, ta_p, 0.0)
    aqi = _safe_div(aq_c, aq_p)

    # SGI = Sales_t / Sales_{t-1}
    sgi = _safe_div(rev_c, rev_p)

    # DEPI = [Dep_{t-1} / (Dep_{t-1} + PPE_{t-1})] / [Dep_t / (Dep_t + PPE_t)]
    dep_rate_c = _safe_div(dep_c, dep_c + ppe_c, 0.0)
    dep_rate_p = _safe_div(dep_p, dep_p + ppe_p, 0.0)
    depi = _safe_div(dep_rate_p, dep_rate_c)

    # SGAI = (SGA_t / Sales_t) / (SGA_{t-1} / Sales_{t-1})
    sgai = _safe_div(
        _safe_div(sga_c, rev_c),
        _safe_div(sga_p, rev_p),
    )

    # TATA = (ΔCA - ΔCash - ΔCL + ΔSTD - Dep) / TA_t
    delta_ca = ca_c - ca_p
    delta_cash = cash_c - cash_p
    delta_cl = cl_c - cl_p
    # Using accruals approach: Net Income - Operating Cash Flow (Sloan 1996)
    accruals = (ni_c - ocf_c) if ocf_c != 0 else (delta_ca - delta_cash - delta_cl - dep_c)
    tata = _safe_div(accruals, ta_c, 0.0)

    # LVGI = [(LTD_t + CL_t) / TA_t] / [(LTD_{t-1} + CL_{t-1}) / TA_{t-1}]
    lev_c = _safe_div(ltd_c + cl_c, ta_c, 0.0)
    lev_p = _safe_div(ltd_p + cl_p, ta_p, 0.0)
    lvgi = _safe_div(lev_c, lev_p)

    # Cap indices to reasonable ranges to prevent outliers dominating the score
    def _cap(x: float, lo: float = 0.01, hi: float = 10.0) -> float:
        return max(lo, min(hi, x))

    dsri = _cap(dsri)
    gmi = _cap(gmi)
    aqi = _cap(aqi)
    sgi = _cap(sgi)
    depi = _cap(depi)
    sgai = _cap(sgai)
    lvgi = _cap(lvgi)

    m_score = (
        _INTERCEPT
        + _WEIGHTS["dsri"] * dsri
        + _WEIGHTS["gmi"] * gmi
        + _WEIGHTS["aqi"] * aqi
        + _WEIGHTS["sgi"] * sgi
        + _WEIGHTS["depi"] * depi
        + _WEIGHTS["sgai"] * sgai
        + _WEIGHTS["tata"] * tata
        + _WEIGHTS["lvgi"] * lvgi
    )
    m_score = round(m_score, 4)

    if m_score > _MANIPULATOR_THRESHOLD:
        verdict = "manipulator"
        interpretation = (
            f"M-Score {m_score:.2f} > {_MANIPULATOR_THRESHOLD}: HIGH probability of "
            "earnings manipulation. Scrutinize receivables, accruals, and margin trends."
        )
    elif m_score > _SAFE_THRESHOLD:
        verdict = "grey_zone"
        interpretation = (
            f"M-Score {m_score:.2f} in grey zone ({_SAFE_THRESHOLD} to {_MANIPULATOR_THRESHOLD}). "
            "Some signs of pressure — monitor closely."
        )
    else:
        verdict = "non_manipulator"
        interpretation = (
            f"M-Score {m_score:.2f} < {_SAFE_THRESHOLD}: LOW probability of earnings manipulation."
        )

    return BeneishResult(
        symbol=fs.symbol,
        m_score=m_score,
        verdict=verdict,
        dsri=round(dsri, 4),
        gmi=round(gmi, 4),
        aqi=round(aqi, 4),
        sgi=round(sgi, 4),
        depi=round(depi, 4),
        sgai=round(sgai, 4),
        tata=round(tata, 6),
        lvgi=round(lvgi, 4),
        data_quality=fs.data_quality,
        interpretation=interpretation,
    )
