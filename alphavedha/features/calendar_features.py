"""Calendar features — 18 features from dates.

Pure computation from DatetimeIndex — no external data needed.
Column naming: cal_{indicator}.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

CALENDAR_FEATURE_COUNT = 18

RBI_POLICY_MONTHS = {2, 4, 6, 8, 10, 12}
RESULT_SEASON_MONTHS = {1, 4, 7, 10}
BUDGET_MONTH = 2
MONSOON_MONTHS = {6, 7, 8, 9}


def _last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of a given month (F&O monthly expiry)."""
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    days_back = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=days_back)


def _days_to_next_expiry(d: date) -> int:
    """Days from date d to the next monthly F&O expiry (last Thursday)."""
    expiry = _last_thursday_of_month(d.year, d.month)
    if d > expiry:
        if d.month == 12:
            expiry = _last_thursday_of_month(d.year + 1, 1)
        else:
            expiry = _last_thursday_of_month(d.year, d.month + 1)
    return (expiry - d).days


def _is_expiry_day(d: date) -> int:
    """1 if date is a monthly F&O expiry day (last Thursday)."""
    expiry = _last_thursday_of_month(d.year, d.month)
    return 1 if d == expiry else 0


def _is_expiry_week(d: date) -> int:
    """1 if date falls in the week of monthly F&O expiry."""
    expiry = _last_thursday_of_month(d.year, d.month)
    week_start = expiry - timedelta(days=expiry.weekday())
    week_end = week_start + timedelta(days=4)
    return 1 if week_start <= d <= week_end else 0


def _days_to_next_rbi(d: date) -> int:
    """Approximate days to next RBI policy meeting (bi-monthly, first week)."""
    for offset in range(13):
        m = d.month + offset
        y = d.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        if m in RBI_POLICY_MONTHS:
            target = date(y, m, 7)
            if target >= d:
                return (target - d).days
    return 30


def _week_of_month(d: date) -> int:
    """Week number within the month (1-5)."""
    return (d.day - 1) // 7 + 1


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 18 calendar features from DatetimeIndex.

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        DataFrame with 18 cal_* columns, same index as input.
    """
    idx = df.index
    dates = idx.date if hasattr(idx, "date") else pd.to_datetime(idx).date

    result = pd.DataFrame(index=df.index)

    result["cal_dow"] = idx.dayofweek
    result["cal_month"] = idx.month
    result["cal_quarter"] = idx.quarter
    result["cal_week_of_month"] = pd.Series(
        [_week_of_month(d) for d in dates],
        index=df.index,
    )

    result["cal_days_to_monthly_expiry"] = pd.Series(
        [_days_to_next_expiry(d) for d in dates],
        index=df.index,
    )
    result["cal_is_expiry_week"] = pd.Series(
        [_is_expiry_week(d) for d in dates],
        index=df.index,
    )
    result["cal_is_expiry_day"] = pd.Series(
        [_is_expiry_day(d) for d in dates],
        index=df.index,
    )

    result["cal_days_to_rbi"] = pd.Series(
        [_days_to_next_rbi(d) for d in dates],
        index=df.index,
    )
    result["cal_is_budget_month"] = (idx.month == BUDGET_MONTH).astype(int)

    result["cal_is_january"] = (idx.month == 1).astype(int)
    result["cal_is_december"] = (idx.month == 12).astype(int)
    result["cal_monsoon_flag"] = idx.month.isin(MONSOON_MONTHS).astype(int)

    result["cal_is_result_season"] = idx.month.isin(RESULT_SEASON_MONTHS).astype(int)

    result["cal_doy"] = idx.dayofyear
    result["cal_year"] = idx.year
    result["cal_week_of_year"] = idx.isocalendar().week.astype(int).to_numpy()

    result["cal_is_monday"] = (idx.dayofweek == 0).astype(int)
    result["cal_days_in_quarter"] = pd.Series(
        [(d - date(d.year, ((d.month - 1) // 3) * 3 + 1, 1)).days + 1 for d in dates],
        index=df.index,
    )

    logger.info(
        "calendar_features_computed",
        n_features=len(result.columns),
        n_rows=len(result),
    )
    return result
