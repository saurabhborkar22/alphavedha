from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

CORP_EVENTS_FEATURE_COUNT = 3

_BOARD_TYPES = frozenset({"BOARD_MEETING"})
_DIV_TYPES = frozenset({"DIVIDEND"})
_ANY_EVENT_TYPES = frozenset(
    {"BOARD_MEETING", "DIVIDEND", "BONUS", "RIGHTS", "BUYBACK", "SPLIT", "AGM", "EGM", "OTHER"}
)
_NO_DATA = 999


def compute_corporate_event_features(
    symbol: str,
    as_of_date: date,
    announcements_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute 3 corporate event features for a symbol as of a given date.

    All features use only data where announced_date <= as_of_date (no look-ahead).

    Features:
        corp_days_to_next_board: days until next board meeting (0 if today, 999 if none)
        corp_days_since_dividend: days since last dividend announcement (999 if none in history)
        corp_event_this_week: 1.0 if any event within next 7 days, else 0.0

    Args:
        symbol: NSE symbol e.g. "TCS.NS"
        as_of_date: The date we are computing features for
        announcements_df: DataFrame with columns [symbol, announced_date, event_type]

    Returns:
        dict with exactly 3 keys matching the feature names above
    """
    defaults: dict[str, float] = {
        "corp_days_to_next_board": float(_NO_DATA),
        "corp_days_since_dividend": float(_NO_DATA),
        "corp_event_this_week": 0.0,
    }

    if announcements_df is None or announcements_df.empty:
        return defaults

    sym_df = announcements_df[announcements_df["symbol"] == symbol].copy()
    if sym_df.empty:
        return defaults

    sym_df["announced_date"] = pd.to_datetime(sym_df["announced_date"]).dt.date

    # corp_days_to_next_board: next board meeting on or after today
    future_board = sym_df[
        sym_df["event_type"].isin(_BOARD_TYPES) & (sym_df["announced_date"] >= as_of_date)
    ]
    days_to_board = (
        (future_board["announced_date"].min() - as_of_date).days
        if not future_board.empty
        else _NO_DATA
    )

    # corp_days_since_dividend: most recent past dividend
    past_div = sym_df[
        sym_df["event_type"].isin(_DIV_TYPES) & (sym_df["announced_date"] < as_of_date)
    ]
    days_since_div = (
        (as_of_date - past_div["announced_date"].max()).days if not past_div.empty else _NO_DATA
    )

    # corp_event_this_week: any event in next 7 calendar days
    week_end = as_of_date + timedelta(days=7)
    upcoming = sym_df[
        sym_df["event_type"].isin(_ANY_EVENT_TYPES)
        & (sym_df["announced_date"] >= as_of_date)
        & (sym_df["announced_date"] <= week_end)
    ]
    event_this_week = 1.0 if not upcoming.empty else 0.0

    logger.debug(
        "corp_events.computed",
        symbol=symbol,
        as_of=str(as_of_date),
        days_to_board=days_to_board,
        days_since_div=days_since_div,
        event_this_week=event_this_week,
    )

    return {
        "corp_days_to_next_board": float(days_to_board),
        "corp_days_since_dividend": float(days_since_div),
        "corp_event_this_week": event_this_week,
    }
