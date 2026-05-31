from __future__ import annotations

from datetime import date

import pandas as pd

from alphavedha.features.corporate_events import (
    CORP_EVENTS_FEATURE_COUNT,
    compute_corporate_event_features,
)


def make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_feature_count_is_3() -> None:
    assert CORP_EVENTS_FEATURE_COUNT == 3


def test_returns_3_keys() -> None:
    result = compute_corporate_event_features("TCS.NS", date(2026, 5, 30), pd.DataFrame())
    assert set(result.keys()) == {
        "corp_days_to_next_board",
        "corp_days_since_dividend",
        "corp_event_this_week",
    }


def test_no_announcements_returns_defaults() -> None:
    result = compute_corporate_event_features("TCS.NS", date(2026, 5, 30), pd.DataFrame())
    assert result["corp_days_to_next_board"] == 999
    assert result["corp_days_since_dividend"] == 999
    assert result["corp_event_this_week"] == 0.0


def test_none_df_returns_defaults() -> None:
    result = compute_corporate_event_features("TCS.NS", date(2026, 5, 30), None)
    assert result["corp_days_to_next_board"] == 999


def test_board_meeting_today_gives_zero_days() -> None:
    df = make_df(
        [{"symbol": "TCS.NS", "announced_date": date(2026, 5, 30), "event_type": "BOARD_MEETING"}]
    )
    result = compute_corporate_event_features("TCS.NS", date(2026, 5, 30), df)
    assert result["corp_days_to_next_board"] == 0.0


def test_board_meeting_in_5_days() -> None:
    df = make_df(
        [{"symbol": "TCS.NS", "announced_date": date(2026, 6, 4), "event_type": "BOARD_MEETING"}]
    )
    result = compute_corporate_event_features("TCS.NS", date(2026, 5, 30), df)
    assert result["corp_days_to_next_board"] == 5.0
    assert result["corp_event_this_week"] == 1.0


def test_board_meeting_past_not_counted() -> None:
    df = make_df(
        [{"symbol": "TCS.NS", "announced_date": date(2026, 5, 25), "event_type": "BOARD_MEETING"}]
    )
    result = compute_corporate_event_features("TCS.NS", date(2026, 5, 30), df)
    assert result["corp_days_to_next_board"] == 999.0


def test_dividend_3_days_ago() -> None:
    df = make_df(
        [{"symbol": "TCS.NS", "announced_date": date(2026, 5, 27), "event_type": "DIVIDEND"}]
    )
    result = compute_corporate_event_features("TCS.NS", date(2026, 5, 30), df)
    assert result["corp_days_since_dividend"] == 3.0


def test_unrelated_symbol_ignored() -> None:
    df = make_df(
        [{"symbol": "INFY.NS", "announced_date": date(2026, 5, 30), "event_type": "BOARD_MEETING"}]
    )
    result = compute_corporate_event_features("TCS.NS", date(2026, 5, 30), df)
    assert result["corp_days_to_next_board"] == 999.0


def test_event_outside_7_days_not_counted() -> None:
    df = make_df([{"symbol": "TCS.NS", "announced_date": date(2026, 6, 8), "event_type": "AGM"}])
    result = compute_corporate_event_features("TCS.NS", date(2026, 5, 30), df)
    assert result["corp_event_this_week"] == 0.0


def test_all_features_are_float() -> None:
    df = make_df(
        [
            {"symbol": "TCS.NS", "announced_date": date(2026, 6, 4), "event_type": "BOARD_MEETING"},
            {"symbol": "TCS.NS", "announced_date": date(2026, 5, 20), "event_type": "DIVIDEND"},
        ]
    )
    result = compute_corporate_event_features("TCS.NS", date(2026, 5, 30), df)
    for key, val in result.items():
        assert isinstance(val, float), f"{key} is not float: {type(val)}"
