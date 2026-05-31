from __future__ import annotations

import math
from datetime import date

import pandas as pd

from alphavedha.features.trends_features import (
    TRENDS_FEATURE_COUNT,
    compute_trends_features,
)


def make_trends_df(sector: str, values: list[float], start_date: date) -> pd.DataFrame:
    from datetime import timedelta

    dates = [start_date + timedelta(days=i) for i in range(len(values))]
    return pd.DataFrame({sector: values}, index=pd.to_datetime(dates))


def test_feature_count_is_2() -> None:
    assert TRENDS_FEATURE_COUNT == 2


def test_returns_2_keys() -> None:
    result = compute_trends_features("TCS.NS", date(2026, 5, 30), None)
    assert set(result.keys()) == {"trends_sector_7d", "trends_sector_change"}


def test_none_trends_returns_nan() -> None:
    result = compute_trends_features("TCS.NS", date(2026, 5, 30), None)
    assert math.isnan(result["trends_sector_7d"])
    assert math.isnan(result["trends_sector_change"])


def test_empty_df_returns_nan() -> None:
    result = compute_trends_features("TCS.NS", date(2026, 5, 30), pd.DataFrame())
    assert math.isnan(result["trends_sector_7d"])


def test_unknown_symbol_returns_nan() -> None:
    df = make_trends_df("it", [50] * 14, date(2026, 5, 17))
    result = compute_trends_features("UNKNOWN.NS", date(2026, 5, 30), df)
    assert math.isnan(result["trends_sector_7d"])


def test_sector_7d_is_average_of_last_7_days() -> None:
    # 14 days of data; TCS.NS maps to "it"
    values = [40.0] * 7 + [60.0] * 7  # last 7 days average = 60
    df = make_trends_df("it", values, date(2026, 5, 17))
    result = compute_trends_features("TCS.NS", date(2026, 5, 30), df)
    assert abs(result["trends_sector_7d"] - 60.0) < 0.01


def test_sector_change_is_positive_when_rising() -> None:
    # prior 7 days avg = 40, recent 7 days avg = 60 → change = +20
    values = [40.0] * 7 + [60.0] * 7
    df = make_trends_df("it", values, date(2026, 5, 17))
    result = compute_trends_features("TCS.NS", date(2026, 5, 30), df)
    assert result["trends_sector_change"] > 0


def test_no_prior_data_change_is_nan() -> None:
    # only 7 days of data, no prior window
    values = [50.0] * 7
    df = make_trends_df("it", values, date(2026, 5, 24))
    result = compute_trends_features("TCS.NS", date(2026, 5, 30), df)
    assert not math.isnan(result["trends_sector_7d"])
    assert math.isnan(result["trends_sector_change"])


def test_all_values_are_float() -> None:
    values = [50.0] * 14
    df = make_trends_df("it", values, date(2026, 5, 17))
    result = compute_trends_features("TCS.NS", date(2026, 5, 30), df)
    for k, v in result.items():
        assert isinstance(v, float), f"{k} is not float"
