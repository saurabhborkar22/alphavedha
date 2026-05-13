"""Tests for calendar feature computation."""

from __future__ import annotations

from datetime import date

import pandas as pd

from alphavedha.features.calendar_features import (
    CALENDAR_FEATURE_COUNT,
    _is_expiry_day,
    _last_thursday_of_month,
    compute_calendar_features,
)


class TestLastThursday:
    def test_jan_2024(self) -> None:
        assert _last_thursday_of_month(2024, 1) == date(2024, 1, 25)

    def test_feb_2024(self) -> None:
        assert _last_thursday_of_month(2024, 2) == date(2024, 2, 29)

    def test_dec_2024(self) -> None:
        assert _last_thursday_of_month(2024, 12) == date(2024, 12, 26)

    def test_result_is_thursday(self) -> None:
        for month in range(1, 13):
            result = _last_thursday_of_month(2024, month)
            assert result.weekday() == 3, f"Month {month}: {result} is not Thursday"


class TestCalendarFeatures:
    def test_returns_correct_count(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_calendar_features(sample_ohlcv_long)
        assert len(result.columns) == CALENDAR_FEATURE_COUNT

    def test_dow_bounded(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_calendar_features(sample_ohlcv)
        assert result["cal_dow"].between(0, 4).all()

    def test_month_bounded(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_calendar_features(sample_ohlcv_long)
        assert result["cal_month"].between(1, 12).all()

    def test_monsoon_flag(self) -> None:
        dates = pd.bdate_range("2024-01-01", "2024-12-31", freq="B")
        df = pd.DataFrame({"close": 100.0}, index=dates)
        result = compute_calendar_features(df)
        jun_sep = result.loc[result["cal_month"].isin([6, 7, 8, 9]), "cal_monsoon_flag"]
        assert (jun_sep == 1).all()
        non_monsoon = result.loc[~result["cal_month"].isin([6, 7, 8, 9]), "cal_monsoon_flag"]
        assert (non_monsoon == 0).all()

    def test_expiry_day_detection(self) -> None:
        assert _is_expiry_day(date(2024, 1, 25)) == 1
        assert _is_expiry_day(date(2024, 1, 24)) == 0

    def test_days_to_expiry_nonnegative(self, sample_ohlcv_long: pd.DataFrame) -> None:
        result = compute_calendar_features(sample_ohlcv_long)
        assert (result["cal_days_to_monthly_expiry"] >= 0).all()

    def test_budget_month(self) -> None:
        dates = pd.bdate_range("2024-02-01", periods=20, freq="B")
        df = pd.DataFrame({"close": 100.0}, index=dates)
        result = compute_calendar_features(df)
        assert (result["cal_is_budget_month"] == 1).all()

    def test_result_season(self) -> None:
        dates = pd.bdate_range("2024-01-01", "2024-12-31", freq="B")
        df = pd.DataFrame({"close": 100.0}, index=dates)
        result = compute_calendar_features(df)
        result_months = result.loc[result["cal_month"].isin([1, 4, 7, 10]), "cal_is_result_season"]
        assert (result_months == 1).all()
