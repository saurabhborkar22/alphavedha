"""Unit tests for the preprocessing pipeline orchestrator."""

from __future__ import annotations

from datetime import date

import pandas as pd

from alphavedha.data.preprocessing.corporate_actions import CorporateActionRecord
from alphavedha.data.preprocessing.pipeline import PreprocessingResult, run_pipeline


class TestPipeline:
    def test_pipeline_returns_result(self, sample_ohlcv: pd.DataFrame):
        result = run_pipeline(sample_ohlcv, symbol="TCS", skip_frac_diff=True, skip_outlier=True)
        assert isinstance(result, PreprocessingResult)
        assert result.symbol == "TCS"
        assert result.rows_after >= result.rows_before

    def test_pipeline_adds_circuit_column(self, sample_ohlcv: pd.DataFrame):
        result = run_pipeline(sample_ohlcv, symbol="TCS", skip_frac_diff=True, skip_outlier=True)
        assert "circuit_hit" in result.df.columns

    def test_pipeline_with_corporate_actions(self, sample_ohlcv: pd.DataFrame):
        actions = [
            CorporateActionRecord(
                symbol="TCS",
                ex_date=date(2024, 1, 15),
                action_type="split",
                ratio=2.0,
            )
        ]
        result = run_pipeline(
            sample_ohlcv,
            symbol="TCS",
            corporate_actions=actions,
            skip_frac_diff=True,
            skip_outlier=True,
        )
        assert result.df["is_adjusted"].all()
        assert "raw_close" in result.df.columns

    def test_pipeline_fills_missing_data(self, sample_ohlcv_with_gaps: pd.DataFrame):
        result = run_pipeline(
            sample_ohlcv_with_gaps,
            symbol="TCS",
            skip_frac_diff=True,
            skip_outlier=True,
        )
        assert result.filled_rows > 0

    def test_pipeline_empty_df(self):
        result = run_pipeline(pd.DataFrame(), symbol="EMPTY")
        assert result.df.empty

    def test_pipeline_full_run(self, sample_ohlcv_long: pd.DataFrame):
        result = run_pipeline(
            sample_ohlcv_long,
            symbol="TCS",
            frac_diff_d={"close": 0.4},
        )
        assert "close_fracdiff" in result.df.columns
        assert result.rows_after > 0
