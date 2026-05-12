"""Preprocessing pipeline — orchestrates all preprocessing steps in correct order.

Pipeline order:
1. Corporate action adjustment (must be first — all downstream depends on adjusted prices)
2. Missing data handling (fill gaps before computing anything)
3. Circuit limit detection (flag circuit hits)
4. Fractional differentiation (stationarity transform)
5. Outlier treatment (winsorize computed features — last step)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import structlog

from alphavedha.data.preprocessing.circuit_handler import detect_circuit_hits
from alphavedha.data.preprocessing.corporate_actions import (
    CorporateActionRecord,
    adjust_ohlcv,
)
from alphavedha.data.preprocessing.fractional_diff import frac_diff_dataframe
from alphavedha.data.preprocessing.missing_data import (
    detect_suspensions,
    handle_missing_data,
)
from alphavedha.data.preprocessing.outlier_treatment import winsorize_features

logger = structlog.get_logger(__name__)


@dataclass
class PreprocessingResult:
    """Result of running the full preprocessing pipeline."""

    df: pd.DataFrame
    symbol: str
    rows_before: int = 0
    rows_after: int = 0
    circuit_hits: int = 0
    filled_rows: int = 0
    outlier_counts: dict[str, int] = field(default_factory=dict)
    frac_diff_d: float | None = None
    suspensions_detected: int = 0


def run_pipeline(
    df: pd.DataFrame,
    symbol: str,
    corporate_actions: list[CorporateActionRecord] | None = None,
    frac_diff_d: dict[str, float] | None = None,
    skip_frac_diff: bool = False,
    skip_outlier: bool = False,
) -> PreprocessingResult:
    """Run the full preprocessing pipeline on a single stock's OHLCV data."""
    rows_before = len(df)

    if df.empty:
        return PreprocessingResult(df=df, symbol=symbol)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if corporate_actions:
        df = adjust_ohlcv(df, corporate_actions)

    filled_before = len(df)
    df = handle_missing_data(df)
    filled_rows = len(df) - filled_before

    suspensions = detect_suspensions(df)
    n_suspensions = int(suspensions.sum())
    if n_suspensions > 0:
        df["is_suspended"] = suspensions

    df = detect_circuit_hits(df)
    n_circuits = int((df["circuit_hit"].notna()).sum()) if "circuit_hit" in df.columns else 0

    computed_d: float | None = None
    if not skip_frac_diff and len(df) > 150:
        df, used_d = frac_diff_dataframe(df, d_values=frac_diff_d, columns=["close"])
        computed_d = used_d.get("close")

    outlier_counts: dict[str, int] = {}
    if not skip_outlier:
        df, outlier_counts = winsorize_features(df)

    logger.info(
        "preprocessing_complete",
        symbol=symbol,
        rows_before=rows_before,
        rows_after=len(df),
        circuit_hits=n_circuits,
        filled_rows=filled_rows,
        suspensions=n_suspensions,
    )

    return PreprocessingResult(
        df=df,
        symbol=symbol,
        rows_before=rows_before,
        rows_after=len(df),
        circuit_hits=n_circuits,
        filled_rows=filled_rows,
        outlier_counts=outlier_counts,
        frac_diff_d=computed_d,
        suspensions_detected=n_suspensions,
    )
