"""Feature pipeline — orchestrates all 7 feature modules.

Calls each module, concatenates into a single DataFrame, validates
(no NaN after fill, no inf), and optionally stores to the feature store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

from alphavedha.features.calendar_features import compute_calendar_features
from alphavedha.features.derivatives import compute_derivatives_features
from alphavedha.features.fundamental_features import compute_fundamental_features
from alphavedha.features.macro import compute_macro_features
from alphavedha.features.microstructure import compute_microstructure_features
from alphavedha.features.returns import compute_return_features
from alphavedha.features.sentiment import compute_sentiment_features
from alphavedha.features.technical import compute_technical_features

logger = structlog.get_logger(__name__)

EXPECTED_FEATURE_COUNT = 150


@dataclass
class FeatureResult:
    """Result of computing all features for a symbol."""

    df: pd.DataFrame
    symbol: str
    feature_count: int = 0
    nan_filled_count: int = 0
    computation_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


def compute_all_features(
    symbol: str,
    ohlcv_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    fii_dii_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
    deriv_df: pd.DataFrame | None = None,
    daily_articles: dict[str, list[str]] | None = None,
    earnings_df: pd.DataFrame | None = None,
    frac_diff_col: str | None = None,
) -> FeatureResult:
    """Compute all features for a single symbol.

    Pipeline order:
    1. technical (40) — needs only OHLCV
    2. returns (20) — needs OHLCV + optional frac_diff
    3. calendar (18) — needs only DatetimeIndex
    4. microstructure (13) — needs delivery_pct column
    5. macro (25) — needs macro_df
    6. derivatives (20) — needs deriv_df
    7. sentiment (8) — needs daily_articles
    8. fundamental (5) — needs earnings_df
    """
    start_time = time.perf_counter()
    warnings: list[str] = []

    technical = compute_technical_features(ohlcv_df)
    returns = compute_return_features(ohlcv_df, frac_diff_col=frac_diff_col)
    calendar = compute_calendar_features(ohlcv_df)
    micro = compute_microstructure_features(ohlcv_df)
    macro = compute_macro_features(ohlcv_df, macro_df, fii_dii_df, sector_df)
    deriv = compute_derivatives_features(ohlcv_df, deriv_df)
    sentiment = compute_sentiment_features(ohlcv_df, daily_articles)
    fundamental = compute_fundamental_features(ohlcv_df, earnings_df)

    all_features = pd.concat(
        [technical, returns, calendar, micro, macro, deriv, sentiment, fundamental],
        axis=1,
    )

    all_features = all_features.replace([np.inf, -np.inf], np.nan)

    high_nan_cols = []
    for col in all_features.columns:
        nan_pct = all_features[col].isna().mean()
        if nan_pct > 0.5:
            high_nan_cols.append(f"{col} ({nan_pct:.0%})")
    if high_nan_cols:
        warnings.append(f"High NaN columns: {', '.join(high_nan_cols[:10])}")

    nan_before = int(all_features.isna().sum().sum())
    all_features = all_features.ffill().bfill().fillna(0.0)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    n_features = len(all_features.columns)
    if n_features != EXPECTED_FEATURE_COUNT:
        warnings.append(f"Expected {EXPECTED_FEATURE_COUNT} features, got {n_features}")

    logger.info(
        "all_features_computed",
        symbol=symbol,
        n_features=n_features,
        n_rows=len(all_features),
        nan_filled=nan_before,
        time_ms=round(elapsed_ms, 1),
        warnings=warnings or None,
    )

    return FeatureResult(
        df=all_features,
        symbol=symbol,
        feature_count=n_features,
        nan_filled_count=nan_before,
        computation_time_ms=elapsed_ms,
        warnings=warnings,
    )
