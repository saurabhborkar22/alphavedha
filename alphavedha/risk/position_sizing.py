"""Half-Kelly position sizing — compute optimal position % from meta-confidence and magnitude."""

from __future__ import annotations

import structlog

from alphavedha.config import PositionSizingConfig

logger = structlog.get_logger(__name__)


def compute_position_size(
    meta_confidence: float,
    magnitude: float,
    config: PositionSizingConfig,
) -> float:
    if meta_confidence < config.min_confidence:
        return 0.0

    if magnitude <= 0.0:
        return 0.0

    # Symmetric Kelly (assumes avg_win ≈ avg_loss). TODO: upgrade to
    # generalized Kelly f = (p·b - q) / b once we have calibrated
    # win/loss ratio estimates per regime.
    kelly_fraction = 2 * meta_confidence - 1

    if kelly_fraction <= 0.0:
        return 0.0

    half_kelly_pct = kelly_fraction * 0.5 * 100

    position_pct = min(half_kelly_pct, config.max_single_stock_pct)

    logger.debug(
        "position_size_computed",
        meta_confidence=round(meta_confidence, 4),
        magnitude=round(magnitude, 6),
        kelly_fraction=round(kelly_fraction, 4),
        position_pct=round(position_pct, 4),
    )

    return position_pct
