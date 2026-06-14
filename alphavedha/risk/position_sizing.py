"""Generalized half-Kelly position sizing.

Formula: f* = p - q/b  (generalized Kelly)
  p   = meta_confidence (P(direction correct))
  q   = 1 - p
  b   = magnitude / magnitude_loss_ref  (expected-win / expected-loss ratio)

Symmetric Kelly (b=1) gives f* = 2p-1, which the old code used.  Wiring in
magnitude means larger expected moves justify larger positions and vice-versa.
Half-Kelly (f*/2) is applied for conservatism, then capped at max_single_stock_pct.
"""

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

    q = 1.0 - meta_confidence
    b = magnitude / config.magnitude_loss_ref  # win/loss ratio

    kelly_fraction = meta_confidence - q / b  # generalized Kelly: p - q/b

    if kelly_fraction <= 0.0:
        return 0.0

    half_kelly_pct = kelly_fraction * 0.5 * 100.0

    position_pct = min(half_kelly_pct, config.max_single_stock_pct)

    logger.debug(
        "position_size_computed",
        meta_confidence=round(meta_confidence, 4),
        magnitude=round(magnitude, 6),
        win_loss_ratio=round(b, 4),
        kelly_fraction=round(kelly_fraction, 4),
        position_pct=round(position_pct, 4),
    )

    return position_pct
