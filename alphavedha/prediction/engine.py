"""PredictionEngine — orchestrates the full prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class StockPrediction:
    symbol: str
    timestamp: datetime
    direction: int
    magnitude: float
    composite_score: float
    meta_confidence: float
    is_tradeable: bool
    regime: str
    regime_probabilities: np.ndarray
    price_target_low: float
    price_target_mid: float
    price_target_high: float
    model_disagreement: float
    position_size_pct: float
    model_version: str
    warnings: list[str] = field(default_factory=list)
