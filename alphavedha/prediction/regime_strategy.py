"""Regime-conditional strategy selection.

Maps HMM regime output to per-regime model weights, position sizing
multipliers, and meta-labeling thresholds. The prediction engine uses
these to adjust its behavior based on market conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

REGIME_NAMES = ("bull", "bear", "sideways", "high_volatility")


class RegimeStrategyParams(BaseModel):
    """Per-regime configuration for model weights and risk."""

    model_weights: dict[str, float] = Field(
        default_factory=lambda: {"xgboost": 0.4, "lstm": 0.3, "tft": 0.3}
    )
    kelly_fraction: float = 0.5
    meta_confidence_threshold: float = 0.55
    require_all_models_agree: bool = False


class RegimeStrategyConfig(BaseModel):
    """Configuration for all four regimes."""

    bull: RegimeStrategyParams = Field(
        default_factory=lambda: RegimeStrategyParams(
            model_weights={"xgboost": 0.45, "lstm": 0.25, "tft": 0.30},
            kelly_fraction=0.25,
            meta_confidence_threshold=0.0,
        )
    )
    bear: RegimeStrategyParams = Field(
        default_factory=lambda: RegimeStrategyParams(
            model_weights={"xgboost": 0.30, "lstm": 0.40, "tft": 0.30},
            kelly_fraction=0.15,
            meta_confidence_threshold=0.0,
        )
    )
    sideways: RegimeStrategyParams = Field(
        default_factory=lambda: RegimeStrategyParams(
            model_weights={"xgboost": 0.33, "lstm": 0.34, "tft": 0.33},
            kelly_fraction=0.25,
            meta_confidence_threshold=0.0,
        )
    )
    high_volatility: RegimeStrategyParams = Field(
        default_factory=lambda: RegimeStrategyParams(
            model_weights={"xgboost": 0.33, "lstm": 0.34, "tft": 0.33},
            kelly_fraction=0.05,
            meta_confidence_threshold=0.0,
            require_all_models_agree=True,
        )
    )


@dataclass
class StrategySelection:
    """Output of regime-conditional strategy selection."""

    regime: str
    model_weights: dict[str, float]
    kelly_fraction: float
    meta_confidence_threshold: float
    require_all_models_agree: bool
    warnings: list[str] = field(default_factory=list)


class RegimeStrategySelector:
    """Select strategy parameters based on detected regime."""

    def __init__(self, config: RegimeStrategyConfig | None = None) -> None:
        self._config = config or RegimeStrategyConfig()
        self._regime_map: dict[str, RegimeStrategyParams] = {
            "bull": self._config.bull,
            "bear": self._config.bear,
            "sideways": self._config.sideways,
            "high_volatility": self._config.high_volatility,
        }

    def select(self, regime: str) -> StrategySelection:
        """Select strategy based on regime name.

        Falls back to sideways config for unknown regimes.
        """
        warnings: list[str] = []

        if regime not in self._regime_map:
            warnings.append(f"Unknown regime '{regime}', using sideways defaults")
            regime = "sideways"

        params = self._regime_map[regime]

        logger.info(
            "regime_strategy_selected",
            regime=regime,
            kelly_fraction=params.kelly_fraction,
            meta_threshold=params.meta_confidence_threshold,
            require_unanimous=params.require_all_models_agree,
        )

        return StrategySelection(
            regime=regime,
            model_weights=dict(params.model_weights),
            kelly_fraction=params.kelly_fraction,
            meta_confidence_threshold=params.meta_confidence_threshold,
            require_all_models_agree=params.require_all_models_agree,
            warnings=warnings,
        )
