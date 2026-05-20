"""Market impact model — Almgren-Chriss inspired impact estimation for Indian markets.

Estimates temporary and permanent price impact for a given order size,
and recommends execution strategies based on participation rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ImpactEstimate:
    temporary_impact_pct: float
    permanent_impact_pct: float
    total_impact_pct: float
    execution_cost_pct: float  # total cost including slippage + impact
    participation_rate: float  # order_size / avg_volume
    is_feasible: bool  # True if participation_rate < 20%
    recommendation: str


class MarketImpactModel:
    """Almgren-Chriss inspired market impact model adapted for Indian markets.

    Temporary impact: eta * sigma * (n/V)^0.6
    Permanent impact: gamma * sigma * (n/V)

    Where:
      eta = temporary impact coefficient (calibrated per cap tier)
      gamma = permanent impact coefficient
      sigma = daily volatility
      n = order size (shares)
      V = average daily volume
    """

    IMPACT_PARAMS: ClassVar[dict[str, dict[str, float]]] = {
        "large": {"eta": 0.05, "gamma": 0.01},
        "mid": {"eta": 0.15, "gamma": 0.03},
        "small": {"eta": 0.40, "gamma": 0.08},
    }

    def __init__(self) -> None:
        pass

    def estimate_impact(
        self,
        order_size_shares: int,
        avg_daily_volume: float,
        daily_volatility: float,
        cap_tier: str = "large",
        price: float = 100.0,
    ) -> ImpactEstimate:
        """Estimate market impact for a given order."""
        cap_tier = cap_tier.lower()
        params = self.IMPACT_PARAMS.get(cap_tier, self.IMPACT_PARAMS["mid"])
        eta = params["eta"]
        gamma = params["gamma"]

        participation_rate = order_size_shares / avg_daily_volume if avg_daily_volume > 0 else 1.0
        is_feasible = participation_rate < 0.20

        temporary_impact = eta * daily_volatility * (participation_rate ** 0.6)
        permanent_impact = gamma * daily_volatility * participation_rate
        total_impact = temporary_impact + permanent_impact

        bid_ask_cost = 0.0005 if cap_tier == "large" else (0.001 if cap_tier == "mid" else 0.002)
        execution_cost = total_impact + bid_ask_cost

        recommendation = self._recommend(participation_rate)

        logger.info(
            "impact_estimated",
            cap_tier=cap_tier,
            participation_rate=round(participation_rate, 4),
            temporary_impact_pct=round(temporary_impact, 4),
            permanent_impact_pct=round(permanent_impact, 4),
            total_impact_pct=round(total_impact, 4),
            is_feasible=is_feasible,
        )

        return ImpactEstimate(
            temporary_impact_pct=round(temporary_impact, 6),
            permanent_impact_pct=round(permanent_impact, 6),
            total_impact_pct=round(total_impact, 6),
            execution_cost_pct=round(execution_cost, 6),
            participation_rate=round(participation_rate, 6),
            is_feasible=is_feasible,
            recommendation=recommendation,
        )

    def optimal_execution_horizon(
        self,
        order_size_shares: int,
        avg_daily_volume: float,
        urgency: float = 0.5,
    ) -> int:
        """Return optimal execution time in minutes.

        Based on participation rate and urgency (0=patient, 1=urgent).
        Higher urgency = faster execution but more impact.
        """
        urgency = max(0.0, min(1.0, urgency))
        participation_rate = order_size_shares / avg_daily_volume if avg_daily_volume > 0 else 1.0

        base_minutes = participation_rate * 375 * 4  # 375 = market minutes per day
        patience_factor = 2.0 - urgency  # patient=2.0x, urgent=1.0x
        horizon = base_minutes * patience_factor

        horizon = max(5, min(round(horizon), 375))

        logger.debug(
            "optimal_horizon_computed",
            participation_rate=round(participation_rate, 4),
            urgency=urgency,
            horizon_minutes=horizon,
        )

        return horizon

    @staticmethod
    def _recommend(participation_rate: float) -> str:
        if participation_rate < 0.05:
            return "execute normally"
        if participation_rate < 0.10:
            return "split into tranches"
        if participation_rate < 0.20:
            return "use VWAP/TWAP over 2+ hours"
        return "order too large for daily volume"
