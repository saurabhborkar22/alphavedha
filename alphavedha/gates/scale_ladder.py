"""Scale decision ladder — capital tier promotion for live strategies.

Defines the capital tiers: 50k -> 2L -> 5L, each requiring a G2 gate
pass to advance. Tracks per-strategy tier state and enforces position
caps that scale with capital.

The capital path decision (own capital, family mandate, SEBI RA) is
a human decision tracked in the progress log, not in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import structlog

from alphavedha.gates.reviewer import GateVerdict

logger = structlog.get_logger(__name__)


class CapitalTier(StrEnum):
    SHADOW = "shadow"
    SMALL_LIVE = "small_live"
    MEDIUM_LIVE = "medium_live"
    FULL_LIVE = "full_live"


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a single capital tier."""

    tier: CapitalTier
    capital: float
    position_cap: float
    position_pct_cap: float
    min_weeks_before_promotion: int
    requires_gate: str


_DEFAULT_TIERS: list[TierConfig] = [
    TierConfig(
        tier=CapitalTier.SHADOW,
        capital=0.0,
        position_cap=0.0,
        position_pct_cap=0.0,
        min_weeks_before_promotion=0,
        requires_gate="G1",
    ),
    TierConfig(
        tier=CapitalTier.SMALL_LIVE,
        capital=50_000.0,
        position_cap=10_000.0,
        position_pct_cap=20.0,
        min_weeks_before_promotion=4,
        requires_gate="G2",
    ),
    TierConfig(
        tier=CapitalTier.MEDIUM_LIVE,
        capital=200_000.0,
        position_cap=40_000.0,
        position_pct_cap=20.0,
        min_weeks_before_promotion=4,
        requires_gate="G2",
    ),
    TierConfig(
        tier=CapitalTier.FULL_LIVE,
        capital=500_000.0,
        position_cap=100_000.0,
        position_pct_cap=20.0,
        min_weeks_before_promotion=0,
        requires_gate="G3",
    ),
]


@dataclass
class StrategyTierState:
    """Tracks a strategy's current position on the scale ladder."""

    strategy: str
    current_tier: CapitalTier = CapitalTier.SHADOW
    tier_entered_date: date | None = None
    weeks_at_tier: int = 0
    g2_passes: int = 0
    g2_failures: int = 0
    demoted_from: CapitalTier | None = None
    history: list[str] = field(default_factory=list)


class ScaleLadder:
    """Manages capital tier promotions and demotions for strategies."""

    def __init__(self, tiers: list[TierConfig] | None = None) -> None:
        self._tiers = tiers or list(_DEFAULT_TIERS)
        self._tier_map = {t.tier: t for t in self._tiers}
        self._tier_order = [t.tier for t in self._tiers]

    def get_tier_config(self, tier: CapitalTier) -> TierConfig:
        return self._tier_map[tier]

    def next_tier(self, current: CapitalTier) -> CapitalTier | None:
        idx = self._tier_order.index(current)
        if idx + 1 < len(self._tier_order):
            return self._tier_order[idx + 1]
        return None

    def prev_tier(self, current: CapitalTier) -> CapitalTier | None:
        idx = self._tier_order.index(current)
        if idx > 0:
            return self._tier_order[idx - 1]
        return None

    def can_promote(
        self,
        state: StrategyTierState,
        gate_verdict: GateVerdict,
    ) -> tuple[bool, str]:
        """Check if a strategy can be promoted to the next tier.

        Returns (can_promote, reason).
        """
        current_config = self._tier_map.get(state.current_tier)
        if current_config is None:
            return False, f"Unknown tier: {state.current_tier}"

        next_t = self.next_tier(state.current_tier)
        if next_t is None:
            return False, "Already at highest tier"

        if not gate_verdict.passed:
            failed = [c.name for c in gate_verdict.failed_criteria]
            return False, f"Gate review failed: {', '.join(failed)}"

        if state.weeks_at_tier < current_config.min_weeks_before_promotion:
            return (
                False,
                f"Need {current_config.min_weeks_before_promotion} weeks at tier, "
                f"have {state.weeks_at_tier}",
            )

        return True, f"Ready to promote to {next_t.value}"

    def promote(
        self,
        state: StrategyTierState,
        gate_verdict: GateVerdict,
    ) -> StrategyTierState:
        """Promote a strategy to the next tier if eligible.

        Returns updated state (or unchanged if not eligible).
        """
        can, reason = self.can_promote(state, gate_verdict)
        if not can:
            logger.warning(
                "scale_ladder_promotion_blocked",
                strategy=state.strategy,
                tier=state.current_tier.value,
                reason=reason,
            )
            return state

        next_t = self.next_tier(state.current_tier)
        assert next_t is not None

        old_tier = state.current_tier
        state.current_tier = next_t
        state.tier_entered_date = date.today()
        state.weeks_at_tier = 0
        state.g2_passes += 1
        state.history.append(f"{date.today()}: promoted {old_tier.value} -> {next_t.value}")

        logger.info(
            "scale_ladder_promoted",
            strategy=state.strategy,
            from_tier=old_tier.value,
            to_tier=next_t.value,
            capital=self._tier_map[next_t].capital,
        )

        return state

    def demote(
        self,
        state: StrategyTierState,
        reason: str = "",
    ) -> StrategyTierState:
        """Demote a strategy back one tier (e.g., slippage over budget)."""
        prev_t = self.prev_tier(state.current_tier)
        if prev_t is None:
            logger.warning(
                "scale_ladder_demote_at_bottom",
                strategy=state.strategy,
            )
            return state

        old_tier = state.current_tier
        state.demoted_from = old_tier
        state.current_tier = prev_t
        state.tier_entered_date = date.today()
        state.weeks_at_tier = 0
        state.history.append(
            f"{date.today()}: demoted {old_tier.value} -> {prev_t.value} ({reason})"
        )

        logger.warning(
            "scale_ladder_demoted",
            strategy=state.strategy,
            from_tier=old_tier.value,
            to_tier=prev_t.value,
            reason=reason,
        )

        return state

    def format_ladder(self, states: list[StrategyTierState]) -> str:
        """Format current ladder state for display."""
        lines = ["Scale Ladder Status", ""]
        for tier_config in self._tiers:
            tier = tier_config.tier
            at_tier = [s for s in states if s.current_tier == tier]
            cap_str = f"(capital: {tier_config.capital:,.0f})" if tier_config.capital > 0 else ""
            lines.append(f"  {tier.value} {cap_str}")
            if at_tier:
                for s in at_tier:
                    weeks = f"{s.weeks_at_tier}w" if s.weeks_at_tier > 0 else "new"
                    lines.append(f"    - {s.strategy} [{weeks}]")
            else:
                lines.append("    (empty)")
        return "\n".join(lines)
