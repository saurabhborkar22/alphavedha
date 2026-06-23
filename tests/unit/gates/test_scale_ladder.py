"""Tests for the scale decision ladder."""

from __future__ import annotations

from datetime import date

import pytest

from alphavedha.gates.reviewer import (
    CriterionResult,
    GateLevel,
    GateVerdict,
)
from alphavedha.gates.scale_ladder import (
    CapitalTier,
    ScaleLadder,
    StrategyTierState,
    TierConfig,
)


@pytest.fixture
def ladder() -> ScaleLadder:
    return ScaleLadder()


def _passing_verdict(gate: GateLevel = GateLevel.G1) -> GateVerdict:
    return GateVerdict(
        strategy="ensemble_v1",
        gate=gate,
        review_date=date.today(),
        passed=True,
        criteria_results=[CriterionResult(name="test", passed=True, observed="ok", threshold="ok")],
        recommendation="PROMOTE",
    )


def _failing_verdict(gate: GateLevel = GateLevel.G1) -> GateVerdict:
    return GateVerdict(
        strategy="ensemble_v1",
        gate=gate,
        review_date=date.today(),
        passed=False,
        criteria_results=[
            CriterionResult(name="min_cohorts", passed=False, observed="20", threshold=">= 30")
        ],
        recommendation="STAY",
    )


class TestTierNavigation:
    def test_next_tier_from_shadow(self, ladder: ScaleLadder) -> None:
        assert ladder.next_tier(CapitalTier.SHADOW) == CapitalTier.SMALL_LIVE

    def test_next_tier_from_small(self, ladder: ScaleLadder) -> None:
        assert ladder.next_tier(CapitalTier.SMALL_LIVE) == CapitalTier.MEDIUM_LIVE

    def test_next_tier_from_full(self, ladder: ScaleLadder) -> None:
        assert ladder.next_tier(CapitalTier.FULL_LIVE) is None

    def test_prev_tier_from_shadow(self, ladder: ScaleLadder) -> None:
        assert ladder.prev_tier(CapitalTier.SHADOW) is None

    def test_prev_tier_from_medium(self, ladder: ScaleLadder) -> None:
        assert ladder.prev_tier(CapitalTier.MEDIUM_LIVE) == CapitalTier.SMALL_LIVE

    def test_get_tier_config(self, ladder: ScaleLadder) -> None:
        config = ladder.get_tier_config(CapitalTier.SMALL_LIVE)
        assert config.capital == 50_000.0
        assert config.position_cap == 10_000.0


class TestPromotion:
    def test_shadow_to_small_live(self, ladder: ScaleLadder) -> None:
        state = StrategyTierState(strategy="ensemble_v1")
        verdict = _passing_verdict(GateLevel.G1)
        can, _reason = ladder.can_promote(state, verdict)
        assert can is True

    def test_promote_updates_state(self, ladder: ScaleLadder) -> None:
        state = StrategyTierState(strategy="ensemble_v1")
        verdict = _passing_verdict(GateLevel.G1)
        state = ladder.promote(state, verdict)
        assert state.current_tier == CapitalTier.SMALL_LIVE
        assert state.weeks_at_tier == 0
        assert state.g2_passes == 1
        assert len(state.history) == 1

    def test_small_to_medium_requires_weeks(self, ladder: ScaleLadder) -> None:
        state = StrategyTierState(
            strategy="ensemble_v1",
            current_tier=CapitalTier.SMALL_LIVE,
            weeks_at_tier=2,
        )
        verdict = _passing_verdict(GateLevel.G2)
        can, reason = ladder.can_promote(state, verdict)
        assert can is False
        assert "4 weeks" in reason

    def test_small_to_medium_after_4_weeks(self, ladder: ScaleLadder) -> None:
        state = StrategyTierState(
            strategy="ensemble_v1",
            current_tier=CapitalTier.SMALL_LIVE,
            weeks_at_tier=4,
        )
        verdict = _passing_verdict(GateLevel.G2)
        can, _reason = ladder.can_promote(state, verdict)
        assert can is True

    def test_promotion_blocked_by_gate_failure(self, ladder: ScaleLadder) -> None:
        state = StrategyTierState(strategy="ensemble_v1")
        verdict = _failing_verdict()
        can, reason = ladder.can_promote(state, verdict)
        assert can is False
        assert "failed" in reason.lower()

    def test_already_at_highest_tier(self, ladder: ScaleLadder) -> None:
        state = StrategyTierState(
            strategy="ensemble_v1",
            current_tier=CapitalTier.FULL_LIVE,
        )
        verdict = _passing_verdict(GateLevel.G3)
        can, reason = ladder.can_promote(state, verdict)
        assert can is False
        assert "highest" in reason.lower()

    def test_promote_when_blocked_returns_unchanged(self, ladder: ScaleLadder) -> None:
        state = StrategyTierState(strategy="ensemble_v1")
        verdict = _failing_verdict()
        result = ladder.promote(state, verdict)
        assert result.current_tier == CapitalTier.SHADOW


class TestDemotion:
    def test_demote_from_medium(self, ladder: ScaleLadder) -> None:
        state = StrategyTierState(
            strategy="ensemble_v1",
            current_tier=CapitalTier.MEDIUM_LIVE,
            weeks_at_tier=3,
        )
        state = ladder.demote(state, "slippage over budget")
        assert state.current_tier == CapitalTier.SMALL_LIVE
        assert state.demoted_from == CapitalTier.MEDIUM_LIVE
        assert state.weeks_at_tier == 0
        assert "demoted" in state.history[-1]

    def test_demote_from_shadow_stays(self, ladder: ScaleLadder) -> None:
        state = StrategyTierState(strategy="ensemble_v1")
        state = ladder.demote(state, "test")
        assert state.current_tier == CapitalTier.SHADOW

    def test_demotion_records_reason(self, ladder: ScaleLadder) -> None:
        state = StrategyTierState(
            strategy="ensemble_v1",
            current_tier=CapitalTier.SMALL_LIVE,
        )
        state = ladder.demote(state, "kill switch breach")
        assert "kill switch breach" in state.history[-1]


class TestDefaultTiers:
    def test_four_tiers(self, ladder: ScaleLadder) -> None:
        assert len(ladder._tiers) == 4

    def test_capital_progression(self, ladder: ScaleLadder) -> None:
        capitals = [t.capital for t in ladder._tiers]
        assert capitals == [0.0, 50_000.0, 200_000.0, 500_000.0]

    def test_position_cap_progression(self, ladder: ScaleLadder) -> None:
        caps = [t.position_cap for t in ladder._tiers]
        assert caps == [0.0, 10_000.0, 40_000.0, 100_000.0]


class TestCustomTiers:
    def test_custom_two_tier(self) -> None:
        tiers = [
            TierConfig(
                tier=CapitalTier.SHADOW,
                capital=0,
                position_cap=0,
                position_pct_cap=0,
                min_weeks_before_promotion=0,
                requires_gate="G1",
            ),
            TierConfig(
                tier=CapitalTier.SMALL_LIVE,
                capital=25_000,
                position_cap=5_000,
                position_pct_cap=20,
                min_weeks_before_promotion=2,
                requires_gate="G2",
            ),
        ]
        ladder = ScaleLadder(tiers=tiers)
        assert len(ladder._tiers) == 2
        config = ladder.get_tier_config(CapitalTier.SMALL_LIVE)
        assert config.capital == 25_000.0


class TestFormatLadder:
    def test_format_empty(self, ladder: ScaleLadder) -> None:
        text = ladder.format_ladder([])
        assert "Scale Ladder" in text
        assert "(empty)" in text

    def test_format_with_strategies(self, ladder: ScaleLadder) -> None:
        states = [
            StrategyTierState(
                strategy="ensemble_v1",
                current_tier=CapitalTier.SMALL_LIVE,
                weeks_at_tier=3,
            ),
            StrategyTierState(
                strategy="event_drift_v1",
                current_tier=CapitalTier.SHADOW,
            ),
        ]
        text = ladder.format_ladder(states)
        assert "ensemble_v1" in text
        assert "event_drift_v1" in text
        assert "3w" in text
