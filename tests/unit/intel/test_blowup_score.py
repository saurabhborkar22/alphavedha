"""Tests for the blowup detector — composite risk scoring."""

from __future__ import annotations

from alphavedha.intel.signals.blowup_score import (
    AVOID_THRESHOLD,
    STRATEGY_NAME,
    BlowupScore,
    compute_avoid_list,
    compute_blowup_score,
    is_vetoed,
)


def _event(event_type: str = "other", direction: int = 0) -> dict[str, object]:
    return {"event_type": event_type, "direction": direction, "symbol": "TCS.NS"}


class TestComputeBlowupScore:
    def test_clean_symbol_scores_zero(self) -> None:
        score = compute_blowup_score("TCS.NS", [], [], [], [])
        assert score.total_score == 0
        assert score.on_avoid_list is False
        assert score.flags == []

    def test_pledge_critical_50pct(self) -> None:
        pledges = [{"promoter_pledge_pct": 55}]
        score = compute_blowup_score("X.NS", [], [], pledges, [])
        assert score.pledge_score == 25
        assert "pledge_critical_50pct" in score.flags

    def test_pledge_high_30pct(self) -> None:
        pledges = [{"promoter_pledge_pct": 35}]
        score = compute_blowup_score("X.NS", [], [], pledges, [])
        assert score.pledge_score == 15

    def test_pledge_rising_trend(self) -> None:
        pledges = [{"promoter_pledge_pct": 20}, {"promoter_pledge_pct": 10}]
        score = compute_blowup_score("X.NS", [], [], pledges, [])
        assert score.pledge_score == 10
        assert "pledge_rising" in score.flags

    def test_rating_downgrade(self) -> None:
        ratings = [{"action": "Downgrade", "outlook": "stable", "agency": "CRISIL"}]
        score = compute_blowup_score("X.NS", [], ratings, [], [])
        assert score.rating_score == 20
        assert "rating_downgrade_CRISIL" in score.flags

    def test_outlook_negative(self) -> None:
        ratings = [{"action": "Affirmed", "outlook": "Negative", "agency": "ICRA"}]
        score = compute_blowup_score("X.NS", [], ratings, [], [])
        assert score.rating_score == 10
        assert "outlook_negative_ICRA" in score.flags

    def test_auditor_resignation(self) -> None:
        events = [_event("auditor_resignation")]
        score = compute_blowup_score("X.NS", events, [], [], [])
        assert score.governance_score == 20
        assert "auditor_resignation" in score.flags

    def test_kmp_resignation(self) -> None:
        events = [_event("kmp_resignation")]
        score = compute_blowup_score("X.NS", events, [], [], [])
        assert score.governance_score == 15

    def test_default_or_delay(self) -> None:
        events = [_event("default_or_delay")]
        score = compute_blowup_score("X.NS", events, [], [], [])
        assert score.default_score == 25
        assert "default_or_delay" in score.flags

    def test_surveillance_active_flag(self) -> None:
        surv = [{"list_name": "ASM", "removed_on": None}]
        score = compute_blowup_score("X.NS", [], [], [], surv)
        assert score.surveillance_score == 15
        assert "surveillance_ASM" in score.flags

    def test_surveillance_removed_flag_ignored(self) -> None:
        surv = [{"list_name": "ASM", "removed_on": "2026-01-01"}]
        score = compute_blowup_score("X.NS", [], [], [], surv)
        assert score.surveillance_score == 0

    def test_beneish_manipulator(self) -> None:
        beneish = {"verdict": "manipulator", "m_score": -1.5}
        score = compute_blowup_score("X.NS", [], [], [], [], beneish_result=beneish)
        assert score.beneish_score == 15

    def test_beneish_grey_zone(self) -> None:
        beneish = {"verdict": "grey_zone", "m_score": -2.0}
        score = compute_blowup_score("X.NS", [], [], [], [], beneish_result=beneish)
        assert score.beneish_score == 5

    def test_beneish_clean(self) -> None:
        beneish = {"verdict": "non_manipulator", "m_score": -3.0}
        score = compute_blowup_score("X.NS", [], [], [], [], beneish_result=beneish)
        assert score.beneish_score == 0

    def test_insider_sell_cluster_3plus(self) -> None:
        events = [_event("insider_sell") for _ in range(3)]
        score = compute_blowup_score("X.NS", events, [], [], [])
        assert score.insider_sell_score == 15

    def test_insider_sell_cluster_2(self) -> None:
        events = [_event("insider_sell") for _ in range(2)]
        score = compute_blowup_score("X.NS", events, [], [], [])
        assert score.insider_sell_score == 10

    def test_single_insider_sell_no_score(self) -> None:
        events = [_event("insider_sell")]
        score = compute_blowup_score("X.NS", events, [], [], [])
        assert score.insider_sell_score == 0

    def test_total_capped_at_100(self) -> None:
        pledges = [{"promoter_pledge_pct": 55}]
        ratings = [{"action": "Downgrade", "outlook": "Negative", "agency": "X"}]
        events = [_event("auditor_resignation"), _event("default_or_delay")]
        events.extend([_event("insider_sell") for _ in range(3)])
        surv = [{"list_name": "GSM", "removed_on": None}]
        beneish = {"verdict": "manipulator"}
        score = compute_blowup_score("X.NS", events, ratings, pledges, surv, beneish)
        assert score.total_score == 100

    def test_avoid_list_threshold(self) -> None:
        pledges = [{"promoter_pledge_pct": 55}]
        events = [_event("default_or_delay")]
        ratings = [{"action": "Downgrade", "outlook": "stable", "agency": "X"}]
        score = compute_blowup_score("X.NS", events, ratings, pledges, [])
        assert score.total_score == 70
        assert score.on_avoid_list is True


class TestComputeAvoidList:
    def test_filters_below_threshold(self) -> None:
        scores = [
            BlowupScore(symbol="A.NS", total_score=80, on_avoid_list=True),
            BlowupScore(symbol="B.NS", total_score=30, on_avoid_list=False),
            BlowupScore(symbol="C.NS", total_score=70, on_avoid_list=True),
        ]
        avoid = compute_avoid_list(scores)
        assert len(avoid) == 2
        assert {s.symbol for s in avoid} == {"A.NS", "C.NS"}

    def test_empty_list(self) -> None:
        assert compute_avoid_list([]) == []


class TestIsVetoed:
    def test_symbol_on_avoid_list(self) -> None:
        avoid = [BlowupScore(symbol="A.NS", total_score=80, on_avoid_list=True)]
        assert is_vetoed("A.NS", avoid) is True

    def test_symbol_not_on_avoid_list(self) -> None:
        avoid = [BlowupScore(symbol="A.NS", total_score=80, on_avoid_list=True)]
        assert is_vetoed("B.NS", avoid) is False


class TestConstants:
    def test_strategy_name(self) -> None:
        assert STRATEGY_NAME == "blowup_short_v1"

    def test_avoid_threshold(self) -> None:
        assert AVOID_THRESHOLD == 70
