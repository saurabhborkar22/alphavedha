"""Event taxonomy for disclosure extraction.

Defines the canonical set of event types that the LLM extraction layer
maps every corporate filing into. Each type carries a default direction
(bullish/bearish/neutral) and a flag indicating whether it is a red-flag
event that requires ≥0.9 recall in the golden-set evaluation.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class EventType(StrEnum):
    """Canonical event types for Indian corporate disclosures."""

    ORDER_WIN = "order_win"
    CAPACITY_EXPANSION = "capacity_expansion"
    RESULTS_GUIDANCE = "results_guidance"
    GUIDANCE_CUT = "guidance_cut"
    FUND_RAISE = "fund_raise"
    M_AND_A = "m_and_a"
    RATING_UPGRADE = "rating_upgrade"
    RATING_DOWNGRADE = "rating_downgrade"
    OUTLOOK_CHANGE = "outlook_change"
    PLEDGE_INCREASE = "pledge_increase"
    PLEDGE_RELEASE = "pledge_release"
    INSIDER_BUY = "insider_buy"
    INSIDER_SELL = "insider_sell"
    AUDITOR_RESIGNATION = "auditor_resignation"
    KMP_RESIGNATION = "kmp_resignation"
    RELATED_PARTY_TXN = "related_party_txn"
    LITIGATION_REGULATORY = "litigation_regulatory"
    DEFAULT_OR_DELAY = "default_or_delay"
    SURVEILLANCE_ACTION = "surveillance_action"
    DIVIDEND_BUYBACK = "dividend_buyback"
    OTHER = "other"


class EventMeta(BaseModel):
    """Metadata for each event type: default direction and red-flag status."""

    event_type: EventType
    default_direction: int
    is_red_flag: bool
    description: str


EVENT_CATALOG: dict[EventType, EventMeta] = {
    EventType.ORDER_WIN: EventMeta(
        event_type=EventType.ORDER_WIN,
        default_direction=1,
        is_red_flag=False,
        description="New order/contract win with disclosed value",
    ),
    EventType.CAPACITY_EXPANSION: EventMeta(
        event_type=EventType.CAPACITY_EXPANSION,
        default_direction=1,
        is_red_flag=False,
        description="Capex announcement, new plant, capacity addition",
    ),
    EventType.RESULTS_GUIDANCE: EventMeta(
        event_type=EventType.RESULTS_GUIDANCE,
        default_direction=0,
        is_red_flag=False,
        description="Quarterly/annual results or forward guidance (direction from content)",
    ),
    EventType.GUIDANCE_CUT: EventMeta(
        event_type=EventType.GUIDANCE_CUT,
        default_direction=-1,
        is_red_flag=False,
        description="Downward revision of revenue/profit/margin guidance",
    ),
    EventType.FUND_RAISE: EventMeta(
        event_type=EventType.FUND_RAISE,
        default_direction=0,
        is_red_flag=False,
        description="QIP, rights issue, preferential allotment, NCD, debt raise",
    ),
    EventType.M_AND_A: EventMeta(
        event_type=EventType.M_AND_A,
        default_direction=0,
        is_red_flag=False,
        description="Merger, acquisition, demerger, stake sale/purchase",
    ),
    EventType.RATING_UPGRADE: EventMeta(
        event_type=EventType.RATING_UPGRADE,
        default_direction=1,
        is_red_flag=False,
        description="Credit rating upgrade by any agency",
    ),
    EventType.RATING_DOWNGRADE: EventMeta(
        event_type=EventType.RATING_DOWNGRADE,
        default_direction=-1,
        is_red_flag=True,
        description="Credit rating downgrade by any agency",
    ),
    EventType.OUTLOOK_CHANGE: EventMeta(
        event_type=EventType.OUTLOOK_CHANGE,
        default_direction=0,
        is_red_flag=False,
        description="Rating outlook change (positive/negative/stable) without notch move",
    ),
    EventType.PLEDGE_INCREASE: EventMeta(
        event_type=EventType.PLEDGE_INCREASE,
        default_direction=-1,
        is_red_flag=True,
        description="Promoter pledge creation or increase",
    ),
    EventType.PLEDGE_RELEASE: EventMeta(
        event_type=EventType.PLEDGE_RELEASE,
        default_direction=1,
        is_red_flag=False,
        description="Promoter pledge release or reduction",
    ),
    EventType.INSIDER_BUY: EventMeta(
        event_type=EventType.INSIDER_BUY,
        default_direction=1,
        is_red_flag=False,
        description="Insider/promoter purchase (PIT Reg disclosure)",
    ),
    EventType.INSIDER_SELL: EventMeta(
        event_type=EventType.INSIDER_SELL,
        default_direction=-1,
        is_red_flag=False,
        description="Insider/promoter sale (PIT Reg disclosure)",
    ),
    EventType.AUDITOR_RESIGNATION: EventMeta(
        event_type=EventType.AUDITOR_RESIGNATION,
        default_direction=-1,
        is_red_flag=True,
        description="Statutory auditor resignation or removal mid-term",
    ),
    EventType.KMP_RESIGNATION: EventMeta(
        event_type=EventType.KMP_RESIGNATION,
        default_direction=-1,
        is_red_flag=True,
        description="Key Managerial Personnel resignation (CFO, CS, CEO, MD)",
    ),
    EventType.RELATED_PARTY_TXN: EventMeta(
        event_type=EventType.RELATED_PARTY_TXN,
        default_direction=0,
        is_red_flag=False,
        description="Related party transaction disclosure (material or otherwise)",
    ),
    EventType.LITIGATION_REGULATORY: EventMeta(
        event_type=EventType.LITIGATION_REGULATORY,
        default_direction=-1,
        is_red_flag=False,
        description="Litigation, SEBI order, tax demand, regulatory action",
    ),
    EventType.DEFAULT_OR_DELAY: EventMeta(
        event_type=EventType.DEFAULT_OR_DELAY,
        default_direction=-1,
        is_red_flag=True,
        description="Loan default, interest payment delay, NPA classification",
    ),
    EventType.SURVEILLANCE_ACTION: EventMeta(
        event_type=EventType.SURVEILLANCE_ACTION,
        default_direction=-1,
        is_red_flag=True,
        description="ASM/GSM stage addition or escalation by exchange",
    ),
    EventType.DIVIDEND_BUYBACK: EventMeta(
        event_type=EventType.DIVIDEND_BUYBACK,
        default_direction=1,
        is_red_flag=False,
        description="Dividend declaration or share buyback announcement",
    ),
    EventType.OTHER: EventMeta(
        event_type=EventType.OTHER,
        default_direction=0,
        is_red_flag=False,
        description="Event that does not fit any specific category",
    ),
}

RED_FLAG_TYPES: frozenset[EventType] = frozenset(
    et for et, meta in EVENT_CATALOG.items() if meta.is_red_flag
)

BOILERPLATE_CATEGORIES: frozenset[str] = frozenset(
    {
        # Original 12
        "Trading Window",
        "Trading Window-Loss of UPSI",
        "ESOP/ESOS",
        "Allotment Of ESOP / ESOS",
        "Newspaper Publication",
        "Newspaper Ad Copy",
        "Notice Of Book Closure",
        "Record Date",
        "Annual General Meeting",
        "Annual Report",
        "Compliances-Reg. 39 (3) - Details of Loss of Certificate / Duplicate Certificate",
        "Certificate Under Reg. 74 (5) Of SEBI (DP) Regulations, 2018",
        # Routine compliance and procedural filings
        "Board Meeting - Intimation",
        "Shareholders meeting",
        "Postal Ballot",
        "Change In Address",
        "Cessation",
        "Reg. 13(3) - Investor Grievance",
        "Compliances-Certificate",
        "Reg.31(1)(a)/31(2)/75(2)",
        "Annual Secretarial Compliance",
        "Certificate under Reg. 40(9)",
        "Proceedings of AGM/EGM",
        "Investor Presentation",
        "Corporate Action-Stock Split",
    }
)

BOILERPLATE_HEADLINE_PATTERNS: tuple[str, ...] = (
    r"(?i)trading\s+window",
    r"(?i)loss\s+of\s+share\s+certificate",
    r"(?i)duplicate\s+share\s+certificate",
    r"(?i)newspaper\s+(publication|advertisement|ad)",
    r"(?i)book\s+closure",
    r"(?i)investor\s+grievance",
    r"(?i)compliance\s+certificate",
    r"(?i)secretarial\s+compliance",
    r"(?i)proceedings\s+of\s+(agm|egm)",
    r"(?i)reg\.?\s*(13|31|40|74)\s*\(",
)

ALWAYS_EXTRACT_PATTERNS: tuple[str, ...] = (
    r"(?i)(cfo|ceo|md|managing\s*director|chief\s*financial|company\s*secretary)\s*.*(resign|step\s*down|vacate)",
    r"(?i)default",
    r"(?i)auditor\s*(resign|removal|step)",
    r"(?i)fraud",
    r"(?i)sebi\s*(order|penalty|action|directive)",
    r"(?i)(asm|gsm|surveillance)",
    r"(?i)downgrad",
    r"(?i)pledge\s*(creat|increas|invoc)",
)
