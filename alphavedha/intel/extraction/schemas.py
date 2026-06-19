"""Pydantic v2 schemas for LLM-structured extraction output.

These schemas define the contract between the LLM and the extraction
pipeline.  They are used with structured-output APIs (Gemini JSON mode,
Anthropic tool_use, etc.) so the LLM returns validated, typed data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from alphavedha.intel.extraction.taxonomy import EventType


class ExtractedNumbers(BaseModel):
    """Numeric values extracted from the filing text."""

    order_value_cr: float | None = Field(
        default=None,
        description="Order/contract value in crores INR",
    )
    capacity_pct_change: float | None = Field(
        default=None,
        description="Capacity change as percentage (e.g. 30 for 30% expansion)",
    )
    rating_notches: int | None = Field(
        default=None,
        description="Rating notch change (+1 upgrade, -1 downgrade, -2 multi-notch)",
    )
    pledge_pct: float | None = Field(
        default=None,
        description="Promoter pledge as percentage of total holding",
    )
    deal_value_cr: float | None = Field(
        default=None,
        description="M&A or fund-raise deal value in crores INR",
    )
    insider_value_cr: float | None = Field(
        default=None,
        description="Insider trade value in crores INR",
    )
    revenue_cr: float | None = Field(
        default=None,
        description="Revenue figure in crores INR (from results)",
    )
    profit_cr: float | None = Field(
        default=None,
        description="Net profit figure in crores INR (from results)",
    )
    margin_pct: float | None = Field(
        default=None,
        description="Operating or net margin as percentage",
    )
    dividend_per_share: float | None = Field(
        default=None,
        description="Dividend per share in INR",
    )


class DisclosureExtraction(BaseModel):
    """Single structured event extracted from a corporate disclosure.

    This is the schema the LLM must return. One disclosure may yield
    zero (boilerplate) or one event.  Multi-event filings (rare) are
    handled by the extractor calling the LLM once and picking the
    primary event.
    """

    event_type: EventType = Field(
        description="The canonical event category from the taxonomy",
    )
    direction: int = Field(
        description="Expected stock price impact: +1 bullish, -1 bearish, 0 neutral/unclear",
        ge=-1,
        le=1,
    )
    materiality: int = Field(
        description=(
            "How material is this event for the stock price on a 0-10 scale? "
            "0 = no impact, 3 = minor, 5 = moderate, 7 = significant, 10 = transformative"
        ),
        ge=0,
        le=10,
    )
    confidence: float = Field(
        description=(
            "How confident are you in the event_type classification and direction? "
            "0.0 = guessing, 0.5 = uncertain, 0.8 = fairly sure, 1.0 = unambiguous"
        ),
        ge=0.0,
        le=1.0,
    )
    summary: str = Field(
        description="One-line summary of the event in plain English, max 200 characters",
        max_length=200,
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description=(
            "List of specific red-flag concerns, if any. Examples: "
            "'auditor resigned mid-term', 'promoter pledge above 50%', "
            "'third CFO resignation in 2 years', 'default on interest payment'"
        ),
    )
    numbers: ExtractedNumbers = Field(
        default_factory=ExtractedNumbers,
        description="Numeric values extracted from the filing",
    )


class TriageResult(BaseModel):
    """Quick triage: is this disclosure relevant or boilerplate?

    Used by the cheap T1 model to filter before full extraction.
    """

    is_relevant: bool = Field(
        description="True if this disclosure contains a material corporate event worth extracting",
    )
    category: EventType | None = Field(
        default=None,
        description="Coarse event category if relevant, None if boilerplate",
    )
    reason: str = Field(
        default="",
        description="One-line reason for the triage decision",
        max_length=150,
    )


class TranscriptDelta(BaseModel):
    """Quarter-over-quarter transcript comparison output.

    Produced by the T3 (deep) model comparing management sections
    of two consecutive quarterly transcripts for the same company.
    """

    guidance_delta: int = Field(
        description=(
            "Guidance change: -2 = significantly worse, -1 = slightly worse, "
            "0 = unchanged, +1 = slightly better, +2 = significantly better"
        ),
        ge=-2,
        le=2,
    )
    tone_delta: int = Field(
        description=(
            "Management tone change: -2 = much more defensive/evasive, "
            "-1 = slightly worse, 0 = unchanged, +1 = more confident, "
            "+2 = significantly more confident"
        ),
        ge=-2,
        le=2,
    )
    dropped_commitments: list[str] = Field(
        default_factory=list,
        description=(
            "Specific commitments or targets mentioned last quarter but NOT mentioned this quarter"
        ),
    )
    new_commitments: list[str] = Field(
        default_factory=list,
        description="New commitments or targets mentioned this quarter for the first time",
    )
    evasiveness_score: int = Field(
        description=(
            "How evasive was management in Q&A? "
            "0 = direct and transparent, 5 = moderately evasive, "
            "10 = highly evasive or deflecting"
        ),
        ge=0,
        le=10,
    )
    summary: str = Field(
        description="One-paragraph summary of what changed between quarters",
        max_length=500,
    )
