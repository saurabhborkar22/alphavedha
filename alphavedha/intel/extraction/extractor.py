"""Disclosure extraction pipeline.

Takes raw disclosures, applies rule-based pre-filter, calls LLM for
structured extraction, and returns validated DisclosureExtraction objects.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from alphavedha.intel.extraction.llm import LLMProvider, load_system_prompt
from alphavedha.intel.extraction.schemas import (
    DisclosureExtraction,
    TriageResult,
)
from alphavedha.intel.extraction.taxonomy import BOILERPLATE_CATEGORIES

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

CURRENT_PROMPT_VERSION = "v1"


def is_boilerplate(category: str) -> bool:
    """Check if a disclosure category is boilerplate (skip LLM)."""
    return category in BOILERPLATE_CATEGORIES


def build_user_prompt(
    symbol: str,
    category: str,
    headline: str,
    text: str | None = None,
) -> str:
    """Build the user prompt for extraction from disclosure fields."""
    parts = [
        f"Company: {symbol}",
        f"Filing category: {category}",
        f"Headline: {headline}",
    ]
    if text:
        truncated = text[:8000]
        parts.append(f"Filing text:\n{truncated}")
    return "\n\n".join(parts)


def build_triage_prompt(
    symbol: str,
    category: str,
    headline: str,
) -> str:
    """Build a shorter prompt for T1 triage (relevant vs boilerplate)."""
    return (
        f"Company: {symbol}\n"
        f"Category: {category}\n"
        f"Headline: {headline}\n\n"
        "Is this disclosure relevant (contains a material corporate event) "
        "or boilerplate (routine compliance, no trading signal)? "
        "If relevant, what is the coarse event category?"
    )


def _coerce_llm_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM JSON output before Pydantic validation.

    Some providers (Groq/Llama) omit fields or return ints as strings.
    """
    if "confidence" not in raw:
        raw["confidence"] = 0.5
    if "summary" not in raw:
        raw["summary"] = raw.get("event_type", "unknown")
    if "red_flags" not in raw:
        raw["red_flags"] = []
    if "numbers" not in raw:
        raw["numbers"] = {}
    for int_field in ("direction", "materiality"):
        if int_field in raw and isinstance(raw[int_field], str):
            with contextlib.suppress(ValueError):
                raw[int_field] = int(raw[int_field])
    if "confidence" in raw and isinstance(raw["confidence"], str):
        with contextlib.suppress(ValueError):
            raw["confidence"] = float(raw["confidence"])
    return raw


def extract_one(
    provider: LLMProvider,
    symbol: str,
    category: str,
    headline: str,
    text: str | None = None,
    prompt_version: str = CURRENT_PROMPT_VERSION,
) -> DisclosureExtraction | None:
    """Extract a structured event from a single disclosure.

    Returns None if the disclosure is boilerplate or extraction fails.
    """
    if is_boilerplate(category):
        logger.debug("skipped_boilerplate", symbol=symbol, category=category)
        return None

    system_prompt = load_system_prompt(prompt_version)
    user_prompt = build_user_prompt(symbol, category, headline, text)

    schema = DisclosureExtraction.model_json_schema()

    result = provider.extract_json(system_prompt, user_prompt, schema)
    if result is None:
        logger.warning("extraction_returned_none", symbol=symbol)
        return None

    result = _coerce_llm_output(result)

    try:
        extraction = DisclosureExtraction.model_validate(result)
        return extraction
    except Exception as e:
        logger.warning(
            "extraction_validation_failed",
            symbol=symbol,
            error=str(e),
            raw=str(result)[:200],
        )
        return None


def triage_one(
    provider: LLMProvider,
    symbol: str,
    category: str,
    headline: str,
    prompt_version: str = CURRENT_PROMPT_VERSION,
) -> TriageResult | None:
    """Quick triage: is this disclosure relevant or boilerplate?"""
    if is_boilerplate(category):
        return TriageResult(is_relevant=False, reason="Boilerplate category")

    system_prompt = load_system_prompt(prompt_version)
    user_prompt = build_triage_prompt(symbol, category, headline)

    schema = TriageResult.model_json_schema()

    result = provider.extract_json(system_prompt, user_prompt, schema)
    if result is None:
        return None

    try:
        return TriageResult.model_validate(result)
    except Exception as e:
        logger.warning("triage_validation_failed", symbol=symbol, error=str(e))
        return None


def extract_batch(
    provider: LLMProvider,
    disclosures: list[dict[str, Any]],
    prompt_version: str = CURRENT_PROMPT_VERSION,
) -> list[dict[str, Any]]:
    """Extract events from a batch of disclosures.

    Each disclosure dict must have: id, symbol, category, headline.
    Optionally: text (PDF-extracted body).

    Returns list of dicts ready for store_disclosure_events().
    """
    events: list[dict[str, Any]] = []
    skipped = 0
    failed = 0

    for disc in disclosures:
        symbol = disc["symbol"]
        category = disc.get("category", "")
        headline = disc.get("headline", "")
        text = disc.get("text")

        extraction = extract_one(provider, symbol, category, headline, text, prompt_version)

        if extraction is None:
            if is_boilerplate(category):
                skipped += 1
            else:
                failed += 1
            continue

        events.append(
            {
                "disclosure_id": disc["id"],
                "symbol": symbol,
                "event_type": extraction.event_type.value,
                "direction": extraction.direction,
                "materiality": extraction.materiality,
                "confidence": extraction.confidence,
                "summary": extraction.summary,
                "red_flags": extraction.red_flags if extraction.red_flags else None,
                "llm_model": provider.name,
                "prompt_version": prompt_version,
                "extracted_at": datetime.now(IST),
            }
        )

    logger.info(
        "extraction_batch_complete",
        total=len(disclosures),
        extracted=len(events),
        skipped_boilerplate=skipped,
        failed=failed,
    )
    return events
