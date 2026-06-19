"""Quarter-over-quarter transcript comparison.

Compares management sections of two consecutive quarterly transcripts
for the same company and produces a TranscriptDelta: guidance change,
tone shift, dropped/new commitments, and evasiveness score.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from alphavedha.intel.extraction.llm import LLMProvider, get_provider
from alphavedha.intel.extraction.schemas import TranscriptDelta
from alphavedha.intel.store import (
    load_transcript_pairs,
    store_disclosure_events,
)

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

MAX_TRANSCRIPT_CHARS = 12000


def _truncate_text(text: str | None, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    if not text:
        return "(no transcript text available)"
    return text[:max_chars]


def build_delta_prompt(
    symbol: str,
    current_quarter: str,
    previous_quarter: str,
    current_text: str,
    previous_text: str,
) -> str:
    """Build a comparison prompt for two consecutive transcripts."""
    return (
        f"Company: {symbol}\n\n"
        f"=== PREVIOUS QUARTER ({previous_quarter}) ===\n"
        f"{_truncate_text(previous_text)}\n\n"
        f"=== CURRENT QUARTER ({current_quarter}) ===\n"
        f"{_truncate_text(current_text)}\n\n"
        "Compare these two quarterly transcripts. Focus on:\n"
        "1. Has management guidance improved or deteriorated?\n"
        "2. Has the overall tone shifted (more confident vs defensive)?\n"
        "3. What commitments from last quarter were dropped (not mentioned)?\n"
        "4. What new commitments or targets appeared this quarter?\n"
        "5. How evasive was management in the Q&A section?"
    )


DELTA_SYSTEM_PROMPT = """You are an expert equity analyst specializing in Indian markets.
You compare two consecutive quarterly earnings call transcripts for the same company.

Return a structured JSON analysis with:
- guidance_delta: -2 (significantly worse) to +2 (significantly better), 0 = unchanged
- tone_delta: -2 (much more defensive/evasive) to +2 (much more confident)
- dropped_commitments: list of specific things management promised last quarter but didn't mention now
- new_commitments: list of new targets/promises this quarter
- evasiveness_score: 0 (direct and transparent) to 10 (highly evasive)
- summary: one paragraph on what changed, max 500 chars

Be conservative — most quarters show 0 delta. Only flag -2/+2 for dramatic shifts.
Empty transcript sections mean you should note limited data and score conservatively.

Respond with valid JSON only."""


def compare_one(
    provider: LLMProvider,
    symbol: str,
    current_quarter: str,
    previous_quarter: str,
    current_text: str | None,
    previous_text: str | None,
) -> TranscriptDelta | None:
    """Compare two transcripts and return a delta."""
    user_prompt = build_delta_prompt(
        symbol,
        current_quarter,
        previous_quarter,
        current_text or "",
        previous_text or "",
    )

    schema = TranscriptDelta.model_json_schema()
    result = provider.extract_json(DELTA_SYSTEM_PROMPT, user_prompt, schema)

    if result is None:
        logger.warning("transcript_delta_failed", symbol=symbol)
        return None

    result = _coerce_delta_output(result)

    try:
        return TranscriptDelta.model_validate(result)
    except Exception as e:
        logger.warning(
            "transcript_delta_validation_failed",
            symbol=symbol,
            error=str(e),
            raw=str(result)[:200],
        )
        return None


def _coerce_delta_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM output for TranscriptDelta."""
    if "summary" not in raw:
        raw["summary"] = "No summary provided"
    if "dropped_commitments" not in raw:
        raw["dropped_commitments"] = []
    if "new_commitments" not in raw:
        raw["new_commitments"] = []
    for field in ("guidance_delta", "tone_delta", "evasiveness_score"):
        if field in raw and isinstance(raw[field], str):
            try:
                raw[field] = int(raw[field])
            except ValueError:
                raw[field] = 0
    return raw


async def process_symbol_deltas(
    symbol: str,
    provider: LLMProvider | None = None,
) -> list[dict[str, Any]]:
    """Process all transcript pairs for a symbol, return event dicts."""
    if provider is None:
        provider = get_provider()

    pairs = await load_transcript_pairs(symbol)
    if not pairs:
        logger.debug("no_transcript_pairs", symbol=symbol)
        return []

    events: list[dict[str, Any]] = []

    for current, previous in pairs:
        delta = compare_one(
            provider,
            symbol,
            current["fiscal_quarter"],
            previous["fiscal_quarter"],
            current.get("text"),
            previous.get("text"),
        )

        if delta is None:
            continue

        direction = 0
        if delta.guidance_delta > 0 and delta.tone_delta >= 0:
            direction = 1
        elif delta.guidance_delta < 0 or delta.tone_delta < -1:
            direction = -1

        materiality = min(
            10,
            abs(delta.guidance_delta) * 2
            + abs(delta.tone_delta)
            + (1 if delta.evasiveness_score >= 7 else 0),
        )

        events.append(
            {
                "disclosure_id": current["id"],
                "symbol": symbol,
                "event_type": "results_guidance",
                "direction": direction,
                "materiality": materiality,
                "confidence": 0.7,
                "summary": delta.summary[:200] if delta.summary else "Transcript delta",
                "red_flags": (
                    [f"evasiveness_score={delta.evasiveness_score}"]
                    if delta.evasiveness_score >= 7
                    else None
                ),
                "llm_model": provider.name,
                "prompt_version": "transcript_delta_v1",
                "extracted_at": datetime.now(IST),
            }
        )

    return events


async def run_transcript_deltas(
    symbols: list[str] | None = None,
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    """Run transcript delta analysis for given symbols (or all with ≥2 transcripts)."""
    if provider is None:
        provider = get_provider()

    if symbols is None:
        from alphavedha.intel.store import load_transcripts

        df = await load_transcripts(limit=10000)
        if df.empty:
            return {"status": "empty", "symbols_processed": 0, "events_stored": 0}
        counts = df.groupby("symbol").size()
        symbols = list(counts[counts >= 2].index)

    total_events = 0
    symbols_processed = 0

    for symbol in symbols:
        events = await process_symbol_deltas(symbol, provider)
        if events:
            stored = await store_disclosure_events(events)
            total_events += stored
        symbols_processed += 1

    logger.info(
        "transcript_deltas_complete",
        symbols=symbols_processed,
        events=total_events,
    )

    return {
        "status": "ok",
        "symbols_processed": symbols_processed,
        "events_stored": total_events,
    }
