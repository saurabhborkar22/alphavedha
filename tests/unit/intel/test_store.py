"""Tests for intel store functions — import and signature validation."""

from __future__ import annotations

import inspect


def test_store_module_importable() -> None:
    from alphavedha.intel import store

    assert hasattr(store, "store_disclosures")
    assert hasattr(store, "load_disclosures")
    assert hasattr(store, "store_disclosure_events")
    assert hasattr(store, "load_disclosure_events")
    assert hasattr(store, "store_rating_events")
    assert hasattr(store, "store_pledge_snapshots")
    assert hasattr(store, "store_surveillance_flags")
    assert hasattr(store, "store_bulk_block_deals")
    assert hasattr(store, "store_transcripts")
    assert hasattr(store, "mark_disclosures_processed")
    assert hasattr(store, "load_disclosures_by_ids")


def test_store_functions_are_async() -> None:
    from alphavedha.intel.store import (
        load_disclosure_events,
        load_disclosures,
        load_disclosures_by_ids,
        mark_disclosures_processed,
        store_bulk_block_deals,
        store_disclosure_events,
        store_disclosures,
        store_pledge_snapshots,
        store_rating_events,
        store_surveillance_flags,
        store_transcripts,
    )

    for fn in [
        store_disclosures,
        load_disclosures,
        store_disclosure_events,
        load_disclosure_events,
        store_rating_events,
        store_pledge_snapshots,
        store_surveillance_flags,
        store_bulk_block_deals,
        store_transcripts,
        mark_disclosures_processed,
        load_disclosures_by_ids,
    ]:
        assert inspect.iscoroutinefunction(fn), f"{fn.__name__} should be async"


def test_store_disclosures_accepts_list() -> None:
    from alphavedha.intel.store import store_disclosures

    sig = inspect.signature(store_disclosures)
    params = list(sig.parameters.keys())
    assert "rows" in params


def test_load_disclosures_has_filters() -> None:
    from alphavedha.intel.store import load_disclosures

    sig = inspect.signature(load_disclosures)
    params = set(sig.parameters.keys())
    assert "symbol" in params
    assert "since" in params
    assert "until" in params
    assert "category" in params
    assert "unprocessed_only" in params
    assert "limit" in params


async def test_store_empty_returns_zero() -> None:
    """Verify early-return for empty input without hitting DB."""
    from alphavedha.intel.store import (
        store_bulk_block_deals,
        store_disclosure_events,
        store_disclosures,
        store_pledge_snapshots,
        store_rating_events,
        store_surveillance_flags,
        store_transcripts,
    )

    for fn in [
        store_disclosures,
        store_disclosure_events,
        store_rating_events,
        store_pledge_snapshots,
        store_surveillance_flags,
        store_bulk_block_deals,
        store_transcripts,
    ]:
        result = await fn([])
        assert result == 0, f"{fn.__name__} should return 0 for empty input"
