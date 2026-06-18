"""Tests for intel ORM model definitions."""

from __future__ import annotations


def test_intel_models_importable() -> None:
    from alphavedha.data.models import (
        BulkBlockDeal,
        Disclosure,
        DisclosureEvent,
        PledgeSnapshot,
        RatingEvent,
        SurveillanceFlag,
        Transcript,
    )

    assert Disclosure.__tablename__ == "disclosures"
    assert DisclosureEvent.__tablename__ == "disclosure_events"
    assert Transcript.__tablename__ == "transcripts"
    assert RatingEvent.__tablename__ == "rating_events"
    assert PledgeSnapshot.__tablename__ == "pledge_snapshots"
    assert SurveillanceFlag.__tablename__ == "surveillance_flags"
    assert BulkBlockDeal.__tablename__ == "bulk_block_deals"


def test_disclosure_unique_constraint() -> None:
    from alphavedha.data.models import Disclosure

    constraint_names = [
        c.name for c in Disclosure.__table__.constraints if hasattr(c, "name") and c.name
    ]
    assert "uq_disclosure" in constraint_names


def test_disclosure_event_has_fk() -> None:
    from alphavedha.data.models import DisclosureEvent

    fk_cols = [col.name for col in DisclosureEvent.__table__.columns if col.foreign_keys]
    assert "disclosure_id" in fk_cols


def test_disclosure_filed_at_is_timezone_aware() -> None:
    from alphavedha.data.models import Disclosure

    col = Disclosure.__table__.columns["filed_at"]
    assert col.type.timezone is True


def test_transcript_unique_constraint() -> None:
    from alphavedha.data.models import Transcript

    constraint_names = [
        c.name for c in Transcript.__table__.constraints if hasattr(c, "name") and c.name
    ]
    assert "uq_transcript" in constraint_names


def test_rating_event_unique_constraint() -> None:
    from alphavedha.data.models import RatingEvent

    constraint_names = [
        c.name for c in RatingEvent.__table__.constraints if hasattr(c, "name") and c.name
    ]
    assert "uq_rating_event" in constraint_names


def test_pledge_snapshot_unique_constraint() -> None:
    from alphavedha.data.models import PledgeSnapshot

    constraint_names = [
        c.name for c in PledgeSnapshot.__table__.constraints if hasattr(c, "name") and c.name
    ]
    assert "uq_pledge_snapshot" in constraint_names


def test_surveillance_flag_unique_constraint() -> None:
    from alphavedha.data.models import SurveillanceFlag

    constraint_names = [
        c.name for c in SurveillanceFlag.__table__.constraints if hasattr(c, "name") and c.name
    ]
    assert "uq_surveillance_flag" in constraint_names
