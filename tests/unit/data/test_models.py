"""Tests for ORM model definitions."""

from __future__ import annotations


def test_d7_orm_models_importable() -> None:
    from alphavedha.data.models import (
        CorporateAnnouncement,
        DataLineage,
        DataQualityReport,
        IntradayOHLCV,
    )

    assert CorporateAnnouncement.__tablename__ == "corporate_announcements"
    assert DataLineage.__tablename__ == "data_lineage"
    assert DataQualityReport.__tablename__ == "data_quality_reports"
    assert IntradayOHLCV.__tablename__ == "intraday_ohlcv"
