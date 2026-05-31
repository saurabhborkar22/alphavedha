from __future__ import annotations


def test_scheduler_has_bse_ingestion_method() -> None:
    from alphavedha.scheduler import AlphaVedhaScheduler

    scheduler = object.__new__(AlphaVedhaScheduler)
    assert hasattr(scheduler, "run_bse_ingestion")
    assert callable(scheduler.run_bse_ingestion)


def test_scheduler_has_trends_ingestion_method() -> None:
    from alphavedha.scheduler import AlphaVedhaScheduler

    scheduler = object.__new__(AlphaVedhaScheduler)
    assert hasattr(scheduler, "run_trends_ingestion")
    assert callable(scheduler.run_trends_ingestion)


def test_scheduler_state_has_ingestion_fields() -> None:
    from alphavedha.scheduler import SchedulerState

    state = SchedulerState()
    assert state.last_bse_ingestion is None
    assert state.last_trends_ingestion is None


def test_bse_ingestion_constants_defined() -> None:
    from alphavedha.scheduler import (
        BSE_INGESTION_DAY,
        BSE_INGESTION_TIME,
        TRENDS_INGESTION_TIME,
    )

    assert BSE_INGESTION_DAY == "sunday"
    assert ":" in BSE_INGESTION_TIME
    assert ":" in TRENDS_INGESTION_TIME
