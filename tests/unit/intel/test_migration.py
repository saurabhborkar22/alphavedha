"""Tests for intel migration — verifies revision chain and table creation ops."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_migration() -> object:
    spec = importlib.util.spec_from_file_location(
        "intel_migration",
        Path("alembic/versions/e1f2a3b4c5d6_add_intel_tables.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_revision_chain() -> None:
    mig = _load_migration()
    assert mig.revision == "e1f2a3b4c5d6"
    assert mig.down_revision == "d0e1f2a3b4c5"


def test_migration_has_upgrade_and_downgrade() -> None:
    mig = _load_migration()
    assert callable(mig.upgrade)
    assert callable(mig.downgrade)
