"""Tests for database engine configuration and URL handling."""

from __future__ import annotations

import os
from unittest.mock import patch

from alphavedha.data.database import get_database_url


class TestGetDatabaseUrl:
    def test_env_var_used_when_set(self) -> None:
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@host:5432/db"}):
            url = get_database_url()
            assert "asyncpg" in url
            assert "user:pass@host:5432/db" in url

    def test_already_asyncpg_url(self) -> None:
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql+asyncpg://user:pass@host/db"}):
            url = get_database_url()
            assert url == "postgresql+asyncpg://user:pass@host/db"

    def test_fallback_to_config(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DATABASE_URL", None)
            url = get_database_url()
            assert "asyncpg" in url
            assert "alphavedha" in url
