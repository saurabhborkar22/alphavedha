"""Tests for API dependency injection and authentication."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from alphavedha.api.deps import get_service, set_service, verify_api_key


class TestSetGetService:
    def test_get_service_raises_when_not_set(self) -> None:
        set_service(None)  # type: ignore[arg-type]
        with pytest.raises(HTTPException) as exc_info:
            get_service()
        assert exc_info.value.status_code == 503

    def test_set_and_get_service(self) -> None:
        sentinel = object()
        set_service(sentinel)  # type: ignore[arg-type]
        assert get_service() is sentinel
        set_service(None)  # type: ignore[arg-type]


class TestVerifyApiKey:
    def test_no_env_var_passes_any_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ALPHAVEDHA_API_KEY", None)
            result = verify_api_key(None)
            assert result is None

    def test_no_env_var_passes_with_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ALPHAVEDHA_API_KEY", None)
            result = verify_api_key("some-key")
            assert result is None

    def test_env_set_missing_key_raises_401(self) -> None:
        with patch.dict(os.environ, {"ALPHAVEDHA_API_KEY": "secret-key"}):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key(None)
            assert exc_info.value.status_code == 401

    def test_env_set_wrong_key_raises_403(self) -> None:
        with patch.dict(os.environ, {"ALPHAVEDHA_API_KEY": "secret-key"}):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key("wrong-key")
            assert exc_info.value.status_code == 403

    def test_env_set_correct_key_passes(self) -> None:
        with patch.dict(os.environ, {"ALPHAVEDHA_API_KEY": "secret-key"}):
            result = verify_api_key("secret-key")
            assert result == "secret-key"

    def test_empty_env_var_passes(self) -> None:
        with patch.dict(os.environ, {"ALPHAVEDHA_API_KEY": ""}):
            result = verify_api_key(None)
            assert result is None
