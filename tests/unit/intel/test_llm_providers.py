"""Tests for LLM provider factory, round-robin, and Cerebras provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alphavedha.intel.extraction.llm import (
    CerebrasProvider,
    GeminiProvider,
    GroqProvider,
    RoundRobinProvider,
    get_available_providers,
    get_provider,
)


class TestGetProvider:
    def test_gemini(self) -> None:
        p = get_provider("gemini")
        assert isinstance(p, GeminiProvider)

    def test_groq(self) -> None:
        p = get_provider("groq")
        assert isinstance(p, GroqProvider)

    def test_cerebras(self) -> None:
        p = get_provider("cerebras")
        assert isinstance(p, CerebrasProvider)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("openai")


class TestCerebrasProvider:
    def test_name(self) -> None:
        p = CerebrasProvider(api_key="test")
        assert p.name == "cerebras/gpt-oss-120b"

    def test_custom_model(self) -> None:
        p = CerebrasProvider(api_key="test", model="zai-glm-4.7")
        assert p.name == "cerebras/zai-glm-4.7"

    def test_extract_json_success(self) -> None:
        p = CerebrasProvider(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"event_type": "order_win", "direction": 1}'}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            result = p.extract_json("system", "user", {})
            assert result is not None
            assert result["event_type"] == "order_win"

    def test_extract_json_failure(self) -> None:
        p = CerebrasProvider(api_key="test-key")

        with patch("httpx.post", side_effect=Exception("connection error")):
            result = p.extract_json("system", "user", {})
            assert result is None

    def test_reads_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CEREBRAS_API_KEY", "env-key-123")
        p = CerebrasProvider()
        assert p._api_key == "env-key-123"


class TestGetAvailableProviders:
    def test_returns_providers_with_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "gk")
        monkeypatch.setenv("GROQ_API_KEY", "qk")
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        providers = get_available_providers()
        assert len(providers) == 2
        names = [p.name for p in providers]
        assert any("gemini" in n for n in names)
        assert any("groq" in n for n in names)

    def test_empty_when_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        providers = get_available_providers()
        assert providers == []

    def test_preserves_priority_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "gk")
        monkeypatch.setenv("GROQ_API_KEY", "qk")
        monkeypatch.setenv("CEREBRAS_API_KEY", "ck")
        providers = get_available_providers()
        assert len(providers) == 3
        assert "gemini" in providers[0].name
        assert "groq" in providers[1].name
        assert "cerebras" in providers[2].name


class TestRoundRobinProvider:
    def _mock_provider(self, name: str, result: dict[str, Any] | None = None) -> MagicMock:
        p = MagicMock()
        p.name = name
        p.extract_json.return_value = result
        return p

    def test_cycles_through_providers(self) -> None:
        p1 = self._mock_provider("p1", {"ok": 1})
        p2 = self._mock_provider("p2", {"ok": 2})
        rr = RoundRobinProvider([p1, p2])

        rr.extract_json("sys", "user1", {})
        p1.extract_json.assert_called_once()
        p2.extract_json.assert_not_called()

        rr.extract_json("sys", "user2", {})
        p2.extract_json.assert_called_once()

    def test_falls_back_on_failure(self) -> None:
        p1 = self._mock_provider("p1", None)
        p2 = self._mock_provider("p2", {"ok": True})
        rr = RoundRobinProvider([p1, p2])

        result = rr.extract_json("sys", "user", {})
        assert result == {"ok": True}

    def test_returns_none_when_all_fail(self) -> None:
        p1 = self._mock_provider("p1", None)
        p2 = self._mock_provider("p2", None)
        rr = RoundRobinProvider([p1, p2])

        result = rr.extract_json("sys", "user", {})
        assert result is None

    def test_name_reflects_current_provider(self) -> None:
        p1 = self._mock_provider("gemini/flash", {"ok": 1})
        p2 = self._mock_provider("groq/llama", {"ok": 1})
        rr = RoundRobinProvider([p1, p2])

        assert rr.name == "gemini/flash"
        rr.extract_json("sys", "user", {})
        assert rr.name == "groq/llama"

    def test_single_provider(self) -> None:
        p1 = self._mock_provider("p1", {"ok": 1})
        rr = RoundRobinProvider([p1])

        result = rr.extract_json("sys", "user", {})
        assert result == {"ok": 1}

    def test_empty_providers_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one provider"):
            RoundRobinProvider([])
