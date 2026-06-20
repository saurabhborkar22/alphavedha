"""Provider-agnostic LLM interface for disclosure extraction.

Supports Gemini (primary) and Groq (fallback). Designed for structured
JSON output using each provider's native JSON mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_system_prompt(version: str = "v1") -> str:
    """Load a versioned system prompt from the prompts directory."""
    path = PROMPTS_DIR / f"{version}_system.md"
    return path.read_text()


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @property
    def name(self) -> str: ...

    def extract_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Call the LLM and return parsed JSON, or None on failure."""
        ...


class GeminiProvider:
    """Google Gemini Flash provider using the generativeai SDK."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash") -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model = model
        self._client: Any = None

    @property
    def name(self) -> str:
        return f"gemini/{self._model}"

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def extract_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        import json

        try:
            from google.genai import types

            client = self._get_client()

            response = client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=json_schema,
                    temperature=0.1,
                ),
            )

            if response.text:
                result: dict[str, Any] = json.loads(response.text)
                return result
            return None

        except Exception as e:
            logger.error("gemini_extraction_failed", error=str(e), model=self._model)
            return None


class GroqProvider:
    """Groq provider (Llama models) as fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._model = model
        self._client: Any = None

    @property
    def name(self) -> str:
        return f"groq/{self._model}"

    def _get_client(self) -> Any:
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self._api_key)
        return self._client

    def extract_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        import json

        try:
            client = self._get_client()

            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt + "\n\nRespond with valid JSON only.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1024,
            )

            text = response.choices[0].message.content
            if text:
                result: dict[str, Any] = json.loads(text)
                return result
            return None

        except Exception as e:
            logger.error("groq_extraction_failed", error=str(e), model=self._model)
            return None


class CerebrasProvider:
    """Cerebras provider (Llama 3.3 70B) via OpenAI-compatible REST API."""

    CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b",
    ) -> None:
        self._api_key = api_key or os.environ.get("CEREBRAS_API_KEY", "")
        self._model = model

    @property
    def name(self) -> str:
        return f"cerebras/{self._model}"

    def extract_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        import json

        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt + "\n\nRespond with valid JSON only.",
                },
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 1024,
        }

        try:
            resp = httpx.post(
                self.CEREBRAS_API_URL,
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            if text:
                result: dict[str, Any] = json.loads(text)
                return result
            return None

        except Exception as e:
            logger.error("cerebras_extraction_failed", error=str(e), model=self._model)
            return None


_PROVIDER_FACTORIES: dict[str, type] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "cerebras": CerebrasProvider,
}

_PROVIDER_ORDER: tuple[str, ...] = ("gemini", "groq", "cerebras")


def get_provider(
    provider_name: str = "gemini",
    api_key: str | None = None,
) -> LLMProvider:
    """Factory function to get an LLM provider by name."""
    cls = _PROVIDER_FACTORIES.get(provider_name)
    if cls is not None:
        provider: LLMProvider = cls(api_key=api_key)
        return provider

    msg = f"Unknown provider: {provider_name}"
    raise ValueError(msg)


def get_available_providers() -> list[LLMProvider]:
    """Return providers that have API keys configured, in priority order."""
    env_keys = {
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
    }
    providers: list[LLMProvider] = []
    for name in _PROVIDER_ORDER:
        env_var = env_keys[name]
        if os.environ.get(env_var):
            providers.append(get_provider(name))
    return providers


class RoundRobinProvider:
    """Cycles through multiple LLM providers to spread rate-limit load."""

    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            msg = "At least one provider is required"
            raise ValueError(msg)
        self._providers = providers
        self._index = 0

    @property
    def name(self) -> str:
        return self._providers[self._index].name

    def extract_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        provider = self._providers[self._index]
        self._index = (self._index + 1) % len(self._providers)

        result = provider.extract_json(system_prompt, user_prompt, json_schema)
        if result is not None:
            return result

        # On failure, try remaining providers once
        for i in range(len(self._providers) - 1):
            fallback_idx = (self._index + i) % len(self._providers)
            fallback = self._providers[fallback_idx]
            logger.info(
                "llm_fallback",
                failed=provider.name,
                trying=fallback.name,
            )
            result = fallback.extract_json(system_prompt, user_prompt, json_schema)
            if result is not None:
                return result

        return None
