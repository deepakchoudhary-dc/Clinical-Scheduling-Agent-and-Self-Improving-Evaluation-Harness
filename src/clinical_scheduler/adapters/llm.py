"""Model adapters: real providers only.

There is no fake model in this project. The reasoning is deliberate: a scripted
"fake LLM" produces transcripts that look like evidence but do not exercise the
prompt, the tool schema, the model's tool-calling behaviour, or its failure
modes. The evaluation would then be measuring the fake.

What exists instead:

* :class:`OllamaAdapter`      -- local models via Ollama's native API;
* :class:`LlamaCppAdapter`    -- a local ``llama-server`` (OpenAI-compatible);
* :class:`OpenAICompatAdapter`-- any third-party OpenAI-compatible endpoint the
  user configures with their own key and base URL.

All three speak real wire protocols. ``tests/unit/test_adapters.py`` verifies
each adapter against a **loopback HTTP server that reproduces the provider's
documented request/response shape**. That is protocol-conformance testing of the
adapter, not a simulated model: it answers "does this adapter serialise the
request and parse the response correctly", and it is never used to produce a
delivered evaluation result.

Determinism: temperature and seed are always sent when the provider supports
them, and the run manifest records exactly what was sent. Providers that ignore
seeds are recorded as such rather than assumed deterministic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


class AdapterError(Exception):
    """Raised when a provider call fails. Carries whether a retry is sane."""

    def __init__(self, message: str, *, retriable: bool = False, status: int | None = None) -> None:
        super().__init__(message)
        self.retriable = retriable
        self.status = status


@dataclass(frozen=True)
class ToolSpec:
    """A tool offered to the model. Derived from the broker's expose set, so the
    model is never shown a tool the session cannot actually use."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ModelMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ModelResponse:
    """Normalised across providers.

    ``tool_calls`` is normalised to a common shape:
    ``{"id": str, "name": str, "arguments": dict}``. Providers differ in how they
    express this; the adapter's job is to remove that difference so the
    orchestrator has one code path.
    """

    text: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)
    model_id: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int = 0
    finish_reason: str | None = None


class ModelAdapter(Protocol):
    """The single interface the orchestrator depends on."""

    provider_name: str
    model_id: str

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: tuple[ToolSpec, ...] = (),
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
        timeout_seconds: float = 60.0,
    ) -> ModelResponse: ...

    def describe(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _to_openai_tools(tools: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON *string* from OpenAI-compatible servers
    and as an object from others. Malformed JSON is reported as an empty dict
    rather than raising: the executor's strict contract validation is the layer
    that rejects it, and doing so there keeps one rejection path instead of two.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# Ollama (native API)
# ---------------------------------------------------------------------------


class OllamaAdapter:
    """Local models via Ollama.

    Uses the native ``/api/chat`` endpoint. Ollama reports token counts in
    ``prompt_eval_count`` / ``eval_count``; those names are mapped so the
    manifest has consistent keys across providers.
    """

    provider_name = "ollama"

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        keep_alive: str | None = "5m",
        client: httpx.Client | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self._client = client or httpx.Client(timeout=120.0)

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: tuple[ToolSpec, ...] = (),
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
        timeout_seconds: float = 60.0,
    ) -> ModelResponse:
        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {
                    "role": m.role,
                    **({"content": m.content} if m.content else {}),
                    **({"tool_calls": list(m.tool_calls)} if m.tool_calls else {}),
                    **({"name": m.name} if m.name else {}),
                }
                for m in messages
            ],
            "stream": False,
            "options": options,
        }
        if tools:
            payload["tools"] = _to_openai_tools(tools)
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        t0 = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat", json=payload, timeout=timeout_seconds
            )
        except httpx.HTTPError as exc:
            raise AdapterError(f"ollama transport failure: {exc}", retriable=True) from exc

        if response.status_code >= 400:
            raise AdapterError(
                f"ollama returned {response.status_code}: {response.text[:300]}",
                retriable=response.status_code >= 500 or response.status_code == 429,
                status=response.status_code,
            )

        data = response.json()
        message = data.get("message", {}) or {}
        calls = []
        for i, tc in enumerate(message.get("tool_calls", []) or []):
            fn = tc.get("function", {}) or {}
            calls.append(
                {
                    "id": tc.get("id") or f"ollama_call_{i}",
                    "name": fn.get("name", ""),
                    "arguments": _parse_tool_arguments(fn.get("arguments")),
                }
            )

        return ModelResponse(
            text=message.get("content", "") or "",
            tool_calls=tuple(calls),
            raw=data,
            model_id=data.get("model", self.model_id),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=int((time.perf_counter() - t0) * 1000),
            finish_reason=data.get("done_reason"),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "seed_supported": True,
            "note": "local inference; determinism depends on the runtime and hardware",
        }


# ---------------------------------------------------------------------------
# llama.cpp server (OpenAI-compatible)
# ---------------------------------------------------------------------------


class LlamaCppAdapter:
    """A local ``llama-server`` instance.

    llama.cpp's server exposes an OpenAI-compatible ``/v1/chat/completions``.
    It supports ``seed`` and ``temperature``. GBNF grammar constraints are also
    available via ``grammar``; the orchestrator does not use them because the
    strict tool contract already constrains arguments, and adding a second
    constraint surface would make failures harder to attribute.
    """

    provider_name = "llamacpp"

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str = "http://127.0.0.1:8080",
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=120.0)

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: tuple[ToolSpec, ...] = (),
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
        timeout_seconds: float = 60.0,
    ) -> ModelResponse:
        return _openai_compatible_complete(
            client=self._client,
            base_url=self.base_url,
            api_key=self.api_key,
            provider=self.provider_name,
            model_id=self.model_id,
            messages=messages,
            tools=tools,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "seed_supported": True,
            "note": "local inference via llama.cpp server; backend is OpenAI-compatible",
        }


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible third-party endpoint
# ---------------------------------------------------------------------------


class OpenAICompatAdapter:
    """Any OpenAI-compatible endpoint the operator configures.

    The API key is held only in memory and is never logged: ``redaction``
    hard-denies fields named ``api_key``/``token``/``secret``.
    """

    provider_name = "openai_compatible"

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str,
        api_key: str,
        organization: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise AdapterError("an API key is required for an OpenAI-compatible endpoint")
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.organization = organization
        self._client = client or httpx.Client(timeout=120.0)

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: tuple[ToolSpec, ...] = (),
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
        timeout_seconds: float = 60.0,
    ) -> ModelResponse:
        return _openai_compatible_complete(
            client=self._client,
            base_url=self.base_url,
            api_key=self.api_key,
            provider=self.provider_name,
            model_id=self.model_id,
            messages=messages,
            tools=tools,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            organization=self.organization,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "seed_supported": "unknown",
            # Deliberately not recorded: presence of a key is noted, the value
            # never is, so a manifest can be shared without leaking credentials.
            "api_key_present": bool(self.api_key),
            "organization_present": bool(self.organization),
        }


def _openai_compatible_complete(
    *,
    client: httpx.Client,
    base_url: str,
    api_key: str | None,
    provider: str,
    model_id: str,
    messages: list[ModelMessage],
    tools: tuple[ToolSpec, ...],
    temperature: float,
    seed: int | None,
    max_tokens: int,
    timeout_seconds: float,
    organization: str | None = None,
) -> ModelResponse:
    """One implementation for both llama.cpp and third-party endpoints, because
    they expose the same documented shape."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if organization:
        headers["OpenAI-Organization"] = organization

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {
                "role": m.role,
                **({"content": m.content} if m.content else {}),
                **({"tool_calls": [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    }
                    for i, tc in enumerate(m.tool_calls)
                ]} if m.tool_calls else {}),
                **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
                **({"name": m.name} if m.name else {}),
            }
            for m in messages
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if tools:
        payload["tools"] = _to_openai_tools(tools)

    t0 = time.perf_counter()
    try:
        response = client.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as exc:
        raise AdapterError(f"{provider} transport failure: {exc}", retriable=True) from exc

    if response.status_code >= 400:
        raise AdapterError(
            f"{provider} returned {response.status_code}: {response.text[:300]}",
            retriable=response.status_code >= 500 or response.status_code == 429,
            status=response.status_code,
        )

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise AdapterError(f"{provider} returned no choices", retriable=True)

    choice = choices[0]
    message = choice.get("message", {}) or {}
    calls = []
    for i, tc in enumerate(message.get("tool_calls", []) or []):
        fn = tc.get("function", {}) or {}
        calls.append(
            {
                "id": tc.get("id") or f"call_{i}",
                "name": fn.get("name", ""),
                "arguments": _parse_tool_arguments(fn.get("arguments")),
            }
        )

    usage = data.get("usage", {}) or {}
    return ModelResponse(
        text=message.get("content", "") or "",
        tool_calls=tuple(calls),
        raw=data,
        model_id=data.get("model", model_id),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        latency_ms=int((time.perf_counter() - t0) * 1000),
        finish_reason=choice.get("finish_reason"),
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_adapter(config: dict[str, Any]) -> ModelAdapter:
    """Build an adapter from a config mapping.

    Recognised providers: ``ollama``, ``llamacpp``, ``openai_compatible``.

    Unknown providers raise rather than silently falling back. A silent fallback
    would mean the manifest records one provider while a different one produced
    the results, which would make the evidence false.
    """
    provider = str(config.get("provider", "")).lower()
    model_id = config.get("model_id") or config.get("model")
    if not model_id:
        raise AdapterError("model_id is required")

    if provider == "ollama":
        return OllamaAdapter(
            model_id,
            base_url=config.get("base_url", "http://127.0.0.1:11434"),
            keep_alive=config.get("keep_alive", "5m"),
        )
    if provider in {"llamacpp", "llama_cpp", "llama.cpp"}:
        return LlamaCppAdapter(
            model_id,
            base_url=config.get("base_url", "http://127.0.0.1:8080"),
            api_key=config.get("api_key"),
        )
    if provider in {"openai_compatible", "openai", "compatible"}:
        key = config.get("api_key") or ""
        if not key:
            raise AdapterError(
                "api_key is required for an openai_compatible endpoint; "
                "set it via the environment, never in a committed file"
            )
        return OpenAICompatAdapter(
            model_id,
            base_url=config.get("base_url", "https://api.openai.com"),
            api_key=key,
            organization=config.get("organization"),
        )
    raise AdapterError(
        f"unknown provider {provider!r}; expected one of ollama, llamacpp, openai_compatible"
    )


__all__ = [
    "AdapterError",
    "LlamaCppAdapter",
    "ModelAdapter",
    "ModelMessage",
    "ModelResponse",
    "OllamaAdapter",
    "OpenAICompatAdapter",
    "ToolSpec",
    "build_adapter",
]
