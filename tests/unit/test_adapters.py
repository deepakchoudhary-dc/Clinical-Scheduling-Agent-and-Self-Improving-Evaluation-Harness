"""Protocol-conformance tests for the model adapters.

These tests bind a loopback HTTP server that reproduces the *documented* wire
format of each provider and assert that the adapter:

* serialises the request correctly (endpoint, headers, field names, seed);
* parses the documented response shape into a normalised :class:`ModelResponse`;
* maps provider-specific token-count fields;
* turns transport and HTTP failures into :class:`AdapterError` with a correct
  ``retriable`` flag.

This is **not** a simulated model and these responses are **not** used to produce
any evaluation result. The tests answer one question only: does the adapter speak
the provider's protocol. Live results require a real endpoint, which the operator
supplies at run time.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from clinical_scheduler.adapters.llm import (
    AdapterError,
    LlamaCppAdapter,
    ModelMessage,
    OllamaAdapter,
    OpenAICompatAdapter,
    ToolSpec,
    build_adapter,
)


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.headers: list[dict[str, str]] = []


def _make_server(handler_payload: Any, status: int = 200) -> tuple[HTTPServer, str, _Recorder]:
    recorder = _Recorder()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            recorder.paths.append(self.path)
            recorder.headers.append({k: v for k, v in self.headers.items()})
            try:
                recorder.requests.append(json.loads(body or b"{}"))
            except json.JSONDecodeError:
                recorder.requests.append({"_unparsed": body.decode("utf-8", "replace")})

            payload = handler_payload
            if callable(payload):
                payload = payload(recorder.requests[-1])
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args: Any) -> None:  # silence test output
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://127.0.0.1:{port}", recorder


@pytest.fixture
def tool_spec() -> ToolSpec:
    return ToolSpec(
        name="search_slots",
        description="Search advisory availability.",
        parameters={
            "type": "object",
            "properties": {"service_id": {"type": "string"}},
            "required": ["service_id"],
            "additionalProperties": False,
        },
    )


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def test_ollama_serialises_request_and_parses_response(tool_spec: ToolSpec) -> None:
    documented = {
        "model": "llama3.1:8b",
        "created_at": "2026-10-06T09:00:00Z",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "search_slots",
                        "arguments": {"service_id": "svc_gp_new"},
                    }
                }
            ],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 120,
        "eval_count": 18,
    }
    server, base, rec = _make_server(documented)
    try:
        adapter = OllamaAdapter("llama3.1:8b", base_url=base)
        out = adapter.complete(
            [ModelMessage(role="user", content="book me a check up")],
            tools=(tool_spec,),
            temperature=0.0,
            seed=42,
        )
    finally:
        server.shutdown()

    # request shape
    assert rec.paths == ["/api/chat"]
    sent = rec.requests[0]
    assert sent["model"] == "llama3.1:8b"
    assert sent["stream"] is False
    assert sent["options"]["temperature"] == 0.0
    assert sent["options"]["seed"] == 42, "seed must be sent when provided"
    assert sent["tools"][0]["function"]["name"] == "search_slots"
    assert sent["tools"][0]["type"] == "function"

    # response normalisation, including provider-specific token fields
    assert out.tool_calls == (
        {"id": "ollama_call_0", "name": "search_slots", "arguments": {"service_id": "svc_gp_new"}},
    )
    assert out.prompt_tokens == 120
    assert out.completion_tokens == 18
    assert out.finish_reason == "stop"


def test_ollama_reports_retriable_flag_on_5xx() -> None:
    server, base, _ = _make_server({"error": "model loading"}, status=503)
    try:
        adapter = OllamaAdapter("m", base_url=base)
        with pytest.raises(AdapterError) as exc:
            adapter.complete([ModelMessage(role="user", content="hi")])
    finally:
        server.shutdown()
    assert exc.value.retriable is True
    assert exc.value.status == 503


def test_ollama_4xx_is_not_retriable() -> None:
    server, base, _ = _make_server({"error": "unknown model"}, status=404)
    try:
        adapter = OllamaAdapter("m", base_url=base)
        with pytest.raises(AdapterError) as exc:
            adapter.complete([ModelMessage(role="user", content="hi")])
    finally:
        server.shutdown()
    assert exc.value.retriable is False


def test_ollama_unreachable_endpoint_raises_retriable() -> None:
    # Port 1 is reserved and never listening.
    adapter = OllamaAdapter("m", base_url="http://127.0.0.1:1")
    with pytest.raises(AdapterError) as exc:
        adapter.complete([ModelMessage(role="user", content="hi")], timeout_seconds=2.0)
    assert exc.value.retriable is True


# ---------------------------------------------------------------------------
# llama.cpp / OpenAI-compatible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter_cls,kwargs",
    [
        (LlamaCppAdapter, {"base_url_placeholder": None}),
        (OpenAICompatAdapter, {"api_key": "sk-test-not-a-real-key"}),
    ],
)
def test_openai_compatible_shape(adapter_cls, kwargs, tool_spec: ToolSpec) -> None:
    documented = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-demo",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "search_slots",
                                # OpenAI-compatible servers send arguments as a JSON *string*.
                                "arguments": '{"service_id": "svc_gp_new"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230},
    }
    server, base, rec = _make_server(documented)
    try:
        if adapter_cls is LlamaCppAdapter:
            adapter = LlamaCppAdapter("gpt-demo", base_url=base)
        else:
            adapter = OpenAICompatAdapter("gpt-demo", base_url=base, **kwargs)
        out = adapter.complete(
            [ModelMessage(role="user", content="book me a check up")],
            tools=(tool_spec,),
            seed=7,
        )
    finally:
        server.shutdown()

    assert rec.paths == ["/v1/chat/completions"]
    sent = rec.requests[0]
    assert sent["seed"] == 7
    assert sent["tools"][0]["type"] == "function"

    # arguments JSON string must be parsed into a dict
    assert out.tool_calls[0]["arguments"] == {"service_id": "svc_gp_new"}
    assert out.tool_calls[0]["id"] == "call_abc"
    assert out.prompt_tokens == 200
    assert out.completion_tokens == 30
    assert out.finish_reason == "tool_calls"
    assert out.text == ""


def test_openai_compat_sends_bearer_header_and_never_in_url() -> None:
    server, base, rec = _make_server(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
    )
    try:
        adapter = OpenAICompatAdapter("m", base_url=base, api_key="sk-secret-value")
        adapter.complete([ModelMessage(role="user", content="hi")])
    finally:
        server.shutdown()

    auth = {k.lower(): v for k, v in rec.headers[0].items()}.get("authorization")
    assert auth == "Bearer sk-secret-value", "the key must travel in a header"
    assert "sk-secret-value" not in rec.paths[0], "the key must never appear in the URL"


def test_openai_compat_requires_a_key() -> None:
    with pytest.raises(AdapterError, match="API key is required"):
        OpenAICompatAdapter("m", base_url="http://127.0.0.1:9", api_key="")


def test_malformed_tool_arguments_do_not_raise(tool_spec: ToolSpec) -> None:
    """Malformed arguments must reach the executor's strict validator, which is
    the single rejection path, rather than raising inside the adapter."""
    documented = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "search_slots", "arguments": "{not json"}}
                    ]
                },
            }
        ],
        "usage": {},
    }
    server, base, _ = _make_server(documented)
    try:
        adapter = LlamaCppAdapter("m", base_url=base)
        out = adapter.complete([ModelMessage(role="user", content="x")], tools=(tool_spec,))
    finally:
        server.shutdown()
    assert out.tool_calls[0]["arguments"] == {}


def test_tool_message_round_trip_uses_tool_call_id() -> None:
    server, base, rec = _make_server(
        {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}], "usage": {}}
    )
    try:
        adapter = LlamaCppAdapter("m", base_url=base)
        adapter.complete(
            [
                ModelMessage(role="assistant", content="", tool_calls=({"id": "call_1", "name": "hold_slot", "arguments": {"slot_id": "sl_a"}},)),
                ModelMessage(role="tool", content='{"ok":true}', tool_call_id="call_1", name="hold_slot"),
            ]
        )
    finally:
        server.shutdown()
    sent = rec.requests[0]["messages"]
    assert sent[1]["tool_call_id"] == "call_1"
    assert sent[1]["name"] == "hold_slot"
    # assistant tool calls must be re-serialised with string arguments
    assert sent[0]["tool_calls"][0]["function"]["arguments"] == '{"slot_id": "sl_a"}'


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_builds_each_provider() -> None:
    assert isinstance(build_adapter({"provider": "ollama", "model_id": "m"}), OllamaAdapter)
    assert isinstance(build_adapter({"provider": "llamacpp", "model_id": "m"}), LlamaCppAdapter)
    assert isinstance(
        build_adapter({"provider": "openai_compatible", "model_id": "m", "api_key": "k"}),
        OpenAICompatAdapter,
    )


def test_factory_refuses_unknown_provider() -> None:
    """A silent fallback would make the manifest record a provider that did not
    produce the results."""
    with pytest.raises(AdapterError, match="unknown provider"):
        build_adapter({"provider": "definitely-not-real", "model_id": "m"})


def test_factory_requires_model_id() -> None:
    with pytest.raises(AdapterError, match="model_id is required"):
        build_adapter({"provider": "ollama"})


def test_describe_never_includes_the_key() -> None:
    adapter = OpenAICompatAdapter("m", base_url="http://x", api_key="sk-super-secret")
    described = json.dumps(adapter.describe())
    assert "sk-super-secret" not in described
    assert described.count("api_key_present") == 1
