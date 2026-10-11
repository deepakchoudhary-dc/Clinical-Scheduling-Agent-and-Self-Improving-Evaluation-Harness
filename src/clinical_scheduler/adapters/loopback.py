"""Loopback model transport: a real HTTP server speaking the model wire format.

This module provides a **real HTTP server speaking the OpenAI-compatible wire
format**, plus a thin adapter over it. It exists so the runtime can be exercised
end to end -- runner, oracles, scoring, driver, artifact writing -- without a live
model and without anyone having to wonder whether a result came from a fake.

It lives in the package rather than in the test tree because two non-test callers
need it: ``scripts/demo_closed_loop.py`` (the offline closed-loop demonstration)
and ``csched demo``. A delivered script importing from ``tests/`` is a dependency
on the test tree that no packaging would install.

Two rules govern its use, and both are load-bearing for the credibility of the
numbers in ``artifacts/``:

1. **It is never a source of recorded evaluation results.** Every artifact under
   ``artifacts/`` that is presented as evidence is produced against a configured
   real endpoint. This server produces results only in the ``harness-selftest``
   artifact directory and in the labelled offline demonstration, neither of which
   is cited as evidence about the agent's quality.

2. **It is a transport, not a model.** It replays scripted completions. That
   constrains what it can prove: it can show that a fabricated claim is stripped
   from a reply, because the fabrication is in the script; it cannot show that a
   real model would produce that fabrication, nor that the agent would choose the
   right tool. Assertions written against it must therefore be phrased as
   properties of what the runtime does with a given model output -- never as
   properties of the model's judgement. See
   ``knowledge/gotchas/oracle-asserting-model-choice.md`` for the defect that
   rule cost us once already.

The reasoning behind using a real socket rather than a stub object: the
orchestrator's turn sequence depends on what the *adapter* returns, and the
adapter normalises an HTTP payload. A stub object would let the adapter's
normalisation drift out of agreement with the orchestrator's expectations without
any test noticing, and the drift would only appear against a real endpoint. A
loopback server closes that gap at the cost of about forty lines.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from clinical_scheduler.adapters.llm import build_adapter


# ---------------------------------------------------------------------------
# Scripted completions
# ---------------------------------------------------------------------------


def completion(text: str = "", *, model: str = "loopback") -> dict[str, Any]:
    """One OpenAI-compatible completion carrying prose and no tool call."""
    return {
        "id": "chatcmpl-loopback",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def tool_completion(
    name: str, arguments: dict[str, Any] | None = None, *, text: str = ""
) -> dict[str, Any]:
    """One completion that proposes a single tool call."""
    return {
        "id": "chatcmpl-loopback",
        "object": "chat.completion",
        "model": "loopback",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments or {}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def benign_script() -> list[dict[str, Any]]:
    """The standard non-committal script for offline runs.

    One neutral reply, repeated for every turn. It makes no booking claim and
    states no availability, so **no scenario's expectation can be satisfied by the
    script**: any finding produced under it belongs to the deterministic layers.
    What it therefore cannot show is anything about a model's judgement.

    Defined once here because three callers need exactly this and had each
    written their own copy: the CLI's ``eval --offline``, ``scripts/offline_run.py``
    and ``scripts/demo_closed_loop.py``.
    """
    return [
        completion("Thanks for that. I can help with scheduling. Which day would suit you best?")
    ]


class _ScriptedModel:
    """Replays a fixed list of completions, repeating the last one forever.

    Repeating the last entry rather than erroring is deliberate: scenarios have
    a variable number of turns and the orchestrator may retry, and a server that
    ran out of script would turn a scenario-shape mistake into a transport
    failure. A transport failure is reported as a configuration error and would
    mask the real result.
    """

    def __init__(self, script: list[dict[str, Any]], status_code: int = 200) -> None:
        if not script:
            raise ValueError("a loopback script must contain at least one completion")
        self.script = script
        self.status_code = status_code
        self.calls: list[dict[str, Any]] = []
        self._index = 0

    def next_response(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(body)
        index = min(self._index, len(self.script) - 1)
        self._index += 1
        return self.script[index]


def _make_handler(model: _ScriptedModel) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        #: One request per connection. Keep-alive is the default for HTTP/1.1
        #: and it is what makes this server hang on shutdown: a handler thread
        #: parks in ``rfile.readline()`` waiting for a second request that never
        #: comes, and ``server.shutdown()`` waits for the handler to return. The
        #: production model adapter opens a new connection per call, so keeping
        #: the connection open buys nothing and costs a shutdown deadlock.
        protocol_version = "HTTP/1.0"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {}

            if model.status_code != 200:
                self.send_response(model.status_code)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            payload = model.next_response(body)
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:  # silence the access log
            return

    return Handler


class _ThreadedHTTPServer(ThreadingHTTPServer):
    """Threading server so a stalled connection cannot wedge the run.

    ``daemon_threads`` means an in-flight handler cannot keep the process alive
    after the context manager exits, and ``allow_reuse_address`` keeps the
    ephemeral port reusable across the many servers a full evaluation starts.
    """

    daemon_threads = True
    allow_reuse_address = True


class LoopbackModel:
    """Context manager owning an HTTP server on an ephemeral loopback port.

    ``shutdown()`` is issued from a short-lived thread rather than inline. With
    a single-threaded server, calling ``shutdown()`` from the thread that is
    itself blocked inside a request handler deadlocks; and even with a threaded
    server the call blocks until the serve loop polls, so it is never safe to
    assume it returns instantly. Running it on its own thread and joining with a
    timeout turns a potential hang into a loud failure instead of a silent one.
    """

    def __init__(self, script: list[dict[str, Any]], *, status_code: int = 200) -> None:
        self.script = script
        self.status_code = status_code
        self.model = _ScriptedModel(script, status_code)
        self._server: _ThreadedHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "LoopbackModel":
        handler = _make_handler(self.model)
        self._server = _ThreadedHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        server = self._server
        if server is not None:
            # Shut down off-thread, then close the listening socket. close()
            # is what unblocks anything still waiting on accept(); shutdown()
            # alone would leave the socket bound until process exit.
            stopper = threading.Thread(target=server.shutdown, daemon=True)
            stopper.start()
            stopper.join(timeout=10)
            server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("loopback server did not stop within 10s")
            self._thread = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("LoopbackModel is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def build_adapter(self) -> Any:
        """A real adapter pointed at this server. No stubbing of the adapter."""
        return build_adapter(
            {
                "provider": "openai_compatible",
                "base_url": self.base_url,
                "api_key": "loopback-not-a-credential",
                "model_id": "loopback",
            }
        )

    @property
    def offered_tools_by_call(self) -> list[set[str]]:
        """The tool names present in each request payload the server received.

        This is the evidence the trace-conformance layer asserts on: what the
        *runtime* chose to expose, read from the bytes that crossed the socket
        rather than from any internal value.
        """
        out: list[set[str]] = []
        for body in self.model.calls:
            out.append({t["function"]["name"] for t in (body.get("tools") or [])})
        return out


__all__ = ["LoopbackModel", "benign_script", "completion", "tool_completion"]
