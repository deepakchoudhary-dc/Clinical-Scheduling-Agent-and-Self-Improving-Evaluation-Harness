"""Orchestrator integration tests over a real HTTP loopback model endpoint.

These tests do not use a stub model object. They stand up a real HTTP server
that speaks the documented OpenAI-compatible wire format and point the real
``OpenAICompatAdapter`` at it. That matters because the thing being tested is a
*protocol boundary*: the orchestrator's turn sequence depends on what the
adapter returns, and a hand-written stub would silently paper over any mismatch
between the adapter's normalisation and what the orchestrator expects.

What the loopback server is NOT: a fake model. It is transport-level test
scaffolding. It returns scripted responses so the orchestrator's *reaction* to
model output can be asserted deterministically. No evaluation result in this
repository is produced by it; evaluation runs against a real configured endpoint
(see ``.agent/eval/scripts/verify.py``).

The scenarios below are chosen to cover the failure classes that matter most:

* a clean booking flow advances the FSM through declared edges only;
* a fabricated availability claim never reaches the patient;
* a safety trigger removes every mutation tool before the model is asked;
* an out-of-band tool request is denied rather than executed;
* a fabricated escalation disclaimer is corrected;
* a model outage degrades honestly instead of inventing facts.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from clinical_scheduler.adapters.llm import build_adapter
from clinical_scheduler.agent.orchestrator import Orchestrator
from clinical_scheduler.agent.response_grounding import ResponseComposer
from clinical_scheduler.db.repository import Repository
from clinical_scheduler.db.seed import FIXTURE_NOW, build_fixture_database
from clinical_scheduler.domain.states import AuthTier, SessionState
from clinical_scheduler.kernel.scheduling import FixedClock, SchedulingKernel
from clinical_scheduler.policy.engine import load_baseline, load_hardened
from clinical_scheduler.tools.broker import CapabilityBroker
from clinical_scheduler.tools.executor import ToolExecutor

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Loopback model server (transport only)
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """Serves scripted completions over the OpenAI-compatible wire format."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = script
        self.calls: list[dict[str, Any]] = []
        self._index = 0

    def next_response(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(body)
        index = min(self._index, len(self.script) - 1)
        self._index += 1
        return self.script[index]


def _make_handler(model: _ScriptedModel, status_code: int = 200) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {}
            if status_code != 200:
                self.send_response(status_code)
                self.end_headers()
                return
            payload = model.next_response(body)
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:  # silence
            return

    return Handler


class LoopbackModel:
    """Context manager owning a real HTTP server on an ephemeral port."""

    def __init__(self, script: list[dict[str, Any]], status_code: int = 200) -> None:
        self.model = _ScriptedModel(script)
        self.status_code = status_code
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "LoopbackModel":
        handler = _make_handler(self.model, self.status_code)
        self._server = HTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def openai_completion(self, text: str = "", tool: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build one canonical scripted completion."""
        message: dict[str, Any] = {"role": "assistant", "content": text}
        finish = "stop"
        if tool is not None:
            message["tool_calls"] = [
                {
                    "id": tool.get("id", "call_0"),
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "arguments": json.dumps(tool.get("arguments", {})),
                    },
                }
            ]
            finish = "tool_calls"
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "loopback-test",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env(tmp_path: Path) -> dict[str, Any]:
    """A complete, wired runtime over a fresh fixture database."""
    db_path = tmp_path / "scheduler.sqlite"
    bundle = build_fixture_database(str(db_path))
    repo = Repository(str(db_path))

    clock = FixedClock(FIXTURE_NOW)
    kernel = SchedulingKernel(repo, clock=clock)
    broker = CapabilityBroker(secret=b"test-secret-not-for-production")
    executor = ToolExecutor(repo=repo, broker=broker, kernel=kernel, clock=lambda: FIXTURE_NOW)
    policy = load_baseline()

    yield {
        "db_path": str(db_path),
        "bundle": bundle,
        "repo": repo,
        "clock": clock,
        "kernel": kernel,
        "broker": broker,
        "executor": executor,
        "policy": policy,
    }
    repo.close()


def build_orchestrator(
    env: dict[str, Any],
    *,
    base_url: str,
    policy: Any | None = None,
    tier: AuthTier = AuthTier.TIER2_VERIFIED_IDENTITY,
    subject_id: str | None = None,
    limits: Any | None = None,
) -> Orchestrator:
    adapter = build_adapter(
        {
            "provider": "openai_compatible",
            "base_url": base_url,
            "api_key": "test-key",
            "model_id": "loopback-test",
        }
    )
    return Orchestrator(
        repo=env["repo"],
        adapter=adapter,
        policy=policy or env["policy"],
        kernel=env["kernel"],
        broker=env["broker"],
        executor=env["executor"],
        composer=ResponseComposer(),
        limits=limits,
        session_id="sess_test",
        session_id_hash="hash_test",
        trace_id="trace_test_0001",
        tier=tier,
        subject_id=subject_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clean_turn_advances_only_along_declared_edges(env: dict[str, Any]) -> None:
    """A benign first turn moves START -> SAFETY_SCREENED and nothing further
    when identity has not been established."""
    script = [
        {
            "id": "c", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "How can I help with your appointment?"}, "finish_reason": "stop"}],
        }
    ]
    with LoopbackModel(script) as server:
        orch = build_orchestrator(
            env, base_url=server.base_url,
            tier=AuthTier.TIER0_PUBLIC, subject_id=None,
        )
        result = orch.run_turn("Hello, I would like to book an appointment.")

    assert result.state_before is SessionState.START
    assert result.state_after is SessionState.SAFETY_SCREENED
    assert result.safety_verdict.value == "CLEAR"
    # Sentinel ran on this turn, evidenced by the log rather than by control flow.
    assert orch.sentinel_log.unscreened_turns([0]) == []


def test_safety_trigger_removes_all_mutation_tools_before_model_call(env: dict[str, Any]) -> None:
    """When SAFE-001 fires, the model must never even be offered a mutation tool.

    This is the INV-02 check done from the request the model actually received,
    which is stronger than checking the broker's return value: it proves the
    tool was absent from the payload, not merely that a later gate refused it.
    """
    script = [{
        "id": "c", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Please call 911."}, "finish_reason": "stop"}],
    }]
    with LoopbackModel(script) as server:
        orch = build_orchestrator(env, base_url=server.base_url, subject_id="pt_collide_a")
        result = orch.run_turn("I have crushing chest pain and I cannot breathe.")

        sent = server.model.calls[0]
        offered = {
            t["function"]["name"]
            for t in (sent.get("tools") or [])
        }

    assert result.safety_verdict.value == "TRIGGERED"
    assert "SAFE-001" in result.policy_decision.fired_rule_ids
    assert not ({"hold_slot", "commit_appointment", "atomic_reschedule", "cancel_appointment"} & offered), \
        f"mutation tools leaked to the model: {offered}"
    assert result.action_executed is False


def test_out_of_band_tool_request_is_denied_not_executed(env: dict[str, Any]) -> None:
    """A model proposing a tool that is not exposed this turn is denied.

    The capability broker's allowlist is the authority. This test drives the
    orchestrator with a tool the turn genuinely does not offer and asserts the
    denial is recorded rather than the call being attempted.
    """
    script = [{
        "id": "c", "object": "chat.completion", "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Let me book that for you.",
                "tool_calls": [{
                    "id": "call_0", "type": "function",
                    "function": {"name": "commit_appointment", "arguments": json.dumps({"lease_id": "lease_000001", "confirmation_token": "made-up"})},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }]
    with LoopbackModel(script) as server:
        orch = build_orchestrator(env, base_url=server.base_url, subject_id="pt_collide_a")
        result = orch.run_turn("Book it now.")

    assert result.proposed_action == "commit_appointment"
    assert result.action_executed is False
    assert result.action_denied_reason is not None
    # No appointment may exist as a result of a refused proposal.
    rows = env["repo"].query_one(
        "SELECT COUNT(*) AS n FROM appointments WHERE patient_id = ?", ("pt_collide_a",)
    )
    assert rows["n"] == 0


def test_fabricated_availability_is_stripped(env: dict[str, Any]) -> None:
    """The model asserts specific openings it was never shown. The patient must
    not receive those times."""
    script = [{
        "id": "c", "object": "chat.completion", "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I have openings this Friday at 9am and 11am. Which suits you?",
            },
            "finish_reason": "stop",
        }],
    }]
    with LoopbackModel(script) as server:
        orch = build_orchestrator(env, base_url=server.base_url, subject_id="pt_collide_a")
        result = orch.run_turn("I would like a follow-up appointment.")

    assert "9am" not in result.assistant_message
    assert "11am" not in result.assistant_message
    assert any(u.startswith("unsourced_claim") for u in result.unsourced_claims)


def test_fabricated_escalation_contact_is_corrected(env: dict[str, Any]) -> None:
    """Claiming emergency services were contacted is the single most harmful
    statement available here. It must be replaced, not merely dropped."""
    script = [{
        "id": "c", "object": "chat.completion", "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I have contacted 911 on your behalf and an ambulance is on the way.",
            },
            "finish_reason": "stop",
        }],
    }]
    with LoopbackModel(script) as server:
        orch = build_orchestrator(env, base_url=server.base_url, subject_id="pt_collide_a")
        result = orch.run_turn("I have crushing chest pain.")

    lowered = result.assistant_message.lower()
    assert "ambulance is on the way" not in lowered
    assert "have not contacted anyone" in lowered
    assert any(u.startswith("prohibited_claim") for u in result.unsourced_claims)


def test_model_outage_degrades_honestly(env: dict[str, Any]) -> None:
    """When the model endpoint fails, no facts are invented and the session
    moves to a human handoff rather than pretending to have done something."""
    with LoopbackModel([{}], status_code=503) as server:
        orch = build_orchestrator(env, base_url=server.base_url, subject_id="pt_collide_a")
        result = orch.run_turn("I would like to book something next week please.")

    assert result.action_executed is False
    assert result.action_denied_reason == "UPSTREAM_UNAVAILABLE"
    assert result.tool_calls == ()
    assert result.assistant_message.strip()
    assert orch.state in {SessionState.HUMAN_HANDOFF, SessionState.SAFETY_SCREENED}


def test_hardened_policy_catches_the_paraphrase_the_baseline_misses(env: dict[str, Any]) -> None:
    """The closed-loop improvement is only real if a fixed bundle changes the
    outcome for the exact input that previously slipped through."""
    paraphrase = "Since this morning my speech went funny and my arm feels heavy."

    baseline = load_baseline().evaluate(paraphrase)
    hardened = load_hardened().evaluate(paraphrase)

    assert baseline.verdict.value != "TRIGGERED", "baseline is expected to miss this"
    assert hardened.verdict.value == "TRIGGERED", "hardened bundle must catch it"
    assert "SAFE-002" in hardened.fired_rule_ids


def test_hardened_policy_does_not_over_trigger_on_counterfactuals(env: dict[str, Any]) -> None:
    """A fix that fires on negated or historical statements would convert a
    false negative into a denial-of-service. The counterfactuals must stay clear."""
    hardened = load_hardened()
    counterfactuals = [
        "I am not having trouble speaking and my arm does not feel heavy.",
        "I had a mini stroke last year but nothing like that right now.",
        "My father had trouble speaking, but I am fine.",
    ]
    for text in counterfactuals:
        decision = hardened.evaluate(text)
        assert decision.verdict.value != "TRIGGERED", f"over-triggered on: {text!r}"


# ---------------------------------------------------------------------------
# Session limits (plan.md section 14.2: input size, rate and turn limits)
# ---------------------------------------------------------------------------


def test_the_rate_limit_refuses_once_the_window_is_full() -> None:
    from clinical_scheduler.agent.orchestrator import SessionLimits

    limits = SessionLimits(max_requests_per_minute=2)
    assert limits.allow_request(1000.0) is None
    assert limits.allow_request(1000.5) is None
    reason = limits.allow_request(1001.0)
    assert reason is not None
    assert "rate limit" in reason


def test_the_rate_window_slides() -> None:
    """A limit that never forgets would end a long legitimate session."""
    from clinical_scheduler.agent.orchestrator import SessionLimits

    limits = SessionLimits(max_requests_per_minute=2)
    limits.allow_request(1000.0)
    limits.allow_request(1000.5)
    assert limits.allow_request(1001.0) is not None  # refused
    assert limits.allow_request(1061.0) is None  # the earlier two have aged out


def test_cost_accumulates_from_the_tokens_the_transport_reported() -> None:
    from clinical_scheduler.agent.orchestrator import SessionLimits

    limits = SessionLimits(
        price_per_1k_input_tokens=0.01, price_per_1k_output_tokens=0.03
    )
    limits.record_usage(1000, 1000)
    assert limits.spent_usd == pytest.approx(0.01 + 0.03)


def test_without_a_price_there_is_no_cost_to_report() -> None:
    """Inventing a cost would be worse than reporting none."""
    from clinical_scheduler.agent.orchestrator import SessionLimits

    limits = SessionLimits()
    limits.record_usage(1000, 1000)
    assert limits.spent_usd == 0.0
    assert limits.cost_exceeded() is None


def test_the_cost_ceiling_is_reported_once_passed() -> None:
    from clinical_scheduler.agent.orchestrator import SessionLimits

    limits = SessionLimits(max_cost_usd=0.02, price_per_1k_input_tokens=0.01)
    assert limits.cost_exceeded() is None
    limits.record_usage(3000, None)  # $0.03 against a $0.02 ceiling
    reason = limits.cost_exceeded()
    assert reason is not None
    assert "cost budget" in reason


def test_an_exhausted_budget_makes_no_model_call_and_no_tool_call(
    env: dict[str, Any],
) -> None:
    """A limit must not degrade silently, and must not be able to mutate state.

    The turn is refused with a reason on the trace, so the control is auditable
    rather than merely present -- and because no model call happens, no tool can be
    proposed, so an exhausted budget cannot become an unsafe booking.
    """
    from clinical_scheduler.agent.orchestrator import SessionLimits

    # Zero requests per minute: every request is refused. No server is started,
    # because a refused turn makes no model call -- if the wiring were wrong the
    # test would fail trying to reach a closed port rather than pass quietly.
    limits = SessionLimits(max_requests_per_minute=0)
    orch = build_orchestrator(env, base_url="http://127.0.0.1:9/v1", limits=limits)
    result = orch.run_turn("I would like to book an appointment.")

    assert result.limit_reason is not None
    assert "rate limit" in result.limit_reason
    assert result.prompt_tokens is None  # no model call was made
    assert result.action_executed is False  # so no tool could be proposed


# ---------------------------------------------------------------------------
# Escalation delivery. A live-session defect, 2026-10-09.
# ---------------------------------------------------------------------------


def _escalation_facts(patient_safe_text: str | None = "Please call 911 now.") -> tuple[Any, ...]:
    from clinical_scheduler.domain.states import GroundedFact

    value: dict[str, Any] = {"escalation_id": "esc_1", "human_contacted": False}
    if patient_safe_text is not None:
        value["patient_safe_text"] = patient_safe_text
    return (
        GroundedFact(
            claim_key="escalation_recorded",
            value=value,
            source="verified_tool_result",
            evidence="route_urgent_concern returned SUCCESS",
        ),
    )


def test_a_restart_against_the_same_database_does_not_collide(
    env: dict[str, Any],
) -> None:
    """Found 2026-10-09 by running `csched chat` twice against one session file.

    `_row_id` derives the audit id from `(correlation_id, sequence)`, and the
    sequence was **per-process**. Restarting against an existing database reset it
    to 0 and regenerated ids already written:

        sqlite3.IntegrityError: UNIQUE constraint failed: audit_events.audit_id

    For an append-only, tamper-evident log that is worse than a crash -- a restart
    could collide with history. The sequence is now seeded from the existing audit
    rows, so a restart continues rather than replays.
    """
    from clinical_scheduler.tools.executor import ToolExecutor

    existing = env["repo"].query_one("SELECT COUNT(*) AS n FROM audit_events")
    assert existing is not None
    # A second executor over the same repository, exactly as a restart creates.
    restarted = ToolExecutor(repo=env["repo"], broker=env["broker"], kernel=env["kernel"])
    assert restarted._audit_seq == int(existing["n"])


def test_a_recorded_escalation_reaches_the_patient(env: dict[str, Any]) -> None:
    """The defect that started this: the correct sentence existed and was discarded.

    In a live session, `route_urgent_concern` and `create_human_handoff` both
    returned SUCCESS, both were grounded as `escalation_recorded`, and the caller
    was still asked to confirm their date of birth -- the state clarification won.
    A caller describing a friend's heart attack was told nothing actionable.

    A recorded escalation is the strongest thing this system can know, so it must
    outrank every state clarification.
    """
    from clinical_scheduler.agent.response_grounding import ResponseComposer
    from clinical_scheduler.domain.states import SessionState

    composer = ResponseComposer()
    decision = env["policy"].evaluate("I would like to book an appointment")

    body, _ = composer.compose(
        model_text="",
        facts=_escalation_facts(),
        decision=decision,
        state=SessionState.SAFETY_SCREENED,
    )
    assert body == "Please call 911 now."


def test_an_escalation_without_approved_text_still_says_something_usable(
    env: dict[str, Any],
) -> None:
    """The template may be absent; the person must still be told to seek help."""
    from clinical_scheduler.agent.response_grounding import ResponseComposer
    from clinical_scheduler.domain.states import SessionState

    composer = ResponseComposer()
    decision = env["policy"].evaluate("I would like to book an appointment")

    body, _ = composer.compose(
        model_text="",
        facts=_escalation_facts(patient_safe_text=None),
        decision=decision,
        state=SessionState.SAFETY_SCREENED,
    )
    assert "flagged this to the clinic team" in body
    assert "emergency" in body


def test_without_an_escalation_the_state_clarification_still_wins(
    env: dict[str, Any],
) -> None:
    """The precedence rule must not swallow the ordinary path."""
    from clinical_scheduler.agent.response_grounding import ResponseComposer
    from clinical_scheduler.domain.states import SessionState

    composer = ResponseComposer()
    decision = env["policy"].evaluate("I would like to book an appointment")

    body, _ = composer.compose(
        model_text="",
        facts=(),
        decision=decision,
        state=SessionState.SAFETY_SCREENED,
    )
    assert "date of birth" in body
    assert "YYYY-MM-DD" in body
