"""D-045 and D-046: the booking half of the FSM was unreachable in conversation.

D-045: ``_settle_state`` had no path past AUTHORIZATION_VERIFIED. ``OPTIONS_PRESENTED``,
``USER_SELECTED``, ``SLOT_HELD`` and ``SUMMARY_CONFIRMED`` were reachable only by
tests that constructed the session context directly -- a verified session could
search slots forever and never be offered ``hold_slot``. The acceptance evidence
for the fix is a *conversation*: search then hold, driven only through
``run_turn``, reaches SLOT_HELD.

D-046: ``_lease_is_live`` compared lease expiry against the wall clock while the
kernel stamps leases from its own (possibly fixed) clock. Under a FixedClock --
determinism runs, fixtures, replay -- every live lease read as expired and the
USER_SELECTED -> SLOT_HELD guard silently refused immediately after a successful
hold. The orchestrator now receives the same clock the kernel uses, and the test
below fails under the old behaviour because the fixture clock's instant is in
the past relative to the real wall clock.
"""

from __future__ import annotations

import sys
import tempfile
from itertools import cycle
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clinical_scheduler.adapters.llm import ModelResponse
from clinical_scheduler.agent.orchestrator import Orchestrator
from clinical_scheduler.agent.response_grounding import ResponseComposer
from clinical_scheduler.db.repository import Repository
from clinical_scheduler.db.seed import FIXTURE_NOW, build_fixture_database
from clinical_scheduler.domain.states import AuthTier, SessionState
from clinical_scheduler.kernel.scheduling import FixedClock, SchedulingKernel
from clinical_scheduler.policy.engine import load_baseline
from clinical_scheduler.tools.broker import CapabilityBroker
from clinical_scheduler.tools.executor import ToolExecutor


class ScriptedAdapter:
    """Replays calls; transparently labelled as a transport, not a model."""

    provider_name = "test_transport"
    model_id = "scripted"

    def __init__(self, steps: list[tuple[str, dict]]):
        self._steps = cycle(steps) if steps else None
        self._queue = list(steps)
        self._index = -1

    def describe(self) -> dict:
        return {"provider": self.provider_name, "model_id": self.model_id}

    def complete(self, messages, *, tools=(), temperature=0.0, seed=None,
                 max_tokens=1024, timeout_seconds=60.0):
        self._index += 1
        if self._index < len(self._queue):
            name, args = self._queue[self._index]
            return ModelResponse(
                text="", tool_calls=({"id": f"c{self._index}", "name": name, "arguments": args},),
                latency_ms=1, model_id=self.model_id,
            )
        return ModelResponse(text="Anything else I can help with?", latency_ms=1, model_id=self.model_id)


def _orchestrator(clock: FixedClock):
    db = Path(tempfile.mkdtemp()) / "fsm.sqlite"
    build_fixture_database(str(db))
    repo = Repository(str(db))
    kernel = SchedulingKernel(repo, clock=clock)
    broker = CapabilityBroker(secret=b"fsm-test")
    executor = ToolExecutor(repo=repo, broker=broker, kernel=kernel, clock=lambda: FIXTURE_NOW)

    def build(steps: list[tuple[str, dict]]):
        return Orchestrator(
            repo=repo, adapter=ScriptedAdapter(steps), policy=load_baseline(),
            kernel=kernel, broker=broker, executor=executor,
            composer=ResponseComposer(),
            session_id="sess_fsm", session_id_hash="c" * 64,
            trace_id="trace_fsm", tier=AuthTier.TIER2_VERIFIED_IDENTITY,
            subject_id="pt_ada", max_turns=8, clock=clock,
        )
    return build, repo


STEPS = [
    ("search_slots", {"service_id": "svc_gp_est",
                      "earliest_utc": "2026-10-13T00:00:00Z",
                      "latest_utc": "2026-10-16T00:00:00Z"}),
    ("hold_slot", {"slot_id": "sl_gp_est_01"}),
]


def test_d045_conversational_search_then_hold_reaches_slot_held() -> None:
    """The whole point: two honest turns, no state forcing, reach SLOT_HELD."""
    build, repo = _orchestrator(FixedClock(FIXTURE_NOW))
    o = build(STEPS)
    o.session.state = SessionState.AUTHORIZATION_VERIFIED
    o.session.verified_identity = True

    t1 = o.run_turn("What do you have next week?")
    assert t1.state_after is SessionState.OPTIONS_PRESENTED, (
        f"a successful search must present options; got {t1.state_after.value}"
    )
    t2 = o.run_turn("The Tuesday 14:00, hold it please.")
    assert t2.tool_calls[0]["status"] == "SUCCESS"
    assert t2.state_after is SessionState.SLOT_HELD, (
        f"a successful hold must reach SLOT_HELD; got {t2.state_after.value}"
    )


def test_d046_lease_is_judged_on_the_kernel_clock_not_the_wall_clock() -> None:
    """A live hold under a FixedClock in the past must read as live.

    Under D-046's old behaviour this failed: the guard compared the lease's
    expiry (stamped from the fixture clock) against the real wall clock, found
    the fixture instant long past, and silently refused SLOT_HELD even though
    the hold had just succeeded.
    """
    build, repo = _orchestrator(FixedClock(FIXTURE_NOW))
    o = build(STEPS)
    o.session.state = SessionState.AUTHORIZATION_VERIFIED
    o.session.verified_identity = True

    o.run_turn("What do you have next week?")
    t2 = o.run_turn("Hold the Tuesday 14:00.")
    assert t2.action_executed
    assert o._lease_is_live(), (
        "a hold just taken under the kernel's clock must be live to the FSM "
        "regardless of the wall clock"
    )


def test_d045_commit_completes_a_natural_booking() -> None:
    """search -> hold -> explicit confirm -> commit -> a real appointment row."""
    steps: list[tuple[str, dict]] = [
        *STEPS,
        ("commit_appointment", {
            "lease_id": "lease_000001",
            "service_id": "svc_gp_est",
            "idempotency_key": "idem-fsm-test-01",
            "confirmation_token": "tok-fsm-explicit",
        }),
    ]
    build, repo = _orchestrator(FixedClock(FIXTURE_NOW))
    o = build(steps)
    o.session.state = SessionState.AUTHORIZATION_VERIFIED
    o.session.verified_identity = True

    o.run_turn("What do you have next week?")
    o.run_turn("Hold the Tuesday 14:00.")
    o.note_explicit_confirmation("tok-fsm-explicit")
    t3 = o.run_turn("Yes -- confirm and book it.")
    assert t3.tool_calls[-1]["status"] == "SUCCESS", t3.tool_calls
    rows = [dict(r) for r in repo.query("select appointment_id, status from appointments")]
    assert rows and rows[0]["status"] == "BOOKED", rows
