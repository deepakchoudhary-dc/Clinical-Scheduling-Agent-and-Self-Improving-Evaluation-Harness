"""The executor must not report a refusal as a success.

Regression test for defect D-015, found by an independent adversarial reviewer.

The executor returned ``ToolStatus.SUCCESS`` / ``KNOWN_SUCCESS`` for *every* call
that returned, including the ones the kernel refused. The response composer
converts a ``SUCCESS`` envelope into grounded facts and nothing else, so a refused
hold became the fact "the slot is held" and the agent told the patient:

* "I have held that slot for you. It is reserved for you." — when the hold failed;
* "Your appointment is booked and confirmed." — when the commit was refused;
* "Your appointment has been cancelled." — for another patient's appointment,
  while the database still showed ``BOOKED``.

Every one of those is a false statement about state, and the third is a false
statement about an action taken on someone else's record.

The distinction the tests below pin is the one the executor lost: **a call that
returns is not a call that succeeded.**
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_scheduler.agent.response_grounding import ResponseComposer
from clinical_scheduler.db.repository import Repository
from clinical_scheduler.db.seed import FIXTURE_NOW, build_fixture_database
from clinical_scheduler.domain.states import AuthTier, SessionState
from clinical_scheduler.kernel.scheduling import FixedClock, SchedulingKernel
from clinical_scheduler.tools.broker import CapabilityBroker, SessionContext
from clinical_scheduler.tools.executor import ExecutorContext, ToolExecutor

pytestmark = pytest.mark.integration

SECRET = b"executor-truthfulness-test-secret"


@pytest.fixture()
def harness(tmp_path: Path):
    db_path = str(tmp_path / "truthfulness.sqlite")
    bundle = build_fixture_database(db_path)
    repo = Repository(db_path)
    broker = CapabilityBroker(secret=SECRET)
    kernel = SchedulingKernel(repo, clock=FixedClock(FIXTURE_NOW))
    executor = ToolExecutor(
        repo=repo, broker=broker, kernel=kernel, clock=lambda: FIXTURE_NOW
    )

    def make(
        session_id: str,
        patient_id: str,
        correlation_id: str,
        state: SessionState = SessionState.OPTIONS_PRESENTED,
    ):
        session = SessionContext(
            session_id=session_id,
            subject_id=patient_id,
            tier=AuthTier.TIER2_VERIFIED_IDENTITY,
            state=state,
            verified_identity=True,
        )
        ctx = ExecutorContext(
            session=session,
            turn_index=0,
            correlation_id=correlation_id,
            policy_version="2026.10-demo.1",
            trace_id="trace_truthfulness",
            session_id_hash="h" * 64,
        )
        return session, ctx

    try:
        yield bundle, broker, executor, make
    finally:
        repo.close()


def _call(executor, broker, session, ctx, tool: str, **arguments):
    capability = broker.mint(session, tool_name=tool)
    return executor.invoke(
        capability=capability, tool_name=tool, raw_arguments=arguments, ctx=ctx
    )


def test_a_successful_hold_reports_success_and_produces_a_fact(harness) -> None:
    """The positive case, so the fix cannot be "always report failure"."""
    bundle, broker, executor, make = harness
    session, ctx = make("sess_ok", bundle.patient_ids[0], "corr_ok_1")

    out = _call(executor, broker, session, ctx, "hold_slot", slot_id=bundle.slot_ids[0])

    assert out["envelope"]["status"] == "SUCCESS"
    assert out["envelope"]["outcome"] == "KNOWN_SUCCESS"
    facts = ResponseComposer().facts_from_tool_result("hold_slot", out)
    assert [f.claim_key for f in facts] == ["slot_hold"]


def test_a_refused_hold_reports_failure_and_produces_no_fact(harness) -> None:
    """The defect. A slot taken by someone else must not become "it is yours"."""
    bundle, broker, executor, make = harness
    first, ctx_first = make("sess_first", bundle.patient_ids[0], "corr_first")
    second, ctx_second = make("sess_second", bundle.patient_ids[1], "corr_second")
    slot = bundle.slot_ids[0]

    _call(executor, broker, first, ctx_first, "hold_slot", slot_id=slot)
    refused = _call(executor, broker, second, ctx_second, "hold_slot", slot_id=slot)

    assert refused["envelope"]["status"] == "FAILURE", (
        "a refused hold reported SUCCESS; the composer will turn that into a fact "
        "and the agent will tell the patient the slot is theirs"
    )
    assert refused["envelope"]["outcome"] == "KNOWN_FAILURE"
    assert refused["envelope"]["reason"] == "LEASE_CONFLICT"
    assert ResponseComposer().facts_from_tool_result("hold_slot", refused) == []


def test_a_commit_without_a_lease_reports_failure_and_produces_no_fact(harness) -> None:
    """The worst of the three: "your appointment is booked" with nothing booked."""
    bundle, broker, executor, make = harness
    session, ctx = make(
        "sess_commit",
        bundle.patient_ids[0],
        "corr_commit_1",
        # commit_appointment is only on the ceiling once a summary has been
        # confirmed, so the capability has to be minted from that state.
        state=SessionState.SUMMARY_CONFIRMED,
    )

    out = _call(
        executor,
        broker,
        session,
        ctx,
        "commit_appointment",
        lease_id="lease_does_not_exist",
        service_id="svc_gp_new",
        explicit_confirmation=True,
    )

    assert out["envelope"]["status"] == "FAILURE"
    assert out["envelope"]["outcome"] == "KNOWN_FAILURE"
    assert ResponseComposer().facts_from_tool_result("commit_appointment", out) == []


def test_a_refusal_does_not_count_as_an_executed_mutation(harness) -> None:
    """The trace must not record a refused mutation as one that happened.

    The oracles count successful mutations per turn and check scenario
    expectations against them. A refused mutation counted as successful would
    make the trace claim an action the database never saw.
    """
    bundle, broker, executor, make = harness
    first, ctx_first = make("sess_m1", bundle.patient_ids[0], "corr_m1")
    second, ctx_second = make("sess_m2", bundle.patient_ids[1], "corr_m2")
    slot = bundle.slot_ids[0]

    _call(executor, broker, first, ctx_first, "hold_slot", slot_id=slot)
    _call(executor, broker, second, ctx_second, "hold_slot", slot_id=slot)

    assert ctx_second.mutation_log, "the attempt should still be recorded"
    assert all(
        entry["committed"] is False for entry in ctx_second.mutation_log
    ), f"a refused mutation was logged as committed: {ctx_second.mutation_log}"
