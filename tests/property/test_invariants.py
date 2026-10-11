"""Property tests for the global invariants.

These tests exist to answer one question per invariant: **can it fail?**

An invariant that cannot be made to fail is not a check, it is a comment. The
same reasoning applies here as to the scenario expectations: an expectation no
oracle reads is documentation, and a predicate no test can violate is decoration.
So each invariant gets a hand-built counterexample, and a meta-test asserts that
*every* registered invariant has one -- which means a newly added invariant
cannot arrive untested.

Before this suite existed, ``domain/invariants.py`` had no importer at all: the
module declared the definition of "correct" and nothing called it. That is defect
D-013 in ``.agent/state.json``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from clinical_scheduler.domain.invariants import (
    INVARIANTS,
    InvariantContext,
    TurnRecord,
    all_pass,
    evaluate_all,
)
from clinical_scheduler.domain.states import TimeWindow

pytestmark = pytest.mark.property


def _clean_context(**overrides) -> InvariantContext:
    """A context that violates nothing.

    Every field is set to a value that passes, so an override in a test isolates
    exactly one failure and a counterexample cannot pass for the wrong reason.
    """
    defaults = dict(
        turns=(
            TurnRecord(
                index=0,
                safety_verdict="CLEAR",
                tool_calls=("search_slots",),
                mutations_attempted=(),
                mutations_committed=(),
                response_claim_keys=(),
            ),
        ),
        identity_verified=True,
        safety_route_active=False,
        mutation_tools_exposed=(),
        candidate_records_enumerated_to_unverified=(),
        patient_id_from_model_args=None,
        patient_id_from_session="pt_ada",
        slot_hold_ids=("lease_1",),
        appointment_ids=("appt_1",),
        response_claims={},
        unsourced_response_claims=(),
        prohibited_claim_hits=(),
        logged_fields=("event_type", "trace_id", "policy_version"),
        naive_datetimes=(),
        idempotency_records=({"key": "k1", "request_hash": "h1", "accepted": True},),
        commit_attempts=(
            {
                "committed": True,
                "fresh_lease": True,
                "explicit_confirmation": True,
                "slot_matches": True,
                "patient_matches": True,
                "outcome": "KNOWN_SUCCESS",
                "kind": "commit_appointment",
                "succeeded": True,
                "original_preserved": True,
                "retried_without_reconcile": False,
            },
        ),
        cross_session_reads=({"crossed": False},),
    )
    defaults.update(overrides)
    return InvariantContext(**defaults)


def test_the_clean_context_passes_every_invariant() -> None:
    """The baseline must be clean, or the counterexamples prove nothing.

    If this context already failed an invariant, a counterexample elsewhere could
    be failing for an unrelated reason and the suite would be lying.
    """
    results = evaluate_all(_clean_context())
    failures = [r for r in results if not r.passed]
    assert not failures, f"clean context failed: {[(r.invariant_id, r.detail) for r in failures]}"
    assert all_pass(results)


# ---------------------------------------------------------------------------
# One counterexample per invariant
# ---------------------------------------------------------------------------


def _turn(**overrides) -> TurnRecord:
    base = dict(
        index=0,
        safety_verdict="CLEAR",
        tool_calls=(),
        mutations_attempted=(),
        mutations_committed=(),
        response_claim_keys=(),
    )
    base.update(overrides)
    return TurnRecord(**base)


#: invariant id -> the context that must violate it. Built as a mapping rather
#: than as individual test functions so the meta-test below can compare it
#: against the registry, which is the property that actually matters.
COUNTEREXAMPLES: dict[str, InvariantContext] = {
    # A turn with no safety verdict: the screening step was skipped entirely.
    "INV-01": _clean_context(
        turns=(_turn(safety_verdict=""),),
    ),
    # An appointment committed on a turn the sentinel had escalated.
    "INV-02": _clean_context(
        safety_route_active=True,
        turns=(_turn(safety_verdict="TRIGGERED", mutations_committed=("commit_appointment",)),),
    ),
    # Candidate records enumerated to a caller whose identity was not verified.
    "INV-03": _clean_context(
        identity_verified=False,
        candidate_records_enumerated_to_unverified=("pt_ada", "pt_bob"),
    ),
    # Same defect, reported by the invariant that owns it specifically.
    "INV-04": _clean_context(
        candidate_records_enumerated_to_unverified=("pt_ada",),
    ),
    # The model supplied a patient id that disagrees with the session subject.
    "INV-05": _clean_context(
        patient_id_from_model_args="pt_bob",
        patient_id_from_session="pt_ada",
    ),
    # A lease identifier also reported as an appointment: a hold became a booking.
    "INV-06": _clean_context(
        slot_hold_ids=("lease_1",),
        appointment_ids=("lease_1",),
    ),
    # The reply claims the appointment is booked with no commit behind it.
    "INV-07": _clean_context(
        turns=(_turn(response_claim_keys=("appointment.booked",)),),
    ),
    # A commit that succeeded without an explicit confirmation.
    "INV-08": _clean_context(
        commit_attempts=(
            {
                "committed": True,
                "fresh_lease": True,
                "explicit_confirmation": False,
                "slot_matches": True,
                "patient_matches": True,
                "outcome": "KNOWN_SUCCESS",
                "kind": "commit_appointment",
                "succeeded": True,
                "original_preserved": True,
                "retried_without_reconcile": False,
            },
        ),
    ),
    # The same idempotency key accepted with a different canonical request hash.
    "INV-09": _clean_context(
        idempotency_records=(
            {"key": "k1", "request_hash": "h1", "accepted": True},
            {"key": "k1", "request_hash": "h2", "accepted": True},
        ),
    ),
    # An UNKNOWN commit outcome retried without reconciliation first.
    "INV-10": _clean_context(
        commit_attempts=(
            {
                "committed": False,
                "outcome": "UNKNOWN",
                "retried_without_reconcile": True,
                "kind": "commit_appointment",
                "succeeded": False,
                "original_preserved": True,
            },
        ),
    ),
    # A failed atomic reschedule that did not leave the original appointment intact.
    "INV-11": _clean_context(
        commit_attempts=(
            {
                "committed": False,
                "outcome": "KNOWN_FAILURE",
                "kind": "atomic_reschedule",
                "succeeded": False,
                "original_preserved": False,
                "retried_without_reconcile": False,
            },
        ),
    ),
    # The reply asserts emergency services were contacted when they were not.
    "INV-12": _clean_context(
        prohibited_claim_hits=("emergency_services_contacted",),
    ),
    # A claim in the reply that no verified fact authorises.
    "INV-13": _clean_context(
        unsourced_response_claims=("appointment.booked",),
    ),
    # A field the log allowlist forbids was logged.
    "INV-14": _clean_context(
        logged_fields=("event_type", "full_name", "date_of_birth"),
    ),
    # A naive datetime reached a comparison, which is how DST bugs happen.
    "INV-15": _clean_context(
        naive_datetimes=("2026-10-06T09:00:00",),
    ),
    # A session read another session's subject.
    "INV-CS": _clean_context(
        cross_session_reads=({"crossed": True, "session": "sess_other"},),
    ),
}


#: The two sides of a spring-forward transition, as fixed offsets.
#:
#: Deliberately **not** ``ZoneInfo("America/New_York")``. Windows ships no IANA
#: timezone database, so a named-zone test needs the ``tzdata`` package and skips
#: without it -- and a test that skips on the machine it is developed on is not
#: coverage. Two fixed offsets reproduce the property that matters: the wall clock
#: moves two hours while the instants move one.
_EST = timezone(timedelta(hours=-5))
_EDT = timezone(timedelta(hours=-4))


def test_a_window_across_a_dst_transition_uses_absolute_instants() -> None:
    """The DST edge case ``plan.md`` section 19 asks for.

    01:30 EST to 03:30 EDT is two hours of wall clock and **one** hour of elapsed
    time. The domain stores aware instants plus a display zone precisely so this
    cannot corrupt a booking, and this asserts the elapsed time follows the
    instants rather than the wall clock. A system that stored local wall-clock
    times would book a two-hour appointment that lasts one hour -- or the reverse
    in the autumn.
    """
    window = TimeWindow(
        start=datetime(2026, 3, 8, 1, 30, tzinfo=_EST),
        end=datetime(2026, 3, 8, 3, 30, tzinfo=_EDT),
        display_zone="America/New_York",
    )
    assert (window.end - window.start).total_seconds() == 3600
    assert window.start.utcoffset() != window.end.utcoffset(), (
        "the two instants must sit on opposite sides of the transition"
    )
    # The same instants expressed in UTC: 06:30Z -> 07:30Z.
    assert window.start.astimezone(timezone.utc).hour == 6
    assert window.end.astimezone(timezone.utc).hour == 7


def test_a_naive_datetime_is_rejected_on_the_transition_day_too() -> None:
    """INV-15 has to hold at the edge, not only in general."""
    with pytest.raises(ValidationError):
        TimeWindow(
            start=datetime(2026, 3, 8, 1, 30),  # naive, on the transition day
            end=datetime(2026, 3, 8, 3, 30, tzinfo=_EDT),
            display_zone="America/New_York",
        )


#: Invariants that necessarily fail together, with the reason. Anything not
#: listed here is expected to fail alone, so an accidental coupling shows up as a
#: test failure rather than as a silently weaker assertion.
EXPECTED_OVERLAP: dict[str, tuple[str, ...]] = {
    # INV-03 is "a PHI read requires a verified identity"; INV-04 is "candidate
    # records are never enumerated to an unverified caller". The second is a
    # specific instance of the first, so one event trips both.
    "INV-03": ("INV-04",),
}


@pytest.mark.parametrize("invariant_id", sorted(COUNTEREXAMPLES))
def test_each_invariant_can_fail(invariant_id: str) -> None:
    """The counterexample must make its own invariant fail, and no others.

    Asserting "and no others" rather than "at least its own" is deliberate: a
    counterexample that trips several invariants is not evidence that the named
    one is checked, it is evidence that the context is broadly malformed.

    One overlap is real rather than accidental and is declared in
    ``EXPECTED_OVERLAP``. INV-03 ("PHI read requires identity") and INV-04 ("no
    candidate enumeration") describe the same event from two directions -- an
    unverified caller receiving candidate records -- so the same counterexample
    necessarily trips both. Declaring it keeps the check strict: any *other*
    co-failure is still a test failure.
    """
    results = {r.invariant_id: r for r in evaluate_all(COUNTEREXAMPLES[invariant_id])}
    target = results[invariant_id]
    assert not target.passed, (
        f"{invariant_id} passed a context that should violate it; the predicate "
        f"is not checking what its name claims"
    )

    others = sorted(k for k, r in results.items() if not r.passed and k != invariant_id)
    assert others == sorted(EXPECTED_OVERLAP.get(invariant_id, ())), (
        f"the counterexample for {invariant_id} tripped {others}; "
        f"expected only {sorted(EXPECTED_OVERLAP.get(invariant_id, ()))}"
    )


def test_every_registered_invariant_has_a_counterexample() -> None:
    """The coverage forcing function.

    Without this, an invariant can be added to the registry and never exercised --
    which is exactly how ``domain/invariants.py`` came to have no importer at all.
    """
    registered = set(INVARIANTS)
    covered = set(COUNTEREXAMPLES)
    assert not registered - covered, (
        f"registered but never shown to be able to fail: {sorted(registered - covered)}"
    )
    assert not covered - registered, (
        f"counterexample for an invariant that is not registered: {sorted(covered - registered)}"
    )


def test_every_invariant_names_its_structural_enforcement() -> None:
    """A predicate that names no mechanism is a claim with nothing behind it."""
    for invariant_id, (_predicate, enforcement) in INVARIANTS.items():
        assert enforcement.strip(), f"{invariant_id} names no structural enforcement"


def test_all_pass_reports_the_conjunction() -> None:
    """``all_pass`` must not be true when one invariant failed."""
    clean = evaluate_all(_clean_context())
    assert all_pass(clean)
    broken = evaluate_all(COUNTEREXAMPLES["INV-12"])
    assert not all_pass(broken)
