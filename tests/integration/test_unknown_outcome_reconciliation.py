"""Invariant 10, on the real transaction path: UNKNOWN is reconciled, never retried.

A commit whose acknowledgement is lost may or may not have taken effect. The
design says the only way forward is to *ask the database*, and that a blind retry
is forbidden. That claim had no test: the two mechanisms that can produce an
UNKNOWN commit -- ``commit_with_unknown_outcome`` and the ``simulate_unknown``
parameter -- were never invoked by anything, so the whole path was unexercised.

The static audit of 2026-10-08 found this. ``simulate_unknown`` turned out to be
declared with a rationale and never read, and was removed; these tests use
``commit_with_unknown_outcome``, which is now the single producer of an UNKNOWN
commit and therefore what keeps invariant 10 falsifiable.

The negative control matters here too: reconciling a key that was never committed
must refuse rather than invent a success.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_scheduler.db.repository import Repository
from clinical_scheduler.db.seed import FIXTURE_NOW, build_fixture_database
from clinical_scheduler.domain.states import OutcomeKind, ReasonCode
from clinical_scheduler.kernel.scheduling import FixedClock, SchedulingKernel

pytestmark = pytest.mark.integration


@pytest.fixture()
def kernel(tmp_path: Path):
    db_path = str(tmp_path / "unknown_outcome.sqlite")
    bundle = build_fixture_database(db_path)
    repo = Repository(db_path)
    scheduling = SchedulingKernel(repo, clock=FixedClock(FIXTURE_NOW))
    try:
        yield bundle, repo, scheduling
    finally:
        repo.close()


def _service_for_slot(repo: Repository, slot_id: str) -> str:
    row = repo.query_one(
        "SELECT s.service_id AS sid FROM slots sl "
        "JOIN schedules s ON sl.schedule_id = s.schedule_id WHERE sl.slot_id = ?",
        (slot_id,),
    )
    assert row is not None, f"fixture has no schedule for slot {slot_id}"
    return row["sid"]


def _lost_commit(repo: Repository, scheduling: SchedulingKernel, slot_id: str, patient_id: str, key: str) -> None:
    held = scheduling.hold_slot(slot_id=slot_id, patient_id=patient_id, idempotency_key=f"{key}-hold")
    assert held.ok, held
    scheduling.commit_with_unknown_outcome(
        lease_id=held.payload["lease_id"],
        patient_id=patient_id,
        service_id=_service_for_slot(repo, slot_id),
        idempotency_key=key,
    )


def test_an_unknown_commit_leaves_something_to_reconcile(kernel) -> None:
    bundle, repo, scheduling = kernel
    slot_id, patient_id = bundle.slot_ids[0], bundle.patient_ids[0]

    _lost_commit(repo, scheduling, slot_id, patient_id, "c1")

    outstanding = repo.query(
        "SELECT * FROM commit_reconciliations WHERE idempotency_key = ?", ("c1",)
    )
    assert len(outstanding) == 1, "the UNKNOWN commit left nothing to reconcile"
    assert outstanding[0]["resolved_at"] is None

    # The write did not survive: the acknowledgement was lost, so the row was
    # rolled back and there is no appointment.
    assert repo.query("SELECT * FROM appointments") == []


def test_reconciliation_returns_a_definite_answer_and_keeps_the_hold(kernel) -> None:
    bundle, repo, scheduling = kernel
    slot_id, patient_id = bundle.slot_ids[1], bundle.patient_ids[0]

    _lost_commit(repo, scheduling, slot_id, patient_id, "c2")
    resolved = scheduling.reconcile_commit(idempotency_key="c2")

    assert resolved.outcome is OutcomeKind.KNOWN_FAILURE
    assert resolved.reason is ReasonCode.RECONCILED_FAILURE
    assert resolved.payload["resolved"] is True
    assert resolved.payload["appointment_id"] is None
    assert resolved.payload["hold_still_valid"] is True, (
        "the hold must survive a failed commit so the patient does not lose their place"
    )


def test_reconciliation_is_idempotent_and_creates_no_duplicate(kernel) -> None:
    bundle, repo, scheduling = kernel
    slot_id, patient_id = bundle.slot_ids[2], bundle.patient_ids[0]

    _lost_commit(repo, scheduling, slot_id, patient_id, "c3")
    first = scheduling.reconcile_commit(idempotency_key="c3")
    second = scheduling.reconcile_commit(idempotency_key="c3")

    assert first.reason is ReasonCode.RECONCILED_FAILURE
    assert second.reason is ReasonCode.RECONCILED_FAILURE
    assert repo.query("SELECT * FROM appointments") == []


def test_reconciling_nothing_is_refused_rather_than_guessed(kernel) -> None:
    """The negative control: no outstanding reconciliation is not a success."""
    _bundle, _repo, scheduling = kernel

    out = scheduling.reconcile_commit(idempotency_key="never-committed")

    assert out.ok is False
    assert out.reason is ReasonCode.APPOINTMENT_NOT_FOUND
