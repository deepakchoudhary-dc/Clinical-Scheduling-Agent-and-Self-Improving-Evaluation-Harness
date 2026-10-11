"""Concurrency tests over real connections.

Every test here uses **separate ``Repository`` instances against one database
file**, each on its own thread. That is the only way these tests can fail for the
right reason: a shared connection would serialise the calls in Python and the
race would never happen, so a passing test would prove nothing about the
transaction boundaries.

What is being defended is the two-phase booking. ``hold_slot`` takes a lease and
``commit_appointment`` consumes it, and the whole design rests on the claim that
two callers cannot both end up holding, or both end up booked, for one slot. That
claim is enforced by ``BEGIN IMMEDIATE`` plus partial unique indexes in the
schema -- not by application-level checking, which would have a time-of-check to
time-of-use gap.

The assertions are deliberately about **outcomes that must hold under any
interleaving** rather than about a specific ordering. A test that asserted "the
second thread fails" would be asserting a scheduling coincidence; the durable
claims are "exactly one succeeds" and "the database never contains two live rows
for one slot".
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from clinical_scheduler.db.repository import Repository
from clinical_scheduler.db.seed import FIXTURE_NOW, build_fixture_database
from clinical_scheduler.domain.states import OutcomeKind
from clinical_scheduler.kernel.scheduling import FixedClock, SchedulingKernel

pytestmark = pytest.mark.concurrency


@pytest.fixture()
def fixture(tmp_path: Path) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """A fresh fixture database plus its slot and patient identifiers."""
    db_path = str(tmp_path / "concurrency.sqlite")
    bundle = build_fixture_database(db_path)
    return db_path, bundle.slot_ids, bundle.patient_ids


def _kernel(db_path: str) -> tuple[Repository, SchedulingKernel]:
    repo = Repository(db_path)
    return repo, SchedulingKernel(repo, clock=FixedClock(FIXTURE_NOW))


def _service_for_slot(repo: Repository, slot_id: str) -> str:
    """The service a slot belongs to, read from the fixture rather than assumed.

    The commit path checks that the lease's slot serves the requested service, so
    a hard-coded service id makes the test depend on which slot happens to be
    first in the fixture.
    """
    row = repo.query_one(
        "SELECT s.service_id AS service_id FROM schedules s "
        "JOIN slots sl ON sl.schedule_id = s.schedule_id WHERE sl.slot_id = ?",
        (slot_id,),
    )
    assert row is not None, f"no schedule for slot {slot_id}"
    return str(row["service_id"])


def _run_concurrently(worker, count: int, barrier: threading.Barrier) -> list:
    """Run ``worker(index)`` on ``count`` threads that start together.

    The barrier is what makes the race real: without it the first thread usually
    finishes before the second starts, and the test would pass on any
    implementation, including a broken one.
    """

    def run(index: int):
        barrier.wait(timeout=10)
        return worker(index)

    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(run, index) for index in range(count)]
        return [future.result(timeout=30) for future in futures]


def _live_leases(repo: Repository, slot_id: str) -> list[dict]:
    return [
        dict(row)
        for row in repo.query(
            "SELECT * FROM slot_leases WHERE slot_id = ? AND released = 0", (slot_id,)
        )
    ]


def _live_appointments(repo: Repository, slot_id: str) -> list[dict]:
    return [
        dict(row)
        for row in repo.query(
            "SELECT * FROM appointments WHERE slot_id = ? AND status IN ('BOOKED','RESCHEDULED')",
            (slot_id,),
        )
    ]


# ---------------------------------------------------------------------------
# Holding
# ---------------------------------------------------------------------------


def test_two_callers_cannot_hold_the_same_slot(fixture) -> None:
    """Exactly one lease may exist for a slot, whoever asks first.

    If two leases could coexist, both callers would be told the slot was theirs
    and only one could ever commit -- so the second patient would be turned away
    after being reassured.
    """
    db_path, slot_ids, patient_ids = fixture
    slot_id = slot_ids[0]
    barrier = threading.Barrier(2)

    def worker(index: int):
        repo, kernel = _kernel(db_path)
        try:
            return kernel.hold_slot(
                slot_id=slot_id,
                patient_id=patient_ids[index],
                idempotency_key=f"hold-race-{index}",
            )
        finally:
            repo.close()

    results = _run_concurrently(worker, 2, barrier)

    successes = [r for r in results if r.outcome is OutcomeKind.KNOWN_SUCCESS]
    assert len(successes) == 1, (
        f"expected exactly one successful hold, got {[(r.outcome.value, r.reason.value) for r in results]}"
    )

    repo, _ = _kernel(db_path)
    try:
        leases = _live_leases(repo, slot_id)
        assert len(leases) == 1, f"expected one live lease, found {len(leases)}"
    finally:
        repo.close()


def test_a_lost_hold_race_is_a_clean_failure_not_an_exception(fixture) -> None:
    """The loser must be told, in a typed way, that the slot went to someone else.

    A lost race surfacing as an exception would be caught somewhere upstream and
    very plausibly retried -- turning a normal contention event into a retry storm
    on a slot that is already gone.
    """
    db_path, slot_ids, patient_ids = fixture
    slot_id = slot_ids[0]
    barrier = threading.Barrier(2)

    def worker(index: int):
        repo, kernel = _kernel(db_path)
        try:
            return kernel.hold_slot(
                slot_id=slot_id,
                patient_id=patient_ids[index],
                idempotency_key=f"hold-clean-{index}",
            )
        finally:
            repo.close()

    results = _run_concurrently(worker, 2, barrier)
    for result in results:
        assert result.outcome in (OutcomeKind.KNOWN_SUCCESS, OutcomeKind.KNOWN_FAILURE), (
            f"a contended hold produced {result.outcome.value}; contention is not an "
            "unknown outcome"
        )
    assert any(r.outcome is OutcomeKind.KNOWN_FAILURE for r in results)


def test_holds_on_different_slots_do_not_contend(fixture) -> None:
    """The lock must be per-slot, not global.

    A global write lock would serialise every booking in the clinic and make the
    concurrency guarantee above trivially true. This test is what distinguishes
    "safe" from "safe but useless".
    """
    db_path, slot_ids, patient_ids = fixture
    assert len(slot_ids) >= 2
    barrier = threading.Barrier(2)

    def worker(index: int):
        repo, kernel = _kernel(db_path)
        try:
            return kernel.hold_slot(
                slot_id=slot_ids[index],
                patient_id=patient_ids[index],
                idempotency_key=f"hold-distinct-{index}",
            )
        finally:
            repo.close()

    results = _run_concurrently(worker, 2, barrier)
    assert all(r.outcome is OutcomeKind.KNOWN_SUCCESS for r in results), (
        f"holds on different slots contended: {[(r.outcome.value, r.reason.value) for r in results]}"
    )


# ---------------------------------------------------------------------------
# Committing
# ---------------------------------------------------------------------------


def test_two_callers_cannot_commit_the_same_lease(fixture) -> None:
    """A lease is consumed once. One appointment, not two.

    This is the duplicate-booking guarantee the whole two-phase design exists for.
    """
    db_path, slot_ids, patient_ids = fixture
    slot_id = slot_ids[0]

    repo, kernel = _kernel(db_path)
    try:
        held = kernel.hold_slot(
            slot_id=slot_id, patient_id=patient_ids[0], idempotency_key="commit-race-hold"
        )
        assert held.outcome is OutcomeKind.KNOWN_SUCCESS, held.reason.value
        lease_id = held.evidence_ids["lease_id"]
        # Resolved while this repository is open; the worker threads below use
        # their own connections.
        service_id = _service_for_slot(repo, slot_id)
    finally:
        repo.close()

    barrier = threading.Barrier(2)

    def worker(index: int):
        thread_repo, thread_kernel = _kernel(db_path)
        try:
            return thread_kernel.commit_appointment(
                lease_id=lease_id,
                patient_id=patient_ids[0],
                service_id=service_id,
                idempotency_key=f"commit-race-{index}",
                explicit_confirmation=True,
            )
        finally:
            thread_repo.close()

    results = _run_concurrently(worker, 2, barrier)

    successes = [r for r in results if r.outcome is OutcomeKind.KNOWN_SUCCESS]
    assert len(successes) == 1, (
        f"expected exactly one successful commit, got "
        f"{[(r.outcome.value, r.reason.value) for r in results]}"
    )

    repo, _ = _kernel(db_path)
    try:
        appointments = _live_appointments(repo, slot_id)
        assert len(appointments) == 1, (
            f"expected exactly one live appointment for the slot, found {len(appointments)}"
        )
    finally:
        repo.close()


def test_the_same_idempotency_key_under_concurrency_yields_one_appointment(fixture) -> None:
    """A retry that races the original must not book twice.

    This is the real-world shape of the problem: a client times out, retries with
    the same key, and the retry arrives while the original is still committing.
    """
    db_path, slot_ids, patient_ids = fixture
    slot_id = slot_ids[0]

    repo, kernel = _kernel(db_path)
    try:
        held = kernel.hold_slot(
            slot_id=slot_id, patient_id=patient_ids[0], idempotency_key="idem-race-hold"
        )
        assert held.outcome is OutcomeKind.KNOWN_SUCCESS, held.reason.value
        lease_id = held.evidence_ids["lease_id"]
        service_id = _service_for_slot(repo, slot_id)
    finally:
        repo.close()

    barrier = threading.Barrier(2)

    def worker(index: int):
        thread_repo, thread_kernel = _kernel(db_path)
        try:
            return thread_kernel.commit_appointment(
                lease_id=lease_id,
                patient_id=patient_ids[0],
                service_id=service_id,
                # Identical key and identical arguments on both threads.
                idempotency_key="same-key-under-race",
                explicit_confirmation=True,
            )
        finally:
            thread_repo.close()

    results = _run_concurrently(worker, 2, barrier)

    repo, _ = _kernel(db_path)
    try:
        appointments = _live_appointments(repo, slot_id)
        assert len(appointments) == 1, (
            f"a raced retry with an identical idempotency key produced "
            f"{len(appointments)} appointments"
        )
    finally:
        repo.close()

    # At least one caller saw success. Whether the other replayed or lost the
    # lease race is an interleaving detail; what must never happen is a second
    # appointment, asserted above.
    assert any(r.outcome is OutcomeKind.KNOWN_SUCCESS for r in results)


# ---------------------------------------------------------------------------
# Identifier allocation across processes
# ---------------------------------------------------------------------------


def test_a_second_kernel_on_the_same_database_allocates_fresh_identifiers(fixture) -> None:
    """A restart must not collide with identifiers already in the database.

    Regression test for the identifier-allocation defect. The kernel used a
    process-local counter, so a second kernel -- a restart, a second worker, any
    new process -- restarted at ``lease_000001`` and its first hold raised an
    unhandled ``UNIQUE`` violation. No concurrency was needed to trigger it,
    which is what made it the more dangerous half of the defect: it would fail
    on the first booking after every deploy.

    Identifiers are now allocated from the ``id_sequences`` table inside the
    caller's transaction.
    """
    db_path, slot_ids, patient_ids = fixture

    repo, kernel = _kernel(db_path)
    try:
        first = kernel.hold_slot(
            slot_id=slot_ids[0], patient_id=patient_ids[0], idempotency_key="restart-1"
        )
        assert first.outcome is OutcomeKind.KNOWN_SUCCESS, first.reason.value
        first_lease = first.payload["lease_id"]
    finally:
        repo.close()

    # A brand-new repository and kernel: exactly what a restarted process has.
    second_repo, second_kernel = _kernel(db_path)
    try:
        second = second_kernel.hold_slot(
            slot_id=slot_ids[1], patient_id=patient_ids[1], idempotency_key="restart-2"
        )
        assert second.outcome is OutcomeKind.KNOWN_SUCCESS, (
            f"a second kernel could not take a hold on a different slot: "
            f"{second.reason.value}"
        )
        assert second.payload["lease_id"] != first_lease
    finally:
        second_repo.close()


def test_identifier_allocation_requires_an_open_transaction(fixture) -> None:
    """Allocating outside a transaction must fail loudly, not race silently.

    The allocation is a read-modify-write on shared state. Outside a transaction
    it is not serialised, which is the defect this design removed -- so a future
    call site that forgets the transaction must not be able to reintroduce it
    quietly.
    """
    db_path, _slot_ids, _patient_ids = fixture
    repo, kernel = _kernel(db_path)
    try:
        assert not repo.conn.in_transaction
        with pytest.raises(RuntimeError) as excinfo:
            kernel._id_factory("lease_")
        assert "open transaction" in str(excinfo.value)
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# The guarantee at the database level
# ---------------------------------------------------------------------------


def test_the_database_refuses_a_second_live_appointment_for_one_slot(fixture) -> None:
    """The last line of defence, asserted directly.

    Application code can be wrong. This inserts a duplicate by hand, bypassing the
    kernel entirely, and requires the *schema* to refuse it. If this test can be
    made to pass by writing directly to SQLite, then the uniqueness of a booking
    is an application convention rather than a database guarantee.
    """
    db_path, slot_ids, patient_ids = fixture
    slot_id = slot_ids[0]

    repo, kernel = _kernel(db_path)
    try:
        held = kernel.hold_slot(
            slot_id=slot_id, patient_id=patient_ids[0], idempotency_key="dup-hold"
        )
        assert held.outcome is OutcomeKind.KNOWN_SUCCESS, held.reason.value
        committed = kernel.commit_appointment(
            lease_id=held.evidence_ids["lease_id"],
            patient_id=patient_ids[0],
            service_id=_service_for_slot(repo, slot_id),
            idempotency_key="dup-commit",
            explicit_confirmation=True,
        )
        assert committed.outcome is OutcomeKind.KNOWN_SUCCESS, committed.reason.value

        existing = _live_appointments(repo, slot_id)
        assert len(existing) == 1
        # Copy the real row and change only the primary key and the patient, so
        # the insert cannot fail for a schema reason (a missing NOT NULL column)
        # and the test keeps asserting the uniqueness rule rather than the
        # column list.
        duplicate = dict(existing[0])
        duplicate["appointment_id"] = "appt_duplicate_probe"
        duplicate["patient_id"] = patient_ids[1]
        columns = ", ".join(duplicate)
        placeholders = ", ".join("?" for _ in duplicate)

        with pytest.raises(sqlite3.IntegrityError):
            repo.execute(
                f"INSERT INTO appointments ({columns}) VALUES ({placeholders})",
                tuple(duplicate.values()),
            )
    finally:
        repo.close()
