"""The deterministic scheduling kernel (plan.md section 8.3).

This module owns every state-changing operation on appointments and slots. It
contains no LLM calls, no prompts and no free text. Its entire contract is:

    operation(request) -> KernelResult

Guarantees it upholds, each with the mechanism that upholds it:

* **Two-phase booking.** ``hold_slot`` creates a lease; ``commit_appointment``
  consumes it. A slot can be held by at most one live lease at a time, enforced
  by ``uniq_live_lease_per_slot``.
* **One active booking per slot.** Enforced by
  ``uniq_active_appointment_per_slot``, so even a concurrent race cannot
  double-book.
* **Idempotency (invariant 9).** Same key + same canonical request hash returns
  the stored original result; same key + different hash is rejected outright.
* **Unknown outcomes (invariant 10).** If the commit cannot be confirmed, the
  kernel raises ``UnknownOutcome`` and records a reconciliation row. It never
  retries blindly.
* **Atomic reschedule (invariant 11).** The replacement is committed first;
  only then is the original marked ``RESCHEDULED``. If anything fails, the
  original stays exactly as it was.
* **Server-derived subject (invariant 5).** ``patient_id`` is a parameter of the
  kernel, passed down from the session capability. No request payload can
  override it.

Clock injection: everything takes ``now`` from a :class:`Clock` so that lease
expiry, DST behaviour and waitlist offers are deterministic in tests.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal

from clinical_scheduler.db.repository import (
    Repository,
    canonical_json,
    canonical_request_hash,
    sha256_hex,
)
from clinical_scheduler.domain.errors import (
    AppointmentNotFound,
    IdempotencyConflict,
    LeaseConflict,
    LeaseExpired,
    ServiceNotFound,
    SlotUnavailable,
    UnknownOutcome,
)
from clinical_scheduler.domain.states import (
    ALLOWED_STATUS_TRANSITIONS,
    AppointmentStatus,
    OutcomeKind,
    ReasonCode,
)

#: Lease lifetime. Short by design: a hold exists to freeze a slot while the
#: user reads a confirmation summary, not to reserve inventory.
DEFAULT_LEASE_SECONDS = 300

#: Idempotency records are retained for this long. Retries arrive within
#: minutes, so a longer window would only accumulate data with no benefit.
IDEMPOTENCY_RETENTION_SECONDS = 86400


class Clock:
    """Injectable clock. Tests substitute a fixed or advancing clock."""

    def now(self) -> datetime:  # pragma: no cover - overridden
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FixedClock(Clock):
    """A clock that only moves when told to. Used by expiry and DST tests."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FixedClock requires an aware datetime")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, **kwargs: float) -> None:
        self._now = self._now + timedelta(**kwargs)

    def set(self, when: datetime) -> None:
        if when.tzinfo is None:
            raise ValueError("FixedClock.set requires an aware datetime")
        self._now = when


@dataclass(frozen=True)
class KernelResult:
    """Uniform kernel return type.

    ``outcome`` distinguishes success, clean failure and unknown, so callers
    cannot accidentally treat UNKNOWN as FAILURE. ``reason`` is a stable code
    from :class:`ReasonCode`. ``evidence_ids`` are database identifiers proving
    what happened, which the trace and DB oracles cross-check.
    """

    outcome: OutcomeKind
    reason: ReasonCode
    payload: dict[str, Any]
    evidence_ids: dict[str, str]
    replayed: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome is OutcomeKind.KNOWN_SUCCESS


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SchedulingKernel:
    """Transactional scheduling operations over a :class:`Repository`."""

    def __init__(
        self,
        repo: Repository,
        *,
        clock: Clock | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        id_factory: Callable[[str], str] | None = None,
    ) -> None:
        self.repo = repo
        self.clock = clock or SystemClock()
        self.lease_seconds = lease_seconds
        self._id_factory = id_factory or self._default_id

    def _default_id(self, prefix: str) -> str:
        """Allocate the next identifier for ``prefix`` from the database.

        Must be called **inside** an open transaction. The allocation is a
        read-modify-write on ``id_sequences``, and ``BEGIN IMMEDIATE`` is what
        makes it atomic -- the same lock that serialises the booking itself, so
        no additional locking is needed.

        This replaces a process-local counter. That counter reset to 1 in every
        new kernel, so the *first* hold taken by any second process on an
        existing database collided with an identifier already in use and raised
        an unhandled ``UNIQUE`` violation. No concurrency was required to trigger
        it: a restart was enough. Identifiers that are allocated from shared
        state have to live in that shared state.
        """
        if not self.repo.conn.in_transaction:
            # Loud rather than silent. Without a transaction the allocation is
            # not serialised and two callers can be handed the same identifier --
            # which is the defect this method was rewritten to remove, so it must
            # not be reintroduced by a future call site.
            raise RuntimeError(
                "identifier allocation requires an open transaction: call "
                "_id_factory from inside repository.transaction()"
            )
        row = self.repo.conn.execute(
            """INSERT INTO id_sequences (prefix, next_val) VALUES (?, 1)
               ON CONFLICT(prefix) DO UPDATE SET next_val = next_val + 1
               RETURNING next_val""",
            (prefix,),
        ).fetchone()
        return f"{prefix}{int(row[0]):06d}"
    # ------------------------------------------------------------------
    # Discovery (read-only, no state change)
    # ------------------------------------------------------------------

    def search_slots(
        self,
        *,
        service_id: str,
        earliest: datetime,
        latest: datetime,
        location_id: str | None = None,
        include_held: bool = False,
    ) -> KernelResult:
        """Advisory availability search.

        plan.md section 8.3: "Search is advisory." A slot returned here is not
        reserved and may be gone by the time a hold is attempted. Callers must
        not present search results as a confirmed appointment.
        """
        if service_id and not self.repo.query_one(
            "SELECT 1 FROM healthcare_services WHERE service_id = ?", (service_id,)
        ):
            return KernelResult(
                outcome=OutcomeKind.KNOWN_FAILURE,
                reason=ReasonCode.SERVICE_NOT_FOUND,
                payload={"slots": []},
                evidence_ids={},
            )

        sql = """
            SELECT s.slot_id, s.start_utc, s.end_utc, s.display_zone,
                   sch.service_id, sch.practitioner_id, sch.location_id,
                   l.modality, p.display_name AS practitioner_name,
                   h.name AS service_name
              FROM slots s
              JOIN schedules sch ON sch.schedule_id = s.schedule_id
              JOIN locations l   ON l.location_id  = sch.location_id
              JOIN practitioners p ON p.practitioner_id = sch.practitioner_id
              JOIN healthcare_services h ON h.service_id = sch.service_id
             WHERE s.is_bookable = 1
               AND s.start_utc >= ?
               AND s.end_utc   <= ?
               AND sch.service_id = ?
        """
        params: list[Any] = [_iso(earliest), _iso(latest), service_id]
        if location_id:
            sql += " AND sch.location_id = ?"
            params.append(location_id)

        rows = [dict(r) for r in self.repo.query(sql, params)]

        # Exclude slots already consumed by an active appointment.
        taken = {
            r["slot_id"]
            for r in self.repo.query(
                "SELECT slot_id FROM appointments WHERE status IN ('BOOKED','RESCHEDULED')"
            )
        }
        rows = [r for r in rows if r["slot_id"] not in taken]

        holds: set[str] = set()
        if not include_held:
            holds = {
                r["slot_id"]
                for r in self.repo.query(
                    "SELECT slot_id FROM slot_leases WHERE released = 0 AND expires_at > ?",
                    (_iso(self.clock.now()),),
                )
            }
            rows = [r for r in rows if r["slot_id"] not in holds]

        for r in rows:
            r["available"] = True
        return KernelResult(
            outcome=OutcomeKind.KNOWN_SUCCESS,
            reason=ReasonCode.OK,
            payload={"slots": rows, "held_excluded": sorted(holds)},
            evidence_ids={},
        )

    def check_requirements(self, *, service_id: str, request_id: str | None) -> KernelResult:
        """Report outstanding administrative requirements for a request."""
        svc = self.repo.query_one(
            """SELECT requires_referral, requires_prior_authorization,
                      requires_preparation_ack, preparation_ack_key
                 FROM healthcare_services WHERE service_id = ?""",
            (service_id,),
        )
        if not svc:
            return KernelResult(
                outcome=OutcomeKind.KNOWN_FAILURE,
                reason=ReasonCode.SERVICE_NOT_FOUND,
                payload={},
                evidence_ids={},
            )

        unmet: list[dict[str, Any]] = []
        satisfied: list[dict[str, Any]] = []

        if request_id:
            for row in self.repo.query(
                "SELECT requirement_id, kind, status FROM requirements WHERE appointment_request_id = ?",
                (request_id,),
            ):
                entry = {"requirement_id": row["requirement_id"], "kind": row["kind"], "status": row["status"]}
                (satisfied if row["status"] in ("SATISFIED", "WAIVED") else unmet).append(entry)

        # Declared-but-not-yet-materialised requirements are unmet by definition.
        declared_kinds = {
            k
            for k, flag in (
                ("REFERRAL", svc["requires_referral"]),
                ("PRIOR_AUTHORIZATION", svc["requires_prior_authorization"]),
                ("PREPARATION_ACKNOWLEDGEMENT", svc["requires_preparation_ack"]),
            )
            if flag
        }
        materialised = {e["kind"] for e in unmet} | {e["kind"] for e in satisfied}
        for kind in sorted(declared_kinds - materialised):
            unmet.append({"requirement_id": None, "kind": kind, "status": "PENDING"})

        payload = {
            "requirements_met": not unmet,
            "unmet": unmet,
            "satisfied": satisfied,
            "preparation_ack_key": svc["preparation_ack_key"],
        }
        return KernelResult(
            outcome=OutcomeKind.KNOWN_SUCCESS,
            reason=ReasonCode.OK if not unmet else ReasonCode.REQUIREMENTS_UNMET,
            payload=payload,
            evidence_ids={},
        )

    # ------------------------------------------------------------------
    # Phase 1: hold
    # ------------------------------------------------------------------

    def hold_slot(
        self,
        *,
        slot_id: str,
        patient_id: str,
        idempotency_key: str,
    ) -> KernelResult:
        """Create a short lease on a slot. This is NOT a booking.

        Invariant 6: a hold is not an appointment. The returned lease_id lives in
        a different identifier space from appointment_id, so a caller cannot
        confuse the two.
        """
        canonical_hash = canonical_request_hash("hold_slot", {"slot_id": slot_id, "patient_id": patient_id})

        with self.repo.transaction() as conn:
            replay = self._idempotency_lookup(conn, idempotency_key, canonical_hash)
            if replay is not None:
                return replay

            slot = conn.execute(
                "SELECT slot_id, start_utc, end_utc, display_zone, is_bookable FROM slots WHERE slot_id = ?",
                (slot_id,),
            ).fetchone()
            if slot is None or not slot["is_bookable"]:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "hold_slot",
                    ReasonCode.SLOT_UNAVAILABLE, {"slot_id": slot_id},
                )

            now = self.clock.now()

            # Already booked?
            booked = conn.execute(
                "SELECT appointment_id FROM appointments WHERE slot_id = ? AND status IN ('BOOKED','RESCHEDULED')",
                (slot_id,),
            ).fetchone()
            if booked is not None:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "hold_slot",
                    ReasonCode.SLOT_UNAVAILABLE,
                    {"slot_id": slot_id, "conflicting_appointment_id": booked["appointment_id"]},
                )

            # Live lease held by someone else?
            live = conn.execute(
                "SELECT lease_id, patient_id, expires_at FROM slot_leases WHERE slot_id = ? AND released = 0",
                (slot_id,),
            ).fetchone()
            if live is not None:
                if _parse(live["expires_at"]) <= now:
                    # Expired but never marked released: reclaim it. Doing this
                    # inside the write transaction is what makes reclaim safe.
                    conn.execute("UPDATE slot_leases SET released = 1 WHERE lease_id = ?", (live["lease_id"],))
                    conn.execute(
                        """INSERT INTO outbox_events (outbox_id, event_type, aggregate_id, payload_json, created_at)
                           VALUES (?,?,?,?,?)""",
                        (
                            self._id_factory("obx_"), "SLOT_LEASE_EXPIRED_RECLAIMED", live["lease_id"],
                            canonical_json({"slot_id": slot_id, "lease_id": live["lease_id"]}), _iso(now),
                        ),
                    )
                elif live["patient_id"] == patient_id:
                    # Same patient re-holding the same slot: refresh the lease
                    # rather than refusing, which is what a user pressing the
                    # button twice expects.
                    conn.execute("UPDATE slot_leases SET released = 1 WHERE lease_id = ?", (live["lease_id"],))
                else:
                    return self._fail(
                        conn, idempotency_key, canonical_hash, "hold_slot",
                        ReasonCode.LEASE_CONFLICT,
                        {"slot_id": slot_id, "held_by": live["patient_id"]},
                    )

            lease_id = self._id_factory("lease_")
            expires = now + timedelta(seconds=self.lease_seconds)
            conn.execute(
                """INSERT INTO slot_leases
                   (lease_id, slot_id, patient_id, issued_at, expires_at, released, canonical_request_hash)
                   VALUES (?,?,?,?,?,0,?)""",
                (lease_id, slot_id, patient_id, _iso(now), _iso(expires), canonical_hash),
            )
            conn.execute(
                """INSERT INTO outbox_events (outbox_id, event_type, aggregate_id, payload_json, created_at)
                   VALUES (?,?,?,?,?)""",
                (
                    self._id_factory("obx_"), "SLOT_HELD", lease_id,
                    canonical_json({"slot_id": slot_id, "patient_id": patient_id, "expires_at": _iso(expires)}),
                    _iso(now),
                ),
            )

            payload = {
                "lease_id": lease_id,
                "slot_id": slot_id,
                "expires_at": _iso(expires),
                "start_utc": slot["start_utc"],
                "end_utc": slot["end_utc"],
                "display_zone": slot["display_zone"],
                "is_appointment": False,
            }
            result = KernelResult(
                outcome=OutcomeKind.KNOWN_SUCCESS,
                reason=ReasonCode.OK,
                payload=payload,
                evidence_ids={"lease_id": lease_id},
            )
            self._idempotency_store(conn, idempotency_key, canonical_hash, "hold_slot", result)
            return result

    # ------------------------------------------------------------------
    # Phase 2: commit
    # ------------------------------------------------------------------

    def commit_appointment(
        self,
        *,
        lease_id: str,
        patient_id: str,
        service_id: str,
        idempotency_key: str,
        explicit_confirmation: bool,
        fail_after_write: bool = False,
    ) -> KernelResult:
        """Consume a lease and create the appointment.

        Preconditions checked inside the write transaction (invariant 8):
        fresh lease; lease belongs to this patient; lease slot's service matches;
        explicit confirmation supplied. Any failure leaves the slot held and the
        appointment uncreated.

        ``fail_after_write`` injects a failure *after* the appointment row is
        written but before commit, which is how a lost-acknowledgement scenario is
        made both recoverable and testable. It is the only producer of an UNKNOWN
        commit; see :meth:`commit_with_unknown_outcome` and
        ``tests/integration/test_unknown_outcome_reconciliation.py``.

        (A ``simulate_unknown`` parameter used to sit beside it, documented as the
        way the harness exercises invariant 10 and never read by this method. The
        2026-10-08 static audit caught it; it was removed rather than left as a
        claim.)
        """
        canonical_hash = canonical_request_hash(
            "commit_appointment",
            {
                "lease_id": lease_id,
                "patient_id": patient_id,
                "service_id": service_id,
                "explicit_confirmation": explicit_confirmation,
            },
        )

        with self.repo.transaction() as conn:
            replay = self._idempotency_lookup(conn, idempotency_key, canonical_hash)
            if replay is not None:
                return replay

            if not explicit_confirmation:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "commit_appointment",
                    ReasonCode.CONFIRMATION_REQUIRED, {"lease_id": lease_id},
                )

            lease = conn.execute(
                """SELECT lease_id, slot_id, patient_id, issued_at, expires_at, released
                     FROM slot_leases WHERE lease_id = ?""",
                (lease_id,),
            ).fetchone()
            if lease is None:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "commit_appointment",
                    ReasonCode.LEASE_NOT_FOUND, {"lease_id": lease_id},
                )
            if lease["released"]:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "commit_appointment",
                    ReasonCode.LEASE_EXPIRED, {"lease_id": lease_id},
                )

            now = self.clock.now()
            if _parse(lease["expires_at"]) <= now:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "commit_appointment",
                    ReasonCode.LEASE_EXPIRED,
                    {"lease_id": lease_id, "expired_at": lease["expires_at"]},
                )

            # Server-derived subject: the lease must belong to the caller.
            if lease["patient_id"] != patient_id:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "commit_appointment",
                    ReasonCode.NOT_OWNER,
                    {"lease_owner": lease["patient_id"], "caller": patient_id},
                )

            slot = conn.execute(
                """SELECT s.slot_id, s.start_utc, s.end_utc, s.display_zone,
                          sch.service_id, sch.practitioner_id, sch.location_id
                     FROM slots s JOIN schedules sch ON sch.schedule_id = s.schedule_id
                    WHERE s.slot_id = ?""",
                (lease["slot_id"],),
            ).fetchone()
            if slot is None:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "commit_appointment",
                    ReasonCode.SLOT_UNAVAILABLE, {"slot_id": lease["slot_id"]},
                )
            if slot["service_id"] != service_id:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "commit_appointment",
                    ReasonCode.SERVICE_MISMATCH,
                    {"slot_service": slot["service_id"], "requested_service": service_id},
                )

            appointment_id = self._id_factory("appt_")
            key_hash = sha256_hex(idempotency_key)

            try:
                conn.execute(
                    """INSERT INTO appointments
                       (appointment_id, patient_id, slot_id, service_id, practitioner_id,
                        location_id, status, start_utc, end_utc, display_zone, created_at,
                        idempotency_key_hash)
                       VALUES (?,?,?,?,?,?,'BOOKED',?,?,?,?,?)""",
                    (
                        appointment_id, patient_id, slot["slot_id"], slot["service_id"],
                        slot["practitioner_id"], slot["location_id"], slot["start_utc"],
                        slot["end_utc"], slot["display_zone"], _iso(now), key_hash,
                    ),
                )
                conn.execute(
                    "INSERT INTO slot_consumption (slot_id, consumed_by, consumed_at) VALUES (?,?,?)",
                    (slot["slot_id"], appointment_id, _iso(now)),
                )
                conn.execute(
                    """INSERT INTO appointment_status_history
                       (history_id, appointment_id, from_status, to_status, changed_at, reason_code)
                       VALUES (?,?,NULL,'BOOKED',?,?)""",
                    (self._id_factory("ash_"), appointment_id, _iso(now), ReasonCode.OK.value),
                )
                conn.execute(
                    "UPDATE slot_leases SET released = 1 WHERE lease_id = ?", (lease_id,)
                )
                conn.execute(
                    """INSERT INTO outbox_events (outbox_id, event_type, aggregate_id, payload_json, created_at)
                       VALUES (?,?,?,?,?)""",
                    (
                        self._id_factory("obx_"), "APPOINTMENT_BOOKED", appointment_id,
                        canonical_json(
                            {
                                "appointment_id": appointment_id,
                                "patient_id": patient_id,
                                "slot_id": slot["slot_id"],
                                "service_id": slot["service_id"],
                            }
                        ),
                        _iso(now),
                    ),
                )

                if fail_after_write:
                    # Simulates a crash or lost acknowledgement between the write
                    # and the commit. The transaction rolls back, so the database
                    # stays consistent; the *caller* is the one left uncertain.
                    raise UnknownOutcome(
                        "commit acknowledgement lost after write; transaction rolled back",
                        lease_id=lease_id,
                        slot_id=slot["slot_id"],
                    )

                payload = {
                    "appointment_id": appointment_id,
                    "slot_id": slot["slot_id"],
                    "service_id": slot["service_id"],
                    "practitioner_id": slot["practitioner_id"],
                    "location_id": slot["location_id"],
                    "start_utc": slot["start_utc"],
                    "end_utc": slot["end_utc"],
                    "display_zone": slot["display_zone"],
                    "status": AppointmentStatus.BOOKED.value,
                }
                result = KernelResult(
                    outcome=OutcomeKind.KNOWN_SUCCESS,
                    reason=ReasonCode.OK,
                    payload=payload,
                    evidence_ids={"appointment_id": appointment_id, "slot_id": slot["slot_id"]},
                )
                self._idempotency_store(conn, idempotency_key, canonical_hash, "commit_appointment", result)
                return result

            except sqlite3.IntegrityError as exc:
                # The unique index fired: somebody else won the race. This is the
                # structural refusal, not an error to be retried blindly.
                return self._fail(
                    conn, idempotency_key, canonical_hash, "commit_appointment",
                    ReasonCode.SLOT_UNAVAILABLE,
                    {"slot_id": slot["slot_id"], "db_error": str(exc)},
                )

    def commit_with_unknown_outcome(
        self,
        *,
        lease_id: str,
        patient_id: str,
        service_id: str,
        idempotency_key: str,
    ) -> None:
        """Perform a commit that deliberately ends in an UNKNOWN outcome.

        This is the only producer of an UNKNOWN commit, which makes it what keeps
        invariant 10 testable: without a producer, "never retry blindly, reconcile
        first" would be a claim with no scenario behind it.

        It writes the reconciliation row *outside* the rolled-back transaction,
        because the whole point of reconciliation is that the caller does not know
        whether the write landed. Exercised by
        ``tests/integration/test_unknown_outcome_reconciliation.py``.
        """
        try:
            self.commit_appointment(
                lease_id=lease_id,
                patient_id=patient_id,
                service_id=service_id,
                idempotency_key=idempotency_key,
                explicit_confirmation=True,
                fail_after_write=True,
            )
        except UnknownOutcome:
            pass  # expected

        with self.repo.transaction() as conn:
            conn.execute(
                """INSERT INTO commit_reconciliations
                   (reconciliation_id, idempotency_key, slot_id, patient_id,
                    canonical_request_hash, created_at, resolved_at, resolution)
                   SELECT ?, ?, slot_id, patient_id, canonical_request_hash, ?, NULL, NULL
                     FROM slot_leases WHERE lease_id = ?""",
                (
                    self._id_factory("recon_"), idempotency_key,
                    _iso(self.clock.now()), lease_id,
                ),
            )

    def reconcile_commit(self, *, idempotency_key: str) -> KernelResult:
        """Resolve an UNKNOWN commit by inspecting the database (invariant 10).

        Reconciliation asks the database what actually happened rather than
        asking the network again. If the appointment exists, the outcome is
        success; if it does not, the outcome is failure and the hold is still
        valid. Either way the caller gets a definite answer and no duplicate is
        created.
        """
        canonical_hash = canonical_request_hash("reconcile_commit", {"idempotency_key": idempotency_key})

        with self.repo.transaction() as conn:
            replay = self._idempotency_lookup(conn, idempotency_key, canonical_hash)
            if replay is not None:
                return replay

            recon = conn.execute(
                """SELECT reconciliation_id, slot_id, patient_id, created_at
                     FROM commit_reconciliations
                    WHERE idempotency_key = ? AND resolved_at IS NULL
                    ORDER BY created_at DESC LIMIT 1""",
                (idempotency_key,),
            ).fetchone()
            if recon is None:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "reconcile_commit",
                    ReasonCode.APPOINTMENT_NOT_FOUND,
                    {"idempotency_key": idempotency_key, "detail": "no outstanding reconciliation"},
                )

            now = self.clock.now()
            key_hash = sha256_hex(idempotency_key)
            appt = conn.execute(
                """SELECT appointment_id, status, start_utc, end_utc, display_zone
                     FROM appointments
                    WHERE slot_id = ? AND patient_id = ? AND status IN ('BOOKED','RESCHEDULED')
                    ORDER BY created_at DESC LIMIT 1""",
                (recon["slot_id"], recon["patient_id"]),
            ).fetchone()

            if appt is not None:
                conn.execute(
                    "UPDATE commit_reconciliations SET resolved_at = ?, resolution = 'RECONCILED_SUCCESS' WHERE reconciliation_id = ?",
                    (_iso(now), recon["reconciliation_id"]),
                )
                payload = {
                    "resolved": True,
                    "appointment_id": appt["appointment_id"],
                    "slot_id": recon["slot_id"],
                    "status": appt["status"],
                    "start_utc": appt["start_utc"],
                    "end_utc": appt["end_utc"],
                    "display_zone": appt["display_zone"],
                }
                result = KernelResult(
                    outcome=OutcomeKind.KNOWN_SUCCESS,
                    reason=ReasonCode.RECONCILED_SUCCESS,
                    payload=payload,
                    evidence_ids={"appointment_id": appt["appointment_id"]},
                )
            else:
                conn.execute(
                    "UPDATE commit_reconciliations SET resolved_at = ?, resolution = 'RECONCILED_FAILURE' WHERE reconciliation_id = ?",
                    (_iso(now), recon["reconciliation_id"]),
                )
                # The hold survives, so the user can be asked to confirm again
                # without losing their place in the queue.
                live = conn.execute(
                    "SELECT lease_id, expires_at FROM slot_leases WHERE slot_id = ? AND released = 0",
                    (recon["slot_id"],),
                ).fetchone()
                payload = {
                    "resolved": True,
                    "appointment_id": None,
                    "slot_id": recon["slot_id"],
                    "hold_still_valid": bool(live and _parse(live["expires_at"]) > now),
                    "lease_id": live["lease_id"] if live else None,
                }
                result = KernelResult(
                    outcome=OutcomeKind.KNOWN_FAILURE,
                    reason=ReasonCode.RECONCILED_FAILURE,
                    payload=payload,
                    evidence_ids={},
                )

            # Record under both the original key and the reconcile key so that
            # repeated reconciliation is also idempotent.
            self._idempotency_store(conn, idempotency_key, canonical_hash, "reconcile_commit", result)
            self._idempotency_store(
                conn, f"reconcile::{idempotency_key}",
                canonical_request_hash("commit_appointment::orig", {"key": idempotency_key}),
                "reconcile_commit", result,
            )
            return result

    # ------------------------------------------------------------------
    # Reschedule
    # ------------------------------------------------------------------

    def atomic_reschedule(
        self,
        *,
        original_appointment_id: str,
        new_lease_id: str,
        patient_id: str,
        service_id: str,
        idempotency_key: str,
        explicit_confirmation: bool,
    ) -> KernelResult:
        """Replace an appointment atomically (invariant 11).

        Order of operations is the whole point:

        1. validate the original exists and belongs to this patient;
        2. commit the replacement against the new lease;
        3. only then mark the original ``RESCHEDULED`` and link ``superseded_by``.

        If step 2 fails, the original is untouched -- the patient keeps the
        appointment they already had. That is the difference between a safe
        reschedule and one that can silently leave a patient with nothing.
        """
        canonical_hash = canonical_request_hash(
            "atomic_reschedule",
            {
                "original": original_appointment_id,
                "new_lease": new_lease_id,
                "patient_id": patient_id,
                "service_id": service_id,
            },
        )

        with self.repo.transaction() as conn:
            replay = self._idempotency_lookup(conn, idempotency_key, canonical_hash)
            if replay is not None:
                return replay

            original = conn.execute(
                "SELECT * FROM appointments WHERE appointment_id = ?",
                (original_appointment_id,),
            ).fetchone()
            if original is None:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "atomic_reschedule",
                    ReasonCode.APPOINTMENT_NOT_FOUND, {"appointment_id": original_appointment_id},
                )
            if original["patient_id"] != patient_id:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "atomic_reschedule",
                    ReasonCode.NOT_OWNER,
                    {"appointment_owner": original["patient_id"], "caller": patient_id},
                )
            target_status = AppointmentStatus.RESCHEDULED
            if target_status not in ALLOWED_STATUS_TRANSITIONS[AppointmentStatus(original["status"])]:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "atomic_reschedule",
                    ReasonCode.STATUS_TRANSITION_INVALID,
                    {"from": original["status"]},
                )

            if not explicit_confirmation:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "atomic_reschedule",
                    ReasonCode.CONFIRMATION_REQUIRED, {"lease_id": new_lease_id},
                )

            # -- step 2: commit the replacement first -------------------------
            lease = conn.execute(
                "SELECT * FROM slot_leases WHERE lease_id = ?", (new_lease_id,)
            ).fetchone()
            now = self.clock.now()
            if lease is None or lease["released"]:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "atomic_reschedule",
                    ReasonCode.LEASE_NOT_FOUND, {"lease_id": new_lease_id},
                )
            if _parse(lease["expires_at"]) <= now:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "atomic_reschedule",
                    ReasonCode.LEASE_EXPIRED, {"lease_id": new_lease_id},
                )
            if lease["patient_id"] != patient_id:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "atomic_reschedule",
                    ReasonCode.NOT_OWNER, {"lease_owner": lease["patient_id"]},
                )

            slot = conn.execute(
                """SELECT s.*, sch.service_id, sch.practitioner_id, sch.location_id
                     FROM slots s JOIN schedules sch ON sch.schedule_id = s.schedule_id
                    WHERE s.slot_id = ?""",
                (lease["slot_id"],),
            ).fetchone()
            if slot is None or slot["service_id"] != service_id:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "atomic_reschedule",
                    ReasonCode.SERVICE_MISMATCH, {"lease_id": new_lease_id},
                )

            new_appointment_id = self._id_factory("appt_")
            key_hash = sha256_hex(idempotency_key)
            try:
                conn.execute(
                    """INSERT INTO appointments
                       (appointment_id, patient_id, slot_id, service_id, practitioner_id,
                        location_id, status, start_utc, end_utc, display_zone, created_at,
                        idempotency_key_hash)
                       VALUES (?,?,?,?,?,?,'BOOKED',?,?,?,?,?)""",
                    (
                        new_appointment_id, patient_id, slot["slot_id"], slot["service_id"],
                        slot["practitioner_id"], slot["location_id"], slot["start_utc"],
                        slot["end_utc"], slot["display_zone"], _iso(now), key_hash,
                    ),
                )
                conn.execute(
                    "INSERT INTO slot_consumption (slot_id, consumed_by, consumed_at) VALUES (?,?,?)",
                    (slot["slot_id"], new_appointment_id, _iso(now)),
                )
            except sqlite3.IntegrityError as exc:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "atomic_reschedule",
                    ReasonCode.SLOT_UNAVAILABLE,
                    {"slot_id": slot["slot_id"], "db_error": str(exc)},
                )

            # -- step 3: retire the original ----------------------------------
            conn.execute(
                "UPDATE appointments SET status = 'RESCHEDULED', superseded_by = ? WHERE appointment_id = ?",
                (new_appointment_id, original_appointment_id),
            )
            conn.execute(
                """INSERT INTO appointment_status_history
                   (history_id, appointment_id, from_status, to_status, changed_at, reason_code)
                   VALUES (?,?,?,?,?,?)""",
                (
                    self._id_factory("ash_"), original_appointment_id, original["status"],
                    "RESCHEDULED", _iso(now), ReasonCode.OK.value,
                ),
            )
            conn.execute(
                """INSERT INTO appointment_status_history
                   (history_id, appointment_id, from_status, to_status, changed_at, reason_code)
                   VALUES (?,?,NULL,'BOOKED',?,?)""",
                (self._id_factory("ash_"), new_appointment_id, _iso(now), ReasonCode.OK.value),
            )
            conn.execute("UPDATE slot_leases SET released = 1 WHERE lease_id = ?", (new_lease_id,))

            # Free the old slot so it can be resold, but only after the new
            # appointment is safely written.
            conn.execute("DELETE FROM slot_consumption WHERE consumed_by = ?", (original_appointment_id,))

            conn.execute(
                """INSERT INTO outbox_events (outbox_id, event_type, aggregate_id, payload_json, created_at)
                   VALUES (?,?,?,?,?)""",
                (
                    self._id_factory("obx_"), "APPOINTMENT_RESCHEDULED", new_appointment_id,
                    canonical_json(
                        {
                            "original_appointment_id": original_appointment_id,
                            "new_appointment_id": new_appointment_id,
                            "new_slot_id": slot["slot_id"],
                        }
                    ),
                    _iso(now),
                ),
            )

            payload = {
                "original_appointment_id": original_appointment_id,
                "appointment_id": new_appointment_id,
                "slot_id": slot["slot_id"],
                "service_id": slot["service_id"],
                "start_utc": slot["start_utc"],
                "end_utc": slot["end_utc"],
                "display_zone": slot["display_zone"],
                "status": "BOOKED",
            }
            result = KernelResult(
                outcome=OutcomeKind.KNOWN_SUCCESS,
                reason=ReasonCode.OK,
                payload=payload,
                evidence_ids={
                    "appointment_id": new_appointment_id,
                    "original_appointment_id": original_appointment_id,
                    "slot_id": slot["slot_id"],
                },
            )
            self._idempotency_store(conn, idempotency_key, canonical_hash, "atomic_reschedule", result)
            return result

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel_appointment(
        self,
        *,
        appointment_id: str,
        patient_id: str,
        reason_code: str,
        idempotency_key: str,
    ) -> KernelResult:
        """Cancel an appointment.

        plan.md section 5.2 / invariant 12: cancellation is only reversible if
        upstream policy supports it; here it is terminal, so the status trigger
        prevents resurrection. A cancelled appointment frees its slot.
        """
        canonical_hash = canonical_request_hash(
            "cancel_appointment",
            {"appointment_id": appointment_id, "patient_id": patient_id, "reason_code": reason_code},
        )

        with self.repo.transaction() as conn:
            replay = self._idempotency_lookup(conn, idempotency_key, canonical_hash)
            if replay is not None:
                return replay

            appt = conn.execute(
                "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
            ).fetchone()
            if appt is None:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "cancel_appointment",
                    ReasonCode.APPOINTMENT_NOT_FOUND, {"appointment_id": appointment_id},
                )
            if appt["patient_id"] != patient_id:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "cancel_appointment",
                    ReasonCode.NOT_OWNER, {"appointment_owner": appt["patient_id"]},
                )
            target = AppointmentStatus.CANCELLED
            if target not in ALLOWED_STATUS_TRANSITIONS[AppointmentStatus(appt["status"])]:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "cancel_appointment",
                    ReasonCode.STATUS_TRANSITION_INVALID,
                    {"from": appt["status"], "to": target.value},
                )

            now = self.clock.now()
            conn.execute(
                "UPDATE appointments SET status = 'CANCELLED' WHERE appointment_id = ?",
                (appointment_id,),
            )
            conn.execute(
                """INSERT INTO appointment_status_history
                   (history_id, appointment_id, from_status, to_status, changed_at, reason_code)
                   VALUES (?,?,?,?,?,?)""",
                (
                    self._id_factory("ash_"), appointment_id, appt["status"], "CANCELLED",
                    _iso(now), reason_code,
                ),
            )
            conn.execute("DELETE FROM slot_consumption WHERE consumed_by = ?", (appointment_id,))
            conn.execute(
                """INSERT INTO outbox_events (outbox_id, event_type, aggregate_id, payload_json, created_at)
                   VALUES (?,?,?,?,?)""",
                (
                    self._id_factory("obx_"), "APPOINTMENT_CANCELLED", appointment_id,
                    canonical_json({"appointment_id": appointment_id, "reason_code": reason_code}),
                    _iso(now),
                ),
            )

            payload = {"appointment_id": appointment_id, "status": "CANCELLED", "slot_freed": appt["slot_id"]}
            result = KernelResult(
                outcome=OutcomeKind.KNOWN_SUCCESS,
                reason=ReasonCode.OK,
                payload=payload,
                evidence_ids={"appointment_id": appointment_id},
            )
            self._idempotency_store(conn, idempotency_key, canonical_hash, "cancel_appointment", result)
            return result

    # ------------------------------------------------------------------
    # Waitlist
    # ------------------------------------------------------------------

    def join_waitlist(
        self,
        *,
        patient_id: str,
        service_id: str,
        earliest: datetime,
        latest: datetime,
        display_zone: str,
        idempotency_key: str,
    ) -> KernelResult:
        canonical_hash = canonical_request_hash(
            "join_waitlist",
            {
                "patient_id": patient_id,
                "service_id": service_id,
                "earliest": _iso(earliest),
                "latest": _iso(latest),
            },
        )
        with self.repo.transaction() as conn:
            replay = self._idempotency_lookup(conn, idempotency_key, canonical_hash)
            if replay is not None:
                return replay

            if conn.execute(
                "SELECT 1 FROM healthcare_services WHERE service_id = ?", (service_id,)
            ).fetchone() is None:
                return self._fail(
                    conn, idempotency_key, canonical_hash, "join_waitlist",
                    ReasonCode.SERVICE_NOT_FOUND, {"service_id": service_id},
                )

            now = self.clock.now()
            entry_id = self._id_factory("wl_")
            conn.execute(
                """INSERT INTO waitlist_entries
                   (waitlist_id, patient_id, service_id, preferred_start_utc, preferred_end_utc,
                    display_zone, created_at, expires_at, status)
                   VALUES (?,?,?,?,?,?,?,?,'ACTIVE')""",
                (
                    entry_id, patient_id, service_id, _iso(earliest), _iso(latest),
                    display_zone, _iso(now), _iso(now + timedelta(days=30)),
                ),
            )
            payload = {"waitlist_id": entry_id, "service_id": service_id, "status": "ACTIVE"}
            result = KernelResult(
                outcome=OutcomeKind.KNOWN_SUCCESS,
                reason=ReasonCode.WAITLIST_JOINED,
                payload=payload,
                evidence_ids={"waitlist_id": entry_id},
            )
            self._idempotency_store(conn, idempotency_key, canonical_hash, "join_waitlist", result)
            return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _idempotency_lookup(
        self, conn: sqlite3.Connection, key: str, canonical_hash: str
    ) -> KernelResult | None:
        row = conn.execute(
            "SELECT * FROM idempotency_records WHERE idempotency_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        if row["canonical_request_hash"] != canonical_hash:
            raise IdempotencyConflict(
                "idempotency key reused with a different canonical request",
                key=key,
                stored_hash=row["canonical_request_hash"],
                incoming_hash=canonical_hash,
            )
        import json as _json

        payload = _json.loads(row["response_json"])
        return KernelResult(
            outcome=OutcomeKind(row["outcome_kind"]),
            reason=ReasonCode(row["reason_code"]),
            payload=payload.get("payload", {}),
            evidence_ids=payload.get("evidence_ids", {}),
            replayed=True,
        )

    def _idempotency_store(
        self,
        conn: sqlite3.Connection,
        key: str,
        canonical_hash: str,
        operation: str,
        result: KernelResult,
    ) -> None:
        self.repo.put_idempotency(
            key=key,
            canonical_hash=canonical_hash,
            operation=operation,
            outcome_kind=result.outcome.value,
            reason_code=result.reason.value,
            response={
                "payload": _json_safe(result.payload),
                "evidence_ids": result.evidence_ids,
                "replayed": True,
            },
        )

    def _fail(
        self,
        conn: sqlite3.Connection,
        key: str,
        canonical_hash: str,
        operation: str,
        reason: ReasonCode,
        payload: dict[str, Any],
    ) -> KernelResult:
        """Record and return a failure.

        Failures are stored in the idempotency table too, so a retry of the same
        failed request returns the same failure instead of re-executing side
        effects. That is what stops a retry loop from repeatedly probing state.
        """
        result = KernelResult(
            outcome=OutcomeKind.KNOWN_FAILURE,
            reason=reason,
            payload=payload,
            evidence_ids={},
        )
        self._idempotency_store(conn, key, canonical_hash, operation, result)
        return result


def _json_safe(value: Any) -> Any:
    """Payloads must round-trip through JSON. Anything exotic is stringified
    rather than dropped, so evidence is never silently lost."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "Clock",
    "DEFAULT_LEASE_SECONDS",
    "FixedClock",
    "IDEMPOTENCY_RETENTION_SECONDS",
    "KernelResult",
    "SchedulingKernel",
    "SystemClock",
]
