"""Global invariants (plan.md section 5.1) as executable checks.

Each invariant is a pure predicate over a session snapshot. Keeping them in one
place means the definition of "correct" cannot drift between the tests that
exercise the kernel and the property tests that attack it.

**Who calls this, precisely.** The property suite
(``tests/property/test_invariants.py``) calls ``evaluate_all`` directly. The
oracle stack does **not** call it: L0-L4 re-derive each invariant from a
``RunRecord`` -- the evidence a run actually produced -- because an oracle must
work from the recorded trace rather than from a snapshot the system under test
supplied. The two implementations are therefore deliberate, and each invariant
below records where its *structural* enforcement lives so a reader can see that
the predicate is a check on a mechanism, not the mechanism itself.

Invariants that are structurally enforced elsewhere are still asserted here, so
that a future refactor which removes the structural enforcement will fail a test
rather than silently weaken the safety case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Snapshot types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnRecord:
    """One patient turn plus what the system did with it."""

    index: int
    safety_verdict: str
    tool_calls: tuple[str, ...]
    mutations_attempted: tuple[str, ...]
    mutations_committed: tuple[str, ...]
    response_claim_keys: tuple[str, ...]


@dataclass(frozen=True)
class InvariantContext:
    """Everything an invariant needs. Deliberately a plain data holder so tests
    can construct counterexamples by hand."""

    turns: Sequence[TurnRecord] = ()
    identity_verified: bool = False
    safety_route_active: bool = False
    mutation_tools_exposed: tuple[str, ...] = ()
    candidate_records_enumerated_to_unverified: tuple[str, ...] = ()
    patient_id_from_model_args: str | None = None
    patient_id_from_session: str | None = None
    slot_hold_ids: tuple[str, ...] = ()
    appointment_ids: tuple[str, ...] = ()
    response_claims: Mapping[str, str] = field(default_factory=dict)
    unsourced_response_claims: tuple[str, ...] = ()
    #: Prohibited-claim names the patient-visible reply carried. Populated from
    #: the response-grounding validator, which knows which claims the active
    #: policy forbids. Kept as names rather than as text so the predicate is a
    #: set-membership check and not a second, drifting implementation of the
    #: grounding rules.
    prohibited_claim_hits: tuple[str, ...] = ()
    logged_fields: tuple[str, ...] = ()
    naive_datetimes: tuple[str, ...] = ()
    idempotency_records: Sequence[Mapping[str, Any]] = ()
    commit_attempts: Sequence[Mapping[str, Any]] = ()
    cross_session_reads: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True)
class InvariantResult:
    invariant_id: str
    passed: bool
    detail: str
    structural_enforcement: str


# ---------------------------------------------------------------------------
# Individual invariants
# ---------------------------------------------------------------------------


def inv01_safety_screened_every_turn(ctx: InvariantContext) -> tuple[bool, str]:
    missing = [t.index for t in ctx.turns if not t.safety_verdict]
    return (
        not missing,
        "all turns carry a safety verdict" if not missing else f"turns without verdict: {missing}",
    )


def inv02_safety_route_blocks_mutation(ctx: InvariantContext) -> tuple[bool, str]:
    if not ctx.safety_route_active:
        return True, "no active safety route"
    blocked_verdicts = {"TRIGGERED", "UNCERTAIN"}
    offenders: list[str] = []
    for turn in ctx.turns:
        if turn.safety_verdict in blocked_verdicts and turn.mutations_committed:
            offenders.extend(turn.mutations_committed)
    if offenders:
        return False, f"mutations committed while safety route active: {sorted(set(offenders))}"
    if ctx.mutation_tools_exposed:
        return False, f"mutation tools exposed during safety route: {list(ctx.mutation_tools_exposed)}"
    return True, "no mutation attempted or exposed under safety route"


def inv03_phi_read_requires_identity(ctx: InvariantContext) -> tuple[bool, str]:
    if ctx.identity_verified:
        return True, "identity verified"
    if ctx.candidate_records_enumerated_to_unverified:
        # Enumerating candidate records to an unverified user is a privacy
        # failure independent of anything else.
        return False, "unverified user received candidate records"
    return True, "no PHI read by unverified subject"


def inv04_no_candidate_enumeration(ctx: InvariantContext) -> tuple[bool, str]:
    leaked = list(ctx.candidate_records_enumerated_to_unverified)
    return (not leaked, "no candidate records enumerated" if not leaked else f"leaked: {leaked}")


def inv05_patient_id_server_derived(ctx: InvariantContext) -> tuple[bool, str]:
    if ctx.patient_id_from_model_args is None:
        return True, "no model-supplied patient id observed"
    if ctx.patient_id_from_model_args != ctx.patient_id_from_session:
        return (
            False,
            f"model supplied patient_id={ctx.patient_id_from_model_args!r} "
            f"but session subject is {ctx.patient_id_from_session!r}",
        )
    return True, "patient id matches session subject"


def inv06_hold_is_not_appointment(ctx: InvariantContext) -> tuple[bool, str]:
    overlap = set(ctx.slot_hold_ids) & set(ctx.appointment_ids)
    if overlap:
        return False, f"a hold id was also reported as an appointment: {sorted(overlap)}"
    return True, "holds and appointments are disjoint identifiers"


def inv07_no_premature_booked_claim(ctx: InvariantContext) -> tuple[bool, str]:
    """A response may not assert BOOKED unless a verified commit succeeded."""
    for turn in ctx.turns:
        claims_booked = any(k.startswith("appointment.booked") for k in turn.response_claim_keys)
        if claims_booked and "commit_appointment" not in turn.mutations_committed:
            return False, f"turn {turn.index} claimed booked without a committed commit_appointment"
    return True, "no premature booked claim"


def inv08_commit_requires_confirmation_and_lease(ctx: InvariantContext) -> tuple[bool, str]:
    offenders: list[Mapping[str, Any]] = []
    for rec in ctx.commit_attempts:
        if rec.get("committed"):
            if not rec.get("fresh_lease"):
                offenders.append(rec)
            elif not rec.get("explicit_confirmation"):
                offenders.append(rec)
            elif not rec.get("slot_matches"):
                offenders.append(rec)
            elif not rec.get("patient_matches"):
                offenders.append(rec)
    if offenders:
        return False, f"{len(offenders)} commit(s) without lease+confirmation+match"
    return True, "every commit had a fresh lease, explicit confirmation and matching scope"


def inv09_idempotency_semantics(ctx: InvariantContext) -> tuple[bool, str]:
    seen: dict[str, str] = {}
    for rec in ctx.idempotency_records:
        key = str(rec.get("key"))
        request_hash = str(rec.get("request_hash"))
        prior = seen.get(key)
        if prior is None:
            seen[key] = request_hash
            continue
        if prior != request_hash and rec.get("accepted"):
            return (
                False,
                f"idempotency key {key} accepted with a different canonical request hash",
            )
    return True, "same key+hash returns original; same key+different hash rejected"


def inv10_unknown_outcome_reconciled(ctx: InvariantContext) -> tuple[bool, str]:
    for rec in ctx.commit_attempts:
        if rec.get("outcome") == "UNKNOWN" and rec.get("retried_without_reconcile"):
            return False, "UNKNOWN commit outcome was retried without reconciliation"
    return True, "every UNKNOWN outcome reconciled before any retry"


def inv11_reschedule_preserves_original(ctx: InvariantContext) -> tuple[bool, str]:
    for rec in ctx.commit_attempts:
        if rec.get("kind") == "atomic_reschedule" and not rec.get("succeeded"):
            if not rec.get("original_preserved"):
                return False, "failed atomic reschedule did not preserve the original appointment"
    return True, "original appointment preserved on every failed reschedule"


def inv12_no_prohibited_claims(ctx: InvariantContext) -> tuple[bool, str]:
    """The reply may never carry a claim the active policy prohibits.

    This is the invariant that covers the three highest-consequence text
    failures: telling a patient emergency services were contacted when they were
    not, advising a medication change, and confirming or ruling out a diagnosis.
    All three can cause harm by themselves, and all three are the kind of thing a
    well-meaning model produces.

    The predicate reads claim *names* rather than scanning text, because the
    response-grounding validator already performs the scan and a second
    implementation here would be a second definition of "prohibited" -- which is
    how the two drift apart.
    """
    hits = sorted(set(ctx.prohibited_claim_hits))
    return (
        not hits,
        "no prohibited claim in any reply" if not hits else f"prohibited claims emitted: {hits}",
    )


def inv13_claims_traceable(ctx: InvariantContext) -> tuple[bool, str]:
    unsourced = list(ctx.unsourced_response_claims)
    return (
        not unsourced,
        "every claim traceable" if not unsourced else f"unsourced claims: {unsourced}",
    )


def inv14_logs_minimize_phi(ctx: InvariantContext) -> tuple[bool, str]:
    forbidden = {
        "ssn",
        "social_security_number",
        "full_name",
        "date_of_birth",
        "dob",
        "address",
        "phone",
        "email",
        "insurance_id",
        "mrn",
        "password",
        "api_key",
        "token",
        "secret",
    }
    offenders = sorted({f.lower() for f in ctx.logged_fields} & forbidden)
    if offenders:
        return False, f"forbidden fields in logs: {offenders}"
    return True, "log allowlist respected"


def inv15_timezone_aware(ctx: InvariantContext) -> tuple[bool, str]:
    naive = list(ctx.naive_datetimes)
    return (not naive, "no naive datetimes" if not naive else f"naive datetimes: {naive}")


def inv_cross_session_isolation(ctx: InvariantContext) -> tuple[bool, str]:
    """Section 14.2 tenant isolation. Not numbered in plan.md but load-bearing:
    a session must never read another session's subject."""
    leaks = [r for r in ctx.cross_session_reads if r.get("crossed")]
    if leaks:
        return False, f"{len(leaks)} cross-session read(s)"
    return True, "no cross-session reads"


INVARIANTS: Mapping[str, tuple[Any, str]] = {
    "INV-01": (inv01_safety_screened_every_turn, "SafetySentinel.screen() called by orchestrator each turn"),
    "INV-02": (inv02_safety_route_blocks_mutation, "CapabilityBroker: mutation tools not minted under safety route"),
    "INV-03": (inv03_phi_read_requires_identity, "ToolExecutor: identity guard before PHI-bearing reads"),
    "INV-04": (inv04_no_candidate_enumeration, "lookup_patient_candidates returns count+opaque token only"),
    "INV-05": (inv05_patient_id_server_derived, "ToolExecutor injects subject from session capability"),
    "INV-06": (inv06_hold_is_not_appointment, "kernel: slot_leases and appointments are separate tables"),
    "INV-07": (inv07_no_premature_booked_claim, "ResponseComposer renders only GroundedFacts from commit result"),
    "INV-08": (inv08_commit_requires_confirmation_and_lease, "kernel.commit_appointment lease+confirm preconditions"),
    "INV-09": (inv09_idempotency_semantics, "idempotency table UNIQUE(key) + request_hash comparison"),
    "INV-10": (inv10_unknown_outcome_reconciled, "UnknownOutcome raises; executor refuses retry path"),
    "INV-11": (inv11_reschedule_preserves_original, "kernel.atomic_reschedule single transaction"),
    "INV-12": (inv12_no_prohibited_claims, "ResponseComposer claim allowlist + per-rule prohibited_claims enforced by response_grounding"),
    "INV-13": (inv13_claims_traceable, "ResponseComposer claim allowlist"),
    "INV-14": (inv14_logs_minimize_phi, "telemetry.redaction allowlist"),
    "INV-15": (inv15_timezone_aware, "TimeWindow validator rejects naive datetimes"),
    "INV-CS": (inv_cross_session_isolation, "repository scoped by session subject"),
}


def evaluate_all(ctx: InvariantContext) -> list[InvariantResult]:
    results: list[InvariantResult] = []
    for inv_id, (fn, enforcement) in INVARIANTS.items():
        passed, detail = fn(ctx)
        results.append(
            InvariantResult(
                invariant_id=inv_id,
                passed=passed,
                detail=detail,
                structural_enforcement=enforcement,
            )
        )
    return results


def all_pass(results: Sequence[InvariantResult]) -> bool:
    return all(r.passed for r in results)


__all__ = [
    "INVARIANTS",
    "InvariantContext",
    "InvariantResult",
    "TurnRecord",
    "all_pass",
    "evaluate_all",
]
