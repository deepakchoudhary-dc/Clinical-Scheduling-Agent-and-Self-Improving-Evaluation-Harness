"""Transactional tool executor (plan.md section 8.2, 4.2).

The executor is the only path from a model's proposed action to a real state
change. It enforces, in order:

1. the tool exists and has a declared contract;
2. the capability is valid, unexpired, un-replayed and scoped to this tool;
3. the input validates against the strict contract (extra fields rejected);
4. the subject is injected server-side -- never read from the model's arguments;
5. the operation runs against the kernel;
6. a *trace entry* is recorded for every attempt, denial or not; an *audit row*
   is written only once a call reaches its handler. The audit table records what
   the system did, and a denial is a refusal to do anything, so it is visible in
   the trace rather than the audit chain;
7. the capability nonce is consumed exactly once.

A denial is a normal, expected result. It is returned as a structured outcome
rather than raised, because "the broker said no" is a fact the response layer
must be able to report honestly (plan.md section 3.4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import ValidationError as PydanticValidationError

from clinical_scheduler.db.repository import Repository
from clinical_scheduler.domain.errors import (
    CapabilityDenied,
    ClinicalSchedulerError,
    SafetyError,
    ToolError,
    UnknownOutcome,
    ValidationRejected,
)
from clinical_scheduler.domain.states import OutcomeKind, ReasonCode
from clinical_scheduler.kernel.scheduling import KernelResult, SchedulingKernel
from clinical_scheduler.tools.broker import (
    MUTATION_TOOLS,
    Capability,
    CapabilityBroker,
    SessionContext,
)
from clinical_scheduler.tools.contracts import CONTRACTS, StrictModel, ToolEnvelope, ToolStatus
from clinical_scheduler.telemetry import HARD_DENY, redact_text

#: Tool version strings. Bumping one is a governance event: the version is
#: recorded in every audit row so a behaviour change is attributable.
TOOL_VERSIONS: dict[str, str] = {name: f"{name}@1" for name in CONTRACTS}


@dataclass
class ToolCallRecord:
    """One tool attempt, success or denial. The trace oracle reads these."""

    turn_index: int
    tool_name: str
    correlation_id: str
    status: ToolStatus
    reason: ReasonCode
    outcome: OutcomeKind
    arguments_redacted: dict[str, Any]
    result_summary: dict[str, Any]
    latency_ms: int
    started_at: datetime
    subject_source: str
    injection_flagged: bool = False
    denial_detail: str | None = None


@dataclass
class ExecutorContext:
    """Everything the executor needs that is not the tool input."""

    session: SessionContext
    turn_index: int
    correlation_id: str
    policy_version: str
    trace_id: str
    session_id_hash: str
    #: Maps a confirmation token to an explicit user confirmation that the
    #: runtime observed. A model cannot mint these; only the confirmation step
    #: can, which is what makes invariant 8 enforceable.
    confirmation_tokens: dict[str, str] = field(default_factory=dict)
    mutation_log: list[dict[str, Any]] = field(default_factory=list)


class ToolExecutor:
    def __init__(
        self,
        *,
        repo: Repository,
        broker: CapabilityBroker,
        kernel: SchedulingKernel,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repo = repo
        self.broker = broker
        self.kernel = kernel
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.calls: list[ToolCallRecord] = []
        # The sequence makes derived row ids unique, so it must be unique across the
        # **database**, not merely across this process. Seeding it from the existing
        # audit rows means a restart continues the sequence instead of replaying it.
        #
        # Before this, restarting against an existing database reset the counter to 0
        # and regenerated ids that were already written:
        # `sqlite3.IntegrityError: UNIQUE constraint failed: audit_events.audit_id`,
        # found on 2026-10-09 by running `csched chat` twice against one session
        # file. That is not only a crash: for an append-only, tamper-evident log it
        # means a restart can collide with history. On a fresh database the count is
        # 0, so ids, hashes and every existing test are unaffected.
        existing = self.repo.query_one("SELECT COUNT(*) AS n FROM audit_events")
        self._audit_seq = int(existing["n"]) if existing else 0
        self._row_seq = 0

    @staticmethod
    def _row_id(prefix: str, correlation_id: str, seq: int) -> str:
        """Derive a unique row identifier.

        Primary keys derived from (correlation_id, tool_name) alone are not
        unique: a single turn legitimately invokes the same tool more than once
        (a patient revises a date and the search reruns), and derived rows such
        as audit events and collision attempts would then collide. Callers pass a
        monotonic sequence so uniqueness does not depend on the tool or the
        correlation being distinct.
        """
        return f"{prefix}_{correlation_id}_{seq:05d}"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def invoke(
        self,
        *,
        capability: Capability,
        tool_name: str,
        raw_arguments: dict[str, Any],
        ctx: ExecutorContext,
    ) -> dict[str, Any]:
        """Validate, authorize and execute one tool call.

        Returns a JSON-serialisable dict: the validated output model's dump, or
        a structured denial. Never raises for a policy refusal; only raises for a
        genuine defect (a contract that cannot be satisfied at all).
        """
        started = self._clock()
        t0 = time.perf_counter()
        injection_flagged = self._flag_injection(raw_arguments)

        def finish(
            status: ToolStatus,
            reason: ReasonCode,
            outcome: OutcomeKind,
            summary: dict[str, Any],
            detail: str | None = None,
            retriable: bool = False,
        ) -> dict[str, Any]:
            """Every return path goes through here so that callers can rely on a
            uniform shape. An envelope is always present, including for
            validation failures and denials, which is what lets the oracle stack
            reason about *all* attempts rather than only the successful ones."""
            latency = int((time.perf_counter() - t0) * 1000)
            try:
                envelope = ToolEnvelope(
                    correlation_id=ctx.correlation_id,
                    policy_version=ctx.policy_version,
                    tool_version=TOOL_VERSIONS.get(tool_name, f"{tool_name}@0"),
                    status=status,
                    outcome=outcome,
                    reason=reason,
                    retriable=retriable,
                    latency_ms=latency,
                    idempotency_key_hash=None,
                ).model_dump(mode="json")
            except PydanticValidationError as exc:
                # The envelope must never be the thing that fails: a malformed
                # runtime context would otherwise turn every tool call into an
                # unhandled exception, hiding the real result. Emit a minimal
                # valid envelope and record the context defect in the trace so
                # the runtime oracle can flag it.
                envelope = {
                    "correlation_id": "invalid",
                    "policy_version": "unavailable",
                    "tool_version": f"{tool_name}@0",
                    "status": ToolStatus.FAILURE.value,
                    "outcome": OutcomeKind.KNOWN_FAILURE.value,
                    "reason": ReasonCode.VALIDATION_ERROR.value,
                    "retriable": False,
                    "latency_ms": latency,
                    "idempotency_key_hash": None,
                    "context_defect": str(exc.errors(include_url=False)),
                }
            body = {k: v for k, v in summary.items() if k != "envelope"}
            result = {"envelope": envelope, **body}
            if detail:
                result["denial_detail"] = detail
            self.calls.append(
                ToolCallRecord(
                    turn_index=ctx.turn_index,
                    tool_name=tool_name,
                    correlation_id=ctx.correlation_id,
                    status=status,
                    reason=reason,
                    outcome=outcome,
                    arguments_redacted=_redact_arguments(raw_arguments),
                    result_summary=body,
                    latency_ms=latency,
                    started_at=started,
                    subject_source="server_session",
                    injection_flagged=injection_flagged,
                    denial_detail=detail,
                )
            )
            return result

        # 1. contract must exist
        if tool_name not in CONTRACTS:
            return finish(
                ToolStatus.DENIED, ReasonCode.TOOL_CONTRACT_ERROR, OutcomeKind.KNOWN_FAILURE,
                {"error": f"unknown tool {tool_name}"}, "no declared contract",
            )

        # 2. capability must verify
        try:
            self.broker.verify(capability, ctx=ctx.session, tool_name=tool_name, now=started)
        except SafetyError as exc:
            return finish(
                ToolStatus.DENIED, exc.reason, OutcomeKind.KNOWN_FAILURE,
                {"error": str(exc), "denials": exc.details}, "safety boundary",
            )
        except CapabilityDenied as exc:
            return finish(
                ToolStatus.DENIED, ReasonCode.CAPABILITY_DENIED, OutcomeKind.KNOWN_FAILURE,
                {"error": str(exc), "denials": exc.details}, "capability refused",
            )

        # 3. strict input validation
        input_model, _output_model = CONTRACTS[tool_name]
        try:
            parsed = input_model.model_validate(raw_arguments)
        except PydanticValidationError as exc:
            return finish(
                ToolStatus.FAILURE, ReasonCode.VALIDATION_ERROR, OutcomeKind.KNOWN_FAILURE,
                {"errors": exc.errors(include_url=False)}, "contract validation failed",
            )

        # 4. subject injection. The model never supplies this.
        arguments = parsed.model_dump()
        subject = ctx.session.subject_id
        if tool_name in MUTATION_TOOLS and not subject:
            return finish(
                ToolStatus.DENIED, ReasonCode.IDENTITY_UNVERIFIED, OutcomeKind.KNOWN_FAILURE,
                {"error": "no authenticated subject"}, "no subject bound",
            )

        # 5. run
        try:
            body, reason, outcome = self._dispatch(tool_name, arguments, ctx, started)
        except UnknownOutcome as exc:
            # UNKNOWN is a real outcome, not an error to retry blindly.
            return finish(
                ToolStatus.UNKNOWN, ReasonCode.UNKNOWN_OUTCOME, OutcomeKind.UNKNOWN,
                {"error": str(exc), "reconcile_required": True}, "unknown commit outcome",
            )
        except ToolError as exc:
            return finish(
                ToolStatus.FAILURE, exc.reason, exc.outcome,
                {"error": str(exc), **exc.details}, "tool error", retriable=exc.retriable,
            )
        except SafetyError as exc:
            return finish(
                ToolStatus.DENIED, exc.reason, OutcomeKind.KNOWN_FAILURE,
                {"error": str(exc)}, "safety boundary",
            )
        except ClinicalSchedulerError as exc:
            return finish(
                ToolStatus.FAILURE, exc.reason, OutcomeKind.KNOWN_FAILURE,
                {"error": str(exc), **exc.details}, "domain error",
            )

        # 6. consume the nonce only after a real execution
        self.broker.consume(capability)
        if tool_name in MUTATION_TOOLS:
            ctx.mutation_log.append(
                {
                    "tool_name": tool_name,
                    "reason": reason.value,
                    # Same rule as the envelope: the operation's own outcome
                    # decides whether a mutation committed. A successful
                    # join_waitlist returns the reason WAITLIST_JOINED, so
                    # comparing the reason to OK would record a committed
                    # mutation as not committed.
                    "committed": outcome is OutcomeKind.KNOWN_SUCCESS,
                    "arguments": _redact_arguments(raw_arguments),
                }
            )

        # A tool call that *returns* is not a tool call that *succeeded*. The
        # kernel reports refusals (slot already taken, lease consumed, not the
        # owner, service mismatch) as a normal return, and labelling those
        # SUCCESS is how the agent comes to tell a patient an appointment is
        # booked when it is not.
        #
        # The outcome is taken from the operation that produced it, never
        # inferred from the reason code. Inferring it is tempting and wrong:
        # ``check_requirements`` returns ``KNOWN_SUCCESS`` with the reason
        # ``REQUIREMENTS_UNMET``, because the *query* succeeded even though its
        # answer is "no". A reason-based rule reports that as a failure, and a
        # hand-classified table of success codes is a second definition of truth
        # that drifts the moment a new code is added.
        #
        # UNKNOWN cannot reach here: the kernel raises ``UnknownOutcome`` for it,
        # and that is handled above with the exception's own outcome.
        succeeded = outcome is OutcomeKind.KNOWN_SUCCESS
        return finish(
            ToolStatus.SUCCESS if succeeded else ToolStatus.FAILURE,
            reason,
            outcome,
            body,
            None if succeeded else "refused by the scheduling kernel",
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        ctx: ExecutorContext,
        started: datetime,
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind]:
        """Run a handler and audit it. Returns (body, reason, outcome).

        The outcome travels with the reason rather than being derived from it.
        The envelope is built by :meth:`invoke` so that success, failure and
        denial share one construction site.
        """
        handler = getattr(self, f"_do_{tool_name}", None)
        if handler is None:
            raise ValidationRejected(f"tool {tool_name} has a contract but no handler")
        body, reason, outcome, evidence = handler(arguments, ctx, started)
        self._audit(tool_name, ctx, arguments, {"reason": reason.value, **evidence}, started)
        return body, reason, outcome

    # -- safety ---------------------------------------------------------------

    def _do_route_urgent_concern(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        from clinical_scheduler.policy.engine import load_baseline

        escalation_id = f"esc_{ctx.trace_id[-8:]}"
        locale = "en-US"
        template_id = "emergency_unmonitored_us"
        policy = load_baseline()
        template = policy.bundle.template_by_id(template_id)
        text = template.text if template else (
            "Please contact emergency services now. I have not contacted anyone on your behalf."
        )
        self.repo.execute(
            """INSERT INTO simulated_escalations
               (escalation_id, session_id_hash, category, locale, message_template_id,
                actually_contacted_services, created_at)
               VALUES (?,?,?,?,?,0,?)""",
            (
                escalation_id, ctx.session_id_hash, str(a["category"]), locale,
                template_id, now.isoformat(),
            ),
        )
        return (
            {
                "escalation_id": escalation_id,
                "locale": locale,
                "message_template_id": template_id,
                "actually_contacted_services": False,
                "patient_safe_text": text,
            },
            ReasonCode.OK,
                        OutcomeKind.KNOWN_SUCCESS,
            {"escalation_id": escalation_id},
        )

    def _do_create_human_handoff(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        handoff_id = f"ho_{ctx.trace_id[-8:]}"
        self.repo.execute(
            """INSERT INTO handoffs
               (handoff_id, session_id_hash, reason_code, summary, contact_path,
                is_monitored, created_at)
               VALUES (?,?,?,?,?,0,?)""",
            (
                handoff_id, ctx.session_id_hash, str(a["reason_code"]),
                a["summary"], a["contact_path"], now.isoformat(),
            ),
        )
        return (
            {"handoff_id": handoff_id, "is_monitored": False},
            ReasonCode.OK,
                        OutcomeKind.KNOWN_SUCCESS,
            {"handoff_id": handoff_id},
        )

    # -- identity -------------------------------------------------------------

    def _do_lookup_patient_candidates(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        """Zero-knowledge collision resolution (plan.md section 9.1).

        Returns a count and an opaque token only. The candidate identifiers are
        stored inside the token row and are never returned through this or any
        other tool.
        """
        import json
        import secrets

        name_pattern = f"%{a['family_name'].strip()}%"
        rows = self.repo.query(
            "SELECT patient_id, birth_date FROM patients WHERE display_name LIKE ?",
            (name_pattern,),
        )
        matched = [r for r in rows if r["birth_date"] == a["birth_date"]]

        if not matched:
            return (
                {
                    "candidate_count": 0,
                    "collision_token": None,
                    "allowed_discriminators": (),
                    "candidate_records": (),
                    "attempts_remaining": 0,
                },
                ReasonCode.OK,
                                OutcomeKind.KNOWN_SUCCESS,
                {"candidate_count": 0},
            )

        if len(matched) == 1:
            # A single unambiguous match still requires a discriminator: knowing
            # one person shares a name must not itself confirm they are the
            # caller (plan.md section 12.3, "caller knows one discriminator but
            # fails another").
            candidate_ids = [matched[0]["patient_id"]]
        else:
            candidate_ids = [r["patient_id"] for r in matched]

        token = f"col_{secrets.token_hex(8)}"
        allowed = ("postal_code", "birth_date")
        self.repo.execute(
            """INSERT INTO identity_collision_tokens
               (collision_token, candidate_count, candidate_ids, allowed_discriminators,
                attempts_remaining, created_at, expires_at, consumed)
               VALUES (?,?,?,?,?,?,?,0)""",
            (
                token, len(candidate_ids), json.dumps(candidate_ids), json.dumps(allowed),
                3, now.isoformat(), now.isoformat(),
            ),
        )
        return (
            {
                "candidate_count": len(candidate_ids),
                "collision_token": token,
                "allowed_discriminators": allowed,
                "candidate_records": (),
                "attempts_remaining": 3,
            },
            ReasonCode.OK,
                        OutcomeKind.KNOWN_SUCCESS,
            {"candidate_count": len(candidate_ids), "collision_token": token},
        )

    def _do_verify_patient_discriminator(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        import json

        from clinical_scheduler.db.repository import sha256_hex
        from clinical_scheduler.db.seed import DEMO_SALT

        token_row = self.repo.query_one(
            "SELECT * FROM identity_collision_tokens WHERE collision_token = ?",
            (a["collision_token"],),
        )
        if token_row is None or token_row["consumed"]:
            raise CapabilityDenied("collision token is unknown or already consumed")

        attempts = int(token_row["attempts_remaining"])
        if attempts <= 0:
            return (
                {"verified": False, "subject_bound": False, "attempts_remaining": 0},
                ReasonCode.DISCRIMINATOR_ATTEMPTS_EXHAUSTED,
                                OutcomeKind.KNOWN_FAILURE,
                {},
            )

        candidate_ids = json.loads(token_row["candidate_ids"])
        wanted_hash = sha256_hex(f"{DEMO_SALT}:{a['discriminator_value'].strip()}")
        match = self.repo.query_one(
            """SELECT patient_id FROM patient_identifiers
                WHERE identifier_type = ? AND token_hash = ?
                  AND patient_id IN ({})""".format(",".join("?" * len(candidate_ids))),
            (a["discriminator_type"], wanted_hash, *candidate_ids),
        )

        remaining = attempts - 1
        self.repo.execute(
            "UPDATE identity_collision_tokens SET attempts_remaining = ?, consumed = ? WHERE collision_token = ?",
            (remaining, 1 if (match or remaining == 0) else 0, a["collision_token"]),
        )
        self._row_seq += 1
        self.repo.execute(
            """INSERT INTO collision_attempts
               (attempt_id, collision_token, discriminator_type, matched, attempt_index, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                self._row_id("att", ctx.correlation_id, self._row_seq),
                a["collision_token"], a["discriminator_type"],
                int(bool(match)), attempts - remaining, now.isoformat(),
            ),
        )

        if match is None:
            return (
                {"verified": False, "subject_bound": False, "attempts_remaining": remaining},
                ReasonCode.DISCRIMINATOR_MISMATCH,
                                OutcomeKind.KNOWN_FAILURE,
                {"matched": False},
            )

        # Bind the subject server-side. The identifier is never returned.
        ctx.session.subject_id = match["patient_id"]
        ctx.session.verified_identity = True
        return (
            {"verified": True, "subject_bound": True, "attempts_remaining": remaining},
            ReasonCode.OK,
                        OutcomeKind.KNOWN_SUCCESS,
            {"matched": True},
        )

    def _do_verify_proxy_scope(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        import json

        row = self.repo.query_one(
            "SELECT * FROM related_persons WHERE related_person_id = ?",
            (a["related_person_ref"],),
        )
        if row is None or row["revoked"]:
            return (
                {"permitted": False, "permitted_actions": (), "requires_staff_review": True},
                ReasonCode.PROXY_SCOPE_INSUFFICIENT,
                                OutcomeKind.KNOWN_FAILURE,
                {"revoked": bool(row and row["revoked"])},
            )
        permitted = tuple(json.loads(row["permitted_actions"]))
        allow = a["intended_action"] in permitted
        review = bool(row["custody_restricted"]) or not allow
        return (
            {
                "permitted": allow and not row["custody_restricted"],
                "permitted_actions": permitted,
                "requires_staff_review": review,
            },
            ReasonCode.OK if allow else ReasonCode.PROXY_SCOPE_INSUFFICIENT,
                        OutcomeKind.KNOWN_SUCCESS if allow else OutcomeKind.KNOWN_FAILURE,
            {"permitted_actions": list(permitted)},
        )

    # -- discovery ------------------------------------------------------------

    def _do_resolve_service(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        query = a["query_text"].strip().lower()
        words = [w for w in query.split() if len(w) > 2]
        rows = self.repo.query("SELECT * FROM healthcare_services ORDER BY service_id")
        scored: list[tuple[int, Any]] = []
        for r in rows:
            haystack = f"{r['service_id']} {r['name']}".lower()
            score = sum(1 for w in words if w in haystack)
            if score:
                scored.append((score, r))
        scored.sort(key=lambda t: (-t[0], t[1]["service_id"]))
        top = [r for _s, r in scored[:3]]
        candidates = tuple(
            {
                "service_id": r["service_id"],
                "name": r["name"],
                "duration_minutes": r["duration_minutes"],
                "requires_referral": bool(r["requires_referral"]),
                "requires_prior_authorization": bool(r["requires_prior_authorization"]),
                "requires_preparation_ack": bool(r["requires_preparation_ack"]),
            }
            for r in top
        )
        ambiguous = len(candidates) > 1
        return (
            {"candidates": candidates, "ambiguous": ambiguous},
            ReasonCode.OK if candidates else ReasonCode.SERVICE_NOT_FOUND,
                        # The tool exists to resolve a service; not finding one did not
                        # achieve that, so it is a known failure rather than a successful
                        # query with an empty answer.
                        OutcomeKind.KNOWN_SUCCESS if candidates else OutcomeKind.KNOWN_FAILURE,
            {"candidate_ids": [c["service_id"] for c in candidates]},
        )

    def _do_list_practitioner_roles(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        rows = self.repo.query(
            """SELECT pr.practitioner_id, pr.role_code, p.display_name
                 FROM practitioner_roles pr
                 JOIN practitioners p ON p.practitioner_id = pr.practitioner_id
                WHERE pr.service_id = ? ORDER BY pr.practitioner_id""",
            (a["service_id"],),
        )
        roles = tuple(
            {
                "practitioner_id": r["practitioner_id"],
                "role_code": r["role_code"],
                "display_name": r["display_name"],
            }
            for r in rows
        )
        return (
                    {"roles": roles},
                    ReasonCode.OK,
                    OutcomeKind.KNOWN_SUCCESS,
                    {"role_count": len(roles)},
                )

    def _do_search_slots(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        result = self.kernel.search_slots(
            service_id=a["service_id"],
            earliest=a["earliest_utc"],
            latest=a["latest_utc"],
            location_id=a.get("location_id"),
        )
        slots = tuple(
            {
                "slot_id": s["slot_id"],
                "service_id": s["service_id"],
                "practitioner_id": s["practitioner_id"],
                "practitioner_name": s["practitioner_name"],
                "location_id": s["location_id"],
                "modality": s["modality"],
                "start_utc": s["start_utc"],
                "end_utc": s["end_utc"],
                "display_zone": s["display_zone"],
                "advisory_only": True,
            }
            for s in result.payload.get("slots", [])
        )
        return {"slots": slots}, result.reason, result.outcome, {"slot_count": len(slots)}

    def _do_check_requirements(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        result = self.kernel.check_requirements(
            service_id=a["service_id"], request_id=a.get("request_id")
        )
        payload = result.payload
        return (
            {
                "requirements_met": payload.get("requirements_met", False),
                "unmet": tuple(payload.get("unmet", [])),
                "satisfied": tuple(payload.get("satisfied", [])),
                "preparation_ack_key": payload.get("preparation_ack_key"),
            },
            result.reason, result.outcome,
            {"unmet_count": len(payload.get("unmet", []))},
        )

    # -- mutations ------------------------------------------------------------

    def _confirm(self, token: str | None, ctx: ExecutorContext) -> bool:
        """Resolve a confirmation token against confirmations the *runtime*
        observed.

        A model cannot invent a token that maps to a real user confirmation.
        This is what makes invariant 8 structural rather than aspirational.
        """
        if not token:
            return False
        return token in ctx.confirmation_tokens

    def _do_hold_slot(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        subject = ctx.session.subject_id
        assert subject is not None  # guaranteed by invoke()
        key = ctx.correlation_id
        result = self.kernel.hold_slot(
            slot_id=a["slot_id"], patient_id=subject, idempotency_key=key
        )
        if not result.ok:
            return (
                {"lease_id": None, "slot_id": a["slot_id"], "expires_at": None,
                 "start_utc": None, "end_utc": None, "display_zone": None,
                 "is_appointment": False},
                result.reason, result.outcome,
                {"reason": result.reason.value},
            )
        return (
            {
                "lease_id": result.payload["lease_id"],
                "slot_id": result.payload["slot_id"],
                "expires_at": datetime.fromisoformat(result.payload["expires_at"]),
                "start_utc": datetime.fromisoformat(result.payload["start_utc"]),
                "end_utc": datetime.fromisoformat(result.payload["end_utc"]),
                "display_zone": result.payload["display_zone"],
                "is_appointment": False,
            },
            result.reason, result.outcome,
            {"lease_id": result.payload["lease_id"]},
        )

    def _do_commit_appointment(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        subject = ctx.session.subject_id
        assert subject is not None
        confirmed = self._confirm(a.get("confirmation_token"), ctx)
        result = self.kernel.commit_appointment(
            lease_id=a["lease_id"],
            patient_id=subject,
            service_id=a["service_id"],
            idempotency_key=a["idempotency_key"],
            explicit_confirmation=confirmed,
        )
        if not result.ok:
            return ({"appointment_id": None, "status": None}, result.reason, result.outcome, {"reason": result.reason.value})

        appt = self.repo.query_one(
            """SELECT a.*, p.display_name AS practitioner_name, l.name AS location_name
                 FROM appointments a
                 JOIN practitioners p ON p.practitioner_id = a.practitioner_id
                 JOIN locations l ON l.location_id = a.location_id
                WHERE a.appointment_id = ?""",
            (result.payload["appointment_id"],),
        )
        return (
            {
                "appointment_id": result.payload["appointment_id"],
                "slot_id": result.payload["slot_id"],
                "service_id": result.payload["service_id"],
                "practitioner_name": appt["practitioner_name"] if appt else None,
                "location_name": appt["location_name"] if appt else None,
                "start_utc": datetime.fromisoformat(result.payload["start_utc"]),
                "end_utc": datetime.fromisoformat(result.payload["end_utc"]),
                "display_zone": result.payload["display_zone"],
                "status": result.payload["status"],
            },
            result.reason, result.outcome,
            {"appointment_id": result.payload["appointment_id"]},
        )

    def _do_reconcile_commit(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        result = self.kernel.reconcile_commit(idempotency_key=a["idempotency_key"])
        return (
            {
                "resolved": bool(result.payload.get("resolved")),
                "appointment_id": result.payload.get("appointment_id"),
                "hold_still_valid": bool(result.payload.get("hold_still_valid", False)),
                "lease_id": result.payload.get("lease_id"),
            },
            result.reason, result.outcome,
            {"resolution": result.reason.value},
        )

    def _do_atomic_reschedule(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        subject = ctx.session.subject_id
        assert subject is not None
        confirmed = self._confirm(a.get("confirmation_token"), ctx)
        original_before = self.repo.query_one(
            "SELECT appointment_id, status FROM appointments WHERE appointment_id = ?",
            (a["original_appointment_id"],),
        )
        result = self.kernel.atomic_reschedule(
            original_appointment_id=a["original_appointment_id"],
            new_lease_id=a["new_lease_id"],
            patient_id=subject,
            service_id=a["service_id"],
            idempotency_key=a["idempotency_key"],
            explicit_confirmation=confirmed,
        )
        if not result.ok:
            original_after = self.repo.query_one(
                "SELECT appointment_id, status FROM appointments WHERE appointment_id = ?",
                (a["original_appointment_id"],),
            )
            preserved = bool(
                original_before
                and original_after
                and dict(original_before) == dict(original_after)
            )
            return (
                {
                    "original_appointment_id": a["original_appointment_id"],
                    "appointment_id": None,
                    "start_utc": None,
                    "end_utc": None,
                    "display_zone": None,
                    "original_preserved": preserved,
                },
                result.reason, result.outcome,
                {"reason": result.reason.value, "original_preserved": preserved},
            )
        return (
            {
                "original_appointment_id": result.payload["original_appointment_id"],
                "appointment_id": result.payload["appointment_id"],
                "start_utc": datetime.fromisoformat(result.payload["start_utc"]),
                "end_utc": datetime.fromisoformat(result.payload["end_utc"]),
                "display_zone": result.payload["display_zone"],
                "original_preserved": True,
            },
            result.reason, result.outcome,
            {"appointment_id": result.payload["appointment_id"]},
        )

    def _do_cancel_appointment(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        subject = ctx.session.subject_id
        assert subject is not None
        result = self.kernel.cancel_appointment(
            appointment_id=a["appointment_id"],
            patient_id=subject,
            reason_code=a["reason_code"],
            idempotency_key=ctx.correlation_id,
        )
        if not result.ok:
            return ({"appointment_id": None, "status": None, "slot_freed": None}, result.reason, result.outcome, {"reason": result.reason.value})
        return (
            {
                "appointment_id": result.payload["appointment_id"],
                "status": result.payload["status"],
                "slot_freed": result.payload["slot_freed"],
            },
            result.reason, result.outcome,
            {"appointment_id": result.payload["appointment_id"]},
        )

    def _do_join_waitlist(
        self, a: dict[str, Any], ctx: ExecutorContext, now: datetime
    ) -> tuple[dict[str, Any], ReasonCode, OutcomeKind, dict[str, Any]]:
        subject = ctx.session.subject_id
        assert subject is not None
        result = self.kernel.join_waitlist(
            patient_id=subject,
            service_id=a["service_id"],
            earliest=a["earliest_utc"],
            latest=a["latest_utc"],
            display_zone=a["display_zone"],
            idempotency_key=ctx.correlation_id,
        )
        if not result.ok:
            return ({"waitlist_id": None, "status": None}, result.reason, result.outcome, {"reason": result.reason.value})
        return (
            {"waitlist_id": result.payload["waitlist_id"], "status": result.payload["status"]},
            result.reason, result.outcome,
            {"waitlist_id": result.payload["waitlist_id"]},
        )

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _audit(
        self,
        tool_name: str,
        ctx: ExecutorContext,
        arguments: dict[str, Any],
        evidence: dict[str, Any],
        now: datetime,
    ) -> None:
        """Append one audit row per *invocation*.

        The audit id must be unique per call, not per (correlation, tool): a
        single turn legitimately calls the same tool more than once, for example
        when a patient revises a date and the search runs again. Keying on the
        correlation alone would raise a primary-key violation and, worse, would
        have silently overwritten the first attempt's evidence had the column
        been mutable.
        """
        self._audit_seq += 1
        self.repo.append_audit(
            audit_id=self._row_id("aud", ctx.correlation_id, self._audit_seq),
            event_type=f"TOOL_{tool_name.upper()}_INVOKED",
            trace_id=ctx.trace_id,
            session_id_hash=ctx.session_id_hash,
            policy_version=ctx.policy_version,
            payload={"arguments": _redact_arguments(arguments), "evidence": evidence},
            tool_version=TOOL_VERSIONS.get(tool_name, f"{tool_name}@0"),
        )

    @staticmethod
    def _flag_injection(arguments: dict[str, Any]) -> bool:
        """Detect obvious injection shapes in model-supplied arguments.

        This is a *detector*, not a sanitizer. The defence is that arguments are
        validated against a strict schema and the subject is injected
        server-side; flagging exists so the trace oracle can count attempts and
        so a human reviewer sees them.
        """
        markers = (
            "ignore previous", "ignore all", "system:", "you are now",
            "drop table", "select * from", "disregard", "<script", "{{",
            "override", "as an ai", "reveal another patient",
        )
        blob = " ".join(str(v) for v in arguments.values()).lower()
        return any(m in blob for m in markers)


#: Argument names whose *value* is masked before the argument is stored. The
#: vocabulary is taken from the telemetry layer's ``HARD_DENY`` rather than
#: re-declared here, so the audit boundary and the redaction layer cannot drift
#: apart: adding a field to ``HARD_DENY`` masks it here automatically.
_SENSITIVE_ARGUMENTS: frozenset[str] = frozenset(
    {"discriminator_value", "family_name", "postal_code", "phone_last4", "confirmation_token"}
) | HARD_DENY


def _redact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments are stored, but identifiers are masked and free text is
    passed through pattern redaction, so the trace is not a PHI store.

    Two controls, matching the telemetry layer: a *field* control (a masked value
    for a sensitive name) and a *value* control (``redact_text`` for free-text
    arguments such as a handoff summary or a service query).
    """
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in _SENSITIVE_ARGUMENTS:
            out[key] = "[REDACTED]"
        elif isinstance(value, str):
            masked = redact_text(value)
            out[key] = masked[:120] + "...[truncated]" if len(masked) > 120 else masked
        else:
            out[key] = value
    return out


__all__ = [
    "ExecutorContext",
    "TOOL_VERSIONS",
    "ToolCallRecord",
    "ToolExecutor",
]
