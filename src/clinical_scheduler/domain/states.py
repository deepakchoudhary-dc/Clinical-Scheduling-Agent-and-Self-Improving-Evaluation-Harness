"""Canonical domain vocabulary.

Design note (plan.md section 9 / agent.md "Domain Context"): domain language is
declared explicitly here and imported everywhere else. Nothing in this project is
allowed to invent a synonym for a concept defined in this module.

All datetimes crossing this boundary are timezone-aware. The storage layer keeps
UTC instants plus an IANA zone for clinic-local display (invariant 15).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base for every domain and contract model.

    `extra="forbid"` is not stylistic. plan.md section 8.1 requires that
    unexpected fields are rejected: an extra field is either a model bug or an
    injection attempt, and both must fail closed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=False)


class FrozenStrEnum(str, Enum):
    """String enum that serialises to its value, for stable audit/DB storage."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# ---------------------------------------------------------------------------
# Conversation / FSM states (plan.md section 5)
# ---------------------------------------------------------------------------


class SessionState(FrozenStrEnum):
    START = "START"
    SAFETY_SCREENED = "SAFETY_SCREENED"
    URGENT_HANDOFF = "URGENT_HANDOFF"
    IDENTITY_PENDING = "IDENTITY_PENDING"
    IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS"
    IDENTITY_VERIFIED = "IDENTITY_VERIFIED"
    AUTHORIZATION_VERIFIED = "AUTHORIZATION_VERIFIED"
    REQUEST_SCOPED = "REQUEST_SCOPED"
    REQUIREMENTS_PENDING = "REQUIREMENTS_PENDING"
    REQUIREMENTS_SATISFIED = "REQUIREMENTS_SATISFIED"
    OPTIONS_PRESENTED = "OPTIONS_PRESENTED"
    USER_SELECTED = "USER_SELECTED"
    SLOT_HELD = "SLOT_HELD"
    SUMMARY_CONFIRMED = "SUMMARY_CONFIRMED"
    COMMITTING = "COMMITTING"
    BOOKED = "BOOKED"
    COMMIT_UNKNOWN = "COMMIT_UNKNOWN"
    HOLD_EXPIRED = "HOLD_EXPIRED"
    RESCHEDULE_PENDING = "RESCHEDULE_PENDING"
    NEW_SLOT_HELD = "NEW_SLOT_HELD"
    ATOMIC_RESCHEDULED = "ATOMIC_RESCHEDULED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    HUMAN_HANDOFF = "HUMAN_HANDOFF"


TERMINAL_STATES: frozenset[SessionState] = frozenset(
    {SessionState.URGENT_HANDOFF, SessionState.HUMAN_HANDOFF, SessionState.CANCELLED,
     SessionState.ATOMIC_RESCHEDULED}
)


# ---------------------------------------------------------------------------
# Authorization tiers (plan.md section 9.2)
# ---------------------------------------------------------------------------


class AuthTier(int, Enum):
    TIER0_PUBLIC = 0
    TIER1_NEW_PATIENT_REQUEST = 1
    TIER2_VERIFIED_IDENTITY = 2
    TIER3_STEP_UP = 3


# ---------------------------------------------------------------------------
# Safety routing (plan.md section 10)
# ---------------------------------------------------------------------------


class SafetyVerdict(FrozenStrEnum):
    """Outcome of the layered safety sentinel."""

    CLEAR = "CLEAR"
    CLEAR_NEGATED = "CLEAR_NEGATED"
    CLEAR_HISTORICAL = "CLEAR_HISTORICAL"
    CLEAR_THIRD_PERSON = "CLEAR_THIRD_PERSON"
    CLEAR_META = "CLEAR_META"
    UNCERTAIN = "UNCERTAIN"
    TRIGGERED = "TRIGGERED"


#: Verdicts that must remove every mutation capability for the turn (invariant 2).
MUTATION_BLOCKING_VERDICTS: frozenset[SafetyVerdict] = frozenset(
    {SafetyVerdict.TRIGGERED, SafetyVerdict.UNCERTAIN}
)

#: Verdicts that are safe to proceed on, each carrying a distinct audit reason.
CLEAR_VERDICTS: frozenset[SafetyVerdict] = frozenset(
    {
        SafetyVerdict.CLEAR,
        SafetyVerdict.CLEAR_NEGATED,
        SafetyVerdict.CLEAR_HISTORICAL,
        SafetyVerdict.CLEAR_THIRD_PERSON,
        SafetyVerdict.CLEAR_META,
    }
)


class UrgencyCategory(FrozenStrEnum):
    """Coarse routing categories. These are NOT diagnoses or triage levels.

    plan.md section 2.2/A forbids the agent from assigning clinical acuity
    levels. This enum names the *routing action bucket* only.
    """

    MEDICAL_EMERGENCY = "MEDICAL_EMERGENCY"
    BEHAVIOURAL_CRISIS = "BEHAVIOURAL_CRISIS"
    POISONING_OVERDOSE = "POISONING_OVERDOSE"
    OBSTETRIC_URGENT = "OBSTETRIC_URGENT"
    UNSPECIFIED_HIGH_RISK = "UNSPECIFIED_HIGH_RISK"


# ---------------------------------------------------------------------------
# Appointment lifecycle
# ---------------------------------------------------------------------------


class AppointmentStatus(FrozenStrEnum):
    PROPOSED = "PROPOSED"
    BOOKED = "BOOKED"
    CANCELLED = "CANCELLED"
    RESCHEDULED = "RESCHEDULED"
    NO_SHOW = "NO_SHOW"
    COMPLETED = "COMPLETED"


#: Legal appointment status transitions (plan.md section 7.2).
ALLOWED_STATUS_TRANSITIONS: Mapping[AppointmentStatus, frozenset[AppointmentStatus]] = {
    AppointmentStatus.PROPOSED: frozenset(
        {AppointmentStatus.BOOKED, AppointmentStatus.CANCELLED}
    ),
    AppointmentStatus.BOOKED: frozenset(
        {
            AppointmentStatus.CANCELLED,
            AppointmentStatus.RESCHEDULED,
            AppointmentStatus.COMPLETED,
            AppointmentStatus.NO_SHOW,
        }
    ),
    AppointmentStatus.RESCHEDULED: frozenset(
        {AppointmentStatus.CANCELLED, AppointmentStatus.COMPLETED, AppointmentStatus.NO_SHOW}
    ),
    AppointmentStatus.CANCELLED: frozenset(),
    AppointmentStatus.NO_SHOW: frozenset({AppointmentStatus.CANCELLED}),
    AppointmentStatus.COMPLETED: frozenset(),
}


class RequirementKind(FrozenStrEnum):
    REFERRAL = "REFERRAL"
    PRIOR_AUTHORIZATION = "PRIOR_AUTHORIZATION"
    QUESTIONNAIRE = "QUESTIONNAIRE"
    PREPARATION_ACKNOWLEDGEMENT = "PREPARATION_ACKNOWLEDGEMENT"
    INTERPRETER = "INTERPRETER"
    ACCESSIBILITY_ACCOMMODATION = "ACCESSIBILITY_ACCOMMODATION"


class RequirementStatus(FrozenStrEnum):
    PENDING = "PENDING"
    SATISFIED = "SATISFIED"
    WAIVED = "WAIVED"
    EXPIRED = "EXPIRED"


class OutcomeKind(FrozenStrEnum):
    """Tool outcome certainty (plan.md section 8.2).

    UNKNOWN is a first-class outcome, not an error. Invariant 10 forbids blind
    retry after UNKNOWN; the caller must reconcile first.
    """

    KNOWN_SUCCESS = "KNOWN_SUCCESS"
    KNOWN_FAILURE = "KNOWN_FAILURE"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Reason codes (plan.md section 8.2: "stable reason codes plus patient-safe text")
# ---------------------------------------------------------------------------


class ReasonCode(FrozenStrEnum):
    OK = "OK"
    SLOT_UNAVAILABLE = "SLOT_UNAVAILABLE"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    LEASE_NOT_FOUND = "LEASE_NOT_FOUND"
    LEASE_CONFLICT = "LEASE_CONFLICT"
    IDEMPOTENT_REPLAY = "IDEMPOTENT_REPLAY"
    IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_REQUEST = "IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_REQUEST"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    REQUIREMENTS_UNMET = "REQUIREMENTS_UNMET"
    IDENTITY_UNVERIFIED = "IDENTITY_UNVERIFIED"
    IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS"
    DISCRIMINATOR_MISMATCH = "DISCRIMINATOR_MISMATCH"
    DISCRIMINATOR_ATTEMPTS_EXHAUSTED = "DISCRIMINATOR_ATTEMPTS_EXHAUSTED"
    PROXY_SCOPE_INSUFFICIENT = "PROXY_SCOPE_INSUFFICIENT"
    SERVICE_NOT_FOUND = "SERVICE_NOT_FOUND"
    SERVICE_MISMATCH = "SERVICE_MISMATCH"
    APPOINTMENT_NOT_FOUND = "APPOINTMENT_NOT_FOUND"
    NOT_OWNER = "NOT_OWNER"
    STATUS_TRANSITION_INVALID = "STATUS_TRANSITION_INVALID"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    UNKNOWN_OUTCOME = "UNKNOWN_OUTCOME"
    RECONCILED_SUCCESS = "RECONCILED_SUCCESS"
    RECONCILED_FAILURE = "RECONCILED_FAILURE"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    WAITLIST_JOINED = "WAITLIST_JOINED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    SAFETY_ROUTE_ACTIVE = "SAFETY_ROUTE_ACTIVE"
    CAPABILITY_DENIED = "CAPABILITY_DENIED"
    STATE_MACHINE_BYPASS = "STATE_MACHINE_BYPASS"
    PRIVACY_DISCLOSURE = "PRIVACY_DISCLOSURE"
    TOOL_CONTRACT_ERROR = "TOOL_CONTRACT_ERROR"
    ORACLE_ERROR = "ORACLE_ERROR"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


class TimeWindow(StrictModel):
    """A half-open instant window [start, end) in UTC plus a display zone."""

    start: datetime
    end: datetime
    display_zone: str = Field(min_length=1)

    @field_validator("start", "end")
    @classmethod
    def _must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError(
                "timezone-naive datetime rejected: all instants must be aware (invariant 15)"
            )
        return v

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError("window end must be strictly after start")
        return self

    def to_utc_iso(self) -> tuple[str, str]:
        return (
            self.start.astimezone(timezone.utc).isoformat(),
            self.end.astimezone(timezone.utc).isoformat(),
        )


class Provenance(StrictModel):
    """Where a field value came from (plan.md section 4.2: 'extract candidate
    fields with provenance').

    Provenance is what lets Layer 3 assert that no value used in a decision was
    invented by the model.
    """

    field_name: str
    source: Literal["user_message", "verified_tool_result", "approved_static", "server_session"]
    turn_index: int = Field(ge=0)
    verbatim_span: str | None = None


class GroundedFact(StrictModel):
    """A single factual claim the response is allowed to make.

    plan.md section 5.1 invariant 13: every factual claim in a response must be
    traceable to tool output or approved static content. The response composer
    may only render GroundedFacts, never free-form model prose about state.
    """

    claim_key: str = Field(min_length=1)
    value: Any
    source: Literal["verified_tool_result", "approved_static"]
    evidence: str = Field(min_length=1)


__all__ = [
    "ALLOWED_STATUS_TRANSITIONS",
    "AppointmentStatus",
    "AuthTier",
    "CLEAR_VERDICTS",
    "FrozenStrEnum",
    "GroundedFact",
    "MUTATION_BLOCKING_VERDICTS",
    "OutcomeKind",
    "Provenance",
    "ReasonCode",
    "RequirementKind",
    "RequirementStatus",
    "SafetyVerdict",
    "SessionState",
    "StrictModel",
    "TERMINAL_STATES",
    "TimeWindow",
    "UrgencyCategory",
]
