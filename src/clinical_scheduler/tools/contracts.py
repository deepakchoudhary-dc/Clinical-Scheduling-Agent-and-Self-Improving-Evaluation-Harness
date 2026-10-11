"""Tool contracts (plan.md section 8.2).

Every tool has a typed input model and a typed output model. Contracts are
strict: extra fields are rejected, enums are closed, and free text is never
interpreted as an instruction.

Two rules that shape the whole file:

* **No patient identifier is an input.** Tools that act on a patient's behalf
  take a ``patient_id`` only where the executor injects it from the session
  capability. The model-facing schemas here therefore omit it for anything
  consequential; the executor fills it in from trusted state.
* **Reason codes carry meaning.** Outputs pair a stable ``ReasonCode`` with
  patient-safe text, so the response layer never has to infer meaning from
  prose.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from clinical_scheduler.domain.states import (
    FrozenStrEnum,
    OutcomeKind,
    ReasonCode,
    StrictModel,
    UrgencyCategory,
)


class ToolStatus(FrozenStrEnum):
    """Coarse tool outcome, mirroring :class:`OutcomeKind` for wire use."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNKNOWN = "UNKNOWN"
    DENIED = "DENIED"


class ToolEnvelope(StrictModel):
    """Common metadata every tool result carries.

    ``correlation_id`` ties the call to the trace; ``policy_version`` ties it to
    the governance record; ``idempotency_key_hash`` ties a mutation to its
    deduplication record without exposing the key.
    """

    correlation_id: str = Field(min_length=4)
    policy_version: str = Field(min_length=3)
    tool_version: str = Field(min_length=3)
    status: ToolStatus
    outcome: OutcomeKind
    reason: ReasonCode
    retriable: bool = False
    latency_ms: int = Field(ge=0)
    idempotency_key_hash: str | None = None


# ---------------------------------------------------------------------------
# Safety tools
# ---------------------------------------------------------------------------


class RouteUrgentConcernInput(StrictModel):
    category: UrgencyCategory
    matched_phrases: tuple[str, ...] = ()
    #: Free text the model may supply for the human reviewer. It is stored but
    #: never executed and never rendered verbatim to the patient.
    context_note: str = Field(default="", max_length=500)


class RouteUrgentConcernOutput(StrictModel):
    envelope: ToolEnvelope
    escalation_id: str
    locale: str
    message_template_id: str
    #: Always False for the prototype, and asserted by the safety oracle.
    actually_contacted_services: Literal[False] = False
    patient_safe_text: str


class CreateHumanHandoffInput(StrictModel):
    reason_code: ReasonCode
    summary: str = Field(min_length=5, max_length=400)
    contact_path: str = Field(min_length=3)


class CreateHumanHandoffOutput(StrictModel):
    envelope: ToolEnvelope
    handoff_id: str
    is_monitored: Literal[False] = False


# ---------------------------------------------------------------------------
# Identity tools
# ---------------------------------------------------------------------------


class LookupPatientCandidatesInput(StrictModel):
    """Deliberately minimal.

    The model supplies only what the patient said. It cannot request a
    disclosure of candidates, because the output has no field for them
    (plan.md section 9.1).
    """

    family_name: str = Field(min_length=1, max_length=80)
    birth_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class LookupPatientCandidatesOutput(StrictModel):
    envelope: ToolEnvelope
    #: Number of matches. May be suppressed by local policy; suppressed is
    #: reported as ``None`` rather than 0, because those are different facts.
    candidate_count: int | None = Field(default=None, ge=0)
    collision_token: str | None = None
    allowed_discriminators: tuple[str, ...] = ()
    #: Never populated. Present so the contract is explicit that it must not be.
    candidate_records: tuple[dict[str, Any], ...] = ()
    attempts_remaining: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _never_disclose_records(self) -> LookupPatientCandidatesOutput:
        if self.candidate_records:
            raise ValueError(
                "lookup_patient_candidates must never return candidate records; "
                "this field exists only to make the prohibition structural"
            )
        return self


class VerifyPatientDiscriminatorInput(StrictModel):
    collision_token: str = Field(pattern=r"^col_[0-9a-f]+$")
    discriminator_type: Literal["postal_code", "birth_date", "phone_last4"]
    discriminator_value: str = Field(min_length=2, max_length=40)


class VerifyPatientDiscriminatorOutput(StrictModel):
    envelope: ToolEnvelope
    verified: bool
    #: The session subject is set server-side as a result. This field reports
    #: that it changed; it does not carry the identifier to the model.
    subject_bound: bool = False
    attempts_remaining: int = Field(ge=0)


class VerifyProxyScopeInput(StrictModel):
    related_person_ref: str = Field(min_length=3)
    intended_action: Literal["book", "cancel", "reschedule", "view_details"]


class VerifyProxyScopeOutput(StrictModel):
    envelope: ToolEnvelope
    permitted: bool
    permitted_actions: tuple[str, ...] = ()
    #: True when the decision required staff review rather than an automatic
    #: answer. The response layer must not disclose the appointment reason when
    #: this is true.
    requires_staff_review: bool = False


# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------


class ResolveServiceInput(StrictModel):
    query_text: str = Field(min_length=2, max_length=200)


class ServiceCandidate(StrictModel):
    service_id: str
    name: str
    duration_minutes: int
    requires_referral: bool
    requires_prior_authorization: bool
    requires_preparation_ack: bool


class ResolveServiceOutput(StrictModel):
    envelope: ToolEnvelope
    candidates: tuple[ServiceCandidate, ...] = ()
    ambiguous: bool = False


class ListPractitionerRolesInput(StrictModel):
    service_id: str = Field(min_length=3)


class PractitionerRoleView(StrictModel):
    practitioner_id: str
    role_code: str
    display_name: str


class ListPractitionerRolesOutput(StrictModel):
    envelope: ToolEnvelope
    roles: tuple[PractitionerRoleView, ...] = ()


class SearchSlotsInput(StrictModel):
    service_id: str = Field(min_length=3)
    earliest_utc: datetime
    latest_utc: datetime
    location_id: str | None = None

    @field_validator("earliest_utc", "latest_utc")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("slot search bounds must be timezone-aware (invariant 15)")
        return v

    @model_validator(mode="after")
    def _ordered(self) -> SearchSlotsInput:
        if self.latest_utc <= self.earliest_utc:
            raise ValueError("latest_utc must be after earliest_utc")
        return self


class SlotView(StrictModel):
    slot_id: str
    service_id: str
    practitioner_id: str
    practitioner_name: str
    location_id: str
    modality: str
    start_utc: datetime
    end_utc: datetime
    display_zone: str
    #: Search results are advisory; the response layer must not present a slot
    #: as reserved (plan.md section 8.3).
    advisory_only: Literal[True] = True


class SearchSlotsOutput(StrictModel):
    envelope: ToolEnvelope
    slots: tuple[SlotView, ...] = ()


class CheckRequirementsInput(StrictModel):
    service_id: str = Field(min_length=3)
    request_id: str | None = None


class RequirementView(StrictModel):
    kind: str
    status: str
    requirement_id: str | None = None


class CheckRequirementsOutput(StrictModel):
    envelope: ToolEnvelope
    requirements_met: bool
    unmet: tuple[RequirementView, ...] = ()
    satisfied: tuple[RequirementView, ...] = ()
    preparation_ack_key: str | None = None


# ---------------------------------------------------------------------------
# Mutation tools
# ---------------------------------------------------------------------------


class HoldSlotInput(StrictModel):
    slot_id: str = Field(pattern=r"^sl_[a-z0-9_]+$")


class HoldSlotOutput(StrictModel):
    envelope: ToolEnvelope
    lease_id: str | None = None
    slot_id: str | None = None
    expires_at: datetime | None = None
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    display_zone: str | None = None
    #: Reinforces invariant 6 at the wire level: a hold is not an appointment,
    #: and the field name makes that visible in every trace.
    is_appointment: Literal[False] = False


class CommitAppointmentInput(StrictModel):
    lease_id: str = Field(pattern=r"^lease_[0-9a-f]+$")
    service_id: str = Field(min_length=3)
    #: Must be an explicit token produced by the confirmation step, not a bare
    #: boolean the model could set. The runtime maps the user's explicit
    #: confirmation to this token.
    confirmation_token: str = Field(min_length=8)
    idempotency_key: str = Field(min_length=8, max_length=128)


class CommitAppointmentOutput(StrictModel):
    envelope: ToolEnvelope
    appointment_id: str | None = None
    slot_id: str | None = None
    service_id: str | None = None
    practitioner_name: str | None = None
    location_name: str | None = None
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    display_zone: str | None = None
    status: str | None = None


class ReconcileCommitInput(StrictModel):
    idempotency_key: str = Field(min_length=8, max_length=128)


class ReconcileCommitOutput(StrictModel):
    envelope: ToolEnvelope
    resolved: bool
    appointment_id: str | None = None
    hold_still_valid: bool = False
    lease_id: str | None = None


class AtomicRescheduleInput(StrictModel):
    original_appointment_id: str = Field(pattern=r"^appt_[a-z0-9_]+$")
    new_lease_id: str = Field(pattern=r"^lease_[0-9a-f]+$")
    service_id: str = Field(min_length=3)
    confirmation_token: str = Field(min_length=8)
    idempotency_key: str = Field(min_length=8, max_length=128)


class AtomicRescheduleOutput(StrictModel):
    envelope: ToolEnvelope
    original_appointment_id: str | None = None
    appointment_id: str | None = None
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    display_zone: str | None = None
    #: True when the original was left intact because the replacement failed.
    original_preserved: bool = False


class CancelAppointmentInput(StrictModel):
    appointment_id: str = Field(pattern=r"^appt_[a-z0-9_]+$")
    reason_code: str = Field(min_length=3, max_length=40)


class CancelAppointmentOutput(StrictModel):
    envelope: ToolEnvelope
    appointment_id: str | None = None
    status: str | None = None
    slot_freed: str | None = None


class JoinWaitlistInput(StrictModel):
    service_id: str = Field(min_length=3)
    earliest_utc: datetime
    latest_utc: datetime
    display_zone: str = Field(min_length=3)

    @field_validator("earliest_utc", "latest_utc")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("waitlist bounds must be timezone-aware (invariant 15)")
        return v


class JoinWaitlistOutput(StrictModel):
    envelope: ToolEnvelope
    waitlist_id: str | None = None
    status: str | None = None


#: Maps tool name -> (input model, output model). The executor and the schema
#: validator both read this, so a tool added to the broker but not here is
#: immediately visible as a gap.
CONTRACTS: dict[str, tuple[type[StrictModel], type[StrictModel]]] = {
    "route_urgent_concern": (RouteUrgentConcernInput, RouteUrgentConcernOutput),
    "create_human_handoff": (CreateHumanHandoffInput, CreateHumanHandoffOutput),
    "lookup_patient_candidates": (LookupPatientCandidatesInput, LookupPatientCandidatesOutput),
    "verify_patient_discriminator": (VerifyPatientDiscriminatorInput, VerifyPatientDiscriminatorOutput),
    "verify_proxy_scope": (VerifyProxyScopeInput, VerifyProxyScopeOutput),
    "resolve_service": (ResolveServiceInput, ResolveServiceOutput),
    "list_practitioner_roles": (ListPractitionerRolesInput, ListPractitionerRolesOutput),
    "search_slots": (SearchSlotsInput, SearchSlotsOutput),
    "check_requirements": (CheckRequirementsInput, CheckRequirementsOutput),
    "hold_slot": (HoldSlotInput, HoldSlotOutput),
    "commit_appointment": (CommitAppointmentInput, CommitAppointmentOutput),
    "reconcile_commit": (ReconcileCommitInput, ReconcileCommitOutput),
    "atomic_reschedule": (AtomicRescheduleInput, AtomicRescheduleOutput),
    "cancel_appointment": (CancelAppointmentInput, CancelAppointmentOutput),
    "join_waitlist": (JoinWaitlistInput, JoinWaitlistOutput),
}


__all__ = [
    "CONTRACTS",
    "AtomicRescheduleInput",
    "AtomicRescheduleOutput",
    "CancelAppointmentInput",
    "CancelAppointmentOutput",
    "CheckRequirementsInput",
    "CheckRequirementsOutput",
    "CommitAppointmentInput",
    "CommitAppointmentOutput",
    "CreateHumanHandoffInput",
    "CreateHumanHandoffOutput",
    "HoldSlotInput",
    "HoldSlotOutput",
    "JoinWaitlistInput",
    "JoinWaitlistOutput",
    "ListPractitionerRolesInput",
    "ListPractitionerRolesOutput",
    "LookupPatientCandidatesInput",
    "LookupPatientCandidatesOutput",
    "ReconcileCommitInput",
    "ReconcileCommitOutput",
    "ResolveServiceInput",
    "ResolveServiceOutput",
    "RouteUrgentConcernInput",
    "RouteUrgentConcernOutput",
    "SearchSlotsInput",
    "SearchSlotsOutput",
    "ToolEnvelope",
    "ToolStatus",
    "VerifyPatientDiscriminatorInput",
    "VerifyPatientDiscriminatorOutput",
    "VerifyProxyScopeInput",
    "VerifyProxyScopeOutput",
]
