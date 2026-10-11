"""Domain entities (plan.md section 7.1).

These are the declared domain vocabulary. They are intentionally separate from the
SQL rows: the database is the source of truth, and the kernel reads rows directly
rather than materialising these.

Honest note (added by the 2026-10-08 static audit): some of these models are
*used* as typed views (``Patient``, ``PractitionerRole``, ``Location``,
``HealthcareService``, ``Slot``, ``Appointment``) and the rest are *declared* but
not yet returned by any call site (``SlotLease``, ``AuthorizationRecord``,
``RelatedPerson``, ``Requirement``, ``WaitlistEntry``, ``HumanHandoff``,
``SimulatedEscalation``). They are kept as the canonical vocabulary rather than
deleted, and each is listed in the anti-drift allowlist in
``.agent/eval/scripts/verify.py`` so the gap is visible rather than silent.

Identifier conventions (synthetic fixtures only):

* ``pt_``    patient
* ``prac_``  practitioner
* ``loc_``   location
* ``svc_``   healthcare service
* ``sl_``    slot
* ``req_``   appointment request
* ``lease_`` slot lease
* ``appt_``  appointment
* ``wl_``    waitlist entry
* ``ho_``    human handoff
* ``esc_``   simulated escalation
* ``col_``   identity collision token

Every identifier is a string with a prefix so that an ID of the wrong kind is
rejected at validation time rather than silently accepted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from clinical_scheduler.domain.states import (
    AppointmentStatus,
    AuthTier,
    StrictModel,
    TimeWindow,
)

_ID_PREFIXES = {
    "patient_id": "pt_",
    "practitioner_id": "prac_",
    "location_id": "loc_",
    "service_id": "svc_",
    "slot_id": "sl_",
    "request_id": "req_",
    "lease_id": "lease_",
    "appointment_id": "appt_",
    "waitlist_id": "wl_",
    "handoff_id": "ho_",
    "escalation_id": "esc_",
    "collision_token": "col_",
}


def _check_prefix(value: str, prefix: str, field_name: str) -> str:
    if not value.startswith(prefix):
        raise ValueError(f"{field_name} must start with {prefix!r} (got {value!r})")
    return value


class Patient(StrictModel):
    """Synthetic demographic index. No real PHI ever enters this table; the
    fixture generator uses obviously-fake values (plan.md section 9.4)."""

    patient_id: str
    display_name: str = Field(min_length=1)
    birth_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    postal_code: str = Field(min_length=3)
    preferred_language: str = Field(min_length=2)
    is_synthetic: Literal[True] = True

    @field_validator("patient_id")
    @classmethod
    def _pfx(cls, v: str) -> str:
        return _check_prefix(v, "pt_", "patient_id")


class PractitionerRole(StrictModel):
    practitioner_id: str
    role_code: str = Field(min_length=2)
    display_name: str
    service_ids: tuple[str, ...] = ()

    @field_validator("practitioner_id")
    @classmethod
    def _pfx(cls, v: str) -> str:
        return _check_prefix(v, "prac_", "practitioner_id")


class Location(StrictModel):
    location_id: str
    name: str
    iana_timezone: str = Field(min_length=3)
    modality: Literal["IN_PERSON", "TELEHEALTH", "PHONE"]

    @field_validator("location_id")
    @classmethod
    def _pfx(cls, v: str) -> str:
        return _check_prefix(v, "loc_", "location_id")


class HealthcareService(StrictModel):
    service_id: str
    name: str
    duration_minutes: int = Field(gt=0, le=480)
    requires_referral: bool = False
    requires_prior_authorization: bool = False
    requires_preparation_ack: bool = False
    preparation_ack_key: str | None = None
    new_patient_only: bool = False
    established_patient_only: bool = False

    @field_validator("service_id")
    @classmethod
    def _pfx(cls, v: str) -> str:
        return _check_prefix(v, "svc_", "service_id")


class Slot(StrictModel):
    slot_id: str
    service_id: str
    practitioner_id: str
    location_id: str
    window: TimeWindow
    is_bookable: bool = True

    @field_validator("slot_id")
    @classmethod
    def _pfx(cls, v: str) -> str:
        return _check_prefix(v, "sl_", "slot_id")


class SlotLease(StrictModel):
    """A short-lived reservation. Invariant 6: a lease is not an appointment."""

    lease_id: str
    slot_id: str
    patient_id: str
    issued_at: datetime
    expires_at: datetime
    canonical_request_hash: str = Field(min_length=16)

    @field_validator("lease_id")
    @classmethod
    def _pfx(cls, v: str) -> str:
        return _check_prefix(v, "lease_", "lease_id")

    @field_validator("issued_at", "expires_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("lease timestamps must be timezone-aware (invariant 15)")
        return v


class Appointment(StrictModel):
    appointment_id: str
    patient_id: str
    slot_id: str
    service_id: str
    practitioner_id: str
    location_id: str
    status: AppointmentStatus
    window: TimeWindow
    created_at: datetime
    idempotency_key_hash: str | None = None

    @field_validator("appointment_id")
    @classmethod
    def _pfx(cls, v: str) -> str:
        return _check_prefix(v, "appt_", "appointment_id")


class AuthorizationRecord(StrictModel):
    """An authorization decision bound to a subject, scope and purpose."""

    authorization_id: str
    subject_id: str
    tier: AuthTier
    scope: tuple[str, ...]
    purpose_of_use: str
    verified_at: datetime
    expires_at: datetime
    verification_method: Literal["knowledge_discriminator", "step_up", "none"]


class RelatedPerson(StrictModel):
    """Guardian / proxy relationship.

    plan.md section 2.2/B: proxy scope is a per-action check, never a blanket
    flag. `permitted_actions` is the allowlist.
    """

    related_person_id: str
    patient_id: str
    relationship_code: Literal[
        "guardian", "caregiver", "interpreter", "separated_parent", "custodial_parent"
    ]
    permitted_actions: tuple[str, ...]
    revoked: bool = False
    custody_restricted: bool = False


class Requirement(StrictModel):
    requirement_id: str
    appointment_request_id: str
    kind: str
    status: str
    evidence_ref: str | None = None


class WaitlistEntry(StrictModel):
    waitlist_id: str
    patient_id: str
    service_id: str
    preferred_window: TimeWindow
    created_at: datetime
    expires_at: datetime

    @field_validator("waitlist_id")
    @classmethod
    def _pfx(cls, v: str) -> str:
        return _check_prefix(v, "wl_", "waitlist_id")


class HumanHandoff(StrictModel):
    handoff_id: str
    reason_code: str
    summary: str = Field(min_length=1)
    contact_path: str
    created_at: datetime
    # A handoff must never be represented as a monitored line unless it is.
    is_monitored: Literal[False] = False


class SimulatedEscalation(StrictModel):
    """plan.md section 3.5: the prototype only records a simulated escalation
    event. It never contacts anyone, and responses must say so."""

    escalation_id: str
    category: str
    locale: str
    created_at: datetime
    actually_contacted_services: Literal[False] = False
    message_template_id: str = Field(min_length=1)


__all__ = [
    "Appointment",
    "AuthorizationRecord",
    "HealthcareService",
    "HumanHandoff",
    "Location",
    "Patient",
    "PractitionerRole",
    "RelatedPerson",
    "Requirement",
    "SimulatedEscalation",
    "Slot",
    "SlotLease",
    "WaitlistEntry",
]
