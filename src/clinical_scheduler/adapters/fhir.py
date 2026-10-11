"""FHIR R4 mapping (``plan.md`` section 7.3).

Translation between this system's domain models and FHIR R4 resources. **No
network, no credentials, no client** -- this is the seam, not an integration. What
it buys is three things:

1. **A deployable shape.** In a real clinic the scheduler does not own the
   calendar; it reads and writes against the EHR's FHIR API. This module is the
   layer that would sit between the kernel and that API.
2. **A shared vocabulary that exposes modelling gaps.** Forcing every internal
   field into a standard surfaces the ones that have no standard home. That is the
   single most useful output of this file, and it is recorded rather than hidden --
   see ``INTERNAL_FIELDS_WITHOUT_A_STANDARD_HOME`` and ``UNMAPPED_ENTITIES``.
3. **A round-trip property.** ``from_fhir(to_fhir(x)) == x`` for every mapped
   entity, asserted by tests. A mapping that quietly loses a field is worse than no
   mapping, because the loss appears only in production.

**The design decision that makes the round trip lossless.** FHIR R4 cannot express
every distinction this system makes -- it has no "rescheduled" appointment status,
no location timezone, no notion of a proxy's permitted actions. Rather than
flattening those away, each is carried in a **local extension** under
``urn:clinical-scheduler``, and the standard element carries the interoperable
projection:

* a FHIR consumer that ignores the extensions gets a correct, standard, *lossy*
  view -- which is the right behaviour;
* this system round-trips exactly, because it reads its own extension back.

The namespace is deliberately not an HL7 URL. A consumer can tell at a glance that
the field is ours, and nobody can mistake a local extension for a standard one.

**The asymmetry, stated plainly.** The mapping is lossy in **one direction only**.

* ``to_fhir`` is *total*: every domain model maps, with the extra distinctions
  carried in extensions.
* ``from_fhir`` is *exact* for resources this system wrote -- it reads its own
  extensions back -- and **best-effort for anything else**. FHIR R4 cannot carry a
  location timezone, a per-action proxy scope or a rescheduled status, so a foreign
  resource is genuinely missing fields these models require. Rather than guess,
  ``from_fhir`` raises ``MissingLocalDataError`` naming the extension it needs.

That asymmetry is not a defect to be engineered away; it is what "richer internal
model, narrower standard" means, and hiding it would produce plausible domain
objects from incomplete data.

**What this is not.** Not a conformance claim. No profile validation, no
terminology server, no `$validate` -- and the code systems are local, because this
system's synthetic services have no SNOMED or LOINC codes and inventing them would
be a lie dressed as interoperability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from clinical_scheduler.domain.models import (
    Appointment,
    HealthcareService,
    HumanHandoff,
    Location,
    Patient,
    PractitionerRole,
    RelatedPerson,
    Requirement,
    SimulatedEscalation,
    Slot,
)
from clinical_scheduler.domain.states import AppointmentStatus

#: Local namespace for every extension and code system this module emits. Not an
#: HL7 URL: a consumer must be able to tell instantly that the field is ours.
LOCAL_NAMESPACE = "urn:clinical-scheduler"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _system(name: str) -> str:
    return f"{LOCAL_NAMESPACE}/CodeSystem/{name}"


def _extension_url(name: str) -> str:
    return f"{LOCAL_NAMESPACE}/StructureDefinition/{name}"


def _identifier(name: str, value: str) -> dict[str, Any]:
    return {"system": f"{LOCAL_NAMESPACE}/identifier/{name}", "value": value}


def _coding(name: str, code: str, display: str | None = None) -> dict[str, Any]:
    coding: dict[str, Any] = {"system": _system(name), "code": code}
    if display:
        coding["display"] = display
    return coding


def _codeable(name: str, code: str, display: str | None = None) -> dict[str, Any]:
    return {"coding": [_coding(name, code, display)], "text": display or code}


def _extension(name: str, value: Any) -> dict[str, Any]:
    """A local extension. The value type follows the Python type.

    ``bool`` is checked before ``int`` because ``isinstance(True, int)`` is true in
    Python, and a boolean written as ``valueInteger`` would come back as ``1``.
    """
    url = _extension_url(name)
    if isinstance(value, bool):
        return {"url": url, "valueBoolean": value}
    if isinstance(value, int):
        return {"url": url, "valueInteger": value}
    if isinstance(value, (list, tuple)):
        return {"url": url, "valueString": ",".join(str(item) for item in value)}
    return {"url": url, "valueString": str(value)}


def _ext_raw(resource: dict[str, Any], name: str) -> Any:
    url = _extension_url(name)
    for entry in resource.get("extension") or []:
        if entry.get("url") == url:
            for key in ("valueString", "valueInteger", "valueBoolean"):
                if key in entry:
                    return entry[key]
            return None
    return None


class MissingLocalDataError(ValueError):
    """A resource lacks a local extension needed to rebuild a domain model.

    Raised rather than guessed at, and the distinction is the whole point: **FHIR
    R4 is less expressive than these models**, so a resource that did not originate
    here cannot be reconstructed faithfully. Inventing a default would silently
    produce a wrong domain object, which is worse than refusing to produce one.

    This is a limitation of the mapping, not a bug in it, and it is documented as
    such: ``from_fhir`` is reliable for round-tripping resources this system wrote,
    and is a best-effort reader for anything else.
    """


def _ext_str(resource: dict[str, Any], name: str) -> str:
    value = _ext_raw(resource, name)
    if value is None:
        raise MissingLocalDataError(
            f"resource {resource.get('resourceType', '?')}/{resource.get('id', '?')} has no "
            f"local extension {name!r}. FHIR R4 cannot carry this field, so a resource written "
            "by another system cannot be reconstructed without guessing."
        )
    return str(value)


def _ext_str_or(resource: dict[str, Any], name: str, default: str) -> str:
    """Read an optional local extension, for fields where a default is safe.

    Used only for presentation data. A display zone does not change the instant a
    window covers -- the instants are absolute -- so a foreign resource falls back
    to UTC, which is correct for the instants and merely loses a local preference.
    """
    value = _ext_raw(resource, name)
    return default if value is None else str(value)


def _ext_int(resource: dict[str, Any], name: str) -> int:
    return int(_ext_raw(resource, name))


def _ext_bool(resource: dict[str, Any], name: str) -> bool:
    return bool(_ext_raw(resource, name))


def _ext_list(resource: dict[str, Any], name: str) -> tuple[str, ...]:
    """Read a list extension. An absent or empty extension is an empty tuple."""
    raw = _ext_raw(resource, name)
    if raw is None or raw == "":
        return ()
    return tuple(part for part in str(raw).split(",") if part)


def _reference(kind: str, identifier: str) -> dict[str, Any]:
    return {"reference": f"{kind}/{identifier}"}


def _reference_id(reference: dict[str, Any]) -> str:
    """The identifier out of ``{"reference": "Patient/pt_x"}``."""
    text = str(reference.get("reference", ""))
    return text.split("/", 1)[1] if "/" in text else text


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _require(resource: dict[str, Any], key: str) -> Any:
    if key not in resource:
        raise KeyError(f"FHIR resource is missing required element {key!r}")
    return resource[key]


# ---------------------------------------------------------------------------
# Status projections
# ---------------------------------------------------------------------------

#: Internal appointment status -> FHIR R4 ``Appointment.status``.
#:
#: ``RESCHEDULED`` projects to ``booked`` because FHIR R4 has no rescheduled
#: status: a rescheduled appointment *is* a booked one, and the distinction this
#: system keeps is carried in ``appointment-internal-status`` so the round trip
#: stays exact. This is the clearest example of why the extension exists.
APPOINTMENT_STATUS_TO_FHIR: dict[AppointmentStatus, str] = {
    AppointmentStatus.PROPOSED: "proposed",
    AppointmentStatus.BOOKED: "booked",
    AppointmentStatus.CANCELLED: "cancelled",
    AppointmentStatus.RESCHEDULED: "booked",
    AppointmentStatus.NO_SHOW: "noshow",
    AppointmentStatus.COMPLETED: "fulfilled",
}

#: The reverse projection, used only when the local extension is absent -- i.e.
#: when reading a resource written by somebody else.
FHIR_STATUS_TO_APPOINTMENT: dict[str, AppointmentStatus] = {
    "proposed": AppointmentStatus.PROPOSED,
    "pending": AppointmentStatus.PROPOSED,
    "booked": AppointmentStatus.BOOKED,
    "arrived": AppointmentStatus.BOOKED,
    "checked-in": AppointmentStatus.BOOKED,
    "fulfilled": AppointmentStatus.COMPLETED,
    "cancelled": AppointmentStatus.CANCELLED,
    "noshow": AppointmentStatus.NO_SHOW,
}

#: Internal requirement status -> ``ServiceRequest.status``.
REQUIREMENT_STATUS_TO_FHIR: dict[str, str] = {
    "SATISFIED": "completed",
    "PENDING": "active",
    "UNMET": "active",
    "WAIVED": "revoked",
}


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------


def patient_to_fhir(patient: Patient) -> dict[str, Any]:
    return {
        "resourceType": "Patient",
        "id": patient.patient_id,
        "identifier": [_identifier("patient", patient.patient_id)],
        "name": [{"text": patient.display_name}],
        "birthDate": patient.birth_date,
        "address": [{"postalCode": patient.postal_code}],
        "communication": [
            {
                "language": {"coding": [_coding("language", patient.preferred_language)]},
                "preferred": True,
            }
        ],
        # Provenance, not decoration: a consumer must be able to filter synthetic
        # records out before anyone mistakes them for real ones.
        "meta": {"tag": [_coding("synthetic", "true", "Synthetic record")]},
    }


def patient_from_fhir(resource: dict[str, Any]) -> Patient:
    return Patient(
        patient_id=str(_require(resource, "id")),
        display_name=str(resource["name"][0]["text"]),
        birth_date=str(resource["birthDate"]),
        postal_code=str(resource["address"][0]["postalCode"]),
        preferred_language=str(resource["communication"][0]["language"]["coding"][0]["code"]),
    )


# ---------------------------------------------------------------------------
# PractitionerRole
# ---------------------------------------------------------------------------


def practitioner_role_to_fhir(role: PractitionerRole) -> dict[str, Any]:
    return {
        "resourceType": "PractitionerRole",
        "id": role.practitioner_id,
        "identifier": [_identifier("practitioner", role.practitioner_id)],
        "practitioner": {"display": role.display_name},
        "code": [_codeable("practitioner-role", role.role_code)],
        "healthcareService": [
            _reference("HealthcareService", service_id) for service_id in role.service_ids
        ],
    }


def practitioner_role_from_fhir(resource: dict[str, Any]) -> PractitionerRole:
    return PractitionerRole(
        practitioner_id=str(_require(resource, "id")),
        role_code=str(resource["code"][0]["coding"][0]["code"]),
        display_name=str(resource["practitioner"]["display"]),
        service_ids=tuple(
            _reference_id(entry) for entry in resource.get("healthcareService") or []
        ),
    )


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


def location_to_fhir(location: Location) -> dict[str, Any]:
    """``iana_timezone`` has no FHIR R4 element -- it becomes a local extension."""
    return {
        "resourceType": "Location",
        "id": location.location_id,
        "identifier": [_identifier("location", location.location_id)],
        "name": location.name,
        "type": [_codeable("location-modality", location.modality)],
        "extension": [_extension("location-iana-timezone", location.iana_timezone)],
    }


def location_from_fhir(resource: dict[str, Any]) -> Location:
    return Location(
        location_id=str(_require(resource, "id")),
        name=str(resource["name"]),
        iana_timezone=_ext_str(resource, "location-iana-timezone"),
        modality=str(resource["type"][0]["coding"][0]["code"]),
    )


# ---------------------------------------------------------------------------
# HealthcareService
# ---------------------------------------------------------------------------


def healthcare_service_to_fhir(service: HealthcareService) -> dict[str, Any]:
    """Every eligibility flag becomes a local extension.

    FHIR R4's ``HealthcareService.eligibility`` is a free-text CodeableConcept
    without a valueset for "requires a referral" or "requires prior
    authorisation", so a standard coding would be invented. Extensions keep the
    claim honest and the round trip exact.
    """
    extensions = [
        _extension("service-duration-minutes", service.duration_minutes),
        _extension("service-requires-referral", service.requires_referral),
        _extension("service-requires-prior-authorization", service.requires_prior_authorization),
        _extension("service-requires-preparation-ack", service.requires_preparation_ack),
        _extension("service-new-patient-only", service.new_patient_only),
        _extension("service-established-patient-only", service.established_patient_only),
    ]
    if service.preparation_ack_key is not None:
        extensions.append(_extension("service-preparation-ack-key", service.preparation_ack_key))
    return {
        "resourceType": "HealthcareService",
        "id": service.service_id,
        "identifier": [_identifier("service", service.service_id)],
        "name": service.name,
        "extension": extensions,
    }


def healthcare_service_from_fhir(resource: dict[str, Any]) -> HealthcareService:
    key = _ext_raw(resource, "service-preparation-ack-key")
    return HealthcareService(
        service_id=str(_require(resource, "id")),
        name=str(resource["name"]),
        duration_minutes=_ext_int(resource, "service-duration-minutes"),
        requires_referral=_ext_bool(resource, "service-requires-referral"),
        requires_prior_authorization=_ext_bool(
            resource, "service-requires-prior-authorization"
        ),
        requires_preparation_ack=_ext_bool(resource, "service-requires-preparation-ack"),
        preparation_ack_key=None if key is None else str(key),
        new_patient_only=_ext_bool(resource, "service-new-patient-only"),
        established_patient_only=_ext_bool(resource, "service-established-patient-only"),
    )


# ---------------------------------------------------------------------------
# Slot
# ---------------------------------------------------------------------------


def slot_to_fhir(slot: Slot) -> dict[str, Any]:
    """A slot's schedule is derived, not stored.

    FHIR requires ``Slot.schedule`` to point at a ``Schedule`` whose ``actor`` is
    the practitioner. This system models one schedule per practitioner, so the
    reference is derived as ``Schedule/{practitioner_id}``. That assumption is
    stated here because it is the kind of thing an integration discovers late.
    """
    return {
        "resourceType": "Slot",
        "id": slot.slot_id,
        "identifier": [_identifier("slot", slot.slot_id)],
        "serviceType": [_codeable("service", slot.service_id)],
        "schedule": _reference("Schedule", slot.practitioner_id),
        "status": "free" if slot.is_bookable else "busy",
        "start": _iso(slot.window.start),
        "end": _iso(slot.window.end),
        "extension": [
            _extension("slot-location", slot.location_id),
            _extension("slot-display-zone", slot.window.display_zone),
        ],
    }


def slot_from_fhir(resource: dict[str, Any]) -> Slot:
    from clinical_scheduler.domain.states import TimeWindow

    return Slot(
        slot_id=str(_require(resource, "id")),
        service_id=str(resource["serviceType"][0]["coding"][0]["code"]),
        practitioner_id=_reference_id(resource["schedule"]),
        location_id=_ext_str(resource, "slot-location"),
        window=TimeWindow(
            start=datetime.fromisoformat(str(resource["start"])),
            end=datetime.fromisoformat(str(resource["end"])),
            display_zone=_ext_str_or(resource, "slot-display-zone", "UTC"),
        ),
        is_bookable=str(resource["status"]) == "free",
    )


# ---------------------------------------------------------------------------
# Appointment
# ---------------------------------------------------------------------------


def appointment_to_fhir(appointment: Appointment) -> dict[str, Any]:
    """Participants carry the patient, practitioner and location.

    FHIR R4 ``Appointment`` has no location element; the location travels as a
    ``Location`` participant, which is the standard's own answer and not a local
    invention.
    """
    return {
        "resourceType": "Appointment",
        "id": appointment.appointment_id,
        "identifier": [_identifier("appointment", appointment.appointment_id)],
        "status": APPOINTMENT_STATUS_TO_FHIR[appointment.status],
        "serviceType": [_codeable("service", appointment.service_id)],
        "slot": [_reference("Slot", appointment.slot_id)],
        "start": _iso(appointment.window.start),
        "end": _iso(appointment.window.end),
        "created": _iso(appointment.created_at),
        "participant": [
            {"actor": _reference("Patient", appointment.patient_id), "status": "accepted"},
            {
                "actor": _reference("Practitioner", appointment.practitioner_id),
                "status": "accepted",
            },
            {"actor": _reference("Location", appointment.location_id), "status": "accepted"},
        ],
        "extension": [
            # The internal status, always. Without it RESCHEDULED would round-trip
            # as BOOKED, because FHIR R4 has no rescheduled status.
            _extension("appointment-internal-status", appointment.status.value),
            _extension("appointment-display-zone", appointment.window.display_zone),
        ]
        + (
            [_extension("appointment-idempotency-key-hash", appointment.idempotency_key_hash)]
            if appointment.idempotency_key_hash
            else []
        ),
    }


def appointment_from_fhir(resource: dict[str, Any]) -> Appointment:
    from clinical_scheduler.domain.states import TimeWindow

    internal = _ext_raw(resource, "appointment-internal-status")
    if internal is not None:
        status = AppointmentStatus(str(internal))
    else:
        status = FHIR_STATUS_TO_APPOINTMENT.get(
            str(resource["status"]), AppointmentStatus.PROPOSED
        )

    participants = {
        _reference_id(entry["actor"]).split("_", 1)[0]: _reference_id(entry["actor"])
        for entry in resource.get("participant") or []
    }
    created = resource.get("created")
    key_hash = _ext_raw(resource, "appointment-idempotency-key-hash")

    return Appointment(
        appointment_id=str(_require(resource, "id")),
        patient_id=participants["pt"],
        slot_id=_reference_id(resource["slot"][0]),
        service_id=str(resource["serviceType"][0]["coding"][0]["code"]),
        practitioner_id=participants["prac"],
        location_id=participants["loc"],
        status=status,
        window=TimeWindow(
            start=datetime.fromisoformat(str(resource["start"])),
            end=datetime.fromisoformat(str(resource["end"])),
            display_zone=_ext_str_or(resource, "appointment-display-zone", "UTC"),
        ),
        created_at=datetime.fromisoformat(str(created)),
        idempotency_key_hash=None if key_hash is None else str(key_hash),
    )


# ---------------------------------------------------------------------------
# RelatedPerson
# ---------------------------------------------------------------------------


def related_person_to_fhir(related: RelatedPerson) -> dict[str, Any]:
    """``permitted_actions`` has no FHIR element, and it is the security-relevant
    one: plan.md section 2.2/B makes proxy scope a per-action allowlist, never a
    blanket flag. It travels as a local extension, and a consumer that ignores it
    must not infer any proxy permission at all.
    """
    return {
        "resourceType": "RelatedPerson",
        "id": related.related_person_id,
        "identifier": [_identifier("related-person", related.related_person_id)],
        "patient": _reference("Patient", related.patient_id),
        "relationship": [_codeable("related-person-relationship", related.relationship_code)],
        "active": not related.revoked,
        "extension": [
            _extension("related-person-permitted-actions", list(related.permitted_actions)),
            _extension("related-person-custody-restricted", related.custody_restricted),
        ],
    }


def related_person_from_fhir(resource: dict[str, Any]) -> RelatedPerson:
    return RelatedPerson(
        related_person_id=str(_require(resource, "id")),
        patient_id=_reference_id(resource["patient"]),
        relationship_code=str(resource["relationship"][0]["coding"][0]["code"]),
        permitted_actions=_ext_list(resource, "related-person-permitted-actions"),
        revoked=not bool(resource.get("active", True)),
        custody_restricted=_ext_bool(resource, "related-person-custody-restricted"),
    )


# ---------------------------------------------------------------------------
# Requirement -> ServiceRequest
# ---------------------------------------------------------------------------


def requirement_to_fhir(requirement: Requirement) -> dict[str, Any]:
    """A prerequisite is a ``ServiceRequest`` -- with a documented mismatch.

    ``Requirement`` points at an ``appointment_request_id``, and **FHIR R4 has no
    ``AppointmentRequest`` resource**: the request concept is ``ServiceRequest``.
    The local reference is carried in ``basedOn`` with a display string that says
    so, rather than being silently dropped or dressed up as a real reference.
    """
    return {
        "resourceType": "ServiceRequest",
        "id": requirement.requirement_id,
        "identifier": [_identifier("requirement", requirement.requirement_id)],
        "status": REQUIREMENT_STATUS_TO_FHIR.get(requirement.status, "active"),
        "intent": "order",
        "code": _codeable("requirement-kind", requirement.kind),
        "basedOn": [
            {
                "reference": f"AppointmentRequest/{requirement.appointment_request_id}",
                "display": "local appointment request; FHIR R4 has no AppointmentRequest",
            }
        ],
        "extension": [
            _extension("requirement-internal-status", requirement.status),
        ]
        + (
            [_extension("requirement-evidence-ref", requirement.evidence_ref)]
            if requirement.evidence_ref
            else []
        ),
    }


def requirement_from_fhir(resource: dict[str, Any]) -> Requirement:
    internal = _ext_raw(resource, "requirement-internal-status")
    evidence = _ext_raw(resource, "requirement-evidence-ref")
    return Requirement(
        requirement_id=str(_require(resource, "id")),
        appointment_request_id=_reference_id(resource["basedOn"][0]),
        kind=str(resource["code"]["coding"][0]["code"]),
        status=str(internal) if internal is not None else str(resource["status"]).upper(),
        evidence_ref=None if evidence is None else str(evidence),
    )


# ---------------------------------------------------------------------------
# HumanHandoff -> Task
# ---------------------------------------------------------------------------


def human_handoff_to_fhir(handoff: HumanHandoff) -> dict[str, Any]:
    """A handoff is a ``Task`` for a human to pick up.

    ``is_monitored`` is ``Literal[False]`` in the domain model and is carried
    explicitly: a handoff must never be *represented* as a monitored line unless
    it is one, and a consumer reading only ``Task.status`` would have no way to
    know. That is a safety property, so it is explicit rather than implied.
    """
    return {
        "resourceType": "Task",
        "id": handoff.handoff_id,
        "identifier": [_identifier("handoff", handoff.handoff_id)],
        "status": "requested",
        "intent": "order",
        "description": handoff.summary,
        "authoredOn": _iso(handoff.created_at),
        "extension": [
            _extension("handoff-reason-code", handoff.reason_code),
            _extension("handoff-contact-path", handoff.contact_path),
            _extension("handoff-is-monitored", handoff.is_monitored),
        ],
    }


def human_handoff_from_fhir(resource: dict[str, Any]) -> HumanHandoff:
    return HumanHandoff(
        handoff_id=str(_require(resource, "id")),
        reason_code=_ext_str(resource, "handoff-reason-code"),
        summary=str(resource["description"]),
        contact_path=_ext_str(resource, "handoff-contact-path"),
        created_at=datetime.fromisoformat(str(resource["authoredOn"])),
    )


# ---------------------------------------------------------------------------
# SimulatedEscalation -> DetectedIssue
# ---------------------------------------------------------------------------


def escalation_to_fhir(escalation: SimulatedEscalation) -> dict[str, Any]:
    """An escalation is a ``DetectedIssue`` -- something the system flagged.

    ``actually_contacted_services`` is ``Literal[False]`` and travels explicitly.
    The prototype only *records* an escalation; a consumer that assumed a
    ``DetectedIssue`` implied a notification would be making a false assumption
    about patient safety.
    """
    return {
        "resourceType": "DetectedIssue",
        "id": escalation.escalation_id,
        "identifier": [_identifier("escalation", escalation.escalation_id)],
        "status": "final",
        "code": _codeable("escalation-category", escalation.category),
        "identifiedDateTime": _iso(escalation.created_at),
        "extension": [
            _extension("escalation-locale", escalation.locale),
            _extension("escalation-message-template", escalation.message_template_id),
            _extension(
                "escalation-actually-contacted-services",
                escalation.actually_contacted_services,
            ),
        ],
    }


def escalation_from_fhir(resource: dict[str, Any]) -> SimulatedEscalation:
    return SimulatedEscalation(
        escalation_id=str(_require(resource, "id")),
        category=str(resource["code"]["coding"][0]["code"]),
        locale=_ext_str(resource, "escalation-locale"),
        created_at=datetime.fromisoformat(str(resource["identifiedDateTime"])),
        message_template_id=_ext_str(resource, "escalation-message-template"),
    )


# ---------------------------------------------------------------------------
# Coverage registries -- the anti-silence guarantee
# ---------------------------------------------------------------------------

#: Every domain entity that has a FHIR projection, and the resource it becomes.
RESOURCE_MAP: dict[str, str] = {
    "Patient": "Patient",
    "PractitionerRole": "PractitionerRole",
    "Location": "Location",
    "HealthcareService": "HealthcareService",
    "Slot": "Slot",
    "Appointment": "Appointment",
    "RelatedPerson": "RelatedPerson",
    "Requirement": "ServiceRequest",
    "HumanHandoff": "Task",
    "SimulatedEscalation": "DetectedIssue",
}

#: Domain entities with **no** FHIR counterpart, each with the reason. Listed
#: rather than omitted, because an entity that is silently absent from an
#: interoperability layer is the kind of gap that surfaces during go-live.
UNMAPPED_ENTITIES: dict[str, str] = {
    "SlotLease": (
        "no FHIR resource exists for a hold. FHIR models a held slot as "
        "Slot.status = busy-tentative, so the lease is a *state of* a Slot rather "
        "than an entity. Emitting one would invent a resource."
    ),
    "AuthorizationRecord": (
        "an authorization decision is a security artefact of this system's "
        "capability broker, not clinical data. FHIR has Consent and "
        "Provenance for adjacent concerns, but neither is a faithful projection of "
        "a server-minted, expiring capability -- mapping it would imply the "
        "authorization is portable, which it is not."
    ),
    "WaitlistEntry": (
        "FHIR R4 has no waitlist resource. The nearest standard pattern is a Task "
        "or a Slot with a provisional status, and neither preserves the preferred "
        "window. Recorded as a gap rather than approximated."
    ),
}

#: Internal fields that no FHIR R4 element can hold, and where they went instead.
#: This is the output of the mapping exercise that is worth the most: each row is
#: a modelling gap an integration would otherwise discover late.
INTERNAL_FIELDS_WITHOUT_A_STANDARD_HOME: dict[str, dict[str, str]] = {
    "Patient": {},
    "PractitionerRole": {},
    "Location": {
        "iana_timezone": "local extension location-iana-timezone; FHIR R4 Location has no timezone",
        "modality": "local code system location-modality; no R4 valueset for in-person/telehealth",
    },
    "HealthcareService": {
        "duration_minutes": "local extension; R4 puts duration on Slot/Schedule, not the service",
        "requires_referral": "local extension; R4 eligibility has no referral code",
        "requires_prior_authorization": "local extension; no R4 element",
        "requires_preparation_ack": "local extension; no R4 element",
        "preparation_ack_key": "local extension; no R4 element",
        "new_patient_only": "local extension; no R4 valueset",
        "established_patient_only": "local extension; no R4 valueset",
    },
    "Slot": {
        "location_id": "local extension slot-location; R4 Slot has no location element",
        "window.display_zone": "local extension slot-display-zone; R4 has no display zone",
    },
    "Appointment": {
        "status": "standard Appointment.status, plus a local extension for the internal value",
        "window.display_zone": "local extension appointment-display-zone",
        "idempotency_key_hash": "local extension; an internal idempotency artefact, not an identifier",
    },
    "RelatedPerson": {
        "permitted_actions": "local extension; R4 has no per-action proxy scope",
        "custody_restricted": "local extension; no R4 element",
    },
    "Requirement": {
        "appointment_request_id": "basedOn, with a display noting R4 has no AppointmentRequest",
        "status": "standard ServiceRequest.status, plus a local extension for the internal value",
        "evidence_ref": "local extension; R4 would use supportInfo, which is a Reference not a string",
    },
    "HumanHandoff": {
        "is_monitored": "local extension; carried explicitly because it is a safety property",
        "reason_code": "local extension; R4 Task has no reason code without an invented valueset",
    },
    "SimulatedEscalation": {
        "locale": "local extension; R4 DetectedIssue has no locale",
        "message_template_id": "local extension; internal template identifier",
        "actually_contacted_services": "local extension; carried explicitly, it is a safety property",
    },
}


def fhir_coverage_report() -> str:
    """A human-readable summary of what maps and what does not."""
    lines = ["entities with a FHIR projection"]
    for entity, resource in sorted(RESOURCE_MAP.items()):
        gaps = len(INTERNAL_FIELDS_WITHOUT_A_STANDARD_HOME.get(entity, {}))
        lines.append(f"  {entity:22} -> {resource:18} {gaps} field(s) via local extension")
    lines.append("")
    lines.append("entities with no FHIR counterpart")
    for entity, reason in sorted(UNMAPPED_ENTITIES.items()):
        lines.append(f"  {entity}: {reason}")
    return "\n".join(lines)


def collection_bundle(resources: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Wrap resources in a FHIR ``collection`` Bundle, for inspection or transfer."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": resource} for resource in resources],
    }


__all__ = [
    "APPOINTMENT_STATUS_TO_FHIR",
    "FHIR_STATUS_TO_APPOINTMENT",
    "INTERNAL_FIELDS_WITHOUT_A_STANDARD_HOME",
    "LOCAL_NAMESPACE",
    "MissingLocalDataError",
    "REQUIREMENT_STATUS_TO_FHIR",
    "RESOURCE_MAP",
    "UNMAPPED_ENTITIES",
    "appointment_from_fhir",
    "appointment_to_fhir",
    "collection_bundle",
    "escalation_from_fhir",
    "escalation_to_fhir",
    "fhir_coverage_report",
    "healthcare_service_from_fhir",
    "healthcare_service_to_fhir",
    "human_handoff_from_fhir",
    "human_handoff_to_fhir",
    "location_from_fhir",
    "location_to_fhir",
    "patient_from_fhir",
    "patient_to_fhir",
    "practitioner_role_from_fhir",
    "practitioner_role_to_fhir",
    "related_person_from_fhir",
    "related_person_to_fhir",
    "requirement_from_fhir",
    "requirement_to_fhir",
    "slot_from_fhir",
    "slot_to_fhir",
]
