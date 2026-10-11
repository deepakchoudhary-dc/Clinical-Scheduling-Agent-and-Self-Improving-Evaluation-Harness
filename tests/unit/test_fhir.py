"""FHIR R4 mapping: the round trip, the standard placements, and the gaps.

Three properties are pinned here, and the third is the one that matters most.

1. **Round trip.** ``from_fhir(to_fhir(x)) == x`` for every mapped entity. A
   mapping that quietly loses a field is worse than no mapping, because the loss
   surfaces in production rather than in review.
2. **Standard placement.** The interoperable projection must actually be in the
   standard element -- ``Slot.status``, ``Appointment.status``, ``Patient.meta.tag``
   -- not stashed entirely in an extension. Otherwise a FHIR consumer would see
   nothing it recognises, which defeats the purpose.
3. **Completeness.** Every declared domain entity is either mapped or recorded in
   ``UNMAPPED_ENTITIES`` with a reason. An entity silently absent from an
   interoperability layer is the gap that appears during go-live, so the registry
   is asserted rather than trusted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from clinical_scheduler.adapters import fhir
from clinical_scheduler.domain import models
from clinical_scheduler.domain.states import AppointmentStatus, TimeWindow

AWARE = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
WINDOW = TimeWindow(start=AWARE, end=AWARE.replace(hour=13), display_zone="America/New_York")


def _entities() -> dict[str, tuple[Any, Any, Any]]:
    """name -> (instance, to_fhir, from_fhir)."""
    return {
        "Patient": (
            models.Patient(
                patient_id="pt_1",
                display_name="Ada (SYNTHETIC)",
                birth_date="1970-01-01",
                postal_code="M5V",
                preferred_language="en",
            ),
            fhir.patient_to_fhir,
            fhir.patient_from_fhir,
        ),
        "PractitionerRole": (
            models.PractitionerRole(
                practitioner_id="prac_1",
                role_code="GP",
                display_name="Dr Test (SYNTHETIC)",
                service_ids=("svc_a", "svc_b"),
            ),
            fhir.practitioner_role_to_fhir,
            fhir.practitioner_role_from_fhir,
        ),
        "Location": (
            models.Location(
                location_id="loc_1",
                name="Demo Clinic (SYNTHETIC)",
                iana_timezone="America/New_York",
                modality="IN_PERSON",
            ),
            fhir.location_to_fhir,
            fhir.location_from_fhir,
        ),
        "HealthcareService": (
            models.HealthcareService(
                service_id="svc_1",
                name="GP new patient (SYNTHETIC)",
                duration_minutes=20,
                requires_referral=True,
                requires_prior_authorization=False,
                requires_preparation_ack=True,
                preparation_ack_key="ack_1",
                new_patient_only=True,
                established_patient_only=False,
            ),
            fhir.healthcare_service_to_fhir,
            fhir.healthcare_service_from_fhir,
        ),
        "Slot": (
            models.Slot(
                slot_id="sl_1",
                service_id="svc_1",
                practitioner_id="prac_1",
                location_id="loc_1",
                window=WINDOW,
                is_bookable=False,
            ),
            fhir.slot_to_fhir,
            fhir.slot_from_fhir,
        ),
        "Appointment": (
            models.Appointment(
                appointment_id="appt_1",
                patient_id="pt_1",
                slot_id="sl_1",
                service_id="svc_1",
                practitioner_id="prac_1",
                location_id="loc_1",
                status=AppointmentStatus.RESCHEDULED,
                window=WINDOW,
                created_at=AWARE,
                idempotency_key_hash="k" * 64,
            ),
            fhir.appointment_to_fhir,
            fhir.appointment_from_fhir,
        ),
        "RelatedPerson": (
            models.RelatedPerson(
                related_person_id="rp_1",
                patient_id="pt_1",
                relationship_code="guardian",
                permitted_actions=("book", "cancel"),
                revoked=False,
                custody_restricted=True,
            ),
            fhir.related_person_to_fhir,
            fhir.related_person_from_fhir,
        ),
        "Requirement": (
            models.Requirement(
                requirement_id="req_1",
                appointment_request_id="areq_1",
                kind="REFERRAL",
                status="PENDING",
                evidence_ref="doc_1",
            ),
            fhir.requirement_to_fhir,
            fhir.requirement_from_fhir,
        ),
        "HumanHandoff": (
            models.HumanHandoff(
                handoff_id="ho_1",
                reason_code="URGENT_SAFETY_ROUTE",
                summary="Safety route triggered during scheduling.",
                contact_path="phone",
                created_at=AWARE,
            ),
            fhir.human_handoff_to_fhir,
            fhir.human_handoff_from_fhir,
        ),
        "SimulatedEscalation": (
            models.SimulatedEscalation(
                escalation_id="esc_1",
                category="CARDIAC",
                locale="en",
                created_at=AWARE,
                message_template_id="tmpl_1",
            ),
            fhir.escalation_to_fhir,
            fhir.escalation_from_fhir,
        ),
    }


@pytest.mark.parametrize("name", sorted(_entities()))
def test_round_trip_is_lossless(name: str) -> None:
    entity, to_fhir, from_fhir = _entities()[name]
    resource = to_fhir(entity)
    assert from_fhir(resource) == entity, f"{name} did not survive the round trip"


@pytest.mark.parametrize("name", sorted(_entities()))
def test_the_resource_type_matches_the_registry(name: str) -> None:
    entity, to_fhir, _from_fhir = _entities()[name]
    assert to_fhir(entity)["resourceType"] == fhir.RESOURCE_MAP[name]


@pytest.mark.parametrize("name", sorted(_entities()))
def test_every_extension_is_in_the_local_namespace(name: str) -> None:
    """No extension may masquerade as a standard one."""
    entity, to_fhir, _from_fhir = _entities()[name]
    for extension in to_fhir(entity).get("extension") or []:
        assert extension["url"].startswith(fhir.LOCAL_NAMESPACE + "/"), extension["url"]


# -- standard placements ------------------------------------------------------


def test_the_interoperable_projection_lands_in_the_standard_element() -> None:
    """Extensions carry the extras; the standard element carries the meaning.

    If everything were stashed in extensions, a FHIR consumer would see a resource
    it could not read, which would defeat the point of mapping at all.
    """
    slot = fhir.slot_to_fhir(_entities()["Slot"][0])
    assert slot["status"] == "busy"  # is_bookable=False
    assert slot["start"].startswith("2026-10-08T12:00:00")
    assert slot["serviceType"][0]["coding"][0]["code"] == "svc_1"
    assert slot["schedule"]["reference"] == "Schedule/prac_1"

    appointment = fhir.appointment_to_fhir(_entities()["Appointment"][0])
    assert appointment["status"] == "booked"  # RESCHEDULED projects to booked
    assert appointment["slot"][0]["reference"] == "Slot/sl_1"

    patient = fhir.patient_to_fhir(_entities()["Patient"][0])
    assert patient["birthDate"] == "1970-01-01"
    assert patient["meta"]["tag"][0]["code"] == "true"  # synthetic provenance


def test_a_free_slot_projects_to_status_free() -> None:
    slot = models.Slot(
        slot_id="sl_2",
        service_id="svc_1",
        practitioner_id="prac_1",
        location_id="loc_1",
        window=WINDOW,
        is_bookable=True,
    )
    assert fhir.slot_to_fhir(slot)["status"] == "free"
    assert fhir.slot_from_fhir(fhir.slot_to_fhir(slot)).is_bookable is True


def _without_extension(resource: dict[str, Any], suffix: str) -> dict[str, Any]:
    """A copy with one named local extension removed.

    Removing *one* extension models a foreign resource that carries the standard
    elements but not our local ones. Removing *all* of them would also strip fields
    the models genuinely require, which is a different (and separately tested)
    situation.
    """
    copy = {**resource}
    copy["extension"] = [
        entry for entry in resource.get("extension") or [] if not entry["url"].endswith(suffix)
    ]
    return copy


def test_rescheduled_survives_only_because_of_the_local_extension() -> None:
    """The clearest justification for the extension mechanism.

    FHIR R4 has no rescheduled status, so the standard element says ``booked``.
    The internal value travels in an extension, and the round trip is exact --
    while a consumer that ignores extensions still sees a correct, lossy view.
    """
    appointment = _entities()["Appointment"][0]
    resource = fhir.appointment_to_fhir(appointment)
    assert resource["status"] == "booked"
    assert fhir.appointment_from_fhir(resource).status is AppointmentStatus.RESCHEDULED

    # Without our extension, the fallback is the standard, lossy projection.
    stripped = _without_extension(resource, "appointment-internal-status")
    assert fhir.appointment_from_fhir(stripped).status is AppointmentStatus.BOOKED


def test_an_unknown_fhir_status_falls_back_rather_than_crashing() -> None:
    """Reading a resource written by somebody else must not raise."""
    resource = _without_extension(
        fhir.appointment_to_fhir(_entities()["Appointment"][0]),
        "appointment-internal-status",
    )
    resource["status"] = "arrived"
    assert fhir.appointment_from_fhir(resource).status is AppointmentStatus.BOOKED


def test_the_display_zone_falls_back_to_utc_for_a_foreign_resource() -> None:
    """A safe default, because the instants are absolute and the zone is not.

    The window's start and end do not change with the display zone, so UTC is
    correct for what the window covers and merely loses a local preference.
    """
    resource = _without_extension(
        fhir.slot_to_fhir(_entities()["Slot"][0]), "slot-display-zone"
    )
    assert fhir.slot_from_fhir(resource).window.display_zone == "UTC"


def test_a_foreign_resource_fails_loudly_rather_than_guessing() -> None:
    """The asymmetry, made explicit.

    FHIR R4 cannot carry every field these models require, so a resource this
    system did not write cannot be reconstructed faithfully. Refusing is the
    correct behaviour; inventing a default would produce a plausible-looking wrong
    domain object, which is worse than an error.
    """
    resource = _without_extension(
        fhir.slot_to_fhir(_entities()["Slot"][0]), "slot-location"
    )
    with pytest.raises(fhir.MissingLocalDataError) as excinfo:
        fhir.slot_from_fhir(resource)
    assert "slot-location" in str(excinfo.value)
    assert "without guessing" in str(excinfo.value)


# -- the gap registries -------------------------------------------------------


def test_every_domain_entity_is_mapped_or_recorded_as_a_gap() -> None:
    """The anti-silence guarantee. An entity absent from both registries fails."""
    declared = set(models.__all__)
    accounted = set(fhir.RESOURCE_MAP) | set(fhir.UNMAPPED_ENTITIES)
    assert declared == accounted, (
        "domain entities missing from the FHIR coverage registries: "
        f"{sorted(declared - accounted)}; listed but no longer declared: "
        f"{sorted(accounted - declared)}"
    )


def test_every_unmapped_entity_gives_a_reason() -> None:
    for entity, reason in fhir.UNMAPPED_ENTITIES.items():
        assert len(reason) > 40, f"{entity} is unmapped without a real reason"


def test_the_gap_registry_names_fields_that_exist() -> None:
    """A gap entry naming a field that no longer exists is a stale claim."""
    for entity, fields in fhir.INTERNAL_FIELDS_WITHOUT_A_STANDARD_HOME.items():
        model = getattr(models, entity)
        declared = set(model.model_fields)
        for path in fields:
            root = path.split(".", 1)[0]
            assert root in declared, f"{entity}.{path} is not a field of the model"


def test_the_gap_registry_covers_only_mapped_entities() -> None:
    assert set(fhir.INTERNAL_FIELDS_WITHOUT_A_STANDARD_HOME) <= set(fhir.RESOURCE_MAP)


def test_the_coverage_report_names_both_halves() -> None:
    report = fhir.fhir_coverage_report()
    assert "entities with a FHIR projection" in report
    assert "entities with no FHIR counterpart" in report
    for entity in fhir.UNMAPPED_ENTITIES:
        assert entity in report


def test_a_collection_bundle_wraps_resources() -> None:
    bundle = fhir.collection_bundle([fhir.patient_to_fhir(_entities()["Patient"][0])])
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert bundle["entry"][0]["resource"]["resourceType"] == "Patient"


# -- the safety-carrying extensions -------------------------------------------


def test_a_handoff_never_claims_to_be_monitored() -> None:
    """A safety property, carried explicitly rather than implied by Task.status."""
    resource = fhir.human_handoff_to_fhir(_entities()["HumanHandoff"][0])
    monitored = [
        e for e in resource["extension"] if e["url"].endswith("handoff-is-monitored")
    ]
    assert monitored and monitored[0]["valueBoolean"] is False


def test_an_escalation_never_claims_services_were_contacted() -> None:
    resource = fhir.escalation_to_fhir(_entities()["SimulatedEscalation"][0])
    contacted = [
        e
        for e in resource["extension"]
        if e["url"].endswith("escalation-actually-contacted-services")
    ]
    assert contacted and contacted[0]["valueBoolean"] is False


def test_proxy_scope_travels_with_the_related_person() -> None:
    """plan.md 2.2/B: proxy scope is a per-action allowlist, never a blanket flag."""
    resource = fhir.related_person_to_fhir(_entities()["RelatedPerson"][0])
    actions = [
        e for e in resource["extension"] if e["url"].endswith("permitted-actions")
    ]
    assert actions and actions[0]["valueString"] == "book,cancel"
    assert fhir.related_person_from_fhir(resource).permitted_actions == ("book", "cancel")
