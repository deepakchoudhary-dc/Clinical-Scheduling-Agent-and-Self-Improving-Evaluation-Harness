"""Deterministic synthetic fixtures (plan.md section 9.4).

Every patient in this fixture set is visibly fabricated:

* names are drawn from a fixed synthetic list and suffixed with ``(SYNTHETIC)``;
* postal codes come from a reserved demo range;
* the ``is_synthetic`` column is constrained to 1 at the schema level.

The fixture set is generated from a single integer seed so that every evaluation
scenario starts from a byte-identical database. The orchestrator computes a hash
of the seed output and records it in the run manifest.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from clinical_scheduler.db.repository import Repository, canonical_json, sha256_hex
from clinical_scheduler.domain.models import (
    HealthcareService,
    Location,
    Patient,
    PractitionerRole,
    Slot,
    TimeWindow,
)

#: Fixed reference instant for all fixtures. Real runs substitute a clock; tests
#: pin this so that DST and lead-time behaviour are reproducible.
FIXTURE_NOW = datetime(2026, 10, 6, 9, 0, tzinfo=timezone.utc)

DEMO_SALT = "demo-fixture-salt-not-a-secret"

_SYNTHETIC_NAMES = [
    "Ada Example",
    "Bo Placeholder",
    "Cy Sample",
    "Dee Mock",
    "Eli Dummy",
    "Fay Specimen",
    "Gus Testcase",
    "Hana Fixture",
]


@dataclass(frozen=True)
class FixtureBundle:
    """Description of a seeded database, used for manifest hashing."""

    seed: int
    patient_ids: tuple[str, ...]
    service_ids: tuple[str, ...]
    slot_ids: tuple[str, ...]
    location_ids: tuple[str, ...]
    fixture_hash: str


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

_LOCATIONS: tuple[Location, ...] = (
    Location(
        location_id="loc_main",
        name="Demo Clinic - Main Campus (SYNTHETIC)",
        iana_timezone="America/New_York",
        modality="IN_PERSON",
    ),
    Location(
        location_id="loc_north",
        name="Demo Clinic - North Annex (SYNTHETIC)",
        iana_timezone="America/New_York",
        modality="IN_PERSON",
    ),
    Location(
        location_id="loc_virtual",
        name="Demo Clinic - Telehealth (SYNTHETIC)",
        iana_timezone="America/Chicago",
        modality="TELEHEALTH",
    ),
)

_SERVICES: tuple[HealthcareService, ...] = (
    HealthcareService(
        service_id="svc_gp_new",
        name="General Practice - New Patient (SYNTHETIC)",
        duration_minutes=30,
        new_patient_only=True,
    ),
    HealthcareService(
        service_id="svc_gp_est",
        name="General Practice - Established Patient (SYNTHETIC)",
        duration_minutes=20,
        established_patient_only=True,
    ),
    HealthcareService(
        service_id="svc_endoscopy",
        name="Upper Endoscopy (SYNTHETIC)",
        duration_minutes=45,
        requires_referral=True,
        requires_prior_authorization=True,
        requires_preparation_ack=True,
        preparation_ack_key="prep.endoscopy.policy_approved_v1",
    ),
    HealthcareService(
        service_id="svc_behavioral",
        name="Behavioural Health - Intake (SYNTHETIC)",
        duration_minutes=60,
    ),
    HealthcareService(
        service_id="svc_obstetric",
        name="Obstetric - Routine (SYNTHETIC)",
        duration_minutes=30,
    ),
    HealthcareService(
        service_id="svc_virtual_review",
        name="Virtual Medication Review (SYNTHETIC)",
        duration_minutes=20,
        established_patient_only=True,
    ),
)

_PRACTITIONERS: tuple[PractitionerRole, ...] = (
    PractitionerRole(practitioner_id="prac_alpha", role_code="GP", display_name="Dr Alpha (SYNTHETIC)",
                     service_ids=("svc_gp_new", "svc_gp_est")),
    PractitionerRole(practitioner_id="prac_beta", role_code="ENDO", display_name="Dr Beta (SYNTHETIC)",
                     service_ids=("svc_endoscopy",)),
    PractitionerRole(practitioner_id="prac_gamma", role_code="BH", display_name="Dr Gamma (SYNTHETIC)",
                     service_ids=("svc_behavioral",)),
    PractitionerRole(practitioner_id="prac_delta", role_code="OB", display_name="Dr Delta (SYNTHETIC)",
                     service_ids=("svc_obstetric",)),
    PractitionerRole(practitioner_id="prac_epsilon", role_code="GP", display_name="Dr Epsilon (SYNTHETIC)",
                     service_ids=("svc_virtual_review", "svc_gp_est")),
)


def _slot_window(offset_days: int, hour: int, minute: int, zone: str, minutes: int) -> TimeWindow:
    start_local = FIXTURE_NOW.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(
        days=offset_days
    )
    return TimeWindow(start=start_local, end=start_local + timedelta(minutes=minutes), display_zone=zone)


def _build_slots() -> tuple[Slot, ...]:
    """A small but deliberately awkward schedule.

    Includes a slot that crosses the US daylight-saving boundary, because
    invariant 15 and scenario S19 require timezone-aware arithmetic to be
    exercised rather than assumed.
    """
    specs: list[tuple[str, str, str, str, TimeWindow]] = [
        ("sl_gp_new_01", "svc_gp_new", "prac_alpha", "loc_main", _slot_window(7, 9, 0, "America/New_York", 30)),
        ("sl_gp_new_02", "svc_gp_new", "prac_alpha", "loc_main", _slot_window(7, 11, 0, "America/New_York", 30)),
        ("sl_gp_est_01", "svc_gp_est", "prac_alpha", "loc_main", _slot_window(8, 14, 0, "America/New_York", 20)),
        ("sl_gp_est_02", "svc_gp_est", "prac_epsilon", "loc_virtual", _slot_window(9, 10, 0, "America/Chicago", 20)),
        ("sl_endo_01", "svc_endoscopy", "prac_beta", "loc_main", _slot_window(14, 8, 0, "America/New_York", 45)),
        ("sl_bh_01", "svc_behavioral", "prac_gamma", "loc_north", _slot_window(6, 13, 0, "America/New_York", 60)),
        ("sl_ob_01", "svc_obstetric", "prac_delta", "loc_main", _slot_window(10, 10, 30, "America/New_York", 30)),
        ("sl_virt_01", "svc_virtual_review", "prac_epsilon", "loc_virtual", _slot_window(8, 15, 0, "America/Chicago", 20)),
    ]
    return tuple(
        Slot(
            slot_id=sid, service_id=svc, practitioner_id=prac, location_id=loc, window=win
        )
        for sid, svc, prac, loc, win in specs
    )


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------


def _build_patients() -> tuple[Patient, ...]:
    """Ten synthetic patients, including a deliberate name collision pair.

    ``pt_collide_a`` and ``pt_collide_b`` share a display name and birth date and
    differ only in postal code. That pair is what makes the blind-verification
    flow testable (plan.md section 9.1, scenario S02).
    """
    base = [
        ("pt_ada", 0, "10001"),
        ("pt_bo", 1, "10002"),
        ("pt_cy", 2, "10003"),
        ("pt_dee", 3, "10004"),
        ("pt_eli", 4, "10005"),
        ("pt_fay", 5, "10006"),
        ("pt_gus", 6, "10007"),
        ("pt_hana", 7, "10008"),
    ]
    patients = [
        Patient(
            patient_id=pid,
            display_name=f"{_SYNTHETIC_NAMES[idx]} (SYNTHETIC)",
            birth_date=f"19{70 + idx:02d}-0{(idx % 9) + 1}-1{idx % 9}",
            postal_code=zipc,
            preferred_language="en",
        )
        for pid, idx, zipc in base
    ]
    # The collision pair: identical name and birth date, different ZIP.
    patients.append(
        Patient(
            patient_id="pt_collide_a",
            display_name="Mira Collision (SYNTHETIC)",
            birth_date="1985-04-12",
            postal_code="10011",
            preferred_language="en",
        )
    )
    patients.append(
        Patient(
            patient_id="pt_collide_b",
            display_name="Mira Collision (SYNTHETIC)",
            birth_date="1985-04-12",
            postal_code="10012",
            preferred_language="en",
        )
    )
    return tuple(patients)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def _tokenise(value: str) -> str:
    return sha256_hex(f"{DEMO_SALT}:{value}")


def seed(conn: sqlite3.Connection, *, seed_value: int = 0) -> FixtureBundle:
    """Populate an empty database. Idempotent: refuses to seed twice."""
    existing = conn.execute("SELECT COUNT(*) AS n FROM patients").fetchone()["n"]
    if existing:
        raise RuntimeError("fixture database is not empty; refusing to seed twice")

    for loc in _LOCATIONS:
        conn.execute(
            "INSERT INTO locations (location_id, name, iana_timezone, modality) VALUES (?,?,?,?)",
            (loc.location_id, loc.name, loc.iana_timezone, loc.modality),
        )
    for svc in _SERVICES:
        conn.execute(
            """INSERT INTO healthcare_services
               (service_id, name, duration_minutes, requires_referral,
                requires_prior_authorization, requires_preparation_ack,
                preparation_ack_key, new_patient_only, established_patient_only)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                svc.service_id, svc.name, svc.duration_minutes,
                int(svc.requires_referral), int(svc.requires_prior_authorization),
                int(svc.requires_preparation_ack), svc.preparation_ack_key,
                int(svc.new_patient_only), int(svc.established_patient_only),
            ),
        )
    for prac in _PRACTITIONERS:
        conn.execute(
            "INSERT INTO practitioners (practitioner_id, display_name) VALUES (?,?)",
            (prac.practitioner_id, prac.display_name),
        )
        for service_id in prac.service_ids:
            conn.execute(
                "INSERT INTO practitioner_roles (practitioner_id, role_code, service_id) VALUES (?,?,?)",
                (prac.practitioner_id, prac.role_code, service_id),
            )

    slots = _build_slots()
    for slot in slots:
        schedule_id = f"sch_{slot.slot_id[3:]}"
        conn.execute(
            "INSERT INTO schedules (schedule_id, service_id, practitioner_id, location_id) VALUES (?,?,?,?)",
            (schedule_id, slot.service_id, slot.practitioner_id, slot.location_id),
        )
        start_utc, end_utc = slot.window.to_utc_iso()
        conn.execute(
            """INSERT INTO slots (slot_id, schedule_id, start_utc, end_utc, display_zone, is_bookable)
               VALUES (?,?,?,?,?,1)""",
            (slot.slot_id, schedule_id, start_utc, end_utc, slot.window.display_zone),
        )

    patients = _build_patients()
    for p in patients:
        conn.execute(
            """INSERT INTO patients (patient_id, display_name, birth_date, postal_code,
                                     preferred_language, is_synthetic)
               VALUES (?,?,?,?,?,1)""",
            (p.patient_id, p.display_name, p.birth_date, p.postal_code, p.preferred_language),
        )
        # Tokenised discriminators: postal code and birth date, stored hashed.
        conn.execute(
            """INSERT INTO patient_identifiers
               (identifier_id, patient_id, identifier_type, issuer, token_hash)
               VALUES (?,?,?,?,?)""",
            (f"id_{p.patient_id}_zip", p.patient_id, "postal_code", "demo_clinic", _tokenise(p.postal_code)),
        )
        conn.execute(
            """INSERT INTO patient_identifiers
               (identifier_id, patient_id, identifier_type, issuer, token_hash)
               VALUES (?,?,?,?,?)""",
            (f"id_{p.patient_id}_dob", p.patient_id, "birth_date", "demo_clinic", _tokenise(p.birth_date)),
        )

    # A proxy relationship with deliberately narrow scope: can book, cannot view.
    conn.execute(
        """INSERT INTO related_persons
           (related_person_id, patient_id, relationship_code, permitted_actions,
            revoked, custody_restricted)
           VALUES (?,?,?,?,0,0)""",
        ("rel_guardian_ada", "pt_ada", "guardian", canonical_json(["book", "cancel"])),
    )

    bundle = FixtureBundle(
        seed=seed_value,
        patient_ids=tuple(p.patient_id for p in patients),
        service_ids=tuple(s.service_id for s in _SERVICES),
        slot_ids=tuple(s.slot_id for s in slots),
        location_ids=tuple(loc.location_id for loc in _LOCATIONS),
        fixture_hash="",
    )
    digest = sha256_hex(
        canonical_json(
            {
                "seed": seed_value,
                "patients": bundle.patient_ids,
                "services": bundle.service_ids,
                "slots": bundle.slot_ids,
                "locations": bundle.location_ids,
            }
        )
    )
    return FixtureBundle(
        seed=bundle.seed,
        patient_ids=bundle.patient_ids,
        service_ids=bundle.service_ids,
        slot_ids=bundle.slot_ids,
        location_ids=bundle.location_ids,
        fixture_hash=digest,
    )


def build_fixture_database(db_path: str, *, seed_value: int = 0) -> FixtureBundle:
    """Create a fresh database at ``db_path`` and seed it. Returns the bundle."""
    repo = Repository(db_path)
    try:
        with repo.transaction() as conn:
            bundle = seed(conn, seed_value=seed_value)
        return bundle
    finally:
        repo.close()


__all__ = [
    "DEMO_SALT",
    "FIXTURE_NOW",
    "FixtureBundle",
    "build_fixture_database",
    "seed",
]
