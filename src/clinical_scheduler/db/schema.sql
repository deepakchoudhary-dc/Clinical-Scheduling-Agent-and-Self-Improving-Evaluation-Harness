-- Clinical Scheduler schema (plan.md section 7).
--
-- Design rules encoded here rather than in application code, because the
-- database is the last line of defence and cannot be bypassed:
--
--   * foreign keys are ON and enforced;
--   * one active booking per slot (partial unique index);
--   * lease uniqueness per slot while live (partial unique index);
--   * idempotency key uniqueness plus stored canonical request hash;
--   * audit tables have no ON DELETE CASCADE, ever;
--   * appointment status transitions are validated by trigger;
--   * UTC storage with an IANA zone carried alongside for display.
--
-- SQLite is used for the demo. Nothing here depends on a SQLite-only feature in
-- a way that would block a PostgreSQL port except the partial unique indexes,
-- which PostgreSQL also supports.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Identifier allocation
-- ---------------------------------------------------------------------------
--
-- Human-readable identifiers (lease_000001, appt_000042, ...) are allocated
-- from here rather than from a counter in the process. A process-local counter
-- collides the moment a second kernel touches the same database -- which
-- happens on a restart and on every additional worker -- and the collision
-- surfaces as an unhandled UNIQUE violation on a perfectly legitimate booking.
--
-- Allocation happens inside the caller's BEGIN IMMEDIATE transaction, so the
-- read-modify-write is serialised by the same lock that protects the booking
-- itself. This is why the table exists instead of a SQLite AUTOINCREMENT: the
-- identifiers are textual and shared across several tables, so there is no
-- single rowid to hang a sequence on.

CREATE TABLE IF NOT EXISTS id_sequences (
    prefix    TEXT PRIMARY KEY,
    next_val  INTEGER NOT NULL CHECK (next_val >= 1)
);

-- ---------------------------------------------------------------------------
-- Reference / catalogue
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS locations (
    location_id      TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    iana_timezone    TEXT NOT NULL,
    modality         TEXT NOT NULL CHECK (modality IN ('IN_PERSON','TELEHEALTH','PHONE'))
);

CREATE TABLE IF NOT EXISTS healthcare_services (
    service_id                  TEXT PRIMARY KEY,
    name                        TEXT NOT NULL,
    duration_minutes            INTEGER NOT NULL CHECK (duration_minutes > 0 AND duration_minutes <= 480),
    requires_referral           INTEGER NOT NULL DEFAULT 0 CHECK (requires_referral IN (0,1)),
    requires_prior_authorization INTEGER NOT NULL DEFAULT 0 CHECK (requires_prior_authorization IN (0,1)),
    requires_preparation_ack    INTEGER NOT NULL DEFAULT 0 CHECK (requires_preparation_ack IN (0,1)),
    preparation_ack_key         TEXT,
    new_patient_only            INTEGER NOT NULL DEFAULT 0 CHECK (new_patient_only IN (0,1)),
    established_patient_only    INTEGER NOT NULL DEFAULT 0 CHECK (established_patient_only IN (0,1)),
    CHECK (NOT (new_patient_only = 1 AND established_patient_only = 1))
);

CREATE TABLE IF NOT EXISTS practitioners (
    practitioner_id  TEXT PRIMARY KEY,
    display_name     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS practitioner_roles (
    practitioner_id  TEXT NOT NULL REFERENCES practitioners(practitioner_id) ON DELETE RESTRICT,
    role_code        TEXT NOT NULL,
    service_id       TEXT NOT NULL REFERENCES healthcare_services(service_id) ON DELETE RESTRICT,
    PRIMARY KEY (practitioner_id, role_code, service_id)
);

-- ---------------------------------------------------------------------------
-- Identity
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS patients (
    patient_id          TEXT PRIMARY KEY,
    display_name        TEXT NOT NULL,
    birth_date          TEXT NOT NULL,
    postal_code         TEXT NOT NULL,
    preferred_language  TEXT NOT NULL,
    is_synthetic        INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic = 1)
);

-- Identifiers are tokenised: the raw value never appears in this table. The
-- salt lives in configuration, not in the database, so a database dump alone
-- cannot be used to confirm a guessed identifier value.
CREATE TABLE IF NOT EXISTS patient_identifiers (
    identifier_id    TEXT PRIMARY KEY,
    patient_id       TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE RESTRICT,
    identifier_type  TEXT NOT NULL CHECK (identifier_type IN ('postal_code','birth_date','phone_last4','mrn_token')),
    issuer           TEXT NOT NULL DEFAULT 'demo_clinic',
    token_hash       TEXT NOT NULL,
    UNIQUE (patient_id, identifier_type, issuer, token_hash)
);

CREATE INDEX IF NOT EXISTS idx_identifier_lookup
    ON patient_identifiers (identifier_type, issuer, token_hash);

CREATE TABLE IF NOT EXISTS related_persons (
    related_person_id  TEXT PRIMARY KEY,
    patient_id         TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE RESTRICT,
    relationship_code  TEXT NOT NULL CHECK (relationship_code IN
                          ('guardian','caregiver','interpreter','separated_parent','custodial_parent')),
    permitted_actions  TEXT NOT NULL,          -- JSON array of action names
    revoked            INTEGER NOT NULL DEFAULT 0 CHECK (revoked IN (0,1)),
    custody_restricted INTEGER NOT NULL DEFAULT 0 CHECK (custody_restricted IN (0,1))
);

CREATE TABLE IF NOT EXISTS authorizations (
    authorization_id    TEXT PRIMARY KEY,
    subject_id          TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE RESTRICT,
    tier                INTEGER NOT NULL CHECK (tier IN (0,1,2,3)),
    scope               TEXT NOT NULL,          -- JSON array
    purpose_of_use      TEXT NOT NULL,
    verification_method TEXT NOT NULL CHECK (verification_method IN
                          ('knowledge_discriminator','step_up','none')),
    verified_at         TEXT NOT NULL,          -- UTC ISO-8601
    expires_at          TEXT NOT NULL           -- UTC ISO-8601
);

CREATE TABLE IF NOT EXISTS identity_collision_tokens (
    collision_token     TEXT PRIMARY KEY,
    candidate_count     INTEGER NOT NULL CHECK (candidate_count >= 0),
    candidate_ids       TEXT NOT NULL,          -- JSON array, never disclosed
    allowed_discriminators TEXT NOT NULL,       -- JSON array
    attempts_remaining  INTEGER NOT NULL CHECK (attempts_remaining >= 0),
    created_at          TEXT NOT NULL,
    expires_at          TEXT NOT NULL,
    consumed            INTEGER NOT NULL DEFAULT 0 CHECK (consumed IN (0,1))
);

-- Redacted attempt metadata only (plan.md section 9.1).
CREATE TABLE IF NOT EXISTS collision_attempts (
    attempt_id       TEXT PRIMARY KEY,
    collision_token  TEXT NOT NULL,
    discriminator_type TEXT NOT NULL,
    matched          INTEGER NOT NULL CHECK (matched IN (0,1)),
    attempt_index    INTEGER NOT NULL,
    created_at       TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Scheduling
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schedules (
    schedule_id      TEXT PRIMARY KEY,
    service_id       TEXT NOT NULL REFERENCES healthcare_services(service_id) ON DELETE RESTRICT,
    practitioner_id  TEXT NOT NULL REFERENCES practitioners(practitioner_id) ON DELETE RESTRICT,
    location_id      TEXT NOT NULL REFERENCES locations(location_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS slots (
    slot_id        TEXT PRIMARY KEY,
    schedule_id    TEXT NOT NULL REFERENCES schedules(schedule_id) ON DELETE RESTRICT,
    start_utc      TEXT NOT NULL,
    end_utc        TEXT NOT NULL,
    display_zone   TEXT NOT NULL,
    is_bookable    INTEGER NOT NULL DEFAULT 1 CHECK (is_bookable IN (0,1)),
    CHECK (end_utc > start_utc)
);

CREATE INDEX IF NOT EXISTS idx_slots_window ON slots (start_utc, end_utc);

-- A slot may be consumed only once. This partial unique index is the structural
-- guarantee behind "one active booking per slot".
CREATE TABLE IF NOT EXISTS slot_consumption (
    slot_id          TEXT PRIMARY KEY REFERENCES slots(slot_id) ON DELETE RESTRICT,
    consumed_by      TEXT NOT NULL,            -- appointment_id
    consumed_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appointment_requests (
    request_id       TEXT PRIMARY KEY,
    patient_id       TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE RESTRICT,
    service_id       TEXT NOT NULL REFERENCES healthcare_services(service_id) ON DELETE RESTRICT,
    requested_window_start_utc TEXT,
    requested_window_end_utc   TEXT,
    display_zone     TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS requirements (
    requirement_id          TEXT PRIMARY KEY,
    appointment_request_id  TEXT NOT NULL REFERENCES appointment_requests(request_id) ON DELETE RESTRICT,
    kind                    TEXT NOT NULL CHECK (kind IN
                              ('REFERRAL','PRIOR_AUTHORIZATION','QUESTIONNAIRE',
                               'PREPARATION_ACKNOWLEDGEMENT','INTERPRETER','ACCESSIBILITY_ACCOMMODATION')),
    status                  TEXT NOT NULL CHECK (status IN ('PENDING','SATISFIED','WAIVED','EXPIRED')),
    evidence_ref            TEXT
);

CREATE TABLE IF NOT EXISTS requirement_acknowledgements (
    acknowledgement_id  TEXT PRIMARY KEY,
    requirement_id      TEXT NOT NULL REFERENCES requirements(requirement_id) ON DELETE RESTRICT,
    acknowledged_at     TEXT NOT NULL,
    acknowledgement_text_key TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slot_leases (
    lease_id                TEXT PRIMARY KEY,
    slot_id                 TEXT NOT NULL REFERENCES slots(slot_id) ON DELETE RESTRICT,
    patient_id              TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE RESTRICT,
    issued_at               TEXT NOT NULL,
    expires_at              TEXT NOT NULL,
    released                INTEGER NOT NULL DEFAULT 0 CHECK (released IN (0,1)),
    canonical_request_hash  TEXT NOT NULL,
    CHECK (expires_at > issued_at)
);

-- At most one live lease per slot. Released or expired leases are historical.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_live_lease_per_slot
    ON slot_leases (slot_id) WHERE released = 0;

CREATE TABLE IF NOT EXISTS appointments (
    appointment_id         TEXT PRIMARY KEY,
    patient_id             TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE RESTRICT,
    slot_id                TEXT NOT NULL REFERENCES slots(slot_id) ON DELETE RESTRICT,
    service_id             TEXT NOT NULL REFERENCES healthcare_services(service_id) ON DELETE RESTRICT,
    practitioner_id        TEXT NOT NULL REFERENCES practitioners(practitioner_id) ON DELETE RESTRICT,
    location_id            TEXT NOT NULL REFERENCES locations(location_id) ON DELETE RESTRICT,
    status                 TEXT NOT NULL CHECK (status IN
                             ('PROPOSED','BOOKED','CANCELLED','RESCHEDULED','NO_SHOW','COMPLETED')),
    start_utc              TEXT NOT NULL,
    end_utc                TEXT NOT NULL,
    display_zone           TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    idempotency_key_hash   TEXT,
    superseded_by          TEXT REFERENCES appointments(appointment_id) ON DELETE RESTRICT,
    CHECK (end_utc > start_utc),
    CHECK (superseded_by IS NULL OR superseded_by <> appointment_id)
);

CREATE INDEX IF NOT EXISTS idx_appointments_patient ON appointments (patient_id, status);

-- Only ONE appointment row per slot may be in a non-terminal-occupied state.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_active_appointment_per_slot
    ON appointments (slot_id) WHERE status IN ('BOOKED','RESCHEDULED');

CREATE TABLE IF NOT EXISTS appointment_participants (
    appointment_id  TEXT NOT NULL REFERENCES appointments(appointment_id) ON DELETE RESTRICT,
    related_person_id TEXT NOT NULL REFERENCES related_persons(related_person_id) ON DELETE RESTRICT,
    role_code       TEXT NOT NULL,
    PRIMARY KEY (appointment_id, related_person_id, role_code)
);

CREATE TABLE IF NOT EXISTS appointment_status_history (
    history_id      TEXT PRIMARY KEY,
    appointment_id  TEXT NOT NULL REFERENCES appointments(appointment_id) ON DELETE RESTRICT,
    from_status     TEXT,
    to_status       TEXT NOT NULL,
    changed_at      TEXT NOT NULL,
    reason_code     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resource_requirements (
    resource_requirement_id  TEXT PRIMARY KEY,
    slot_id                  TEXT NOT NULL REFERENCES slots(slot_id) ON DELETE RESTRICT,
    resource_kind            TEXT NOT NULL CHECK (resource_kind IN
                               ('room','equipment','interpreter','modality')),
    resource_ref             TEXT NOT NULL,
    satisfied                INTEGER NOT NULL DEFAULT 1 CHECK (satisfied IN (0,1))
);

CREATE TABLE IF NOT EXISTS waitlist_entries (
    waitlist_id       TEXT PRIMARY KEY,
    patient_id        TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE RESTRICT,
    service_id        TEXT NOT NULL REFERENCES healthcare_services(service_id) ON DELETE RESTRICT,
    preferred_start_utc TEXT NOT NULL,
    preferred_end_utc   TEXT NOT NULL,
    display_zone      TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    expires_at        TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE','OFFERED','EXPIRED','CONVERTED')),
    CHECK (preferred_end_utc > preferred_start_utc)
);

-- ---------------------------------------------------------------------------
-- Reliability
-- ---------------------------------------------------------------------------

-- Idempotency: the key is unique; the canonical request hash is stored so that
-- "same key, different data" is detectable and rejected (invariant 9).
CREATE TABLE IF NOT EXISTS idempotency_records (
    idempotency_key      TEXT PRIMARY KEY,
    canonical_request_hash TEXT NOT NULL,
    operation            TEXT NOT NULL,
    outcome_kind         TEXT NOT NULL CHECK (outcome_kind IN ('KNOWN_SUCCESS','KNOWN_FAILURE','UNKNOWN')),
    reason_code          TEXT NOT NULL,
    response_json        TEXT NOT NULL,
    created_at           TEXT NOT NULL
);

-- Outbox rows are written in the same transaction as the mutation they describe.
CREATE TABLE IF NOT EXISTS outbox_events (
    outbox_id       TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    aggregate_id    TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    published_at    TEXT
);

-- UNKNOWN commit outcomes are recorded so reconciliation has something to work
-- with (invariant 10).
CREATE TABLE IF NOT EXISTS commit_reconciliations (
    reconciliation_id  TEXT PRIMARY KEY,
    idempotency_key    TEXT NOT NULL,
    slot_id            TEXT NOT NULL,
    patient_id         TEXT NOT NULL,
    canonical_request_hash TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    resolved_at        TEXT,
    resolution         TEXT CHECK (resolution IS NULL OR resolution IN
                         ('RECONCILED_SUCCESS','RECONCILED_FAILURE'))
);

-- ---------------------------------------------------------------------------
-- Governance / audit
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS policy_versions (
    policy_version   TEXT PRIMARY KEY,
    jurisdiction     TEXT NOT NULL,
    intended_use     TEXT NOT NULL,
    bundle_sha256    TEXT NOT NULL,
    owner            TEXT NOT NULL,
    effective_from   TEXT NOT NULL,
    effective_until  TEXT,
    approved_by      TEXT,
    rollback_target  TEXT
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    prompt_version_id TEXT PRIMARY KEY,
    prompt_sha256     TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    owner             TEXT NOT NULL,
    rollback_target   TEXT
);

CREATE TABLE IF NOT EXISTS model_configs (
    model_config_id  TEXT PRIMARY KEY,
    provider         TEXT NOT NULL,
    model_id         TEXT NOT NULL,
    parameters_json  TEXT NOT NULL,
    seed             INTEGER,
    config_sha256    TEXT NOT NULL
);

-- Append-only. No CASCADE on delete anywhere in this table's graph.
CREATE TABLE IF NOT EXISTS audit_events (
    audit_id        TEXT PRIMARY KEY,
    sequence_no     INTEGER NOT NULL,
    event_type      TEXT NOT NULL,
    trace_id        TEXT NOT NULL,
    session_id_hash TEXT NOT NULL,
    policy_version  TEXT NOT NULL,
    prompt_hash     TEXT,
    model_config_hash TEXT,
    tool_version    TEXT,
    aggregate_id    TEXT,
    payload_json    TEXT NOT NULL,
    prev_hash       TEXT NOT NULL,
    row_hash        TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_audit_sequence ON audit_events (sequence_no);
CREATE INDEX IF NOT EXISTS idx_audit_trace ON audit_events (trace_id);

CREATE TABLE IF NOT EXISTS handoffs (
    handoff_id      TEXT PRIMARY KEY,
    session_id_hash TEXT NOT NULL,
    reason_code     TEXT NOT NULL,
    summary         TEXT NOT NULL,
    contact_path    TEXT NOT NULL,
    is_monitored    INTEGER NOT NULL DEFAULT 0 CHECK (is_monitored = 0),
    created_at      TEXT NOT NULL
);

-- The prototype records a *simulated* escalation and must never imply that a
-- service was actually contacted (plan.md section 3.5).
CREATE TABLE IF NOT EXISTS simulated_escalations (
    escalation_id  TEXT PRIMARY KEY,
    session_id_hash TEXT NOT NULL,
    category       TEXT NOT NULL,
    locale         TEXT NOT NULL,
    message_template_id TEXT NOT NULL,
    actually_contacted_services INTEGER NOT NULL DEFAULT 0 CHECK (actually_contacted_services = 0),
    created_at     TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Evaluation (kept in the same database so DB-diff oracles can see it, but
-- scoped so production paths never read it)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id           TEXT PRIMARY KEY,
    suite            TEXT NOT NULL,
    config_json      TEXT NOT NULL,
    policy_version   TEXT NOT NULL,
    prompt_hash      TEXT NOT NULL,
    model_config_hash TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    manifest_json    TEXT
);

CREATE TABLE IF NOT EXISTS scenario_results (
    result_id        TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE RESTRICT,
    scenario_id      TEXT NOT NULL,
    severity         TEXT NOT NULL CHECK (severity IN ('S0','S1','S2','S3','NONE')),
    passed           INTEGER NOT NULL CHECK (passed IN (0,1)),
    eligibility_gate INTEGER NOT NULL CHECK (eligibility_gate IN (0,1)),
    detail_json      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patch_candidates (
    patch_id         TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE RESTRICT,
    target_scenario  TEXT NOT NULL,
    patch_layer      TEXT NOT NULL,
    patch_json       TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('PROPOSED','VALIDATED','ACCEPTED','REJECTED')),
    owner            TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Status-transition trigger. The FSM validates in application code; this
-- trigger means that even a direct SQL update cannot perform an illegal
-- appointment transition.
-- ---------------------------------------------------------------------------

CREATE TRIGGER IF NOT EXISTS trg_appointment_status_transition
BEFORE UPDATE OF status ON appointments
FOR EACH ROW
WHEN OLD.status <> NEW.status
BEGIN
    SELECT CASE
        WHEN OLD.status = 'PROPOSED'
             AND NEW.status NOT IN ('BOOKED','CANCELLED')
            THEN RAISE(ABORT, 'illegal appointment transition from PROPOSED')
        WHEN OLD.status = 'BOOKED'
             AND NEW.status NOT IN ('CANCELLED','RESCHEDULED','COMPLETED','NO_SHOW')
            THEN RAISE(ABORT, 'illegal appointment transition from BOOKED')
        WHEN OLD.status = 'RESCHEDULED'
             AND NEW.status NOT IN ('CANCELLED','COMPLETED','NO_SHOW')
            THEN RAISE(ABORT, 'illegal appointment transition from RESCHEDULED')
        WHEN OLD.status = 'CANCELLED'
            THEN RAISE(ABORT, 'illegal appointment transition from CANCELLED (terminal)')
        WHEN OLD.status = 'COMPLETED'
            THEN RAISE(ABORT, 'illegal appointment transition from COMPLETED (terminal)')
        WHEN OLD.status = 'NO_SHOW'
             AND NEW.status NOT IN ('CANCELLED')
            THEN RAISE(ABORT, 'illegal appointment transition from NO_SHOW')
    END;
END;

-- A slot cannot be consumed by an appointment that does not exist.
CREATE TRIGGER IF NOT EXISTS trg_consumption_matches_appointment
BEFORE INSERT ON slot_consumption
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN NOT EXISTS (SELECT 1 FROM appointments WHERE appointment_id = NEW.consumed_by)
            THEN RAISE(ABORT, 'slot_consumption.consumed_by must reference a real appointment')
    END;
END;
