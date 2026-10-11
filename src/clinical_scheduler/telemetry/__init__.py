"""Telemetry: structured events and PHI-aware redaction."""

from clinical_scheduler.telemetry.redaction import (
    HARD_DENY,
    LOG_ALLOWLIST,
    RedactionError,
    assert_no_canaries,
    assert_no_hard_denied_keys,
    idempotency_key_hash,
    redact_text,
    sanitise_log_record,
    session_id_hash,
)

__all__ = [
    "HARD_DENY",
    "LOG_ALLOWLIST",
    "RedactionError",
    "assert_no_canaries",
    "assert_no_hard_denied_keys",
    "idempotency_key_hash",
    "redact_text",
    "sanitise_log_record",
    "session_id_hash",
]
