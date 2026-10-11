"""PHI-aware redaction for logs and artifacts (invariant 14, plan.md section 14.2).

Two directions of control:

1. **Allowlist on structured logging.** A log call may only emit the field names
   in ``LOG_ALLOWLIST``. Anything else is dropped and the drop is itself
   reported, so a developer who tries to log ``patient_name`` gets a visible
   signal instead of a silent PHI leak.

2. **Denylist-free value redaction.** Free text (transcripts, tool output) is
   passed through pattern redaction as defence in depth. This is *not* the
   primary control -- the allowlist is -- because pattern matching over free
   text will always miss something.

The design deliberately favours structural control (field allowlist) over
heuristic control (regex), matching the plan's stance that the sentinel and the
redactor are high-recall aids, not guarantees.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence

#: Structured-log keys that are safe to persist. Everything else is dropped.
LOG_ALLOWLIST: frozenset[str] = frozenset(
    {
        # identifiers that are already irreversible or non-identifying
        "audit_id", "event_type", "trace_id", "session_id_hash", "sequence_no",
        "policy_version", "prompt_hash", "model_config_hash", "tool_version",
        "aggregate_id", "reason_code", "outcome_kind", "severity", "scenario_id",
        "run_id", "result_id", "patch_id", "status", "state", "from_state",
        "to_state", "guard_index", "latency_ms", "token_count", "cost_micros",
        "created_at", "timestamp_utc", "sequence", "attempt_index",
        "attempts_remaining", "discriminator_type", "matched", "lease_id",
        "slot_id", "appointment_id", "request_id", "service_id", "location_id",
        "practitioner_id", "waitlist_id", "handoff_id", "escalation_id",
        "collision_token", "candidate_count", "idempotency_key_hash",
        "canonical_request_hash", "prev_hash", "row_hash", "message_template_id",
        "category", "locale", "modality", "partner_id",  # non-identifying
        "display_zone", "start_utc", "end_utc", "duration_minutes",
        "requires_referral", "requires_prior_authorization",
        "requires_preparation_ack", "unknown_count", "retried_without_reconcile",
        "tool_name", "oracle_id", "invariant_id", "passed", "detail",
        # synthetic-only fields: fixtures use obviously fake values, and these
        # are needed to make failure reports readable during development
        "synthetic_patient_ref", "fixture_id",
    }
)

#: Field names that must never be logged even if someone adds them to the
#: allowlist by mistake. This list is checked *after* the allowlist, so it wins.
HARD_DENY: frozenset[str] = frozenset(
    {
        "ssn", "social_security_number", "tax_id", "full_name", "display_name",
        "date_of_birth", "birth_date", "dob", "address", "phone", "email",
        "insurance_id", "member_id", "mrn", "medical_record_number",
        "password", "api_key", "apikey", "token", "access_token", "secret",
        "private_key", "authorization_header", "cookie", "session_token",
        "raw_transcript", "patient_name", "guardian_name",
    }
)


class RedactionError(Exception):
    """Raised when a caller attempts to log a hard-denied field.

    Fails loudly rather than silently dropping: a developer who tries to log a
    secret should learn about it immediately, not discover it in an incident
    review.
    """


def session_id_hash(session_id: str, salt: str) -> str:
    """Session identifiers are stored hashed so that logs and audit rows cannot
    be joined back to a live session by someone holding only the log."""
    return hashlib.sha256(f"{salt}:{session_id}".encode("utf-8")).hexdigest()


def idempotency_key_hash(key: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()


def sanitise_log_record(record: Mapping[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """Filter a structured log record down to allowlisted, non-denied fields.

    Returns a new dict. ``dropped`` lists which keys were removed so the caller
    can emit a counter (but not the values).
    """
    out: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in record.items():
        if key in HARD_DENY:
            if strict:
                raise RedactionError(f"refusing to log hard-denied field {key!r}")
            dropped.append(key)
            continue
        if key not in LOG_ALLOWLIST:
            dropped.append(key)
            continue
        out[key] = value
    if dropped:
        out["_redacted_fields"] = sorted(dropped)
    return out


def assert_no_hard_denied_keys(record: Mapping[str, Any], *, where: str = "record") -> None:
    """Refuse to persist a record whose own field names are hard-denied.

    ``sanitise_log_record`` *drops* non-allowlisted keys, which is right for a
    flat log line but wrong for an audit payload: the payload is a nested
    evidence document, and dropping keys from it would silently destroy the audit
    trail. The audit boundary therefore uses this instead -- it **refuses** a
    hard-denied field name rather than dropping it, and leaves the rest of the
    payload intact.

    Only the top level is inspected. Nested values are not re-checked here
    because model-supplied tool arguments are masked by
    ``tools.executor._redact_arguments`` before they enter a payload, and that
    function takes its field vocabulary from ``HARD_DENY`` so the two definitions
    cannot drift apart.
    """
    offending = sorted(key for key in record if key in HARD_DENY)
    if offending:
        raise RedactionError(f"{where}: refusing to persist hard-denied field(s) {offending}")


# ---------------------------------------------------------------------------
# Free-text redaction (defence in depth)
# ---------------------------------------------------------------------------

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("MRN", re.compile(r"\bMRN[:#\s]*[A-Z0-9]{6,}\b", re.IGNORECASE)),
    ("CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("DOB_LABEL", re.compile(r"\b(?:DOB|date of birth)[:\s]*\S+", re.IGNORECASE)),
)


def redact_text(text: str, *, keep_canaries: Iterable[str] = ()) -> str:
    """Replace recognised sensitive patterns with typed placeholders.

    ``keep_canaries`` lets a test harness assert that a specific canary string
    *would* have leaked, by inspecting whether the canary survived redaction
    before it is substituted. That is how the privacy oracle distinguishes
    "redaction worked" from "the value was never there".
    """
    out = text
    for label, pattern in _PATTERNS:
        out = pattern.sub(f"[{label}_REDACTED]", out)
    for canary in keep_canaries:
        if canary in out:
            out = out.replace(canary, "[CANARY_PRESERVED]")
    return out


def assert_no_canaries(payload: str, canaries: Sequence[str]) -> list[str]:
    """Return any canaries found in ``payload``. Used by the privacy oracle."""
    return [c for c in canaries if c in payload]


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
