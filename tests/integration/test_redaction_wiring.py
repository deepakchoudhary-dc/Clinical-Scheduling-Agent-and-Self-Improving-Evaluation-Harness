"""The PHI-redaction control must be enforced on the real path, not just exist.

Found by a static audit on 2026-10-08. ``telemetry/redaction.py`` implemented the
allowlist, the hard-deny list and the free-text redactor, and the threat model
described "PHI-aware logging: implemented" — but **no runtime module imported the
module**. Its only callers were its own tests. A control that exists and is never
invoked is not a control; it is a claim.

The wiring is now:

* ``db.repository.append_audit`` refuses a hard-denied *field name* at the audit
  boundary — the one place a field name persists forever. It refuses rather than
  drops, because dropping would silently destroy the audit trail;
* ``tools.executor._redact_arguments`` masks sensitive argument *values* and runs
  free text through pattern redaction before it is stored.

The last test pins the property that makes the two stay consistent: the argument
vocabulary is derived from ``HARD_DENY`` rather than re-declared.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_scheduler.db.repository import Repository
from clinical_scheduler.telemetry import HARD_DENY, RedactionError, redact_text
from clinical_scheduler.tools.executor import _SENSITIVE_ARGUMENTS, _redact_arguments

pytestmark = pytest.mark.integration


@pytest.fixture()
def repo(tmp_path: Path):
    repository = Repository(str(tmp_path / "redaction.sqlite"))
    try:
        yield repository
    finally:
        repository.close()


def _audit(repo: Repository, payload: dict) -> str:
    return repo.append_audit(
        audit_id="aud_1",
        event_type="TEST_EVENT",
        trace_id="trace_redaction",
        session_id_hash="h" * 64,
        policy_version="2026.10-demo.1",
        payload=payload,
    )


def test_a_normal_audit_payload_is_stored_unchanged(repo: Repository) -> None:
    """The guard must not break the audit path it now protects.

    An audit payload is a nested evidence document; the executor nests masked
    arguments under ``arguments``. That shape must keep working, which is why the
    boundary uses a refusal check rather than the allowlist filter.
    """
    digest = _audit(
        repo, {"arguments": {"family_name": "[REDACTED]"}, "evidence": {"reason": "OK"}}
    )
    assert len(digest) == 64
    row = repo.query_one("SELECT payload_json FROM audit_events")
    assert "arguments" in row["payload_json"]
    assert "evidence" in row["payload_json"]


def test_the_audit_boundary_refuses_a_hard_denied_field(repo: Repository) -> None:
    """The control, on the real path. This is what "implemented" must mean."""
    with pytest.raises(RedactionError) as excinfo:
        _audit(repo, {"patient_name": "Ada Lovelace"})
    assert "patient_name" in str(excinfo.value)
    assert repo.query("SELECT * FROM audit_events") == [], (
        "the refused row was written anyway; the guard must run before the insert"
    )


def test_tool_arguments_are_masked_and_free_text_is_redacted() -> None:
    """Both directions of control, on the function the executor actually calls."""
    out = _redact_arguments(
        {
            "family_name": "Lovelace",           # sensitive name -> value masked
            "query_text": "reach me at a@b.com",  # free text -> pattern redacted
            "service_id": "svc_gp_new",           # ordinary -> untouched
        }
    )
    assert out["family_name"] == "[REDACTED]"
    assert out["query_text"] == "reach me at [EMAIL_REDACTED]"
    assert out["service_id"] == "svc_gp_new"


def test_the_argument_vocabulary_is_derived_from_the_telemetry_layer() -> None:
    """Guard against the two definitions drifting apart.

    If someone adds a field to ``HARD_DENY`` expecting the audit path to mask it,
    this fails rather than letting the new field be stored verbatim.
    """
    assert HARD_DENY <= _SENSITIVE_ARGUMENTS, (
        "a hard-denied field is not masked in tool arguments: "
        f"{sorted(HARD_DENY - _SENSITIVE_ARGUMENTS)}"
    )


def test_the_free_text_redactor_is_reachable_from_the_telemetry_package() -> None:
    """The package re-exports what the runtime imports, so there is one import path.

    The MRN pattern consumes the label as well as the value, which is the
    intended behaviour: leaving ``MRN:`` in place would still signal that the
    text was a medical record number.
    """
    assert redact_text("MRN: 123456789") == "[MRN_REDACTED]"
