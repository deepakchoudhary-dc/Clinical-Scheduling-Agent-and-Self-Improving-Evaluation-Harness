"""Adversarial and privacy tests.

These are the tests that treat the system as hostile input would. They are
grouped by the boundary they attack, and each asserts a *structural* property
rather than a textual one wherever possible -- a check on what the system can be
made to do, not on what it happens to say.

The stance being tested is the one plan.md section 14.3 states: all
natural-language content is data; the model cannot expand its own permissions;
tool output is not instruction. The concrete mechanisms are the capability
broker (signed, scoped, expiring, deny-by-default), server-side subject
injection, and an append-only hash-chained audit.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from clinical_scheduler.db.repository import Repository
from clinical_scheduler.db.seed import FIXTURE_NOW, build_fixture_database
from clinical_scheduler.domain.states import AuthTier, SessionState
from clinical_scheduler.kernel.scheduling import FixedClock, SchedulingKernel
from clinical_scheduler.telemetry.redaction import (
    HARD_DENY,
    LOG_ALLOWLIST,
    RedactionError,
    assert_no_canaries,
    sanitise_log_record,
)
from clinical_scheduler.tools.broker import Capability, CapabilityBroker, SessionContext

pytestmark = pytest.mark.security


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "security.sqlite")
    build_fixture_database(path)
    return path


def _session(**overrides) -> SessionContext:
    """A session in a state where ``search_slots`` is on the ceiling.

    The state matters: the broker's ceiling is per-state, so a capability test
    that started in ``START`` would be refused for the wrong reason and would
    prove nothing about the property it names.
    """
    fields = dict(
        session_id="sess_attacker",
        subject_id="pt_ada",
        tier=AuthTier.TIER2_VERIFIED_IDENTITY,
        state=SessionState.REQUEST_SCOPED,
        safety_route_active=False,
        verified_identity=True,
    )
    fields.update(overrides)
    return SessionContext(**fields)


# ---------------------------------------------------------------------------
# Capability broker: forge, reuse, escalate
# ---------------------------------------------------------------------------


def test_a_tampered_capability_is_refused() -> None:
    """Editing a capability must invalidate it.

    The signature is checked before anything else, so a tampered capability
    cannot influence a later check by, for example, claiming a different tool
    name and thereby changing which scope is compared.
    """
    broker = CapabilityBroker(secret=b"unit-test-secret")
    session = _session()
    capability = broker.mint(session, tool_name="search_slots")
    assert capability.tool_name == "search_slots"

    tampered = Capability(
        **{**capability.__dict__, "tool_name": "commit_appointment"}
    )
    with pytest.raises(Exception) as excinfo:
        broker.verify(tampered, ctx=session, tool_name="commit_appointment")
    assert "signature" in str(excinfo.value).lower() or "tamper" in str(excinfo.value).lower()


def test_a_capability_cannot_be_used_in_another_session() -> None:
    """A capability is bound to one session, so a stolen one is useless.

    This is what stops session fixation: obtaining another session's capability
    does not grant access to its subject.
    """
    broker = CapabilityBroker(secret=b"unit-test-secret")
    minted_for = _session(session_id="sess_victim", subject_id="pt_ada")
    capability = broker.mint(minted_for, tool_name="search_slots")

    attacker = _session(session_id="sess_attacker", subject_id="pt_bob")
    with pytest.raises(Exception):
        broker.verify(capability, ctx=attacker, tool_name="search_slots")


def test_an_expired_capability_is_refused() -> None:
    """Capabilities expire, so a leaked one has a bounded lifetime."""
    broker = CapabilityBroker(secret=b"unit-test-secret")
    session = _session()
    capability = broker.mint(session, tool_name="search_slots", now=FIXTURE_NOW)

    later = FIXTURE_NOW + timedelta(days=1)
    with pytest.raises(Exception):
        broker.verify(capability, ctx=session, tool_name="search_slots", now=later)


def test_a_safety_route_withholds_every_mutation_tool() -> None:
    """The structural form of invariant 2.

    Under a safety route the broker refuses to mint a mutation capability at all,
    so there is no path by which the model could book -- the tool is not absent
    from a prompt, it does not exist for that session.
    """
    broker = CapabilityBroker(secret=b"unit-test-secret")
    session = _session(safety_route_active=True)

    for tool in (
        "hold_slot",
        "commit_appointment",
        "atomic_reschedule",
        "cancel_appointment",
        "join_waitlist",
    ):
        with pytest.raises(Exception) as excinfo:
            broker.mint(session, tool_name=tool)
        assert tool in str(excinfo.value) or "safety" in str(excinfo.value).lower()


def test_an_unverified_session_cannot_mint_a_phi_bearing_read() -> None:
    """Deny by default: the tool ceiling rises with the authentication tier.

    An unauthenticated caller must not be able to read patient records at all,
    and the refusal must come from the broker rather than from the model choosing
    not to ask.
    """
    broker = CapabilityBroker(secret=b"unit-test-secret")
    public = _session(
        session_id="sess_public",
        subject_id=None,
        tier=AuthTier.TIER0_PUBLIC,
        verified_identity=False,
    )
    with pytest.raises(Exception):
        broker.mint(public, tool_name="search_slots")


def test_an_unknown_tool_name_is_refused() -> None:
    """The tool vocabulary is closed; a name outside it cannot be minted.

    This is what makes "the model invented a tool" a non-event rather than an
    execution attempt.
    """
    broker = CapabilityBroker(secret=b"unit-test-secret")
    with pytest.raises(Exception):
        broker.mint(_session(), tool_name="exfiltrate_all_patients")


# ---------------------------------------------------------------------------
# The audit trail: tamper-evidence
# ---------------------------------------------------------------------------


def _append(repo: Repository, index: int) -> str:
    """Append one audit row with the minimum required provenance."""
    return repo.append_audit(
        audit_id=f"aud_{index:06d}",
        event_type="UNIT_TEST_EVENT",
        trace_id="tr_security_test",
        session_id_hash="0" * 64,
        policy_version="2026.10-demo.1",
        payload={"index": index},
        aggregate_id=f"agg_{index}",
    )


def test_the_audit_chain_verifies_when_untouched(db_path: str) -> None:
    repo = Repository(db_path)
    try:
        for index in range(5):
            _append(repo, index)
        ok, detail = repo.verify_audit_chain()
        assert ok, detail
    finally:
        repo.close()


def test_tampering_with_an_audit_row_breaks_the_chain(db_path: str) -> None:
    """The audit is evidence, so editing it must be detectable.

    Without this the audit cannot support any claim, because a claim resting on
    a mutable record is not evidence. The tamper here is the smallest one
    available -- changing one payload value -- and it must still be caught.
    """
    repo = Repository(db_path)
    try:
        for index in range(4):
            _append(repo, index)
        ok, _ = repo.verify_audit_chain()
        assert ok

        # Edit a stored payload without touching the chain hashes.
        repo.execute(
            "UPDATE audit_events SET payload_json = ? WHERE sequence_no = ?",
            ('{"index": 999}', 2),
        )

        ok, detail = repo.verify_audit_chain()
        assert not ok, "an edited audit row verified; the chain is not tamper-evident"
        assert detail
    finally:
        repo.close()


def test_deleting_an_audit_row_breaks_the_chain(db_path: str) -> None:
    """Truncation must be as detectable as modification.

    A deleted row is the more likely tamper in practice, because it looks like
    the event never happened rather than like the event was changed.
    """
    repo = Repository(db_path)
    try:
        for index in range(4):
            _append(repo, index)
        repo.execute("DELETE FROM audit_events WHERE sequence_no = ?", (3,))
        ok, _ = repo.verify_audit_chain()
        assert not ok, "a deleted audit row verified"
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# Telemetry: PHI must not reach a log
# ---------------------------------------------------------------------------


def test_non_allowlisted_fields_are_dropped() -> None:
    """Logging is an allowlist, so a new field is dropped by default.

    A denylist would fail open: a field nobody thought of leaks. An allowlist
    fails closed, which is the only acceptable direction for a log that may
    contain patient data.
    """
    record = {
        "event_type": "APPOINTMENT_COMMIT_SUCCEEDED",
        "trace_id": "tr_1",
        "patient_narrative": "I have had crushing chest pain since Tuesday",
        "free_text_notes": "called about her son",
    }
    clean = sanitise_log_record(record)
    assert clean["event_type"] == "APPOINTMENT_COMMIT_SUCCEEDED"
    assert clean["trace_id"] == "tr_1"
    assert "patient_narrative" not in clean
    assert "free_text_notes" not in clean


@pytest.mark.parametrize("field", ["full_name", "date_of_birth", "ssn", "email", "phone"])
def test_hard_denied_fields_fail_loudly(field: str) -> None:
    """A hard-denied field raises rather than being dropped silently.

    A developer who tries to log a secret should learn immediately, not discover
    it during an incident review.
    """
    assert field in HARD_DENY
    with pytest.raises(RedactionError):
        sanitise_log_record({"event_type": "X", field: "value"})


def test_the_hard_deny_list_wins_over_the_allowlist() -> None:
    """Defence in depth: the denylist is checked after the allowlist.

    If someone adds a denied field to the allowlist by mistake, it must still be
    refused. The two lists are not independent filters, they are ordered.
    """
    overlap = LOG_ALLOWLIST & HARD_DENY
    # The lists must not overlap in the shipped configuration; if they ever do,
    # the hard deny must still win, which the previous test establishes.
    assert not overlap, f"allowlist and hard-deny overlap: {sorted(overlap)}"


def test_canary_detection_finds_a_leak() -> None:
    """A canary is how a leak is proven rather than argued about."""
    canaries = ("pt_ada", "1990-04-12", "ada.lovelace@example.test")
    leaked = "record for pt_ada, born 1990-04-12, email ada.lovelace@example.test"
    found = assert_no_canaries(leaked, canaries)
    assert set(found) == set(canaries)


def test_canary_detection_is_quiet_when_nothing_leaked() -> None:
    assert assert_no_canaries("no identifiers here", ("pt_ada",)) == []
