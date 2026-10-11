"""Signed audit checkpoints: the anchor the hash chain cannot provide.

The threat model says plainly that an attacker with database write access can
rewrite the chain *consistently* and ``verify_audit_chain`` will report OK. These
tests pin the two halves of the answer: a signature that only the key holder can
produce, and a check that the anchored row still hashes to what was signed.

Every negative case is exercised, because a verification function that cannot
return False is decoration.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from clinical_scheduler.db.checkpoint import (
    CHECKPOINT_KEY_ENV_VAR,
    CheckpointError,
    checkpoint_key_from_env,
    read_checkpoint,
    sign_checkpoint,
    verify_checkpoint,
    write_checkpoint,
)
from clinical_scheduler.db.repository import Repository

KEY = b"a-test-key-the-database-does-not-hold"
WRONG_KEY = b"a-different-key"


@pytest.fixture()
def repo(tmp_path: Path):
    repository = Repository(str(tmp_path / "audit.sqlite"))
    try:
        yield repository
    finally:
        repository.close()


def _append(repo: Repository, count: int = 2) -> None:
    for index in range(count):
        repo.append_audit(
            audit_id=f"aud_{index}",
            event_type="TEST_EVENT",
            trace_id="trace_checkpoint",
            session_id_hash="h" * 64,
            policy_version="2026.10-demo.1",
            payload={"reason_code": "OK"},
        )


def test_a_signed_checkpoint_verifies(repo: Repository) -> None:
    _append(repo)
    checkpoint = sign_checkpoint(repo, key=KEY)
    ok, detail = verify_checkpoint(repo, checkpoint, key=KEY)
    assert ok, detail
    assert "still matches" in detail


def test_the_checkpoint_records_the_head_of_the_chain(repo: Repository) -> None:
    _append(repo, count=3)
    checkpoint = sign_checkpoint(repo, key=KEY)
    assert checkpoint.sequence_no == 3
    assert checkpoint.row_hash == repo.last_audit_hash()


def test_a_rewrite_below_the_anchor_is_detected(repo: Repository) -> None:
    """An inconsistent edit further down leaves the head's stored hash untouched.

    Comparing only the anchored row would pass here -- which is exactly the hole
    the first version of this verifier had, and which this test found. The
    verifier therefore also walks the chain up to the anchor.
    """
    _append(repo)
    checkpoint = sign_checkpoint(repo, key=KEY)

    repo.execute("UPDATE audit_events SET row_hash = 'rewritten' WHERE sequence_no = 1")

    ok, detail = verify_checkpoint(repo, checkpoint, key=KEY)
    assert ok is False
    assert "at or below the anchor was rewritten" in detail


def _rewrite_head_consistently(repo: Repository) -> None:
    """Simulate an attacker who rewrites the head *and* recomputes its hash.

    This is the capability the hash chain alone cannot defend against: the
    recomputation is internally consistent, so ``verify_audit_chain`` reports OK.
    It is what makes the third check -- comparing the head to the *signed* value
    -- load-bearing rather than redundant.
    """
    from clinical_scheduler.db.repository import canonical_json, sha256_hex

    row = repo.query_one("SELECT * FROM audit_events WHERE sequence_no = 2")
    forged_payload = {"reason_code": "FORGED"}
    digest_input = canonical_json(
        {
            "sequence_no": row["sequence_no"],
            "event_type": row["event_type"],
            "trace_id": row["trace_id"],
            "session_id_hash": row["session_id_hash"],
            "policy_version": row["policy_version"],
            "payload": forged_payload,
            "prev_hash": row["prev_hash"],
            "created_at": row["created_at"],
        }
    )
    repo.execute(
        "UPDATE audit_events SET payload_json = ?, row_hash = ? WHERE sequence_no = 2",
        (canonical_json(forged_payload), sha256_hex(digest_input)),
    )


def test_a_consistent_rewrite_of_the_head_is_detected(repo: Repository) -> None:
    """The chain verifies; the anchor is what catches it."""
    _append(repo)
    checkpoint = sign_checkpoint(repo, key=KEY)
    _rewrite_head_consistently(repo)

    # The rewrite is internally consistent, so the chain check passes...
    assert repo.verify_audit_chain()[0] is True

    # ...and the anchor is the only thing that notices.
    ok, detail = verify_checkpoint(repo, checkpoint, key=KEY)
    assert ok is False
    assert "different hash than the signed checkpoint" in detail


def test_a_truncated_chain_is_detected(repo: Repository) -> None:
    _append(repo, count=3)
    checkpoint = sign_checkpoint(repo, key=KEY)

    repo.execute("DELETE FROM audit_events WHERE sequence_no = 3")

    ok, detail = verify_checkpoint(repo, checkpoint, key=KEY)
    assert ok is False
    assert "truncated" in detail


def test_the_wrong_key_is_rejected(repo: Repository) -> None:
    _append(repo)
    checkpoint = sign_checkpoint(repo, key=KEY)
    ok, detail = verify_checkpoint(repo, checkpoint, key=WRONG_KEY)
    assert ok is False
    assert "signature does not verify" in detail


def test_an_edited_checkpoint_is_rejected(repo: Repository) -> None:
    """Editing the anchored hash must not be enough -- the signature covers it."""
    _append(repo)
    checkpoint = sign_checkpoint(repo, key=KEY)
    forged = dataclasses.replace(checkpoint, row_hash="0" * 64)
    ok, detail = verify_checkpoint(repo, forged, key=KEY)
    assert ok is False
    assert "signature does not verify" in detail


def test_an_empty_chain_cannot_be_anchored(repo: Repository) -> None:
    with pytest.raises(CheckpointError):
        sign_checkpoint(repo, key=KEY)


def test_an_empty_key_is_refused(repo: Repository) -> None:
    """An unsigned anchor proves nothing, so it is refused rather than emitted."""
    _append(repo)
    with pytest.raises(CheckpointError):
        sign_checkpoint(repo, key=b"")


def test_the_checkpoint_round_trips_through_a_file(tmp_path: Path, repo: Repository) -> None:
    _append(repo)
    checkpoint = sign_checkpoint(repo, key=KEY)
    path = write_checkpoint(checkpoint, tmp_path / "anchor" / "checkpoint.json")
    assert read_checkpoint(path) == checkpoint


def test_the_checkpoint_file_carries_a_key_label_not_the_key(repo: Repository) -> None:
    _append(repo)
    checkpoint = sign_checkpoint(repo, key=KEY, key_id="ci-2026-10")
    text = checkpoint.to_json()
    assert "ci-2026-10" in text
    assert KEY.decode() not in text


def test_the_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CHECKPOINT_KEY_ENV_VAR, raising=False)
    with pytest.raises(CheckpointError) as excinfo:
        checkpoint_key_from_env()
    assert CHECKPOINT_KEY_ENV_VAR in str(excinfo.value)

    monkeypatch.setenv(CHECKPOINT_KEY_ENV_VAR, "from-the-environment")
    assert checkpoint_key_from_env() == b"from-the-environment"
