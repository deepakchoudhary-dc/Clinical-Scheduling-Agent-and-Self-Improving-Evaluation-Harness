"""Signed audit checkpoints (plan.md section 14.2; threat-model "audit integrity").

The threat model is explicit about the limit of the hash chain: it detects
accidental or partial tampering -- a stray UPDATE, a truncated table, a failed
migration -- and an attacker who holds database write access can rewrite the whole
chain *consistently*, and ``verify_audit_chain`` will report OK. That is a real
limitation, and the threat model marks the control "absent" rather than pretending
otherwise.

Tamper **evidence** needs an anchor the database does not hold. This module emits
a checkpoint -- the sequence number and row hash at a point in time -- signed with
a key supplied from the environment, and stored outside the audit table. An
attacker who rewrites the chain cannot produce a matching signature, because
producing one requires the key.

**What this is not.** It is not proof that the audit is complete, and it does not
anchor anything if the key lives beside the database. The key must come from
outside: an environment variable, a secret manager, or a KMS. ``key_id`` is a
label that records *which* key signed a checkpoint, never the key itself, so a
checkpoint file stays safe to commit.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clinical_scheduler.db.repository import Repository

#: Environment variable holding the checkpoint signing key.
CHECKPOINT_KEY_ENV_VAR = "CLINICAL_SCHEDULER_CHECKPOINT_KEY"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be produced (no rows, no key)."""


@dataclass(frozen=True)
class Checkpoint:
    """An anchored point in the audit chain."""

    sequence_no: int
    row_hash: str
    created_at: str
    key_id: str
    signature: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "sequence_no": self.sequence_no,
                "row_hash": self.row_hash,
                "created_at": self.created_at,
                "key_id": self.key_id,
                "signature": self.signature,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "Checkpoint":
        raw = json.loads(text)
        return cls(
            sequence_no=int(raw["sequence_no"]),
            row_hash=str(raw["row_hash"]),
            created_at=str(raw["created_at"]),
            key_id=str(raw["key_id"]),
            signature=str(raw["signature"]),
        )


def _payload(sequence_no: int, row_hash: str, created_at: str, key_id: str) -> str:
    """The canonical bytes that are signed.

    Canonical JSON with sorted keys, so the signature does not depend on field
    order or incidental whitespace.
    """
    return json.dumps(
        {
            "sequence_no": sequence_no,
            "row_hash": row_hash,
            "created_at": created_at,
            "key_id": key_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def sign_checkpoint(
    repo: Repository,
    *,
    key: bytes,
    key_id: str = "default",
    now: datetime | None = None,
) -> Checkpoint:
    """Anchor the current head of the audit chain."""
    if not key:
        raise CheckpointError("a checkpoint key is required; an unsigned anchor proves nothing")
    row = repo.query_one("SELECT COALESCE(MAX(sequence_no), 0) AS n FROM audit_events")
    sequence_no = int(row["n"])
    if sequence_no == 0:
        raise CheckpointError("the audit chain is empty; there is nothing to anchor")
    row_hash = repo.last_audit_hash()
    created_at = (now or datetime.now(timezone.utc)).isoformat()
    signature = hmac.new(
        key, _payload(sequence_no, row_hash, created_at, key_id).encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return Checkpoint(
        sequence_no=sequence_no,
        row_hash=row_hash,
        created_at=created_at,
        key_id=key_id,
        signature=signature,
    )


def verify_checkpoint(
    repo: Repository, checkpoint: Checkpoint, *, key: bytes
) -> tuple[bool, str]:
    """Check a signature, then check it still describes the chain.

    Three checks, and each catches a different attack:

    1. the **signature** -- the checkpoint file was edited, or this is not the key
       that signed it;
    2. the **chain up to the anchored point** -- a row *below* the anchor was
       rewritten. This is not redundant with check 3: an inconsistent edit further
       down leaves the head's stored hash untouched, so comparing the head alone
       would pass;
    3. the **head row itself** -- a consistent rewrite that recomputed every hash
       after it, which necessarily changes the anchored row's hash.
    """
    expected = hmac.new(
        key,
        _payload(
            checkpoint.sequence_no, checkpoint.row_hash, checkpoint.created_at, checkpoint.key_id
        ).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, checkpoint.signature):
        return (
            False,
            "signature does not verify: the checkpoint was edited, or this is not the key "
            "that signed it",
        )

    chain_ok, chain_detail = repo.verify_audit_chain(up_to=checkpoint.sequence_no)
    if not chain_ok:
        return (
            False,
            f"the chain up to the anchored point does not verify ({chain_detail}): a row "
            "at or below the anchor was rewritten",
        )

    row = repo.query_one(
        "SELECT row_hash FROM audit_events WHERE sequence_no = ?", (checkpoint.sequence_no,)
    )
    if row is None:
        return (
            False,
            f"no audit row at sequence {checkpoint.sequence_no}: the chain was truncated "
            "below the anchored point",
        )
    if row["row_hash"] != checkpoint.row_hash:
        return (
            False,
            f"audit row {checkpoint.sequence_no} has a different hash than the signed "
            "checkpoint: the chain was rewritten after it was anchored",
        )
    return True, f"verified: audit row {checkpoint.sequence_no} still matches the signed checkpoint"


def checkpoint_key_from_env(var: str = CHECKPOINT_KEY_ENV_VAR) -> bytes:
    """Read the signing key from the environment.

    From the environment rather than from a file beside the database, because an
    anchor stored next to the thing it anchors is not an anchor.
    """
    value = os.environ.get(var)
    if not value:
        raise CheckpointError(
            f"no checkpoint key: set {var} to a secret the database does not hold"
        )
    return value.encode("utf-8")


def write_checkpoint(checkpoint: Checkpoint, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(checkpoint.to_json(), encoding="utf-8")
    return destination


def read_checkpoint(path: str | Path) -> Checkpoint:
    return Checkpoint.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "CHECKPOINT_KEY_ENV_VAR",
    "Checkpoint",
    "CheckpointError",
    "checkpoint_key_from_env",
    "read_checkpoint",
    "sign_checkpoint",
    "verify_checkpoint",
    "write_checkpoint",
]
