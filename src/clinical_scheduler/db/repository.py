"""Repository: the only module permitted to touch the database.

Rules enforced here rather than trusted to callers:

* connections are opened per-thread with ``PRAGMA foreign_keys=ON``;
* every write goes through :meth:`Repository.transaction`, which takes an
  immediate write lock, so the check-then-act sequences in the kernel cannot
  interleave;
* :class:`Repository` never reads another session's subject -- PHI-bearing reads
  take an explicit ``subject_id`` and the caller must have obtained it from a
  server-side capability (invariant 5, INV-CS);
* append-only audit rows are chained by hash so tampering is detectable.

SQLite note: ``BEGIN IMMEDIATE`` is what makes the two-phase booking safe. A
deferred transaction would allow two connections to both read "slot free" and
then one to fail at commit time with a confusing error instead of a clean
``LEASE_CONFLICT``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from clinical_scheduler.telemetry import assert_no_hard_denied_keys

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    The canonical request hash depends on this. If canonicalisation were not
    stable, idempotency would silently break by producing different hashes for
    logically identical requests (invariant 9).
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(payload: str | bytes) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()


def canonical_request_hash(operation: str, arguments: dict[str, Any]) -> str:
    """Hash the operation together with its canonical arguments.

    Deliberately excludes timestamps and correlation ids: retrying the *same*
    logical request must produce the *same* hash, otherwise the "same key,
    different data" check fires on every retry.
    """
    return sha256_hex(canonical_json({"op": operation, "args": arguments}))


def dump_snapshot_sql(snapshot: Mapping[str, list[dict[str, Any]]]) -> str:
    """Deterministic SQL text for a logical snapshot.

    This is what produces the ``db_before.sql`` / ``db_after.sql`` run artifacts
    required by ``plan.md`` section 18. It is a module function rather than only a
    method because the evaluation driver writes those artifacts from the
    *recorded* snapshots, after the database connection is long gone.

    Restored 2026-10-09. The method that wrapped it was deleted as dead code during
    the Track A audit -- correctly by the letter of "no dead code", and wrongly
    here: it was dead because nothing called it *yet*, not because nothing needed
    it. ``plan.md`` required the artifact. Wire the capability; do not delete it.
    """
    lines: list[str] = []
    for table in sorted(snapshot):
        lines.append(f"-- {table}")
        for row in snapshot[table]:
            columns = ",".join(sorted(row))
            values = ",".join(repr(row[column]) for column in sorted(row))
            lines.append(f"INSERT INTO {table} ({columns}) VALUES ({values});")
        lines.append("")
    return "\n".join(lines)


class Repository:
    """SQLite-backed repository.

    One ``Repository`` instance owns one file. Concurrency tests open several
    instances against the same file to exercise real locking behaviour.
    """

    def __init__(self, db_path: str | Path, *, read_only: bool = False) -> None:
        self.db_path = str(db_path)
        self.read_only = read_only
        self._local = threading.local()
        if not read_only:
            self._initialise()

    # -- connection management -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            uri = f"file:{Path(self.db_path).as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=10.0)
        else:
            conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL keeps readers from blocking the writer, which matters for the
        # concurrency tests; busy_timeout makes contention wait rather than fail.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _initialise(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        finally:
            conn.close()

    # -- transactions ----------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Immediate-write transaction with rollback on any exception.

        Nesting is not supported and not needed: the kernel composes operations
        by calling one transaction per logical unit of work. Nested use would
        risk a silent partial commit, so it is detected and refused.
        """
        conn = self.conn
        if getattr(self._local, "in_txn", False):
            raise RuntimeError(
                "nested transaction refused: compose operations at the kernel level, "
                "not by nesting repository.transaction()"
            )
        self._local.in_txn = True
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            self._local.in_txn = False

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params))

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        cur = self.conn.execute(sql, params)
        return cur.fetchone()

    # -- audit -----------------------------------------------------------------

    def last_audit_hash(self) -> str:
        row = self.query_one(
            "SELECT row_hash FROM audit_events ORDER BY sequence_no DESC LIMIT 1"
        )
        return row["row_hash"] if row else "GENESIS"

    def append_audit(
        self,
        *,
        audit_id: str,
        event_type: str,
        trace_id: str,
        session_id_hash: str,
        policy_version: str,
        payload: dict[str, Any],
        prompt_hash: str | None = None,
        model_config_hash: str | None = None,
        tool_version: str | None = None,
        aggregate_id: str | None = None,
    ) -> str:
        """Append one audit row and return its hash.

        The hash chain (``prev_hash`` -> ``row_hash``) makes silent tampering
        detectable: changing any historical row breaks every subsequent hash.
        This is the "tamper-evident" control referenced in plan.md section 14.2.

        The PHI-redaction control is enforced here rather than trusted to
        callers. An audit row is the one place a field name would persist
        forever, so a hard-denied name is *refused* -- not dropped, which would
        silently destroy the audit trail.
        """
        assert_no_hard_denied_keys(payload, where=f"audit payload for {event_type}")
        row = self.query_one("SELECT COALESCE(MAX(sequence_no), 0) AS n FROM audit_events")
        seq = int(row["n"]) + 1
        prev = self.last_audit_hash()
        created = datetime.now(timezone.utc).isoformat()
        payload_json = canonical_json(payload)
        digest_input = canonical_json(
            {
                "sequence_no": seq,
                "event_type": event_type,
                "trace_id": trace_id,
                "session_id_hash": session_id_hash,
                "policy_version": policy_version,
                "payload": json.loads(payload_json),
                "prev_hash": prev,
                "created_at": created,
            }
        )
        row_hash = sha256_hex(digest_input)
        self.execute(
            """
            INSERT INTO audit_events (
                audit_id, sequence_no, event_type, trace_id, session_id_hash,
                policy_version, prompt_hash, model_config_hash, tool_version,
                aggregate_id, payload_json, prev_hash, row_hash, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                audit_id, seq, event_type, trace_id, session_id_hash,
                policy_version, prompt_hash, model_config_hash, tool_version,
                aggregate_id, payload_json, prev, row_hash, created,
            ),
        )
        return row_hash

    def verify_audit_chain(self, up_to: int | None = None) -> tuple[bool, str]:
        """Recompute the chain. Returns (ok, detail).

        ``up_to`` limits the walk to a sequence number. A checkpoint anchors a
        head, and the rows *below* that head are part of what it anchors, so the
        checkpoint verifier asks for the chain up to its sequence. Without this,
        a row rewritten below the anchor would pass unnoticed: the head's stored
        hash is unchanged by an inconsistent edit further down.
        """
        rows = self.query("SELECT * FROM audit_events ORDER BY sequence_no ASC")
        if up_to is not None:
            rows = [r for r in rows if int(r["sequence_no"]) <= up_to]
        prev = "GENESIS"
        for r in rows:
            if r["prev_hash"] != prev:
                return False, f"chain break at sequence {r['sequence_no']}"
            digest_input = canonical_json(
                {
                    "sequence_no": r["sequence_no"],
                    "event_type": r["event_type"],
                    "trace_id": r["trace_id"],
                    "session_id_hash": r["session_id_hash"],
                    "policy_version": r["policy_version"],
                    "payload": json.loads(r["payload_json"]),
                    "prev_hash": r["prev_hash"],
                    "created_at": r["created_at"],
                }
            )
            if sha256_hex(digest_input) != r["row_hash"]:
                return False, f"hash mismatch at sequence {r['sequence_no']}"
            prev = r["row_hash"]
        return True, f"{len(rows)} audit rows verified"

    # -- idempotency -----------------------------------------------------------
    #
    # The kernel writes outbox rows with its own SQL inside the mutation
    # transaction, so there is deliberately no ``append_outbox`` helper here. One
    # existed and was never called, which turned its "must be called inside the
    # transaction" docstring into a claim about a code path that did not exist;
    # the 2026-10-08 static audit removed it.

    def put_idempotency(
        self,
        *,
        key: str,
        canonical_hash: str,
        operation: str,
        outcome_kind: str,
        reason_code: str,
        response: dict[str, Any],
    ) -> None:
        self.execute(
            """
            INSERT INTO idempotency_records
                (idempotency_key, canonical_request_hash, operation, outcome_kind,
                 reason_code, response_json, created_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                key, canonical_hash, operation, outcome_kind, reason_code,
                canonical_json(response), datetime.now(timezone.utc).isoformat(),
            ),
        )

    # -- snapshotting (used by the DB-diff oracle) -----------------------------

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Full logical snapshot of domain tables.

        Excludes the append-only audit chain, which is expected to grow on every
        call and is verified by its own oracle (L2 ``AUDIT_CHAIN_BROKEN``) rather
        than by a row diff. Evaluation tables are excluded because they are not
        clinical state.

        ``outbox_events`` *is* included: an outbox row written but never reported
        is precisely the divergence the state diff exists to catch.
        """
        tables = [
            "slots", "slot_leases", "slot_consumption", "appointments",
            "appointment_status_history", "requirements",
            "requirement_acknowledgements", "waitlist_entries",
            "idempotency_records", "commit_reconciliations", "handoffs",
            "simulated_escalations", "identity_collision_tokens",
            "collision_attempts", "authorizations", "outbox_events",
        ]
        out: dict[str, list[dict[str, Any]]] = {}
        for t in tables:
            rows = self.query(f"SELECT * FROM {t} ORDER BY 1")
            out[t] = [dict(r) for r in rows]
        return out

__all__ = [
    "Repository",
    "SCHEMA_PATH",
    "canonical_json",
    "canonical_request_hash",
    "dump_snapshot_sql",
    "sha256_hex",
]
