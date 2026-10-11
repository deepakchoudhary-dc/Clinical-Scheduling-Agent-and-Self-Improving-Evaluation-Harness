"""The evaluator's declarations must match the schema they describe.

``runner.SNAPSHOT_TABLES`` documents which tables the L2 oracle diffs between the
pre- and post-run snapshot. Its docstring claimed it was "asserted against the
snapshot in tests so a new table cannot be added to the schema and silently go
unwatched" -- and no test referenced it. The static audit of 2026-10-08 also found
it declared ``audit_events`` while ``Repository.snapshot()`` did not return that
table, i.e. the drift the claim was supposed to prevent had already happened.

The tests below make the claim true in both directions:

* ``SNAPSHOT_TABLES`` equals the keys of ``Repository.snapshot()``;
* every table in ``schema.sql`` is accounted for -- either diffed, or in the
  explicit exclusion list with a reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from clinical_scheduler.db.repository import SCHEMA_PATH, Repository
from clinical_scheduler.eval.runner import SNAPSHOT_TABLES

#: Schema tables deliberately outside the state diff, each with its reason. A new
#: table must be added either to ``SNAPSHOT_TABLES`` or here -- there is no third
#: option, which is what makes the promise checkable.
EXCLUDED_FROM_SNAPSHOT: dict[str, str] = {
    # identifier allocation and reference data: not mutated by a scenario
    "id_sequences": "identifier allocation; grows monotonically",
    "locations": "reference data",
    "healthcare_services": "reference data",
    "practitioners": "reference data",
    "practitioner_roles": "reference data",
    "schedules": "reference data",
    # identity: read for verification, written only by the fixture
    "patients": "fixture-only; PHI-bearing, never diffed",
    "patient_identifiers": "fixture-only; tokenised identifiers",
    "related_persons": "fixture-only",
    # request records written before the booking flow proper
    "appointment_requests": "written once at request scope; covered by the trace oracle",
    "appointment_participants": "written with the appointment; covered by the trace oracle",
    "resource_requirements": "fixture-only",
    # governance and evaluation: append-only or not clinical state
    "policy_versions": "governance record",
    "prompt_versions": "governance record",
    "model_configs": "governance record",
    "audit_events": "append-only; verified by its own oracle (L2 AUDIT_CHAIN_BROKEN)",
    "evaluation_runs": "evaluation bookkeeping",
    "scenario_results": "evaluation bookkeeping",
    "patch_candidates": "evaluation bookkeeping",
}


def _schema_tables() -> set[str]:
    return set(
        re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", SCHEMA_PATH.read_text(encoding="utf-8"))
    )


@pytest.fixture()
def repository(tmp_path: Path):
    repo = Repository(str(tmp_path / "declarations.sqlite"))
    try:
        yield repo
    finally:
        repo.close()


def test_snapshot_tables_equal_the_repository_snapshot(repository: Repository) -> None:
    """The declaration and the implementation must be the same set."""
    assert set(SNAPSHOT_TABLES) == set(repository.snapshot()), (
        "SNAPSHOT_TABLES and Repository.snapshot() have drifted apart: "
        f"declared-only={sorted(set(SNAPSHOT_TABLES) - set(repository.snapshot()))}, "
        f"implemented-only={sorted(set(repository.snapshot()) - set(SNAPSHOT_TABLES))}"
    )


def test_every_schema_table_is_either_diffed_or_explicitly_excluded() -> None:
    """The promise in the docstring, made mechanical.

    A table added to the schema and forgotten here fails this test, rather than
    being silently unwatched by the state diff.
    """
    accounted = set(SNAPSHOT_TABLES) | set(EXCLUDED_FROM_SNAPSHOT)
    unaccounted = sorted(_schema_tables() - accounted)
    assert not unaccounted, (
        f"schema table(s) neither diffed nor explicitly excluded: {unaccounted}. "
        "Add each to SNAPSHOT_TABLES or to EXCLUDED_FROM_SNAPSHOT with a reason."
    )


def test_the_exclusion_list_has_no_stale_entries() -> None:
    """An exclusion naming a table that no longer exists is a stale claim."""
    stale = sorted(set(EXCLUDED_FROM_SNAPSHOT) - _schema_tables())
    assert not stale, f"exclusion list names table(s) that are not in the schema: {stale}"
