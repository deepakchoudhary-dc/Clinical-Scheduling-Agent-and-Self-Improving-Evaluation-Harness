"""Determinism: the same inputs must produce the same experiment.

The closed-loop demonstration regenerates its artifact bundle, and the checksums
were once observed to verify *after* a regeneration -- which is how the run's
byte-determinism was discovered. That was an accident. These tests make it a
property.

The negative control matters as much as the positive ones here: a fingerprint that
ignores every input is trivially constant, so the tests below require that it
*does* change when a behavioural input changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clinical_scheduler.db.seed import build_fixture_database
from clinical_scheduler.eval.scoring import build_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _manifest(**overrides: Any):
    started = overrides.pop("started_at", datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc))
    finished = overrides.pop("finished_at", datetime(2026, 10, 8, 12, 1, tzinfo=timezone.utc))
    base: dict[str, Any] = {
        "run_id": "run_a",
        "started_at": started,
        "finished_at": finished,
        "policy_version": "2026.10-demo.1",
        "policy_bundle_hash": "b" * 64,
        "prompt": "a system prompt",
        "tool_versions": {"hold_slot": "hold_slot@1"},
        "schema_version": "1",
        "fixture_hash": "f" * 64,
        "model_provider": "LoopbackModel",
        "model_id": "loopback",
        "model_parameters": {"seed": 0, "temperature": 0.0, "max_turns": 12},
        "scenario_set_versions": {"core": "1.0"},
        "scenario_ids": ["S01", "S02"],
        "holdout_included": False,
        "repo_root": REPO_ROOT,
        "judge_version": "human:layer5-human-review/1",
        "notes": ["test"],
    }
    base.update(overrides)
    return build_manifest(**base)


# -- fixture ------------------------------------------------------------------


def test_the_fixture_is_deterministic(tmp_path: Path) -> None:
    """Two builds of the same seed must hash the same, or no run is comparable."""
    first = build_fixture_database(str(tmp_path / "a.sqlite"))
    second = build_fixture_database(str(tmp_path / "b.sqlite"))
    assert first.fixture_hash == second.fixture_hash
    assert first.slot_ids == second.slot_ids
    assert first.patient_ids == second.patient_ids


# -- manifest fingerprint -----------------------------------------------------


def test_the_fingerprint_ignores_the_run_id_and_the_timestamps() -> None:
    """Two runs on different machines at different times are the same experiment."""
    first = _manifest()
    second = _manifest(
        run_id="run_b",
        started_at=datetime(2026, 10, 9, 3, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 10, 9, 3, 5, tzinfo=timezone.utc),
    )
    assert first.fingerprint() == second.fingerprint()


def test_the_fingerprint_changes_with_the_seed() -> None:
    """The negative control: a fingerprint that never changes would prove nothing."""
    first = _manifest(model_parameters={"seed": 0, "temperature": 0.0, "max_turns": 12})
    second = _manifest(model_parameters={"seed": 1, "temperature": 0.0, "max_turns": 12})
    assert first.fingerprint() != second.fingerprint()


def test_the_fingerprint_changes_with_the_fixture() -> None:
    assert _manifest(fixture_hash="a" * 64).fingerprint() != _manifest(
        fixture_hash="b" * 64
    ).fingerprint()


def test_the_fingerprint_changes_with_the_policy() -> None:
    """Changing policy changes behaviour, so it must change the experiment's name."""
    assert _manifest(policy_version="2026.10-demo.1").fingerprint() != _manifest(
        policy_version="2026.10-demo.2"
    ).fingerprint()


def test_the_fingerprint_changes_with_the_scenario_set() -> None:
    assert _manifest(scenario_ids=["S01"]).fingerprint() != _manifest(
        scenario_ids=["S01", "S02"]
    ).fingerprint()


# -- the manifest records what it should --------------------------------------


def test_the_manifest_records_the_behavioural_inputs() -> None:
    manifest = _manifest()
    assert manifest.policy_version == "2026.10-demo.1"
    assert manifest.model_id == "loopback"
    assert manifest.schema_version == "1"
    assert manifest.supports_seed is True
    assert manifest.holdout_included is False


def test_a_run_without_a_seed_is_recorded_as_not_supporting_one() -> None:
    manifest = _manifest(model_parameters={"temperature": 0.0})
    assert manifest.supports_seed is False
