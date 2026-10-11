"""Operational metrics (plan.md section 15.2).

The metrics are read from the audit trail rather than instrumented separately, so
these tests are about two things: that the reading is correct, and that the metrics
the trail *cannot* support are named with their reason instead of being omitted. An
absent metric reads as a healthy one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_scheduler.db.repository import Repository
from clinical_scheduler.db.seed import build_fixture_database
from clinical_scheduler.telemetry.events import UNMEASURABLE, collect


@pytest.fixture()
def repo(tmp_path: Path):
    path = str(tmp_path / "metrics.sqlite")
    build_fixture_database(path)
    repository = Repository(path)
    try:
        yield repository
    finally:
        repository.close()


def _append(repo: Repository, event_type: str, payload: dict) -> None:
    repo.append_audit(
        audit_id=f"aud_{event_type}_{repo.query_one('SELECT COUNT(*) AS n FROM audit_events')['n']}",
        event_type=event_type,
        trace_id="trace_metrics",
        session_id_hash="h" * 64,
        policy_version="2026.10-demo.1",
        payload=payload,
    )


def test_an_empty_trail_reports_zero_and_says_so(repo: Repository) -> None:
    metrics = collect(repo)
    assert metrics.total_events == 0
    assert metrics.events_by_type == {}
    assert metrics.appointments_by_status == {}
    assert "none recorded" in metrics.report()


def test_events_are_counted_by_type(repo: Repository) -> None:
    _append(repo, "HOLD_SLOT_SUCCEEDED", {"status": "SUCCESS"})
    _append(repo, "HOLD_SLOT_SUCCEEDED", {"status": "SUCCESS"})
    _append(repo, "SEARCH_SLOTS_SUCCEEDED", {"status": "SUCCESS"})

    metrics = collect(repo)
    assert metrics.total_events == 3
    assert metrics.events_by_type == {"HOLD_SLOT_SUCCEEDED": 2, "SEARCH_SLOTS_SUCCEEDED": 1}


def test_failures_are_counted_from_the_payload_envelope(repo: Repository) -> None:
    _append(repo, "HOLD_SLOT_SUCCEEDED", {"status": "SUCCESS"})
    _append(repo, "HOLD_SLOT_FAILED", {"status": "FAILURE", "reason_code": "SLOT_TAKEN"})

    assert collect(repo).failed_operations == 1


def test_the_policy_version_is_recorded(repo: Repository) -> None:
    _append(repo, "HOLD_SLOT_SUCCEEDED", {"status": "SUCCESS"})
    assert collect(repo).policy_versions == ("2026.10-demo.1",)


def test_handoffs_and_escalations_are_counted(repo: Repository) -> None:
    repo.execute(
        "INSERT INTO handoffs (handoff_id, session_id_hash, reason_code, summary, "
        "contact_path, created_at) VALUES (?,?,?,?,?,?)",
        ("ho_1", "h" * 64, "URGENT", "s", "phone", "2026-10-09T00:00:00+00:00"),
    )
    repo.execute(
        "INSERT INTO simulated_escalations (escalation_id, session_id_hash, category, "
        "locale, message_template_id, created_at) VALUES (?,?,?,?,?,?)",
        ("esc_1", "h" * 64, "CARDIAC", "en", "tmpl", "2026-10-09T00:00:00+00:00"),
    )
    metrics = collect(repo)
    assert metrics.human_handoffs == 1
    assert metrics.simulated_escalations == 1


def test_every_unmeasurable_metric_gives_a_reason() -> None:
    assert UNMEASURABLE, "the unmeasurable registry must not be empty"
    for name, reason in UNMEASURABLE.items():
        assert len(reason) > 40, f"{name} is listed as unmeasurable without a real reason"


def test_the_report_names_the_unmeasurable_metrics(repo: Repository) -> None:
    report = collect(repo).report()
    assert "not measurable in this build" in report
    for name in UNMEASURABLE:
        assert name in report
