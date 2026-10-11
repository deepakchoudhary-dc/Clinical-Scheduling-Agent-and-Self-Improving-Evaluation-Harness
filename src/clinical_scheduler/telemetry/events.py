"""Operational metrics (``plan.md`` section 15.2).

**Derived from the audit trail, not instrumented separately.** The audit table
already records every tool invocation and outcome with an event type, a policy
version, a trace id and a timestamp, so a parallel metrics subsystem would be a
second source of truth that could drift out of agreement with the first. This
module reads the one that already exists.

Metrics the trail cannot support are **named with what they would need** rather
than omitted, for the same reason the scorecard names its unmeasured metrics: an
absent metric reads as a healthy one, and "we never measured that" and "we measured
it and it was fine" are different claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from clinical_scheduler.db.repository import Repository

#: Metrics section 15.2 names that this build cannot compute, with the input each
#: would need. Printed on every report.
UNMEASURABLE: dict[str, str] = {
    "stale_availability_rate": (
        "needs a demand signal -- how often a presented slot was no longer bookable "
        "when the patient chose it. The run record captures the presented set but "
        "not the refusal, so the denominator does not exist"
    ),
    "disparity_slices": (
        "needs demographic slices plus a minimum-sample rule (plan.md 12.6). No "
        "slice attribute is collected, deliberately: collecting one to measure "
        "fairness creates the dataset that makes the measurement sensitive"
    ),
    "latency_p50_p95": (
        "measured per run from the transport's own timings and reported on the "
        "scorecard; the audit trail deliberately does not carry latency"
    ),
    "rate_limit_events": (
        "no rate limiter exists (threat-model.md lists it as an absent control), so "
        "there is nothing to count"
    ),
    "judge_errors": (
        "the judge is human (eval/review.py); a model-as-judge is future work, so "
        "there are no judge errors to record"
    ),
    "tool_errors": (
        "tool failures are visible as FAILURE envelopes in the audit payload, and "
        "are counted below as failed_operations; a per-tool breakdown needs the "
        "tool name denormalised out of the payload"
    ),
}


@dataclass(frozen=True)
class OperationalMetrics:
    """A snapshot of section 15.2, as far as the audit trail supports it."""

    total_events: int
    events_by_type: dict[str, int]
    failed_operations: int
    human_handoffs: int
    simulated_escalations: int
    leases_held: int
    leases_expired: int
    appointments_by_status: dict[str, int]
    policy_versions: tuple[str, ...]
    unmeasurable: dict[str, str] = field(default_factory=lambda: dict(UNMEASURABLE))

    def report(self) -> str:
        lines = [
            f"audit events: {self.total_events}",
            f"policy versions seen: {list(self.policy_versions) or 'none'}",
            "",
            "events by type",
        ]
        if self.events_by_type:
            for event_type, count in sorted(
                self.events_by_type.items(), key=lambda kv: (-kv[1], kv[0])
            ):
                lines.append(f"  {count:>6}  {event_type}")
        else:
            lines.append("  none recorded")
        lines.append("")

        lines.append("operational counters")
        lines.append(f"  failed operations      : {self.failed_operations}")
        lines.append(f"  human handoffs         : {self.human_handoffs}")
        lines.append(f"  simulated escalations  : {self.simulated_escalations}")
        lines.append(f"  leases held            : {self.leases_held}")
        lines.append(f"  leases expired         : {self.leases_expired}")
        lines.append("")

        lines.append("appointments by status")
        if self.appointments_by_status:
            for status, count in sorted(self.appointments_by_status.items()):
                lines.append(f"  {count:>6}  {status}")
        else:
            lines.append("  none")
        lines.append("")

        lines.append("not measurable in this build (with the input each needs)")
        for name, reason in self.unmeasurable.items():
            lines.append(f"  {name}: {reason}")
        return "\n".join(lines)


def _count(repo: Repository, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = repo.query_one(sql, params)
    return int(row["n"]) if row else 0


def collect(repo: Repository) -> OperationalMetrics:
    """Read the operational metrics out of a database."""
    events_by_type = {
        str(row["event_type"]): int(row["n"])
        for row in repo.query(
            "SELECT event_type, COUNT(*) AS n FROM audit_events GROUP BY event_type"
        )
    }

    # A tool failure is recorded as a FAILURE envelope in the audit payload. The
    # match is on the serialised payload because the envelope status is not a
    # column; it is exact rather than a substring guess, since the executor writes
    # canonical JSON.
    failed = _count(
        repo,
        "SELECT COUNT(*) AS n FROM audit_events WHERE payload_json LIKE ?",
        ('%"status":"FAILURE"%',),
    )

    policy_versions = tuple(
        str(row["policy_version"])
        for row in repo.query(
            "SELECT DISTINCT policy_version FROM audit_events ORDER BY policy_version"
        )
    )

    now = datetime.now(timezone.utc).isoformat()
    return OperationalMetrics(
        total_events=_count(repo, "SELECT COUNT(*) AS n FROM audit_events"),
        events_by_type=events_by_type,
        failed_operations=failed,
        human_handoffs=_count(repo, "SELECT COUNT(*) AS n FROM handoffs"),
        simulated_escalations=_count(repo, "SELECT COUNT(*) AS n FROM simulated_escalations"),
        leases_held=_count(repo, "SELECT COUNT(*) AS n FROM slot_leases"),
        leases_expired=_count(
            repo, "SELECT COUNT(*) AS n FROM slot_leases WHERE expires_at < ?", (now,)
        ),
        appointments_by_status={
            str(row["status"]): int(row["n"])
            for row in repo.query(
                "SELECT status, COUNT(*) AS n FROM appointments GROUP BY status"
            )
        },
        policy_versions=policy_versions,
    )


__all__ = ["UNMEASURABLE", "OperationalMetrics", "collect"]
