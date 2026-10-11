"""RCA package: what is known about a failure, and what is only hypothesised.

The distinction this module exists to preserve is between **observation** and
**hypothesis**. A run produces a large amount of raw material -- traces, tool
arguments, database snapshots, screening records, transcript spans -- and it is
tempting to let a model summarise it into a paragraph that reads like an
explanation. That paragraph would be the most-quoted artifact in the repository
and the least checkable.

So the package keeps the two apart structurally. Everything under
``observed`` is copied from the run and is verifiable by re-reading the trace.
Everything under ``hypothesis`` is a claim, is labelled as one, and must name the
alternatives that were considered. A reviewer who disagrees with the hypothesis
can still use the observations, which is not true of a narrative.

The package also carries the two fields that make a failure *actionable* rather
than merely described: ``first_divergence`` -- the earliest point at which the
run departed from what the scenario required -- and ``minimal_reproducer``, which
is the shortest input that still produces the failure. A failure without a
reproducer cannot be shown to be fixed.

PHI handling: the package is built from a :class:`RunRecord`, whose assistant
messages may contain patient-visible text. :meth:`EvidencePackage.redacted`
rewrites the transcript -- the one field that can carry free text a patient wrote
-- and :meth:`EvidencePackage.to_dict` serialises that redacted view, so a package
written to disk is subject to the same redaction as any other artifact. (An
earlier version of this docstring claimed that and ``to_dict`` did not do it; the
2026-10-08 static audit caught the gap, and the code was corrected rather than the
claim.)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from clinical_scheduler.improvement.taxonomy import (
    FailureClass,
    FailureClassification,
    PatchLayer,
)
from clinical_scheduler.eval.oracles import Finding, OracleResult, RunRecord

#: Redaction marker. Kept identical to the telemetry layer's so a reader of
#: either artifact sees one convention.
REDACTED = "[REDACTED]"


@dataclass(frozen=True)
class Divergence:
    """The earliest point at which the run departed from the requirement.

    "Earliest" matters more than "clearest". A later symptom is usually easier to
    describe -- a wrong sentence, a missing row -- but patching at the symptom
    leaves the cause alive. Recording the first divergence is what makes the
    classification testable: a hypothesis that does not explain the first
    divergence is a hypothesis about something else.
    """

    turn_index: int | None
    #: Which channel diverged: ``route``, ``tool_offer``, ``tool_execution``,
    #: ``state``, ``database``, ``text`` or ``audit``.
    channel: str
    expected: str
    observed: str
    #: The oracle that noticed, for cross-reference.
    oracle_id: str
    detail: str


@dataclass
class EvidencePackage:
    """The RCA package for one failed scenario.

    ``observed`` is copied, ``hypothesis`` is claimed. The type of each field
    says which it is, so a reader never has to guess, and a serialiser never has
    to decide.
    """

    package_id: str
    scenario_id: str
    scenario_title: str
    scenario_severity: str
    scenario_rationale: str
    key_assertion: str
    #: What the scenario requires, stated from the scenario itself.
    expected_outcome: str
    #: What happened, in one line, derived from the findings.
    actual_outcome: str

    # -- observed (copied from the run; re-derivable) -------------------------
    first_divergence: Divergence | None
    all_findings: list[dict[str, Any]]
    state_before: str
    state_after: str
    state_trail: list[dict[str, Any]]
    #: Tool calls with the arguments the executor saw, redacted.
    tool_calls: list[dict[str, Any]]
    #: The tools the broker actually offered, per turn. This is the evidence the
    #: INV-02 class of finding rests on, so it is copied verbatim.
    offered_tools: list[dict[str, Any]]
    #: Screening records, so the route decision can be read from evidence.
    screening: list[dict[str, Any]] = field(default_factory=list)
    #: Database delta: only the tables whose row counts changed.
    db_delta: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Patient-visible replies. Rewritten at serialisation by ``redacted()``.
    transcript: list[str] = field(default_factory=list)
    #: Policy version in force, and the rule ids that fired.
    policy_version: str = ""
    fired_rule_ids: list[str] = field(default_factory=list)

    # -- hypothesis (claimed; must name alternatives) -------------------------
    classification: FailureClassification | None = None
    causal_hypothesis: str = ""
    alternatives_considered: list[str] = field(default_factory=list)
    owner: str = ""
    proposed_change_surface: str = ""
    proposed_layer: PatchLayer | None = None
    minimal_reproducer: str = ""
    new_regression_tests: list[str] = field(default_factory=list)

    #: Whether the failure is reproducible from the reproducer alone. False means
    #: the classification rests on a single observation and the acceptance gate
    #: should treat it accordingly.
    reproducible: bool = False

    def redacted(self) -> "EvidencePackage":
        """Return a copy with patient-visible text replaced by placeholders.

        Only the transcript is rewritten. The findings, tool arguments and
        database deltas carry identifiers rather than clinical narrative, and
        stripping them would remove the evidence the package exists to carry --
        so the redaction is applied to the one field that can contain free text a
        patient wrote.
        """
        copy = EvidencePackage(**{**self.__dict__})
        copy.transcript = [REDACTED if _looks_like_free_text(line) else line for line in self.transcript]
        copy.package_id = self.package_id
        return copy

    def to_dict(self) -> dict[str, Any]:
        # Serialise the *redacted* view. ``redacted()`` rewrites only the
        # transcript, so this is where the PHI control on the package actually
        # takes effect; the earlier version of this method serialised ``self``
        # directly, which made the module docstring's claim false.
        payload = asdict(self.redacted())
        if self.classification is not None:
            payload["classification"] = {
                "failure_class": self.classification.failure_class.value,
                "required_layer": self.classification.required_layer.value,
                "layer_reason": self.classification.layer_reason,
                "evidence": list(self.classification.evidence),
                "counter_hypotheses": list(self.classification.counter_hypotheses),
                "hypothesis": self.classification.hypothesis,
                "implicated_rule_id": self.classification.implicated_rule_id,
            }
        else:
            payload["classification"] = None
        payload["proposed_layer"] = self.proposed_layer.value if self.proposed_layer else None
        if self.first_divergence is not None:
            payload["first_divergence"] = asdict(self.first_divergence)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


def _looks_like_free_text(line: str) -> bool:
    """Whether a line is patient-visible prose that may carry clinical narrative.

    The heuristic is intentionally crude and biased towards redacting: a line of
    six or more whitespace-separated words is treated as prose. An over-redacted
    transcript loses a little review value; an under-redacted one puts patient
    narrative into a committed artifact, and those costs are not comparable.
    """
    if not line:
        return False
    words = line.split()
    return len(words) >= 6


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _state_at(record: RunRecord, index: int) -> tuple[str, str]:
    if not record.state_trail:
        return "", ""
    if index < 0 or index >= len(record.state_trail):
        entry = record.state_trail[-1]
    else:
        entry = record.state_trail[index]
    return str(entry.get("from", "")), str(entry.get("to", ""))


def _db_delta(record: RunRecord) -> dict[str, dict[str, int]]:
    """Row-count changes per table. Only changed tables are reported.

    Reporting every table would bury the one that moved. The delta is what a
    reviewer reads to answer "did anything get written that should not have", and
    a wall of zeros makes that harder, not easier.
    """
    out: dict[str, dict[str, int]] = {}
    keys = set(record.snapshot_before) | set(record.snapshot_after)
    for table in sorted(keys):
        before = len(record.snapshot_before.get(table, []) or [])
        after = len(record.snapshot_after.get(table, []) or [])
        if before != after:
            out[table] = {"before": before, "after": after, "delta": after - before}
    return out


def _finding_to_dict(finding: Finding) -> dict[str, Any]:
    return {
        "oracle_id": finding.oracle_id,
        "layer": finding.layer,
        "severity": finding.severity,
        "turn_index": finding.turn_index,
        "message": finding.message,
        "evidence": finding.evidence,
        "reason_code": finding.reason_code,
        "is_oracle_error": finding.is_oracle_error,
    }


#: Which channel a finding's oracle id belongs to, for the divergence record.
_CHANNEL_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("L1.INV02_MUTATION_OFFERED", "tool_offer"),
    ("L1.INV02_MUTATION_EXECUTED", "tool_execution"),
    ("L1.INV02_BOOKED", "database"),
    ("L1.INV14", "text"),
    ("L1.CROSS_SESSION", "text"),
    ("L1.INV12", "text"),
    ("L1.INV06", "text"),
    ("L1.INV08", "tool_execution"),
    ("L1.INV10", "tool_execution"),
    ("L1.INV01", "route"),
    ("L2.", "database"),
    ("L3.SCREENING", "route"),
    ("L3.TOOL", "tool_offer"),
    ("L3.SCENARIO", "tool_offer"),
    ("L3.MULTIPLE", "tool_execution"),
    ("L4.", "text"),
    ("L0.", "state"),
)


def _channel_for(oracle_id: str) -> str:
    for prefix, channel in _CHANNEL_BY_PREFIX:
        if oracle_id.startswith(prefix):
            return channel
    return "unknown"


def _first_divergence(record: RunRecord, findings: Sequence[Finding]) -> Divergence | None:
    """Pick the earliest divergence, by turn then by channel precedence.

    Channel precedence matters when several findings sit on the same turn. The
    order is the order in which the runtime acts, so the earliest channel in that
    order is the earliest actual divergence: routing is decided first, then the
    tool payload is built, then the tool runs, then the reply is composed, then
    the database is read back. A text finding on turn 0 cannot be the cause of a
    tool-offer finding on turn 0 if the tool payload was built first.
    """
    if not findings:
        return None
    precedence = {
        "route": 0,
        "state": 1,
        "tool_offer": 2,
        "tool_execution": 3,
        "text": 4,
        "database": 5,
        "audit": 6,
        "unknown": 7,
    }

    def sort_key(finding: Finding) -> tuple[int, int, str]:
        turn = finding.turn_index if finding.turn_index is not None else -1
        return (turn, precedence.get(_channel_for(finding.oracle_id), 7), finding.oracle_id)

    chosen = sorted(findings, key=sort_key)[0]
    channel = _channel_for(chosen.oracle_id)

    expected = "a run with no finding on this channel"
    if channel == "route":
        expected = "the verdict the scenario declared for this turn"
    elif channel == "tool_offer":
        expected = "the broker's permitted set for the session's state and tier"
    elif channel == "tool_execution":
        expected = "no execution, or an execution the runtime authorised"
    elif channel == "database":
        expected = "the state the trace reports"
    elif channel == "text":
        expected = "a reply containing only claims a grounded fact authorises"

    return Divergence(
        turn_index=chosen.turn_index,
        channel=channel,
        expected=expected,
        observed=chosen.evidence or chosen.message,
        oracle_id=chosen.oracle_id,
        detail=chosen.message,
    )


def build_package(
    *,
    package_id: str,
    record: RunRecord,
    result: OracleResult,
    classification: FailureClassification | None = None,
    policy_version: str = "",
    fired_rule_ids: Sequence[str] = (),
    minimal_reproducer: str = "",
    reproducible: bool = False,
) -> EvidencePackage:
    """Assemble the RCA package for one failed scenario.

    Only real findings are considered for the divergence. Oracle errors are
    carried in ``all_findings`` so the reader sees them, but they cannot be the
    first divergence: an oracle that raised produced no observation about the
    system, and treating it as one would attribute the run's own brokenness to
    the agent.
    """
    scenario = record.scenario
    findings = result.real_findings
    divergence = _first_divergence(record, findings)

    if findings:
        worst = sorted(findings, key=lambda f: ("S0", "S1", "S2", "S3").index(f.severity))[0]
        actual = f"{len(findings)} finding(s); worst {worst.severity} {worst.oracle_id}: {worst.message}"
    elif result.oracle_errors:
        actual = f"no system finding; {len(result.oracle_errors)} oracle error(s)"
    else:
        actual = "no finding"

    expected = (
        f"{scenario.key_assertion} "
        f"(severity if wrong: {scenario.severity})"
    )

    return EvidencePackage(
        package_id=package_id,
        scenario_id=scenario.scenario_id,
        scenario_title=scenario.title,
        scenario_severity=scenario.severity,
        scenario_rationale=scenario.rationale,
        key_assertion=scenario.key_assertion,
        expected_outcome=expected,
        actual_outcome=actual,
        first_divergence=divergence,
        all_findings=[_finding_to_dict(f) for f in result.findings],
        state_before=_state_at(record, 0)[0],
        state_after=_state_at(record, len(record.state_trail) - 1)[1],
        state_trail=list(record.state_trail),
        tool_calls=[
            {
                "turn_index": call.get("turn_index"),
                "name": call.get("name"),
                "status": call.get("status"),
                "arguments": call.get("arguments"),
            }
            for call in record.tool_calls
        ],
        offered_tools=[
            {"turn_index": index, "tools": sorted(names)}
            for index, names in record.offered_tools
        ],
        screening=[
            {
                "turn_index": s.get("turn_index"),
                "verdict": s.get("verdict"),
                "fired_rule_ids": s.get("fired_rule_ids"),
                "matched_phrases": s.get("matched_phrases"),
                "before_model": s.get("before_model"),
            }
            for s in record.screenings
        ],
        db_delta=_db_delta(record),
        transcript=list(record.assistant_messages),
        policy_version=policy_version,
        fired_rule_ids=list(fired_rule_ids),
        classification=classification,
        causal_hypothesis=classification.hypothesis if classification else "",
        alternatives_considered=list(classification.counter_hypotheses) if classification else [],
        owner=_owner_for(classification),
        proposed_change_surface=(
            classification.required_layer.value if classification else ""
        ),
        proposed_layer=classification.required_layer if classification else None,
        minimal_reproducer=minimal_reproducer,
        new_regression_tests=_suggested_tests(scenario.scenario_id, classification),
        reproducible=reproducible,
    )


#: Which role owns a class of fix. A failure whose owner is unclear is a failure
#: nobody fixes, so the mapping is exhaustive rather than best-effort.
_OWNER_BY_CLASS: Mapping[FailureClass, str] = {
    FailureClass.CLINICAL_POLICY_GAP: "clinical_safety_officer",
    FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE: "clinical_safety_officer",
    FailureClass.SAFETY_SENTINEL_FALSE_POSITIVE: "clinical_safety_officer",
    FailureClass.IDENTITY_AUTHORIZATION_FAILURE: "privacy_legal",
    FailureClass.PRIVACY_DISCLOSURE: "privacy_legal",
    FailureClass.PROMPT_INJECTION_SUCCESS: "privacy_legal",
    FailureClass.STATE_MACHINE_BYPASS: "engineering_lead",
    FailureClass.TOOL_CONTRACT_ERROR: "engineering_lead",
    FailureClass.TRANSACTION_CONCURRENCY_ERROR: "engineering_lead",
    FailureClass.IDEMPOTENCY_ERROR: "engineering_lead",
    FailureClass.TIMEZONE_TIME_ERROR: "engineering_lead",
    FailureClass.RESPONSE_GROUNDING_ERROR: "engineering_lead",
    FailureClass.ACCESSIBILITY_QUALITY_FAILURE: "operations_manager",
    FailureClass.ORACLE_ERROR: "engineering_lead",
    FailureClass.FIXTURE_ERROR: "engineering_lead",
    FailureClass.MODEL_FLAKINESS: "engineering_lead",
    FailureClass.INTEGRATION_OUTAGE: "operations_manager",
}


def _owner_for(classification: FailureClassification | None) -> str:
    if classification is None:
        return "engineering_lead"
    return _OWNER_BY_CLASS.get(classification.failure_class, "engineering_lead")


def _suggested_tests(scenario_id: str, classification: FailureClassification | None) -> list[str]:
    """Name the regression tests the patch is expected to add.

    These are *names*, not implementations. The point is that the patch's own
    record says which test must exist before the patch can be accepted, so a
    patch that fixes a failure without adding a test is refused at validation
    rather than at review.
    """
    if classification is None:
        return []
    tests = [f"tests/regression/{scenario_id}_{classification.failure_class.value.lower()}.py"]
    if classification.failure_class in {
        FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE,
        FailureClass.SAFETY_SENTINEL_FALSE_POSITIVE,
    }:
        # The counterfactual half is not optional. A recall fix without a
        # precision test is how a false negative becomes a false positive.
        tests.append(f"tests/regression/{scenario_id}_counterfactual_negatives.py")
    return tests


__all__ = [
    "Divergence",
    "EvidencePackage",
    "REDACTED",
    "build_package",
]
