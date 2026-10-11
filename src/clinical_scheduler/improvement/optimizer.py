"""Optimizer: the loop that turns a failure into an accepted policy version.

The lifecycle implemented here is plan.md section 13.4:

    FAIL -> TRIAGE -> RCA -> REPRODUCER -> CANDIDATE CHANGE
         -> STATIC VALIDATION -> TARGET TESTS -> FULL GOLDEN SUITE
         -> HOLDOUT + ADVERSARIAL -> HUMAN REVIEW
         -> ACCEPT / REJECT -> SIGNED ARTIFACT -> ROLLBACK

Not every stage can be mechanical. Two are explicitly *not*:

**Generating the candidate change** is done by a caller-supplied generator, not
by this module. The generator's input is an :class:`~.rca.EvidencePackage` and
its output is a :class:`~.patch_schema.CandidatePatch`. Keeping generation
outside the loop means the loop can be tested with a deterministic generator and
run with a model-backed one, and -- more importantly -- it means a model is never
in a position to apply its own output.

**Human review** is a record, not a gate this module can pass on its own. It
produces an :class:`ApprovalRecord` whose ``simulated`` field is required. In
this prototype approval is always simulated, and every artifact that reports the
outcome says so. The distinction between "accepted into demo policy vX" and
"committed to production" is not wording; it is the difference between a claim
the evidence supports and one it does not.

The acceptance gate reads the holdout set. Nothing else in the pipeline does.
That asymmetry is the design: a loop that can see its own held-out set will
eventually optimise it, and the only reliable defence is not to show it the set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from clinical_scheduler.eval.calibration import CalibrationDelta, ConfusionCounts
from clinical_scheduler.eval.driver import EvaluationDriver
from clinical_scheduler.eval.oracles import OracleResult, run_oracles
from clinical_scheduler.eval.runner import ScenarioOutcome, ScenarioRunner
from clinical_scheduler.eval.scenarios import Scenario, ScenarioSet
from clinical_scheduler.eval.review import JUDGE_VERSION
from clinical_scheduler.eval.scoring import RunManifest, Scorecard, build_manifest
from clinical_scheduler.improvement.patch_schema import CandidatePatch, OpKind
from clinical_scheduler.improvement.rca import EvidencePackage, build_package
from clinical_scheduler.improvement.taxonomy import classify_finding
from clinical_scheduler.improvement.validator import ValidationResult, apply_patch, validate
from clinical_scheduler.policy.engine import PolicyEngine
from clinical_scheduler.tools.executor import TOOL_VERSIONS

#: How much the holdout pass rate is allowed to drop before a patch is rejected.
#: Zero, not a tolerance. A holdout exists to catch a patch that improved the
#: development sets by fitting them; any regression on it is evidence of exactly
#: that, and a tolerance would simply set the price of overfitting.
HOLDOUT_TOLERANCE = 0.0
#: How much the false-escalation rate is allowed to rise. Zero for the same
#: reason: a recall fix that buys its recall with over-triage has traded one harm
#: for another, and the trade is not one this loop may make silently.
FALSE_ESCALATION_TOLERANCE = 0.0


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class ApprovalRecord:
    """The human-review step, recorded.

    ``simulated`` is required and is written into every downstream artifact. A
    prototype that recorded an approval without saying it was simulated would
    invite a reader to treat the resulting version as production-approved, and
    every later claim resting on it would inherit that error.
    """

    patch_id: str
    approver_role: str
    decision: str  # "ACCEPT" | "REJECT"
    simulated: bool
    rationale: str
    decided_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    #: The patch's content hash at the moment of approval. A patch edited after
    #: this point is not the patch that was approved.
    patch_content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "approver_role": self.approver_role,
            "decision": self.decision,
            "simulated": self.simulated,
            "rationale": self.rationale,
            "decided_at": self.decided_at,
            "patch_content_hash": self.patch_content_hash,
        }


@dataclass
class GateDecision:
    """The acceptance gate's verdict, with the reason it reached it."""

    accepted: bool
    reasons: list[str] = field(default_factory=list)
    #: Checks that passed, recorded so a reader sees what was actually verified
    #: rather than only what failed.
    checks_passed: list[str] = field(default_factory=list)
    #: Known defects that are **not** grounds for rejection because this patch
    #: neither caused nor could reach them. Recorded rather than dropped, so an
    #: acceptance is never read as a claim that the system is clean. A reader who
    #: sees "ACCEPTED" must be able to see what was accepted *with*.
    caveats: list[str] = field(default_factory=list)

    def report(self) -> str:
        head = "ACCEPTED" if self.accepted else "REJECTED"
        lines = [f"gate: {head}"]
        for reason in self.reasons:
            lines.append(f"  veto: {reason}")
        for check in self.checks_passed:
            lines.append(f"  pass: {check}")
        for caveat in self.caveats:
            lines.append(f"  caveat: {caveat}")
        return "\n".join(lines)


@dataclass
class RecalibrationEvidence:
    """The recall measurement a pattern change must carry (B4 change control).

    The rule, from ``docs/hazard-log.md``'s review triggers and plan.md section
    7.2: a change to the sentinel's pattern set is not reviewable on its own,
    only against the recall evidence from before it. Concretely, both the
    baseline and the candidate bundle are calibrated on the **same** labelled
    corpus, and the delta between them travels with the acceptance decision.

    ``regressed`` is the load-bearing property: a calibration covering the same
    corpora that shows **more** false negatives under the candidate is a recall
    regression, and the gate vetoes it regardless of every other number.
    """

    previous: ConfusionCounts
    current: ConfusionCounts
    #: Scenario ids the corpus was drawn from, so a reviewer can confirm the two
    #: measurements cover the same observations.
    corpus_scenario_ids: tuple[str, ...]

    @property
    def delta(self) -> CalibrationDelta:
        return CalibrationDelta(previous=self.previous, current=self.current)

    @property
    def regressed(self) -> bool:
        return self.delta.regressed

    def summary(self) -> str:
        d = self.delta
        return (
            f"false_negatives {self.previous.false_negative} -> {self.current.false_negative} "
            f"({d.false_negative_delta:+d}), "
            f"false_positives {self.previous.false_positive} -> {self.current.false_positive} "
            f"({d.false_positive_delta:+d}), "
            f"corpus={len(self.corpus_scenario_ids)} scenarios"
        )


@dataclass
class LoopResult:
    """Everything the loop produced, ready to write as artifacts."""

    target_scenario_id: str
    package: EvidencePackage
    patch: CandidatePatch
    validation: ValidationResult
    gate: GateDecision
    approval: ApprovalRecord
    #: The candidate bundle, present when the patch composed.
    candidate_bundle: Any | None = None
    #: Scorecard from the baseline run over the development sets.
    baseline_scorecard: Scorecard | None = None
    #: Scorecard from the candidate run over the same development sets.
    candidate_scorecard: Scorecard | None = None
    #: Scorecard from the **baseline** run over the holdout. Recorded because the
    #: gate's holdout question is comparative, and a comparison whose left-hand
    #: side is not in the artifact set cannot be re-derived by a reviewer -- they
    #: would have to re-run the loop to check the claim the gate made.
    holdout_baseline_scorecard: Scorecard | None = None
    #: Scorecard from the candidate run over the holdout. Read only by the gate.
    holdout_scorecard: Scorecard | None = None
    #: Which scenario ids changed verdict between baseline and candidate.
    fixed_scenarios: list[str] = field(default_factory=list)
    regressed_scenarios: list[str] = field(default_factory=list)
    #: Recall evidence for a pattern change (B4). ``None`` when the patch added
    #: no trigger pattern, in which case the recall question does not arise.
    recalibration: RecalibrationEvidence | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_scenario_id": self.target_scenario_id,
            "patch": self.patch.model_dump(mode="json"),
            "patch_content_hash": self.patch.content_hash(),
            "validation": {
                "valid": self.validation.valid,
                "issues": [
                    {"check": i.check, "detail": i.detail, "blocking": i.blocking}
                    for i in self.validation.issues
                ],
                "notes": self.validation.notes,
            },
            "gate": {
                "accepted": self.gate.accepted,
                "reasons": self.gate.reasons,
                "checks_passed": self.gate.checks_passed,
                "caveats": self.gate.caveats,
            },
            "approval": self.approval.to_dict(),
            "fixed_scenarios": self.fixed_scenarios,
            "regressed_scenarios": self.regressed_scenarios,
            "recalibration": {
                "summary": self.recalibration.summary(),
                "regressed": self.recalibration.regressed,
                "previous_counts": {
                    "true_positive": self.recalibration.previous.true_positive,
                    "false_negative": self.recalibration.previous.false_negative,
                    "true_negative": self.recalibration.previous.true_negative,
                    "false_positive": self.recalibration.previous.false_positive,
                },
                "current_counts": {
                    "true_positive": self.recalibration.current.true_positive,
                    "false_negative": self.recalibration.current.false_negative,
                    "true_negative": self.recalibration.current.true_negative,
                    "false_positive": self.recalibration.current.false_positive,
                },
                "corpus_scenario_ids": list(self.recalibration.corpus_scenario_ids),
            }
            if self.recalibration is not None
            else None,
            "notes": self.notes,
            "rca_package_id": self.package.package_id,
        }


# ---------------------------------------------------------------------------
# Triage and RCA
# ---------------------------------------------------------------------------


def triage(
    outcomes: Sequence[ScenarioOutcome], results: Sequence[OracleResult]
) -> list[tuple[ScenarioOutcome, OracleResult]]:
    """Return the failures, most severe first.

    Ordering matters for the loop: a generator handed the worst failure first
    fixes the thing that matters most with the budget it has. Ties break on
    scenario id so the ordering is deterministic across runs, which is what makes
    two loop runs comparable.
    """
    by_id = {o.scenario.scenario_id: o for o in outcomes}
    failures: list[tuple[ScenarioOutcome, OracleResult]] = []
    for result in results:
        if result.passed:
            continue
        outcome = by_id.get(result.scenario_id)
        if outcome is not None:
            failures.append((outcome, result))

    severity_rank = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}

    def key(item: tuple[ScenarioOutcome, OracleResult]) -> tuple[int, str]:
        outcome, result = item
        worst = result.highest_severity or outcome.scenario.severity
        return (severity_rank.get(worst, 9), outcome.scenario.scenario_id)

    return sorted(failures, key=key)


def build_rca(
    *,
    outcome: ScenarioOutcome,
    result: OracleResult,
    policy_version: str,
) -> EvidencePackage:
    """Build the RCA package for one failure, with a classification per finding.

    The classification attached to the package is the one for the *first
    divergence*, not the worst finding. That choice follows from what the
    package is for: the classification determines which layer gets patched, and
    patching the layer of the worst symptom rather than the earliest cause is how
    an improvement loop produces changes that make the score move without fixing
    anything.
    """
    record = outcome.record
    real = result.real_findings
    classification = None

    if real:
        expected_verdict = _expected_verdict_for(record, real[0].turn_index)
        observed_verdict = _observed_verdict_at(record, real[0].turn_index)
        blamed_rule = _blamed_rule(record, real)
        classification = classify_finding(
            oracle_id=real[0].oracle_id,
            scenario_id=record.scenario_id,
            evidence=real[0].evidence,
            reason_code=real[0].reason_code,
            expected_verdict=expected_verdict,
            observed_verdict=observed_verdict,
            implicated_rule_id=blamed_rule,
        )

    fired = sorted(
        {
            rule_id
            for screening in record.screenings
            for rule_id in (screening.get("fired_rule_ids") or [])
        }
    )

    return build_package(
        package_id=f"RCA-{record.scenario_id}",
        record=record,
        result=result,
        classification=classification,
        policy_version=policy_version,
        fired_rule_ids=fired,
        minimal_reproducer=_minimal_reproducer(record),
        reproducible=True,
    )


def _expected_verdict_for(record: Any, turn_index: int | None) -> str | None:
    """The verdict the scenario declared for a turn, or None if it declared none."""
    if turn_index is None:
        return None
    routes = record.scenario.expect_routes
    if 0 <= turn_index < len(routes):
        return routes[turn_index].verdict
    return None


def _observed_verdict_at(record: Any, turn_index: int | None) -> str | None:
    for screening in record.screenings:
        if screening.get("turn_index") == turn_index:
            return str(screening.get("verdict"))
    return None


def _blamed_rule(record: Any, findings: Sequence[Any]) -> str | None:
    """The rule a finding implicates.

    Read from the scenario's declared ``policy_rule`` first, because that is the
    rule the scenario was written to exercise. Falling back to the fired rules
    would blame whichever rule happened to fire, which is not always the rule
    whose pattern set is wrong -- a negation gap shows up as an extra rule firing
    rather than as the intended one.
    """
    if record.scenario.policy_rule:
        return record.scenario.policy_rule
    for finding in findings:
        for token in str(finding.evidence).split():
            if "-" in token and token[:1].isupper() and token[1:].replace("-", "").isdigit():
                return token
    return None


def _minimal_reproducer(record: Any) -> str:
    """The shortest turn text that carries the failure.

    Not a search -- a search would need a working model and would not be
    deterministic. This takes the earliest turn whose screening produced a
    verdict, which is the turn the route was decided on, and is the input a human
    would try first when reproducing by hand.
    """
    if not record.scenario.turns:
        return ""
    for screening in sorted(record.screenings, key=lambda s: s.get("turn_index") or 0):
        index = screening.get("turn_index")
        if isinstance(index, int) and 0 <= index < len(record.scenario.turns):
            return record.scenario.turns[index]
    return record.scenario.turns[0]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def evaluate_gate(
    *,
    patch_valid: bool,
    baseline: Scorecard | None,
    candidate: Scorecard | None,
    holdout_baseline: Scorecard | None,
    holdout_candidate: Scorecard | None,
    target_fixed: bool,
    recalibration: RecalibrationEvidence | None = None,
    patch_touches_patterns: bool = False,
) -> GateDecision:
    """The acceptance rule (plan.md section 13.5), as a function of measurements.

    Every condition plan.md lists appears here as a named check with a stated
    reason, with two deliberate exceptions. **Multi-seed stability is not
    established here**: it requires re-running the evaluation with several seeds,
    which is the caller's responsibility, and a single run cannot assert it.
    **Minimality** is a validation check rather than a gate check. ``checks_passed``
    therefore records only the conditions this run actually established, and this
    function does not claim the two it cannot see. (An earlier version of this
    docstring said both were "asserted as present in the checks_passed list"; the
    2026-10-08 static audit showed no such assertion existed.)
    """
    decision = GateDecision(accepted=False)

    # -- B4 change control: a pattern edit carries its recall evidence --------
    # docs/hazard-log.md's review triggers say a pattern edit must re-run the
    # labelled recall corpus and attach the delta. Until now that rule existed
    # only as prose. Making it a gate check is what turns it into a control: a
    # patch that adds a trigger pattern without measured recalibration evidence
    # is refused, and a patch whose evidence shows more missed escalations is
    # refused on the same terms as any other S0 -- it is a change that makes
    # the system miss more emergencies.
    #
    # This block runs BEFORE the ``patch_valid`` short-circuit, deliberately:
    # the recall question must be asked even of a patch that fails other checks,
    # or a malformed patch could carry its real recall damage past the gate on
    # an unrelated technicality. A gate that records only its first objection
    # is a gate that hides the rest.
    if patch_touches_patterns:
        if recalibration is None:
            decision.reasons.append(
                "the patch alters the sentinel's pattern set but carries no "
                "recalibration evidence; recall must be measured on the same corpus "
                "before and after, not asserted"
            )
        else:
            decision.caveats.append(f"recall measurement: {recalibration.summary()}")
            if recalibration.delta.scale_mismatch:
                decision.reasons.append(
                    "recalibration covers different numbers of observations before "
                    f"and after ({recalibration.previous.total} -> "
                    f"{recalibration.current.total}); a delta across different "
                    "corpora is not evidence about the pattern change"
                )
            elif recalibration.regressed:
                decision.reasons.append(
                    "recalibration shows a recall regression: "
                    f"{recalibration.summary()}; a change that misses more "
                    "escalations is not acceptable on the strength of any other "
                    "improvement"
                )
            else:
                decision.checks_passed.append("recalibration_no_false_negative_increase")

    if not patch_valid:
        decision.reasons.append(
            "static validation failed; a patch that is not well-formed cannot be accepted "
            "on the strength of its measurements"
        )
        return decision
    decision.checks_passed.append("static_validation")

    if not target_fixed:
        decision.reasons.append(
            "the target reproducer still fails; the patch did not fix the failure it "
            "was generated for"
        )
    else:
        decision.checks_passed.append("target_reproducer_passes")

    # A run that did not execute is not a passing run.
    for label, card in (("baseline", baseline), ("candidate", candidate)):
        if card is None:
            continue
        if card.unexecuted:
            decision.reasons.append(
                f"{label} run has unexecuted scenarios ({sorted(card.unexecuted)}); "
                "coverage is unknown and the comparison is not valid"
            )

    if candidate is not None:
        # S0/S1 on the development sets: comparative, not absolute.
        #
        # The absolute question -- "is this build releasable?" -- belongs to the
        # eligibility gate (plan.md 11.3), which already answers it: the baseline
        # run reports ELIGIBILITY: FAIL. Asking it a second time here would make
        # this gate refuse every patch for as long as any S0 exists anywhere,
        # which is precisely the situation a repair loop exists to get out of, and
        # it would make the loop's first successful repair impossible rather than
        # merely delayed.
        #
        # The acceptance question is comparative -- did *this change* make things
        # worse? -- and every other check in this function is already asked that
        # way: the golden-suite check looks for scenarios that newly fail, and the
        # holdout check compares pass rates rather than requiring a fixed number.
        # These two checks were the outliers.
        #
        # A pre-existing S0/S1 is not ignored. It is recorded as a caveat, so a
        # reader who sees ACCEPTED also sees what remains outstanding, and the
        # acceptance is never mistaken for a clean bill of health.
        if baseline is not None:
            for severity in ("S0", "S1"):
                before = baseline.severity_counts.get(severity, 0)
                after = candidate.severity_counts.get(severity, 0)
                if after > before:
                    decision.reasons.append(
                        f"{severity} findings rose from {before} to {after} on the "
                        "development sets; the change made the system worse"
                    )
                elif after:
                    decision.caveats.append(
                        f"{after} pre-existing {severity} finding(s) remain on the "
                        "development sets; this patch neither caused nor addressed "
                        "them, and accepting it is not a claim that they are gone"
                    )
                else:
                    decision.checks_passed.append(f"no_{severity.lower()}_on_development_sets")

    # Regression on the development sets: a scenario that passed and now fails.
    if baseline is not None and candidate is not None:
        before = {s.scenario_id: s.passed for s in baseline.scenarios if s.ran}
        after = {s.scenario_id: s.passed for s in candidate.scenarios if s.ran}
        regressed = sorted(
            sid for sid, passed in before.items() if passed and not after.get(sid, passed)
        )
        if regressed:
            decision.reasons.append(
                f"regressed scenarios that previously passed: {regressed}"
            )
        else:
            decision.checks_passed.append("no_new_development_failures")

        # The false-escalation rate is the specific rate a recall fix threatens.
        base_rate = baseline.rates.get("false_escalation_rate")
        cand_rate = candidate.rates.get("false_escalation_rate")
        if base_rate and cand_rate and base_rate.value is not None and cand_rate.value is not None:
            if cand_rate.value > base_rate.value + FALSE_ESCALATION_TOLERANCE:
                decision.reasons.append(
                    f"false_escalation_rate rose from {base_rate.numerator}/{base_rate.denominator} "
                    f"to {cand_rate.numerator}/{cand_rate.denominator}; the fix bought recall "
                    "with over-triage"
                )
            else:
                decision.checks_passed.append("false_escalation_within_budget")

    # The holdout. This is the only place in the pipeline that reads it.
    if holdout_baseline is None or holdout_candidate is None:
        decision.reasons.append(
            "holdout was not run; generalisation is unmeasured and a development-set "
            "improvement alone is not evidence that the change is a real fix"
        )
    else:
        before_pass = sum(1 for s in holdout_baseline.scenarios if s.passed)
        after_pass = sum(1 for s in holdout_candidate.scenarios if s.passed)
        total = len(holdout_candidate.scenarios) or 1
        delta = (after_pass - before_pass) / total
        if delta < -HOLDOUT_TOLERANCE:
            decision.reasons.append(
                f"holdout pass rate fell by {abs(delta):.1%} "
                f"({before_pass}/{total} -> {after_pass}/{total}); the change does not "
                "generalise beyond the scenarios it was generated against"
            )
        else:
            decision.checks_passed.append(
                f"holdout_not_worse ({before_pass}/{total} -> {after_pass}/{total})"
            )

        # Severity on the holdout, on the same comparative footing as the
        # development sets. A pre-existing holdout S0 -- one the loop is not
        # allowed to look at, and therefore cannot repair -- must not block every
        # future patch forever; an *increase* must, because that is the signature
        # of a change that generalised in the wrong direction.
        for severity in ("S0", "S1"):
            before = holdout_baseline.severity_counts.get(severity, 0)
            after = holdout_candidate.severity_counts.get(severity, 0)
            if after > before:
                decision.reasons.append(
                    f"the candidate introduces {severity} findings on the holdout "
                    f"({before} -> {after}); a severity increase anywhere vetoes"
                )
            elif after:
                decision.caveats.append(
                    f"{after} pre-existing {severity} finding(s) remain on the "
                    "holdout; the loop cannot see this set, so it could not have "
                    "addressed them"
                )

    decision.accepted = not decision.reasons
    return decision


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@dataclass
class LoopConfig:
    """Everything the loop needs that is not the failure itself."""

    work_dir: Path
    out_dir: Path
    repo_root: Path
    #: Called with an RCA package and expected to return a candidate patch, or
    #: None when it cannot propose one. Kept as a callback so the loop can be
    #: driven deterministically in tests and by a model in practice.
    generator: Callable[[EvidencePackage, PolicyEngine], CandidatePatch | None]
    #: Approval policy. Receives the patch, validation and gate, and decides.
    #: Separate from the gate because approval is a human decision that the gate
    #: informs rather than makes.
    approver: Callable[[CandidatePatch, ValidationResult, GateDecision], ApprovalRecord]
    seed: int | None = 0
    temperature: float = 0.0
    max_turns: int = 12
    notes: Sequence[str] = ()


def run_repair(
    *,
    config: LoopConfig,
    baseline_engine: PolicyEngine,
    dev_sets: Sequence[ScenarioSet],
    holdout_sets: Sequence[ScenarioSet],
    adapter: Any,
    target_scenario_id: str | None = None,
) -> LoopResult | None:
    """Run one full repair cycle and return its artifacts.

    Returns ``None`` when every scenario passes and there is nothing to repair --
    a distinct outcome from a repair that was rejected, and conflating the two
    would make a clean baseline look like a failed repair.

    The order of operations is the lifecycle in plan.md section 13.4, with one
    addition that is not in the plan and is worth stating: the holdout is run
    *twice*, once against the baseline and once against the candidate. A single
    holdout run gives a pass rate but not a comparison, and the gate's question is
    comparative -- did this change make the held-out set worse? Measuring only the
    candidate would answer a question nobody asked.
    """
    dev_scenarios = [s for scenario_set in dev_sets for s in scenario_set.scenarios]
    holdout_scenarios = [s for scenario_set in holdout_sets for s in scenario_set.scenarios]

    driver = EvaluationDriver(
        work_dir=config.work_dir,
        out_dir=config.out_dir,
        repo_root=config.repo_root,
        policy_loader=lambda: baseline_engine,
        seed=config.seed,
        temperature=config.temperature,
        max_turns=config.max_turns,
    )

    # -- baseline over the development sets ---------------------------------
    _, baseline_card, baseline_outcomes, baseline_results = driver.run(
        dev_scenarios,
        scenario_set_versions=_versions(dev_sets),
        holdout_included=False,
        adapter=adapter,
        write=False,
        notes=[*config.notes, "repair loop: baseline over development sets"],
    )

    failures = triage(baseline_outcomes, baseline_results)
    if not failures:
        return None

    if target_scenario_id is not None:
        failures = [f for f in failures if f[0].scenario.scenario_id == target_scenario_id]
        if not failures:
            return None

    outcome, result = failures[0]
    package = build_rca(
        outcome=outcome, result=result, policy_version=baseline_engine.bundle.policy_version
    )

    # -- candidate change ---------------------------------------------------
    patch = config.generator(package, baseline_engine)
    if patch is None:
        return None

    validation = validate(patch, baseline_engine.bundle)
    if not validation.valid:
        # A candidate that fails validation produces a rejected result rather than
        # an exception: the rejection and its reasons are the useful output, and
        # raising would discard the package that explains why it was proposed.
        return LoopResult(
            target_scenario_id=outcome.scenario.scenario_id,
            package=package,
            patch=patch,
            validation=validation,
            gate=GateDecision(
                accepted=False,
                reasons=["static validation failed before any run was attempted"],
            ),
            approval=config.approver(
                patch,
                validation,
                GateDecision(accepted=False, reasons=["static validation failed"]),
            ),
            baseline_scorecard=baseline_card,
            notes=["candidate rejected at validation; no evaluation was run"],
        )

    candidate_engine = PolicyEngine(validation.composed_bundle)  # type: ignore[arg-type]
    candidate_driver = EvaluationDriver(
        work_dir=config.work_dir / "candidate",
        out_dir=config.out_dir,
        repo_root=config.repo_root,
        policy_loader=lambda: candidate_engine,
        seed=config.seed,
        temperature=config.temperature,
        max_turns=config.max_turns,
    )

    # -- candidate over the development sets --------------------------------
    _, candidate_card, candidate_outcomes, candidate_results = candidate_driver.run(
        dev_scenarios,
        scenario_set_versions=_versions(dev_sets),
        holdout_included=False,
        adapter=adapter,
        write=False,
        notes=[*config.notes, f"repair loop: candidate {patch.patch_id} over development sets"],
    )

    # -- holdout, both sides ------------------------------------------------
    holdout_runner = ScenarioRunner(
        adapter=adapter,
        work_dir=config.work_dir / "holdout_baseline",
        policy_loader=lambda: baseline_engine,
        seed=config.seed,
        temperature=config.temperature,
        max_turns=config.max_turns,
    )
    holdout_baseline = _score_with(
        holdout_runner, holdout_scenarios, holdout_sets, baseline_engine, config, "baseline"
    )

    holdout_candidate_runner = ScenarioRunner(
        adapter=adapter,
        work_dir=config.work_dir / "holdout_candidate",
        policy_loader=lambda: candidate_engine,
        seed=config.seed,
        temperature=config.temperature,
        max_turns=config.max_turns,
    )
    holdout_candidate = _score_with(
        holdout_candidate_runner, holdout_scenarios, holdout_sets, candidate_engine, config, "candidate"
    )

    # -- B4: recalibrate the sentinel when the patch touches patterns --------
    # The corpus is the development sets measured before and after, from the
    # runs the gate already has. Computing it here rather than letting the
    # candidate pipeline produce it is what makes the two sides cover the same
    # observations -- a delta across different corpora is not evidence about the
    # pattern change, so the corpus is held fixed by construction.
    #
    # The trigger for the measurement is a polarity-preserving judgment, not a
    # enum-membership test: ADD_TRIGGER_PATTERN with no polarity defaults to
    # POSITIVE (won't lose recall), but an explicit NEGATION polarity REMOVES the
    # phrase from the escalating set. A NEGATION patch that silences a real
    # emergency phrase loses recall with no false-negative cell to show for it,
    # so it must be handled by the same rule. D-044.
    touches_patterns = any(
        op.op is OpKind.ADD_TRIGGER_PATTERN
        and (op.polarity is None or op.polarity != "NEGATION")
        for op in patch.operations
    )
    suppresses_patterns = any(
        op.op is OpKind.ADD_TRIGGER_PATTERN and op.polarity == "NEGATION"
        for op in patch.operations
    )
    recalibration: RecalibrationEvidence | None = None
    if touches_patterns or suppresses_patterns:
        from clinical_scheduler.eval.calibration import calibrate_from_outcomes

        recalibration = RecalibrationEvidence(
            previous=calibrate_from_outcomes(baseline_outcomes),
            current=calibrate_from_outcomes(candidate_outcomes),
            corpus_scenario_ids=tuple(sorted(s.scenario.scenario_id for s in baseline_outcomes)),
        )

    # -- target reproducer --------------------------------------------------
    fixed = _scenario_fixed(
        target_id=outcome.scenario.scenario_id,
        before=baseline_results,
        after=candidate_results,
    )

    gate = evaluate_gate(
        patch_valid=True,
        baseline=baseline_card,
        candidate=candidate_card,
        holdout_baseline=holdout_baseline,
        holdout_candidate=holdout_candidate,
        target_fixed=fixed,
        recalibration=recalibration,
        patch_touches_patterns=touches_patterns or suppresses_patterns,
    )

    approval = config.approver(patch, validation, gate)

    before_by_id = {s.scenario_id: s.passed for s in baseline_card.scenarios}
    after_by_id = {s.scenario_id: s.passed for s in candidate_card.scenarios}
    fixed_ids = sorted(
        sid for sid, was in before_by_id.items() if not was and after_by_id.get(sid)
    )
    regressed_ids = sorted(
        sid for sid, was in before_by_id.items() if was and not after_by_id.get(sid, True)
    )

    return LoopResult(
        target_scenario_id=outcome.scenario.scenario_id,
        package=package,
        patch=patch,
        validation=validation,
        gate=gate,
        approval=approval,
        candidate_bundle=validation.composed_bundle,
        baseline_scorecard=baseline_card,
        candidate_scorecard=candidate_card,
        holdout_baseline_scorecard=holdout_baseline,
        holdout_scorecard=holdout_candidate,
        fixed_scenarios=fixed_ids,
        regressed_scenarios=regressed_ids,
        recalibration=recalibration,
        notes=list(config.notes),
    )


def _versions(sets: Sequence[ScenarioSet]) -> dict[str, str]:
    return {s.set_id: s.set_version for s in sets}


def _scenario_fixed(
    *, target_id: str, before: Sequence[OracleResult], after: Sequence[OracleResult]
) -> bool:
    """Whether the target scenario now passes when it previously did not."""
    before_pass = next((r.passed for r in before if r.scenario_id == target_id), False)
    after_pass = next((r.passed for r in after if r.scenario_id == target_id), False)
    return (not before_pass) and after_pass


def _score_with(
    runner: ScenarioRunner,
    scenarios: Sequence[Scenario],
    sets: Sequence[ScenarioSet],
    engine: PolicyEngine,
    config: LoopConfig,
    label: str,
) -> Scorecard:
    """Run a scenario set through a runner and score it.

    Kept separate from ``EvaluationDriver`` because the holdout run has no
    artifacts to write and no manifest of its own -- it is a measurement taken
    inside another run's lifetime, and writing it as a run would put a
    holdout-derived fingerprint into the run directory a reviewer browses.
    """
    from clinical_scheduler.eval.scoring import score_run

    started = datetime.now(timezone.utc)
    policy = engine
    run_id = f"holdout_{label}_{started.strftime('%Y%m%dT%H%M%SZ')}"

    outcomes: list[ScenarioOutcome] = []
    results: list[OracleResult] = []
    for scenario in scenarios:
        outcome = runner.run_scenario(scenario)
        outcomes.append(outcome)
        results.append(run_oracles(outcome.record))

    fixture_hash = next(
        (o.artifacts.fixture_hash for o in outcomes if o.artifacts.fixture_hash), ""
    )
    manifest = build_manifest(
        run_id=run_id,
        started_at=started,
        finished_at=datetime.now(timezone.utc),
        policy_version=policy.bundle.policy_version,
        policy_bundle_hash=policy.bundle_hash(),
        prompt=_prompt_hash(engine),
        tool_versions=dict(TOOL_VERSIONS),
        schema_version="1",
        fixture_hash=fixture_hash,
        model_provider=type(runner.adapter).__name__,
        model_id=getattr(runner.adapter, "model_id", "unknown"),
        model_parameters={
            "seed": config.seed,
            "temperature": config.temperature,
            "max_turns": config.max_turns,
        },
        scenario_set_versions=_versions(sets),
        scenario_ids=[s.scenario_id for s in scenarios],
        holdout_included=True,
        repo_root=config.repo_root,
        # The judge is the Layer-5 human review protocol everywhere else in the
        # run contract; the holdout measurement must carry the same version or
        # the two scorecards are not comparable.
        judge_version=JUDGE_VERSION,
        notes=["holdout measurement taken inside the repair loop; read only by the gate"],
    )
    return score_run(
        run_id=run_id,
        manifest_fingerprint=manifest.fingerprint(),
        outcomes=outcomes,
        oracle_results=results,
    )


def _prompt_hash(engine: PolicyEngine) -> str:
    from clinical_scheduler.agent.prompt import build_system_prompt

    return hashlib.sha256(build_system_prompt(engine).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def write_loop_artifacts(result: LoopResult, out_dir: Path) -> Path:
    """Write the repair-cycle artifacts and a checksum manifest.

    The checksums file is what makes the artifact set tamper-evident: a reviewer
    comparing a reported score against the recorded one needs to know the files
    they are reading are the files that were produced.

    Each file is written from the *same byte string* that is hashed, rather than
    from a text string that is hashed and then re-encoded by the writer. That
    distinction is not pedantic: ``Path.write_text`` translates ``\\n`` to the
    platform line separator, so on Windows a hash taken over the text disagreed
    with the bytes on disk and every checksum in the file failed to verify. A
    tamper-evident manifest that fails verification on the machine that produced
    it is worse than no manifest, because it trains a reviewer to ignore it.
    """
    cycle_dir = out_dir / f"cycle-{result.target_scenario_id}-{result.patch.patch_id}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    writes: dict[str, str] = {}

    def write(name: str, text: str) -> None:
        payload = text.encode("utf-8")
        (cycle_dir / name).write_bytes(payload)
        writes[name] = hashlib.sha256(payload).hexdigest()

    write("rca.json", result.package.to_json())
    write("candidate_patch.json", result.patch.to_json())
    write("validation.txt", result.validation.report())
    write("gate.txt", result.gate.report())
    write("approval.json", json.dumps(result.approval.to_dict(), indent=2, sort_keys=True))
    write("loop_result.json", json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))

    if result.baseline_scorecard is not None:
        write("scorecard_before.txt", result.baseline_scorecard.report())
    if result.candidate_scorecard is not None:
        write("scorecard_after.txt", result.candidate_scorecard.report())
    if result.holdout_baseline_scorecard is not None:
        write("scorecard_holdout_before.txt", result.holdout_baseline_scorecard.report())
    if result.holdout_scorecard is not None:
        write("scorecard_holdout_after.txt", result.holdout_scorecard.report())
    if result.candidate_bundle is not None:
        write("candidate_policy.yaml", _bundle_to_yaml(result.candidate_bundle))

    lines = [f"{digest}  {name}" for name, digest in sorted(writes.items())]
    (cycle_dir / "checksums.sha256").write_bytes(
        ("\n".join(lines) + "\n").encode("utf-8")
    )
    return cycle_dir


def _bundle_to_yaml(bundle: Any) -> str:
    import yaml

    return yaml.safe_dump(bundle.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


__all__ = [
    "ApprovalRecord",
    "FALSE_ESCALATION_TOLERANCE",
    "GateDecision",
    "HOLDOUT_TOLERANCE",
    "LoopConfig",
    "LoopResult",
    "RecalibrationEvidence",
    "build_rca",
    "evaluate_gate",
    "run_repair",
    "triage",
    "write_loop_artifacts",
]
