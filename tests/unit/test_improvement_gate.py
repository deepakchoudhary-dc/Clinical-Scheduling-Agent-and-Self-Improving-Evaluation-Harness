"""Tests for the acceptance gate.

The gate decides whether a candidate patch enters a demo policy version. It is
the one place in the pipeline that reads the frozen holdout, and the one decision
in the loop that is a *judgement* rather than a measurement.

The property under test throughout is that the gate is **comparative**: it asks
whether *this change* made things worse, not whether the system is currently
perfect. The absolute question belongs to the eligibility gate in `scoring.py`,
which already answers it. An earlier implementation asked it in both places,
which made the gate refuse every patch for as long as any S0 existed anywhere —
a state the loop exists to get out of, and one it could never leave.

The second property is that a pre-existing defect is recorded as a **caveat**,
not dropped. An acceptance must never be readable as a clean bill of health.
"""

from __future__ import annotations

import pytest

from clinical_scheduler.eval.scoring import Rate, ScenarioScore, Scorecard
from clinical_scheduler.improvement.optimizer import evaluate_gate


def _scenario(
    scenario_id: str,
    *,
    passed: bool,
    severity: str = "S2",
    ran: bool = True,
) -> ScenarioScore:
    return ScenarioScore(
        scenario_id=scenario_id,
        title=f"scenario {scenario_id}",
        family="core",
        declared_severity=severity,
        ran=ran,
        passed=passed,
        highest_severity=None if passed else severity,
        finding_count=0 if passed else 1,
        counts={} if passed else {severity: 1},
        key_assertion="a scenario used by the gate tests",
    )


def _card(
    *,
    s0: int = 0,
    s1: int = 0,
    s2: int = 0,
    passed: int = 10,
    failed: int = 0,
    unexecuted: list[str] | None = None,
    run_id: str = "run_test",
) -> Scorecard:
    scenarios = [_scenario(f"P{index:02d}", passed=True) for index in range(passed)]
    scenarios += [_scenario(f"F{index:02d}", passed=False) for index in range(failed)]
    return Scorecard(
        run_id=run_id,
        manifest_fingerprint="f" * 64,
        eligible=s0 == 0 and s1 == 0,
        vetoes=[],
        scenarios=scenarios,
        rates={"false_escalation_rate": Rate("false_escalation_rate", 1, 20)},
        severity_counts={"S0": s0, "S1": s1, "S2": s2, "S3": 0},
        unexecuted=list(unexecuted or []),
        oracle_error_scenarios=[],
    )


def _gate(**overrides):
    """A gate evaluation that passes every check, with overrides applied."""
    fields = {
        "patch_valid": True,
        "baseline": _card(),
        "candidate": _card(),
        "holdout_baseline": _card(run_id="holdout_before"),
        "holdout_candidate": _card(run_id="holdout_after"),
        "target_fixed": True,
    }
    fields.update(overrides)
    return evaluate_gate(**fields)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_a_clean_improvement_is_accepted() -> None:
    decision = _gate()
    assert decision.accepted, decision.report()
    assert "target_reproducer_passes" in decision.checks_passed
    assert "holdout_not_worse" in " ".join(decision.checks_passed) or any(
        check.startswith("holdout_not_worse") for check in decision.checks_passed
    )
    assert not decision.caveats


# ---------------------------------------------------------------------------
# Hard vetoes
# ---------------------------------------------------------------------------


def test_a_patch_that_fails_validation_is_rejected_without_measuring() -> None:
    """A patch that is not well-formed cannot be accepted on its measurements.

    The measurements would be about a policy that could never exist.
    """
    decision = _gate(patch_valid=False)
    assert not decision.accepted
    assert any("static validation" in reason for reason in decision.reasons)


def test_a_patch_that_does_not_fix_its_target_is_rejected() -> None:
    """The patch must fix the failure it was generated for.

    This is the check that stops a loop from reporting progress by moving a
    different number.
    """
    decision = _gate(target_fixed=False)
    assert not decision.accepted
    assert any("target reproducer still fails" in reason for reason in decision.reasons)


def test_a_run_with_unexecuted_scenarios_is_not_a_passing_run() -> None:
    decision = _gate(candidate=_card(unexecuted=["ADV-11"]))
    assert not decision.accepted
    assert any("unexecuted" in reason for reason in decision.reasons)


def test_a_new_development_failure_vetoes() -> None:
    """A scenario that passed and now fails is a regression, whatever else improved."""
    baseline = _card(passed=10, failed=0)
    candidate = _card(passed=9, failed=1)
    # Make the failing scenario one that previously passed.
    candidate.scenarios[9] = _scenario("P09", passed=False)
    decision = _gate(baseline=baseline, candidate=candidate)
    assert not decision.accepted
    assert any("regressed scenarios" in reason for reason in decision.reasons)


def test_a_missing_holdout_run_vetoes() -> None:
    """A development-set improvement alone is not evidence the change is a real fix."""
    decision = _gate(holdout_baseline=None, holdout_candidate=None)
    assert not decision.accepted
    assert any("holdout was not run" in reason for reason in decision.reasons)


# ---------------------------------------------------------------------------
# Comparative severity: the D-010 regression guard
# ---------------------------------------------------------------------------


def test_a_rise_in_s0_is_rejected() -> None:
    decision = _gate(baseline=_card(s0=0), candidate=_card(s0=1))
    assert not decision.accepted
    assert any("S0 findings rose" in reason for reason in decision.reasons)


def test_a_rise_in_s1_is_rejected() -> None:
    decision = _gate(baseline=_card(s1=0), candidate=_card(s1=1))
    assert not decision.accepted
    assert any("S1 findings rose" in reason for reason in decision.reasons)


def test_a_preexisting_s0_is_a_caveat_not_a_veto() -> None:
    """The behaviour that makes incremental repair possible at all.

    A pre-existing S0 is not evidence against *this* patch. Rejecting on it would
    make the gate refuse every change while the backlog is non-empty, which is
    precisely the situation the loop exists to clear.
    """
    decision = _gate(baseline=_card(s0=1), candidate=_card(s0=1))
    assert decision.accepted, decision.report()
    assert any("pre-existing S0" in caveat for caveat in decision.caveats)


def test_an_improvement_in_s0_is_accepted_and_records_no_caveat_for_it() -> None:
    """Going from two S0 findings to one is progress, not a caveat."""
    decision = _gate(baseline=_card(s0=2), candidate=_card(s0=1))
    assert decision.accepted, decision.report()
    assert any("pre-existing S0" in caveat for caveat in decision.caveats)
    assert not any("rose" in reason for reason in decision.reasons)


def test_a_clean_candidate_records_the_positive_check() -> None:
    decision = _gate(baseline=_card(s0=1), candidate=_card(s0=0))
    assert decision.accepted
    assert "no_s0_on_development_sets" in decision.checks_passed


def test_a_preexisting_s1_is_a_caveat_not_a_veto() -> None:
    decision = _gate(baseline=_card(s1=1), candidate=_card(s1=1))
    assert decision.accepted, decision.report()
    assert any("pre-existing S1" in caveat for caveat in decision.caveats)


# ---------------------------------------------------------------------------
# The holdout, on the same comparative footing
# ---------------------------------------------------------------------------


def test_a_holdout_severity_rise_vetoes_even_when_the_pass_rate_holds() -> None:
    """The exact case that rejected the first real patch.

    A patch can leave the holdout pass rate unchanged while making a failing
    scenario fail *worse*. A gate that compared only pass rates would accept it,
    which is why severity is compared separately.
    """
    decision = _gate(
        holdout_baseline=_card(s1=2, run_id="holdout_before"),
        holdout_candidate=_card(s1=4, run_id="holdout_after"),
    )
    assert not decision.accepted
    assert any("introduces S1 findings on the holdout" in reason for reason in decision.reasons)


def test_a_preexisting_holdout_s0_is_a_caveat_not_a_veto() -> None:
    """The loop is forbidden to look at the holdout, so it cannot repair it.

    Vetoing on a pre-existing holdout S0 would block every future patch forever
    on the strength of a defect the loop is structurally unable to address.
    """
    decision = _gate(
        holdout_baseline=_card(s0=1, run_id="holdout_before"),
        holdout_candidate=_card(s0=1, run_id="holdout_after"),
    )
    assert decision.accepted, decision.report()
    assert any("holdout" in caveat and "pre-existing S0" in caveat for caveat in decision.caveats)


def test_a_holdout_pass_rate_drop_vetoes() -> None:
    decision = _gate(
        holdout_baseline=_card(passed=10, failed=2, run_id="holdout_before"),
        holdout_candidate=_card(passed=8, failed=4, run_id="holdout_after"),
    )
    assert not decision.accepted
    assert any("holdout pass rate fell" in reason for reason in decision.reasons)


# ---------------------------------------------------------------------------
# The report must not be able to hide a caveat
# ---------------------------------------------------------------------------


def test_the_report_prints_caveats_alongside_an_acceptance() -> None:
    """A reader who sees ACCEPTED must also see what was accepted *with*.

    "Accepted into demo policy vX with 1 pre-existing S0 outstanding" is a
    truthful sentence. "Accepted" on its own is not.
    """
    decision = _gate(baseline=_card(s0=1), candidate=_card(s0=1))
    report = decision.report()
    assert "gate: ACCEPTED" in report
    assert "caveat:" in report
    assert "pre-existing S0" in report
