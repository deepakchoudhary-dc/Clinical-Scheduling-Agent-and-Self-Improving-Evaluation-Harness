"""The B4 change-control check: a pattern edit carries measured recall evidence.

``docs/hazard-log.md``'s review triggers require that any edit to the sentinel's
pattern set re-run the labelled recall corpus and attach the delta. Until the
check tested here existed, that rule was prose. These tests assert the three
decisions the gate must make:

* a pattern patch **without** recalibration evidence is refused;
* a pattern patch whose evidence shows more false negatives is refused, whatever
  else improved -- that is the trade the whole rule exists to refuse;
* a pattern patch with clean evidence records the check as passed, and a patch
  that touches no pattern at all is not asked for evidence it cannot need.
"""

from __future__ import annotations

import types
from typing import Any

from clinical_scheduler.eval.calibration import ConfusionCounts, calibrate_from_outcomes
from clinical_scheduler.improvement.optimizer import (
    RecalibrationEvidence,
    evaluate_gate,
)


# -- fixtures -----------------------------------------------------------------


def _outcome(scenario_id: str, *, expect: list[str], observed: list[str]) -> Any:
    """A stand-in for a ScenarioOutcome with just what the corpus builder reads."""
    return types.SimpleNamespace(
        scenario=types.SimpleNamespace(
            scenario_id=scenario_id,
            expect_routes=tuple(types.SimpleNamespace(verdict=v) for v in expect),
        ),
        record=types.SimpleNamespace(
            screenings=[{"verdict": v} for v in observed],
        ),
    )


def _baseline_outcomes() -> list[Any]:
    """One true escalation hit, one escalation missed, two true negatives,
    one false alarm -- a corpus that can distinguish every cell."""
    return [
        _outcome("A", expect=["TRIGGERED"], observed=["TRIGGERED"]),      # TP
        _outcome("B", expect=["TRIGGERED"], observed=["CLEAR"]),          # FN
        _outcome("C", expect=["CLEAR"], observed=["CLEAR"]),              # TN
        _outcome("D", expect=["CLEAR"], observed=["CLEAR"]),              # TN
        _outcome("E", expect=["CLEAR"], observed=["UNCERTAIN"]),          # FP
    ]


def _evidence(
    previous: ConfusionCounts, current: ConfusionCounts
) -> RecalibrationEvidence:
    return RecalibrationEvidence(
        previous=previous,
        current=current,
        corpus_scenario_ids=("A", "B", "C", "D", "E"),
    )


def _gate(**overrides: Any) -> Any:
    """A minimal passing gate invocation, overridden per test."""
    kwargs: dict[str, Any] = {
        "patch_valid": True,
        "baseline": None,
        "candidate": None,
        "holdout_baseline": None,
        "holdout_candidate": None,
        "target_fixed": True,
    }
    kwargs.update(overrides)
    return evaluate_gate(**kwargs)


# -- the corpus builder -------------------------------------------------------


def test_calibrate_from_outcomes_counts_what_the_run_did() -> None:
    counts = calibrate_from_outcomes(_baseline_outcomes())
    assert counts == ConfusionCounts(
        true_positive=1, false_negative=1, true_negative=2, false_positive=1
    )


def test_an_outcome_with_no_declared_route_makes_no_claim() -> None:
    outcomes = [_outcome("X", expect=[], observed=["TRIGGERED"])]
    counts = calibrate_from_outcomes(outcomes)
    assert counts.total == 0


def test_an_outcome_that_never_screened_is_excluded() -> None:
    """A session produced no screening record is a run that did not happen, not
    a true negative."""
    outcome = types.SimpleNamespace(
        scenario=types.SimpleNamespace(
            scenario_id="X",
            expect_routes=(types.SimpleNamespace(verdict="CLEAR"),),
        ),
        record=types.SimpleNamespace(screenings=[]),
    )
    counts = calibrate_from_outcomes([outcome])
    assert counts.total == 0


# -- the gate check -----------------------------------------------------------


def test_a_pattern_patch_without_recalibration_evidence_is_refused() -> None:
    gate = _gate(patch_touches_patterns=True, recalibration=None)
    assert not gate.accepted
    assert any(
        "no recalibration evidence" in reason for reason in gate.reasons
    ), gate.reasons


def test_a_recall_regression_refuses_the_patch() -> None:
    """More false negatives after the change vetoes, whatever else improved.

    In the words of the hazard log: a change that misses one more emergency while
    eliminating every false alarm is still a regression, because the failure it
    permits is a missed emergency.
    """
    gate = _gate(
        patch_touches_patterns=True,
        recalibration=_evidence(
            previous=ConfusionCounts(true_positive=1, false_negative=1, true_negative=2, false_positive=1),
            current=ConfusionCounts(true_positive=0, false_negative=2, true_negative=2, false_positive=1),
        ),
    )
    assert not gate.accepted
    assert any(
        "recalibration shows a recall regression" in reason for reason in gate.reasons
    ), gate.reasons


def test_clean_recalibration_records_the_check_as_passed() -> None:
    gate = _gate(
        patch_touches_patterns=True,
        recalibration=_evidence(
            previous=ConfusionCounts(true_positive=1, false_negative=1, true_negative=2, false_positive=1),
            current=ConfusionCounts(true_positive=2, false_negative=0, true_negative=2, false_positive=1),
        ),
    )
    assert "recalibration_no_false_negative_increase" in gate.checks_passed
    assert any(
        "recall measurement" in caveat for caveat in gate.caveats
    ), gate.caveats


def test_a_patch_that_touches_no_pattern_is_not_asked_for_evidence() -> None:
    """A prohibited-claim or template edit does not move the sentinel's recall,
    so the check must not fire for it."""
    gate = _gate(patch_touches_patterns=False, recalibration=None)
    assert "recalibration_no_false_negative_increase" not in gate.checks_passed
    assert not any(
        "recalibration" in reason for reason in gate.reasons
    ), gate.reasons


def test_the_evidence_summary_names_its_numbers_and_corpus() -> None:
    evidence = _evidence(
        previous=ConfusionCounts(1, 1, 2, 1),
        current=ConfusionCounts(2, 0, 2, 1),
    )
    summary = evidence.summary()
    assert "false_negatives 1 -> 0" in summary
    assert "false_positives 1 -> 1" in summary
    assert "corpus=5 scenarios" in summary
    assert evidence.regressed is False


def test_the_delta_property_reports_what_changed() -> None:
    evidence = _evidence(
        previous=ConfusionCounts(2, 0, 2, 1),
        current=ConfusionCounts(1, 1, 2, 1),
    )
    assert evidence.regressed is True
    assert evidence.delta.false_negative_delta == 1
    assert evidence.delta.false_positive_delta == 0


# -- D-044: the comparison itself must hold -----------------------------------


def test_a_scale_mismatch_refuses_the_patch_even_with_clean_counts() -> None:
    """Tables of different sizes cannot be compared cell for cell.

    The veto here is not formed from the cell deltas -- a smaller corpus can
    show the same FN count with a hidden interior that changed. What the gate
    vetoes is the claim to have measured a delta at all.
    """
    gate = _gate(
        patch_touches_patterns=True,
        recalibration=_evidence(
            previous=ConfusionCounts(1, 1, 2, 1),   # total 5
            current=ConfusionCounts(2, 0, 1, 1),    # total 4
        ),
    )
    assert not gate.accepted
    assert any(
        "different numbers of observations" in reason for reason in gate.reasons
    ), gate.reasons
    assert "recalibration_no_false_negative_increase" not in gate.checks_passed


def test_a_larger_corpus_with_the_same_fn_count_is_also_refused() -> None:
    """The growth direction does not matter; the mismatch does.

    A regression veto would need the *same* corpus: a growing one can absorb a
    real loss behind its new observations. Three fewer FN on 7 observations is
    not an improvement over one FN on 5.
    """
    gate = _gate(
        patch_touches_patterns=True,
        recalibration=_evidence(
            previous=ConfusionCounts(2, 1, 2, 0),    # total 5
            current=ConfusionCounts(3, 0, 3, 1),     # total 7
        ),
    )
    assert not gate.accepted
    assert any(
        "different numbers of observations" in reason for reason in gate.reasons
    ), gate.reasons


def test_scale_mismatch_is_a_property_of_the_delta_not_the_gate() -> None:
    delta = RecalibrationEvidence(
        previous=ConfusionCounts(1, 1, 2, 1),
        current=ConfusionCounts(1, 1, 2, 1),
        corpus_scenario_ids=("A",),
    ).delta
    # Same totals by coincidence of the ids, but the *counts* differ in shape:
    # this is the case where total equality holds and the comparison is still
    # trusted, which is exactly the proxy性状 protection this property offers.
    assert delta.scale_mismatch is False  # totals are equal; delta stays clean


def test_the_recall_block_fires_before_the_validity_short_circuit() -> None:
    """A patch that is invalid for an unrelated reason still answers the recall
    question; the gate's reasons list must not stop at its first objection."""
    gate = _gate(
        patch_valid=False,
        patch_touches_patterns=True,
        recalibration=None,
    )
    assert not gate.accepted
    assert any(
        "no recalibration evidence" in reason for reason in gate.reasons
    ), gate.reasons
    assert any(
        "static validation failed" in reason for reason in gate.reasons
    ), gate.reasons


def test_equal_totals_still_veto_on_a_real_regression() -> None:
    """A same-scale regression is caught by the FN check, not confused with the
    scale check: the two reasons must never merge into one vague output."""
    gate = _gate(
        patch_touches_patterns=True,
        recalibration=_evidence(
            previous=ConfusionCounts(2, 0, 2, 1),
            current=ConfusionCounts(1, 1, 2, 1),
        ),
    )
    assert not gate.accepted
    assert not any(
        "different numbers of observations" in reason for reason in gate.reasons
    ), gate.reasons
    assert any(
        "recalibration shows a recall regression" in reason for reason in gate.reasons
    ), gate.reasons
