"""Sentinel calibration: the arithmetic, and the honesty about its labels.

The numbers here are hand-computed. A statistic tested only against itself is not
tested, and this module's whole purpose is to make a claim (recall) measurable, so
a wrong interval would be worse than no interval.

Two properties matter beyond the arithmetic: the labels are *scenario-declared*,
not clinician-adjudicated, and every report must say so; and predictive values
must be reported against a prevalence, because sensitivity alone makes a
near-useless screen look excellent.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from clinical_scheduler.eval.calibration import (
    SCENARIO_DECLARED_LABELS,
    CalibrationReport,
    ConfusionCounts,
    calibrate,
    delta_report,
    expected_escalation,
    is_escalation,
    observations_from_run,
    observed_escalation,
    wilson_interval,
)


# -- Wilson interval ----------------------------------------------------------


def test_wilson_interval_of_an_empty_sample_is_none() -> None:
    assert wilson_interval(0, 0) is None


def test_wilson_interval_at_zero_successes() -> None:
    """0 of 10: the upper bound is what a reviewer needs, and the normal
    approximation would give 0..0."""
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.2775, abs=1e-3)


def test_wilson_interval_at_half() -> None:
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)


def test_wilson_interval_stays_inside_the_unit_interval() -> None:
    for successes, total in ((10, 10), (0, 3), (1, 3)):
        low, high = wilson_interval(successes, total)
        assert 0.0 <= low <= high <= 1.0


# -- confusion table ----------------------------------------------------------


def _counts() -> ConfusionCounts:
    # sensitivity 9/10, specificity 95/100
    return ConfusionCounts(true_positive=9, false_negative=1, true_negative=95, false_positive=5)


def test_sensitivity_and_specificity() -> None:
    counts = _counts()
    assert counts.sensitivity == pytest.approx(0.9)
    assert counts.specificity == pytest.approx(0.95)
    assert counts.total == 110


def test_a_rate_with_a_zero_denominator_is_none_not_zero() -> None:
    counts = ConfusionCounts(true_positive=0, false_negative=0, true_negative=5, false_positive=0)
    assert counts.sensitivity is None
    assert counts.specificity == pytest.approx(1.0)


def test_positive_predictive_value_against_a_hand_computed_value() -> None:
    """sens .9, spec .95, prevalence .001

    PPV = .9*.001 / (.9*.001 + .05*.999) = .0009 / .05085 = 0.017699...
    """
    assert _counts().positive_predictive_value(0.001) == pytest.approx(0.017699, abs=1e-6)


def test_negative_predictive_value_against_a_hand_computed_value() -> None:
    """NPV = .95*.999 / (.95*.999 + .1*.001) = .94905 / .94915 = 0.999894..."""
    assert _counts().negative_predictive_value(0.001) == pytest.approx(0.999895, abs=1e-6)


def test_predictive_values_are_none_when_the_table_is_degenerate() -> None:
    counts = ConfusionCounts(true_positive=0, false_negative=0, true_negative=5, false_positive=0)
    assert counts.positive_predictive_value(0.01) is None


# -- building the table -------------------------------------------------------


def test_calibrate_counts_every_cell() -> None:
    counts = calibrate([(True, True), (True, False), (False, False), (False, True)])
    assert counts == ConfusionCounts(
        true_positive=1, false_negative=1, true_negative=1, false_positive=1
    )


def test_is_escalation_recognises_only_the_two_escalating_verdicts() -> None:
    assert is_escalation("TRIGGERED")
    assert is_escalation("uncertain")
    assert not is_escalation("CLEAR")
    assert not is_escalation("CLEAR_NEGATED")
    assert not is_escalation("")


def _scenario(*verdicts: str) -> Any:
    return types.SimpleNamespace(
        expect_routes=tuple(types.SimpleNamespace(verdict=v) for v in verdicts)
    )


def test_a_scenario_with_no_declared_route_makes_no_claim() -> None:
    """Excluded rather than counted as a negative: no expectation is not evidence."""
    assert expected_escalation(_scenario()) is None


def test_expected_escalation_is_true_if_any_declared_route_escalates() -> None:
    assert expected_escalation(_scenario("CLEAR", "TRIGGERED")) is True
    assert expected_escalation(_scenario("CLEAR", "CLEAR_META")) is False


def test_observed_escalation_reads_the_screening_records() -> None:
    assert observed_escalation({"screenings": [{"verdict": "CLEAR"}, {"verdict": "TRIGGERED"}]})
    assert not observed_escalation({"screenings": [{"verdict": "CLEAR"}]})
    assert observed_escalation({"screenings": []}) is None
    assert observed_escalation({}) is None


def test_observations_pair_declared_expectations_with_what_happened() -> None:
    rows = [
        {"scenario_id": "S01", "screenings": [{"verdict": "TRIGGERED"}]},
        {"scenario_id": "S02", "screenings": [{"verdict": "CLEAR"}]},
        {"scenario_id": "S03", "screenings": [{"verdict": "CLEAR"}]},  # no declared route
    ]
    index = {
        "S01": _scenario("TRIGGERED"),
        "S02": _scenario("TRIGGERED"),  # a real miss
        "S03": _scenario(),
    }
    pairs = observations_from_run(rows, index)
    assert pairs == [(True, True), (True, False)]


# -- the report ---------------------------------------------------------------


def test_the_report_names_the_label_source_and_disclaims_clinical_validation() -> None:
    report = CalibrationReport(counts=_counts(), label_source=SCENARIO_DECLARED_LABELS).report()
    assert "not clinician adjudication" in report
    assert "consistency measurement, not clinical validation" in report


def test_the_report_prints_intervals_and_a_prevalence_table() -> None:
    report = CalibrationReport(counts=_counts(), label_source=SCENARIO_DECLARED_LABELS).report()
    assert "95% CI" in report
    assert "predictive values by prevalence" in report
    assert "false negative" in report
    assert "missed escalation" in report
    assert "false positive" in report
    assert "alert fatigue" in report


def test_the_prevalence_table_is_a_range_not_a_point_estimate() -> None:
    """The fix for the first version of this report.

    With 0 false positives in 3 negatives, the point estimate is 100% specificity,
    and the table read **100% PPV at every prevalence** -- a claim of perfection
    from five observations. The range across the interval corners says otherwise,
    and the lower bound is the number that matters.
    """
    counts = ConfusionCounts(true_positive=2, false_negative=0, true_negative=3, false_positive=0)
    low, high = counts.positive_predictive_value_range(0.01)
    assert low is not None and high is not None
    assert high == pytest.approx(1.0)
    assert low < 0.05, f"the lower bound should be nowhere near perfect, got {low}"

    report = CalibrationReport(counts=counts, label_source="x").report()
    assert "RANGE" in report
    assert ".." in report


def test_a_small_sample_is_flagged_as_saying_little() -> None:
    counts = ConfusionCounts(true_positive=2, false_negative=0, true_negative=3, false_positive=0)
    report = CalibrationReport(counts=counts, label_source="x").report()
    assert "CAVEAT" in report
    assert "not yet measured" in report


def test_a_large_sample_is_not_flagged() -> None:
    counts = ConfusionCounts(
        true_positive=90, false_negative=10, true_negative=950, false_positive=50
    )
    report = CalibrationReport(counts=counts, label_source="x").report()
    assert "CAVEAT" not in report


def test_the_ppv_range_is_bounded_by_the_point_estimate() -> None:
    counts = _counts()
    low, high = counts.positive_predictive_value_range(0.001)
    point = counts.positive_predictive_value(0.001)
    assert low is not None and high is not None and point is not None
    assert low <= point <= high


def test_the_report_says_not_measured_when_a_rate_is_undefined() -> None:
    counts = ConfusionCounts(true_positive=0, false_negative=0, true_negative=3, false_positive=0)
    report = CalibrationReport(counts=counts, label_source="x").report()
    assert "sensitivity: n/a (0/0, not measured)" in report
    assert "predictive values: not measured" in report


# -- change control (B4) ------------------------------------------------------


def test_a_calibration_round_trips_through_json() -> None:
    """A later run must be able to diff against a saved one."""
    report = CalibrationReport(counts=_counts(), label_source=SCENARIO_DECLARED_LABELS)
    assert CalibrationReport.from_json(report.to_json()) == report


def test_a_recall_regression_is_a_regression_whatever_else_improves() -> None:
    """The change-control rule.

    The candidate loses one escalation and eliminates nine false alarms. That is a
    trade a scorecard would call an improvement and a clinician would not: the
    failure it permits is a missed emergency.
    """
    before = CalibrationReport(
        counts=ConfusionCounts(true_positive=9, false_negative=1, true_negative=90, false_positive=10),
        label_source="x",
    )
    after = CalibrationReport(
        counts=ConfusionCounts(true_positive=8, false_negative=2, true_negative=99, false_positive=1),
        label_source="x",
    )
    delta = delta_report(before, after)
    assert delta.false_negative_delta == 1
    assert delta.false_positive_delta == -9
    assert delta.regressed is True
    assert "REGRESSION" in delta.report()


def test_a_change_that_does_not_lose_recall_is_not_a_regression() -> None:
    before = CalibrationReport(counts=ConfusionCounts(9, 1, 90, 10), label_source="x")
    after = CalibrationReport(counts=ConfusionCounts(10, 0, 95, 5), label_source="x")
    delta = delta_report(before, after)
    assert delta.false_negative_delta == -1
    assert delta.regressed is False
    assert "No recall regression" in delta.report()


def test_a_delta_across_different_corpora_says_so() -> None:
    """Comparing calibrations of different corpora is not evidence about a change."""
    before = CalibrationReport(counts=ConfusionCounts(5, 0, 5, 0), label_source="x")
    after = CalibrationReport(counts=ConfusionCounts(50, 0, 50, 0), label_source="x")
    report = delta_report(before, after).report()
    assert "WARNING" in report
    assert "different numbers of observations" in report


def test_the_delta_report_shows_before_and_after() -> None:
    before = CalibrationReport(counts=ConfusionCounts(9, 1, 90, 10), label_source="x")
    after = CalibrationReport(counts=ConfusionCounts(9, 1, 95, 5), label_source="x")
    report = delta_report(before, after).report()
    assert "false negatives: 1 -> 1" in report
    assert "specificity: 90.0% -> 95.0%" in report
