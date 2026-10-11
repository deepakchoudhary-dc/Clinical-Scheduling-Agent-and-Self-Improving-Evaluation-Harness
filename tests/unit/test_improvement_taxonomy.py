"""Tests for the failure taxonomy: the layer a failure must be repaired at.

The taxonomy is the part of the loop that decides *where* a change belongs, and
its whole value is that the answer is not "the prompt". These tests hold that
line: every class routes to a layer, the routing reasons say what a higher-layer
patch would leave unfixed, and the two directions of a routing failure are
distinguished from each other rather than collapsed into one class.
"""

from __future__ import annotations

import pytest

from clinical_scheduler.improvement.taxonomy import (
    LAYER_RANK,
    FailureClass,
    PatchLayer,
    classify_finding,
    layer_reason,
    required_layer,
)


# ---------------------------------------------------------------------------
# Structural completeness
# ---------------------------------------------------------------------------


def test_every_failure_class_routes_to_a_layer() -> None:
    """An unrouted class would silently become an oracle error at run time.

    Failing here instead means a new class cannot be added without deciding
    where its repair belongs, which is the decision the taxonomy exists to force.
    """
    for failure_class in FailureClass:
        layer = required_layer(failure_class)
        assert isinstance(layer, PatchLayer)
        assert layer_reason(failure_class).strip(), (
            f"{failure_class.value} routes to {layer.value} with no stated reason"
        )


def test_the_layer_order_is_the_plan_order() -> None:
    """Rank must follow plan.md section 13.3, lowest layer first.

    The comparison "is this patch above the required layer" is by rank, so a
    reordering here silently changes what the validator accepts.
    """
    assert LAYER_RANK[PatchLayer.L1_DATABASE_CONSTRAINT] == 1
    assert LAYER_RANK[PatchLayer.L7_MODEL] == 7
    ordered = sorted(PatchLayer, key=lambda layer: LAYER_RANK[layer])
    assert [layer.value for layer in ordered] == [
        "L1_database_constraint",
        "L2_tool_contract",
        "L3_fsm_or_capability",
        "L4_policy_data",
        "L5_template_content",
        "L6_prompt",
        "L7_model",
    ]


def test_no_failure_class_routes_to_the_model() -> None:
    """The outermost layer must never be the *required* fix.

    A failure whose only repair is "the model should behave differently" has no
    testable, reviewable, rollback-able change behind it, so it cannot be a
    required layer. Model variance is represented as MODEL_FLAKINESS, which
    routes inward to the harness instead.
    """
    for failure_class in FailureClass:
        assert required_layer(failure_class) is not PatchLayer.L7_MODEL, (
            f"{failure_class.value} routes to the model layer"
        )


def test_a_missed_escalation_requires_policy_data_not_a_prompt() -> None:
    """The headline routing decision, asserted on its own.

    A missed high-risk route is a gap in the clinician-authored pattern set. A
    prompt instruction not to miss it cannot be tested, cannot be reviewed by the
    safety officer who owns the rule, and cannot be rolled back.
    """
    assert required_layer(FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE) is PatchLayer.L4_POLICY_DATA


# ---------------------------------------------------------------------------
# Classifying a routing finding: the direction is the whole question
# ---------------------------------------------------------------------------


def test_a_required_escalation_that_did_not_happen_is_a_false_negative() -> None:
    classification = classify_finding(
        oracle_id="L3.ROUTE_MISMATCH",
        scenario_id="ADV-11",
        evidence="expected_verdict=TRIGGERED observed_verdict=CLEAR",
        reason_code="ROUTE_MISMATCH",
        expected_verdict="TRIGGERED",
        observed_verdict="CLEAR",
        implicated_rule_id="SAFE-001",
    )
    assert classification.failure_class is FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE
    assert classification.implicated_rule_id == "SAFE-001"
    assert classification.counter_hypotheses, (
        "a classification must name the alternatives it considered and rejected"
    )
    assert any(
        "SAFETY_SENTINEL_FALSE_POSITIVE" in hypothesis
        for hypothesis in classification.counter_hypotheses
    )


def test_an_unwanted_escalation_is_a_false_positive() -> None:
    classification = classify_finding(
        oracle_id="L3.ROUTE_MISMATCH",
        scenario_id="ADV-20",
        evidence="expected_verdict=CLEAR observed_verdict=TRIGGERED",
        reason_code="ROUTE_MISMATCH",
        expected_verdict="CLEAR",
        observed_verdict="TRIGGERED",
    )
    assert classification.failure_class is FailureClass.SAFETY_SENTINEL_FALSE_POSITIVE


def test_the_same_oracle_id_yields_opposite_classes_by_direction() -> None:
    """The property that makes the direction check necessary.

    One oracle id reports both directions, so the id alone cannot decide the
    class and a table lookup on it would have to pick one direction and be wrong
    about the other.
    """
    common = dict(
        oracle_id="L3.ROUTE_MISMATCH",
        scenario_id="S01",
        evidence="",
        reason_code="ROUTE_MISMATCH",
    )
    missed = classify_finding(**common, expected_verdict="TRIGGERED", observed_verdict="CLEAR")
    spurious = classify_finding(**common, expected_verdict="CLEAR", observed_verdict="TRIGGERED")
    assert missed.failure_class is not spurious.failure_class


def test_two_non_clear_verdicts_are_a_precision_defect_not_a_safety_one() -> None:
    """The patient is still escalated, so the defect is which route, not whether.

    Classifying this as a false negative would send the repair to the pattern
    set when the route was already protective, which is how a loop produces
    changes that move a score without fixing anything.
    """
    classification = classify_finding(
        oracle_id="L3.ROUTE_MISMATCH",
        scenario_id="ADV-13",
        evidence="expected_verdict=TRIGGERED observed_verdict=UNCERTAIN",
        reason_code="ROUTE_MISMATCH",
        expected_verdict="TRIGGERED",
        observed_verdict="UNCERTAIN",
    )
    assert classification.failure_class is FailureClass.CLINICAL_POLICY_GAP


# ---------------------------------------------------------------------------
# Fallbacks that must not guess
# ---------------------------------------------------------------------------


def test_a_quality_finding_routes_to_the_quality_class() -> None:
    """Quality findings are capped at S3 and must never be routed structurally."""
    for oracle_id in ("Q3_NO_FALSE_REASSURANCE", "L4.Q3_NO_FALSE_REASSURANCE"):
        classification = classify_finding(
            oracle_id=oracle_id,
            scenario_id="S01",
            evidence="matched 'don't worry'",
            reason_code="QUALITY",
        )
        assert classification.failure_class is FailureClass.ACCESSIBILITY_QUALITY_FAILURE


def test_an_oracle_error_is_not_a_finding_about_the_system() -> None:
    classification = classify_finding(
        oracle_id="ORACLE",
        scenario_id="S01",
        evidence="ZeroDivisionError",
        reason_code="ORACLE_ERROR",
    )
    assert classification.failure_class is FailureClass.ORACLE_ERROR


def test_an_unmapped_oracle_id_is_an_oracle_error_not_a_guess() -> None:
    """An unclassifiable failure must not be routed to a plausible-looking layer.

    A guess here becomes a patch at a random layer, so the honest answer is that
    the cause cannot be named yet.
    """
    classification = classify_finding(
        oracle_id="L9.SOMETHING_NEW",
        scenario_id="S01",
        evidence="whatever",
    )
    assert classification.failure_class is FailureClass.ORACLE_ERROR
    assert "L9.SOMETHING_NEW" in " ".join(classification.counter_hypotheses)


@pytest.mark.parametrize(
    "oracle_id",
    [
        "L3.ROUTE_MISMATCH",
        "L3.FIRED_RULES_MISMATCH",
        "L3.SCENARIO_TEXT_VIOLATION",
        "L3.SCENARIO_TEXT_MISSING",
        "L3.SCENARIO_DB_MISMATCH",
        "L1.INV02_MUTATION_OFFERED_UNDER_SAFETY_ROUTE",
        "L2.AUDIT_CHAIN_BROKEN",
        "L0.UNHANDLED_EXCEPTION",
    ],
)
def test_known_oracle_ids_never_fall_through_to_oracle_error(oracle_id: str) -> None:
    """Every id the stack can emit must be routable.

    This is the regression guard for the whole class of defect where a new
    oracle id is added and silently degrades every run that produces it into
    "oracle error", which makes the run ineligible.
    """
    classification = classify_finding(
        oracle_id=oracle_id,
        scenario_id="S01",
        evidence="evidence",
        reason_code="ROUTE_MISMATCH",
    )
    assert classification.failure_class is not FailureClass.ORACLE_ERROR
