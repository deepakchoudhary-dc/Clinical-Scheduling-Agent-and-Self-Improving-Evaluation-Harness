"""Tests for the typed patch schema.

The schema is the mechanism that stops a repair loop from becoming a text editor.
These tests hold the four properties that do that work:

* a patch may not operate **above** the layer its failure class requires, so the
  prompt cannot become the default answer to a systems failure;
* an operation may not silently do nothing, so a patch cannot report success
  while changing nothing;
* every operation must name the failure it claims to fix;
* the content hash must be stable across the one field that changes every time a
  patch is built.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from clinical_scheduler.improvement.patch_schema import CandidatePatch, OpKind
from clinical_scheduler.improvement.taxonomy import FailureClass, PatchLayer

# The class value must appear in every operation rationale, lowercased. Written
# once so a typo in the tests cannot masquerade as a policy decision.
_TOKEN = FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE.value.lower()


def _op(**overrides):
    op = {
        "op": "add_trigger_pattern",
        "rule_id": "SAFE-001",
        "value": "having a heart attack",
        "polarity": "POSITIVE",
        "rationale": f"Repairs the {_TOKEN} measured on ADV-11 by naming the condition.",
    }
    op.update(overrides)
    return op


def _patch(**overrides) -> CandidatePatch:
    fields = {
        "patch_id": "P-0001",
        "target_scenario_id": "ADV-11",
        "failure_class": FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE,
        "layer": PatchLayer.L4_POLICY_DATA,
        "base_policy_version": "2026.10-demo.1",
        "target_policy_version": "2026.10-demo.2",
        "rollback_target": "2026.10-demo.1",
        "operations": (_op(),),
        "new_regression_tests": ("ADV-11",),
        "approver_role": "clinical_safety_officer",
        "evidence_refs": ("RCA-ADV-11",),
    }
    fields.update(overrides)
    return CandidatePatch(**fields)


# ---------------------------------------------------------------------------
# The layer ceiling: the mechanical form of "do not default to the prompt"
# ---------------------------------------------------------------------------


def test_a_patch_above_the_required_layer_is_refused_at_construction() -> None:
    """The single most important constraint in the schema.

    A missed escalation requires a policy-data fix. A prompt-layer patch for the
    same failure leaves the cause in place, and the refusal happens in the schema
    so no caller can construct one by forgetting to validate.
    """
    with pytest.raises(ValidationError) as excinfo:
        _patch(layer=PatchLayer.L6_PROMPT)
    assert "requires a patch at" in str(excinfo.value)


def test_a_patch_below_the_required_layer_is_allowed() -> None:
    """A stronger-than-needed fix is permitted.

    The preference order runs database-first, so a lower layer is a *better*
    repair, not an invalid one. The validator notes it rather than blocking it.
    """
    patch = _patch(layer=PatchLayer.L3_FSM_OR_CAPABILITY)
    assert patch.layer is PatchLayer.L3_FSM_OR_CAPABILITY


def test_a_patch_at_exactly_the_required_layer_is_allowed() -> None:
    assert _patch().layer is PatchLayer.L4_POLICY_DATA


# ---------------------------------------------------------------------------
# A patch must not be able to claim a repair it did not make
# ---------------------------------------------------------------------------


def test_a_target_version_equal_to_the_base_is_refused() -> None:
    """A patch that produces the version it started from changes nothing."""
    with pytest.raises(ValidationError) as excinfo:
        _patch(target_policy_version="2026.10-demo.1")
    assert "would produce no new version" in str(excinfo.value)


def test_a_rollback_target_equal_to_the_target_is_refused() -> None:
    """Rolling back to the version just produced is a no-op, not a rollback."""
    with pytest.raises(ValidationError):
        _patch(rollback_target="2026.10-demo.2")


def test_an_operation_missing_a_required_field_is_refused() -> None:
    """A no-op operation is worse than a rejected patch, because it looks like a repair."""
    with pytest.raises(ValidationError) as excinfo:
        _patch(operations=({**_op(), "polarity": None},))
    assert "requires" in str(excinfo.value)


def test_an_operation_carrying_a_field_it_does_not_accept_is_refused() -> None:
    """A stray field means the author expected an operation this schema does not implement.

    Guessing at their intent is how a deliberately narrow schema stops being
    narrow.
    """
    with pytest.raises(ValidationError):
        _patch(operations=({**_op(), "template_text": "some replacement text"},))


def test_a_patch_with_no_regression_test_is_refused() -> None:
    """Acceptance without a test means the same failure can return unnoticed."""
    with pytest.raises(ValidationError):
        _patch(new_regression_tests=())


def test_a_patch_with_no_evidence_reference_is_refused() -> None:
    """A patch that cites nothing cannot be traced back to the failure it answers."""
    with pytest.raises(ValidationError):
        _patch(evidence_refs=())


def test_an_operation_rationale_must_name_the_failure_class() -> None:
    """The cheap guard against a patch being reused to justify an unrelated edit.

    It can be satisfied dishonestly; its value is that doing so requires writing
    the wrong claim into a record a human reads.
    """
    with pytest.raises(ValidationError) as excinfo:
        _patch(operations=({**_op(), "rationale": "General tidy-up of the cardiac rule."},))
    # The rejection names the class it expected, so the author is told what to fix.
    assert FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE.value in str(excinfo.value)


# ---------------------------------------------------------------------------
# Identity: ids, hashing and the rules a patch touches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["P-1", "p-0001", "P-00001", "P0001"])
def test_patch_id_shape_is_enforced(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        _patch(patch_id=bad_id)


@pytest.mark.parametrize("good_id", ["ADV-11", "S04", "HO-01", "MET-12"])
def test_scenario_id_shape_accepts_both_families(good_id: str) -> None:
    assert _patch(target_scenario_id=good_id).target_scenario_id == good_id


def test_content_hash_ignores_the_creation_timestamp() -> None:
    """Two patches built from the same content must hash the same.

    ``created_at`` defaults to "now", so a hash that included it would differ on
    every construction and could not be compared across runs -- which is exactly
    what the approval record does with it.
    """
    first = _patch()
    second = _patch()
    assert first.created_at != second.created_at
    assert first.content_hash() == second.content_hash()


def test_content_hash_changes_when_the_content_changes() -> None:
    """The other half of the previous property: it must still detect a real edit."""
    original = _patch()
    edited = _patch(
        operations=({**_op(), "value": "having a cardiac event"},),
    )
    assert original.content_hash() != edited.content_hash()


def test_touched_rule_ids_reports_every_rule_the_patch_edits() -> None:
    patch = _patch(
        operations=(
            _op(rule_id="SAFE-001"),
            _op(rule_id="SAFE-002", value="arm feels heavy"),
        )
    )
    assert patch.touched_rule_ids() == ("SAFE-001", "SAFE-002")


def test_the_op_vocabulary_is_closed() -> None:
    """Every operation is confined to a policy bundle.

    There is deliberately no operation for changing a tool contract, an FSM
    transition or a test -- each of those would let a patch move the goal rather
    than the system.
    """
    assert {member.value for member in OpKind} == {
        "add_trigger_pattern",
        "add_prohibited_claim",
        "widen_rule_scope",
        "update_template_text",
        "add_rule",
        "bump_review_date",
    }
