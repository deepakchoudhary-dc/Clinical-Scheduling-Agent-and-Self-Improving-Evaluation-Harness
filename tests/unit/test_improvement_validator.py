"""Tests for the static validator.

The validator is what stands between a proposed patch and an evaluation run. Two
properties matter most and are asserted directly here:

* **It refuses.** A patch that names a rule that does not exist, or a pattern
  that is not in normalised surface form, or a base version that has moved on,
  must be rejected *before* anything is measured -- because a run against a patch
  that could never be applied produces a number that means nothing.
* **Composition is all-or-nothing.** A partially applied patch is a bundle that
  exists in no version and that no rollback target describes.
"""

from __future__ import annotations

import pytest

from clinical_scheduler.improvement.patch_schema import CandidatePatch
from clinical_scheduler.improvement.taxonomy import FailureClass, PatchLayer
from clinical_scheduler.improvement.validator import (
    MAX_RULES_PER_PATCH,
    apply_patch,
    validate,
)
from clinical_scheduler.policy.engine import load_baseline
from clinical_scheduler.policy.schema import PolicyBundle

_TOKEN = FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE.value.lower()


def _op(**overrides):
    op = {
        "op": "add_trigger_pattern",
        "rule_id": "SAFE-001",
        "value": "having a heart attack",
        "polarity": "POSITIVE",
        "category": "MEDICAL_EMERGENCY",
        "rationale": f"Repairs the {_TOKEN} measured on ADV-11.",
    }
    op.update(overrides)
    return op


def _patch(bundle: PolicyBundle, **overrides) -> CandidatePatch:
    fields = {
        "patch_id": "P-0001",
        "target_scenario_id": "ADV-11",
        "failure_class": FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE,
        "layer": PatchLayer.L4_POLICY_DATA,
        "base_policy_version": bundle.policy_version,
        "target_policy_version": "2026.10-demo.2",
        "rollback_target": bundle.policy_version,
        "operations": (_op(),),
        "new_regression_tests": ("ADV-11",),
        "approver_role": "clinical_safety_officer",
        "evidence_refs": ("RCA-ADV-11",),
    }
    fields.update(overrides)
    return CandidatePatch(**fields)


@pytest.fixture()
def bundle() -> PolicyBundle:
    return load_baseline().bundle


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_a_well_formed_patch_is_valid_and_composes(bundle: PolicyBundle) -> None:
    result = validate(_patch(bundle), bundle)
    assert result.valid, result.report()
    assert result.composed_bundle is not None
    assert result.composed_bundle.policy_version == "2026.10-demo.2"

    before = {p.phrase for p in bundle.rule_by_id("SAFE-001").trigger_patterns}
    after = {p.phrase for p in result.composed_bundle.rule_by_id("SAFE-001").trigger_patterns}
    assert after - before == {"having a heart attack"}


def test_the_composed_bundle_changes_its_fingerprint(bundle: PolicyBundle) -> None:
    """Adding a pattern must change the bundle hash, or the run manifest lies.

    The manifest records the bundle hash so two runs can be compared. A change
    that did not move the hash would make two genuinely different policies look
    like the same experiment.
    """
    from clinical_scheduler.policy.engine import PolicyEngine

    before = PolicyEngine(bundle).bundle_hash()
    after = PolicyEngine(validate(_patch(bundle), bundle).composed_bundle).bundle_hash()
    assert before != after


# ---------------------------------------------------------------------------
# Refusals: base version, references, pattern form
# ---------------------------------------------------------------------------


def test_a_patch_against_a_stale_base_version_is_refused(bundle: PolicyBundle) -> None:
    """The rule it names may not exist there, or may have changed.

    Either way the patch's evidence no longer applies to the bundle in front of it.
    """
    result = validate(_patch(bundle, base_policy_version="2026.09-demo.0"), bundle)
    assert not result.valid
    assert any(issue.check == "base_policy_version" for issue in result.issues)


def test_a_patch_naming_a_rule_that_does_not_exist_is_refused(bundle: PolicyBundle) -> None:
    result = validate(_patch(bundle, operations=(_op(rule_id="SAFE-999"),)), bundle)
    assert not result.valid
    assert any(issue.check == "rule_exists" for issue in result.issues)


def test_a_pattern_that_is_not_normalised_is_refused(bundle: PolicyBundle) -> None:
    """Patterns are authored in normalised surface form, and the validator says so.

    An unnormalised pattern ("Chest Pain") would never match, because matching
    happens against normalised text. It would be a patch that reports success and
    changes no behaviour -- the exact failure the schema's no-op guards exist to
    prevent, arriving through a different door.
    """
    result = validate(_patch(bundle, operations=(_op(value="Having A Heart Attack"),)), bundle)
    assert not result.valid
    assert any(issue.check == "pattern_normalisation" for issue in result.issues)


def test_an_unknown_polarity_is_refused(bundle: PolicyBundle) -> None:
    """The polarity vocabulary is closed, and the schema closes it.

    Rejected at construction rather than at validation, which is the better
    place: the generator is told immediately, with the illegal value named.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _patch(bundle, operations=(_op(polarity="MAYBE"),))


def test_a_duplicate_pattern_is_refused_at_composition(bundle: PolicyBundle) -> None:
    """Adding a phrase the rule already has changes nothing but claims a repair.

    "chest pain" is already on SAFE-001 in the baseline bundle.
    """
    result = validate(_patch(bundle, operations=(_op(value="chest pain"),)), bundle)
    assert not result.valid
    assert any(issue.check == "composition" for issue in result.issues)


def test_a_patch_touching_too_many_rules_is_refused(bundle: PolicyBundle) -> None:
    """Beyond a handful of rules, a reviewer cannot hold the change in their head.

    An unreviewable patch is an unreviewable approval, so the limit is enforced
    rather than suggested. The bundle has six rules, so four distinct ones
    exceeds the limit without being contrived.
    """
    rules = ["SAFE-001", "SAFE-002", "SAFE-003", "MED-001"]
    assert len(rules) > MAX_RULES_PER_PATCH
    ops = tuple(
        _op(rule_id=rule_id, value=f"phrase for {rule_id.lower()}")
        for rule_id in rules
    )
    result = validate(_patch(bundle, operations=ops), bundle)
    assert not result.valid
    assert any(issue.check == "minimality_size" for issue in result.issues)


# ---------------------------------------------------------------------------
# Defence in depth: the layer ceiling, re-checked
# ---------------------------------------------------------------------------


def test_the_validator_re_checks_the_layer_even_if_the_schema_was_bypassed(
    bundle: PolicyBundle,
) -> None:
    """The schema already refuses a too-high layer; the validator refuses it again.

    The bypass is ``model_copy``, which changes a field **without re-running
    validators** -- a realistic route by which a caller could hold a patch the
    schema would have rejected. The rule is therefore enforced in two places, and
    a caller that skips construction validation still cannot get past the
    validator.
    """
    legal = _patch(bundle)
    assert legal.layer is PatchLayer.L4_POLICY_DATA
    smuggled = legal.model_copy(update={"layer": PatchLayer.L6_PROMPT})

    result = validate(smuggled, bundle)
    assert not result.valid
    assert any(issue.check == "layer_correctness" for issue in result.issues)


# ---------------------------------------------------------------------------
# Composition is all-or-nothing
# ---------------------------------------------------------------------------


def test_apply_patch_raises_rather_than_returning_a_partial_bundle(
    bundle: PolicyBundle,
) -> None:
    """A half-applied patch is a bundle that exists in no version.

    Two operations, the second of which cannot apply. If the first had been
    applied and the second skipped, the result would be a policy that no
    rollback target describes and no version identifies.
    """
    patch = _patch(
        bundle,
        operations=(
            _op(value="having a heart attack"),
            _op(value="chest pain"),  # already present -> composition fails
        ),
    )
    with pytest.raises(Exception) as excinfo:
        apply_patch(patch, bundle)
    assert "chest pain" in str(excinfo.value) or "duplicate" in str(excinfo.value).lower()


def test_apply_patch_leaves_the_input_bundle_untouched(bundle: PolicyBundle) -> None:
    """The function must be pure: the base bundle is shared by every other run."""
    before = bundle.model_dump(mode="json")
    apply_patch(_patch(bundle), bundle)
    assert bundle.model_dump(mode="json") == before


# ---------------------------------------------------------------------------
# The other five operations
# ---------------------------------------------------------------------------


def test_add_prohibited_claim(bundle: PolicyBundle) -> None:
    from clinical_scheduler.policy.schema import ProhibitedClaim

    claim = sorted(ProhibitedClaim, key=lambda c: c.value)[0].value
    result = validate(
        _patch(
            bundle,
            operations=(
                {
                    "op": "add_prohibited_claim",
                    "rule_id": "SAFE-001",
                    "value": claim,
                    "rationale": f"Repairs the {_TOKEN} by prohibiting a claim class.",
                },
            ),
        ),
        bundle,
    )
    assert result.valid, result.report()


def test_bump_review_date(bundle: PolicyBundle) -> None:
    result = validate(
        _patch(
            bundle,
            operations=(
                {
                    "op": "bump_review_date",
                    "rule_id": "SAFE-001",
                    "rationale": f"Repairs the {_TOKEN} by bringing the review forward.",
                },
            ),
        ),
        bundle,
    )
    assert result.valid, result.report()


def test_widen_rule_scope(bundle: PolicyBundle) -> None:
    result = validate(
        _patch(
            bundle,
            operations=(
                {
                    "op": "widen_rule_scope",
                    "rule_id": "SAFE-001",
                    "value": "gp_new",
                    "rationale": f"Repairs the {_TOKEN} by widening the rule's service scope.",
                },
            ),
        ),
        bundle,
    )
    assert result.valid, result.report()
