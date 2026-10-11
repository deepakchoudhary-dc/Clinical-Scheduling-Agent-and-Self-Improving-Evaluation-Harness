"""Patch validation: the checks a candidate must pass before it may be scored.

Validation is deliberately split from scoring. Validation answers **is this
patch well-formed, minimal and aimed at the right layer?** -- questions that can
be answered about the patch alone, without running anything. Scoring answers **did
it actually help?** and requires a full evaluation run.

Keeping them apart has a practical consequence that matters more than tidiness:
a patch that fails validation never costs a run. On a real system that is the
difference between a loop that can be run continuously and one that spends its
budget evaluating malformed proposals.

Five checks, in increasing subtlety:

1. **Schema validity.** Handled by ``CandidatePatch`` itself.
2. **Base version matches.** A patch generated against one policy version must
   not be applied to another; the rule it names may not exist there.
3. **Referential integrity.** Every rule it names must exist, and every claim,
   polarity and template id must be a value the policy schema accepts.
4. **Layer correctness.** The patch must operate at or below the layer its
   failure class requires. This is the mechanical form of plan.md section 13.3.
5. **Minimality.** A patch touching more rules than the classification justifies
   is refused even when it is otherwise valid, because an unreviewable patch is
   an unapproved patch however loudly it is approved.

Every check returns a :class:`ValidationIssue`, and the result carries all of
them rather than stopping at the first. A generator that has to run five times to
learn five problems learns nothing about the shape of the errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from clinical_scheduler.improvement.patch_schema import CandidatePatch, OpKind
from clinical_scheduler.improvement.taxonomy import (
    FailureClass,
    LAYER_RANK,
    PatchLayer,
    required_layer,
)
from clinical_scheduler.policy.schema import (
    PolicyBundle,
    PolicyRule,
    ProhibitedClaim,
    RuleAction,
    RuleOwner,
)
from clinical_scheduler.policy.schema import SentinelPattern

#: A patch may touch at most this many rules. Three is chosen so that a patch
#: which fixes a rule, its guard and the template that renders it still fits,
#: while a bulk edit does not. The cost of a size limit is that a genuinely broad
#: fix must be split; the benefit is that no single accepted change is too large
#: to read.
MAX_RULES_PER_PATCH = 3


@dataclass(frozen=True)
class ValidationIssue:
    """One reason a patch was refused.

    ``remedy`` is required. An issue that says only what is wrong leaves the
    generator to guess, and a generator that guesses at a schema will produce
    something that passes the letter of the check and misses its purpose.
    """

    check: str
    detail: str
    remedy: str
    #: True when the issue cannot be waived by an approver. Layer and schema
    #: violations are structural; a minimality concern could in principle be
    #: accepted with a stated reason, but this prototype treats it as blocking so
    #: that the exception has to be made in code and therefore in review.
    blocking: bool = True


@dataclass
class ValidationResult:
    """The outcome of validating one patch."""

    patch_id: str
    issues: list[ValidationIssue] = field(default_factory=list)
    #: Non-blocking observations, kept so a reviewer sees them.
    notes: list[str] = field(default_factory=list)
    #: The composed bundle, when composition succeeded. Present so the caller
    #: does not have to render again to inspect the result.
    composed_bundle: PolicyBundle | None = None

    @property
    def valid(self) -> bool:
        return not any(issue.blocking for issue in self.issues)

    def report(self) -> str:
        if self.valid:
            lines = [f"{self.patch_id}: VALID"]
        else:
            lines = [f"{self.patch_id}: INVALID"]
            for issue in self.issues:
                marker = "blocking" if issue.blocking else "advisory"
                lines.append(f"  [{marker}] {issue.check}: {issue.detail}")
                lines.append(f"      remedy: {issue.remedy}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_base_version(patch: CandidatePatch, bundle: PolicyBundle) -> list[ValidationIssue]:
    if patch.base_policy_version != bundle.policy_version:
        return [
            ValidationIssue(
                check="base_policy_version",
                detail=(
                    f"patch was generated against {patch.base_policy_version} but the "
                    f"bundle in force is {bundle.policy_version}"
                ),
                remedy=(
                    "Regenerate the patch against the current bundle. The rule it names "
                    "may have changed or been removed, so its evidence no longer applies."
                ),
            )
        ]
    return []


def _check_rule_references(patch: CandidatePatch, bundle: PolicyBundle) -> list[ValidationIssue]:
    """Every referenced rule must exist, and every value must be schema-legal.

    The value checks are duplicated from the policy schema on purpose. If the
    validator accepted anything and relied on the rendered bundle to fail, the
    error a generator saw would be a pydantic traceback about a bundle it did not
    know it had produced. Naming the exact illegal value here produces a message
    the generator can act on.
    """
    issues: list[ValidationIssue] = []
    valid_claims = {c.value for c in ProhibitedClaim}
    valid_polarities = {"POSITIVE", "NEGATION", "HISTORICAL", "THIRD_PERSON", "META"}
    template_ids = {t.template_id for t in bundle.message_templates}

    for operation in patch.operations:
        if operation.op is not OpKind.ADD_RULE:
            if operation.rule_id and bundle.rule_by_id(operation.rule_id) is None:
                issues.append(
                    ValidationIssue(
                        check="rule_exists",
                        detail=f"{operation.op.value} names rule {operation.rule_id}, which does not exist",
                        remedy=(
                            "Use ADD_RULE to introduce a new rule, or correct the id to "
                            "one of: " + ", ".join(sorted(r.rule_id for r in bundle.rules))
                        ),
                    )
                )

        if operation.op is OpKind.ADD_PROHIBITED_CLAIM and operation.value:
            if operation.value not in valid_claims:
                issues.append(
                    ValidationIssue(
                        check="prohibited_claim_value",
                        detail=f"{operation.value!r} is not a declared prohibited claim",
                        remedy="Use one of: " + ", ".join(sorted(valid_claims)),
                    )
                )

        if operation.op is OpKind.ADD_TRIGGER_PATTERN:
            if operation.polarity and operation.polarity not in valid_polarities:
                issues.append(
                    ValidationIssue(
                        check="polarity_value",
                        detail=f"{operation.polarity!r} is not a declared polarity",
                        remedy="Use one of: " + ", ".join(sorted(valid_polarities)),
                    )
                )
            if operation.value and operation.value != operation.value.strip().lower():
                # A pattern is matched against normalised text, so an unnormalised
                # pattern is a pattern that will never fire. This is the single
                # most common way a sentinel patch silently does nothing, and it
                # is worth refusing rather than quietly normalising: the author
                # needs to know the matcher works on the normalised form.
                issues.append(
                    ValidationIssue(
                        check="pattern_normalisation",
                        detail=f"pattern {operation.value!r} is not fully normalised",
                        remedy=(
                            "Rewrite the pattern in the normalised surface form the "
                            "matcher sees: lowercase, single spaces, no surrounding "
                            "punctuation. Use the same form the patient's text is "
                            "reduced to, not the form they typed."
                        ),
                    )
                )

        if operation.op is OpKind.UPDATE_TEMPLATE_TEXT and operation.template_id:
            if operation.template_id not in template_ids:
                issues.append(
                    ValidationIssue(
                        check="template_exists",
                        detail=f"template {operation.template_id!r} does not exist",
                        remedy="Use one of: " + ", ".join(sorted(template_ids)),
                    )
                )

        if operation.op is OpKind.ADD_RULE and operation.new_rule:
            issues.extend(_check_new_rule(operation.new_rule, template_ids))

    return issues


def _check_new_rule(raw: dict, template_ids: set[str]) -> list[ValidationIssue]:
    """Validate a new rule by constructing it, and translate the error.

    Constructing it is the honest check: the schema is the authority, and
    re-implementing its rules here would let the two drift. The translation step
    is for the generator's benefit -- a pydantic error about a field path is
    harder to act on than a sentence naming the field.
    """
    try:
        PolicyRule.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 -- the message is the product here
        return [
            ValidationIssue(
                check="new_rule_schema",
                detail=f"the new rule does not satisfy the policy schema: {str(exc)[:600]}",
                remedy=(
                    "Every rule needs rule_id, owner, rationale, action, jurisdiction, "
                    "effective_from, review_due, evidence_source, rollback_target and at "
                    "least one test_id. A routing action may not carry "
                    "mutation_permissions, and human_clinical_review must declare "
                    "prohibited_claims."
                ),
            )
        ]
    return []


def _check_layer(patch: CandidatePatch) -> list[ValidationIssue]:
    """The patch must not operate above the layer its failure class requires."""
    required = required_layer(patch.failure_class)
    if LAYER_RANK[patch.layer] > LAYER_RANK[required]:
        return [
            ValidationIssue(
                check="layer_correctness",
                detail=(
                    f"failure class {patch.failure_class.value} requires a patch at "
                    f"{required.value} or lower; this patch operates at {patch.layer.value}"
                ),
                remedy=(
                    "Patch the required layer. A higher-layer change leaves the cause "
                    "in place, and the next scenario on the same cause fails again. If "
                    "the required layer genuinely cannot express the fix, that is a "
                    "taxonomy question and belongs in a reviewed change to the mapping, "
                    "not in a one-off exception here."
                ),
            )
        ]
    if LAYER_RANK[patch.layer] < LAYER_RANK[required]:
        # Lower than required is a stronger fix than necessary. Allowed, but worth
        # saying: a structural fix for a content-shaped cause is a much larger
        # intervention than the classification justified.
        return []
    return []


def _check_minimality(patch: CandidatePatch, bundle: PolicyBundle) -> tuple[list[ValidationIssue], list[str]]:
    """A patch must touch few rules, and must not touch a rule for another reason.

    Two distinct concerns, both about reviewability:

    * **Size.** Beyond ``MAX_RULES_PER_PATCH`` rules, a reader cannot hold the
      change in mind, and an unreviewable change is not really approved.
    * **Scope.** Every rule it touches must be named in the classification's
      evidence or be the rule the relevant oracle blames. A patch that also
      tidies an unrelated rule is a patch whose acceptance cannot be attributed.

    The second check is advisory rather than blocking when the patch is small.
    There are legitimate cases -- adding a guard to a neighbouring rule that
    shares a pattern -- and refusing them outright would push the author to split
    a coherent change in two, which makes review harder rather than easier.
    """
    touched = patch.touched_rule_ids()
    issues: list[ValidationIssue] = []
    notes: list[str] = []

    if len(touched) > MAX_RULES_PER_PATCH:
        issues.append(
            ValidationIssue(
                check="minimality_size",
                detail=f"patch touches {len(touched)} rules ({', '.join(touched)}); limit is {MAX_RULES_PER_PATCH}",
                remedy=(
                    "Split the patch. One patch addresses one cause; a single patch "
                    "spanning several rules cannot be attributed when one of them "
                    "regresses."
                ),
            )
        )

    implicated = patch.evidence_refs
    for rule_id in touched:
        if not any(rule_id in ref for ref in implicated):
            notes.append(
                f"rule {rule_id} is touched but is not named in the evidence refs "
                f"({list(implicated)}); confirm at review that it belongs to this cause"
            )
        rule = bundle.rule_by_id(rule_id)
        if rule is not None and not rule.test_ids:
            notes.append(
                f"rule {rule_id} declares no test_ids; the bundle will refuse to load "
                "until that is corrected"
            )

    return issues, notes


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def apply_patch(patch: CandidatePatch, bundle: PolicyBundle) -> PolicyBundle:
    """Render a patch into a new bundle.

    Composition is pure: the input bundle is not modified, and the output is a
    new bundle that has passed its own schema validation. If any operation cannot
    be applied, this raises -- it never partially applies. A partially applied
    patch would produce a bundle whose version claims a change it does not
    contain, which is worse than a failed patch because the artifact looks right.
    """
    if patch.base_policy_version != bundle.policy_version:
        raise ValueError(
            f"{patch.patch_id}: base {patch.base_policy_version} != bundle "
            f"{bundle.policy_version}; refusing to compose across versions"
        )

    rules = [r.model_copy(deep=True) for r in bundle.rules]
    templates = [t.model_copy(deep=True) for t in bundle.message_templates]

    def index_of(rule_id: str) -> int:
        for position, rule in enumerate(rules):
            if rule.rule_id == rule_id:
                return position
        raise ValueError(f"{patch.patch_id}: rule {rule_id} does not exist")

    for operation in patch.operations:
        if operation.op is OpKind.ADD_TRIGGER_PATTERN:
            position = index_of(operation.rule_id or "")
            rule = rules[position]
            existing = {(p.phrase, p.polarity) for p in rule.trigger_patterns}
            new_pattern = SentinelPattern(
                phrase=operation.value or "",
                polarity=operation.polarity or "POSITIVE",  # type: ignore[arg-type]
                category=operation.category or "UNSPECIFIED_HIGH_RISK",
            )
            if (new_pattern.phrase, new_pattern.polarity) in existing:
                raise ValueError(
                    f"{patch.patch_id}: pattern {new_pattern.phrase!r} with polarity "
                    f"{new_pattern.polarity} already exists on {rule.rule_id}"
                )
            rules[position] = rule.model_copy(
                update={"trigger_patterns": (*rule.trigger_patterns, new_pattern)}
            )

        elif operation.op is OpKind.ADD_PROHIBITED_CLAIM:
            position = index_of(operation.rule_id or "")
            rule = rules[position]
            claim = ProhibitedClaim(operation.value or "")
            if claim in rule.prohibited_claims:
                raise ValueError(
                    f"{patch.patch_id}: claim {claim.value} already declared on {rule.rule_id}"
                )
            rules[position] = rule.model_copy(
                update={"prohibited_claims": (*rule.prohibited_claims, claim)}
            )

        elif operation.op is OpKind.WIDEN_RULE_SCOPE:
            position = index_of(operation.rule_id or "")
            rule = rules[position]
            service_id = operation.value or ""
            if service_id in rule.applies_to_service_ids:
                raise ValueError(
                    f"{patch.patch_id}: service {service_id} already in scope for {rule.rule_id}"
                )
            rules[position] = rule.model_copy(
                update={
                    "applies_to_service_ids": (*rule.applies_to_service_ids, service_id)
                }
            )

        elif operation.op is OpKind.UPDATE_TEMPLATE_TEXT:
            for position, template in enumerate(templates):
                if template.template_id == operation.template_id:
                    templates[position] = template.model_copy(
                        update={"text": operation.template_text or template.text}
                    )
                    break
            else:
                raise ValueError(
                    f"{patch.patch_id}: template {operation.template_id} does not exist"
                )

        elif operation.op is OpKind.ADD_RULE:
            new_rule = PolicyRule.model_validate(operation.new_rule or {})
            if any(r.rule_id == new_rule.rule_id for r in rules):
                raise ValueError(
                    f"{patch.patch_id}: rule {new_rule.rule_id} already exists; use an "
                    "operation against the existing rule instead of ADD_RULE"
                )
            rules.append(new_rule)

        elif operation.op is OpKind.BUMP_REVIEW_DATE:
            position = index_of(operation.rule_id or "")
            rule = rules[position]
            rules[position] = rule.model_copy(
                update={"review_due": rule.review_due.replace(year=rule.review_due.year + 1)}
            )

        else:  # pragma: no cover -- the enum is closed
            raise ValueError(f"{patch.patch_id}: unsupported operation {operation.op}")

    # Version the result. The bundle's own validators run here, so an inconsistent
    # composition is caught rather than returned.
    return PolicyBundle.model_validate(
        {
            **bundle.model_dump(mode="json"),
            "policy_version": patch.target_policy_version,
            "rollback_target": patch.rollback_target,
            "rules": [r.model_dump(mode="json") for r in rules],
            "message_templates": [t.model_dump(mode="json") for t in templates],
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


#: Every check that can block, in the order it is run. Ordered cheapest-first so
#: a malformed patch is refused before any work is done on it.
def validate(
    patch: CandidatePatch,
    bundle: PolicyBundle,
    *,
    composed: PolicyBundle | None = None,
) -> ValidationResult:
    """Validate a patch against the bundle in force.

    ``composed`` lets a caller pass an already-rendered bundle to avoid rendering
    twice. When omitted, composition is attempted here -- and a composition
    failure is reported as a validation issue rather than raised, because a patch
    that cannot be composed is a patch the generator needs to correct.
    """
    result = ValidationResult(patch_id=patch.patch_id)

    result.issues.extend(_check_base_version(patch, bundle))
    result.issues.extend(_check_layer(patch))

    # Rule and value references need the base bundle, so they run regardless of
    # whether the base version matched: the messages are more useful that way.
    result.issues.extend(_check_rule_references(patch, bundle))

    minimality_issues, minimality_notes = _check_minimality(patch, bundle)
    result.issues.extend(minimality_issues)
    result.notes.extend(minimality_notes)

    if result.valid:
        try:
            result.composed_bundle = composed if composed is not None else apply_patch(patch, bundle)
        except Exception as exc:  # noqa: BLE001
            result.issues.append(
                ValidationIssue(
                    check="composition",
                    detail=f"the patch could not be composed: {exc}",
                    remedy=(
                        "Correct the operation so it applies cleanly. A patch that "
                        "cannot be composed has not been validated at all, because "
                        "validation runs on the composed result."
                    ),
                )
            )

    return result


__all__ = [
    "MAX_RULES_PER_PATCH",
    "ValidationIssue",
    "ValidationResult",
    "apply_patch",
    "validate",
]
