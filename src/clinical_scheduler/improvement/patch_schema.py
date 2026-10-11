"""Typed patch schema: what a candidate change is allowed to say.

The schema is the load-bearing safety mechanism of the whole loop, and the way it
works is by being *narrow*. A patch is not a diff, not a script, and not a
string that a model composed. It is a structured object with a closed set of
operations, each of which names a field that already exists in the policy schema.

Three properties follow from that, and each one closes a specific hazard:

**A patch cannot contain arbitrary code.** There is no operation that evaluates
anything. The worst a malformed patch can do is produce a policy bundle that
fails its own validation.

**A patch cannot touch the kernel, the broker or the oracles.** Those are the
layers where a "fix" could hide the evidence of its own success. The only file a
patch may modify is a policy bundle, and the only identifiers it may reference
are rule ids that already exist or that it declares in full.

**A patch cannot be silent about its provenance.** Every patch carries the
failure class it claims to address, the required layer for that class, and its
own rollback target. The validator then refuses a patch whose layer is higher
than the classification requires -- which is the mechanical form of plan.md
section 13.3's "a prompt patch is not the default answer to a systems failure".

The operations are expressed as *intents* rather than as text edits, so that a
reviewer reads "add a NEGATION guard to SAFE-002" rather than inspecting a
diff of quoted YAML. The renderer turns intents into a bundle; the reviewer
approves the intents.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from clinical_scheduler.domain.states import FrozenStrEnum, StrictModel
from clinical_scheduler.improvement.taxonomy import (
    FailureClass,
    LAYER_RANK,
    PatchLayer,
    required_layer,
)

#: Policy versions are ``YYYY.MM-<slug>``; the slug identifies the lineage.
POLICY_VERSION_PATTERN = r"^\d{4}\.\d{2}-[a-z0-9.\-]+$"


class OpKind(FrozenStrEnum):
    """The closed set of operations a patch may perform.

    Every one is confined to a policy bundle. There is deliberately no operation
    for adding a rule action, no operation for changing a tool contract, and no
    operation for editing a test -- each of those would let a patch move the goal
    rather than the system.
    """

    ADD_TRIGGER_PATTERN = "add_trigger_pattern"
    ADD_PROHIBITED_CLAIM = "add_prohibited_claim"
    WIDEN_RULE_SCOPE = "widen_rule_scope"
    UPDATE_TEMPLATE_TEXT = "update_template_text"
    ADD_RULE = "add_rule"
    BUMP_REVIEW_DATE = "bump_review_date"


class PatchOperation(StrictModel):
    """One change to one policy bundle.

    ``rationale`` is required and is not free-form decoration: it appears in the
    approval record, and the validator rejects a patch whose rationale does not
    name the failure class it addresses. That is a cheap check against a patch
    that was generated for one failure and quietly used to justify an unrelated
    edit.
    """

    op: OpKind
    #: The rule this operation applies to. Required for every op except
    #: ADD_RULE, which declares its rule in full via ``new_rule``.
    rule_id: str | None = Field(default=None, pattern=r"^[A-Z]+-\d{3}$")
    #: For ADD_TRIGGER_PATTERN: the phrase to match, already in normalised
    #: surface form. For ADD_PROHIBITED_CLAIM: the claim name.
    value: str | None = None
    #: For ADD_TRIGGER_PATTERN: the polarity of the new pattern.
    polarity: Literal["POSITIVE", "NEGATION", "HISTORICAL", "THIRD_PERSON", "META"] | None = None
    category: str | None = None
    #: For UPDATE_TEMPLATE_TEXT: the template id and the new body.
    template_id: str | None = None
    template_text: str | None = None
    #: For ADD_RULE: the complete new rule as a mapping, validated against
    #: ``PolicyRule`` by the validator before the patch is accepted. Kept as a
    #: mapping rather than a typed model so that a malformed new rule fails
    #: policy validation with the same error a hand-authored bundle would give.
    new_rule: dict[str, Any] | None = None
    #: Why this operation, in terms of the failure. Must mention the failure
    #: class value, which the validator checks.
    rationale: str = Field(min_length=20)

    @model_validator(mode="after")
    def _required_fields_per_op(self) -> Self:
        """Fill in what each operation needs, and refuse what it does not accept.

        Two directions, both needed. A missing field would produce an operation
        that silently does nothing -- a patch that claims to fix a failure and
        changes nothing, which is worse than a rejected patch because it looks
        like a successful repair. An extra field is a sign the author expected an
        operation this schema does not implement, and guessing at their intent is
        how a narrow schema stops being narrow.
        """
        needs: dict[OpKind, tuple[str, ...]] = {
            OpKind.ADD_TRIGGER_PATTERN: ("rule_id", "value", "polarity"),
            OpKind.ADD_PROHIBITED_CLAIM: ("rule_id", "value"),
            OpKind.WIDEN_RULE_SCOPE: ("rule_id", "value"),
            OpKind.UPDATE_TEMPLATE_TEXT: ("template_id", "template_text"),
            OpKind.ADD_RULE: ("new_rule",),
            OpKind.BUMP_REVIEW_DATE: ("rule_id",),
        }[self.op]

        missing = [name for name in needs if getattr(self, name) in (None, "")]
        if missing:
            raise ValueError(
                f"{self.op.value} requires {missing}; an operation with a missing "
                "field would render to a no-op and report as a successful repair"
            )

        # The polarity is only meaningful for a pattern, and a pattern without a
        # polarity would default to POSITIVE -- which is exactly the default that
        # turns a negation guard into a trigger.
        if self.op is not OpKind.ADD_TRIGGER_PATTERN and self.polarity is not None:
            raise ValueError(f"{self.op.value} must not carry a polarity")
        if self.op is not OpKind.ADD_RULE and self.new_rule is not None:
            raise ValueError(f"{self.op.value} must not carry new_rule")
        if self.op is not OpKind.UPDATE_TEMPLATE_TEXT and self.template_text is not None:
            raise ValueError(f"{self.op.value} must not carry template_text")
        return self


class CandidatePatch(StrictModel):
    """A proposed repair: typed, minimal, and traceable to one failure.

    ``target_scenario_id`` is singular on purpose. A patch that addresses several
    scenarios at once cannot be attributed when it regresses one of them, and the
    acceptance gate's whole job is attribution. If several scenarios share a
    cause, that is several patches on one shared rule -- which the renderer
    handles by composing them in order.
    """

    patch_id: str = Field(pattern=r"^P-\d{4}$")
    #: The scenario this patch exists to repair.
    target_scenario_id: str = Field(pattern=r"^S\d{2}$|^[A-Z]{2,4}-\d{2}$")
    #: The failure class this patch claims to address.
    failure_class: FailureClass
    #: The layer this patch operates at. The validator compares this against the
    #: required layer for ``failure_class``.
    layer: PatchLayer
    #: The policy version this patch was generated against. A patch applied to a
    #: different base is refused: the rule it names may not exist there, or may
    #: have changed, and either way the patch's evidence no longer applies.
    base_policy_version: str = Field(pattern=POLICY_VERSION_PATTERN)
    #: The version this patch produces when accepted.
    target_policy_version: str = Field(pattern=POLICY_VERSION_PATTERN)
    #: The version to return to if the patch is rolled back. Must name a real
    #: version; existence is checked by the validator against the loadable
    #: bundles.
    rollback_target: str = Field(pattern=POLICY_VERSION_PATTERN)
    operations: tuple[PatchOperation, ...] = Field(min_length=1)
    #: The regression tests this patch adds. Required and non-empty: acceptance
    #: without a test means the same failure can return unnoticed.
    new_regression_tests: tuple[str, ...] = Field(min_length=1)
    #: Owner who must approve. Drawn from the policy's own owner vocabulary.
    approver_role: str = Field(min_length=3)
    #: The evidence this patch is based on -- typically the RCA package id.
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def _target_version_must_differ(self) -> Self:
        """A patch that produces the version it started from changes nothing.

        Catches the case where a generator built a patch, then a second
        generation run produced the same target version and silently shadowed the
        first. Two patches cannot share a produced version, because the accepted
        bundle is identified by it.
        """
        if self.target_policy_version == self.base_policy_version:
            raise ValueError(
                f"{self.patch_id}: target_policy_version equals base_policy_version "
                f"({self.base_policy_version}); the patch would produce no new version"
            )
        if self.rollback_target == self.target_policy_version:
            raise ValueError(
                f"{self.patch_id}: rollback_target equals target_policy_version; "
                "rolling back would be a no-op"
            )
        return self

    @model_validator(mode="after")
    def _layer_must_not_exceed_the_required_layer(self) -> Self:
        """The mechanical form of "do not default to the prompt".

        A patch may operate at or below the layer its failure class requires. It
        may not operate above: a layer *below* the required one would be a
        stronger fix than needed (allowed, and noted), while a layer above is the
        shortcut this rule exists to prevent.

        The comparison is by rank, where rank 1 is the database and rank 7 is the
        model, so "above" means a higher number.
        """
        required = required_layer(self.failure_class)
        if LAYER_RANK[self.layer] > LAYER_RANK[required]:
            raise ValueError(
                f"{self.patch_id}: failure class {self.failure_class.value} requires a "
                f"patch at {required.value} or lower, but this patch operates at "
                f"{self.layer.value}. A higher-layer patch leaves the cause in place; "
                "see the layer_reason for this class."
            )
        return self

    @model_validator(mode="after")
    def _rationales_must_name_the_failure_class(self) -> Self:
        """Each operation's rationale must name the class it addresses.

        A cheap guard against a patch generated for one failure being reused to
        justify an unrelated edit. It is a string check and can be satisfied
        dishonestly; its value is that satisfying it dishonestly requires writing
        down the wrong claim in a record a human reads.
        """
        token = self.failure_class.value.lower()
        for operation in self.operations:
            if token not in operation.rationale.lower():
                raise ValueError(
                    f"{self.patch_id}/{operation.op.value}: rationale must name "
                    f"{self.failure_class.value!r} so the operation is traceable to "
                    "the failure it claims to fix"
                )
        return self

    def content_hash(self) -> str:
        """Hash of the patch's *content*, excluding the timestamps.

        Used to detect a patch that was edited after approval: the approval
        record stores this value, and a mismatch means the artifact a reviewer
        approved is not the artifact being applied.
        """
        payload = self.model_dump(mode="json")
        payload.pop("created_at", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def touched_rule_ids(self) -> tuple[str, ...]:
        """Every rule id this patch references, for the minimality check."""
        ids: list[str] = []
        for operation in self.operations:
            if operation.rule_id:
                ids.append(operation.rule_id)
            elif operation.new_rule:
                rule_id = operation.new_rule.get("rule_id")
                if isinstance(rule_id, str):
                    ids.append(rule_id)
        return tuple(dict.fromkeys(ids))

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True, default=str)


__all__ = [
    "CandidatePatch",
    "OpKind",
    "POLICY_VERSION_PATTERN",
    "PatchOperation",
]
