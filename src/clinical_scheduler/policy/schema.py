"""Policy bundle schema (plan.md section 6).

Policy is data. Every rule carries the metadata needed for a governance review:
who owns it, why it exists, when it takes effect, what evidence backs it, which
tests cover it, and how to roll it back.

This module defines the *shape*. It does not contain any rule content, and it
deliberately has no fields for clinical thresholds. If a future author wants to
encode "stop the blood thinner 7 days before", there is no field that accepts
it -- the refusal is structural rather than a matter of reviewer vigilance.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from clinical_scheduler.domain.states import FrozenStrEnum, StrictModel


class RuleOwner(FrozenStrEnum):
    """Who is accountable for a rule. plan.md section 6 requires an owner per
    rule; a rule with no accountable owner cannot be approved."""

    CLINICAL_SAFETY_OFFICER = "clinical_safety_officer"
    CLINIC_MEDICAL_DIRECTOR = "clinic_medical_director"
    PRIVACY_LEGAL = "privacy_legal"
    OPERATIONS_MANAGER = "operations_manager"
    ENGINEERING_LEAD = "engineering_lead"


class RuleAction(FrozenStrEnum):
    """What the system does when a rule fires. Deliberately coarse: rules route
    or block, they never diagnose or prescribe."""

    URGENT_HANDOFF = "urgent_handoff"
    CRISIS_HANDOFF = "crisis_handoff"
    HUMAN_CLINICAL_REVIEW = "human_clinical_review"
    SPECIALIST_HANDOFF = "specialist_handoff"
    REQUIRE_ACKNOWLEDGEMENT = "require_acknowledgement"
    BLOCK_MUTATION = "block_mutation"
    ALLOW = "allow"


class ProhibitedClaim(FrozenStrEnum):
    """Claims the response layer must never emit when a rule is active.

    This is enforced by the response-grounding validator, not by prompt text.
    A prompt instruction can be argued with; a validator cannot.
    """

    STOP_MEDICATION = "stop_medication"
    CHANGE_DOSE = "change_dose"
    START_MEDICATION = "start_medication"
    SKIP_MEDICATION = "skip_medication"
    BRIDGE_MEDICATION = "bridge_medication"
    CONFIRM_DIAGNOSIS = "confirm_diagnosis"
    RULE_OUT_CONDITION = "rule_out_condition"
    EMERGENCY_SERVICES_CONTACTED = "emergency_services_contacted"
    GUARANTEE_COVERAGE = "guarantee_coverage"
    GUARANTEE_COST = "guarantee_cost"


class SentinelPattern(StrictModel):
    """A deterministic high-recall pattern.

    ``pattern`` is matched case-insensitively against normalised patient text.
    ``polarity`` says what the pattern means when found:

    * ``POSITIVE``   -- the phrase indicates the concern is present right now;
    * ``NEGATION``   -- the phrase cancels a nearby positive ("I do not have...");
    * ``HISTORICAL`` -- the concern is in the past and resolved;
    * ``THIRD_PERSON`` -- the concern belongs to someone else;
    * ``META``       -- the phrase is about the concept, not an experience.
    """

    phrase: str = Field(min_length=3)
    polarity: Literal["POSITIVE", "NEGATION", "HISTORICAL", "THIRD_PERSON", "META"] = "POSITIVE"
    category: str = "UNSPECIFIED_HIGH_RISK"


class PolicyRule(StrictModel):
    rule_id: str = Field(pattern=r"^[A-Z]+-\d{3}$")
    owner: RuleOwner
    rationale: str = Field(min_length=10)
    action: RuleAction
    jurisdiction: str
    effective_from: date
    effective_until: date | None = None
    review_due: date
    #: Evidence for why this rule takes the form it does. For routed clinical
    #: concerns this must point at clinician-authored material, never at model
    #: output.
    evidence_source: str = Field(min_length=5)
    test_ids: tuple[str, ...] = ()
    rollback_target: str = Field(min_length=3)
    trigger_patterns: tuple[SentinelPattern, ...] = ()
    #: Empty for every rule that must not permit a state change.
    mutation_permissions: tuple[str, ...] = ()
    prohibited_claims: tuple[ProhibitedClaim, ...] = ()
    message_template_id: str | None = None
    applies_to_service_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _safety_rules_may_not_grant_mutations(self) -> Self:
        """A routing rule that escalates a concern must never also grant a
        mutation permission. Allowing both would create a path where an
        escalation and a booking coexist, which invariant 2 forbids."""
        routing = {
            RuleAction.URGENT_HANDOFF,
            RuleAction.CRISIS_HANDOFF,
            RuleAction.HUMAN_CLINICAL_REVIEW,
            RuleAction.SPECIALIST_HANDOFF,
            RuleAction.BLOCK_MUTATION,
        }
        if self.action in routing and self.mutation_permissions:
            raise ValueError(
                f"{self.rule_id}: a {self.action.value} rule must not carry mutation "
                f"permissions (got {self.mutation_permissions})"
            )
        return self

    @model_validator(mode="after")
    def _prohibited_claims_required_for_clinical_review(self) -> Self:
        if self.action is RuleAction.HUMAN_CLINICAL_REVIEW and not self.prohibited_claims:
            raise ValueError(
                f"{self.rule_id}: human_clinical_review must declare prohibited_claims "
                "so the response validator has something to enforce"
            )
        return self



class MessageTemplate(StrictModel):
    """Patient-facing messaging. Locale-aware and explicit about what the system
    did or did not do (plan.md section 3.5)."""

    template_id: str = Field(min_length=3)
    locale: str = Field(min_length=2)
    text: str = Field(min_length=5)
    #: True when the message tells the patient to contact emergency services.
    #: The response validator uses this to enforce that the agent never asserts
    #: a service was contacted on the patient's behalf.
    directs_to_emergency_services: bool = False
    crisis_service_number: str | None = None
    emergency_number: str | None = None


class PolicyBundle(StrictModel):
    """A complete, signed, versioned policy bundle."""

    policy_version: str = Field(pattern=r"^\d{4}\.\d{2}-[a-z0-9.\-]+$")
    jurisdiction: str = Field(min_length=2)
    intended_use: Literal["administrative_scheduling"] = "administrative_scheduling"
    effective_from: date
    approved_by: tuple[str, ...] = ()
    rollback_target: str = Field(min_length=3)
    #: Locale used when the deployment locale is unknown. Emergency numbers must
    #: never be guessed at runtime (plan.md section 3.5).
    default_locale: str = Field(min_length=2)
    rules: tuple[PolicyRule, ...] = ()
    message_templates: tuple[MessageTemplate, ...] = ()

    @field_validator("rules")
    @classmethod
    def _unique_rule_ids(cls, v: tuple[PolicyRule, ...]) -> tuple[PolicyRule, ...]:
        ids = [r.rule_id for r in v]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate rule_id(s): {dupes}")
        return v

    @field_validator("message_templates")
    @classmethod
    def _unique_template_ids(cls, v: tuple[MessageTemplate, ...]) -> tuple[MessageTemplate, ...]:
        ids = [t.template_id for t in v]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate template_id(s): {dupes}")
        return v

    @model_validator(mode="after")
    def _rule_references_resolve(self) -> PolicyBundle:
        """Every referenced template must exist, and every rule must be
        reviewable: an unowned or unreviewable rule is a governance failure."""
        template_ids = {t.template_id for t in self.message_templates}
        problems: list[str] = []
        for rule in self.rules:
            if rule.message_template_id and rule.message_template_id not in template_ids:
                problems.append(f"{rule.rule_id} references missing template {rule.message_template_id}")
            if rule.effective_until and rule.effective_until < rule.effective_from:
                problems.append(f"{rule.rule_id} effective_until precedes effective_from")
            if not rule.test_ids:
                problems.append(f"{rule.rule_id} declares no test_ids")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def active_rules(self, *, on: date | None = None) -> tuple[PolicyRule, ...]:
        today = on or datetime.now(timezone.utc).date()
        return tuple(
            r
            for r in self.rules
            if r.effective_from <= today and (r.effective_until is None or today <= r.effective_until)
        )

    def rules_requiring_review(self, *, on: date | None = None) -> tuple[PolicyRule, ...]:
        """Rules past their review date. A deployment should surface these; for
        the demo they are reported as governance evidence."""
        today = on or datetime.now(timezone.utc).date()
        return tuple(r for r in self.rules if r.review_due < today)

    def rule_by_id(self, rule_id: str) -> PolicyRule | None:
        return next((r for r in self.rules if r.rule_id == rule_id), None)

    def template_by_id(self, template_id: str) -> MessageTemplate | None:
        return next((t for t in self.message_templates if t.template_id == template_id), None)

    def fingerprint(self) -> dict[str, Any]:
        """Serialisable content used to compute the bundle hash recorded in the
        run manifest and in every audit row."""
        return {
            "policy_version": self.policy_version,
            "jurisdiction": self.jurisdiction,
            "rules": [
                {
                    "id": r.rule_id,
                    "owner": r.owner.value,
                    "action": r.action.value,
                    "effective_from": r.effective_from.isoformat(),
                    "patterns": [
                        {"phrase": p.phrase, "polarity": p.polarity, "category": p.category}
                        for p in r.trigger_patterns
                    ],
                    "mutation_permissions": list(r.mutation_permissions),
                    "prohibited_claims": [c.value for c in r.prohibited_claims],
                }
                for r in self.rules
            ],
            "templates": [
                {
                    "id": t.template_id,
                    "locale": t.locale,
                    "directs": t.directs_to_emergency_services,
                    "crisis": t.crisis_service_number,
                    "emergency": t.emergency_number,
                }
                for t in self.message_templates
            ],
        }


__all__ = [
    "MessageTemplate",
    "PolicyBundle",
    "PolicyRule",
    "ProhibitedClaim",
    "RuleAction",
    "RuleOwner",
    "SentinelPattern",
]
