"""The curated repair table used by the demonstration and by the loop tests.

This module exists so that the demonstration is *reproducible from the
repository* rather than reconstructed in a shell history. It holds one entry per
repair the demonstration performs, keyed on the failure signature it answers.

It is deliberately not a general-purpose fixer. Each entry is a claim about one
specific, measured failure, written by hand, and reviewable as such:

``ADV_11_NAMED_CONDITION``
    The baseline's ``SAFE-001`` describes cardiac symptoms (``chest pain``,
    ``cannot breathe``) but has no phrase for a patient who **names the
    condition**. "I think I am having a heart attack" therefore routes CLEAR.

    The repair adds the positive phrase *and* its negation guard in the same
    patch. The guard is not optional polish: adding a bare positive makes
    "I am not having a heart attack" escalate, which trades a missed emergency
    for an over-triage on exactly the people saying they are fine. A recall fix
    that ships without its guard is the failure mode the false-escalation rate
    exists to catch, and the gate will reject the patch without it.
"""

from __future__ import annotations

from clinical_scheduler.improvement.scripted_generator import KnownRepair
from clinical_scheduler.improvement.taxonomy import FailureClass, PatchLayer

#: The failure class each operation's rationale must name, lowercased. The patch
#: schema checks for it, so it is written once here rather than repeated inline
#: where a typo would look like a policy decision.
_CLASS_TOKEN = FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE.value.lower()

#: Version the baseline is repaired to. Kept in the demo series so a reader can
#: see the repaired bundle is a *demo* artifact and not a production version.
REPAIRED_VERSION = "2026.10-demo.2"


ADV_11_NAMED_CONDITION = KnownRepair(
    failure_class=FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE,
    implicated_rule_id="SAFE-001",
    layer=PatchLayer.L4_POLICY_DATA,
    target_version=REPAIRED_VERSION,
    operations=(
        {
            "op": "add_trigger_pattern",
            "rule_id": "SAFE-001",
            "value": "having a heart attack",
            "polarity": "POSITIVE",
            "category": "MEDICAL_EMERGENCY",
            "rationale": (
                "Fixes the safety_sentinel_false_negative measured on ADV-11: the "
                "patient names the condition rather than describing a symptom, and "
                "SAFE-001 had no phrase for it, so a possible myocardial infarction "
                "routed to routine booking. This is a pattern-set gap, so the repair "
                "belongs at the policy-data layer where the clinical safety officer "
                "who owns the rule can review and roll it back."
            ),
        },
        {
            "op": "add_trigger_pattern",
            "rule_id": "SAFE-001",
            "value": "not having a heart attack",
            "polarity": "NEGATION",
            "category": "MEDICAL_EMERGENCY",
            "rationale": (
                "Ships with the positive above, because the safety_sentinel_false_negative "
                "repair introduces a substring that a denial also contains. Without this "
                "guard, 'I am not having a heart attack' escalates: the patch would trade a "
                "missed emergency for an over-triage on the patients who are explicitly "
                "saying they are well, which is the trade the false-escalation rate exists "
                "to refuse."
            ),
        },
    ),
    new_regression_tests=(
        "ADV-11",
        "counterfactual-negative: 'I am not having a heart attack' must not escalate",
        "S04: 'I had a heart attack five years ago' must remain CLEAR_HISTORICAL",
    ),
    note=(
        "Curated repair for the measured ADV-11 false negative; adds a condition-naming "
        "pattern and its negation guard together"
    ),
)


#: The table a demonstration run uses, in the order repairs are offered.
DEMO_REPAIR_TABLE: tuple[KnownRepair, ...] = (ADV_11_NAMED_CONDITION,)


__all__ = ["ADV_11_NAMED_CONDITION", "DEMO_REPAIR_TABLE", "REPAIRED_VERSION", "_CLASS_TOKEN"]
