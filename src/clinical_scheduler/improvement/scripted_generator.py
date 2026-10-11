"""Scripted repair generators: deterministic stand-ins for a model-backed one.

The repair loop takes its candidate change from a caller-supplied generator, so
that the loop itself never contains generation logic and a model is never in a
position to apply its own output. In production that generator is a model
reading an :class:`~clinical_scheduler.improvement.rca.EvidencePackage`.

The generators here are **scripted**, and the name is chosen to make that
unmissable. A scripted generator holds a curated table of repairs and returns
the entry whose failure signature matches the package. It does not reason, and
it must not be mistaken for one that does: turning

    "I think I am having a heart attack"

into the pattern ``having a heart attack`` is natural-language understanding,
and a table lookup is not a substitute for it. What the table *does* provide is
the ability to exercise every mechanical stage of the loop -- static
validation, the target reproducer, the golden suite, the holdout comparison, the
gate, approval, artifact writing and checksums -- deterministically, with no
model, and to assert that the gate rejects specific bad patches rather than only
that it accepts good ones.

A model-backed generator produces a patch of the same *shape* by the same
contract; swapping it in changes no other part of the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from clinical_scheduler.improvement.optimizer import ApprovalRecord, GateDecision
from clinical_scheduler.improvement.patch_schema import CandidatePatch
from clinical_scheduler.improvement.rca import EvidencePackage
from clinical_scheduler.improvement.taxonomy import FailureClass, PatchLayer
from clinical_scheduler.improvement.validator import ValidationResult
from clinical_scheduler.policy.engine import PolicyEngine

__all__ = ["KnownRepair", "ScriptedGenerator", "approver_accept_when_gate_accepts"]


@dataclass(frozen=True)
class KnownRepair:
    """One curated repair, keyed by the failure signature it answers.

    The signature is deliberately narrow -- failure class and implicated rule --
    so that a repair can only be selected by a package that actually describes
    the failure it was written for. A generator that matched on the scenario id
    alone would produce a "fix" for any failure in that scenario, including one
    with a completely different cause.
    """

    failure_class: FailureClass
    implicated_rule_id: str
    layer: PatchLayer
    target_version: str
    operations: tuple[dict[str, Any], ...]
    new_regression_tests: tuple[str, ...]
    #: Why this repair is the one the table holds, in the author's words. Written
    #: into the patch's evidence references so a reviewer can see it is curated.
    note: str = ""

    def matches(self, package: EvidencePackage) -> bool:
        classification = package.classification
        if classification is None:
            return False
        if classification.failure_class != self.failure_class:
            return False
        # An implicated rule the package cannot name cannot select a repair that
        # patches a specific rule: the repair would be applied to a guess.
        return classification.implicated_rule_id == self.implicated_rule_id


@dataclass
class ScriptedGenerator:
    """Returns the first curated repair whose signature matches the package.

    ``patch_id`` is allocated deterministically from the number of patches
    issued, so two runs over the same failure produce the same patch id and their
    artifacts are comparable by name.
    """

    repairs: Sequence[KnownRepair]
    patch_id_prefix: str = "P"
    _issued: int = field(default=0, init=False, repr=False)

    def __call__(
        self, package: EvidencePackage, engine: PolicyEngine
    ) -> CandidatePatch | None:
        repair = next((r for r in self.repairs if r.matches(package)), None)
        if repair is None:
            return None
        return self.build(repair, package, engine)

    def build(
        self, repair: KnownRepair, package: EvidencePackage, engine: PolicyEngine
    ) -> CandidatePatch:
        """Materialise one repair against a specific package and bundle."""
        self._issued += 1
        base_version = engine.bundle.policy_version
        return CandidatePatch(
            patch_id=f"{self.patch_id_prefix}-{self._issued:04d}",
            target_scenario_id=package.scenario_id,
            failure_class=repair.failure_class,
            layer=repair.layer,
            base_policy_version=base_version,
            target_policy_version=repair.target_version,
            rollback_target=base_version,
            operations=repair.operations,
            new_regression_tests=repair.new_regression_tests,
            approver_role="clinical_safety_officer",
            evidence_refs=(package.package_id, repair.note or "scripted repair table"),
        )


def approver_accept_when_gate_accepts(
    *, approver_role: str = "clinical_safety_officer", rationale: str = ""
):
    """An approver that accepts exactly when the gate accepted.

    This is the *mechanical* approval policy, and it is the right one for a
    demonstration of the gate: an approver that could accept a patch the gate
    rejected would make the gate advisory, and an approver that could reject a
    patch the gate accepted would hide the gate's decision behind a second,
    undocumented one.

    It is still a human step in the sense that matters: the record says
    ``simulated=True``, and every artifact that reports the outcome repeats that
    the approval was simulated. A prototype that recorded an approval without
    saying so would invite a reader to treat the resulting version as
    production-approved.
    """

    def approve(
        patch: CandidatePatch, validation: ValidationResult, gate: GateDecision
    ) -> ApprovalRecord:
        accepted = gate.accepted
        return ApprovalRecord(
            patch_id=patch.patch_id,
            approver_role=approver_role,
            decision="ACCEPT" if accepted else "REJECT",
            simulated=True,
            rationale=rationale
            or (
                "Gate accepted: the target reproducer passes, no new development "
                "failure appeared, the false-escalation rate did not rise, and the "
                "holdout did not regress."
                if accepted
                else "Gate rejected: " + "; ".join(gate.reasons)
            ),
            patch_content_hash=patch.content_hash(),
        )

    return approve
