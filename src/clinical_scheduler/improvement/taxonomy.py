"""Failure taxonomy and the mapping from cause to the layer that must be patched.

Two things live here, and they are kept apart on purpose:

* the **taxonomy** -- a closed set of failure classes, mirroring plan.md section
  13.1, so that a failure can be named consistently across runs;
* the **routing** -- for each class, the lowest layer at which the cause can be
  eliminated, mirroring the preference order in plan.md section 13.3.

The routing is the part that matters. Plan.md section 13.3 gives a strict
preference order:

    1. database constraint
    2. tool contract / validation
    3. FSM transition or capability restriction
    4. policy-as-data rule
    5. retrieval content / template
    6. prompt instruction / example
    7. model change (last resort)

and adds the sentence that makes the ordering load-bearing: "A prompt patch is
not the default answer to a systems failure." An improvement loop's dominant
failure mode is that the prompt is the easiest thing to edit, so it is what gets
edited. Encoding the preference as data, with a stated reason per class, is what
turns that sentence from an aspiration into something a reviewer can check
against a patch.

A note on why each class maps to *one* primary layer rather than a list. A list
would let a generator pick the most convenient entry, which is the whole problem.
Where a class genuinely has two plausible homes, the mapping records the lower
one and the higher one is available only through an explicit escalation that the
validator requires a justification for -- so choosing it is a visible decision,
not an invisible default.
"""

from __future__ import annotations

from typing import Mapping

from clinical_scheduler.domain.states import FrozenStrEnum, StrictModel


class FailureClass(FrozenStrEnum):
    """The closed set of failure classes (plan.md section 13.1).

    Closed on purpose. An open set means "unclassified" becomes the most common
    class, and an unclassified failure is one whose owning layer nobody decided.
    A failure that genuinely does not fit is a signal to widen the taxonomy
    deliberately, which is a reviewed change rather than a per-failure judgement.
    """

    CLINICAL_POLICY_GAP = "CLINICAL_POLICY_GAP"
    SAFETY_SENTINEL_FALSE_NEGATIVE = "SAFETY_SENTINEL_FALSE_NEGATIVE"
    SAFETY_SENTINEL_FALSE_POSITIVE = "SAFETY_SENTINEL_FALSE_POSITIVE"
    IDENTITY_AUTHORIZATION_FAILURE = "IDENTITY_AUTHORIZATION_FAILURE"
    PRIVACY_DISCLOSURE = "PRIVACY_DISCLOSURE"
    STATE_MACHINE_BYPASS = "STATE_MACHINE_BYPASS"
    TOOL_CONTRACT_ERROR = "TOOL_CONTRACT_ERROR"
    TRANSACTION_CONCURRENCY_ERROR = "TRANSACTION_CONCURRENCY_ERROR"
    IDEMPOTENCY_ERROR = "IDEMPOTENCY_ERROR"
    TIMEZONE_TIME_ERROR = "TIMEZONE_TIME_ERROR"
    RESPONSE_GROUNDING_ERROR = "RESPONSE_GROUNDING_ERROR"
    PROMPT_INJECTION_SUCCESS = "PROMPT_INJECTION_SUCCESS"
    ACCESSIBILITY_QUALITY_FAILURE = "ACCESSIBILITY_QUALITY_FAILURE"
    ORACLE_ERROR = "ORACLE_ERROR"
    FIXTURE_ERROR = "FIXTURE_ERROR"
    MODEL_FLAKINESS = "MODEL_FLAKINESS"
    INTEGRATION_OUTAGE = "INTEGRATION_OUTAGE"


class PatchLayer(FrozenStrEnum):
    """Where a change can be made, in the preference order of plan.md 13.3.

    The names carry their rank so that ordering is not something a reader has to
    look up, and so a comparison is always available: a patch at a higher-ranked
    layer than the classification requires is a legitimate reason to reject it.
    """

    L1_DATABASE_CONSTRAINT = "L1_database_constraint"
    L2_TOOL_CONTRACT = "L2_tool_contract"
    L3_FSM_OR_CAPABILITY = "L3_fsm_or_capability"
    L4_POLICY_DATA = "L4_policy_data"
    L5_TEMPLATE_CONTENT = "L5_template_content"
    L6_PROMPT = "L6_prompt"
    L7_MODEL = "L7_model"


#: Rank, lowest first. Used to compare a classification's required layer with the
#: layer a candidate actually patches.
LAYER_RANK: Mapping[PatchLayer, int] = {
    PatchLayer.L1_DATABASE_CONSTRAINT: 1,
    PatchLayer.L2_TOOL_CONTRACT: 2,
    PatchLayer.L3_FSM_OR_CAPABILITY: 3,
    PatchLayer.L4_POLICY_DATA: 4,
    PatchLayer.L5_TEMPLATE_CONTENT: 5,
    PatchLayer.L6_PROMPT: 6,
    PatchLayer.L7_MODEL: 7,
}


class FailureClassification(StrictModel):
    """A failure class plus the reasoning that put it there.

    ``evidence`` and ``counter_hypotheses`` are required. This is the part that
    makes the output reviewable: a reviewer needs to see not only what the cause
    was claimed to be, but which alternative causes were considered and why they
    were rejected. A classification with no alternatives listed is a guess that
    has been promoted to a fact.
    """

    failure_class: FailureClass
    #: The scenario the failure came from, when the classification is per-scenario.
    scenario_id: str | None = None
    #: Oracle ids or trace facts that led to this class. Never empty.
    evidence: tuple[str, ...]
    #: Alternative classes considered and the reason each was rejected.
    counter_hypotheses: tuple[str, ...] = ()
    #: The lowest layer at which the cause can be eliminated.
    required_layer: PatchLayer
    #: Why that layer, stated in terms of what would still be wrong if a higher
    #: layer were patched instead.
    layer_reason: str
    #: The policy rule implicated, when the cause is policy-data shaped.
    implicated_rule_id: str | None = None
    #: Free-text hypothesis, for the RCA package. Not used for routing.
    hypothesis: str

    @property
    def required_rank(self) -> int:
        return LAYER_RANK[self.required_layer]


# ---------------------------------------------------------------------------
# Cause -> layer routing
# ---------------------------------------------------------------------------

#: For each class: the layer that must be patched, and the sentence explaining
#: what a patch at a higher layer would leave unfixed. The second element is not
#: decoration -- it is the text that appears in a rejection message when a
#: candidate patches the wrong layer, and it has to be specific enough to be
#: actionable.
_ROUTING: Mapping[FailureClass, tuple[PatchLayer, str]] = {
    FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE: (
        PatchLayer.L4_POLICY_DATA,
        "A missed high-risk route is a gap in the clinician-authored pattern set. "
        "A prompt instruction not to miss it cannot be tested, cannot be reviewed "
        "by the safety officer who owns the rule, and cannot be rolled back.",
    ),
    FailureClass.SAFETY_SENTINEL_FALSE_POSITIVE: (
        PatchLayer.L4_POLICY_DATA,
        "An over-triggering pattern is a precision defect in the same pattern set. "
        "The counterfactual negation or historical form must be declared as data, "
        "because the guard is what the next reviewer has to inspect.",
    ),
    FailureClass.CLINICAL_POLICY_GAP: (
        PatchLayer.L4_POLICY_DATA,
        "A missing routing rule is a gap in the policy bundle itself. It must be "
        "authored with an owner, rationale, evidence source and rollback target, "
        "none of which a prompt change carries.",
    ),
    FailureClass.STATE_MACHINE_BYPASS: (
        PatchLayer.L3_FSM_OR_CAPABILITY,
        "A transition that happened outside the declared table is a kernel defect. "
        "Softening the prompt leaves the code path that allowed it exactly where "
        "it was, and the next caller reaches it too.",
    ),
    FailureClass.IDENTITY_AUTHORIZATION_FAILURE: (
        PatchLayer.L3_FSM_OR_CAPABILITY,
        "An action taken at the wrong tier is a capability-broker defect. The "
        "tier check is enforced in code precisely so that it does not depend on "
        "the conversation, and a prompt change would make it depend on it again.",
    ),
    FailureClass.PRIVACY_DISCLOSURE: (
        PatchLayer.L3_FSM_OR_CAPABILITY,
        "A disclosure is a broker or redaction defect: either a tool returned a "
        "field it must not, or a field reached the patient-visible text. Both are "
        "structural, and both are caught by the tool payload rather than by "
        "wording.",
    ),
    FailureClass.PROMPT_INJECTION_SUCCESS: (
        PatchLayer.L3_FSM_OR_CAPABILITY,
        "If an instruction in patient text changed what the system did, then "
        "patient text is a control channel. The fix is to remove the channel -- "
        "broker exposure or data/instruction separation -- not to ask the model "
        "more firmly to ignore it.",
    ),
    FailureClass.TOOL_CONTRACT_ERROR: (
        PatchLayer.L2_TOOL_CONTRACT,
        "A malformed or mis-validated tool call is an interface defect. The "
        "contract is where the guarantee belongs, so that every caller gets it "
        "rather than only the callers a prompt happened to describe.",
    ),
    FailureClass.TRANSACTION_CONCURRENCY_ERROR: (
        PatchLayer.L1_DATABASE_CONSTRAINT,
        "A race is a data-integrity defect. Only a constraint that holds under "
        "concurrent writers fixes it; a partial unique index or a transactional "
        "boundary is the guarantee, and anything softer is a narrowing of the "
        "window rather than a closure of it.",
    ),
    FailureClass.IDEMPOTENCY_ERROR: (
        PatchLayer.L1_DATABASE_CONSTRAINT,
        "A duplicate write under a repeated request is an integrity defect. The "
        "uniqueness of the idempotency key is the enforcement, and it has to hold "
        "at the database or it does not hold.",
    ),
    FailureClass.TIMEZONE_TIME_ERROR: (
        PatchLayer.L2_TOOL_CONTRACT,
        "A shifted instant is a contract defect: an instant that is not tz-aware, "
        "or a rendering that uses the server zone. The type and the conversion "
        "are where the guarantee belongs.",
    ),
    FailureClass.RESPONSE_GROUNDING_ERROR: (
        PatchLayer.L5_TEMPLATE_CONTENT,
        "A claim reached the patient that no grounded fact authorised. If the "
        "claim key is known and the template should never assert it, the template "
        "is the fix; if the claim is arbitrary, the defect is in the grounding "
        "mechanism and this classification is wrong -- see the counter-hypothesis "
        "recorded with the finding.",
    ),
    FailureClass.ACCESSIBILITY_QUALITY_FAILURE: (
        PatchLayer.L5_TEMPLATE_CONTENT,
        "A readability or verbosity problem is a content problem. It is graded "
        "S2 or S3 by construction and never vetoes, so the fix belongs in the "
        "wording rather than in anything structural.",
    ),
    FailureClass.INTEGRATION_OUTAGE: (
        PatchLayer.L5_TEMPLATE_CONTENT,
        "An unavailable dependency is an operational event, not a defect. The "
        "only thing to improve is how honestly the degradation is described, "
        "which is a template change; the system must not learn to invent "
        "availability during an outage.",
    ),
    FailureClass.ORACLE_ERROR: (
        PatchLayer.L2_TOOL_CONTRACT,
        "An oracle that raised is a defect in the measuring instrument, not in "
        "the system. It is routed to the contract layer by convention to keep it "
        "out of the policy path, and the acceptance gate treats any oracle error "
        "as an ineligible run regardless of which layer is patched.",
    ),
    FailureClass.FIXTURE_ERROR: (
        PatchLayer.L2_TOOL_CONTRACT,
        "A fixture that is not deterministic invalidates every comparison built "
        "on it. Fixing it is a prerequisite for measurement, not an improvement.",
    ),
    FailureClass.MODEL_FLAKINESS: (
        PatchLayer.L6_PROMPT,
        "Only a genuinely nondeterministic model behaviour -- one that varies "
        "across seeds with identical inputs and no other cause -- justifies "
        "touching the prompt. This is the one class where the prompt is the right "
        "layer, and reaching it requires the multi-seed evidence in plan.md "
        "section 11.4 to show the variance is real.",
    ),
}


def required_layer(failure_class: FailureClass) -> PatchLayer:
    """The lowest layer at which this class of cause can be eliminated."""
    return _ROUTING[failure_class][0]


def layer_reason(failure_class: FailureClass) -> str:
    """Why that layer, in terms of what a higher-layer patch would leave unfixed."""
    return _ROUTING[failure_class][1]


# ---------------------------------------------------------------------------
# Classifying a finding
# ---------------------------------------------------------------------------

#: Oracle id -> failure class, by **exact id**. There is deliberately no prefix
#: matching: an id that is not listed here is not guessed at, because a wrong
#: classification routes a repair to the wrong layer, and a wrong-layer patch is
#: the failure this module exists to prevent.
_ORACLE_ID_TO_CLASS: Mapping[str, FailureClass] = {
    "L1.INV14_WRONG_PATIENT_DISCLOSURE": FailureClass.PRIVACY_DISCLOSURE,
    "L1.CROSS_SESSION_CONTAMINATION": FailureClass.PRIVACY_DISCLOSURE,
    "L1.INV02_MUTATION_OFFERED_UNDER_SAFETY_ROUTE": FailureClass.STATE_MACHINE_BYPASS,
    "L1.INV02_MUTATION_EXECUTED_UNDER_SAFETY_ROUTE": FailureClass.STATE_MACHINE_BYPASS,
    "L1.INV02_BOOKED_UNDER_SAFETY_ROUTE": FailureClass.PRIVACY_DISCLOSURE,
    "L1.INV06_HOLD_DESCRIBED_AS_BOOKING": FailureClass.RESPONSE_GROUNDING_ERROR,
    "L1.INV08_COMMIT_WITHOUT_CONFIRMATION": FailureClass.STATE_MACHINE_BYPASS,
    "L1.INV10_BLIND_COMMIT_RETRY": FailureClass.IDEMPOTENCY_ERROR,
    "L1.INV12_FABRICATED_ESCALATION": FailureClass.RESPONSE_GROUNDING_ERROR,
    "L1.INV12_MEDICATION_DIRECTIVE": FailureClass.CLINICAL_POLICY_GAP,
    "L1.INV12_DIAGNOSTIC_CLAIM": FailureClass.CLINICAL_POLICY_GAP,
    "L2.COMMIT_STATE_MISMATCH": FailureClass.TRANSACTION_CONCURRENCY_ERROR,
    "L2.DUPLICATE_LIVE_RESERVATION": FailureClass.TRANSACTION_CONCURRENCY_ERROR,
    "L2.UNDECLARED_APPOINTMENT_STATUS": FailureClass.STATE_MACHINE_BYPASS,
    "L2.UNDECLARED_STATUS_TRANSITION": FailureClass.STATE_MACHINE_BYPASS,
    "L2.AUDIT_CHAIN_BROKEN": FailureClass.TRANSACTION_CONCURRENCY_ERROR,
    "L3.SCREENING_AFTER_MODEL": FailureClass.STATE_MACHINE_BYPASS,
    "L3.MULTIPLE_MUTATIONS_IN_TURN": FailureClass.STATE_MACHINE_BYPASS,
    "L3.TOOL_NOT_ALLOWED_IN_STATE": FailureClass.STATE_MACHINE_BYPASS,
    "L3.TOOL_EXECUTED_WITHOUT_OFFER": FailureClass.STATE_MACHINE_BYPASS,
    "L3.SCENARIO_FORBIDDEN_TOOL_OFFERED": FailureClass.STATE_MACHINE_BYPASS,
    "L3.SCENARIO_FORBIDDEN_TOOL_EXECUTED": FailureClass.STATE_MACHINE_BYPASS,
    "L3.SCENARIO_REQUIRED_TOOL_WITHHELD": FailureClass.IDENTITY_AUTHORIZATION_FAILURE,
    # Route conformance is classified from the verdict direction rather than from
    # this table -- a missed trigger and an unwanted trigger share one oracle id
    # and are opposite classes. The entry below is the fallback for a mismatch
    # that is not a clear/non-clear disagreement (two different non-clear
    # verdicts), which is a policy-data precision defect.
    "L3.ROUTE_MISMATCH": FailureClass.CLINICAL_POLICY_GAP,
    "L3.FIRED_RULES_MISMATCH": FailureClass.CLINICAL_POLICY_GAP,
    "L3.SCENARIO_TEXT_VIOLATION": FailureClass.RESPONSE_GROUNDING_ERROR,
    "L3.SCENARIO_TEXT_MISSING": FailureClass.RESPONSE_GROUNDING_ERROR,
    "L3.SCENARIO_DB_MISMATCH": FailureClass.STATE_MACHINE_BYPASS,
    "L0.UNHANDLED_EXCEPTION": FailureClass.TOOL_CONTRACT_ERROR,
    "L0.MISSING_STATE": FailureClass.STATE_MACHINE_BYPASS,
    "L0.UNKNOWN_STATE": FailureClass.STATE_MACHINE_BYPASS,
    "L0.UNRESOLVED_TOOL_OUTCOME": FailureClass.TOOL_CONTRACT_ERROR,
    "L0.TURN_BOUND": FailureClass.MODEL_FLAKINESS,
}

_QUALITY_PREFIXES = ("Q1_", "Q2_", "Q3_", "Q4_", "Q5_", "Q6_", "Q7_", "Q8_")


def _is_quality_finding(oracle_id: str, reason_code: str) -> bool:
    """Whether an oracle id belongs to the conversation-quality layer.

    The quality layer reports ids like ``Q3_NO_FALSE_REASSURANCE``, but a caller
    that joined the layer name on -- ``L4.Q3_...`` -- should not fall through to
    the unclassified branch and be reported as an oracle error. Both forms are
    accepted, and the reason code ``QUALITY`` is accepted as a third signal,
    because a finding whose severity is capped at S3 must never be misrouted to a
    structural layer.
    """
    if reason_code == "QUALITY":
        return True
    tail = oracle_id.rsplit(".", 1)[-1]
    return tail.startswith(_QUALITY_PREFIXES)


def classify_finding(
    *,
    oracle_id: str,
    scenario_id: str,
    evidence: str,
    reason_code: str = "",
    expected_verdict: str | None = None,
    observed_verdict: str | None = None,
    implicated_rule_id: str | None = None,
) -> FailureClassification:
    """Classify one oracle finding into a failure class.

    The routing table does most of the work, but a finding about *routing* needs
    one extra distinction that the oracle id alone cannot carry: a missed trigger
    is a false negative and an unwanted trigger is a false positive, and they are
    the same oracle id from different directions. The direction is read from the
    expected and observed verdicts, because those are the two facts that decide
    it.

    A finding whose class cannot be determined is classified as an oracle error
    and makes the run ineligible. That is deliberate: an unclassifiable failure is
    an opportunity to guess, and a guess here becomes a patch at a random layer.
    """
    evidence_items = [f"oracle_id={oracle_id}"]
    if evidence:
        evidence_items.append(f"evidence={evidence[:300]}")
    if reason_code:
        evidence_items.append(f"reason_code={reason_code}")

    if _is_quality_finding(oracle_id, reason_code):
        return FailureClassification(
            failure_class=FailureClass.ACCESSIBILITY_QUALITY_FAILURE,
            scenario_id=scenario_id,
            evidence=tuple(evidence_items),
            counter_hypotheses=(
                "Considered RESPONSE_GROUNDING_ERROR and rejected it: a quality "
                "finding is about readability or hygiene in text that was "
                "otherwise authorised, not about a claim with no grounded fact.",
            ),
            required_layer=required_layer(FailureClass.ACCESSIBILITY_QUALITY_FAILURE),
            layer_reason=layer_reason(FailureClass.ACCESSIBILITY_QUALITY_FAILURE),
            hypothesis="The reply was difficult to read or leaked internal framing.",
        )

    if oracle_id.startswith("ORACLE") or reason_code == "ORACLE_ERROR":
        return FailureClassification(
            failure_class=FailureClass.ORACLE_ERROR,
            scenario_id=scenario_id,
            evidence=tuple(evidence_items),
            counter_hypotheses=(
                "Considered every system class and rejected all of them: the "
                "oracle raised, so no observation about the system was produced.",
            ),
            required_layer=required_layer(FailureClass.ORACLE_ERROR),
            layer_reason=layer_reason(FailureClass.ORACLE_ERROR),
            hypothesis="An oracle failed to evaluate and must be fixed before the run counts.",
        )

    # The routing direction: same oracle family, opposite class. This is decided
    # from the verdicts rather than from the oracle id because the routing oracle
    # reports "the verdict did not match the expectation", and which way it
    # mismatched is the whole question.
    if expected_verdict is not None and observed_verdict is not None:
        expected_clear = expected_verdict.startswith("CLEAR")
        observed_clear = observed_verdict.startswith("CLEAR")
        if expected_clear and not observed_clear:
            return FailureClassification(
                failure_class=FailureClass.SAFETY_SENTINEL_FALSE_POSITIVE,
                scenario_id=scenario_id,
                evidence=tuple(
                    evidence_items
                    + [
                        f"expected_verdict={expected_verdict}",
                        f"observed_verdict={observed_verdict}",
                    ]
                ),
                counter_hypotheses=(
                    "Considered SAFETY_SENTINEL_FALSE_NEGATIVE and rejected it: the "
                    "sentinel routed *away* from clear, so the defect is precision.",
                    "Considered CLINICAL_POLICY_GAP and rejected it: the rule exists "
                    "and fired; the gap is in the guard that should have suppressed it.",
                ),
                required_layer=required_layer(FailureClass.SAFETY_SENTINEL_FALSE_POSITIVE),
                layer_reason=layer_reason(FailureClass.SAFETY_SENTINEL_FALSE_POSITIVE),
                implicated_rule_id=implicated_rule_id,
                hypothesis=(
                    f"A pattern in {'the implicated rule' if implicated_rule_id else 'the sentinel'} "
                    f"fired where the expected route was {expected_verdict}."
                ),
            )
        if not expected_clear and observed_clear:
            return FailureClassification(
                failure_class=FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE,
                scenario_id=scenario_id,
                evidence=tuple(
                    evidence_items
                    + [
                        f"expected_verdict={expected_verdict}",
                        f"observed_verdict={observed_verdict}",
                    ]
                ),
                counter_hypotheses=(
                    "Considered SAFETY_SENTINEL_FALSE_POSITIVE and rejected it: the "
                    "sentinel did not fire where it was required to.",
                    "Considered RESPONSE_GROUNDING_ERROR and rejected it: nothing was "
                    "said that a grounded fact failed to authorise; the route itself "
                    "was wrong, so the failure is upstream of the reply.",
                ),
                required_layer=required_layer(FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE),
                layer_reason=layer_reason(FailureClass.SAFETY_SENTINEL_FALSE_NEGATIVE),
                implicated_rule_id=implicated_rule_id,
                hypothesis=(
                    f"The route required {expected_verdict} but the sentinel "
                    f"returned {observed_verdict}; the pattern set misses this surface form."
                ),
            )

    failure_class = _ORACLE_ID_TO_CLASS.get(oracle_id)
    if failure_class is None:
        return FailureClassification(
            failure_class=FailureClass.ORACLE_ERROR,
            scenario_id=scenario_id,
            evidence=tuple(evidence_items),
            counter_hypotheses=(
                f"No routing entry exists for oracle id {oracle_id!r}. This is an "
                "oracle error rather than a guess, because a failure whose cause "
                "cannot be named cannot be routed to a layer.",
            ),
            required_layer=required_layer(FailureClass.ORACLE_ERROR),
            layer_reason=layer_reason(FailureClass.ORACLE_ERROR),
            hypothesis=f"Unmapped oracle id {oracle_id!r}; taxonomy needs widening.",
        )

    return FailureClassification(
        failure_class=failure_class,
        scenario_id=scenario_id,
        evidence=tuple(evidence_items),
        counter_hypotheses=(),
        required_layer=required_layer(failure_class),
        layer_reason=layer_reason(failure_class),
        implicated_rule_id=implicated_rule_id,
        hypothesis=f"Routed from oracle id {oracle_id} to {failure_class.value}.",
    )


#: Instance used by modules that just want the mapping without importing the
#: routing tables directly.
#: Public alias. ``classify`` reads better at call sites than ``classify_finding``
#: and is what the package ``__init__`` exports.
classify = classify_finding


__all__ = [
    "FailureClass",
    "FailureClassification",
    "LAYER_RANK",
    "PatchLayer",
    "Taxonomy",
    "classify",
    "classify_finding",
    "layer_reason",
    "required_layer",
]
