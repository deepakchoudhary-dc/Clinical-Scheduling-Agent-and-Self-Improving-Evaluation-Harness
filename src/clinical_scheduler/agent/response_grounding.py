"""Response grounding: the layer that makes invented facts unable to reach a patient.

plan.md section 5.1 invariant 13 states that every factual claim in an agent
response must be traceable to verified tool output or to approved static
content. This module is the enforcement point.

The design is deliberately **subtractive**. The model writes prose; this module
decides which parts of that prose are allowed to survive. It never adds facts
of its own. Three mechanisms, in order of how blunt they are:

1. **Eligibility.** Facts exist only because a tool returned them this turn
   (``facts_from_tool_result``) or because they follow from server-side session
   state that the runtime owns (``session_facts``). A claim key that is not in
   that set cannot be rendered at all.

2. **Sentence triage.** The model's prose is split into sentences. A sentence
   that asserts a state-bearing claim -- availability, a held slot, a booked
   appointment, an identity match, a requirement, an escalation, a promise
   about who was contacted -- is admitted only if a grounded fact authorises it.
   Everything else is dropped and recorded in ``unsourced_claims``.

3. **Prohibited claims.** Independently of grounding, the active policy rule may
   forbid whole classes of claim (``ProhibitedClaim``). These are matched
   structurally against the model's text and, unlike triage, a hit *replaces*
   the text rather than merely dropping a sentence, because a prohibited claim
   is not a gap to be closed -- it is a statement the service must never make.

Why subtraction rather than generation: if this module generated the reply, a
bug in it would silently invent facts, and the failure would be invisible. By
subtracting, the worst case of a bug is a *less informative* reply. Under-
informing someone about an appointment is recoverable. Misinforming them is not.

The module also never trusts the model to police itself. ``compose`` is the only
public entry point that produces patient-visible text, and it always returns the
unsourced-claim list alongside, so the oracle stack can assert on both.
"""

from __future__ import annotations

# NOTE (2026-10-08 static audit): three module constants -- ``_CLAIM_TRIGGERS``,
# ``_NEGATIVE_MARKERS`` and ``_HEDGE_MARKERS`` -- were removed. They were defined
# and never referenced, and their comments described negation/hedge handling that
# no code implemented: a claim in a comment is still a claim. The live logic reads
# ``_AUTHORISING_KEYS`` and ``_claim_triggers()`` instead.

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from clinical_scheduler.domain.states import (
    GroundedFact,
    SafetyVerdict,
    SessionState,
)
from clinical_scheduler.policy.engine import PolicyDecision
from clinical_scheduler.policy.schema import ProhibitedClaim
from clinical_scheduler.tools.broker import SessionContext


# ---------------------------------------------------------------------------
# Claim authority
# ---------------------------------------------------------------------------

#: Claim keys and the words that signal a sentence is asserting *that* claim.
#: Matching is anchored on the claim vocabulary rather than on the model's
#: phrasing, so a paraphrase of the same assertion is still caught.
_AUTHORISING_KEYS: dict[str, tuple[str, ...]] = {
    "slot_availability": (
        "available", "availability", "opening", "openings", "free at", "has space",
        "slots", "options for", "times on",
    ),
    "slot_hold": (
        "held", "hold on", "on hold", "reserved for you", "reserving", "holding",
        "temporary reservation", "expires",
    ),
    "appointment_booked": (
        "booked", "is scheduled", "you are scheduled", "appointment is confirmed",
        "confirmed for", "confirmation number", "appointment reference",
    ),
    "appointment_cancelled": ("cancelled", "canceled", "has been cancelled"),
    "appointment_details": (
        "your appointment", "the appointment on", "you are seeing", "you will see",
        "at the clinic", "location is", "practitioner is",
    ),
    "requirements": (
        "requirement", "referral", "needs a referral", "preparation", "you need to",
        "must bring", "fasting", "is outstanding", "not yet satisfied",
    ),
    "identity_match": (
        "i found your record", "your record is confirmed", "identity is verified",
        "i have verified", "your identity", "matched your record", "confirmed who you are",
    ),
    "candidate_lookup": (
        "could not find", "cannot find", "no record", "more than one record",
        "several records", "matching records",
    ),
    "service_mapping": ("service", "appointment type", "which clinic", "department"),
    "practitioner_listing": ("practitioner", "clinician who", "doctor who", "specialist who"),
    "escalation_recorded": (
        "flagged", "alerted a human", "a person will follow up", "will follow up with you",
        "noted for the clinic team", "passed this on", "escalated",
    ),
    "waitlist": ("waitlist", "wait list", "waiting list"),
    "policy_routing": (
        "urgent", "emergency", "immediately", "call 911", "emergency department",
        "crisis line", "988", "right away",
    ),
}

#: Markers wrapping a safety replacement written by this module. Two forms: an
#: ASCII form used between ``_scrub_prohibited`` and ``_sentences``, and a
#: control-character form that survives sentence splitting intact.
_PROTECTED_OPEN = "\ue000"
_PROTECTED_CLOSE = "\ue001"
_SENTINEL_OPEN = "\x01P\x01"
_SENTINEL_CLOSE = "\x01Q\x01"


def _sentences(text: str) -> list[str]:
    """Split prose into sentences without touching abbreviations we care about.

    Legal ``911`` and ``988`` must survive, so a naive ``split(".")`` is not
    usable: "call 911." would lose its number and "e.g." would fragment. The
    splitter therefore breaks on terminal punctuation, newlines and semicolons,
    but only when the following character is whitespace or end-of-string, and it
    protects the two numeric literals the policy bundle may instruct the model to
    repeat.
    """
    if not text.strip():
        return []
    protected = (
        text.replace("911", "\x00911\x00")
        .replace("988", "\x00988\x00")
        # The protection markers must not be broken apart by the splitter, so
        # their ASCII markers are neutralised before splitting and the sentence
        # carries the marker through intact.
        .replace(_PROTECTED_OPEN, _SENTINEL_OPEN)
        .replace(_PROTECTED_CLOSE, _SENTINEL_CLOSE)
    )
    parts = re.split(r"(?<=[.!?])[\s\n]+|[\n;]+", protected)
    out: list[str] = []
    for part in parts:
        cleaned = part.replace("\x00", "").strip()
        if cleaned:
            out.append(cleaned)
    return out


def _is_protected(sentence: str) -> bool:
    """True when the sentence is a safety replacement this module wrote.

    Protected sentences bypass triage. They are safe by construction -- they are
    authored in this file, not by the model -- and dropping one would remove a
    disclaimer rather than a fabricated claim.
    """
    return _SENTINEL_OPEN in sentence or _SENTINEL_CLOSE in sentence


def _unprotect(sentence: str) -> str:
    return sentence.replace(_SENTINEL_OPEN, "").replace(_SENTINEL_CLOSE, "").strip()


@dataclass(frozen=True)
class _ProhibitionHit:
    claim: ProhibitedClaim
    matched_span: str


class ResponseComposer:
    """Turns model prose into patient-visible text that cannot assert facts
    nothing produced.

    One instance per turn (it carries no mutable state, but keeping it
    per-session keeps the construction site obvious).
    """

    #: Structure wording, per prohibited claim. Used to replace text, never to
    #: build a claim.
    _PROHIBITION_REPLACEMENTS: dict[ProhibitedClaim, str] = {
        ProhibitedClaim.STOP_MEDICATION: (
            "I am not able to advise you about stopping any medicine. "
            "That has to come from your prescriber."
        ),
        ProhibitedClaim.CHANGE_DOSE: (
            "I am not able to advise you about changing a dose. "
            "Please speak to your prescriber or pharmacist."
        ),
        ProhibitedClaim.START_MEDICATION: (
            "I am not able to advise you about starting any medicine. "
            "That has to come from your prescriber."
        ),
        ProhibitedClaim.SKIP_MEDICATION: (
            "I am not able to advise you about skipping any medicine. "
            "Please speak to your prescriber."
        ),
        ProhibitedClaim.BRIDGE_MEDICATION: (
            "I am not able to advise you about bridging or adjusting any medicine. "
            "That has to come from your prescriber."
        ),
        ProhibitedClaim.CONFIRM_DIAGNOSIS: (
            "I am not able to tell you what condition this is or is not. "
            "Only a clinician who examines you can do that."
        ),
        ProhibitedClaim.RULE_OUT_CONDITION: (
            "I am not able to rule anything in or out. "
            "Please have that assessed by a clinician."
        ),
        ProhibitedClaim.EMERGENCY_SERVICES_CONTACTED: (
            "I have not contacted anyone on your behalf, and I cannot arrange "
            "emergency care."
        ),
        ProhibitedClaim.GUARANTEE_COVERAGE: (
            "I am not able to confirm what insurance will or will not cover. "
            "Please check with your insurer."
        ),
        ProhibitedClaim.GUARANTEE_COST: (
            "I am not able to confirm what this will cost. "
            "Please check with the clinic's billing team."
        ),
    }

    #: Deterministic detectors for each prohibited claim class. These are
    #: structural, not semantic: they look for the assertion, not for a topic.
    #: Deliberately over-broad -- a false positive costs one replaced sentence,
    #: a false negative costs a patient acting on medical advice from a
    #: scheduling bot.
    _PROHIBITION_PATTERNS: dict[ProhibitedClaim, tuple[str, ...]] = {
        ProhibitedClaim.STOP_MEDICATION: (
            r"\b(stop|stopping|discontinue|hold off|come off|quit)\b[^.]{0,40}\b(medication|medicine|meds|pills?|tablet|drug|dose|blood thinner|anticoagulant)\b",
            r"\byou (should|can|may) (stop|skip|hold)\b",
            r"\b(medication|medicine|meds|pills?)\b[^.]{0,30}\b(before|prior to)\b[^.]{0,25}\b(procedure|surgery|appointment)\b",
        ),
        ProhibitedClaim.CHANGE_DOSE: (
            r"\b(increase|decrease|reduce|lower|raise|double|halve|adjust|change)\b[^.]{0,30}\b(dose|dosage|doseage|mg|milligram)\b",
            r"\btake (an? )?(extra|another|higher|lower) dose\b",
            r"\b(dose|dosage)\b[^.]{0,25}\b(should be|needs to be|increase|decrease)\b",
        ),
        ProhibitedClaim.START_MEDICATION: (
            r"\b(start|begin|take|try)\b[^.]{0,35}\b(medication|medicine|meds|pills?|antibiotic|painkiller|ibuprofen|paracetamol|acetaminophen|aspirin)\b",
            r"\byou (should|can|may|need to) take\b",
        ),
        ProhibitedClaim.SKIP_MEDICATION: (
            r"\b(skip|miss|leave out|omit)\b[^.]{0,30}\b(dose|medication|medicine|meds|pills?|tablet)\b",
        ),
        ProhibitedClaim.BRIDGE_MEDICATION: (
            r"\b(bridg\w*|self-?bridg\w*)\b",
        ),
        ProhibitedClaim.CONFIRM_DIAGNOSIS: (
            r"\byou (definitely |certainly |probably )?(have|had|are experiencing|are suffering from)\b[^.]{0,40}\b(condition|syndrome|disease|infection|disorder|attack)\b",
            r"\b(this|it|that) (is|sounds like|looks like|appears to be)\b[^.]{0,30}\b(condition|syndrome|disease|infection|migraine|flu|infection)\b",
            r"\byour diagnosis\b",
            r"\bthis confirms\b",
        ),
        ProhibitedClaim.RULE_OUT_CONDITION: (
            r"\b(rules? out|ruled out|nothing to worry about|no need to worry|not serious|nothing serious)\b",
            r"\b(it is not|this is not|you do not have|you don't have)\b[^.]{0,35}\b(a |an )?(heart attack|stroke|infection|fracture|blood clot|emergency)\b",
            r"\b(is not|isn't) (an? )?emergency\b",
            r"\bno cause for concern\b",
        ),
        ProhibitedClaim.EMERGENCY_SERVICES_CONTACTED: (
            r"\b(i|we|the system|the service)\b[^.]{0,60}\b(contacted|called|alerted|dispatched|notified|arranged|sent)\b[^.]{0,40}\b(911|emergency|ambulance|paramedic|police|services|help)\b",
            r"\b(an? )?(ambulance|paramedic|emergency team|help) (is|are) (on the way|coming|coming to you|en route)\b",
            r"\b(help is coming|someone is on the way|on the way now)\b",
            r"\b(i have|i've|we have|we've) (already )?(informed|told|alerted)\b[^.]{0,30}\b(hospital|clinic team|emergency)\b",
        ),
        ProhibitedClaim.GUARANTEE_COVERAGE: (
            r"\b(insurance|insurer|coverage|covered|policy)\b[^.]{0,40}\b(will (definitely |certainly )?cover|is covered|fully covered|guaranteed)\b",
            r"\byou (will|won't|will not) (be charged|have to pay)\b",
            r"\b(no|zero) (cost|charge|fee)\b",
        ),
        ProhibitedClaim.GUARANTEE_COST: (
            r"\b(the |this |it will )?(cost|price|fee) (will be|is|is going to be)\b[^.]{0,20}(\$|£|€|\d)",
            r"\bcosts? exactly\b",
            r"\byour total will be\b",
        ),
    }

    # ------------------------------------------------------------------
    # Fact derivation
    # ------------------------------------------------------------------

    def session_facts(
        self, state: SessionState, session: SessionContext
    ) -> list[GroundedFact]:
        """Facts that follow from server-side session state.

        These are the only facts not derived from a tool result, and they are
        narrow on purpose: whether identity is verified, whether a safety route
        is active, and the current conversational stage. All three are owned by
        the runtime and cannot be influenced by the model.

        Note what is *absent*: this method never produces a fact about a slot,
        an appointment, a requirement or a patient record. Those can only come
        from a tool, which is what makes ``facts_from_tool_result`` load-bearing.
        """
        facts: list[GroundedFact] = []
        if session.verified_identity or state in {
            SessionState.IDENTITY_VERIFIED,
            SessionState.AUTHORIZATION_VERIFIED,
            SessionState.REQUEST_SCOPED,
            SessionState.REQUIREMENTS_PENDING,
            SessionState.REQUIREMENTS_SATISFIED,
            SessionState.OPTIONS_PRESENTED,
            SessionState.USER_SELECTED,
            SessionState.SLOT_HELD,
            SessionState.SUMMARY_CONFIRMED,
            SessionState.COMMITTING,
            SessionState.BOOKED,
        }:
            facts.append(
                GroundedFact(
                    claim_key="identity_match",
                    value="verified",
                    source="approved_static",
                    evidence="session state is past IDENTITY_VERIFIED (server-owned)",
                )
            )
        facts.append(
            GroundedFact(
                claim_key="conversation_stage",
                value=state.value,
                source="approved_static",
                evidence="session state machine (server-owned)",
            )
        )
        if session.safety_route_active:
            facts.append(
                GroundedFact(
                    claim_key="policy_routing",
                    value="active",
                    source="approved_static",
                    evidence="safety route recorded on the server session",
                )
            )
        return facts

    def facts_from_tool_result(
        self, tool_name: str, out: dict[str, Any]
    ) -> list[GroundedFact]:
        """Derive grounded facts from one *successful* tool result.

        Only ``SUCCESS`` envelopes produce facts. A ``DENIED``, ``ERROR`` or
        ``UNKNOWN`` envelope produces none, deliberately: the honest response to
        "we do not know what happened" is not to convert it into a fact.

        The evidence string always names the tool and cites the runtime
        correlation so a reviewer can trace the claim back to the audit row.
        """
        envelope = out.get("envelope") or {}
        if envelope.get("status") != "SUCCESS":
            return []

        correlation = str(envelope.get("correlation_id") or "")
        produced: list[GroundedFact] = []

        def fact(key: str, value: Any, detail: str) -> None:
            produced.append(
                GroundedFact(
                    claim_key=key,
                    value=value,
                    source="verified_tool_result",
                    evidence=f"{tool_name}: {detail}" + (f" [{correlation}]" if correlation else ""),
                )
            )

        if tool_name == "search_slots":
            slots = out.get("results") or out.get("slots") or []
            fact(
                "slot_availability",
                {"count": len(slots), "window": _window_label(out.get("window"))},
                f"{len(slots)} advisory slot(s) returned; not held",
            )
            if not slots:
                produced[-1] = produced[-1].model_copy(
                    update={"value": {"count": 0, "window": _window_label(out.get("window"))}}
                )

        elif tool_name == "hold_slot":
            lease = out.get("lease") or {}
            fact(
                "slot_hold",
                {
                    "slot_id": lease.get("slot_id"),
                    "expires_at": lease.get("expires_at"),
                    "lease_id": lease.get("lease_id"),
                },
                "lease created; this is a hold, not an appointment",
            )

        elif tool_name == "commit_appointment":
            appt = out.get("appointment") or {}
            fact(
                "appointment_booked",
                {
                    "appointment_id": appt.get("appointment_id"),
                    "slot_id": appt.get("slot_id"),
                    "starts_at": appt.get("starts_at"),
                    "location_id": appt.get("location_id"),
                    "practitioner_id": appt.get("practitioner_id"),
                    "service_id": appt.get("service_id"),
                },
                "appointment committed and persisted",
            )
            fact(
                "appointment_details",
                {
                    "appointment_id": appt.get("appointment_id"),
                    "starts_at": appt.get("starts_at"),
                },
                "details of the committed appointment",
            )

        elif tool_name == "reconcile_commit":
            outcome = out.get("outcome")
            if out.get("appointment"):
                fact(
                    "appointment_booked",
                    out["appointment"],
                    "reconciliation found a persisted appointment",
                )
            else:
                fact(
                    "reconciliation",
                    outcome,
                    "reconciliation resolved the commit outcome",
                )

        elif tool_name == "atomic_reschedule":
            appt = out.get("appointment") or {}
            fact(
                "appointment_booked",
                {
                    "appointment_id": appt.get("appointment_id"),
                    "slot_id": appt.get("slot_id"),
                    "starts_at": appt.get("starts_at"),
                    "previous_appointment_id": appt.get("supersedes") or out.get("supersedes"),
                },
                "replacement committed; original retired only on success",
            )

        elif tool_name == "cancel_appointment":
            fact(
                "appointment_cancelled",
                {"appointment_id": out.get("appointment_id"), "status": out.get("status")},
                "cancellation committed",
            )

        elif tool_name == "check_requirements":
            reqs = out.get("requirements") or []
            outstanding = [r for r in reqs if r.get("status") != "SATISFIED"]
            fact(
                "requirements",
                {"total": len(reqs), "outstanding": len(outstanding)},
                f"{len(reqs)} requirement(s), {len(outstanding)} outstanding",
            )

        elif tool_name == "lookup_patient_candidates":
            fact(
                "candidate_lookup",
                {
                    "count": out.get("candidate_count"),
                    "ambiguous": out.get("ambiguous"),
                },
                "candidate lookup returns a count only, never candidate details",
            )

        elif tool_name == "verify_patient_discriminator":
            fact(
                "identity_match",
                {"verified": bool(out.get("verified"))},
                "discriminator verification result",
            )

        elif tool_name == "verify_proxy_scope":
            fact(
                "authorization",
                {
                    "covered": bool(out.get("covered")),
                    "requires_staff_review": bool(out.get("requires_staff_review")),
                },
                "proxy scope evaluation",
            )

        elif tool_name == "resolve_service":
            fact(
                "service_mapping",
                {
                    "service_id": out.get("service_id"),
                    "ambiguous": bool(out.get("ambiguous")),
                    "candidate_count": len(out.get("candidates") or []),
                },
                "service resolution result",
            )

        elif tool_name == "list_practitioner_roles":
            roles = out.get("roles") or []
            fact(
                "practitioner_listing",
                {"count": len(roles)},
                f"{len(roles)} practitioner role(s) for the service",
            )

        elif tool_name == "join_waitlist":
            fact(
                "waitlist",
                {"entry_id": out.get("entry_id"), "created": bool(out.get("entry_id"))},
                "waitlist entry created",
            )

        elif tool_name == "route_urgent_concern":
            fact(
                "escalation_recorded",
                {
                    "escalation_id": out.get("escalation_id"),
                    "human_contacted": False,
                    # The approved patient-facing wording the escalation produced.
                    # Carried so the composer can deliver it rather than falling back
                    # to a state clarification. See `_fallback_body`.
                    "patient_safe_text": out.get("patient_safe_text"),
                },
                "escalation recorded in-system; no external service contacted",
            )

        elif tool_name == "create_human_handoff":
            fact(
                "escalation_recorded",
                {"handoff_id": out.get("handoff_id"), "human_contacted": False},
                "handoff recorded for a human; no external contact made",
            )

        return produced

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    #: Claims that are always prohibited while a safety route is active,
    #: irrespective of what the bundle declares in ``prohibited_claims``.
    #:
    #: This set exists because of a defect found in review. A ``urgent_handoff``
    #: rule that declares no ``prohibited_claims`` (as SAFE-001 did) would
    #: otherwise let the model append "I have contacted 911 and an ambulance is
    #: on the way" to the emergency template -- producing a single message that
    #: both denies and asserts that help was summoned. These two claim classes
    #: are not policy-tunable: they are the failure modes that make this system
    #: dangerous rather than merely unhelpful, so the deny is unconditional.
    _ALWAYS_PROHIBITED_ON_SAFETY_ROUTE: frozenset[ProhibitedClaim] = frozenset(
        {
            ProhibitedClaim.EMERGENCY_SERVICES_CONTACTED,
            ProhibitedClaim.RULE_OUT_CONDITION,
            ProhibitedClaim.CONFIRM_DIAGNOSIS,
        }
    )

    def _effective_prohibitions(self, decision: PolicyDecision) -> frozenset[ProhibitedClaim]:
        """Union of the bundle's declared prohibitions and the unconditional set.

        The unconditional set only applies when a safety route is active. On a
        clear turn there is no emergency context, so applying it would replace
        innocuous text ("I will call you back") for no benefit.
        """
        if decision.blocks_mutation:
            return frozenset(decision.prohibited_claims) | self._ALWAYS_PROHIBITED_ON_SAFETY_ROUTE
        return frozenset(decision.prohibited_claims)

    def compose(
        self,
        *,
        model_text: str,
        facts: Sequence[GroundedFact],
        decision: PolicyDecision,
        state: SessionState,
    ) -> tuple[str, tuple[str, ...]]:
        """Produce the patient-visible reply and the list of dropped claims.

        Ordering is significant. Prohibited-claim scrubbing runs *first*, on the
        whole text, because a prohibited claim must not survive even inside a
        sentence that grounding happens to drop for another reason -- if the text
        is ever logged or echoed, it should already be clean. Triage then runs on
        the scrubbed text.
        """
        unsourced: list[str] = []

        working = model_text.strip()
        working, hits = self._scrub_prohibited(working, self._effective_prohibitions(decision))
        for hit in hits:
            unsourced.append(f"prohibited_claim:{hit.claim.value}")

        allowed_keys = {f.claim_key for f in facts}
        kept: list[str] = []

        for sentence in _sentences(working):
            if _is_protected(sentence):
                # Authored here, safe by construction: never triage it away.
                kept.append(_unprotect(sentence))
                continue
            triggers = self._claim_triggers(sentence)
            if not triggers:
                kept.append(sentence)
                continue
            if allowed_keys & set(triggers):
                kept.append(sentence)
                continue
            # A sentence claiming state that no fact authorises is dropped.
            # Recorded with the trigger so a reviewer can see *what* was claimed.
            unsourced.append(
                f"unsourced_claim:{','.join(sorted(triggers))}::{_truncate(sentence, 160)}"
            )

        body = " ".join(kept).strip()

        # When a safety route carries an authoritative template, the template is
        # the whole reply.
        #
        # The earlier design prefixed the template to the surviving prose. Review
        # found that unsafe: the prose could restate the very claim the template
        # denies, and concatenation produced a self-contradicting message. The
        # template is written to be complete and safe on its own, so the model's
        # prose is discarded entirely rather than reviewed for compatibility. Any
        # surviving prose is still reported through ``unsourced_claims`` so the
        # discard is visible in the trace rather than silent.
        if decision.blocks_mutation and decision.message_template is not None:
            for sentence in _sentences(body):
                unsourced.append(f"suppressed_under_safety_template::{_truncate(sentence, 160)}")
            return decision.message_template.text.strip(), tuple(unsourced)

        if not body:
            body = self._fallback_body(decision, state, facts)

        return body, tuple(unsourced)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _claim_triggers(sentence: str) -> set[str]:
        lowered = sentence.casefold()
        hits: set[str] = set()
        for key, words in _AUTHORISING_KEYS.items():
            if any(w in lowered for w in words):
                hits.add(key)
        return hits

    def _scrub_prohibited(
        self, text: str, prohibited: frozenset[ProhibitedClaim]
    ) -> tuple[str, list[_ProhibitionHit]]:
        """Replace any prohibited assertion with structurally safe wording.

        A hit is detected on the *sentence* so the replacement unit is a whole
        clause, not a fragment. Cutting out a matched span mid-sentence risks
        producing a mangled sentence that reads as a different claim -- which
        would be worse than the original failure.

        Replacement sentences are wrapped in ``_PROTECTED`` markers. The markers
        are stripped in ``compose`` *after* triage, and they exist because
        replacement text is written to be safe by construction: it says what the
        service cannot do, which is exactly the kind of sentence triage would
        otherwise drop for want of a grounded fact. Dropping it would silently
        delete the disclaimer -- the worst possible outcome, since
        "I have not contacted anyone on your behalf" is the sentence that must
        survive a fabricated-escalation attempt.

        Each prohibited class is emitted at most once per turn. The model may
        restate the same forbidden assertion in several sentences; repeating the
        correction three times reads as a malfunction, and one clear statement
        carries the same information.
        """
        if not prohibited or not text.strip():
            return text, []

        sentences = _sentences(text)
        out: list[str] = []
        hits: list[_ProhibitionHit] = []
        already_replaced: set[ProhibitedClaim] = set()

        for sentence in sentences:
            hit_claim = self._prohibition_in(sentence, prohibited)
            if hit_claim is None:
                out.append(sentence)
                continue

            hits.append(_ProhibitionHit(claim=hit_claim, matched_span=_truncate(sentence, 160)))

            if hit_claim in already_replaced:
                # Suppress the duplicate entirely rather than emitting a stutter.
                continue
            already_replaced.add(hit_claim)

            replacement = self._PROHIBITION_REPLACEMENTS.get(
                hit_claim, self._generic_replacement()
            )
            out.append(f"{_PROTECTED_OPEN}{replacement}{_PROTECTED_CLOSE}")

        if not out:
            return text, []
        # Reassemble preserving the original sentence order and nothing else.
        return " ".join(out), hits

    def _prohibition_in(
        self, sentence: str, prohibited: frozenset[ProhibitedClaim]
    ) -> ProhibitedClaim | None:
        lowered = sentence.casefold()
        for claim in sorted(prohibited, key=lambda c: c.value):
            for pattern in self._PROHIBITION_PATTERNS.get(claim, ()):
                if re.search(pattern, lowered):
                    return claim
        return None

    @staticmethod
    def _generic_replacement() -> str:
        return (
            "I am not able to advise on that myself. "
            "It needs a member of the clinical team rather than the scheduling service."
        )

    def _fallback_body(
        self,
        decision: PolicyDecision,
        state: SessionState,
        facts: Sequence[GroundedFact] = (),
    ) -> str:
        """Used when triage removed everything, or the model produced nothing.

        The fallback is derived from the decision and the state. It states the
        current stage and asks one question. It never asserts anything that
        would need grounding -- which is precisely why it is a safe floor.
        """
        # An escalation or handoff that was *actually recorded* outranks every state
        # clarification. Found in a live session on 2026-10-09: `route_urgent_concern`
        # and `create_human_handoff` both returned SUCCESS, both were grounded as
        # `escalation_recorded`, and the caller was still asked to confirm their date
        # of birth. The correct sentence existed and was thrown away. A recorded
        # escalation is the strongest thing this system can know, so it speaks first.
        for grounded in facts:
            if grounded.claim_key != "escalation_recorded":
                continue
            approved = (
                grounded.value.get("patient_safe_text")
                if isinstance(grounded.value, dict)
                else None
            )
            if approved:
                return str(approved)
            return (
                "I have flagged this to the clinic team and a person will follow up "
                "with you. I have not contacted anyone on your behalf. If you think "
                "this may be an emergency, please call your local emergency number or "
                "go to the nearest emergency department now."
            )

        if decision.message_template is not None:
            return decision.message_template.text

        if decision.verdict is SafetyVerdict.TRIGGERED:
            return (
                "Please contact emergency services or go to your nearest emergency "
                "department now. I have not contacted anyone on your behalf, and I "
                "cannot arrange emergency care. A member of the clinic team will "
                "follow up with you."
            )
        if decision.verdict is SafetyVerdict.UNCERTAIN:
            return (
                "I want a member of the clinical team to look at that before we go "
                "any further with scheduling. I have flagged it for review and will "
                "not book anything in the meantime."
            )

        next_question = {
            SessionState.START: "Could you tell me what you would like to do -- book, move or cancel an appointment?",
            # The format is stated because the tool demands YYYY-MM-DD and the person
            # cannot know that. Making someone guess a wire format while they are in
            # pain is a design failure, not a user error. Found in a live session on
            # 2026-10-09, where a patient typed "1/1/1999" and was never told.
            SessionState.SAFETY_SCREENED: (
                "Could you confirm the full name and date of birth on the record, so I "
                "can find the right one? Please give the date as YYYY-MM-DD, for "
                "example 1999-01-31."
            ),
            SessionState.IDENTITY_PENDING: (
                "Could you confirm the date of birth on the record? Please give it as "
                "YYYY-MM-DD, for example 1999-01-31."
            ),
            SessionState.IDENTITY_AMBIGUOUS: "There is more than one record matching that name and date of birth. Could you give me one discriminator -- for example the postcode or the phone number on the record?",
            SessionState.IDENTITY_VERIFIED: "Which service is this about?",
            SessionState.AUTHORIZATION_VERIFIED: "Which service is this appointment for?",
            SessionState.REQUEST_SCOPED: "What days or times work for you, and is there a particular practitioner you would like?",
            SessionState.REQUIREMENTS_PENDING: "There are administrative requirements outstanding for this service. Would you like me to read them out exactly as the clinic recorded them?",
            SessionState.REQUIREMENTS_SATISFIED: "Would you like me to look for available times now?",
            SessionState.OPTIONS_PRESENTED: "Which of those times would you like me to hold?",
            SessionState.USER_SELECTED: "I can hold that time while we finish the details. Shall I hold it?",
            SessionState.SLOT_HELD: "That time is held for a short period. Would you like me to read back the details before you confirm?",
            SessionState.SUMMARY_CONFIRMED: "Thanks -- I have your confirmation. Would you like me to go ahead and book it?",
            SessionState.COMMITTING: "I am finalising the booking now. Please give me a moment.",
            SessionState.COMMIT_UNKNOWN: "I could not confirm whether the booking completed. I am going to check the record rather than try again, so we do not end up with two appointments.",
            SessionState.HOLD_EXPIRED: "That hold has expired, which is expected. Would you like me to search again?",
            SessionState.BOOKED: "Is there anything else about this appointment you would like to change?",
            SessionState.RESCHEDULE_PENDING: "What days or times would work better for the new appointment?",
            SessionState.NEW_SLOT_HELD: "Would you like me to move your appointment to that time?",
            SessionState.CANCEL_PENDING: "Just to be sure -- would you like me to cancel that appointment?",
            SessionState.URGENT_HANDOFF: "Please contact emergency services now. A member of the clinic team will follow up with you.",
            SessionState.HUMAN_HANDOFF: "A member of staff will follow up with you about this. Is there anything about your appointment I can help with in the meantime?",
            SessionState.ATOMIC_RESCHEDULED: "The appointment has been moved. Is there anything else you would like to change?",
            SessionState.CANCELLED: "The appointment is cancelled. Would you like me to add you to a waitlist for something sooner?",
        }
        return next_question.get(
            state,
            "Could you tell me a little more about what you need so I can point you to the right next step?",
        )

def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _window_label(window: Any) -> str | None:
    if isinstance(window, dict):
        return f"{window.get('start')}..{window.get('end')}"
    return None


__all__ = ["ResponseComposer"]
