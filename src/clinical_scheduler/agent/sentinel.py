"""Safety sentinel: layered screening that produces a routing decision.

Layer structure (plan.md section 10.1), in order:

1. **Deterministic high-recall pattern matching**, delegated to the policy
   engine, which matches only clinician-approved patterns held as data.
2. **Uncertainty route**, also from the policy engine: safety-adjacent
   vocabulary without a matching rule produces UNCERTAIN, which blocks mutation
   and hands off rather than booking.
3. **Mutation gate**, applied by the capability broker using the verdict.
4. **Patient-safe response template**, selected by the policy engine.

What this module deliberately does NOT do: it does not assign acuity, it does
not diagnose, and it does not attempt semantic paraphrase detection. A semantic
classifier is listed in the plan as an optional future layer that must never be
the sole protection; it is out of scope here, and the honest consequence is
recorded in docs/safety-case.md: a paraphrase outside the approved pattern set
is a miss, and misses are found by evaluation and fixed with approved data.

The sentinel keeps a per-session record of every verdict so that the oracle
stack can assert invariant 1 (screening happened on *every* turn) from evidence
rather than trusting the orchestrator's control flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from clinical_scheduler.domain.states import SafetyVerdict
from clinical_scheduler.policy.engine import PolicyDecision, PolicyEngine


@dataclass(frozen=True)
class ScreeningRecord:
    """One turn's screening result, retained as audit evidence."""

    turn_index: int
    verdict: SafetyVerdict
    fired_rule_ids: tuple[str, ...]
    matched_phrases: tuple[str, ...]
    policy_version: str
    screened_at: datetime
    text_length: int
    #: Hash-free: the raw text is never stored here. The transcript lives in the
    #: trace, and only when redaction has run.
    text_sha256: str


@dataclass
class SentinelSessionLog:
    """Append-only-in-practice screening history for one session."""

    session_id: str
    records: list[ScreeningRecord] = field(default_factory=list)

    def append(self, record: ScreeningRecord) -> None:
        self.records.append(record)

    @property
    def turn_count(self) -> int:
        return len(self.records)

    def unscreened_turns(self, observed_turn_indices: Sequence[int]) -> list[int]:
        """Turn indices that were observed by the orchestrator but never
        screened. Used by the runtime oracle to detect a skipped sentinel call
        on any turn, not just the first (invariant 1)."""
        screened = {r.turn_index for r in self.records}
        return [i for i in observed_turn_indices if i not in screened]

    def any_mutation_blocking(self) -> bool:
        from clinical_scheduler.domain.states import MUTATION_BLOCKING_VERDICTS

        return any(r.verdict in MUTATION_BLOCKING_VERDICTS for r in self.records)

    def highest_restriction(self) -> SafetyVerdict:
        """The most restrictive verdict seen so far in this session.

        Once a session has triggered a safety route, it stays restricted for the
        remainder of that session: a patient who reports chest pain and then
        changes the subject to ask about booking hours must not be booked. This
        is a deliberate trade-off in favour of safety and is recorded as such.
        """
        for verdict in (SafetyVerdict.TRIGGERED, SafetyVerdict.UNCERTAIN):
            if any(r.verdict is verdict for r in self.records):
                return verdict
        return self.records[-1].verdict if self.records else SafetyVerdict.CLEAR


class SafetySentinel:
    """Screens every patient turn. Must be called by the orchestrator on each
    turn, including turns after the first."""

    def __init__(self, policy: PolicyEngine) -> None:
        self.policy = policy

    def screen(
        self,
        *,
        session_log: SentinelSessionLog,
        turn_index: int,
        text: str,
        service_id: str | None = None,
    ) -> PolicyDecision:
        """Screen one turn and record the result.

        The decision returned is computed from the *current* message, with the
        caller responsible for passing prior context if it wants cross-turn
        screening. The session log separately enforces the sticky-restriction
        rule, so a clear later message cannot release an earlier route.
        """
        import hashlib

        decision = self.policy.evaluate(text, service_id=service_id)
        record = ScreeningRecord(
            turn_index=turn_index,
            verdict=decision.verdict,
            fired_rule_ids=decision.fired_rule_ids,
            matched_phrases=decision.matched_phrases,
            policy_version=decision.policy_version,
            screened_at=datetime.now(timezone.utc),
            text_length=len(text),
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        session_log.append(record)
        return decision

    @staticmethod
    def effective_decision(session_log: SentinelSessionLog, latest: PolicyDecision) -> PolicyDecision:
        """Combine the newest decision with the session's sticky restriction.

        If the session has already triggered, the earlier restriction wins even
        though the latest turn is clear. Without this, a patient could clear a
        safety route by simply waiting one turn, which would make the whole
        router bypassable.
        """
        top = session_log.highest_restriction()
        if top is SafetyVerdict.TRIGGERED and latest.verdict is not SafetyVerdict.TRIGGERED:
            from clinical_scheduler.policy.engine import MUTATION_TOOLS

            return PolicyDecision(
                verdict=SafetyVerdict.TRIGGERED,
                fired_rule_ids=latest.fired_rule_ids,
                matched_phrases=latest.matched_phrases,
                action=latest.action,
                blocked_mutation_tools=frozenset(MUTATION_TOOLS),
                prohibited_claims=latest.prohibited_claims,
                message_template=latest.message_template,
                urgency_category=latest.urgency_category,
                policy_version=latest.policy_version,
                rationale="session already triggered an urgent route; restriction persists",
            )
        return latest


__all__ = [
    "SafetySentinel",
    "ScreeningRecord",
    "SentinelSessionLog",
]
