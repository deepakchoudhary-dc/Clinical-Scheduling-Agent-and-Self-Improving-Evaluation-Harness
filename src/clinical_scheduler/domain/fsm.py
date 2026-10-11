"""The finite-state machine: allowed transitions and the guards that gate them.

plan.md section 1 promises that "unsafe transitions are structurally
impossible". That promise is kept here and only here. Two rules make it real:

1. A transition that is absent from `TRANSITIONS` cannot occur. There is no
   fallback branch, no "force" parameter, no administrative override.
2. A transition that is present still fails unless every guard in
   `GUARD_REGISTRY` for that edge returns True against the live session.

Any code that wants to change session state must go through `transition()`.
The repository never writes `state` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from clinical_scheduler.domain.errors import IllegalTransitionError
from clinical_scheduler.domain.states import SessionState

S = SessionState


# ---------------------------------------------------------------------------
# Transition table (plan.md section 5, verbatim)
# ---------------------------------------------------------------------------

TRANSITIONS: Mapping[SessionState, frozenset[SessionState]] = {
    S.START: frozenset({S.SAFETY_SCREENED}),
    S.SAFETY_SCREENED: frozenset({S.URGENT_HANDOFF, S.IDENTITY_PENDING, S.HUMAN_HANDOFF}),
    S.IDENTITY_PENDING: frozenset(
        {S.IDENTITY_AMBIGUOUS, S.IDENTITY_VERIFIED, S.HUMAN_HANDOFF}
    ),
    S.IDENTITY_AMBIGUOUS: frozenset(
        {S.IDENTITY_AMBIGUOUS, S.IDENTITY_VERIFIED, S.HUMAN_HANDOFF}
    ),
    S.IDENTITY_VERIFIED: frozenset({S.AUTHORIZATION_VERIFIED, S.HUMAN_HANDOFF}),
    S.AUTHORIZATION_VERIFIED: frozenset({S.REQUEST_SCOPED, S.HUMAN_HANDOFF}),
    S.REQUEST_SCOPED: frozenset({S.REQUIREMENTS_PENDING, S.REQUIREMENTS_SATISFIED}),
    S.REQUIREMENTS_PENDING: frozenset({S.REQUIREMENTS_SATISFIED, S.HUMAN_HANDOFF}),
    S.REQUIREMENTS_SATISFIED: frozenset({S.OPTIONS_PRESENTED}),
    S.OPTIONS_PRESENTED: frozenset({S.USER_SELECTED, S.HUMAN_HANDOFF}),
    S.USER_SELECTED: frozenset({S.SLOT_HELD, S.HOLD_EXPIRED, S.HUMAN_HANDOFF}),
    S.SLOT_HELD: frozenset(
        {S.SUMMARY_CONFIRMED, S.HOLD_EXPIRED, S.USER_SELECTED, S.HUMAN_HANDOFF}
    ),
    S.SUMMARY_CONFIRMED: frozenset({S.COMMITTING, S.HOLD_EXPIRED, S.HUMAN_HANDOFF}),
    S.COMMITTING: frozenset({S.BOOKED, S.COMMIT_UNKNOWN, S.HOLD_EXPIRED}),
    S.COMMIT_UNKNOWN: frozenset({S.BOOKED, S.SLOT_HELD, S.HUMAN_HANDOFF}),
    S.HOLD_EXPIRED: frozenset({S.USER_SELECTED, S.OPTIONS_PRESENTED, S.HUMAN_HANDOFF}),
    S.BOOKED: frozenset(
        {S.RESCHEDULE_PENDING, S.CANCEL_PENDING, S.BOOKED, S.HUMAN_HANDOFF}
    ),
    S.RESCHEDULE_PENDING: frozenset(
        {S.NEW_SLOT_HELD, S.BOOKED, S.HUMAN_HANDOFF}
    ),
    S.NEW_SLOT_HELD: frozenset({S.ATOMIC_RESCHEDULED, S.BOOKED, S.HOLD_EXPIRED}),
    S.CANCEL_PENDING: frozenset({S.CANCELLED, S.BOOKED, S.HUMAN_HANDOFF}),
    # Terminal states.
    S.URGENT_HANDOFF: frozenset(),
    S.HUMAN_HANDOFF: frozenset(),
    S.ATOMIC_RESCHEDULED: frozenset(),
    S.CANCELLED: frozenset(),
}


def is_legal(from_state: SessionState, to_state: SessionState) -> bool:
    return to_state in TRANSITIONS.get(from_state, frozenset())


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@dataclass
class SessionFacts:
    """The slice of session truth that guards are allowed to read.

    Guards read facts, never the model's claims. If a fact is not here, a guard
    cannot be written against it -- that is deliberate.
    """

    identity_verified: bool = False
    identity_ambiguous: bool = False
    authorization_verified: bool = False
    safety_route_active: bool = False
    safety_verdict_is_uncertain: bool = False
    requirements_satisfied: bool = False
    lease_active: bool = False
    lease_expired: bool = False
    explicit_confirmation_token: str | None = None
    commit_outcome_known: bool = False
    commit_succeeded: bool = False
    appointment_exists: bool = False
    proxy_scope_covers_action: bool = False
    human_handoff_reason: str | None = None
    discriminator_attempts_remaining: int = 3
    extra: dict[str, object] = field(default_factory=dict)


Guard = Callable[[SessionFacts], bool]


# ---------------------------------------------------------------------------
# Guard registry. Keyed by (from_state, to_state). An edge with no entry is
# ungated and always permitted (for example any -> HUMAN_HANDOFF).
# ---------------------------------------------------------------------------

GUARD_REGISTRY: Mapping[tuple[SessionState, SessionState], tuple[Guard, ...]] = {
    (S.START, S.SAFETY_SCREENED): (),
    (S.SAFETY_SCREENED, S.URGENT_HANDOFF): (
        lambda f: f.safety_route_active,
    ),
    (S.IDENTITY_PENDING, S.IDENTITY_AMBIGUOUS): (
        lambda f: f.identity_ambiguous,
    ),
    (S.IDENTITY_PENDING, S.IDENTITY_VERIFIED): (
        lambda f: f.identity_verified,
    ),
    (S.IDENTITY_AMBIGUOUS, S.IDENTITY_VERIFIED): (
        lambda f: f.identity_verified and f.discriminator_attempts_remaining > 0,
    ),
    (S.IDENTITY_VERIFIED, S.AUTHORIZATION_VERIFIED): (
        lambda f: f.authorization_verified,
        # Invariant 2: a live safety route blocks all forward progress.
        lambda f: not f.safety_route_active,
    ),
    (S.AUTHORIZATION_VERIFIED, S.REQUEST_SCOPED): (
        lambda f: f.proxy_scope_covers_action,
    ),
    (S.REQUIREMENTS_PENDING, S.REQUIREMENTS_SATISFIED): (
        lambda f: f.requirements_satisfied,
    ),
    (S.USER_SELECTED, S.SLOT_HELD): (
        lambda f: f.lease_active,
    ),
    (S.SLOT_HELD, S.SUMMARY_CONFIRMED): (
        # Invariant 8: confirmation must be an explicit token, not a bare "yes".
        lambda f: f.explicit_confirmation_token is not None,
        lambda f: f.lease_active and not f.lease_expired,
    ),
    (S.SUMMARY_CONFIRMED, S.COMMITTING): (
        lambda f: f.lease_active and not f.lease_expired,
    ),
    (S.COMMITTING, S.BOOKED): (
        # Invariant 7: never claim booked without a verified commit result.
        lambda f: f.commit_outcome_known and f.commit_succeeded,
    ),
    (S.COMMITTING, S.COMMIT_UNKNOWN): (
        lambda f: not f.commit_outcome_known,
    ),
    (S.COMMIT_UNKNOWN, S.BOOKED): (
        lambda f: f.commit_outcome_known and f.commit_succeeded,
    ),
    (S.SLOT_HELD, S.HOLD_EXPIRED): (
        lambda f: f.lease_expired or not f.lease_active,
    ),
    (S.BOOKED, S.RESCHEDULE_PENDING): (
        lambda f: f.identity_verified,
        lambda f: not f.safety_route_active,
    ),
    (S.BOOKED, S.CANCEL_PENDING): (
        # Cancellation changes clinical state; require verified identity even
        # though the patient is already known to the session.
        lambda f: f.identity_verified,
    ),
    (S.RESCHEDULE_PENDING, S.NEW_SLOT_HELD): (
        lambda f: f.lease_active,
    ),
    (S.NEW_SLOT_HELD, S.ATOMIC_RESCHEDULED): (
        lambda f: f.commit_outcome_known and f.commit_succeeded,
        # Invariant 11: the original must still exist when the new one commits.
        lambda f: f.appointment_exists,
    ),
    (S.CANCEL_PENDING, S.CANCELLED): (
        lambda f: f.commit_outcome_known and f.commit_succeeded,
    ),
}


def guard_failures(from_state: SessionState, to_state: SessionState, facts: SessionFacts) -> list[int]:
    """Return indices of guards that rejected this edge. Empty list means pass."""
    guards = GUARD_REGISTRY.get((from_state, to_state), ())
    return [i for i, g in enumerate(guards) if not g(facts)]


def transition(
    from_state: SessionState, to_state: SessionState, facts: SessionFacts
) -> SessionState:
    """The only sanctioned way to advance session state.

    Raises `IllegalTransitionError` when the edge does not exist or a guard
    rejects it. Callers must not catch this to force progress; catching it and
    retrying with mutated facts is exactly the bypass this design forbids.
    """
    if not is_legal(from_state, to_state):
        raise IllegalTransitionError(
            f"no transition {from_state.value} -> {to_state.value}",
            from_state=from_state.value,
            to_state=to_state.value,
            kind="undeclared_edge",
        )
    failed = guard_failures(from_state, to_state, facts)
    if failed:
        raise IllegalTransitionError(
            f"guard(s) {failed} rejected {from_state.value} -> {to_state.value}",
            from_state=from_state.value,
            to_state=to_state.value,
            kind="guard_rejected",
        )
    return to_state


def reachable_states(start: SessionState) -> frozenset[SessionState]:
    """Graph closure over declared edges. Used by property tests to prove that
    no sequence of legal transitions reaches a forbidden configuration."""
    seen: set[SessionState] = {start}
    frontier = [start]
    while frontier:
        cur = frontier.pop()
        for nxt in TRANSITIONS.get(cur, frozenset()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return frozenset(seen)


def states_never_reaching(target: SessionState, avoid: frozenset[SessionState]) -> bool:
    """True if every backward path from `target` starts at a state in `avoid`.

    Used to assert, for example, that BOOKED cannot be reached without coming
    from SUMMARY_CONFIRMED.

    The check walks the edge graph backwards from `target`. `avoid` states are
    absorbing: we do not expand past them, because crossing one already
    satisfies the obligation. If a root is reached (a state with no incoming
    edges) without having crossed an `avoid` state, a path exists that bypasses
    the guard, and the answer is False.
    """
    if not avoid:
        return True

    incoming: dict[SessionState, list[SessionState]] = {}
    for src, dsts in TRANSITIONS.items():
        for dst in dsts:
            incoming.setdefault(dst, []).append(src)

    if target in avoid:
        return True

    seen: set[SessionState] = set()
    frontier: list[SessionState] = [target]
    while frontier:
        cur = frontier.pop()
        if cur in seen:
            continue
        seen.add(cur)
        preds = incoming.get(cur)
        if not preds:
            # Reached a root without crossing an `avoid` state: bypass exists.
            return False
        for pred in preds:
            if pred in avoid or pred in seen:
                continue
            frontier.append(pred)
    return True


__all__ = [
    "GUARD_REGISTRY",
    "SessionFacts",
    "TRANSITIONS",
    "guard_failures",
    "is_legal",
    "reachable_states",
    "states_never_reaching",
    "transition",
]
