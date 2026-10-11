"""The FSM's transition table, checked as a graph rather than read as a comment.

``reachable_states`` and ``states_never_reaching`` shipped with the FSM and had
**no caller**: the properties they compute were asserted in prose and checked
nowhere. A graph helper nobody calls is not a guarantee, and the static audit of
2026-10-08 found them (along with a docstring that said they were used by
"property tests"). These tests are the callers.

The negative control is as important as the positive ones. A property check that
can only ever return ``True`` proves nothing, which is the lesson this project
already learned about its oracles (defect D-007): a green harness is worthless
unless it can disagree.
"""

from __future__ import annotations

import pytest

from clinical_scheduler.domain.fsm import (
    TRANSITIONS,
    reachable_states,
    states_never_reaching,
)
from clinical_scheduler.domain.states import TERMINAL_STATES, SessionState as S

pytestmark = pytest.mark.property


def test_the_check_can_return_false() -> None:
    """The negative control. Without this the tests below prove nothing.

    There *is* a path to BOOKED that never crosses URGENT_HANDOFF -- the ordinary
    booking flow -- so the predicate must say so.
    """
    assert states_never_reaching(S.BOOKED, frozenset({S.URGENT_HANDOFF})) is False


def test_every_declared_state_is_reachable_from_start() -> None:
    """No state is declared and then stranded."""
    assert reachable_states(S.START) == frozenset(S)


def test_booked_cannot_be_reached_without_committing() -> None:
    """The structural half of INV-07: never claim BOOKED without a commit.

    Every backward path from ``BOOKED`` passes through ``COMMITTING``, so no
    sequence of legal transitions can arrive at a booked session without having
    gone through the commit state. The invariant's other half -- that the commit
    succeeded -- is enforced by the guard on ``COMMITTING -> BOOKED``.
    """
    assert states_never_reaching(S.BOOKED, frozenset({S.COMMITTING})) is True


def test_cancelled_cannot_be_reached_without_cancel_pending() -> None:
    assert states_never_reaching(S.CANCELLED, frozenset({S.CANCEL_PENDING})) is True


def test_rescheduled_cannot_be_reached_without_holding_a_new_slot() -> None:
    """INV-11: an atomic reschedule goes through a new hold, never around it."""
    assert (
        states_never_reaching(S.ATOMIC_RESCHEDULED, frozenset({S.NEW_SLOT_HELD})) is True
    )


def test_an_empty_avoid_set_is_vacuously_true() -> None:
    """Documented behaviour: nothing to cross means nothing to prove."""
    assert states_never_reaching(S.BOOKED, frozenset()) is True


def test_terminal_states_have_no_outgoing_edges() -> None:
    """A terminal state that can still move is not terminal."""
    for state in TERMINAL_STATES:
        assert TRANSITIONS[state] == frozenset(), f"{state.value} is declared terminal but has edges"


def test_every_state_has_a_declared_entry_in_the_transition_table() -> None:
    """``is_legal`` falls back to an empty set for an absent key, which would make
    an undeclared state silently terminal. Every state must be explicit."""
    assert set(TRANSITIONS) == set(S)
