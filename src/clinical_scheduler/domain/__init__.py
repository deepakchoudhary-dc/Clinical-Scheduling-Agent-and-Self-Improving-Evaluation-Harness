"""Domain layer: vocabulary, FSM, errors and invariants.

Nothing in this package imports from `kernel`, `agent`, `tools` or `eval`. The
dependency direction is one-way so that the deterministic core can be reasoned
about and tested without a model, a database or a network.
"""

from clinical_scheduler.domain import errors, fsm, invariants, models, states

__all__ = ["errors", "fsm", "invariants", "models", "states"]
