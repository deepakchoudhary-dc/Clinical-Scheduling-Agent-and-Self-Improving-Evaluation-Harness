"""Agent layer: the LLM-facing surface.

This package is where an untrusted model meets the constrained kernel. Every
module here is written on the assumption that model output is *proposals*, not
decisions:

* ``prompt``             -- builds the system prompt from the policy bundle. The
                            prompt shapes usefulness; it is never a control.
* ``sentinel``           -- screens every turn before the model is invoked.
* ``response_grounding`` -- subtracts any factual claim no tool authorised.
* ``orchestrator``       -- orders the per-turn sequence and is the only place
                            that advances the session FSM.

Dependency direction: ``agent`` may import from ``policy``, ``tools``,
``kernel`` and ``domain``. Nothing below ``agent`` may import from it, so the
deterministic core remains testable without a model.
"""

from clinical_scheduler.agent.orchestrator import (
    TOOL_DESCRIPTIONS,
    Orchestrator,
    TurnResult,
    derive_tool_specs,
)
from clinical_scheduler.agent.prompt import build_system_prompt
from clinical_scheduler.agent.response_grounding import ResponseComposer
from clinical_scheduler.agent.sentinel import (
    SafetySentinel,
    ScreeningRecord,
    SentinelSessionLog,
)

__all__ = [
    "Orchestrator",
    "ResponseComposer",
    "SafetySentinel",
    "ScreeningRecord",
    "SentinelSessionLog",
    "TOOL_DESCRIPTIONS",
    "TurnResult",
    "build_system_prompt",
    "derive_tool_specs",
]
