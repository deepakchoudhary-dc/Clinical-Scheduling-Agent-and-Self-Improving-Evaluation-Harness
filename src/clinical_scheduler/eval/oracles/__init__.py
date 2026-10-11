"""Oracle stack (plan.md section 11.2). See ``stack`` for the implementation.

The re-exports below are the package's public surface. ``MUTATION_TOOLS``,
``READ_TOOLS`` and ``EXPECTED_TOOLS_BY_STATE`` are included deliberately: a
caller building a synthetic record to test the oracles, or a reviewer writing a
new check, needs the same vocabulary the oracles use. Keeping them private would
push those callers to re-derive the sets locally, and a second copy of "which
tools mutate state" is exactly the kind of duplication that drifts.
"""

from clinical_scheduler.eval.oracles.stack import (
    ALL_ORACLES,
    EXPECTED_TOOLS_BY_STATE,
    MUTATION_TOOLS,
    READ_TOOLS,
    Finding,
    OracleLayer,
    OracleResult,
    RunRecord,
    l0_schema_and_runtime,
    l1_safety_and_privacy,
    l2_state_diff,
    l3_trace_conformance,
    l4_conversation_quality,
    run_oracles,
)

__all__ = [
    "ALL_ORACLES",
    "EXPECTED_TOOLS_BY_STATE",
    "Finding",
    "MUTATION_TOOLS",
    "OracleLayer",
    "OracleResult",
    "READ_TOOLS",
    "RunRecord",
    "l0_schema_and_runtime",
    "l1_safety_and_privacy",
    "l2_state_diff",
    "l3_trace_conformance",
    "l4_conversation_quality",
    "run_oracles",
]
