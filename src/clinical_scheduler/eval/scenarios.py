"""Scenario schema and loader.

A scenario is *data*, not code. That is deliberate and load-bearing for three
reasons:

1. **The improvement loop patches data, not the harness.** A failure produces a
   typed patch against a policy bundle or a scenario expectation. If scenarios
   were Python, a "fix" could silently edit the test instead of the system, and
   the before/after comparison would be meaningless.
2. **Holdout isolation is enforceable.** The frozen holdout set is a separate
   directory that the patch generator is never given. A schema boundary makes
   that exclusion checkable rather than a matter of discipline.
3. **Every expectation is reviewable.** Each scenario carries a rationale, an
   owning policy rule where one exists, and an explicit severity. A reviewer can
   read the YAML and disagree without reading any Python.

The schema is strict: unknown keys are rejected. A typo in an expectation key
would otherwise silently produce a scenario that asserts nothing, which is worse
than a scenario that fails.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from clinical_scheduler.domain.states import FrozenStrEnum, StrictModel

SCENARIO_DIR = Path(__file__).with_name("scenarios")


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


class Severity(FrozenStrEnum):
    """Failure severity (plan.md section 11.3).

    The matrix is applied to *failure*, so a scenario's severity is the cost of
    getting it wrong, not the difficulty of getting it right. One S0 vetoes
    release regardless of how good the aggregate score is.
    """

    S0 = "S0"  # catastrophic: unsafe booking under an urgent concern,
    #            wrong-patient disclosure, medication-stop advice
    S1 = "S1"  # high: wrong patient/proxy action, duplicate booking,
    #            false confirmed booking
    S2 = "S2"  # medium: missed requirement, stale availability, poor handoff
    S3 = "S3"  # low: verbosity, style, recoverable confusion


# ---------------------------------------------------------------------------
# Expectations
# ---------------------------------------------------------------------------


class ExpectedRoute(StrictModel):
    """What the safety sentinel must decide for a given patient turn."""

    verdict: Literal[
        "TRIGGERED", "UNCERTAIN", "CLEAR", "CLEAR_NEGATED", "CLEAR_HISTORICAL",
        "CLEAR_THIRD_PERSON", "CLEAR_META",
    ]
    #: Rule ids that must appear in ``fired_rule_ids``. Empty means "no rule must
    #: fire", which is a distinct assertion from "any rule may fire".
    fired_rules: tuple[str, ...] = ()
    #: True when the exact rule ids are the whole expectation; False allows extra
    #: rules to fire as well.
    exact_rules: bool = True


class ExpectedTools(StrictModel):
    """Tool-usage expectations, asserted against the trace."""

    #: Tools that must never be *offered* to the model during the scenario. This
    #: is the strongest form of the mutation-blocking assertion: it checks the
    #: request payload, not a post-hoc refusal.
    forbidden_offered: tuple[str, ...] = ()
    #: Tools that must be offered at least once.
    required_offered: tuple[str, ...] = ()
    #: Tools that must never appear as *executed* in any turn.
    forbidden_executed: tuple[str, ...] = ()


class ExpectedDb(StrictModel):
    """Post-run database assertions.

    Counts are exact integers, not "at least", because an extra row is the
    signature of a duplicate-write bug and "at least" would hide it.
    """

    appointments_created: int | None = None
    appointments_cancelled: int | None = None
    live_leases: int | None = None
    waitlist_entries: int | None = None
    handoffs_created: int | None = None
    escalations_created: int | None = None
    #: When True, ``appointments_created`` must hold for the named patient only.
    appointments_for_subject: str | None = None


class ExpectedText(StrictModel):
    """Substring assertions against the patient-visible assistant message.

    ``must_not_contain`` entries are the deceptive-wording checks: a phrase here
    is one whose presence indicates the agent asserted something it did not do.
    Matching is case-insensitive on the concatenation of all assistant turns.
    """

    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()


class Scenario(StrictModel):
    """One evaluable scenario."""

    #: Identifier shape. ``S01``-style for the core MVP set; ``PFX-01`` for the
    #: derived sets, where the prefix names the family the scenario belongs to
    #: (``ADV`` adversarial, ``MET`` metamorphic, ``HO`` holdout). The prefix is
    #: 2-4 letters so a set can name itself without a schema change.
    scenario_id: str = Field(pattern=r"^S\d{2}$|^[A-Z]{2,4}-\d{2}$")
    title: str = Field(min_length=5)
    family: Literal["core", "adversarial", "metamorphic", "holdout", "counterfactual"]
    severity: str = Field(pattern=r"^S[0-3]$")
    #: Why this scenario exists. Required: a scenario with no stated purpose
    #: cannot be reviewed and tends to be kept long after it stops mattering.
    rationale: str = Field(min_length=20)
    #: The policy rule this scenario is meant to exercise, when one exists.
    policy_rule: str | None = None
    #: Test ids this scenario satisfies, for the policy bundle's traceability.
    test_ids: tuple[str, ...] = ()

    #: Authenticated subject. ``None`` means an unauthenticated session.
    subject_id: str | None = None
    #: Authentication tier, named exactly as in ``AuthTier`` so a rename in the
    #: domain layer becomes a load-time error here rather than a silently wrong
    #: scenario. The names are repeated rather than imported because the loader
    #: must reject an unknown tier even if the enum later gains members.
    tier: Literal[
        "TIER0_PUBLIC", "TIER1_NEW_PATIENT_REQUEST",
        "TIER2_VERIFIED_IDENTITY", "TIER3_STEP_UP",
    ] = "TIER2_VERIFIED_IDENTITY"

    #: The patient turns, in order.
    turns: tuple[str, ...] = Field(min_length=1)
    #: Turn indices at which the runtime confirms the summary (1-based, as they
    #: appear to the reader). A commit may only follow one of these.
    confirm_on_turns: tuple[int, ...] = ()

    #: Per-turn route expectations. Index i applies to turn i. A shorter list
    #: than ``turns`` means the trailing turns carry no route expectation.
    expect_routes: tuple[ExpectedRoute, ...] = ()
    expect_tools: ExpectedTools = ExpectedTools()
    expect_db: ExpectedDb = ExpectedDb()
    expect_text: ExpectedText = ExpectedText()

    #: The single most important thing this scenario proves. Used in the
    #: scorecard so a reader sees the point without opening the YAML.
    key_assertion: str = Field(min_length=10)

    @model_validator(mode="after")
    def _routes_align_with_turns(self) -> "Scenario":
        if len(self.expect_routes) > len(self.turns):
            raise ValueError(
                f"{self.scenario_id}: {len(self.expect_routes)} route expectations "
                f"but only {len(self.turns)} turns"
            )
        for turn in self.confirm_on_turns:
            if turn < 1 or turn > len(self.turns):
                raise ValueError(
                    f"{self.scenario_id}: confirm_on_turns mentions turn {turn} "
                    f"but there are {len(self.turns)} turns"
                )
        return self

    @model_validator(mode="after")
    def _safety_scenarios_assert_no_mutation(self) -> "Scenario":
        """A scenario expecting TRIGGERED must also forbid mutation tools.

        This is not redundant with the runtime's own enforcement. It makes the
        expectation explicit in the scenario file, so a reviewer reading only the
        YAML sees the safety claim being tested. A TRIGGERED scenario with no
        mutation assertion would pass even if every tool were available.
        """
        if any(r.verdict == "TRIGGERED" for r in self.expect_routes):
            forbidden = set(self.expect_tools.forbidden_offered)
            if not forbidden & {
                "hold_slot", "commit_appointment", "atomic_reschedule",
                "cancel_appointment", "join_waitlist",
            }:
                raise ValueError(
                    f"{self.scenario_id}: expects TRIGGERED but does not forbid any "
                    "mutation tool; the safety claim would be untested"
                )
        return self


class ScenarioSet(StrictModel):
    """A named, versioned collection of scenarios."""

    set_id: str = Field(min_length=3)
    set_version: str = Field(min_length=3)
    description: str = Field(min_length=10)
    #: True for sets withheld from the patch generator (plan.md section 11.4).
    frozen_holdout: bool = False
    scenarios: tuple[Scenario, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ScenarioSet":
        ids = [s.scenario_id for s in self.scenarios]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate scenario_id(s) in {self.set_id}: {dupes}")
        return self

    def by_severity(self) -> dict[str, list[Scenario]]:
        out: dict[str, list[Scenario]] = {}
        for scenario in self.scenarios:
            out.setdefault(scenario.severity, []).append(scenario)
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_scenario_file(path: Path) -> ScenarioSet:
    """Load one YAML scenario file into a validated set.

    An invalid file raises rather than loading a partial set. A partially-loaded
    scenario set would silently reduce coverage, and the scorecard would report
    a better number for having tested less.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: expected a mapping at the top level")
    return ScenarioSet.model_validate(raw)


def load_all_scenarios(
    directory: Path | None = None, *, include_holdout: bool = False
) -> list[ScenarioSet]:
    """Load every scenario set in the directory.

    Holdout sets are excluded by default. ``include_holdout=True`` is used only
    by the final verification run, never by the patch generator.
    """
    root = directory or SCENARIO_DIR
    sets: list[ScenarioSet] = []
    for path in sorted(root.glob("*.yaml")):
        sets.append(load_scenario_file(path))
    if include_holdout:
        for path in sorted((root / "holdout").glob("*.yaml")):
            sets.append(load_scenario_file(path))
    return sets


def all_scenarios(*, include_holdout: bool = False) -> list[Scenario]:
    out: list[Scenario] = []
    for scenario_set in load_all_scenarios(include_holdout=include_holdout):
        out.extend(scenario_set.scenarios)
    return out


__all__ = [
    "ExpectedDb",
    "ExpectedRoute",
    "ExpectedText",
    "ExpectedTools",
    "SCENARIO_DIR",
    "Scenario",
    "ScenarioSet",
    "Severity",
    "all_scenarios",
    "load_all_scenarios",
    "load_scenario_file",
]
