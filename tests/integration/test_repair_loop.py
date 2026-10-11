"""End-to-end tests for the closed-loop repair pipeline.

This is the only test that runs the whole lifecycle: baseline evaluation, triage,
RCA, generation, static validation, candidate evaluation, the holdout comparison,
the acceptance gate, the approval record and artifact writing.

The model is the loopback transport, so nothing here is a statement about model
quality. What it does establish is that the *machinery* behaves as the design
says, and in particular that:

* the loop repairs its target and the gate still refuses the patch, because the
  hidden holdout shows it over-triggers (see
  ``.agent/knowledge/gotchas/holdout-caught-the-over-trigger.md``);
* every checksum in the artifact set verifies against the bytes on disk -- the
  regression guard for defect D-011, where the manifest hashed text before the
  writer translated newlines and *every* entry failed;
* the holdout baseline scorecard is persisted, so the gate's comparative claim
  can be re-derived from the artifacts alone (defect D-012).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clinical_scheduler.eval.driver import load_sets
from clinical_scheduler.improvement.demo_repairs import DEMO_REPAIR_TABLE
from clinical_scheduler.improvement.optimizer import (
    LoopConfig,
    run_repair,
    write_loop_artifacts,
)
from clinical_scheduler.improvement.scripted_generator import (
    ScriptedGenerator,
    approver_accept_when_gate_accepts,
)
from clinical_scheduler.policy.engine import load_baseline

from tests.support.loopback import LoopbackModel, completion

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]

#: One neutral reply, repeated. Chosen so no scenario's expectations can be
#: satisfied *by* the script: any finding under it is attributable to the
#: deterministic layers rather than to the reply text.
BENIGN_SCRIPT = [
    completion("Thanks for that. I can help with scheduling. Which day would suit you best?")
]


def _config(tmp_path: Path, *, repairs=DEMO_REPAIR_TABLE) -> LoopConfig:
    return LoopConfig(
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "artifacts",
        repo_root=REPO_ROOT,
        generator=ScriptedGenerator(repairs),
        approver=approver_accept_when_gate_accepts(),
        seed=0,
        temperature=0.0,
        notes=["integration test: loopback transport, scripted generator"],
    )


@pytest.fixture(scope="module")
def repair_run(tmp_path_factory) -> dict:
    """One real cycle, shared by the assertions below.

    Module-scoped because a cycle evaluates 52 development scenarios twice and 12
    holdout scenarios twice; re-running it per assertion would make the suite
    slow enough that people stop running it.
    """
    tmp_path = tmp_path_factory.mktemp("repair")
    sets = load_sets(include_holdout=True)
    dev_sets = [s for s in sets if not s.frozen_holdout]
    holdout_sets = [s for s in sets if s.frozen_holdout]

    with LoopbackModel(BENIGN_SCRIPT) as server:
        result = run_repair(
            config=_config(tmp_path),
            baseline_engine=load_baseline(),
            dev_sets=dev_sets,
            holdout_sets=holdout_sets,
            adapter=server.build_adapter(),
            target_scenario_id="ADV-11",
        )
    assert result is not None, "the baseline has a declared failure; a cycle must run"
    return {"result": result, "tmp_path": tmp_path}


# ---------------------------------------------------------------------------
# The cycle reaches a decision
# ---------------------------------------------------------------------------


def test_the_loop_fixes_its_target(repair_run: dict) -> None:
    """The patch does what it was generated to do.

    Without this the rest of the file would be asserting properties of a loop
    that never did anything.
    """
    result = repair_run["result"]
    assert result.target_scenario_id == "ADV-11"
    assert "ADV-11" in result.fixed_scenarios
    assert result.regressed_scenarios == []


def test_the_rca_names_the_cause_and_the_layer(repair_run: dict) -> None:
    """The classification must point at a policy-data gap, not at the prompt."""
    package = repair_run["result"].package
    classification = package.classification
    assert classification is not None
    assert classification.failure_class.value == "SAFETY_SENTINEL_FALSE_NEGATIVE"
    assert classification.required_layer.value == "L4_policy_data"
    assert classification.implicated_rule_id == "SAFE-001"
    assert package.minimal_reproducer.startswith("I think I am having a heart attack")


def test_the_patch_is_typed_and_at_the_required_layer(repair_run: dict) -> None:
    patch = repair_run["result"].patch
    assert patch.layer.value == "L4_policy_data"
    assert patch.base_policy_version != patch.target_policy_version
    assert patch.rollback_target == patch.base_policy_version
    assert patch.new_regression_tests


def test_the_holdout_catches_the_over_trigger(repair_run: dict) -> None:
    """The headline result, asserted rather than described.

    The patch fixes the development sets and is refused because the holdout shows
    the new phrase over-triggers inside a hypothetical frame. If a future change
    makes this pass, that is a finding about the *scenario sets*, not a green run.
    """
    result = repair_run["result"]
    assert result.holdout_baseline_scorecard is not None
    assert result.holdout_scorecard is not None

    before = result.holdout_baseline_scorecard.severity_counts.get("S1", 0)
    after = result.holdout_scorecard.severity_counts.get("S1", 0)
    assert after > before, (
        "the holdout no longer reports a severity rise; the recorded rejection of "
        "this patch depended on it"
    )
    assert not result.gate.accepted
    assert any("holdout" in reason for reason in result.gate.reasons)


def test_the_pass_rate_alone_would_not_have_caught_it(repair_run: dict) -> None:
    """Why severity and pass rate are both compared.

    The holdout pass rate is unchanged while the severity profile worsens,
    because the patch made a *failing* scenario fail worse rather than making a
    passing one fail. A pass-rate-only gate would have accepted this patch.
    """
    result = repair_run["result"]
    before = sum(1 for s in result.holdout_baseline_scorecard.scenarios if s.passed)
    after = sum(1 for s in result.holdout_scorecard.scenarios if s.passed)
    assert before == after, (
        "this test exists because the pass rate held; if it no longer holds, the "
        "point it makes has changed and the test should be rewritten"
    )


def test_a_rejection_is_recorded_with_a_simulated_approval(repair_run: dict) -> None:
    """Approval is a human step, and the record must say it was simulated.

    A prototype that recorded an approval without saying so would invite a reader
    to treat the resulting version as production-approved.
    """
    approval = repair_run["result"].approval
    assert approval.decision == "REJECT"
    assert approval.simulated is True
    assert approval.patch_content_hash == repair_run["result"].patch.content_hash()
    assert approval.rationale


def test_the_approval_is_bound_to_the_patch_content(repair_run: dict) -> None:
    """A patch edited after approval is not the patch that was approved.

    The hash is what makes that checkable.
    """
    result = repair_run["result"]
    edited = result.patch.model_copy(update={"target_policy_version": "2026.10-demo.3"})
    assert edited.content_hash() != result.approval.patch_content_hash


# ---------------------------------------------------------------------------
# Artifacts: present, parseable, and self-verifying
# ---------------------------------------------------------------------------


def test_artifacts_are_written_and_parse(repair_run: dict) -> None:
    cycle_dir = write_loop_artifacts(
        repair_run["result"], repair_run["tmp_path"] / "artifacts"
    )
    for name in (
        "rca.json",
        "candidate_patch.json",
        "validation.txt",
        "gate.txt",
        "approval.json",
        "loop_result.json",
        "scorecard_before.txt",
        "scorecard_after.txt",
        "scorecard_holdout_before.txt",
        "scorecard_holdout_after.txt",
        "candidate_policy.yaml",
        "checksums.sha256",
    ):
        assert (cycle_dir / name).is_file(), f"missing artifact {name}"

    loop_result = json.loads((cycle_dir / "loop_result.json").read_text("utf-8"))
    assert loop_result["gate"]["accepted"] is False
    assert loop_result["approval"]["simulated"] is True
    assert loop_result["patch_content_hash"] == repair_run["result"].patch.content_hash()


def test_every_checksum_verifies_against_the_bytes_on_disk(repair_run: dict) -> None:
    """The D-011 regression guard.

    ``write_loop_artifacts`` hashed the text before ``Path.write_text`` translated
    ``\\n`` to ``\\r\\n``, so on Windows every entry in the manifest failed to
    verify. No test asserted this, which is how it survived. A tamper-evident
    manifest that fails on the machine that produced it trains a reviewer to
    ignore it.
    """
    cycle_dir = write_loop_artifacts(
        repair_run["result"], repair_run["tmp_path"] / "artifacts"
    )
    lines = (cycle_dir / "checksums.sha256").read_text("utf-8").splitlines()
    assert len(lines) >= 11, "the manifest should cover every artifact"

    for line in lines:
        digest, name = line.split("  ", 1)
        actual = hashlib.sha256((cycle_dir / name).read_bytes()).hexdigest()
        assert digest == actual, f"checksum mismatch for {name}"


def test_the_holdout_before_scorecard_is_persisted(repair_run: dict) -> None:
    """The D-012 regression guard.

    The gate's holdout question is comparative. If only the *after* scorecard is
    written, a reviewer cannot re-derive the comparison without re-running the
    loop, and the gate's claim becomes unverifiable.
    """
    cycle_dir = write_loop_artifacts(
        repair_run["result"], repair_run["tmp_path"] / "artifacts"
    )
    before = (cycle_dir / "scorecard_holdout_before.txt").read_text("utf-8")
    after = (cycle_dir / "scorecard_holdout_after.txt").read_text("utf-8")
    assert "ELIGIBILITY" in before and "ELIGIBILITY" in after
    assert before != after, "the two holdout scorecards must record different states"


# ---------------------------------------------------------------------------
# The loop declines to act when there is nothing to repair
# ---------------------------------------------------------------------------


def test_no_matching_repair_returns_none(tmp_path: Path) -> None:
    """An empty repair table means the generator proposes nothing.

    ``None`` is a distinct outcome from a rejected patch: conflating them would
    make a generator that simply had nothing to say look like a failed repair.
    """
    sets = load_sets(include_holdout=True)
    with LoopbackModel(BENIGN_SCRIPT) as server:
        result = run_repair(
            config=_config(tmp_path, repairs=()),
            baseline_engine=load_baseline(),
            dev_sets=[s for s in sets if not s.frozen_holdout],
            holdout_sets=[s for s in sets if s.frozen_holdout],
            adapter=server.build_adapter(),
            target_scenario_id="ADV-11",
        )
    assert result is None


def test_an_unknown_target_returns_none(tmp_path: Path) -> None:
    """A target that is not failing has nothing to repair."""
    sets = load_sets(include_holdout=True)
    with LoopbackModel(BENIGN_SCRIPT) as server:
        result = run_repair(
            config=_config(tmp_path),
            baseline_engine=load_baseline(),
            dev_sets=[s for s in sets if not s.frozen_holdout],
            holdout_sets=[s for s in sets if s.frozen_holdout],
            adapter=server.build_adapter(),
            target_scenario_id="S01",  # a passing core scenario
        )
    assert result is None
