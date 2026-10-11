"""Run the closed-loop repair demonstration end to end, offline.

This script is the reproducible form of the demonstration: it wires the real
components together and writes the resulting artifacts and checksums to disk.
Nothing here is stubbed except the *model transport*.

What is real
------------
* the policy engine, the FSM, the capability broker and the scheduling kernel;
* the SQLite database and its fixtures, built fresh per scenario;
* the five-layer oracle stack, including route conformance;
* the RCA package, the patch schema, the validator, the acceptance gate and the
  holdout comparison;
* the model *adapter* -- the bytes travel over a real HTTP socket to a real
  server and are parsed by the same code path a hosted endpoint would exercise.

What is not real, and is labelled as such
-----------------------------------------
* the **transport** serves a scripted reply from a loopback server rather than
  calling a model. It exists so the loop can be demonstrated with no endpoint and
  no credential. It cannot produce a quality signal, and no number in the output
  is a statement about model behaviour.
* the **generator** is the curated table in ``improvement/demo_repairs.py``, not
  a model. See that module for why the substitution is safe here and what a
  model-backed generator would have to do instead.
* the **approval** is simulated, and every artifact says so.

Usage::

    python scripts/demo_closed_loop.py --target ADV-11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clinical_scheduler.adapters.loopback import LoopbackModel, benign_script  # noqa: E402
from clinical_scheduler.eval.driver import aggregate_versions, load_sets  # noqa: E402
from clinical_scheduler.improvement.demo_repairs import DEMO_REPAIR_TABLE  # noqa: E402
from clinical_scheduler.improvement.optimizer import (  # noqa: E402
    LoopConfig,
    run_repair,
    write_loop_artifacts,
)
from clinical_scheduler.improvement.scripted_generator import (  # noqa: E402
    ScriptedGenerator,
    approver_accept_when_gate_accepts,
)
from clinical_scheduler.policy.engine import load_baseline  # noqa: E402

#: The shared non-committal script (``adapters.loopback.benign_script``): it makes
#: no booking claim and states no availability, so no scenario's expectations can be
#: satisfied *by* the reply and any finding belongs to the deterministic layers.
#:
#: **The consequence, which is easy to miss and was missed for a long time:** it
#: never proposes a tool call, so no tool is ever invoked. The run covers the
#: *safety-routing* path only. No appointment is created, no lease is taken, no
#: outbox row is written and **no audit row is written** -- a scan of every scenario
#: database this demonstration has produced found zero of each. Nothing in its
#: scorecards is evidence about booking, idempotency, concurrency or the audit
#: chain. Those are covered by the test suite, and by ``demo_audit.py``, which drives
#: a real invocation and leaves a database with a verified audit chain.
BENIGN_SCRIPT = benign_script()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default="ADV-11", help="scenario id to repair")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "repair"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    work_dir = out_dir / "_work"

    all_sets = load_sets(include_holdout=True)
    dev_sets = [s for s in all_sets if not s.frozen_holdout]
    holdout_sets = [s for s in all_sets if s.frozen_holdout]

    print(f"project      : {PROJECT_ROOT}")
    print(f"development  : {[s.set_id for s in dev_sets]} "
          f"({sum(len(s.scenarios) for s in dev_sets)} scenarios)")
    print(f"holdout      : {[s.set_id for s in holdout_sets]} "
          f"({sum(len(s.scenarios) for s in holdout_sets)} scenarios)")
    print(f"target       : {args.target}")
    print()

    baseline = load_baseline()
    print(f"baseline policy : {baseline.bundle.policy_version} "
          f"({baseline.bundle_hash()[:16]}…)")
    print()

    config = LoopConfig(
        work_dir=work_dir,
        out_dir=out_dir,
        repo_root=PROJECT_ROOT,
        generator=ScriptedGenerator(DEMO_REPAIR_TABLE),
        approver=approver_accept_when_gate_accepts(),
        seed=args.seed,
        temperature=args.temperature,
        max_turns=args.max_turns,
        notes=[
            "offline demonstration: loopback transport, scripted generator, "
            "simulated approval",
        ],
    )

    with LoopbackModel(BENIGN_SCRIPT) as server:
        result = run_repair(
            config=config,
            baseline_engine=baseline,
            dev_sets=dev_sets,
            holdout_sets=holdout_sets,
            adapter=server.build_adapter(),
            target_scenario_id=args.target,
        )

    if result is None:
        print(f"nothing to repair: {args.target} passed under the baseline.")
        return 0

    package = result.package
    classification = package.classification
    print("=" * 78)
    print("RCA")
    print("=" * 78)
    print(f"  package        : {package.package_id}")
    print(f"  expected       : {package.expected_outcome}")
    print(f"  actual         : {package.actual_outcome}")
    if classification is not None:
        print(f"  failure class  : {classification.failure_class.value}")
        print(f"  required layer : {classification.required_layer.value}")
        print(f"  implicated rule: {classification.implicated_rule_id}")
        print(f"  hypothesis     : {classification.hypothesis}")
    print(f"  reproducer     : {package.minimal_reproducer!r}")
    print()

    print("=" * 78)
    print("CANDIDATE PATCH")
    print("=" * 78)
    print(f"  patch id       : {result.patch.patch_id}")
    print(f"  layer          : {result.patch.layer.value}")
    print(f"  versions       : {result.patch.base_policy_version} "
          f"-> {result.patch.target_policy_version}")
    for op in result.patch.operations:
        target = op.rule_id or op.template_id
        print(f"    - {op.op.value} {target} = {op.value!r} "
              f"polarity={op.polarity}")
    print()

    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    print(result.validation.report())
    print()

    print("=" * 78)
    print("GATE")
    print("=" * 78)
    print(result.gate.report())
    print()

    print("=" * 78)
    print("APPROVAL")
    print("=" * 78)
    approval = result.approval.to_dict()
    for key in ("decision", "approver_role", "simulated", "patch_content_hash"):
        print(f"  {key:<18}: {approval[key]}")
    print(f"  rationale         : {approval['rationale']}")
    print()

    print("=" * 78)
    print("OUTCOME")
    print("=" * 78)
    print(f"  fixed scenarios    : {result.fixed_scenarios}")
    print(f"  regressed scenarios: {result.regressed_scenarios}")
    if result.baseline_scorecard and result.candidate_scorecard:
        b, c = result.baseline_scorecard, result.candidate_scorecard
        print(f"  dev findings before: {b.severity_counts}  eligible={b.eligible}")
        print(f"  dev findings after : {c.severity_counts}  eligible={c.eligible}")
    for label, card in (
        ("holdout before", result.holdout_baseline_scorecard),
        ("holdout after ", result.holdout_scorecard),
    ):
        if card is not None:
            passed = sum(1 for s in card.scenarios if s.passed)
            print(f"  {label}     : {passed}/{len(card.scenarios)} passed, "
                  f"{card.severity_counts}")
    if result.gate.caveats:
        print()
        print("  caveats carried by this decision:")
        for caveat in result.gate.caveats:
            print(f"    - {caveat}")
    print()

    cycle_dir = write_loop_artifacts(result, out_dir)
    print(f"artifacts written to: {cycle_dir}")
    for path in sorted(cycle_dir.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
