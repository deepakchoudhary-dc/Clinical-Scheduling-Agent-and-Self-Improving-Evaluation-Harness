"""Produce an evaluation run directory offline, so the review and calibration
commands have something to read.

This is the evaluation counterpart of ``demo_closed_loop.py``. It runs the real
driver, the real oracle stack and the real scoring over a slice of the core
scenario set, with the **labelled loopback transport** standing in for a model.

What that does and does not establish:

* it **does** exercise the whole path end to end -- driver, oracles, scoring,
  artifact writing -- and produces a run directory whose ``traces.jsonl`` is what
  ``csched calibrate`` and ``csched review export`` consume;
* it does **not** say anything about a model's behaviour. The scripted reply is
  deliberately non-committal, so no scenario's expectation can be satisfied *by*
  the script, and any finding produced under it belongs to the deterministic
  layers. The manifest records the run as an offline one for the same reason.

Usage::

    python scripts/offline_run.py [--scenarios 5] [--out artifacts/offline]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clinical_scheduler.adapters.loopback import LoopbackModel, benign_script  # noqa: E402
from clinical_scheduler.eval.driver import (  # noqa: E402
    EvaluationDriver,
    aggregate_versions,
    flatten,
    load_sets,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", type=int, default=5, help="how many core scenarios to run")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "offline"))
    args = parser.parse_args(argv)

    scenario_sets = [s for s in load_sets() if s.set_id == "core"]
    if not scenario_sets:
        print("error: no core scenario set found", file=sys.stderr)
        return 2
    scenarios = flatten(scenario_sets)[: max(args.scenarios, 1)]
    out_dir = Path(args.out)

    with LoopbackModel(benign_script()) as server:
        driver = EvaluationDriver(
            work_dir=out_dir / "_work",
            out_dir=out_dir,
            repo_root=PROJECT_ROOT,
        )
        _manifest, scorecard, _outcomes, _results = driver.run(
            scenarios,
            scenario_set_versions=aggregate_versions(scenario_sets),
            holdout_included=False,
            adapter=server.build_adapter(),
            notes=["offline run: labelled loopback transport, not evidence about a model"],
        )

    run_dir = out_dir / scorecard.manifest_fingerprint
    print(f"scenarios : {len(scenarios)}")
    print(f"eligible  : {scorecard.eligible}")
    print(f"run dir   : {run_dir}")
    print()
    print("next:")
    print(f'  csched calibrate --run "{run_dir}"')
    print(f'  csched review export --run "{run_dir}" --out artifacts/review --fraction 1.0')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
