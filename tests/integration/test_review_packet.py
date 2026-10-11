"""Layer 5, end to end: a real run directory must yield a usable review packet.

``review.export`` reads ``traces.jsonl`` from a *completed* run rather than from
in-memory objects, because a review is performed on a frozen run -- a reviewer
looking at a live object would be reviewing something that can still change under
them. That decoupling is what this test pins: the driver writes the run, the
review layer reads only what was written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_scheduler.adapters.loopback import LoopbackModel, completion
from clinical_scheduler.eval.driver import (
    EvaluationDriver,
    aggregate_versions,
    flatten,
    load_sets,
)
from clinical_scheduler.eval.review import (
    export_packet,
    load_packet,
    load_run_rows,
    sampling_frame,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]

#: A non-committal reply, chosen so no scenario's expectation can be satisfied
#: *by* the script. Any finding produced under it belongs to the runtime.
BENIGN = [completion("Thanks for that. Which day would suit you best?")]


def test_a_completed_run_is_sampled_into_a_packet(tmp_path: Path) -> None:
    scenario_sets = [s for s in load_sets() if s.set_id == "core"]
    scenarios = flatten(scenario_sets)[:3]
    assert scenarios, "the core set is empty; this test needs scenarios"

    with LoopbackModel(BENIGN) as server:
        driver = EvaluationDriver(
            work_dir=tmp_path / "work",
            out_dir=tmp_path / "out",
            repo_root=REPO_ROOT,
        )
        _manifest, scorecard, _outcomes, _results = driver.run(
            scenarios,
            scenario_set_versions=aggregate_versions(scenario_sets),
            holdout_included=False,
            adapter=server.build_adapter(),
            write=True,
        )

    run_dir = tmp_path / "out" / scorecard.manifest_fingerprint
    rows = load_run_rows(run_dir)
    assert len(rows) == len(scenarios), "the run directory does not carry one row per scenario"

    items = sampling_frame(rows, fraction=1.0, seed=0)
    assert len(items) == len(rows)

    packet = export_packet(items, tmp_path / "packet")
    loaded = load_packet(packet)
    assert [item.item_id for item in loaded] == [item.item_id for item in items]
    assert all(item.reply is not None for item in loaded)
