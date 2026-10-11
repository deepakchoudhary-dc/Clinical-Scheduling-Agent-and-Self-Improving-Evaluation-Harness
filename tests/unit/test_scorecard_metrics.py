"""The scorecard must report the metrics it can compute, and name the ones it cannot.

plan.md section 11.3 lists eleven metrics. Seven were computed; ``p50/p95``
latency, ``cost_per_completed_task``, ``accessibility_quality`` and
``judge_disagreement_rate`` were simply absent from the scorecard. An absent
metric reads as a passing one, which is the failure mode this module exists to
avoid -- so latency is now computed from what the transport reported, and the
three that have no input are printed *with the input each would need*.

The ``None``-not-zero convention is load-bearing and is tested here: "we never
measured this" and "we measured it and it was zero" are different claims.
"""

from __future__ import annotations

import types
from typing import Any

from clinical_scheduler.eval.scoring import (
    UNMEASURED_METRICS,
    Rate,
    Scorecard,
    _compute_latency,
    _percentile,
)


def _outcome(latencies: list[float] | None) -> Any:
    """A minimal stand-in: ``_compute_latency`` only reads model_metadata."""
    metadata: dict[str, Any] = {} if latencies is None else {"latencies_ms": latencies}
    return types.SimpleNamespace(record=types.SimpleNamespace(model_metadata=metadata))


# -- percentile ---------------------------------------------------------------


def test_percentile_of_an_empty_sample_is_none_not_zero() -> None:
    assert _percentile([], 0.5) is None


def test_percentile_is_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert _percentile(values, 0.50) == 20.0
    assert _percentile(values, 0.95) == 40.0
    assert _percentile(values, 0.0) == 10.0
    assert _percentile(values, 1.0) == 40.0


# -- latency ------------------------------------------------------------------


def test_latency_is_none_when_the_transport_reported_nothing() -> None:
    latency = _compute_latency([_outcome(None), _outcome([])])
    assert latency == {"latency_p50_ms": None, "latency_p95_ms": None}


def test_latency_percentiles_are_computed_across_scenarios() -> None:
    latency = _compute_latency([_outcome([1.0, 2.0]), _outcome([3.0, 4.0])])
    assert latency["latency_p50_ms"] == 2.0
    assert latency["latency_p95_ms"] == 4.0


# -- unmeasured metrics -------------------------------------------------------


def test_the_three_metrics_without_inputs_are_named_with_their_reason() -> None:
    assert set(UNMEASURED_METRICS) == {
        "cost_per_completed_task",
        "accessibility_quality",
        "judge_disagreement_rate",
    }
    for name, reason in UNMEASURED_METRICS.items():
        assert reason, f"{name} is listed as unmeasured with no reason"
        assert len(reason) > 20, f"{name}'s reason does not explain what it would need"


# -- rendering ----------------------------------------------------------------


def _card(**overrides: Any) -> Scorecard:
    base: dict[str, Any] = {
        "run_id": "run_test",
        "manifest_fingerprint": "fp",
        "eligible": True,
        "vetoes": [],
        "scenarios": [],
        "rates": {"task_success_rate": Rate("task_success_rate", 1, 2)},
        "severity_counts": {"S0": 0, "S1": 0, "S2": 0, "S3": 0},
        "unexecuted": [],
        "oracle_error_scenarios": [],
    }
    base.update(overrides)
    return Scorecard(**base)


def test_the_report_says_so_when_latency_was_not_measured() -> None:
    report = _card().report()
    assert "model latency" in report
    assert "not measured" in report


def test_the_report_prints_the_latency_percentiles_when_present() -> None:
    report = _card(latency={"latency_p50_ms": 12.5, "latency_p95_ms": 40.0}).report()
    assert "p50: 12.5" in report
    assert "p95: 40.0" in report


def test_the_report_separates_rates_by_family() -> None:
    """plan.md 11.4: separate families so a paraphrase cannot be overweighted.

    The aggregate hides composition. A suite that is mostly paraphrase variants
    reports a task_success_rate that is mostly a statement about paraphrases, and
    section 12.6's disparity question needs the same substrate.
    """
    report = _card(
        rates_by_family={
            "core": {"task_success_rate": Rate("task_success_rate", 9, 10)},
            "metamorphic": {"task_success_rate": Rate("task_success_rate", 2, 13)},
        }
    ).report()
    assert "rates by family" in report
    assert "core: task_success_rate: 9/10 (90.0%)" in report
    assert "metamorphic: task_success_rate: 2/13 (15.4%)" in report


def test_the_report_lists_the_unmeasured_metrics_and_why() -> None:
    report = _card(unmeasured=dict(UNMEASURED_METRICS)).report()
    assert "not measured in this build" in report
    for name in UNMEASURED_METRICS:
        assert name in report
