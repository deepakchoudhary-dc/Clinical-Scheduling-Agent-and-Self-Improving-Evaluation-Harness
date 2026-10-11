"""Multi-seed stability: the input the acceptance gate cannot produce itself.

plan.md section 13.5 lists "multi-seed result is stable" as an acceptance
condition. ``evaluate_gate`` cannot check it -- stability is a property of a set
of runs -- so the gate records only the conditions a single run established, and
this module supplies the missing one.

The property that matters is the distinction between an improvement and a
coincidence. A sweep of one seed is therefore reported as **not established**,
never as stable: a check that answers "stable" for a sample of one is the same
class of error as an oracle that cannot return False.
"""

from __future__ import annotations

from typing import Any

from clinical_scheduler.eval.multi_seed import (
    MultiSeedSummary,
    SeedOutcome,
    summarise_seeds,
)
from clinical_scheduler.eval.scoring import Rate, Scorecard


def _card(*, eligible: bool, s0: int = 0, passed: int = 1, ran: int = 2) -> Scorecard:
    return Scorecard(
        run_id="run_test",
        manifest_fingerprint="fp",
        eligible=eligible,
        vetoes=[],
        scenarios=[],
        rates={"task_success_rate": Rate("task_success_rate", passed, ran)},
        severity_counts={"S0": s0, "S1": 0, "S2": 0, "S3": 0},
        unexecuted=[],
        oracle_error_scenarios=[],
    )


def test_a_single_seed_is_not_evidence_of_stability() -> None:
    summary = summarise_seeds([(0, _card(eligible=True))])
    assert summary.stable is False
    assert summary.is_a_sweep is False
    assert "not established" in summary.report()


def test_identical_outcomes_across_seeds_are_stable() -> None:
    summary = summarise_seeds([(seed, _card(eligible=True)) for seed in (0, 1, 2)])
    assert summary.stable is True
    assert "STABILITY: stable" in summary.report()


def test_a_verdict_that_moves_with_the_seed_is_not_stable() -> None:
    summary = summarise_seeds(
        [(0, _card(eligible=True)), (1, _card(eligible=False, s0=1)), (2, _card(eligible=True))]
    )
    assert summary.stable is False
    report = summary.report()
    assert "NOT stable" in report
    assert "not evidence of an improvement" in report
    assert "seed    1" in report  # the differing seed is named


def test_the_same_verdict_but_different_severity_counts_is_not_stable() -> None:
    """The gate compares severity counts, so equal eligibility is not enough."""
    summary = summarise_seeds(
        [(0, _card(eligible=True, s0=0)), (1, _card(eligible=True, s0=1))]
    )
    assert summary.stable is False


def test_a_rate_with_no_denominator_is_reported_as_not_measured() -> None:
    outcome = SeedOutcome(seed=0, eligible=True, severity_counts={}, task_success=(0, 0))
    assert outcome.task_success_display == "n/a (0/0, not measured)"


def test_the_report_names_every_seed() -> None:
    summary = summarise_seeds([(seed, _card(eligible=True)) for seed in (3, 4)])
    report = summary.report()
    assert "seed    3" in report and "seed    4" in report
    assert "seeds: [3, 4]" in report


def test_summary_carries_the_severity_counts_through() -> None:
    summary: MultiSeedSummary = summarise_seeds([(0, _card(eligible=True, s0=2)), (1, _card(eligible=True, s0=2))])
    assert summary.outcomes[0].severity_counts["S0"] == 2
    assert summary.stable is True
