"""Multi-seed runs, and the flakiness-versus-improvement distinction (plan.md 11.4).

The acceptance rule in plan.md section 13.5 lists "multi-seed result is stable"
as a condition. The gate itself cannot establish it: stability is a property of a
*set* of runs, and one run cannot show it. This module is what supplies it.

The distinction it exists to make is between an improvement and a coincidence. A
single run that goes from failing to passing is consistent with both. If the same
seed range reproduces the change every time, it is an improvement; if the outcome
moves around with the seed, the change was noise and the gate should not treat it
as evidence.

Stability is defined narrowly and stated in code: **the eligibility verdict and
the severity counts must be identical across every seed.** Two runs that are both
"eligible" but carry different severity counts are not stable -- the severity
counts are what the acceptance gate compares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from clinical_scheduler.eval.scoring import SEVERITY_ORDER, Scorecard


@dataclass(frozen=True)
class SeedOutcome:
    """One seed's result, reduced to the fields stability is defined over."""

    seed: int
    eligible: bool
    severity_counts: dict[str, int]
    task_success: tuple[int, int]  # (numerator, denominator)

    @property
    def task_success_display(self) -> str:
        numerator, denominator = self.task_success
        if denominator == 0:
            return "n/a (0/0, not measured)"
        return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


@dataclass(frozen=True)
class MultiSeedSummary:
    """The outcome of a seed sweep, with an explicit stability verdict."""

    outcomes: list[SeedOutcome]

    @property
    def stable(self) -> bool:
        """True when every seed produced the same verdict and severity counts.

        A single run is trivially stable, which is why a sweep of one seed is
        reported as *not evidence about stability* rather than as stable.
        """
        if len(self.outcomes) < 2:
            return False
        first = self.outcomes[0]
        return all(
            outcome.eligible == first.eligible
            and outcome.severity_counts == first.severity_counts
            for outcome in self.outcomes[1:]
        )

    @property
    def is_a_sweep(self) -> bool:
        return len(self.outcomes) >= 2

    def report(self) -> str:
        lines = [f"seeds: {[o.seed for o in self.outcomes]}", ""]
        lines.append("per-seed results")
        for outcome in self.outcomes:
            verdict = "PASS" if outcome.eligible else "FAIL"
            counts = " ".join(
                f"{severity}:{outcome.severity_counts.get(severity, 0)}"
                for severity in SEVERITY_ORDER
            )
            lines.append(
                f"  seed {outcome.seed:>4}  eligibility {verdict}  "
                f"task_success {outcome.task_success_display}  {counts}"
            )
        lines.append("")

        if not self.is_a_sweep:
            lines.append(
                "STABILITY: not established -- a sweep needs at least two seeds. "
                "One run cannot distinguish an improvement from a coincidence."
            )
        elif self.stable:
            lines.append(
                "STABILITY: stable -- every seed produced the same verdict and the "
                "same severity counts."
            )
        else:
            differing = sorted(
                {o.seed for o in self.outcomes if o.eligible != self.outcomes[0].eligible}
            )
            lines.append(
                "STABILITY: NOT stable -- the outcome moves with the seed "
                f"(eligibility differs at seed(s) {differing}). Any change measured "
                "on a single seed of this sweep is not evidence of an improvement."
            )
        return "\n".join(lines)


def summarise_seeds(cards: Sequence[tuple[int, Scorecard]]) -> MultiSeedSummary:
    """Reduce (seed, scorecard) pairs to a stability summary."""
    outcomes: list[SeedOutcome] = []
    for seed, card in cards:
        task_success = card.rates.get("task_success_rate")
        outcomes.append(
            SeedOutcome(
                seed=seed,
                eligible=card.eligible,
                severity_counts=dict(card.severity_counts),
                task_success=(
                    (task_success.numerator, task_success.denominator)
                    if task_success is not None
                    else (0, 0)
                ),
            )
        )
    return MultiSeedSummary(outcomes=outcomes)


__all__ = ["MultiSeedSummary", "SeedOutcome", "summarise_seeds"]
