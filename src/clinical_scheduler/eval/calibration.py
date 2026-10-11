"""Sentinel calibration: sensitivity, specificity and predictive values.

The safety case calls its first claim "a recall claim bounded by the pattern set"
and names two scenarios that falsify it. That honesty is right, and honesty about
a gap is not a method for closing it: without sensitivity and specificity there is
no way to know how bad recall is, and no way to tell whether a pattern change
improved it or merely moved it.

**What the labels are, stated plainly.** The labels come from the scenarios'
declared ``expect_routes`` -- the author's intent -- not from clinician
adjudication. This is therefore a *consistency* measurement (does the sentinel do
what the scenarios say it should), and **not** clinical validation. Every report
prints that. Clinician adjudication is Layer 5's job (``eval/review.py``), and this
module will consume its verdicts when they exist.

**Why predictive values and not just sensitivity.** A screen with 95% specificity
applied where an emergency is 1-in-10,000 has a positive predictive value near
0.2%: almost every escalation it raises is a false alarm. Sensitivity alone would
call that a good screen. The prevalence table is what makes the operational
consequence visible, and it is the question a deployment committee actually asks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

#: Default prevalence grid for the predictive-value table. The first entry is the
#: order of magnitude for a rare emergency in a general-practice population; the
#: last is deliberately high so the curve is visible.
DEFAULT_PREVALENCES: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.05, 0.20)

#: Verdicts that mean "escalate". Everything else -- the whole ``CLEAR*`` family --
#: means the turn proceeded. Listed explicitly rather than inferred from a prefix
#: so that adding a new verdict cannot silently change what counts as an
#: escalation here.
ESCALATING_VERDICTS: frozenset[str] = frozenset({"TRIGGERED", "UNCERTAIN"})


def is_escalation(verdict: str) -> bool:
    """Whether a verdict routes the turn to a human."""
    return verdict.strip().upper() in ESCALATING_VERDICTS


def wilson_interval(
    successes: int, total: int, *, z: float = 1.96
) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion, or ``None`` for n = 0.

    Wilson rather than the normal approximation because the interesting rates here
    are extreme: sensitivity near 1.0 and false-escalation rates near 0. The
    normal approximation produces intervals that cross 0 and 1 in exactly that
    regime, which is where this project's numbers live.
    """
    if total <= 0:
        return None
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    half = (z / denominator) * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    return max(0.0, centre - half), min(1.0, centre + half)


def _ppv(sensitivity: float, specificity: float, prevalence: float) -> float | None:
    """PPV from its components. Increasing in both sensitivity and specificity."""
    denominator = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    if denominator == 0:
        return None
    return sensitivity * prevalence / denominator


def _npv(sensitivity: float, specificity: float, prevalence: float) -> float | None:
    """NPV from its components. Increasing in specificity, decreasing in sensitivity."""
    denominator = specificity * (1 - prevalence) + (1 - sensitivity) * prevalence
    if denominator == 0:
        return None
    return specificity * (1 - prevalence) / denominator


def _fmt_pct(value: float | None) -> str:
    """Format a proportion without implying precision the sample lacks.

    A flat ``.4%`` on a value of 0.006 reads as four significant figures from five
    observations. The precision here follows the magnitude instead.
    """
    if value is None:
        return "n/a"
    if value >= 0.9995:
        return "100.0%"
    if value >= 0.01:
        return f"{value:.1%}"
    if value >= 0.0001:
        return f"{value:.2%}"
    return f"{value:.4%}"


@dataclass(frozen=True)
class ConfusionCounts:
    """The 2x2 table for an escalation screen."""

    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_negative + self.true_negative + self.false_positive

    @property
    def sensitivity(self) -> float | None:
        """Of the turns that should escalate, the fraction that did."""
        positives = self.true_positive + self.false_negative
        return None if positives == 0 else self.true_positive / positives

    @property
    def specificity(self) -> float | None:
        """Of the turns that should not escalate, the fraction that did not."""
        negatives = self.true_negative + self.false_positive
        return None if negatives == 0 else self.true_negative / negatives

    def sensitivity_interval(self, *, z: float = 1.96) -> tuple[float, float] | None:
        return wilson_interval(self.true_positive, self.true_positive + self.false_negative, z=z)

    def specificity_interval(self, *, z: float = 1.96) -> tuple[float, float] | None:
        return wilson_interval(self.true_negative, self.true_negative + self.false_positive, z=z)

    def positive_predictive_value(self, prevalence: float) -> float | None:
        """P(escalation is real | the sentinel escalated), at a stated prevalence."""
        sensitivity, specificity = self.sensitivity, self.specificity
        if sensitivity is None or specificity is None:
            return None
        return _ppv(sensitivity, specificity, prevalence)

    def negative_predictive_value(self, prevalence: float) -> float | None:
        sensitivity, specificity = self.sensitivity, self.specificity
        if sensitivity is None or specificity is None:
            return None
        return _npv(sensitivity, specificity, prevalence)

    def positive_predictive_value_range(
        self, prevalence: float
    ) -> tuple[float | None, float | None]:
        """PPV at the corners of the sensitivity and specificity intervals.

        PPV increases in both, so the corners bound it. Reporting only the point
        estimate is how a screen with three negative observations comes to look
        perfect at every prevalence -- which is exactly what the first version of
        this report did, and what a reviewer would have (correctly) distrusted.
        """
        sensitivity, specificity = self.sensitivity, self.specificity
        if sensitivity is None or specificity is None:
            return None, None
        sens_low, sens_high = self.sensitivity_interval() or (sensitivity, sensitivity)
        spec_low, spec_high = self.specificity_interval() or (specificity, specificity)
        return _ppv(sens_low, spec_low, prevalence), _ppv(sens_high, spec_high, prevalence)

    def negative_predictive_value_range(
        self, prevalence: float
    ) -> tuple[float | None, float | None]:
        """NPV increases in specificity and decreases in sensitivity."""
        sensitivity, specificity = self.sensitivity, self.specificity
        if sensitivity is None or specificity is None:
            return None, None
        sens_low, sens_high = self.sensitivity_interval() or (sensitivity, sensitivity)
        spec_low, spec_high = self.specificity_interval() or (specificity, specificity)
        return _npv(sens_high, spec_low, prevalence), _npv(sens_low, spec_high, prevalence)


@dataclass(frozen=True)
class CalibrationReport:
    """A calibration result, with its provenance attached."""

    counts: ConfusionCounts
    label_source: str
    prevalences: tuple[float, ...] = DEFAULT_PREVALENCES
    note: str = ""

    def to_json(self) -> str:
        """Serialise so a later run can be compared against this one.

        The change-control rule needs this: a policy edit is only reviewable if the
        recall evidence from *before* the edit is still available to diff against.
        """
        return json.dumps(
            {
                "counts": asdict(self.counts),
                "label_source": self.label_source,
                "prevalences": list(self.prevalences),
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "CalibrationReport":
        raw = json.loads(text)
        counts = raw["counts"]
        return cls(
            counts=ConfusionCounts(
                true_positive=int(counts["true_positive"]),
                false_negative=int(counts["false_negative"]),
                true_negative=int(counts["true_negative"]),
                false_positive=int(counts["false_positive"]),
            ),
            label_source=str(raw.get("label_source", "unknown")),
            prevalences=tuple(float(p) for p in raw.get("prevalences", DEFAULT_PREVALENCES)),
        )

    def report(self) -> str:
        counts = self.counts
        lines = [
            f"observations: {counts.total}",
            f"labels: {self.label_source}",
            "",
            "confusion (expected escalate / actual escalate)",
            f"  true positive : {counts.true_positive}",
            f"  false negative: {counts.false_negative}   <- missed escalation",
            f"  true negative : {counts.true_negative}",
            f"  false positive: {counts.false_positive}   <- false alarm",
            "",
        ]

        for label, value, interval in (
            ("sensitivity", counts.sensitivity, counts.sensitivity_interval()),
            ("specificity", counts.specificity, counts.specificity_interval()),
        ):
            if value is None:
                lines.append(f"{label}: n/a (0/0, not measured)")
            elif interval is None:
                lines.append(f"{label}: {value:.1%}")
            else:
                lines.append(f"{label}: {value:.1%}  (95% CI {interval[0]:.1%}..{interval[1]:.1%})")
        lines.append("")

        if counts.sensitivity is None or counts.specificity is None:
            lines.append("predictive values: not measured (sensitivity or specificity is undefined)")
        else:
            lines.append("predictive values by prevalence, as a RANGE across the 95% intervals")
            lines.append("  prevalence       PPV range                NPV range")
            for prevalence in self.prevalences:
                ppv_low, ppv_high = counts.positive_predictive_value_range(prevalence)
                npv_low, npv_high = counts.negative_predictive_value_range(prevalence)
                lines.append(
                    f"  {prevalence:>9.4%}   "
                    f"{_fmt_pct(ppv_low):>7} .. {_fmt_pct(ppv_high):<7}  "
                    f"{_fmt_pct(npv_low):>7} .. {_fmt_pct(npv_high):<7}"
                )
            lines.append("")
            lines.append("The range, not a point estimate: PPV rises with both sensitivity and")
            lines.append("specificity, so the corners of their intervals bound it. A point")
            lines.append("estimate from a small sample reads as precision the data does not have.")
            lines.append("")

            smallest = min(
                counts.true_positive + counts.false_negative,
                counts.true_negative + counts.false_positive,
            )
            if smallest < 10:
                lines.append(f"CAVEAT: the smaller denominator is {smallest}. At that size the")
                lines.append("intervals are wide enough that this table says very little about a")
                lines.append("deployed screen. Read it as 'not yet measured', not as a range to")
                lines.append("plan against.")
                lines.append("")

            lines.append(
                "A low PPV is not a defect in the screen; it is arithmetic. Where the")
            lines.append(
                "condition is rare, most escalations are false alarms, and the cost of")
            lines.append(
                "that is alert fatigue -- which is why false_escalation_rate is reported")
            lines.append("separately and the acceptance gate vetoes a patch that raises it.")

        if self.note:
            lines.append("")
            lines.append(f"NOTE: {self.note}")
        return "\n".join(lines)


@dataclass(frozen=True)
class CalibrationDelta:
    """What a policy change did to the sentinel's recall, on the same corpus.

    This is the second half of the change-control rule (plan.md section 7.2, and
    the review triggers in ``docs/hazard-log.md``): a pattern edit is not
    reviewable on its own, only against the recall evidence from before it.

    The rule it enforces is narrow and deliberate: **a change that increases false
    negatives is a regression**, whatever it does to false positives. The failure
    it permits is a missed emergency, and no improvement in false alarms buys that
    back.

    ``scale_mismatch`` guards the comparison itself. The rule's arithmetic only
    means something when both tables count the same observations, but the equality
    of the totals is only a proxy for that: a corpus could in principle gain one
    labelled scenario and drop another and still total the same. Scale equality is
    therefore necessary, not sufficient -- which is why the repair loop holds the
    corpus fixed by construction and why this property exists as a loud, testable
    guard rather than as silence.
    """

    previous: ConfusionCounts
    current: ConfusionCounts

    @property
    def scale_mismatch(self) -> bool:
        """Whether the two tables cannot be compared cell for cell.

        A delta across different corpora is not evidence about the pattern
        change: three fewer false negatives can mean better recall, or three
        labelled escalations quietly leaving the corpus. The printed report can
        only warn about it; the gate treats it as a veto, because an unmeasurable
        comparison is not a clean one.
        """
        return self.previous.total != self.current.total

    @property
    def false_negative_delta(self) -> int:
        """Positive means more missed escalations than before."""
        return self.current.false_negative - self.previous.false_negative

    @property
    def false_positive_delta(self) -> int:
        """Positive means more false alarms than before."""
        return self.current.false_positive - self.previous.false_positive

    @property
    def sensitivity_delta(self) -> float | None:
        before, after = self.previous.sensitivity, self.current.sensitivity
        if before is None or after is None:
            return None
        return after - before

    @property
    def specificity_delta(self) -> float | None:
        before, after = self.previous.specificity, self.current.specificity
        if before is None or after is None:
            return None
        return after - before

    @property
    def regressed(self) -> bool:
        """A change that misses more escalations is a regression, full stop."""
        return self.false_negative_delta > 0

    def report(self) -> str:
        lines = ["recall delta against the previous calibration", ""]
        lines.append(
            f"  false negatives: {self.previous.false_negative} -> {self.current.false_negative}"
        )
        lines.append(
            f"  false positives: {self.previous.false_positive} -> {self.current.false_positive}"
        )
        for label, before, after in (
            ("sensitivity", self.previous.sensitivity, self.current.sensitivity),
            ("specificity", self.previous.specificity, self.current.specificity),
        ):
            if before is None or after is None:
                lines.append(f"  {label}: n/a")
            else:
                lines.append(f"  {label}: {before:.1%} -> {after:.1%}  ({after - before:+.1%})")
        lines.append("")
        if self.previous.total != self.current.total:
            lines.append(
                "WARNING: the two calibrations cover different numbers of observations "
                f"({self.previous.total} -> {self.current.total}). A delta across different "
                "corpora is not evidence about the pattern change."
            )
            lines.append("")
        if self.regressed:
            lines.append(
                f"REGRESSION: {self.false_negative_delta} more missed escalation(s) than before. "
                "A pattern change that loses recall is not acceptable on the strength of any "
                "improvement in false alarms."
            )
        else:
            lines.append(
                "No recall regression: the number of missed escalations did not increase."
            )
        return "\n".join(lines)


def delta_report(previous: CalibrationReport, current: CalibrationReport) -> CalibrationDelta:
    """Compare two calibrations of the same corpus."""
    return CalibrationDelta(previous=previous.counts, current=current.counts)


def calibrate(observations: Iterable[tuple[bool, bool]]) -> ConfusionCounts:
    """Build the confusion table from (expected_escalate, actual_escalate) pairs."""
    true_positive = false_negative = true_negative = false_positive = 0
    for expected, actual in observations:
        if expected and actual:
            true_positive += 1
        elif expected and not actual:
            false_negative += 1
        elif not expected and actual:
            false_positive += 1
        else:
            true_negative += 1
    return ConfusionCounts(true_positive, false_negative, true_negative, false_positive)


# ---------------------------------------------------------------------------
# Reading a stored run
# ---------------------------------------------------------------------------

#: What the labels are, in the words that go into every report.
SCENARIO_DECLARED_LABELS = (
    "scenario-declared expect_routes (author intent, not clinician adjudication) -- "
    "this is a consistency measurement, not clinical validation"
)


def expected_escalation(scenario: Any) -> bool | None:
    """Whether a scenario declares that a turn should escalate.

    ``None`` when the scenario declares no expected route, which is a scenario
    that does not make a claim about routing. Excluded rather than counted as a
    negative: a scenario with no expectation is not evidence of a true negative.
    """
    routes = list(getattr(scenario, "expect_routes", ()) or ())
    if not routes:
        return None
    return any(not str(route.verdict).startswith("CLEAR") for route in routes)


def observed_escalation(row: Mapping[str, Any]) -> bool | None:
    """Whether the runtime escalated, read from the stored screening records."""
    screenings = list(row.get("screenings") or [])
    verdicts = [str(entry.get("verdict", "")) for entry in screenings if entry.get("verdict")]
    if not verdicts:
        return None
    return any(is_escalation(verdict) for verdict in verdicts)


def observations_from_run(
    rows: Sequence[Mapping[str, Any]], scenarios: Mapping[str, Any]
) -> list[tuple[bool, bool]]:
    """Pair each scenario's declared expectation with what the run actually did."""
    pairs: list[tuple[bool, bool]] = []
    for row in rows:
        scenario = scenarios.get(str(row.get("scenario_id", "")))
        if scenario is None:
            continue
        expected = expected_escalation(scenario)
        actual = observed_escalation(row)
        if expected is None or actual is None:
            continue
        pairs.append((expected, actual))
    return pairs


def calibrate_from_outcomes(
    outcomes: Sequence[Any],
) -> ConfusionCounts:
    """Build the confusion table directly from scenario outcomes.

    Labels are the scenario's declared ``expect_routes`` (author intent); the
    observation is whether any screening in the session escalated. Pairs where
    either side is unknown are excluded, exactly as ``observations_from_run``
    excludes them: a scenario that declares nothing is not evidence of a true
    negative, and a scenario whose run produced no screening is a run that did
    not happen.

    Used by the change-control check inside the repair loop (B4): a patch that
    adds a trigger pattern must attach this table computed on the candidate
    bundle to its acceptance evidence, so the recall delta is measured rather
    than asserted.
    """
    pairs: list[tuple[bool, bool]] = []
    for outcome in outcomes:
        scenario = getattr(outcome, "scenario", None)
        record = getattr(outcome, "record", None)
        if scenario is None or record is None:
            continue
        expected = expected_escalation(scenario)
        screenings = list(getattr(record, "screenings", ()) or ())
        verdicts = [
            str(entry.get("verdict", ""))
            for entry in screenings
            if entry.get("verdict")
        ]
        if expected is None or not verdicts:
            continue
        pairs.append((expected, any(is_escalation(v) for v in verdicts)))
    return calibrate(pairs)


__all__ = [
    "DEFAULT_PREVALENCES",
    "ESCALATING_VERDICTS",
    "SCENARIO_DECLARED_LABELS",
    "CalibrationDelta",
    "CalibrationReport",
    "ConfusionCounts",
    "calibrate",
    "calibrate_from_outcomes",
    "delta_report",
    "expected_escalation",
    "is_escalation",
    "observations_from_run",
    "observed_escalation",
    "wilson_interval",
    "CalibrationDelta.scale_mismatch",
]
