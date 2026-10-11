"""Layer 5: human review of a sample (plan.md section 11.2, layer 5).

Layers 0-4 are deterministic oracles: they can say whether a run did what a
scenario required, and they cannot say whether the scenario required the right
thing. That is the gap this module exists for, and the safety case names the
consequence of leaving it open -- "the evaluator is uncalibrated".

**What is implemented here is the procedure and its arithmetic, not the review.**
This module samples items, exports a packet a human can work through, ingests
their verdicts, and computes the agreement statistics. It does not contain a
reviewer, and it will not invent one: with no verdicts ingested it reports
``not measured`` and says what is missing. A fabricated reviewer would be the
single most damaging thing this repository could contain, because every other
number would then rest on it.

Two questions are answered, and they are different:

* **Do humans agree with each other?** Cohen's kappa for two reviewers, Fleiss'
  kappa for more. This is the calibration question: if two clinicians cannot agree
  on whether a reply is acceptable, no oracle score is meaningful.
* **Do humans agree with the runtime?** The per-reviewer rate at which the
  human's verdict matched the oracle's. This is the accuracy question.

Kappa is reported with its observed and expected agreement, because a kappa of
0.4 from 90% observed agreement and from 60% observed agreement mean different
things, and the number alone hides which one you have.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Severities whose findings are always sampled, however rare. A missed emergency
#: is exactly the case a random sample is most likely to skip.
ALWAYS_SAMPLED_SEVERITIES: frozenset[str] = frozenset({"S0", "S1"})

#: Version of the Layer-5 review protocol. Recorded with every packet and every
#: report so a change in the rubric is attributable. ``plan.md`` section 19 requires
#: "judge version and disagreement are recorded", and **the judge here is human** --
#: a model-as-judge is deliberately deferred, because a non-deterministic judge
#: whose disagreement with the deterministic oracles is itself uncalibrated would
#: add a number nobody could interpret.
PROTOCOL_VERSION = "layer5-human-review/1"

#: The judge this harness uses, as recorded in every run manifest (``plan.md``
#: section 11.1 requires the judge version to be recorded). The judge is human.
JUDGE_VERSION = f"human:{PROTOCOL_VERSION}"

PACKET_FILE = "packet.json"
REVIEWER_PACKET_FILE = "reviewer_packet.json"
VERDICT_TEMPLATE_FILE = "verdicts_template.csv"


# ---------------------------------------------------------------------------
# Items and verdicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewItem:
    """One sampled (scenario, turn) for a human to judge."""

    item_id: str
    scenario_id: str
    severity: str
    oracle_passed: bool
    route: str
    patient_text: str
    reply: str
    oracle_findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlindedItem:
    """What a reviewer is allowed to see: the exchange, and nothing else.

    **Blinding is not decoration.** If a reviewer can see that the oracle marked a
    scenario failed, they will tend to agree that it failed, and the agreement
    statistic would then measure the reviewer's deference rather than their
    judgement. The oracle's verdict therefore lives in ``packet.json``, which is
    kept by whoever runs the analysis, and never in the file handed to a reviewer.
    """

    item_id: str
    scenario_id: str
    patient_text: str
    reply: str


@dataclass(frozen=True)
class Verdict:
    """One reviewer's judgement of one item."""

    item_id: str
    reviewer: str
    agrees_with_runtime: bool
    comment: str = ""


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sampling_frame(
    rows: Sequence[Mapping[str, Any]],
    *,
    fraction: float = 0.1,
    seed: int = 0,
) -> list[ReviewItem]:
    """Select items for review from the rows of a run's ``traces.jsonl``.

    The frame is stratified, and the stratification is the point: **every**
    scenario carrying an S0 or S1 finding is included, because a random sample of
    a rare catastrophic failure is mostly a sample of the failures that do not
    matter. The remainder of the frame is a seeded random fraction of everything
    else, which is what lets the review estimate the false-*pass* rate the oracles
    cannot see.

    ``fraction`` of 1.0 samples everything. The random draw is seeded so a packet
    is reproducible and a reviewer can be handed a different stratum of the same
    run later.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    rng = random.Random(seed)

    items: list[ReviewItem] = []
    for row in rows:
        scenario_id = str(row.get("scenario_id", ""))
        findings = list(row.get("findings") or [])
        worst = _worst_severity(findings)
        must_sample = worst in ALWAYS_SAMPLED_SEVERITIES
        if not must_sample and rng.random() >= fraction:
            continue
        items.append(_item_from_row(row, findings))
    return items


def _worst_severity(findings: Sequence[Mapping[str, Any]]) -> str:
    order = ["S0", "S1", "S2", "S3"]
    severities = [str(f.get("severity", "")) for f in findings]
    for severity in order:
        if severity in severities:
            return severity
    return "NONE"


def _item_from_row(row: Mapping[str, Any], findings: Sequence[Mapping[str, Any]]) -> ReviewItem:
    """Build one review item from a stored trace row.

    The last patient turn and the last assistant reply are what a reviewer judges:
    the earlier turns are context, and shipping the whole transcript would make the
    packet a data-handling problem for no gain in judgement.
    """
    turns = list(row.get("turns") or [])
    replies = list(row.get("assistant_messages") or [])
    patient_text = ""
    if turns:
        last = turns[-1]
        patient_text = str(last.get("patient_text") or last.get("message") or "")
    reply = replies[-1] if replies else ""
    scenario_id = str(row.get("scenario_id", ""))
    return ReviewItem(
        item_id=f"{scenario_id}",
        scenario_id=scenario_id,
        severity=_worst_severity(findings),
        oracle_passed=bool(row.get("passed")),
        route=str((row.get("screenings") or [{}])[-1].get("verdict", "")) if row.get("screenings") else "",
        patient_text=patient_text,
        reply=str(reply),
        oracle_findings=tuple(str(f.get("oracle_id", "")) for f in findings),
    )


# ---------------------------------------------------------------------------
# Packet export / load, verdict ingest
# ---------------------------------------------------------------------------


def blind(item: ReviewItem) -> BlindedItem:
    """Strip everything the oracle knows from an item."""
    return BlindedItem(
        item_id=item.item_id,
        scenario_id=item.scenario_id,
        patient_text=item.patient_text,
        reply=item.reply,
    )


def export_packet(items: Sequence[ReviewItem], out_dir: str | Path) -> Path:
    """Write the analyst packet, the blinded reviewer packet, and the template.

    Three files, and the split is the point:

    * ``packet.json`` -- the full items, *including* the oracle's verdict. Kept by
      whoever runs the analysis. **Not** handed to a reviewer.
    * ``reviewer_packet.json`` -- the blinded view: the exchange, with no oracle
      verdict and no finding list.
    * ``verdicts_template.csv`` -- one row per item, for the reviewer to fill in.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PACKET_FILE).write_text(
        json.dumps([asdict(item) for item in items], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (directory / REVIEWER_PACKET_FILE).write_text(
        json.dumps([asdict(blind(item)) for item in items], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (directory / VERDICT_TEMPLATE_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "reviewer", "agrees_with_runtime", "comment"])
        for item in items:
            writer.writerow([item.item_id, "", "", ""])
    return directory


def load_packet(packet_dir: str | Path) -> list[ReviewItem]:
    raw = json.loads((Path(packet_dir) / PACKET_FILE).read_text(encoding="utf-8"))
    return [
        ReviewItem(
            item_id=entry["item_id"],
            scenario_id=entry["scenario_id"],
            severity=entry["severity"],
            oracle_passed=entry["oracle_passed"],
            route=entry["route"],
            patient_text=entry["patient_text"],
            reply=entry["reply"],
            oracle_findings=tuple(entry.get("oracle_findings") or ()),
        )
        for entry in raw
    ]


def ingest_verdicts(path: str | Path) -> list[Verdict]:
    """Read verdicts from the CSV template or from a JSON list.

    A row with no reviewer or no verdict is skipped rather than guessed at: an
    unfilled template row means "not reviewed", and counting it as agreement would
    manufacture the number this module exists to measure honestly.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        raw = json.loads(text)
        return [
            Verdict(
                item_id=str(entry["item_id"]),
                reviewer=str(entry["reviewer"]),
                agrees_with_runtime=_as_bool(entry["agrees_with_runtime"]),
                comment=str(entry.get("comment", "")),
            )
            for entry in raw
        ]

    verdicts: list[Verdict] = []
    for row in csv.DictReader(text.splitlines()):
        item_id = (row.get("item_id") or "").strip()
        reviewer = (row.get("reviewer") or "").strip()
        raw_verdict = (row.get("agrees_with_runtime") or "").strip()
        if not item_id or not reviewer or not raw_verdict:
            continue
        verdicts.append(
            Verdict(
                item_id=item_id,
                reviewer=reviewer,
                agrees_with_runtime=_as_bool(raw_verdict),
                comment=(row.get("comment") or "").strip(),
            )
        )
    return verdicts


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "agree", "agrees"}:
        return True
    if text in {"false", "0", "no", "n", "disagree", "disagrees"}:
        return False
    raise ValueError(f"cannot read {value!r} as a boolean verdict")


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------


def cohens_kappa(pairs: Sequence[tuple[bool, bool]]) -> tuple[float | None, float, float]:
    """Cohen's kappa for two raters. Returns (kappa, observed, expected).

    ``kappa`` is ``None`` when chance agreement is 1.0 -- which happens when both
    raters put every item in one category. The coefficient is undefined there, and
    reporting 0.0 would read as "no agreement" when the truth is "no variation to
    measure agreement against".
    """
    if not pairs:
        return None, 0.0, 0.0
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    a_true = sum(1 for a, _ in pairs if a) / n
    b_true = sum(1 for _, b in pairs if b) / n
    expected = a_true * b_true + (1 - a_true) * (1 - b_true)
    if expected >= 1.0:
        return None, observed, expected
    return (observed - expected) / (1 - expected), observed, expected


def fleiss_kappa(item_counts: Sequence[Sequence[int]]) -> tuple[float | None, float, float]:
    """Fleiss' kappa. ``item_counts[i][j]`` is how many raters gave item i category j.

    ``kappa`` is ``None`` when every rater used a single category, for the same
    reason as above.
    """
    usable = [counts for counts in item_counts if sum(counts) >= 2]
    if not usable:
        return None, 0.0, 0.0

    per_item: list[float] = []
    totals = [0.0] * len(usable[0])
    for counts in usable:
        raters = sum(counts)
        per_item.append((sum(c * c for c in counts) - raters) / (raters * (raters - 1)))
        for index, count in enumerate(counts):
            totals[index] += count
    grand = sum(totals)
    proportions = [total / grand for total in totals]
    expected = sum(p * p for p in proportions)
    observed = sum(per_item) / len(per_item)
    if expected >= 1.0:
        return None, observed, expected
    return (observed - expected) / (1 - expected), observed, expected


@dataclass(frozen=True)
class AgreementReport:
    """The result of a Layer-5 review, or the reason there is not one."""

    n_items: int
    reviewers: tuple[str, ...]
    statistic: str  # "cohen" | "fleiss" | "none"
    kappa: float | None
    observed_agreement: float | None
    expected_agreement: float | None
    reviewer_agreement_with_runtime: dict[str, float]
    disagreements: tuple[str, ...]
    note: str = ""
    #: How many items the packet contained, so a report of "0 items" is
    #: distinguishable from a packet that was never built.
    #:
    #: It sits *after* every field without a default on purpose: a dataclass
    #: rejects a defaulted field that precedes a non-defaulted one, and the first
    #: version of this field was placed in the middle, which raised TypeError at
    #: import time and took the whole CLI down with it.
    packet_items: int = 0
    #: Who judged, and under which rubric. ``plan.md`` section 19 requires the judge
    #: version to be recorded; the judge is human, so the version is the protocol's.
    judge: str = "human"
    protocol_version: str = PROTOCOL_VERSION
    #: The rate at which the human judge disagreed with the runtime's own verdict.
    #: This is the calibration signal: high disagreement with high inter-rater
    #: agreement means the *evaluator* is measuring the wrong thing.
    judge_disagreement_rate: float | None = None

    def report(self) -> str:
        lines = [
            f"items in packet: {self.packet_items}",
            f"items with verdicts: {self.n_items}",
            f"reviewers: {list(self.reviewers) or 'none'}",
            f"judge: {self.judge} ({self.protocol_version})",
        ]
        if self.judge_disagreement_rate is None:
            lines.append("judge disagreement with the runtime: not measured (no verdicts)")
        else:
            lines.append(
                "judge disagreement with the runtime: "
                f"{self.judge_disagreement_rate:.1%}"
            )
        lines.append("")
        if self.note:
            lines.append(f"STATUS: {self.note}")
            lines.append("")
        if self.kappa is not None:
            lines.append(f"inter-rater agreement ({self.statistic}'s kappa)")
            lines.append(f"  kappa   : {self.kappa:.3f}")
            lines.append(f"  observed: {self.observed_agreement:.3f}")
            lines.append(f"  expected: {self.expected_agreement:.3f}")
            lines.append("")
        if self.reviewer_agreement_with_runtime:
            lines.append("agreement with the runtime (per reviewer)")
            for reviewer, rate in sorted(self.reviewer_agreement_with_runtime.items()):
                lines.append(f"  {reviewer}: {rate:.1%}")
            lines.append("")
        if self.disagreements:
            lines.append(f"items where reviewers disagreed ({len(self.disagreements)})")
            for item_id in self.disagreements:
                lines.append(f"  {item_id}")
            lines.append("")
            lines.append("a disagreement is adjudicated by a third reviewer before it")
            lines.append("counts either way; it is not resolved by averaging.")
        return "\n".join(lines)


def agreement_report(
    items: Sequence[ReviewItem], verdicts: Sequence[Verdict]
) -> AgreementReport:
    """Compute the Layer-5 statistics, or explain why they cannot be computed."""
    reviewers = sorted({v.reviewer for v in verdicts})
    by_item: dict[str, dict[str, bool]] = {}
    for verdict in verdicts:
        by_item.setdefault(verdict.item_id, {})[verdict.reviewer] = verdict.agrees_with_runtime

    # Agreement with the runtime: does the human's verdict match the oracle's?
    runtime_by_item = {item.item_id: item.oracle_passed for item in items}
    per_reviewer: dict[str, float] = {}
    for reviewer in reviewers:
        their = [v for v in verdicts if v.reviewer == reviewer]
        if not their:
            continue
        matched = sum(
            1
            for v in their
            if v.item_id in runtime_by_item
            and v.agrees_with_runtime == runtime_by_item[v.item_id]
        )
        per_reviewer[reviewer] = matched / len(their)

    # Judge disagreement: how often a human verdict differed from the runtime's own.
    judged = [v for v in verdicts if v.item_id in runtime_by_item]
    disagreement_rate = (
        sum(1 for v in judged if v.agrees_with_runtime != runtime_by_item[v.item_id])
        / len(judged)
        if judged
        else None
    )

    if len(reviewers) < 2:
        return AgreementReport(
            n_items=len({v.item_id for v in verdicts}),
            reviewers=tuple(reviewers),
            statistic="none",
            packet_items=len(items),
            judge_disagreement_rate=disagreement_rate,
            kappa=None,
            observed_agreement=None,
            expected_agreement=None,
            reviewer_agreement_with_runtime=per_reviewer,
            disagreements=(),
            note=(
                "inter-rater agreement needs at least two reviewers; "
                f"{len(reviewers)} present. A single reviewer cannot be calibrated "
                "against anyone, and reporting 1.0 would be a fabrication."
            )
            if reviewers
            else "no verdicts ingested: the review has not been performed. "
            "Export a packet with `csched review export`, have it completed by "
            "two independent reviewers, then `csched review ingest` the result.",
        )

    common = sorted(
        item_id for item_id, ratings in by_item.items() if len(ratings) == len(reviewers)
    )
    if len(reviewers) == 2:
        first, second = reviewers
        pairs = [(by_item[item_id][first], by_item[item_id][second]) for item_id in common]
        kappa, observed, expected = cohens_kappa(pairs)
        statistic = "cohen"
    else:
        counts = [
            [sum(1 for r in reviewers if by_item[item_id].get(r) is False),
             sum(1 for r in reviewers if by_item[item_id].get(r) is True)]
            for item_id in common
        ]
        kappa, observed, expected = fleiss_kappa(counts)
        statistic = "fleiss"

    disagreements = tuple(
        item_id
        for item_id in common
        if len({by_item[item_id][r] for r in reviewers}) > 1
    )

    note = ""
    if kappa is None:
        note = (
            "kappa is undefined here: every reviewer used a single category, so "
            "there is no variation to measure agreement against."
        )

    return AgreementReport(
        n_items=len(common),
        reviewers=tuple(reviewers),
        statistic=statistic,
        packet_items=len(items),
        judge_disagreement_rate=disagreement_rate,
        kappa=kappa,
        observed_agreement=observed,
        expected_agreement=expected,
        reviewer_agreement_with_runtime=per_reviewer,
        disagreements=disagreements,
        note=note,
    )


def load_run_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read a stored run's ``traces.jsonl`` into rows."""
    path = Path(run_dir) / "traces.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no traces.jsonl under {run_dir}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = [
    "ALWAYS_SAMPLED_SEVERITIES",
    "JUDGE_VERSION",
    "PROTOCOL_VERSION",
    "AgreementReport",
    "BlindedItem",
    "REVIEWER_PACKET_FILE",
    "ReviewItem",
    "Verdict",
    "agreement_report",
    "blind",
    "cohens_kappa",
    "export_packet",
    "fleiss_kappa",
    "ingest_verdicts",
    "load_packet",
    "load_run_rows",
    "sampling_frame",
]
