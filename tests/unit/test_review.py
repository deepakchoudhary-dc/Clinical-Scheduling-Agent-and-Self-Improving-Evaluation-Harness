"""Layer 5: the sampling frame, the packet round-trip, and the arithmetic.

The kappa tests use values computed by hand, because a statistic that is only
tested against itself is not tested. The negative controls matter here: with no
reviewers the module must say "not measured", and with a single reviewer it must
refuse to report inter-rater agreement rather than reporting 1.0. A fabricated
agreement number would be the most damaging thing this repository could contain,
since every other number would then rest on it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from clinical_scheduler.eval.review import (
    ReviewItem,
    Verdict,
    agreement_report,
    cohens_kappa,
    export_packet,
    fleiss_kappa,
    ingest_verdicts,
    load_packet,
    sampling_frame,
)


def _row(
    scenario_id: str,
    *,
    passed: bool = True,
    severities: tuple[str, ...] = (),
    route: str = "CLEAR",
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "passed": passed,
        "severity": "S2",
        "findings": [
            {"oracle_id": f"L3.F{i}", "severity": s, "message": "m"} for i, s in enumerate(severities)
        ],
        "turns": [{"patient_text": f"patient says {scenario_id}"}],
        "assistant_messages": [f"reply for {scenario_id}"],
        "screenings": [{"verdict": route}],
    }


# -- Cohen's kappa ------------------------------------------------------------


def test_cohens_kappa_perfect_agreement() -> None:
    pairs = [(True, True), (False, False), (True, True)]
    kappa, observed, expected = cohens_kappa(pairs)
    assert observed == 1.0
    assert kappa == pytest.approx(1.0)


def test_cohens_kappa_complete_disagreement() -> None:
    pairs = [(True, False), (False, True)]
    kappa, observed, expected = cohens_kappa(pairs)
    assert observed == 0.0
    assert expected == pytest.approx(0.5)
    assert kappa == pytest.approx(-1.0)


def test_cohens_kappa_against_a_hand_computed_value() -> None:
    """A: 4 true / 6 false, B: 3 true / 7 false, 9 of 10 agreeing.

    observed = 0.9, expected = 0.4*0.3 + 0.6*0.7 = 0.54,
    kappa = (0.9 - 0.54) / (1 - 0.54) = 0.782608...
    """
    a = [True, True, True, True, False, False, False, False, False, False]
    b = [True, True, True, False, False, False, False, False, False, False]
    kappa, observed, expected = cohens_kappa(list(zip(a, b)))
    assert observed == pytest.approx(0.9)
    assert expected == pytest.approx(0.54)
    assert kappa == pytest.approx(0.36 / 0.46)


def test_cohens_kappa_is_undefined_without_variation() -> None:
    """Both raters used one category: chance agreement is 1, kappa is undefined.

    Reporting 0.0 here would read as "no agreement" when the truth is that there
    was nothing to measure agreement against.
    """
    kappa, observed, expected = cohens_kappa([(True, True), (True, True)])
    assert kappa is None
    assert observed == 1.0
    assert expected == 1.0


# -- Fleiss' kappa ------------------------------------------------------------


def test_fleiss_kappa_perfect_agreement_with_variation() -> None:
    # 3 raters; item 1 unanimous category 0, item 2 unanimous category 1.
    kappa, observed, expected = fleiss_kappa([[3, 0], [0, 3]])
    assert observed == pytest.approx(1.0)
    assert expected == pytest.approx(0.5)
    assert kappa == pytest.approx(1.0)


def test_fleiss_kappa_against_a_hand_computed_value() -> None:
    """Two items, three raters, a 2-1 split on each: observed 1/3, expected 5/9.

    kappa = (1/3 - 5/9) / (1 - 5/9) = -0.5
    """
    kappa, observed, expected = fleiss_kappa([[1, 2], [1, 2]])
    assert observed == pytest.approx(1 / 3)
    assert expected == pytest.approx(5 / 9)
    assert kappa == pytest.approx(-0.5)


def test_fleiss_kappa_needs_two_raters_per_item() -> None:
    assert fleiss_kappa([[1, 0], [0, 1]]) == (None, 0.0, 0.0)


# -- sampling frame -----------------------------------------------------------


def test_every_s0_and_s1_finding_is_sampled_even_at_zero_fraction() -> None:
    """The whole point of stratifying: a rare catastrophe must not be skipped."""
    rows = [
        _row("S01"),
        _row("ADV-11", passed=False, severities=("S0",)),
        _row("ADV-12", passed=False, severities=("S1",)),
    ]
    items = sampling_frame(rows, fraction=0.0, seed=0)
    assert {item.scenario_id for item in items} == {"ADV-11", "ADV-12"}


def test_the_random_fraction_is_deterministic_for_a_seed() -> None:
    rows = [_row(f"S{i:02d}") for i in range(40)]
    first = sampling_frame(rows, fraction=0.25, seed=7)
    second = sampling_frame(rows, fraction=0.25, seed=7)
    assert [i.item_id for i in first] == [i.item_id for i in second]
    assert 0 < len(first) < len(rows), "the sample should be a proper subset"


def test_a_fraction_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError):
        sampling_frame([_row("S01")], fraction=1.5)


# -- packet round-trip --------------------------------------------------------


def test_export_then_load_round_trips(tmp_path: Path) -> None:
    items = sampling_frame([_row("ADV-11", passed=False, severities=("S0",))], fraction=0.0)
    directory = export_packet(items, tmp_path / "packet")
    assert (directory / "packet.json").is_file()
    assert (directory / "reviewer_packet.json").is_file()
    assert (directory / "verdicts_template.csv").is_file()
    assert load_packet(directory) == items


def test_the_reviewer_packet_is_blinded(tmp_path: Path) -> None:
    """The file a reviewer receives must not reveal the oracle's verdict.

    A reviewer who can see that the oracle failed will tend to agree that it
    failed, and the agreement statistic would then measure deference.
    """
    items = sampling_frame([_row("ADV-11", passed=False, severities=("S0",))], fraction=0.0)
    directory = export_packet(items, tmp_path / "packet")

    blinded = json.loads((directory / "reviewer_packet.json").read_text(encoding="utf-8"))
    assert len(blinded) == len(items)
    for entry in blinded:
        assert set(entry) == {"item_id", "scenario_id", "patient_text", "reply"}
        assert "oracle_passed" not in entry
        assert "oracle_findings" not in entry
        assert "severity" not in entry

    # The analyst packet keeps them, which is what makes the split meaningful.
    full = json.loads((directory / "packet.json").read_text(encoding="utf-8"))
    assert full[0]["oracle_passed"] is False
    assert full[0]["severity"] == "S0"


def test_the_template_has_one_blank_row_per_item(tmp_path: Path) -> None:
    items = sampling_frame([_row("S01"), _row("S02")], fraction=1.0)
    directory = export_packet(items, tmp_path / "packet")
    with (directory / "verdicts_template.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(items)
    assert all(row["reviewer"] == "" for row in rows)


def test_ingest_skips_unfilled_rows_rather_than_counting_them(tmp_path: Path) -> None:
    """A blank row means "not reviewed". Counting it would manufacture agreement."""
    path = tmp_path / "verdicts.csv"
    path.write_text(
        "item_id,reviewer,agrees_with_runtime,comment\n"
        "S01,dr_a,true,\n"
        "S02,,,\n"
        "S03,dr_a,yes,looks fine\n",
        encoding="utf-8",
    )
    verdicts = ingest_verdicts(path)
    assert [v.item_id for v in verdicts] == ["S01", "S03"]
    assert all(v.agrees_with_runtime for v in verdicts)


def test_ingest_reads_json_too(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.json"
    path.write_text(
        json.dumps([{"item_id": "S01", "reviewer": "dr_a", "agrees_with_runtime": False}]),
        encoding="utf-8",
    )
    assert ingest_verdicts(path)[0].agrees_with_runtime is False


def test_an_unreadable_verdict_is_an_error_not_a_default(tmp_path: Path) -> None:
    """A verdict nobody can read must fail loudly, not be coerced to False."""
    path = tmp_path / "bad.csv"
    path.write_text(
        "item_id,reviewer,agrees_with_runtime,comment\nS01,dr_a,maybe,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        ingest_verdicts(path)


# -- the report ---------------------------------------------------------------


def _items() -> list[ReviewItem]:
    return sampling_frame(
        [_row("S01"), _row("S02"), _row("ADV-11", passed=False, severities=("S0",))],
        fraction=1.0,
    )


def test_with_no_verdicts_the_report_says_the_review_has_not_happened() -> None:
    report = agreement_report(_items(), [])
    assert report.kappa is None
    assert "no verdicts ingested" in report.report()
    assert "not been performed" in report.report()


def test_with_one_reviewer_inter_rater_agreement_is_refused() -> None:
    """One reviewer cannot be calibrated against anyone; 1.0 would be a lie."""
    verdicts = [Verdict("S01", "dr_a", True)]
    report = agreement_report(_items(), verdicts)
    assert report.kappa is None
    assert "at least two reviewers" in report.report()


def test_two_reviewers_produce_a_kappa_and_a_disagreement_list() -> None:
    verdicts = [
        Verdict("S01", "dr_a", True),
        Verdict("S01", "dr_b", True),
        Verdict("S02", "dr_a", True),
        Verdict("S02", "dr_b", False),
        Verdict("ADV-11", "dr_a", False),
        Verdict("ADV-11", "dr_b", False),
    ]
    report = agreement_report(_items(), verdicts)
    assert report.statistic == "cohen"
    assert report.kappa is not None
    assert report.disagreements == ("S02",)
    text = report.report()
    assert "cohen's kappa" in text
    assert "adjudicated by a third reviewer" in text


def test_agreement_with_the_runtime_is_reported_per_reviewer() -> None:
    """The oracle says S01 passed; dr_a agreed, dr_b did not."""
    verdicts = [Verdict("S01", "dr_a", True), Verdict("S01", "dr_b", False)]
    report = agreement_report(_items(), verdicts)
    assert report.reviewer_agreement_with_runtime["dr_a"] == 1.0
    assert report.reviewer_agreement_with_runtime["dr_b"] == 0.0
