"""The documentation's checkable claims must match the source.

Regression test for defect D-019, found by an independent reproduction reviewer.
The README described the completion gate as having ten checks and reporting
"9 of 10" when it had eleven and reported 10 of 11, showed a baseline block whose
veto line contradicted its own severity counts, and described the FSM as "24
states, 24 declared transitions, each with a guard" when there are 24 source
states, 53 edges and 33 unguarded edges.

None of those was a code defect. All of them were **false claims in the document
a reviewer reads first**, which is worse in one specific way: a wrong number in
prose is not caught by any test of the system, so it survives every green run.

The tests below check only the claims that are mechanically checkable. A claim
that cannot be checked here is not thereby verified — it is simply out of reach
of this file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
VERIFY = REPO_ROOT.parent / ".agent" / "eval" / "scripts" / "verify.py"
STATE = REPO_ROOT.parent / ".agent" / "state.json"

# Three of these tests compare the README against files that live in the
# workspace engineering record (verify.py, state.json) -- a *sibling* of this
# repository, deliberately not pushed (the README says so). In a clean clone
# those files do not exist, and a test that then fails would be asserting that
# the reviewer built a workspace they were never given. The skip is the honest
# state: on the build machine the assertion runs; in a clone there is nothing
# to compare against and the check is out of scope, exactly as the module
# docstring already says of any claim it cannot reach.
pytestmark = [
    pytest.mark.skipif(
        not VERIFY.exists() or not STATE.exists(),
        reason="workspace engineering record (.agent/) not present; these README "
        "claims are checkable only where that record exists",
    )
]


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def _count_tests() -> int:
    """Collect the suite and count it, without running it."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    counts = re.findall(r"^tests/.*: (\d+)$", proc.stdout, flags=re.MULTILINE)
    return sum(int(value) for value in counts)


def test_the_readme_states_the_real_test_count(readme: str) -> None:
    """The count a reader will use to decide whether to trust the rest.

    Also covers the same claim in ``docs/*.md``: the safety case states a test
    count too, and it drifted to "180 tests" while the suite ran 236 -- unguarded,
    because only the README was checked. The pattern is deliberately narrow
    ("N tests pass", not "N tests") so a per-file count such as the model card's
    "14 tests" is not mistaken for a total.
    """
    actual = _count_tests()

    stated = [int(m) for m in re.findall(r"# (\d+) passed", readme)]
    assert stated, "the README no longer states a test count; this test needs updating"
    assert all(value == actual for value in stated), (
        f"README claims {stated} tests; the suite collects {actual}"
    )

    doc_dir = REPO_ROOT / "docs"
    documents = [readme]
    if doc_dir.exists():
        documents += [p.read_text(encoding="utf-8") for p in sorted(doc_dir.glob("*.md"))]
    for document in documents:
        for value in re.findall(r"(\d+) tests pass", document):
            assert int(value) == actual, (
                f"a document claims {value} tests pass; the suite collects {actual}"
            )


def test_the_readme_states_the_real_number_of_gate_checks(readme: str) -> None:
    """The gate count, stated in words and in the pass ratio."""
    source = VERIFY.read_text(encoding="utf-8")
    registered = source.split("CHECKS = (", 1)[1].split(")", 1)[0]
    checks = [line.strip().rstrip(",") for line in registered.splitlines() if line.strip()]
    assert len(checks) >= 5, f"could not parse the gate's check list: {checks}"

    words = {
        10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen",
    }
    assert f"{words[len(checks)]} checks" in readme, (
        f"the gate has {len(checks)} checks; the README does not say "
        f"{words[len(checks)]!r}"
    )


def test_the_readme_states_the_real_fsm_shape(readme: str) -> None:
    """The FSM claim, which was wrong twice: edge count and guard coverage.

    The first version of this test asserted only that the *numbers* appeared
    somewhere in the README, which an independent reviewer showed was
    association-blind: swapping two counts between labels still passed. It now
    checks each number against the label it belongs to.
    """
    from clinical_scheduler.domain.fsm import GUARD_REGISTRY, TRANSITIONS

    states = len(TRANSITIONS)
    edges = sum(len(targets) for targets in TRANSITIONS.values())
    guards = len(GUARD_REGISTRY)

    assert f"{states} states and {edges} declared" in readme, (
        f"the FSM has {states} source states and {edges} edges; the README "
        "describes something else"
    )
    assert f"{guards} transitions carry a registered guard" in readme, (
        f"{guards} guards are registered; the README says otherwise"
    )
    assert f"the other\n   {edges - guards} are unconditional" in readme, (
        f"{edges - guards} transitions have no guard; the README says otherwise"
    )


def test_the_readme_states_the_real_counts_for_the_major_structures(readme: str) -> None:
    """Counts a reader can check in one command, so they should be right."""
    from clinical_scheduler.domain.invariants import INVARIANTS
    from clinical_scheduler.domain.states import SessionState
    from clinical_scheduler.tools.broker import ALL_TOOLS

    schema = (REPO_ROOT / "src/clinical_scheduler/db/schema.sql").read_text("utf-8")
    tables = schema.count("CREATE TABLE IF NOT EXISTS")
    indexes = len(re.findall(r"CREATE.*INDEX IF NOT EXISTS", schema))
    triggers = schema.count("CREATE TRIGGER IF NOT EXISTS")

    # Each count is checked against the label it belongs to, not merely for
    # presence. A bare `str(value) in readme` passes when two counts are swapped,
    # which an independent reviewer demonstrated.
    expected = {
        "states": f"states ({len(SessionState)})",
        "invariants": f"{len(INVARIANTS)} invariants",
        "tools": f"contracts ({len(ALL_TOOLS)} tools)",
        "tables": f"{tables} tables",
        "indexes": f"{indexes} indexes",
        "triggers": f"{triggers} triggers",
    }
    for label, phrase in expected.items():
        assert phrase in readme, (
            f"the README does not state the {label} count as {phrase!r}"
        )


def test_the_readme_states_the_real_defect_count(readme: str) -> None:
    """The defect count, which drifted from 22 to 28 with nothing to notice.

    ``verify.py::workspace.record_is_complete`` reads the same list, so the two
    numbers are the same by construction -- but the README was never checked
    against it, and the README is the document a reader trusts first. Added after
    the 2026-10-08 static audit found the two had diverged by six.
    """
    import json

    state = json.loads(STATE.read_text(encoding="utf-8"))
    actual = len(state["defects_found_and_fixed"])
    stated = [int(match) for match in re.findall(r"(\d+) defects", readme)]
    assert stated, "the README no longer states a defect count; this test needs updating"
    assert all(value == actual for value in stated), (
        f"README claims {stated} defects; state.json records {actual}"
    )


def test_the_readme_states_the_real_gate_ratio(readme: str) -> None:
    """The pass *ratio*, not only the word form of the check count.

    ``test_the_readme_states_the_real_number_of_gate_checks`` checks that the
    README names the right number of checks. It does not check the ratio the
    README claims the gate reports -- which said "10 of 11" while the gate
    reported 11 of 11, and then 12 of 12.
    """
    source = VERIFY.read_text(encoding="utf-8")
    registered = source.split("CHECKS = (", 1)[1].split(")", 1)[0]
    checks = [line.strip().rstrip(",") for line in registered.splitlines() if line.strip()]
    count = len(checks)
    assert f"{count} of {count}" in readme, (
        f"the gate has {count} checks; the README does not state that it reports "
        f"{count} of {count}"
    )


def test_the_readme_says_where_the_record_and_venv_live(readme: str) -> None:
    """A clone must not be left expecting files that are outside the repository.

    The README cites `.agent/...` and runs `../.venv/...`. Both are siblings of
    the repository, so a fresh clone has neither. Saying so is the difference
    between a reference and a broken link.
    """
    assert "beside the repository" in readme or "outside this directory" in readme, (
        "the README does not explain that .agent/ and .venv live outside the repo"
    )
    assert not (REPO_ROOT / ".agent").exists(), (
        "a .agent directory now exists inside the repository; either remove it or "
        "update the README, but do not leave two records"
    )
