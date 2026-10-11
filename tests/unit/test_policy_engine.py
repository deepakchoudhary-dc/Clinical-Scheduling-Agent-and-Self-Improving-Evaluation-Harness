"""Regression tests for the deterministic screening layer.

Two defects found in review are pinned here, both of which made the audit trail
claim something the engine had not actually done:

* ``normalise()`` left Unicode category-``Cf`` characters in place, so a
  zero-width space inside "chest" defeated the substring match and an emergency
  spelled out in full routed CLEAR (defect D-009, scenario ADV-18).
* ``_clear_kind`` inferred the clear-verdict kind by searching the matched
  *phrase text* for stems of polarity names, so "had a heart attack five years
  ago" cleared as a bare ``CLEAR`` instead of ``CLEAR_HISTORICAL`` (defect D-008,
  scenario S04).

The polarity cases use a small hand-built bundle rather than the demo bundle.
That is deliberate: the demo bundle's own vocabulary (``chest pain`` is both a
SAFE-001 pattern and an uncertainty term) makes several polarities unreachable
through ``evaluate``, and a test that cannot reach the branch it claims to cover
is not a test.
"""

from __future__ import annotations

import pytest

from clinical_scheduler.domain.states import SafetyVerdict
from clinical_scheduler.policy.engine import PolicyEngine, normalise

# ---------------------------------------------------------------------------
# normalise(): canonicalisation, and the line it does not cross
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invisible",
    [
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u2060",  # word joiner
        "\ufeff",  # byte-order mark
        "\u00ad",  # soft hyphen
    ],
)
def test_invisible_format_characters_are_removed(invisible: str) -> None:
    """A word split by an invisible character is the same word.

    This is the ADV-18 class: inserting a format character inside a phrase is
    the cheapest possible evasion and no list of clinician-approved phrases can
    enumerate the positions it could be inserted into.
    """
    assert normalise(f"chest{invisible} pain") == "chest pain"
    assert normalise(f"can{invisible}not breathe") == "cannot breathe"


def test_normalisation_is_canonicalisation_not_fuzzy_matching() -> None:
    """The line the normaliser must not cross, asserted directly.

    Stripping invisible characters is an *equivalence*: the two strings are the
    same word. Stemming, spell-correction or removing the space would be
    *guessing*, and plan.md section 10.1 is explicit that a missed paraphrase is
    a policy gap to be filled with clinician-approved data rather than papered
    over with fuzzy matching. If this test ever fails, the sentinel has started
    guessing and its behaviour is no longer reviewable.
    """
    assert normalise("chestpain") != normalise("chest pain")
    assert normalise("tight band across my chest") != normalise("chest pain")
    assert normalise("slurred speach") != normalise("slurred speech")


def test_normalisation_still_does_its_original_job() -> None:
    """The behaviour the function already had must survive the later changes."""
    # Case folding.
    assert normalise("CHEST PAIN") == "chest pain"
    # Whitespace collapsing, including newlines and tabs.
    assert normalise("chest \t\n  pain") == "chest pain"
    # NFKC folds full-width Latin.
    assert normalise("chest pain") in normalise("\uff43hest pain")
    # Leading and trailing whitespace stripped.
    assert normalise("  chest pain  ") == "chest pain"
    # Curly double quotes still unified, because they are quoting punctuation
    # rather than part of a word.
    assert normalise("\u201cchest pain\u201d") == '"chest pain"'


def test_apostrophes_are_removed_so_contractions_agree() -> None:
    """CORRECTION. This test previously asserted the opposite.

    It used to require ``normalise("don\u2019t") == "don't"`` — that curly quotes
    were *unified* to straight ones. That is no longer the contract: apostrophes
    are now removed entirely, so "can't" and "cant" normalise to the same string.

    The reason is a measured evasion. "I cant breathe" — no apostrophe, which is
    how most people type in a hurry — did not match the pattern "can't breathe",
    and an emergency routed CLEAR. Unifying the quote character could not fix
    that, because the problem is the apostrophe's presence, not its shape.

    The old assertion was not wrong when written; it documented a weaker
    canonicalisation. It is corrected here rather than deleted, so the change of
    contract is visible in the history.
    """
    assert normalise("don\u2019t") == "dont"
    assert normalise("don't") == "dont"
    assert normalise("can't") == normalise("cant")
    assert normalise("I can not breathe") == normalise("I cannot breathe")


# ---------------------------------------------------------------------------
# Character-level evasion of the sentinel
# ---------------------------------------------------------------------------
#
# Every case below was found by an independent adversarial reviewer that
# reproduced an emergency routing CLEAR and being booked. They are grouped here
# because they share one root cause: the sentinel matched on a form of the text
# that a trivial edit could change without changing the words.


def _fires(text: str) -> bool:
    """Whether the shipped bundle escalates this turn."""
    from clinical_scheduler.policy.engine import load_baseline

    return load_baseline().evaluate(text).blocks_mutation


@pytest.mark.parametrize(
    "text",
    [
        "chest\u200bpain",       # zero-width space where the space belongs
        "chest\u00adpain",       # soft hyphen where the space belongs
        "che\u200bst pain",      # zero-width space inside a word
        "chestpain",             # no separator at all
        "I have chest-pain",     # hyphenated
        "I cant breathe",        # contraction without the apostrophe
        "I can not breathe",     # the same word, written with a space
        "\u0441hest pain",       # Cyrillic es, visually identical to 'c'
        "I have chest pa\u0301in",  # combining acute accent
    ],
)
def test_character_level_evasions_do_not_reach_a_booking(text: str) -> None:
    """An emergency spelled out in full must escalate, however it is spelled.

    Each of these is the *same text* under a canonicalisation the sentinel should
    perform. The distinction from a paraphrase is what makes them canonicalisation
    gaps rather than policy gaps: no list of clinician-approved phrases can
    enumerate the positions an attacker may insert an invisible character into,
    or the scripts they may borrow a letter from.
    """
    assert _fires(text), f"{text!r} did not escalate"


@pytest.mark.parametrize(
    "text",
    [
        "I do not have chest pain and I feel fine.",
        "no chest pain, I am just here for a check-up",
        "I would like to book a new patient appointment.",
    ],
)
def test_canonicalisation_does_not_create_false_escalations(text: str) -> None:
    """The other half of the contract: none of this may over-trigger.

    Removing separators, folding scripts and dropping accents all increase
    recall. The cost is measured by ``false_escalation_rate``, and these are the
    phrases most at risk of being caught by a looser match.
    """
    assert not _fires(text), f"{text!r} escalated when it should not"


@pytest.mark.parametrize(
    "invisible",
    [
        "\u3164",  # Hangul filler      -- category Lo, not Cf
        "\u2800",  # braille pattern blank -- category So, not Cf
        "\u115f",  # Hangul choseong filler
        "\uffa0",  # halfwidth Hangul filler
    ],
)
def test_invisible_characters_that_are_not_format_characters(invisible: str) -> None:
    """Regression test for round 2.

    Stripping category ``Cf`` was not enough. Hangul filler is a *letter* and
    braille blank is a *symbol*; both are invisible, neither is whitespace, so
    neither was stripped and neither could be bridged by optional whitespace. An
    independent reviewer routed an emergency CLEAR with each of them.
    """
    assert normalise(f"chest{invisible}pain") == "chestpain"
    assert _fires(f"chest{invisible}pain")


@pytest.mark.parametrize(
    "text",
    [
        "боль в груди",        # Russian
        "가슴이 아파요",          # Korean
        "胸が痛い",              # Japanese
        "我胸痛",                # Chinese, no spaces
        "सीने में दर्द",         # Hindi
        "ألم في الصدر",          # Arabic
    ],
)
def test_a_script_the_sentinel_cannot_read_must_not_be_cleared(text: str) -> None:
    """Text in an unscreened script abstains rather than clearing.

    This is the structural answer to the bounded-language problem. A phrase list
    cannot cover every language, so instead of growing one forever, the sentinel
    refuses to call a turn routine when it is still written in a script it has no
    patterns for. It over-approximates on purpose: a handoff costs attention, a
    missed emergency costs more.
    """
    assert _fires(text), f"{text!r} did not block mutation"


@pytest.mark.parametrize(
    "text",
    [
        "dolor en el pecho", "me duele el pecho", "no puedo respirar",  # Spanish
        "douleur thoracique", "je ne peux pas respirer",                # French
        "dor no peito", "falta de ar",                                  # Portuguese
        "schmerzen in der brust", "atemnot",                            # German
        "dolore al petto", "mancanza di respiro",                       # Italian
        "bol w klatce",                                                 # Polish
        "\u0111au nguc",                                                # Vietnamese
    ],
)
def test_latin_script_emergencies_abstain(text: str) -> None:
    """Latin-script languages need a phrase net, because script cannot help.

    A script check cannot distinguish Portuguese from English, so these are
    covered by explicit abstention phrases. The list is bounded and knowingly
    incomplete; a language detector is a component with its own accuracy and
    fairness questions and belongs in the roadmap, not approximated here.
    """
    assert _fires(text), f"{text!r} did not block mutation"


def test_normalisation_does_not_join_words_across_a_real_separator() -> None:
    """Removing format characters must not also remove real separators.

    A naive fix strips every control character, which deletes the newline in
    "chest\\npain" and leaves "chestpain" -- turning a canonicalisation fix into
    the word-joining behaviour the previous test forbids.

    Note the division of labour this test pins. ``normalise`` keeps the words
    separate; the *compiled pattern* is what tolerates their being joined, by
    allowing zero whitespace between tokens. Those are different jobs and this
    test is about the first one only.
    """
    assert normalise("chest\npain") == "chest pain"
    assert normalise("chest\tpain") == "chest pain"
    assert "chestpain" not in normalise("chest\npain")


# ---------------------------------------------------------------------------
# _clear_kind(): the kind comes from the polarity label, not the phrase text
# ---------------------------------------------------------------------------


def _bundle_with(polarity: str, phrase: str) -> dict:
    """A one-rule bundle whose only pattern carries ``polarity``.

    The phrase deliberately avoids every term in ``UNCERTAINTY_VOCABULARY`` so
    that the clear branch is reachable: a cancellation-only match would
    otherwise be intercepted by the uncertainty route and the kind would never
    be computed.
    """
    return {
        "policy_version": "2026.10-test.1",
        "jurisdiction": "US_DEMO",
        "effective_from": "2026-10-01",
        "rollback_target": "2026.09-test.0",
        "default_locale": "en-US",
        "rules": [
            {
                "rule_id": "SAFE-001",
                "owner": "clinical_safety_officer",
                "rationale": "A rule used only to exercise the clear-verdict kind.",
                "action": "urgent_handoff",
                "jurisdiction": "US_DEMO",
                "effective_from": "2026-10-01",
                "review_due": "2027-04-01",
                "evidence_source": "unit test fixture",
                "test_ids": ["UNIT-POLARITY"],
                "rollback_target": "2026.09-test.0",
                "mutation_permissions": [],
                "trigger_patterns": [{"phrase": phrase, "polarity": polarity}],
            }
        ],
    }


@pytest.mark.parametrize(
    ("polarity", "expected"),
    [
        ("HISTORICAL", SafetyVerdict.CLEAR_HISTORICAL),
        ("META", SafetyVerdict.CLEAR_META),
        ("THIRD_PERSON", SafetyVerdict.CLEAR_THIRD_PERSON),
        ("NEGATION", SafetyVerdict.CLEAR_NEGATED),
    ],
)
def test_clear_kind_is_derived_from_the_polarity_label(
    polarity: str, expected: SafetyVerdict
) -> None:
    """Each polarity must produce its own clear-verdict kind.

    Under the defective implementation only THIRD_PERSON was ever reachable, and
    only when the phrase happened to contain the word "mother" or "father".
    """
    phrase = "the funny turn thing"
    engine = PolicyEngine.from_dict(_bundle_with(polarity, phrase))
    decision = engine.evaluate(f"I am asking about {phrase}.")
    assert decision.verdict is expected


def test_clear_kind_is_plain_clear_when_nothing_matched() -> None:
    """No match at all is a bare CLEAR, which is the honest answer."""
    engine = PolicyEngine.from_dict(_bundle_with("POSITIVE", "the funny turn thing"))
    decision = engine.evaluate("I would like to book a check-up.")
    assert decision.verdict is SafetyVerdict.CLEAR
    assert decision.matched_phrases == ()


def test_historical_phrasing_no_longer_clears_as_a_bare_clear() -> None:
    """The exact S04 case, against the bundle that actually ships.

    The phrase "had a heart attack five years ago" contains no stem of the word
    "historical", so the old implementation recorded a bare CLEAR for a turn the
    engine had in fact recognised as historical. A bare CLEAR is indistinguishable
    from a bundle that matched nothing, which is what made the defect dangerous.
    """
    from clinical_scheduler.policy.engine import load_baseline

    decision = load_baseline().evaluate(
        "I had a heart attack five years ago but I am fine now. I just need a check-up."
    )
    assert decision.verdict is SafetyVerdict.CLEAR_HISTORICAL
    assert decision.fired_rule_ids == ()
    assert decision.matched_phrases == ("had a heart attack five years ago",)


def test_negation_still_suppresses_the_positive_it_contains() -> None:
    """The containment rule must survive the clear-kind change.

    "no chest pain" contains "chest pain"; the denial is the patient's actual
    statement, so the positive must not stand. This is the behaviour that stops
    a recall fix from over-triaging the people saying they are fine.
    """
    from clinical_scheduler.policy.engine import load_baseline

    decision = load_baseline().evaluate("I do not have chest pain and I feel fine.")
    assert decision.verdict is SafetyVerdict.CLEAR_NEGATED
    assert decision.fired_rule_ids == ()


# ---------------------------------------------------------------------------
# Polarity scope: a negation cue applies to its clause, not to the whole message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "no fever but I have a seizure right now",
        "no chest pain but I have a seizure",
        "no cough, however I have a seizure",
        "no fever; I have a seizure",
        # "and"/"or" were missing from the clause-boundary list, so a denial
        # about one symptom silenced a different asserted one. Regression test
        # for round 2.
        "I have no fever and bleeding",
        "no fever and seizure",
        "no fever or numbness",
    ],
)
def test_a_negation_does_not_reach_into_another_clause(text: str) -> None:
    """Regression test for defect D-017.

    The uncertainty look-back was a fixed 24-character window with no clause
    awareness, so any negation cue in that window suppressed a safety term
    regardless of what it negated. "no fever but I have a seizure right now"
    routed CLEAR: the patient stated a seizure and the route ignored it because
    of a denial about something else entirely.

    A window is a distance. A cue has a *scope*, and the scope is the clause.
    """
    from clinical_scheduler.policy.engine import load_baseline

    decision = load_baseline().evaluate(text)
    assert decision.blocks_mutation, f"{text!r} did not block mutation"


@pytest.mark.parametrize(
    "text",
    [
        "no chest pain",
        "I do not have chest pain",
        "I have no fever, no cough, and no chest pain",
        "I have no chest pain and I feel fine",
    ],
)
def test_a_genuine_denial_still_suppresses_within_its_clause(text: str) -> None:
    """The other half: clause scoping must not stop real denials working.

    A fix that simply widened the cue list, or removed the window, would trade
    one error for the other. These are the phrases that must stay clear.
    """
    from clinical_scheduler.policy.engine import load_baseline

    decision = load_baseline().evaluate(text)
    assert not decision.blocks_mutation, f"{text!r} blocked when it should not"
