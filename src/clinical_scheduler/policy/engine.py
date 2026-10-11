"""Policy engine: load, verify and evaluate policy bundles.

Two responsibilities, kept separate on purpose:

* **Load and verify.** A bundle is parsed into the typed schema, hashed, and
  registered in ``policy_versions``. An invalid bundle is refused outright; it
  never becomes "the current policy" in a degraded form.
* **Evaluate.** Given normalised text and contextual facts, decide which rules
  fire, what the safety verdict is, which mutations are blocked, and which
  claims are forbidden.

The engine contains no interpretation of what is clinically true. It matches
patterns and applies declared actions. That is exactly the constraint the plan
demands: the system routes according to clinician-approved data rather than
generating clinical judgement.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml
from pydantic import ValidationError as PydanticValidationError

from clinical_scheduler.db.repository import Repository, canonical_json, sha256_hex
from clinical_scheduler.domain.errors import PolicyViolation
from clinical_scheduler.domain.states import (
    MUTATION_BLOCKING_VERDICTS,
    SafetyVerdict,
    UrgencyCategory,
)
from clinical_scheduler.policy.schema import (
    MessageTemplate,
    PolicyBundle,
    PolicyRule,
    ProhibitedClaim,
    RuleAction,
    SentinelPattern,
)

BUNDLE_DIR = Path(__file__).with_name("bundles")


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


#: Characters from other scripts that render as, or near enough as, a Latin
#: letter, folded to that letter. Applied after casefolding, so only lowercase
#: forms are needed.
#:
#: This is canonicalisation in the same sense as NFKC, not fuzzy matching: a
#: patient typing a Cyrillic "с" in "сhest pain" has written the same word in a
#: different character set, deliberately or through a keyboard layout. Leaving it
#: unmapped means an emergency is spelled out in full and routes CLEAR.
#:
#: The table is deliberately small and explicit rather than a general confusables
#: database. Each entry is a character a person could actually type in place of a
#: Latin letter; a general table would fold hundreds of pairs and its precision
#: cost could not be reasoned about.
_CONFUSABLE_FOLD = str.maketrans(
    {
        # Cyrillic
        "\u0430": "a", "\u0435": "e", "\u043a": "k", "\u043c": "m",
        "\u043d": "h", "\u043e": "o", "\u0440": "p", "\u0441": "c",
        "\u0442": "t", "\u0443": "y", "\u0445": "x", "\u0456": "i",
        "\u0458": "j", "\u0455": "s", "\u0432": "b",
        # Greek
        "\u03b1": "a", "\u03b5": "e", "\u03b9": "i", "\u03ba": "k",
        "\u03bd": "v", "\u03bf": "o", "\u03c1": "p", "\u03c4": "t",
        "\u03c5": "u", "\u03c7": "x",
    }
)

#: Hyphen-like characters treated as a word separator, so "chest-pain" and
#: "chest pain" are the same string. A hyphen between two words is a spelling
#: choice; a hyphen inside a word ("follow-up") becomes two tokens, which is
#: harmless for phrase matching and is the safer direction for recall.
_HYPHENS = "\u002d\u2010\u2011\u2012\u2013\u2014\u2015\u2212"


#: Substrings in a character's Unicode *name* that mark it as occupying space
#: without rendering. Category ``Cf`` catches most of them, but not all: Hangul
#: filler (U+3164) is a letter and braille blank (U+2800) is a symbol, and both
#: are invisible, are not whitespace, and therefore defeated a matcher that only
#: stripped ``Cf`` and treated whitespace as optional.
_INVISIBLE_NAME_TOKENS = ("FILLER", "BLANK", "INVISIBLE", "ZERO WIDTH")


def _is_invisible(ch: str) -> bool:
    """Whether a character takes up space without showing anything.

    Deliberately does **not** include the space characters: those are real
    separators and are handled by whitespace collapsing. The distinction matters,
    because deleting a separator joins words while deleting an invisible
    character restores them.
    """
    if unicodedata.category(ch) in {"Mn", "Cf"}:
        return True
    name = unicodedata.name(ch, "")
    return any(token in name for token in _INVISIBLE_NAME_TOKENS)


#: Script name prefixes that are not the Latin alphabet. A character whose
#: Unicode name starts with one of these is written in a script the pattern set
#: does not cover.
_NON_LATIN_SCRIPTS: tuple[str, ...] = (
    "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "ARMENIAN", "GEORGIAN",
    "HANGUL", "CJK", "IDEOGRAPH", "HIRAGANA", "KATAKANA", "BOPOMOFO",
    "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL",
    "TELUGU", "KANNADA", "MALAYALAM", "SINHALA", "THAI", "LAO", "TIBETAN",
    "MYANMAR", "KHMER", "ETHIOPIC", "CHEROKEE", "MONGOLIAN",
)


def _has_non_latin_script(text: str) -> bool:
    """Whether the text contains a letter written in a non-Latin script.

    Used to abstain rather than clear. The check is on *letters*, so punctuation
    and symbols borrowed from another block -- a degree sign, a curly quote, a
    currency symbol -- do not trigger it.
    """
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if any(name.startswith(script) for script in _NON_LATIN_SCRIPTS):
            return True
    return False


def normalise(text: str) -> str:
    """Normalise patient text for deterministic pattern matching.

    The transforms, in order, and why each is here:

    * **NFKD, then strip combining marks.** Folds accented letters to their base
      form, so "páin" and "pain" are the same string. Without it, a combining
      acute accent turns an emergency into an unrecognised word.
    * **Strip category-``Cf`` characters.** Zero-width space, joiner/non-joiner,
      word joiner, byte-order mark and the soft hyphen. NFKC does not touch them,
      so ``"chest<ZWSP> pain"`` normalised to a string that did not contain
      ``chest pain``.
    * **NFKC, then casefold.** Compatibility forms (full-width Latin, ligatures)
      and shouting.
    * **Confusable fold.** Latin-lookalike characters from other scripts.
    * **Remove apostrophes.** So "can't" and "cant" agree, which matters because
      the second is how most people type the first.
    * **Fold "can not" to "cannot".** The same word written with a space.
    * **Hyphens to spaces.** "chest-pain" is "chest pain".
    * **Collapse whitespace.**

    What this deliberately does NOT do: stem, spell-correct, or translate. A
    missed paraphrase is a *policy gap to be filled with clinician-approved data*,
    not something to paper over with fuzzy matching. The line is that everything
    above is an **equivalence** — the two strings are the same text — while
    stemming and spell-correction are *guesses*. ``"chestpain"`` is handled not
    here but by making inter-token whitespace optional in the compiled pattern,
    because that is a spelling of the same phrase rather than a different one.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    # Combining marks and invisible characters both vanish. Neither carries
    # meaning in the words being matched.
    stripped = "".join(ch for ch in decomposed if not _is_invisible(ch))
    folded = unicodedata.normalize("NFKC", stripped).casefold()
    folded = folded.translate(_CONFUSABLE_FOLD)
    for apostrophe in ("'", "\u2019", "\u2018", "\u02bc"):
        folded = folded.replace(apostrophe, "")
    folded = folded.replace("\u201c", '"').replace("\u201d", '"')
    folded = re.sub(rf"[{_HYPHENS}]", " ", folded)
    folded = re.sub(r"\bcan not\b", "cannot", folded)
    folded = re.sub(r"\s+", " ", folded)
    return folded.strip()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyDecision:
    """The engine's answer for one turn.

    ``verdict`` drives the FSM and the capability broker. ``fired_rule_ids`` and
    ``matched_phrases`` are audit evidence: they let a reviewer see exactly which
    approved pattern caused a route, rather than having to trust a boolean.
    """

    verdict: SafetyVerdict
    fired_rule_ids: tuple[str, ...]
    matched_phrases: tuple[str, ...]
    action: RuleAction | None
    blocked_mutation_tools: frozenset[str]
    prohibited_claims: frozenset[ProhibitedClaim]
    message_template: MessageTemplate | None
    urgency_category: UrgencyCategory | None
    policy_version: str
    rationale: str

    @property
    def blocks_mutation(self) -> bool:
        return self.verdict in MUTATION_BLOCKING_VERDICTS


#: Tools that change state. Used to compute the mutation block set.
MUTATION_TOOLS: frozenset[str] = frozenset(
    {
        "hold_slot",
        "commit_appointment",
        "reconcile_commit",
        "atomic_reschedule",
        "cancel_appointment",
        "join_waitlist",
    }
)


class PolicyEngine:
    def __init__(self, bundle: PolicyBundle, *, jurisdiction: str | None = None) -> None:
        self.bundle = bundle
        self.jurisdiction = jurisdiction or bundle.jurisdiction
        self._compiled: dict[str, list[re.Pattern[str]]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        for rule in self.bundle.rules:
            self._compiled[rule.rule_id] = [
                re.compile(self._phrase_to_regex(p.phrase)) for p in rule.trigger_patterns
            ]

    @staticmethod
    def _phrase_to_regex(phrase: str) -> re.Pattern[str]:
        """Whole-phrase regex with optional whitespace between tokens.

        Word boundaries prevent "pain" matching "paint" and stop the sentinel
        from firing on substrings. Phrases are escaped, so a pattern can never
        contain regex metacharacters that broaden the match beyond what the
        clinician approved.

        Whitespace between tokens is ``\\s*`` rather than a single space because
        the space itself is the thing an attacker removes. A zero-width character
        sitting where the space belongs is indistinguishable from one sitting
        inside a word once it has been stripped -- ``"chest<ZWSP>pain"`` and
        ``"che<ZWSP>st pain"`` both reduce to a run of letters -- so the pattern
        accepts either spelling rather than guessing which was meant. The same
        optionality makes ``"chestpain"`` match, which is how people type when
        they are in a hurry, and it is the safer direction: a false positive is
        measured by ``false_escalation_rate``, a false negative is an emergency
        that was booked.
        """
        normalised = normalise(phrase)
        tokens = [re.escape(token) for token in normalised.split()]
        if not tokens:
            return re.compile(r"(?!)")  # never matches; an empty phrase is refused upstream
        # The separator is bound to a name because a backslash may not appear
        # inside an f-string's expression part before Python 3.12 (PEP 701),
        # and 3.11 is this project's floor. Writing ``r'\s*'`` inline in the
        # expression was a SyntaxError there -- at import time, in the module
        # almost every test collects through -- so the constant carries it.
        separator = r"\s*"
        return re.compile(rf"(?<!\w){separator.join(tokens)}(?!\w)")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicyEngine:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise PolicyViolation(f"policy bundle {path} did not parse to a mapping")
        try:
            bundle = PolicyBundle.model_validate(raw)
        except PydanticValidationError as exc:
            raise PolicyViolation(
                f"policy bundle {path} failed schema validation",
                errors=exc.errors(include_url=False),
            ) from exc
        return cls(bundle)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyEngine:
        try:
            return cls(PolicyBundle.model_validate(data))
        except PydanticValidationError as exc:
            raise PolicyViolation(
                "policy bundle failed schema validation", errors=exc.errors(include_url=False)
            ) from exc

    def bundle_hash(self) -> str:
        return sha256_hex(canonical_json(self.bundle.fingerprint()))

    def register(self, repo: Repository, *, owner: str, approved_by: str | None) -> None:
        """Record this bundle version in ``policy_versions``.

        Idempotent: re-registering an identical version with an identical hash is
        a no-op, so re-running the harness does not create duplicate governance
        rows or trip the primary key.
        """
        digest = self.bundle_hash()
        existing = repo.query_one(
            "SELECT bundle_sha256 FROM policy_versions WHERE policy_version = ?",
            (self.bundle.policy_version,),
        )
        if existing is not None:
            if existing["bundle_sha256"] != digest:
                raise PolicyViolation(
                    "policy version already registered with a different hash; "
                    "bump policy_version instead of editing a released bundle",
                    policy_version=self.bundle.policy_version,
                )
            return
        repo.execute(
            """INSERT INTO policy_versions
               (policy_version, jurisdiction, intended_use, bundle_sha256, owner,
                effective_from, effective_until, approved_by, rollback_target)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                self.bundle.policy_version,
                self.bundle.jurisdiction,
                self.bundle.intended_use,
                digest,
                owner,
                self.bundle.effective_from.isoformat(),
                None,
                approved_by,
                self.bundle.rollback_target,
            ),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        text: str,
        *,
        on: date | None = None,
        service_id: str | None = None,
        message_history: Sequence[str] = (),
    ) -> PolicyDecision:
        """Screen one patient turn.

        The verdict resolution order matters and is deliberate:

        1. A positive match on an escalating rule wins immediately. High recall
           is the goal, so a true positive is never suppressed by a negation
           elsewhere in the same message.
        2. If no positive match fires but the text contains safety-relevant
           vocabulary that the bundle does not fully cover, the verdict is
           UNCERTAIN, which also blocks mutation. This is the "uncertainty
           route" from plan.md section 10.1.
        3. Otherwise the text is clear; the specific clear-verdict kind records
           *why* (negated, historical, third-person, meta) for the audit trail.
        """
        normalised = normalise(text)
        # Include recent history so that a symptom disclosed late in the
        # conversation is still screened (plan.md section 10.2, "multiple
        # symptoms disclosed late in the conversation").
        haystack = normalise(" ".join([*message_history, text]))

        fired: list[PolicyRule] = []
        matched_phrases: list[str] = []

        for rule in self.bundle.active_rules(on=on):
            if service_id and rule.applies_to_service_ids and service_id not in rule.applies_to_service_ids:
                continue
            if not rule.trigger_patterns:
                continue
            positives, cancels = self._match_rule(rule, haystack)
            if positives:
                # A positive wins even if a cancellation marker is also present.
                # Rationale: in "no chest pain, but I am short of breath" the
                # second clause is the safety-relevant one. Under-triage is the
                # costly error, so the false-positive risk is accepted and
                # measured by the false_escalation_rate metric.
                fired.append(rule)
                matched_phrases.extend(positives)
            elif cancels:
                matched_phrases.extend(cancels)

        if fired:
            # Most restrictive action wins when several rules fire.
            rule = self._most_restrictive(fired)
            template = (
                self.bundle.template_by_id(rule.message_template_id)
                if rule.message_template_id
                else None
            )
            category = self._category_of(rule, matched_phrases)
            return PolicyDecision(
                verdict=SafetyVerdict.TRIGGERED,
                fired_rule_ids=tuple(r.rule_id for r in fired),
                matched_phrases=tuple(matched_phrases),
                action=rule.action,
                blocked_mutation_tools=frozenset(MUTATION_TOOLS),
                prohibited_claims=frozenset(rule.prohibited_claims),
                message_template=template,
                urgency_category=category,
                policy_version=self.bundle.policy_version,
                rationale=rule.rationale,
            )

        uncertain_hits = self._uncertainty_hits(haystack)
        if uncertain_hits:
            return PolicyDecision(
                verdict=SafetyVerdict.UNCERTAIN,
                fired_rule_ids=(),
                matched_phrases=tuple(uncertain_hits),
                action=RuleAction.HUMAN_CLINICAL_REVIEW,
                blocked_mutation_tools=frozenset(MUTATION_TOOLS),
                prohibited_claims=frozenset(),
                message_template=None,
                urgency_category=UrgencyCategory.UNSPECIFIED_HIGH_RISK,
                policy_version=self.bundle.policy_version,
                rationale="safety-relevant vocabulary present without a matching approved rule",
            )

        return PolicyDecision(
            verdict=self._clear_kind(matched_phrases),
            fired_rule_ids=(),
            matched_phrases=tuple(matched_phrases),
            action=None,
            blocked_mutation_tools=frozenset(),
            prohibited_claims=frozenset(),
            message_template=None,
            urgency_category=None,
            policy_version=self.bundle.policy_version,
            rationale="no safety rule fired",
        )

    def _match_rule(
        self, rule: PolicyRule, haystack: str
    ) -> tuple[list[str], list[str]]:
        """Return (positive phrases, cancellation phrases) found for a rule.

        Precedence, and the reasoning behind it:

        A sentence like "I am not having trouble speaking" contains the positive
        phrase "trouble speaking" *and* the negation phrase "not having trouble
        speaking". The positive match is an artefact of substring presence; the
        patient's statement is a denial. If the positive won here, the fix for
        S04 would create false escalations on exactly the people describing that
        they are fine -- which is both a care-experience harm and a measurable
        cost (false_escalation_rate).

        So a cancellation marker for the *same rule* suppresses that rule's
        positives. It does not suppress other rules: "no chest pain but my arm
        feels heavy" must still fire SAFE-002, because the cancellation belongs
        to SAFE-001's concept, not SAFE-002's. Rule-scoped polarity is what makes
        that distinction structural rather than heuristic.

        The one case where a positive must still win over a co-occurring cancel
        is a *different symptom* in the same message, which is handled by the
        rules being separate: each rule evaluates independently.
        """
        positives: list[str] = []
        cancels: list[str] = []
        for pattern, spec in zip(self._compiled[rule.rule_id], rule.trigger_patterns):
            if not pattern.search(haystack):
                continue
            if spec.polarity == "POSITIVE":
                positives.append(spec.phrase)
            else:
                cancels.append(spec.phrase)

        if cancels:
            # Remove positives whose text is subsumed by a cancellation marker.
            # Containment is the right test: the negation phrase is authored to
            # contain the symptom it denies ("not having trouble speaking"
            # contains "trouble speaking").
            surviving = [
                p for p in positives if not any(p in c for c in cancels)
            ]
            # A cancellation marker that denies a symptom the rule would
            # otherwise route on therefore leaves no positive standing.
            return surviving, cancels
        return positives, cancels

    @staticmethod
    def _most_restrictive(rules: Sequence[PolicyRule]) -> PolicyRule:
        order = {
            RuleAction.CRISIS_HANDOFF: 0,
            RuleAction.URGENT_HANDOFF: 1,
            RuleAction.BLOCK_MUTATION: 2,
            RuleAction.HUMAN_CLINICAL_REVIEW: 3,
            RuleAction.SPECIALIST_HANDOFF: 4,
            RuleAction.REQUIRE_ACKNOWLEDGEMENT: 5,
            RuleAction.ALLOW: 6,
        }
        return sorted(rules, key=lambda r: order[r.action])[0]

    def _category_of(self, rule: PolicyRule, phrases: Sequence[str]) -> UrgencyCategory:
        for spec in rule.trigger_patterns:
            if spec.phrase in phrases:
                try:
                    return UrgencyCategory(spec.category)
                except ValueError:
                    return UrgencyCategory.UNSPECIFIED_HIGH_RISK
        return UrgencyCategory.UNSPECIFIED_HIGH_RISK

    def _uncertainty_hits(self, haystack: str) -> list[str]:
        """Safety-adjacent vocabulary that the bundle does not route on.

        This is the deliberate "high recall sentinel is not a triage
        instrument" concession from plan.md section 10.1. When the text is in
        this neighbourhood but no approved rule covers it, the system refuses to
        book and hands off rather than assuming the absence of a match means
        absence of risk.

        Denials are honoured here too. "I have no chest pain" must not become
        UNCERTAIN merely because the phrase "chest pain" appears: the patient has
        explicitly denied the symptom, and abstaining on a clear denial would
        make the agent useless for ordinary conversation. The check is
        deliberately conservative -- it only suppresses a term when an explicit
        negation immediately precedes it -- because an under-suppressed term
        costs a handoff while an over-suppressed term could cost a missed
        emergency.
        """
        hits: list[str] = []
        for term in self.UNCERTAINTY_VOCABULARY:
            # The term is normalised before matching. The haystack has already
            # been normalised, so a term left in its authored form can never
            # match: the Arabic entry "ألم في الصدر" decomposes under NFKD and
            # its combining mark is stripped from the text but not from the
            # pattern, which made every non-Latin entry dead code.
            normalised_term = normalise(term)
            if not normalised_term:
                continue
            # Word boundaries stop "pain" matching "paint". They are meaningless
            # in scripts written without spaces, and actively harmful: in "我胸痛"
            # the look-behind fails because the preceding ideograph is a word
            # character, so a bare "胸痛" matched but the same phrase inside a
            # sentence did not.
            ascii_only = normalised_term.isascii()
            escaped = re.escape(normalised_term)
            pattern = (
                re.compile(rf"(?<!\w){escaped}(?!\w)")
                if ascii_only
                else re.compile(escaped)
            )
            for match in pattern.finditer(haystack):
                if self._is_denied(haystack, match.start()):
                    continue
                hits.append(term)

        # A script the sentinel cannot read is a script it must not clear. This
        # is the bounded-language problem answered structurally rather than by
        # an ever-longer phrase list: if text survives normalisation still
        # written in a non-Latin script, the sentinel has no basis for calling
        # the turn routine, so it abstains and hands off.
        #
        # It is deliberately an over-approximation. A patient whose name is
        # written in another script will be routed to a human, which costs a
        # handoff; the alternative is booking an emergency described in a
        # language nobody screened for.
        if not hits and _has_non_latin_script(haystack):
            hits.append("unrecognised_script")
        return hits

    #: Negation cues that, appearing shortly before a symptom term, invert it.
    #: Kept short and literal on purpose: this is a high-precision list, not a
    #: natural-language inference.
    _NEGATION_CUES: tuple[str, ...] = (
        "no ", "not ", "never ", "without ", "denies ", "deny ", "no longer ",
        "n't ", "dont ", "don't ", "does not ", "do not ", "did not ",
        "have not ", "has not ", "had no ", "zero ",
    )

    #: Clause boundaries. A negation cue does not reach across one: in
    #: "no fever but I have a seizure", the "no" belongs to the fever.
    #:
    #: " and " and " or " are included even though they can join two denials
    #: ("no fever and no cough"). That case still works, because the look-back
    #: for the second term finds "no " *after* the boundary. Leaving them out was
    #: a live defect: "I have no fever and bleeding" cleared, so a denial about
    #: fever silenced an asserted bleed.
    _CLAUSE_BREAKS: tuple[str, ...] = (
        ",", ";", ":", ".", "?", "!", "\n",
        " but ", " and ", " or ", " so ", " then ", " however ", " although ",
        " though ", " yet ", " except ", " apart from ",
    )

    @classmethod
    def _is_denied(cls, haystack: str, position: int, *, window: int = 24) -> bool:
        """True if a negation cue immediately precedes ``position``.

        A short look-back window is correct here: English negation of a symptom
        is local ("no chest pain", "does not feel heavy"). A longer window would
        start suppressing genuine positives in long messages, which is the
        dangerous direction of error.

        The look-back also stops at a **clause boundary**. Without that, any
        negation cue within the window suppressed the term regardless of what it
        negated, so "no fever but I have a seizure right now" cleared -- the
        patient stated a seizure and the route ignored it because of a denial
        about something else entirely. A window alone is a distance; a clause is
        a scope, and scope is what the cue actually has.
        """
        prefix = haystack[max(0, position - window) : position]
        for boundary in cls._CLAUSE_BREAKS:
            cut = prefix.rfind(boundary)
            if cut != -1:
                prefix = prefix[cut + len(boundary) :]
        return any(cue in prefix for cue in cls._NEGATION_CUES)

    #: Vocabulary that raises UNCERTAIN rather than CLEAR when unmatched.
    #: These are symptom words that a scheduling agent must never treat as
    #: routine. Kept here, not in the bundle, because the bundle governs
    #: *routing* while this list governs *abstention*.
    UNCERTAINTY_VOCABULARY: tuple[str, ...] = (
        "chest pain", "chest tightness", "can't breathe", "cannot breathe",
        "shortness of breath", "severe pain", "bleeding", "blood",
        "unconscious", "fainted", "seizure", "numbness", "weakness",
        "confusion", "slurred speech", "vision loss", "suicidal", "harm myself",
        "overdose", "overdosed", "too many pills", "took too many",
        "swallowed", "allergic", "rash", "swelling", "contractions",
        "pregnant and bleeding", "severe headache", "worst headache",
        # Chest pain and breathing difficulty in the languages most spoken in
        # this bundle's demo jurisdiction, plus the highest-frequency European
        # phrasings. These are an **abstention** net, not translation:
        # recognising one routes the turn to UNCERTAIN, which blocks mutation and
        # hands it to a human. The sentinel does not understand the sentence and
        # does not claim to.
        #
        # Without them, a patient describing a heart attack in Spanish routed
        # CLEAR and could be booked -- the opposite of what MET-03 asserts, and
        # nothing caught it because that scenario declared no route expectation.
        #
        # This list is **bounded and knowingly incomplete**. It is the mitigation
        # for Latin-script languages, where a script check cannot help because
        # the letters are the same. The structural answer for everything else is
        # `_has_non_latin_script`: text still written in another script abstains.
        # Covering Latin-script languages completely would require a language
        # detector, which is a component with its own accuracy and fairness
        # questions and is deferred to the roadmap rather than approximated here.
        "dolor en el pecho", "dolor de pecho", "no puedo respirar",
        "me falta el aire", "me duele el pecho",
        "douleur thoracique", "douleur dans la poitrine",
        "je ne peux pas respirer",
        "dor no peito", "falta de ar",
        "schmerzen in der brust", "brustschmerzen", "atemnot",
        "dolore al petto", "dolore toracico", "mancanza di respiro",
        "bol w klatce", "dusznosc",
        # Written in the form `normalise` produces, not the authored form. "đ"
        # (U+0111) is a base letter rather than a combining mark, so NFKD leaves
        # it alone; the authored spelling would never have matched.
        "\u0111au nguc", "kho tho",
        "seene mein dard", "seene mein tez dard", "saans nahi", "dard ho raha",
        "\u80f8\u75db", "\u547c\u5438\u56f0\u96be", "\u65e0\u6cd5\u547c\u5438",
        "\u0623\u0644\u0645 \u0641\u064a \u0627\u0644\u0635\u062f\u0631",
        "\u0644\u0627 \u0623\u0633\u062a\u0637\u064a\u0639 \u0627\u0644\u062a\u0646\u0641\u0633",
    )

    def _polarities_of(self, matched: Sequence[str]) -> set[str]:
        """The polarity labels of the patterns whose phrase matched.

        ``_match_rule`` returns phrase *text*, because that is what belongs in
        the audit trail -- a reviewer wants to see the words the patient used,
        not the label the policy author attached. Recovering the label is
        therefore a lookup, exactly as ``_category_of`` does for the category.
        """
        wanted = set(matched)
        out: set[str] = set()
        for rule in self.bundle.rules:
            for spec in rule.trigger_patterns:
                if spec.phrase in wanted:
                    out.add(str(spec.polarity))
        return out

    def _clear_kind(self, matched: Sequence[str]) -> SafetyVerdict:
        """Classify *why* a turn is clear, so the audit records a reason rather
        than a bare pass. Order is fixed so the result is deterministic.

        The kind is read from the **polarity label** of each matched pattern,
        not from the text of the phrase it matched. An earlier implementation
        searched the phrase text for stems ("histor", "meta", "mother"), which
        could only ever work by accident: a THIRD_PERSON pattern whose text
        happens to contain "mother" was classified, while the HISTORICAL pattern
        "had a heart attack five years ago" was not, and the audit recorded a
        bare CLEAR for a turn the engine had in fact recognised as historical.
        A clear-verdict kind that is only sometimes right is worse than no kind
        at all, because it is read as evidence that the engine understood.
        """
        polarities = self._polarities_of(matched)
        if "NEGATION" in polarities:
            return SafetyVerdict.CLEAR_NEGATED
        if "HISTORICAL" in polarities:
            return SafetyVerdict.CLEAR_HISTORICAL
        if "THIRD_PERSON" in polarities:
            return SafetyVerdict.CLEAR_THIRD_PERSON
        if "META" in polarities:
            return SafetyVerdict.CLEAR_META
        return SafetyVerdict.CLEAR

def load_baseline() -> PolicyEngine:
    """The bundle the demo starts from.

    It intentionally omits one escalated pattern for a defined high-risk
    presentation. That omission is the deliberate, policy-declared failure the
    improvement loop is designed to find (plan.md section 13.6).
    """
    return PolicyEngine.from_yaml(BUNDLE_DIR / "demo_policy.yaml")


def load_hardened() -> PolicyEngine:
    """The bundle the loop produces after adding the missing pattern plus a
    counterfactual negative so the fix cannot over-trigger."""
    return PolicyEngine.from_yaml(BUNDLE_DIR / "demo_policy_hardened.yaml")


__all__ = [
    "BUNDLE_DIR",
    "MUTATION_TOOLS",
    "PolicyDecision",
    "PolicyEngine",
    "load_baseline",
    "load_hardened",
    "normalise",
]
