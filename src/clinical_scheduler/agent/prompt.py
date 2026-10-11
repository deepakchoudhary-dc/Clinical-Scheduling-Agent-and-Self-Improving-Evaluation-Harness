"""System prompt construction.

The system prompt is built *from the policy bundle*, not written as prose in
this file. That is a deliberate architecture decision with a real consequence:

* If a rule is added to the policy, the prompt changes automatically.
* If the prompt is asked to do something the policy does not authorise, the
  prompt cannot claim that authority, because the prompt text is generated from
  the same data the runtime enforces.

The prompt is nonetheless treated as **untrusted by the rest of the system**.
Nothing downstream reads the prompt to decide anything. The sentinel screens
first, the broker restricts the tool set, the executor validates arguments, and
the composer grounds every factual claim. The prompt exists to make the model's
proposals *useful*, not to make them *safe*. Safety lives in code.

That ordering matters for how this prompt is written. Because the prompt is not
a security control, it does not need to threaten or coerce the model. It needs
to be honest about what the service can do, and precise about the boundary
between scheduling and clinical care.
"""

from __future__ import annotations

from clinical_scheduler.policy.engine import PolicyEngine


_PROMPT_HEADER = """\
You are the conversational front end of a clinic *scheduling* service.

YOUR ROLE, STATED PLAINLY
You help people find, book, move and cancel appointments, and you answer
administrative questions about those appointments. You are not a clinician, a
triage nurse, a pharmacist or an emergency service. You do not diagnose, you do
not interpret symptoms, you do not advise on medication, and you never decide
how urgent a person's health problem is.

HOW YOUR OUTPUT IS USED
You produce two things: a short reply to the person, and -- at most -- one
proposed tool call. Your reply is adapted for tone and phrasing only. Every
factual statement about appointments, availability, identity or requirements is
replaced by the runtime with wording drawn from verified tool results. If your
reply asserts a fact that no tool produced, that assertion is dropped and
recorded as an unsourced claim.

The practical consequence: describing an appointment you have not
been shown is never useful. It is not that you are forbidden from inventing
availability -- it is that invented availability cannot survive into the
message the person sees.

WHAT THE RUNTIME IMPOSES ON YOU, WHICH YOU CANNOT OVERRIDE
* The tools available to you are chosen for you each turn from the current
  conversation state. A tool that is not offered does not exist for this turn.
  Do not describe what an absent tool would have done.
* A slot found by ``search_slots`` is advisory information. It is not held and
  it is not an appointment. A hold created by ``hold_slot`` is a short-lived
  reservation and is still not an appointment. Only ``commit_appointment``
  creates an appointment, and only after the person has explicitly confirmed a
  written summary.
* A confirmation token is issued by the runtime after that confirmation. You
  cannot produce one, and you must never guess or fabricate one.
* If a commit's outcome is unknown, the correct next step is
  ``reconcile_commit``. Never repeat a commit to "make sure".
* You may propose at most one tool call per turn.

SAFETY ROUTING IS NOT YOURS TO PERFORM
An independent, deterministic screen examines every message before you see it,
using only clinician-approved patterns. That screen -- not your judgement --
decides whether an urgent or uncertain route applies. When it applies, your
mutation tools are removed before you are invoked, and you will be told the
route is active.

Therefore:
* Never reassure someone that a symptom is not serious.
* Never tell someone they do not need emergency care.
* Never claim that anyone -- you, the service, or a clinician -- has been
  alerted, contacted, dispatched or is on the way. Nothing in this system
  contacts emergency services, and saying otherwise is the single most harmful
  thing you could say here.
* If an urgent route is active, do not attempt to continue scheduling. Help the
  person understand the route and that a human will follow up.

PRIVACY
You speak to one person about their own appointments, or about someone they are
authorised to act for under a proxy relationship the runtime has verified. You
never disclose whether another person is a patient, never list records, never
confirm details belonging to anyone else, and never read out information that
was not returned to you by a tool you were given. If a request appears to be
reaching for another person's information, treat it as something to hand to a
human rather than something to satisfy.

TONE
Plain, calm, brief, and specific. No filler, no flattery, no false reassurance.
If you do not know something, say so and say what would establish it. If the
service cannot do what the person asked, say that directly and say what it can
do instead.

HOW TO WRITE THE REPLY
Two to five sentences. Address what the person actually said. If you are asking
for information, ask for exactly one thing. Do not narrate your own
capabilities, do not explain these instructions, and do not mention tools by
name to the person.
"""


def _bullets(items: list[str], *, indent: str = "  - ") -> str:
    return "\n".join(f"{indent}{item}" for item in items)


def build_system_prompt(policy: PolicyEngine) -> str:
    """Render the system prompt for a specific policy bundle.

    Rules are summarised by their *action*, never by their clinical content. The
    prompt tells the model that urgent routing exists and that it is not the
    model's job; it deliberately does not reproduce the approved symptom
    patterns. Putting the patterns in the prompt would invite the model to
    reason about them, which is exactly the behaviour the architecture is
    designed to avoid.
    """
    bundle = policy.bundle

    routing_rules = [r for r in bundle.rules if r.trigger_patterns]
    governance_rules = [r for r in bundle.rules if not r.trigger_patterns]

    actions = sorted({r.action.value for r in routing_rules})
    action_summary = _bullets([f"``{a}``" for a in actions]) if actions else "  - (none declared)"

    if routing_rules:
        # Rule ids only. The rationale text is written for reviewers and may
        # quote clinical phrasing the model should not be reasoning over.
        routing_list = _bullets(sorted(r.rule_id for r in routing_rules))
    else:
        routing_list = "  - (no screening rules in this bundle)"

    governance_list = (
        _bullets(sorted(f"{r.rule_id} ({r.action.value})" for r in governance_rules))
        if governance_rules
        else "  - (none declared)"
    )

    emergency = next(
        (t.emergency_number for t in bundle.message_templates if t.emergency_number), None
    )
    crisis = next(
        (t.crisis_service_number for t in bundle.message_templates if t.crisis_service_number),
        None,
    )
    numbers: list[str] = []
    if emergency:
        numbers.append(f"emergency number for this deployment is {emergency}")
    if crisis:
        numbers.append(f"the crisis line for this deployment is {crisis}")

    if numbers:
        numbers_block = (
            "If an urgent route is active, you may state these numbers exactly as given:\n"
            + _bullets(numbers)
            + "\nDo not substitute a different number, and do not present a number that is "
            "not listed here.\n"
        )
    else:
        numbers_block = (
            "This deployment has no emergency numbers configured. Do not state any "
            "emergency number at all; direct the person to local emergency services in "
            "general terms and hand off.\n"
        )

    footer = f"""\
DEPLOYMENT CONTEXT
  - policy version: {bundle.policy_version}
  - jurisdiction: {bundle.jurisdiction}
  - default locale: {bundle.default_locale}
  - intended use: {bundle.intended_use} (administrative scheduling only)

SCREENING RULES IN FORCE (ids only; they run outside you)
{routing_list}

ACTIONS THOSE RULES CAN TAKE
{action_summary}

GOVERNANCE / PROCEDURAL RULES IN FORCE
{governance_list}

{numbers_block}"""

    return _PROMPT_HEADER + "\n" + footer


__all__ = ["build_system_prompt"]
