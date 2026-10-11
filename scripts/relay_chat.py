"""Drive a persistent chat session with the agent, one turn per invocation.

This is the manual-mode equivalent of ``csched chat --manual`` split across two
parties: the *patient* types to the operator of this script, and the *model* side
is supplied from outside (an agent process, or a human reading the printed
request). Each call processes **one patient turn** against a **saved sentinel
log**, so session state -- safety route, confirmation tokens, appointment state
-- persists across separate process invocations in the same way the interactive
``csched chat`` loop keeps it in memory.

Why one process per turn: the model side can live anywhere; it only needs the
printed "REQUEST n" block to compose its reply, and the reply travels back as an
argument (``--reply`` or ``--reply-file``). Nothing in the harness treats the
model as a peer process, and nothing here pretends to: the reply is labelled in
the session record as ``relay`` (operator-supplied), never as a model's output.

Usage (two calls of one conversation)::

    # turn 1 -- patient text, no reply yet: prints the REQUEST block
    python scripts/relay_chat.py --patient "I need to move my appointment"

    # turn 2 -- the previous reply was a CALL; the model saw its result
    python scripts/relay_chat.py --patient "yes, confirm it" \\
        --reply-file last_reply.json

Reply file format (what a model side produces)::

    {"text": "...", "tool_calls": [{"name": "...", "arguments": {...}}]}

or the shorthand the manual adapter accepts, as plain text::

    {"text": "CALL hold_slot {\"slot_id\": \"...\"}"}

Both are optional when the previous turn was itself a tool call whose *result
the model must see*; in that case pass ``--result-digest`` off and the Scratchpad
prints what the tool returned, which is what the real orchestrator appends to
the conversation. Showing it here, in the terminal where the model side is
being composed, is what makes a CALL usable by a remote model at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

#: Bump when ScreeningRecord's fields change; resuming an old session with a
#: new field set would silently truncate the safety history.
LOG_VERSION = 1

# ---------------------------------------------------------------------------
# The model side
# ---------------------------------------------------------------------------


class RelayAdapter:
    """One-turn transport: the reply arrives from the command line.

    Sub-class shape mirrors ``_ManualAdapter`` in ``cli.py``: same
    ``complete()`` signature, same normalised ``ModelResponse``. It is a
    **transport, not a model** -- whoever composed the reply could be a person,
    an agent, anything. The harness's job is contracts, not authorship, and the
    session record says ``model_id: "relay"`` rather than implying a provider.
    """

    provider_name = "relay"
    model_id = "operator-via-relay"

    def __init__(self, text: str, *, echo: bool = True) -> None:
        self._text = text
        self._echo = echo

    @staticmethod
    def parse_reply(raw: str) -> tuple[str, tuple[dict, ...]]:
        """Accept the manual adapter's grammar: text, or ``CALL name {json}``."""
        stripped = raw.strip()
        if stripped.upper().startswith("CALL "):
            head, _, tail = stripped[5:].partition(" ")
            try:
                arguments = json.loads(tail) if tail.strip() else {}
            except json.JSONDecodeError as exc:
                print(f"  (relay: arguments did not parse: {exc} -- text reply instead)", flush=True)
                return stripped, ()
            return "", (
                {"id": "call_relay", "name": head, "arguments": arguments},
            )
        return stripped, ()

    def describe(self) -> dict:
        return {"provider": self.provider_name, "model_id": self.model_id, "relay": True}

    def complete(self, messages, *, tools=(), temperature=0.0, seed=None,
                 max_tokens=1024, timeout_seconds=60.0):
        import time
        started = time.monotonic()
        if self._echo:
            print("=" * 72, flush=True)
            print("REQUEST to the model", flush=True)
            print("=" * 72, flush=True)
            for message in messages:
                body = str(getattr(message, "content", ""))
                if len(body) > 1200:
                    body = body[:1200] + " ..."
                print(f"[{getattr(message, 'role', '?')}] {body}", flush=True)
            print("\ntools offered (* = required argument):", flush=True)
            if not tools:
                print("  none", flush=True)
            for spec in tools:
                schema = getattr(spec, "parameters", {}) or {}
                required = set(schema.get("required") or ())
                properties = schema.get("properties") or {}
                args = ", ".join(
                    f"{n}{'*' if n in required else ''}" for n in properties
                )
                print(f"  {getattr(spec, 'name', '?')}({args or 'no arguments'})", flush=True)
            print("=" * 72, flush=True)
        text, calls = self.parse_reply(self._text)
        from clinical_scheduler.adapters.llm import ModelResponse
        return ModelResponse(
            text=text,
            tool_calls=calls,
            latency_ms=int((time.monotonic() - started) * 1000),
            model_id=self.model_id,
        )


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


def _serialise_result(result) -> dict:
    """What the next invocation needs to rebuild exactly where we stopped."""

    def plain(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        return str(getattr(value, "value", value))

    return {
        "assistant_message": result.assistant_message,
        "safety_verdict": result.safety_verdict.value,
        "fired_rule_ids": list(result.policy_decision.fired_rule_ids),
        "state_before": result.state_before.value,
        "state_after": result.state_after.value,
        "exposed_tools": list(result.exposed_tools),
        "action_executed": getattr(result, "action_executed", None),
        "action_denied_reason": getattr(result, "action_denied_reason", None),
        "limit_reason": getattr(result, "limit_reason", None),
        "unsourced_claims": [
            plain(c) for c in getattr(result, "unsourced_claims", ()) or ()
        ],
        "grounded_facts": [
            plain(getattr(f, "text", f))
            for f in getattr(result, "grounded_facts", ()) or ()
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session-id", default="sess_relay")
    parser.add_argument("--db", default=None, help="scheduler database (created/seeded if absent)")
    parser.add_argument("--patient", required=True, help="this turn's patient message (verbatim)")
    parser.add_argument("--reply", default=None, help="the model's reply to the previous turn (text, or CALL name {json})")
    parser.add_argument("--reply-file", default=None, help="JSON file: {text, tool_calls}")
    parser.add_argument("--state-dir", default=None, help="where session/scratchpad state lives")
    parser.add_argument("--max-turns", type=int, default=24)
    args = parser.parse_args(argv)

    from clinical_scheduler.agent.orchestrator import Orchestrator, SessionLimits
    from clinical_scheduler.agent.response_grounding import ResponseComposer
    from clinical_scheduler.agent.sentinel import SentinelSessionLog
    from clinical_scheduler.db.seed import FIXTURE_NOW, build_fixture_database
    from clinical_scheduler.domain.states import AuthTier
    from clinical_scheduler.kernel.scheduling import FixedClock, SchedulingKernel
    from clinical_scheduler.policy.engine import load_baseline
    from clinical_scheduler.tools.broker import CapabilityBroker
    from clinical_scheduler.tools.executor import ToolExecutor

    state_dir = Path(args.state_dir) if args.state_dir else (
        PROJECT_ROOT / "artifacts" / "relay" / args.session_id
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = (
        Path(args.db) if args.db else (state_dir / "scheduler.sqlite")
    )
    if not db_path.exists():
        build_fixture_database(str(db_path))

    sentinel_log_path = state_dir / "sentinel_log.json"
    if sentinel_log_path.exists():
        # SentinelSessionLog and ScreeningRecord are plain dataclasses (not
        # pydantic), so the round trip is explicit. A silent field mismatch
        # would resume the session with a truncated history -- an
        # over-permissive session -- which is the direction this must fail in.
        from clinical_scheduler.domain.states import SafetyVerdict
        from clinical_scheduler.agent.sentinel import ScreeningRecord

        def _log_from_json(payload: dict) -> SentinelSessionLog:
            records = [
                ScreeningRecord(
                    turn_index=r["turn_index"],
                    verdict=SafetyVerdict(r["verdict"]),
                    fired_rule_ids=tuple(r["fired_rule_ids"]),
                    matched_phrases=tuple(r["matched_phrases"]),
                    policy_version=r["policy_version"],
                    screened_at=datetime.fromisoformat(r["screened_at"]),
                    text_length=r["text_length"],
                    text_sha256=r["text_sha256"],
                )
                for r in payload["records"]
            ]
            return SentinelSessionLog(
                session_id=payload["session_id"], records=records
            )

        payload = json.loads(sentinel_log_path.read_text(encoding="utf-8"))
        if payload.get("log_version") != LOG_VERSION:
            print(
                f"error: sentinel log version {payload.get('log_version')!r} != "
                f"{LOG_VERSION!r}; refusing to resume an incompatible session",
                file=sys.stderr,
            )
            return 2
        sentinel_log = _log_from_json(payload)
    else:
        sentinel_log = SentinelSessionLog(session_id=args.session_id)

    # Turn history: what the "conversation so far" looks like to the model side.
    # The orchestrator rebuilds messages from the session DB; this scratchpad is
    # only what the *remote model side* reads to stay coherent across turns.
    scratchpad_path = state_dir / "scratchpad.json"
    scratchpad = (
        json.loads(scratchpad_path.read_text(encoding="utf-8"))
        if scratchpad_path.exists()
        else {"turns": []}
    )

    # The dynamic session state the orchestrator must resume with: a saved
    # confirmation token, if the previous turn minted one and the composer
    # reported it.
    confirmation_path = state_dir / "confirmation_token.txt"
    pending_confirmation = (
        confirmation_path.read_text(encoding="utf-8").strip()
        if confirmation_path.exists()
        else None
    )

    # Identity binding: the executor sets ``ctx.session.verified_identity =
    # True`` and binds ``subject_id`` server-side during the verify tool
    # (executor.py, _do_verify_patient_discriminator). Both live on the
    # orchestrator's in-memory SessionContext, which the eval runner keeps
    # alive for a scenario's whole run -- but a relay turn is its own
    # process, so the binding must be persisted here or every turn after
    # verification would silently lose it and refuse mutations. The file
    # records only that the fact was established and for which subject; the
    # trust comes from the DB's collision_attempts row, not from the file,
    # and the subject id is a fixture identifier, not PHI.

    repo_path = str(db_path)
    from clinical_scheduler.db.repository import Repository as _Repository
    repo = _Repository(repo_path)
    try:
        clock = FixedClock(FIXTURE_NOW)
        kernel = SchedulingKernel(repo, clock=clock)
        broker = CapabilityBroker(secret=b"relay-session-secret-not-for-production")
        executor = ToolExecutor(
            repo=repo, broker=broker, kernel=kernel, clock=lambda: FIXTURE_NOW
        )
        orchestrator = Orchestrator(
            repo=repo,
            adapter=RelayAdapter(args.reply or "", echo=True),
            policy=load_baseline(),
            kernel=kernel,
            broker=broker,
            executor=executor,
            composer=ResponseComposer(),
            session_id=args.session_id,
            session_id_hash=hashlib.sha256(args.session_id.encode()).hexdigest(),
            trace_id=f"trace_relay_{args.session_id}",
            tier=AuthTier.TIER2_VERIFIED_IDENTITY,
            subject_id="pt_ada",
            max_turns=args.max_turns,
            limits=SessionLimits(),
        )
        # Restore the identity binding from the previous invocation, if it was
        # established. The subject id is re-applied to the fresh SessionContext
        # exactly as the executor would have bound it server-side; whether it
        # was *honestly* established is a property of the DB evidence the
        # previous turn wrote, not of this file.
        # Identity binding + FSM position: both restore from files written by
        # the previous invocation (see the write-side below).
        identity_path = state_dir / "identity_binding.json"
        sentinel_state_path = state_dir / "session_state.txt"
        if identity_path.exists():
            binding = json.loads(identity_path.read_text(encoding="utf-8"))
            if binding.get("verified") and binding.get("subject_id"):
                # Only meaningful once the session carries a subject at all;
                # a None subject cannot be verified.
                orchestrator.session.subject_id = binding["subject_id"]
                orchestrator.session.verified_identity = True
                # The verified session continues from IDENTITY_VERIFIED, the
                # state the FSM would have settled at had the process lived.
                if not sentinel_state_path.exists():
                    sentinel_state_path.write_text("IDENTITY_VERIFIED", encoding="utf-8")
        # Resume the FSM where the previous invocation left it. The state
        # file is the contract; an unknown name is a corruption signal, and
        # failing loudly beats restarting at START (which would silently
        # re-expose only the two safety tools and strand the conversation).
        if sentinel_state_path.exists():
            from clinical_scheduler.domain.states import SessionState
            try:
                orchestrator.session.state = SessionState(
                    sentinel_state_path.read_text(encoding="utf-8").strip()
                )
            except ValueError as exc:
                print(f"error: cannot resume session state: {exc}", file=sys.stderr)
                return 2

        turn = orchestrator.run_turn(args.patient)

        # -- persist everything the next turn needs -------------------------
        # Identity binding: if a tool ran on this turn and the orchestrator's
        # session now reports a verified identity, carry it forward. Read the
        # verified flag *after* the turn so the verify turn itself persists it.
        if orchestrator.session.verified_identity and orchestrator.session.subject_id:
            identity_path.write_text(
                json.dumps(
                    {
                        "verified": True,
                        "subject_id": orchestrator.session.subject_id,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        from dataclasses import asdict

        def _log_to_json(log: SentinelSessionLog) -> dict:
            return {
                "log_version": LOG_VERSION,
                "session_id": log.session_id,
                "records": [
                    {
                        "turn_index": r.turn_index,
                        "verdict": r.verdict.value,
                        "fired_rule_ids": list(r.fired_rule_ids),
                        "matched_phrases": list(r.matched_phrases),
                        "policy_version": r.policy_version,
                        "screened_at": r.screened_at.isoformat(),
                        "text_length": r.text_length,
                        "text_sha256": r.text_sha256,
                    }
                    for r in log.records
                ],
            }

        sentinel_log_path.write_text(
            json.dumps(_log_to_json(sentinel_log), indent=2), encoding="utf-8"
        )
        scratchpad["turns"].append(
            {
                "patient": args.patient,
                "agent": turn.assistant_message,
                "verdict": turn.safety_verdict.value,
                "state_after": turn.state_after.value,
                "exposed_tools": list(turn.exposed_tools),
            }
        )
        scratchpad_path.write_text(
            json.dumps(scratchpad, indent=2), encoding="utf-8"
        )

        # Any confirmation token the orchestrator now holds travels to the
        # next invocation; a mid-session restart would otherwise drop it.
        pending = getattr(orchestrator, "_pending_confirmation", None)
        if pending:
            confirmation_path.write_text(pending, encoding="utf-8")

        # Persist the FSM position so the next invocation resumes where this
        # one ended. The base position is where the FSM settled; the two
        # identity-evidence adjustments below mirror what the in-process
        # runner's `_settle_state` would have carried forward had the session
        # lived in one process, so the broker's state table exposes the same
        # tools it would have then.
        resumed_state = turn.state_after.value
        if orchestrator.session.verified_identity:
            # A verified identity settles at IDENTITY_VERIFIED regardless of
            # whether the turn started from SAFETY_SCREENED or an ambiguity.
            resumed_state = "IDENTITY_VERIFIED"
        elif (
            getattr(turn, "tool_calls", None)
            and turn.tool_calls
            and turn.tool_calls[0].get("name") == "lookup_patient_candidates"
            and turn.tool_calls[0].get("status") == "SUCCESS"
        ):
            # A successful zero-knowledge lookup starts the identity chain: the
            # verify tool is exposed only at IDENTITY_AMBIGUOUS (the broker's
            # state table), so resume there. The candidate count stays hidden
            # from the model either way (INV-04).
            if resumed_state == "SAFETY_SCREENED":
                resumed_state = "IDENTITY_AMBIGUOUS"
        (state_dir / "session_state.txt").write_text(resumed_state, encoding="utf-8")

        # Also show the tool result, if a tool ran: the model side needs to see
        # what it did, exactly as the orchestrator's next `complete()` would
        # show it in-conversation.
        if getattr(turn, "tool_calls", None):
            print()
            print(
                "--- tool result for the model side (as the orchestrator "
                "appends it to the conversation) ---",
                flush=True,
            )

        result = _serialise_result(turn)
        print()
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        repo.close()


if __name__ == "__main__":
    raise SystemExit(main())
