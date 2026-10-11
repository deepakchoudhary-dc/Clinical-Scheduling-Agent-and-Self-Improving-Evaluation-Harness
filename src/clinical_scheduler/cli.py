"""Command-line entry point (``plan.md`` section 18).

``pyproject.toml`` declares ``csched = "clinical_scheduler.cli:main"``. This
module is what that resolves to. Before it existed the declared console script
raised ``ModuleNotFoundError`` on first use -- a claim in the packaging that the
code did not honour.

Subcommands, each delegating to a library function or a documented script so the
CLI contains no logic of its own:

``version``
    Print the package version.

``verify``
    Run the completion gate. The gate lives in the workspace engineering record
    (``../.agent``), which is a *sibling* of this repository rather than part of
    it, so the command reports plainly when the record is not present instead of
    pretending the check ran.

``eval``
    Run an evaluation suite against a **real** model endpoint and write the
    artifacts. There is no ``--offline`` here on purpose: the offline
    demonstration (loopback transport, scripted generator) is ``csched demo``, and
    it is labelled as such. A run whose model is unknown is not evidence, so this
    command refuses to substitute one -- with no endpoint configured it says so
    and exits non-zero rather than inventing a result.

``demo``
    Run the offline closed-loop demonstration, which writes
    ``artifacts/repair/cycle-<target>-<patch>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

from clinical_scheduler import __version__
from clinical_scheduler.adapters.llm import ModelResponse, build_adapter
from clinical_scheduler.adapters.loopback import LoopbackModel, benign_script
from clinical_scheduler.db.checkpoint import (
    CHECKPOINT_KEY_ENV_VAR,
    CheckpointError,
    checkpoint_key_from_env,
    read_checkpoint,
    sign_checkpoint,
    verify_checkpoint,
    write_checkpoint,
)
from clinical_scheduler.db.repository import Repository
from clinical_scheduler.eval.calibration import (
    SCENARIO_DECLARED_LABELS,
    CalibrationReport,
    calibrate,
    delta_report,
    observations_from_run,
)
from clinical_scheduler.eval.driver import (
    EvaluationDriver,
    aggregate_versions,
    flatten,
    load_sets,
)
from clinical_scheduler.eval.multi_seed import summarise_seeds
from clinical_scheduler.eval.review import (
    PACKET_FILE,
    REVIEWER_PACKET_FILE,
    VERDICT_TEMPLATE_FILE,
    agreement_report,
    export_packet,
    ingest_verdicts,
    load_packet,
    load_run_rows,
    sampling_frame,
)
from clinical_scheduler.eval.runner import (
    ENDPOINT_ENV_VAR,
    ModelNotConfigured,
    resolve_adapter_from_env,
)

#: ``<repo>/src/clinical_scheduler/cli.py`` -> ``<repo>``
REPO_ROOT = Path(__file__).resolve().parents[2]
#: The completion gate, in the workspace record beside the repository.
GATE_SCRIPT = REPO_ROOT.parent / ".agent" / "eval" / "scripts" / "verify.py"
#: The offline demonstration script.
DEMO_SCRIPT = REPO_ROOT / "scripts" / "demo_closed_loop.py"

SUITES = ("core", "adversarial", "metamorphic", "holdout", "all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: str | None) -> dict[str, Any]:
    """Read a run configuration. Absent path means "no config", not an error."""
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_file():
        raise SystemExit(f"error: config file not found: {config_path}")
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"error: config must be a YAML mapping: {config_path}")
    return data


def _adapter_for(model_config: dict[str, Any] | None):
    """Build the adapter from an explicit ``model:`` block, else from the env.

    A key is never taken from the config file: the file names the *environment
    variable* that holds it, and the value is read here. That is what keeps a
    config blob safe to paste into a ticket.
    """
    if not model_config:
        return resolve_adapter_from_env()
    config = dict(model_config)
    key_env = config.pop("api_key_env", None)
    if key_env:
        key = os.environ.get(str(key_env))
        if not key:
            raise ModelNotConfigured(
                f"model.api_key_env names {key_env!r}, which is not set in the environment"
            )
        config["api_key"] = key
    return build_adapter(config)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"csched {__version__}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    if not GATE_SCRIPT.is_file():
        print(
            f"completion gate not found at {GATE_SCRIPT}\n"
            "The gate lives in the workspace engineering record (.agent/), which is "
            "a sibling of this repository, not part of it. A clone of "
            "clinical_scheduler alone cannot run it.",
            file=sys.stderr,
        )
        return 2
    argv = [sys.executable, str(GATE_SCRIPT)]
    if args.json:
        argv.append("--json")
    return subprocess.run(argv).returncode


def _cmd_repair(args: argparse.Namespace) -> int:
    """Run the closed-loop repair against a target scenario (plan.md section 18).

    This is ``improvement.optimizer --target <id>`` from the plan. Registered under
    both ``repair`` and ``demo``: they are the same operation, and the logic lives
    once -- in the script -- rather than being reimplemented per name.
    """
    if not DEMO_SCRIPT.is_file():
        print(f"error: repair script not found at {DEMO_SCRIPT}", file=sys.stderr)
        return 2
    argv = [sys.executable, str(DEMO_SCRIPT), "--target", args.target]
    if args.out:
        argv += ["--out", args.out]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    return subprocess.run(argv).returncode


def _cmd_metrics(args: argparse.Namespace) -> int:
    """Report the operational metrics a database's audit trail supports (§15.2)."""
    from clinical_scheduler.telemetry.events import collect

    repo = Repository(args.db, read_only=True)
    try:
        metrics = collect(repo)
    finally:
        repo.close()

    print(metrics.report())
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Print what a stored run produced (plan.md section 18: ``eval.report``)."""
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        print(f"error: no run directory at {run_dir}", file=sys.stderr)
        return 2

    printed = False
    for name in ("scorecard.md", "summary.md"):
        path = run_dir / name
        if path.is_file():
            print(path.read_text(encoding="utf-8"))
            print()
            printed = True

    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("manifest")
        print(f"  policy : {manifest.get('policy_version', 'n/a')}")
        print(
            f"  model  : {manifest.get('model_provider', 'n/a')}"
            f"/{manifest.get('model_id', 'n/a')}"
        )
        print(f"  seed   : {manifest.get('model_parameters', {}).get('seed')}")
        print(f"  holdout: {manifest.get('holdout_included')}")
        printed = True

    if not printed:
        print(
            f"error: {run_dir} holds no scorecard, summary or manifest to report",
            file=sys.stderr,
        )
        return 2
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    config = _load_config(args.config)

    suite = args.suite or config.get("suite") or "core"
    if suite not in SUITES:
        print(
            f"error: unknown suite {suite!r}; available: {', '.join(SUITES)}",
            file=sys.stderr,
        )
        return 2
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    temperature = (
        args.temperature if args.temperature is not None else config.get("temperature", 0.0)
    )
    max_turns = args.max_turns if args.max_turns is not None else config.get("max_turns", 12)
    out_dir = Path(args.out or config.get("out") or (REPO_ROOT / "artifacts" / "eval"))

    if args.offline:
        # plan.md section 18: a deterministic transport so a reviewer can validate
        # the kernel with no endpoint and no key. The run is labelled as offline in
        # its manifest notes, because a run whose model is unknown is not evidence.
        server_context: Any = LoopbackModel(benign_script())
        adapter: Any = None
    else:
        server_context = nullcontext(None)
        try:
            adapter = _adapter_for(config.get("model"))
        except ModelNotConfigured as exc:
            print(f"error: {exc}", file=sys.stderr)
            print(
                "\nConfigure an endpoint, for example:\n"
                f'  set {ENDPOINT_ENV_VAR}={{"provider":"ollama","base_url":'
                '"http://127.0.0.1:11434","model_id":"qwen2.5:7b-instruct"}\n'
                "or put a `model:` block in a config file and pass --config, or pass "
                "--offline to run the deterministic transport. "
                "See .env.example and configs/baseline.yaml.",
                file=sys.stderr,
            )
            return 2

    include_holdout = suite in ("holdout", "all")
    scenario_sets = load_sets(include_holdout=include_holdout)
    if suite != "all":
        scenario_sets = [s for s in scenario_sets if s.set_id == suite]
    if not scenario_sets:
        print(f"error: no scenario set named {suite!r}", file=sys.stderr)
        return 2

    scenarios = flatten(scenario_sets)
    driver = EvaluationDriver(
        work_dir=out_dir / "_work",
        out_dir=out_dir,
        repo_root=REPO_ROOT,
        seed=seed,
        temperature=temperature,
        max_turns=max_turns,
    )
    note = f"csched eval --suite {suite}"
    if args.offline:
        note += " (offline: labelled loopback transport, not evidence about a model)"

    with server_context as server:
        if server is not None:
            adapter = server.build_adapter()
        manifest, scorecard, _outcomes, _results = driver.run(
            scenarios,
            scenario_set_versions=aggregate_versions(scenario_sets),
            holdout_included=any(s.frozen_holdout for s in scenario_sets),
            adapter=adapter,
            notes=[note],
        )

    print(scorecard.report())
    print(f"\nartifacts: {out_dir / scorecard.manifest_fingerprint}")
    return 0 if scorecard.eligible else 1


def _cmd_multiseed(args: argparse.Namespace) -> int:
    """Run the same suite under several seeds and report whether it is stable.

    This is the input plan.md section 13.5's "multi-seed result is stable"
    condition needs, and which no single run can supply.
    """
    config = _load_config(args.config)
    suite = args.suite or config.get("suite") or "core"
    if suite not in SUITES:
        print(f"error: unknown suite {suite!r}", file=sys.stderr)
        return 2

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2:
        print(
            "error: a stability sweep needs at least two seeds; "
            "one run cannot distinguish an improvement from a coincidence",
            file=sys.stderr,
        )
        return 2

    try:
        adapter = _adapter_for(config.get("model"))
    except ModelNotConfigured as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("see .env.example and configs/baseline.yaml", file=sys.stderr)
        return 2

    include_holdout = suite in ("holdout", "all")
    scenario_sets = load_sets(include_holdout=include_holdout)
    if suite != "all":
        scenario_sets = [s for s in scenario_sets if s.set_id == suite]
    if not scenario_sets:
        print(f"error: no scenario set named {suite!r}", file=sys.stderr)
        return 2

    scenarios = flatten(scenario_sets)
    out_dir = Path(args.out or config.get("out") or (REPO_ROOT / "artifacts" / "multiseed"))
    temperature = args.temperature if args.temperature is not None else config.get("temperature", 0.0)
    max_turns = args.max_turns if args.max_turns is not None else config.get("max_turns", 12)

    cards = []
    for seed in seeds:
        driver = EvaluationDriver(
            work_dir=out_dir / "_work",
            out_dir=out_dir,
            repo_root=REPO_ROOT,
            seed=seed,
            temperature=temperature,
            max_turns=max_turns,
        )
        _manifest, scorecard, _outcomes, _results = driver.run(
            scenarios,
            scenario_set_versions=aggregate_versions(scenario_sets),
            holdout_included=any(s.frozen_holdout for s in scenario_sets),
            adapter=adapter,
            notes=[f"csched multiseed --suite {suite} --seeds {args.seeds}"],
        )
        cards.append((seed, scorecard))

    summary = summarise_seeds(cards)
    print(summary.report())
    print(f"\nartifacts: {out_dir}")
    return 0 if summary.stable else 1


def _cmd_review_export(args: argparse.Namespace) -> int:
    """Sample a stored run and write a packet two humans can work through."""
    try:
        rows = load_run_rows(args.run)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("point --run at a directory written by `csched eval`", file=sys.stderr)
        return 2

    items = sampling_frame(rows, fraction=args.fraction, seed=args.seed)
    directory = export_packet(items, args.out)
    print(f"sampled {len(items)} of {len(rows)} scenario(s)")
    print(f"packet: {directory}")
    print(f"  {REVIEWER_PACKET_FILE}  blinded view -- give THIS to the reviewers")
    print(f"  {VERDICT_TEMPLATE_FILE}  one row per item; fill reviewer + verdict")
    print(f"  {PACKET_FILE}            carries the oracle's verdicts; keep it back")
    print()
    print("Do NOT hand a reviewer packet.json: seeing the oracle's verdict biases the")
    print("review, and the agreement statistic would then measure deference.")
    print()
    print("Have two independent reviewers complete the template, then:")
    print(f"  csched review ingest --packet {directory} --verdicts <filled.csv>")
    return 0


def _cmd_review_ingest(args: argparse.Namespace) -> int:
    """Compute the Layer-5 statistics from completed verdicts."""
    try:
        items = load_packet(args.packet)
        verdicts = ingest_verdicts(args.verdicts)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: unreadable verdict: {exc}", file=sys.stderr)
        return 2

    print(agreement_report(items, verdicts).report())
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Report the sentinel's sensitivity, specificity and predictive values.

    Labels come from the scenarios' declared expectations, not from clinician
    adjudication, and the report says so: this is a consistency measurement.

    ``--save`` writes the calibration so a later run can be diffed against it, and
    ``--compare`` prints the recall delta and exits non-zero on a regression. That
    pair *is* the change-control rule: a policy edit is reviewable only against the
    recall evidence from before it.
    """
    from clinical_scheduler.eval.scenarios import all_scenarios

    try:
        rows = load_run_rows(args.run)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    index = {s.scenario_id: s for s in all_scenarios(include_holdout=args.include_holdout)}
    pairs = observations_from_run(rows, index)
    if not pairs:
        print(
            "error: no scenario in this run declares an expected route, so there is "
            "nothing to calibrate the sentinel against",
            file=sys.stderr,
        )
        return 2

    report = CalibrationReport(counts=calibrate(pairs), label_source=SCENARIO_DECLARED_LABELS)
    print(report.report())

    if args.save:
        destination = Path(args.save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report.to_json(), encoding="utf-8")
        print()
        print(f"saved: {destination}")

    if args.compare:
        try:
            previous = CalibrationReport.from_json(
                Path(args.compare).read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            print(f"error: no previous calibration at {args.compare}", file=sys.stderr)
            return 2
        delta = delta_report(previous, report)
        print()
        print(delta.report())
        return 1 if delta.regressed else 0

    return 0


def _cmd_checkpoint_create(args: argparse.Namespace) -> int:
    """Anchor the current head of a database's audit chain."""
    try:
        key = checkpoint_key_from_env(args.key_env)
    except CheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    repo = Repository(args.db, read_only=True)
    try:
        checkpoint = sign_checkpoint(repo, key=key, key_id=args.key_id)
    except CheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        repo.close()

    path = write_checkpoint(checkpoint, args.out)
    print(f"anchored audit row {checkpoint.sequence_no} (key id {checkpoint.key_id!r})")
    print(f"checkpoint: {path}")
    return 0


def _cmd_checkpoint_verify(args: argparse.Namespace) -> int:
    """Verify a checkpoint's signature, then verify it still describes the chain."""
    try:
        key = checkpoint_key_from_env(args.key_env)
    except CheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        checkpoint = read_checkpoint(args.checkpoint)
    except FileNotFoundError:
        print(f"error: no checkpoint file at {args.checkpoint}", file=sys.stderr)
        return 2

    repo = Repository(args.db, read_only=True)
    try:
        ok, detail = verify_checkpoint(repo, checkpoint, key=key)
    finally:
        repo.close()

    print(("PASS: " if ok else "FAIL: ") + detail)
    return 0 if ok else 1


class _ManualAdapter:
    """A transport that asks the operator for the completion.

    This is the answer to "let me see what the model sees". It prints the exact
    request the runtime would send -- the conversation so far, and the tools the
    capability broker is currently willing to expose -- and reads the reply from the
    terminal.

    It is a **transport, not a model**: it cannot produce a quality signal, and every
    reply is whatever the operator typed. That is the point. It makes the harness's
    tool path steppable by hand, because the broker, the strict argument contract and
    the executor all run exactly as they would for a real model -- a proposed call is
    refused or executed for real.

    Reply forms::

        any text                 -> the model's reply text
        CALL <tool> {"json": ..}  -> a proposed tool call with those arguments
    """

    provider_name = "manual"
    model_id = "operator"

    def __init__(self, *, echo: bool = True) -> None:
        self.echo = echo
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider_name, "model_id": self.model_id, "manual": True}

    def complete(
        self,
        messages: list[Any],
        *,
        tools: tuple[Any, ...] = (),
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
        timeout_seconds: float = 60.0,
    ) -> ModelResponse:
        started = time.monotonic()
        self.calls += 1

        if self.echo:
            print()
            print("=" * 72)
            print(f"REQUEST {self.calls} to the model")
            print("=" * 72)
            for message in messages:
                body = str(getattr(message, "content", ""))
                if len(body) > 1200:
                    body = body[:1200] + " …"
                print(f"[{getattr(message, 'role', '?')}] {body}")
            print("\ntools offered (* = required argument):")
            if not tools:
                print("  none")
            for spec in tools:
                schema = getattr(spec, "parameters", {}) or {}
                required = set(schema.get("required") or ())
                properties = schema.get("properties") or {}
                arguments = ", ".join(
                    f"{name}{'*' if name in required else ''}" for name in properties
                )
                print(f"  {getattr(spec, 'name', '?')}({arguments or 'no arguments'})")
            print("=" * 72)

        try:
            raw = input("model reply > ").strip()
        except EOFError:
            # No completion available (piped input, or the operator pressed Ctrl-D).
            # Returning empty text is the honest outcome: the composer then answers
            # from grounded facts alone, which is the same path a transport failure
            # takes. Guessing a reply here would be the worst possible behaviour.
            print("  (no completion supplied -- the runtime answers from grounded facts)")
            return ModelResponse(
                text="",
                latency_ms=int((time.monotonic() - started) * 1000),
                model_id=self.model_id,
            )
        latency_ms = int((time.monotonic() - started) * 1000)

        if raw.upper().startswith("CALL "):
            head, _, tail = raw[5:].partition(" ")
            try:
                arguments = json.loads(tail) if tail.strip() else {}
            except json.JSONDecodeError as exc:
                print(f"  (arguments did not parse: {exc} -- sending no tool call)")
                return ModelResponse(text="", latency_ms=latency_ms, model_id=self.model_id)
            return ModelResponse(
                text="",
                tool_calls=(
                    {"id": f"call_manual_{self.calls}", "name": head, "arguments": arguments},
                ),
                latency_ms=latency_ms,
                model_id=self.model_id,
            )
        return ModelResponse(text=raw, latency_ms=latency_ms, model_id=self.model_id)


def _print_turn(result: Any, *, diagnostics: bool = False) -> None:
    """Print the turn.

    **The reply goes to the patient; the diagnostics go to the developer, and they
    are separate on purpose.** A person in pain, on a phone, does not need to read
    ``route: CLEAR`` and ``fired rules: none`` -- that is instrumentation, and
    putting it in front of a patient is a design failure, not a feature. It is
    behind ``--diagnostics`` for exactly that reason.
    """
    print()
    print(f"agent > {result.assistant_message}")
    if not diagnostics:
        return
    print(f"  route          : {result.safety_verdict.value}")
    print(f"  fired rules    : {list(result.policy_decision.fired_rule_ids) or 'none'}")
    print(f"  state          : {result.state_before.value} -> {result.state_after.value}")
    print(f"  tools offered  : {list(result.exposed_tools) or 'none'}")
    print(f"  action executed: {getattr(result, 'action_executed', 'unknown')}")
    if result.limit_reason:
        print(f"  SESSION LIMIT  : {result.limit_reason}")


def _cmd_chat(args: argparse.Namespace) -> int:
    """Talk to the agent one turn at a time (interactive).

    ``--manual`` makes you the model: the harness prints the request and you type the
    completion. Otherwise it uses the configured endpoint.
    """
    from clinical_scheduler.agent.orchestrator import Orchestrator, SessionLimits
    from clinical_scheduler.agent.response_grounding import ResponseComposer
    from clinical_scheduler.db.seed import FIXTURE_NOW, build_fixture_database
    from clinical_scheduler.domain.states import AuthTier
    from clinical_scheduler.kernel.scheduling import FixedClock, SchedulingKernel
    from clinical_scheduler.policy.engine import load_baseline
    from clinical_scheduler.tools.broker import CapabilityBroker
    from clinical_scheduler.tools.executor import ToolExecutor

    config = _load_config(args.config)
    if args.manual:
        adapter: Any = _ManualAdapter()
    else:
        try:
            adapter = _adapter_for(config.get("model"))
        except ModelNotConfigured as exc:
            print(f"error: {exc}", file=sys.stderr)
            print(
                "or pass --manual to type the model's replies yourself.",
                file=sys.stderr,
            )
            return 2

    db_path = (
        Path(args.db)
        if args.db
        else (REPO_ROOT / "artifacts" / "chat" / f"{args.session_id}.sqlite")
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        build_fixture_database(str(db_path))

    repo = Repository(str(db_path))
    try:
        clock = FixedClock(FIXTURE_NOW)
        kernel = SchedulingKernel(repo, clock=clock)
        broker = CapabilityBroker(secret=b"chat-session-secret-not-for-production")
        executor = ToolExecutor(
            repo=repo, broker=broker, kernel=kernel, clock=lambda: FIXTURE_NOW
        )
        orchestrator = Orchestrator(
            repo=repo,
            adapter=adapter,
            policy=load_baseline(),
            kernel=kernel,
            broker=broker,
            executor=executor,
            composer=ResponseComposer(),
            session_id=args.session_id,
            session_id_hash="c" * 64,
            trace_id=f"trace_{args.session_id}",
            # AuthTier is an int-backed enum, so the name is the lookup key.
            tier=AuthTier[args.tier],
            subject_id=args.subject_id,
            max_turns=args.max_turns,
            limits=SessionLimits(max_cost_usd=args.max_cost_usd),
        )

        print(f"database : {db_path}")
        print(f"subject  : {args.subject_id}   tier: {args.tier}")
        print(f"transport: {adapter.provider_name} ({adapter.model_id})")
        print("Type a patient message. 'exit' or Ctrl-C to stop.")

        while True:
            try:
                text = input("\npatient > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not text:
                continue
            if text.lower() in {"exit", "quit"}:
                return 0
            _print_turn(orchestrator.run_turn(text), diagnostics=args.diagnostics)
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="csched",
        description=(
            "Clinical scheduling agent and its evaluation harness. "
            "Research prototype: not a medical device, not for real patients."
        ),
    )
    parser.add_argument("--version", action="version", version=f"csched {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_version = sub.add_parser("version", help="print the package version")
    p_version.set_defaults(func=_cmd_version)

    p_verify = sub.add_parser("verify", help="run the completion gate")
    p_verify.add_argument("--json", action="store_true", help="write a machine-readable report")
    p_verify.set_defaults(func=_cmd_verify)

    # `repair` is plan.md section 18's name for the operation; `demo` is the
    # demonstration form of it. One handler for both, so the logic has one home.
    for name, help_text in (
        ("repair", "run the closed-loop repair against a target scenario"),
        ("demo", "the offline demonstration form of `repair`"),
    ):
        sub_parser = sub.add_parser(name, help=help_text)
        sub_parser.add_argument("--target", default="ADV-11", help="scenario id to repair")
        sub_parser.add_argument("--out", default=None, help="output directory for the cycle artifacts")
        sub_parser.add_argument("--seed", type=int, default=None)
        sub_parser.set_defaults(func=_cmd_repair)

    p_eval = sub.add_parser(
        "eval", help="run an evaluation suite against a configured model endpoint"
    )
    p_eval.add_argument("--config", default=None, help="run configuration (see configs/baseline.yaml)")
    p_eval.add_argument("--suite", choices=SUITES, default=None)
    p_eval.add_argument("--out", default=None)
    p_eval.add_argument("--seed", type=int, default=None)
    p_eval.add_argument("--temperature", type=float, default=None)
    p_eval.add_argument("--max-turns", type=int, default=None)
    p_eval.add_argument(
        "--offline",
        action="store_true",
        help="use the deterministic loopback transport (no endpoint, no key); labelled in the manifest",
    )
    p_eval.set_defaults(func=_cmd_eval)

    p_chat = sub.add_parser(
        "chat", help="talk to the agent one turn at a time, and see what it does"
    )
    p_chat.add_argument("--config", default=None, help="run configuration")
    p_chat.add_argument(
        "--manual",
        action="store_true",
        help="be the model: the harness prints the request and you type the completion",
    )
    p_chat.add_argument("--db", default=None, help="database to use (created and seeded if absent)")
    p_chat.add_argument("--session-id", default="sess_chat")
    p_chat.add_argument("--subject-id", default="pt_ada", help="patient the session is bound to")
    p_chat.add_argument(
        "--tier",
        default="TIER2_VERIFIED_IDENTITY",
        choices=[
            "TIER0_PUBLIC",
            "TIER1_NEW_PATIENT_REQUEST",
            "TIER2_VERIFIED_IDENTITY",
            "TIER3_STEP_UP",
        ],
    )
    p_chat.add_argument("--max-turns", type=int, default=12)
    p_chat.add_argument("--max-cost-usd", type=float, default=None)
    p_chat.add_argument(
        "--diagnostics",
        action="store_true",
        help="also print route/rules/state/tools -- instrumentation, not for patients",
    )
    p_chat.set_defaults(func=_cmd_chat)

    p_metrics = sub.add_parser(
        "metrics", help="operational metrics derived from a database's audit trail"
    )
    p_metrics.add_argument("--db", required=True, help="path to the scheduler database")
    p_metrics.set_defaults(func=_cmd_metrics)

    p_report = sub.add_parser("report", help="print what a stored run produced")
    p_report.add_argument("--run", required=True, help="a run directory")
    p_report.set_defaults(func=_cmd_report)

    p_multi = sub.add_parser(
        "multiseed", help="run a suite under several seeds and report stability"
    )
    p_multi.add_argument("--config", default=None, help="run configuration")
    p_multi.add_argument("--suite", choices=SUITES, default=None)
    p_multi.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    p_multi.add_argument("--out", default=None)
    p_multi.add_argument("--temperature", type=float, default=None)
    p_multi.add_argument("--max-turns", type=int, default=None)
    p_multi.set_defaults(func=_cmd_multiseed)

    p_review = sub.add_parser(
        "review", help="Layer-5 human review: export a packet, ingest verdicts"
    )
    review_sub = p_review.add_subparsers(dest="review_command", required=True)

    p_export = review_sub.add_parser("export", help="sample a run and write a review packet")
    p_export.add_argument("--run", required=True, help="a run directory containing traces.jsonl")
    p_export.add_argument("--out", required=True, help="where to write the packet")
    p_export.add_argument("--fraction", type=float, default=0.1, help="fraction of the rest to sample")
    p_export.add_argument("--seed", type=int, default=0)
    p_export.set_defaults(func=_cmd_review_export)

    p_ingest = review_sub.add_parser("ingest", help="ingest verdicts and report agreement")
    p_ingest.add_argument("--packet", required=True)
    p_ingest.add_argument("--verdicts", required=True)
    p_ingest.set_defaults(func=_cmd_review_ingest)

    p_cal = sub.add_parser(
        "calibrate",
        help="sentinel sensitivity/specificity and predictive values, from a stored run",
    )
    p_cal.add_argument("--run", required=True, help="a run directory containing traces.jsonl")
    p_cal.add_argument("--include-holdout", action="store_true")
    p_cal.add_argument("--save", default=None, help="write this calibration for a later --compare")
    p_cal.add_argument("--compare", default=None, help="diff against a saved calibration")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_cp = sub.add_parser("checkpoint", help="sign or verify an audit-chain anchor")
    cp_sub = p_cp.add_subparsers(dest="checkpoint_command", required=True)

    p_cp_create = cp_sub.add_parser("create", help="sign a checkpoint for a database")
    p_cp_create.add_argument("--db", required=True, help="path to the scheduler database")
    p_cp_create.add_argument("--out", required=True, help="where to write the checkpoint")
    p_cp_create.add_argument("--key-id", default="default", help="a label for the key, not the key")
    p_cp_create.add_argument("--key-env", default=CHECKPOINT_KEY_ENV_VAR)
    p_cp_create.set_defaults(func=_cmd_checkpoint_create)

    p_cp_verify = cp_sub.add_parser("verify", help="verify a checkpoint against a database")
    p_cp_verify.add_argument("--db", required=True)
    p_cp_verify.add_argument("--checkpoint", required=True)
    p_cp_verify.add_argument("--key-env", default=CHECKPOINT_KEY_ENV_VAR)
    p_cp_verify.set_defaults(func=_cmd_checkpoint_verify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
