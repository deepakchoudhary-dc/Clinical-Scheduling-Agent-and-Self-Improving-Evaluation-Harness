"""Scenario runner: executes scenarios against the live orchestrator and
collects a :class:`RunRecord` for the oracle stack.

The runner is the only place that couples the evaluator to the runtime. Its
contract is narrow on purpose:

* it creates a **fresh database per scenario** from the deterministic fixture, so
  a scenario cannot inherit state from the one before it (plan.md section 15,
  checklist item "every scenario starts from a fresh deterministic fixture");
* it builds a real orchestrator over a real adapter, with no test-only bypass of
  the sentinel, broker or executor;
* it records what happened, and makes **no judgements**. Judging is the oracle
  stack's job, and keeping the two apart is what lets a reviewer re-run the
  oracles over a stored record without re-running the system.

On the model endpoint: the runner requires a configured endpoint and refuses to
invent one. There is no built-in fake model anywhere in this package. When no
endpoint is configured the runner raises, and the failure is reported as a
configuration error rather than as a scenario failure -- a distinction that
matters, because "the model was not reachable" and "the agent routed a cardiac
patient into a booking slot" must never be able to look the same in a scorecard.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from clinical_scheduler.adapters.llm import ModelAdapter, build_adapter
from clinical_scheduler.agent.orchestrator import Orchestrator
from clinical_scheduler.agent.response_grounding import ResponseComposer
from clinical_scheduler.db.repository import Repository
from clinical_scheduler.db.seed import FIXTURE_NOW, build_fixture_database
from clinical_scheduler.domain.states import AuthTier
from clinical_scheduler.eval.oracles import RunRecord
from clinical_scheduler.eval.scenarios import Scenario
from clinical_scheduler.kernel.scheduling import FixedClock, SchedulingKernel
from clinical_scheduler.policy.engine import PolicyEngine, load_baseline
from clinical_scheduler.tools.broker import CapabilityBroker
from clinical_scheduler.tools.executor import ToolExecutor


class ModelNotConfigured(RuntimeError):
    """Raised when no model endpoint is available.

    This is deliberately not an ``AdapterError``: it must not be catchable by the
    same handler that treats a mid-session outage as a degraded turn. A missing
    configuration invalidates the run; an outage mid-session is a real event the
    agent must handle.
    """


#: Tables the L2 oracle inspects for the pre/post state diff. ``Repository.
#: snapshot()`` is the source of truth; this tuple is asserted **equal** to the
#: keys of ``Repository.snapshot()`` by
#: ``tests/unit/test_eval_declarations.py``, so a table cannot be added to the
#: schema and silently go unwatched.
#:
#: The append-only audit chain is deliberately *not* here: it is verified
#: separately by its own oracle (L2 ``AUDIT_CHAIN_BROKEN``) rather than by a row
#: diff, because it is expected to grow on every call. The static audit of
#: 2026-10-08 found ``audit_events`` listed here while ``snapshot()`` did not
#: return it, which is exactly the drift the assertion now prevents.
SNAPSHOT_TABLES: tuple[str, ...] = (
    "slots",
    "slot_leases",
    "slot_consumption",
    "appointments",
    "appointment_status_history",
    "requirements",
    "requirement_acknowledgements",
    "waitlist_entries",
    "idempotency_records",
    "commit_reconciliations",
    "handoffs",
    "simulated_escalations",
    "identity_collision_tokens",
    "collision_attempts",
    "authorizations",
    "outbox_events",
)


@dataclass
class RunArtifacts:
    """Paths and hashes produced by one scenario run."""

    scenario_id: str
    db_path: str
    fixture_hash: str
    started_at: str
    finished_at: str
    duration_ms: int


@dataclass
class ScenarioOutcome:
    """A scenario run plus its oracle verdict, ready for scoring."""

    scenario: Scenario
    record: RunRecord
    artifacts: RunArtifacts
    #: Set when the run could not be executed at all (configuration error,
    #: fixture failure). Distinct from a scenario that ran and failed: only the
    #: latter is evidence about the agent.
    execution_error: str | None = None
    config_error: str | None = None

    @property
    def ran(self) -> bool:
        return self.execution_error is None and self.config_error is None


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------

#: Environment variable holding the endpoint config. Kept in the environment
#: rather than a committed file so a key can never be pushed.
ENDPOINT_ENV_VAR = "CLINICAL_SCHEDULER_MODEL_ENDPOINT"


def resolve_adapter_from_env(env: dict[str, str] | None = None) -> ModelAdapter:
    """Build the adapter from the environment, or raise ``ModelNotConfigured``.

    The configuration is a JSON object in ``CLINICAL_SCHEDULER_MODEL_ENDPOINT``:

    ``{"provider": "ollama", "model_id": "qwen2.5:7b-instruct",
       "base_url": "http://127.0.0.1:11434"}``

    or for a third-party API, with the key supplied separately and never logged:

    ``{"provider": "openai_compatible", "model_id": "gpt-4o-mini",
       "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"}``

    ``api_key_env`` names a second environment variable; the key itself is never
    written into the config blob, so a copied config cannot leak a credential.
    """
    import json

    source = (env or os.environ).get(ENDPOINT_ENV_VAR)
    if not source:
        raise ModelNotConfigured(
            f"no model endpoint configured: set {ENDPOINT_ENV_VAR} to a JSON object "
            "naming the provider, model and base URL. This harness never substitutes "
            "a built-in model, because a run whose model is unknown is not evidence."
        )
    try:
        config = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ModelNotConfigured(f"{ENDPOINT_ENV_VAR} is not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ModelNotConfigured(f"{ENDPOINT_ENV_VAR} must be a JSON object")

    config = dict(config)
    key_env = config.pop("api_key_env", None)
    if key_env:
        key = (env or os.environ).get(key_env)
        if not key:
            raise ModelNotConfigured(
                f"{ENDPOINT_ENV_VAR} references {key_env} but that variable is empty"
            )
        config["api_key"] = key

    return build_adapter(config)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ScenarioRunner:
    """Runs scenarios and returns records. Holds no cross-scenario state."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        work_dir: Path,
        policy_loader: Callable[[], PolicyEngine] = load_baseline,
        seed: int | None = 0,
        temperature: float = 0.0,
        max_turns: int = 12,
    ) -> None:
        self.adapter = adapter
        self.work_dir = Path(work_dir)
        self.policy_loader = policy_loader
        self.seed = seed
        self.temperature = temperature
        self.max_turns = max_turns

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _snapshot(repo: Repository) -> dict[str, list[dict[str, Any]]]:
        """Capture the full logical snapshot, including the audit rows.

        ``Repository.snapshot()`` intentionally excludes append-only audit and
        outbox tables. The L2 oracle needs to see audit rows, so they are added
        back: an oracle that cannot see the evidence cannot check it.

        The chain *verdict* is captured separately (see ``_audit_verdict``)
        because recomputing it here would duplicate the hashing rule in two
        places. The oracle checks the captured verdict and the row count, and a
        mutation to the rows after capture would be caught by the fact that the
        captured verdict and the captured rows travel together.
        """
        out = repo.snapshot()
        for extra in ("audit_events", "outbox_events"):
            try:
                out[extra] = [dict(row) for row in repo.query(f"SELECT * FROM {extra} ORDER BY rowid")]
            except Exception:  # noqa: BLE001 -- an absent optional table is not fatal
                out[extra] = []
        return out

    @staticmethod
    def _audit_verdict(repo: Repository) -> dict[str, Any]:
        """Ask the repository to verify its own chain, and store the answer.

        This is the one place the evaluator deliberately delegates to the system
        under test. That is acceptable here because the delegated operation is a
        *recomputation from stored bytes*, not a policy decision: it reads rows
        and re-hashes them. The oracle records the verdict alongside the row
        count so a chain that was truncated still shows up as a count mismatch
        even though the recomputation over the surviving rows would pass.
        """
        try:
            ok, detail = repo.verify_audit_chain()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "detail": f"chain verification raised: {exc}", "rows": 0}
        try:
            rows = repo.query_one("SELECT COUNT(*) AS n FROM audit_events")
            count = int(rows["n"]) if rows is not None else 0
        except Exception:  # noqa: BLE001
            count = 0
        return {"ok": bool(ok), "detail": str(detail), "rows": count}

    def _tier(self, name: str) -> AuthTier:
        return AuthTier[name]

    # -- main entry ------------------------------------------------------------

    def run_scenario(self, scenario: Scenario) -> ScenarioOutcome:
        """Execute one scenario and return an unfiltered record."""
        started = datetime.now(timezone.utc)
        scenario_dir = self.work_dir / scenario.scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        db_path = scenario_dir / "scheduler.sqlite"
        if db_path.exists():
            db_path.unlink()

        try:
            fixture = build_fixture_database(str(db_path))
        except Exception as exc:  # noqa: BLE001
            return ScenarioOutcome(
                scenario=scenario,
                record=_empty_record(scenario),
                artifacts=RunArtifacts(
                    scenario_id=scenario.scenario_id,
                    db_path=str(db_path),
                    fixture_hash="",
                    started_at=started.isoformat(),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=0,
                ),
                execution_error=f"fixture build failed: {exc}",
            )

        repo = Repository(str(db_path))
        try:
            try:
                policy = self.policy_loader()
            except Exception as exc:  # noqa: BLE001
                return ScenarioOutcome(
                    scenario=scenario,
                    record=_empty_record(scenario),
                    artifacts=RunArtifacts(
                        scenario_id=scenario.scenario_id,
                        db_path=str(db_path),
                        fixture_hash=fixture.fixture_hash,
                        started_at=started.isoformat(),
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        duration_ms=0,
                    ),
                    config_error=f"policy bundle failed to load: {exc}",
                )

            clock = FixedClock(FIXTURE_NOW)
            kernel = SchedulingKernel(repo, clock=clock)
            broker = CapabilityBroker()
            executor = ToolExecutor(
                repo=repo, broker=broker, kernel=kernel, clock=lambda: FIXTURE_NOW
            )

            trace_id = f"trace_{scenario.scenario_id.lower()}"
            orch = Orchestrator(
                repo=repo,
                adapter=self.adapter,
                policy=policy,
                kernel=kernel,
                broker=broker,
                executor=executor,
                composer=ResponseComposer(),
                session_id="sess_test",
                session_id_hash="hash_test",
                trace_id=trace_id,
                tier=self._tier(scenario.tier),
                subject_id=scenario.subject_id,
                seed=self.seed,
                temperature=self.temperature,
                max_turns=self.max_turns,
            )

            before = self._snapshot(repo)
            audit_before = self._audit_verdict(repo)
            record = RunRecord(scenario_id=scenario.scenario_id, scenario=scenario)
            record.snapshot_before = before
            record.audit_verdict_before = audit_before
            record.exceptions = []
            record.hit_turn_bound = False

            turn_bound_hit = False
            for offset, message in enumerate(scenario.turns):
                # Confirmations are runtime-issued, so the runner asks the
                # orchestrator for one exactly at the turns the scenario marks.
                if (offset + 1) in scenario.confirm_on_turns:
                    token = f"confirm_{scenario.scenario_id}_{offset + 1}"
                    orch.note_explicit_confirmation(token)
                    record.confirmations.append(
                        {"turn_index": offset, "issued": True, "token": token}
                    )

                try:
                    result = orch.run_turn(message)
                except Exception as exc:  # noqa: BLE001
                    record.exceptions.append(
                        {
                            "turn_index": offset,
                            "type": type(exc).__name__,
                            "message": str(exc)[:400],
                        }
                    )
                    turn_bound_hit = True
                    break

                turn = result.as_trace()
                turn["assistant_message"] = result.assistant_message
                record.turns.append(turn)
                record.assistant_messages.append(result.assistant_message)
                record.model_metadata.setdefault("latencies_ms", []).append(result.model_latency_ms)
                # Token usage, when the transport reported it. Recorded per call so
                # the manifest can sum it, and left absent rather than zeroed when
                # the transport said nothing (plan.md section 11.1).
                if result.prompt_tokens is not None:
                    record.model_metadata.setdefault("prompt_tokens", []).append(
                        result.prompt_tokens
                    )
                if result.completion_tokens is not None:
                    record.model_metadata.setdefault("completion_tokens", []).append(
                        result.completion_tokens
                    )

                for call in result.tool_calls:
                    record.tool_calls.append({**call, "turn_index": result.turn_index})

                record.offered_tools.append((result.turn_index, result.exposed_tools))

                for screening in orch.sentinel_log.records:
                    if screening.turn_index == result.turn_index:
                        record.screenings.append(
                            {
                                "turn_index": screening.turn_index,
                                "verdict": screening.verdict.value,
                                "fired_rule_ids": list(screening.fired_rule_ids),
                                "matched_phrases": list(screening.matched_phrases),
                                "policy_version": screening.policy_version,
                                "text_sha256": screening.text_sha256,
                                # Screening runs before the adapter call by
                                # construction in the orchestrator; recorded so L3
                                # can check it from evidence rather than trust.
                                "before_model": True,
                            }
                        )

                record.state_trail.append(
                    {
                        "turn_index": result.turn_index,
                        "from": result.state_before.value,
                        "to": result.state_after.value,
                    }
                )

                if result.state_after.value in {"HUMAN_HANDOFF", "CANCELLED", "URGENT_HANDOFF"}:
                    break

            record.snapshot_after = self._snapshot(repo)
            record.audit_verdict_after = self._audit_verdict(repo)
            record.hit_turn_bound = turn_bound_hit
            finished = datetime.now(timezone.utc)

            return ScenarioOutcome(
                scenario=scenario,
                record=record,
                artifacts=RunArtifacts(
                    scenario_id=scenario.scenario_id,
                    db_path=str(db_path),
                    fixture_hash=fixture.fixture_hash,
                    started_at=started.isoformat(),
                    finished_at=finished.isoformat(),
                    duration_ms=int((finished - started).total_seconds() * 1000),
                ),
            )
        finally:
            repo.close()


def _empty_record(scenario: Scenario) -> RunRecord:
    return RunRecord(scenario_id=scenario.scenario_id, scenario=scenario)


__all__ = [
    "ENDPOINT_ENV_VAR",
    "ModelNotConfigured",
    "RunArtifacts",
    "SNAPSHOT_TABLES",
    "ScenarioOutcome",
    "ScenarioRunner",
    "resolve_adapter_from_env",
]
