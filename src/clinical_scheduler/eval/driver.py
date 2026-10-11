"""Evaluation driver: run a scenario set, score it, and write the artifacts.

This is the module the CLI calls. It is the only place that knows about the
on-disk artifact layout, and it is deliberately boring: it wires the runner, the
oracles and the scoring together and writes files. All the interesting decisions
live in those three modules, each of which can be tested without touching a
filesystem layout.

Artifact layout under the run directory::

    <run_dir>/
      manifest.json         run manifest (fingerprint of everything behavioural)
      scorecard.txt         human-readable scorecard
      scorecard.json        machine-readable scorecard
      traces.jsonl          one row per scenario: trace + findings
      summary.md            short prose summary for a reviewer

A run directory is named by its manifest fingerprint, so two runs with identical
behavioural inputs land in the same directory and their artifacts are directly
comparable. A run that differs in any behavioural input gets a different
directory, which makes "was this the same experiment?" answerable by path alone.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from clinical_scheduler.agent.prompt import build_system_prompt
from clinical_scheduler.db.repository import dump_snapshot_sql
from clinical_scheduler.eval.review import JUDGE_VERSION
from clinical_scheduler.eval.oracles import OracleResult, run_oracles
from clinical_scheduler.eval.runner import (
    ModelNotConfigured,
    ScenarioOutcome,
    ScenarioRunner,
    resolve_adapter_from_env,
)
from clinical_scheduler.eval.scenarios import (
    SCENARIO_DIR,
    Scenario,
    ScenarioSet,
    load_all_scenarios,
)
from clinical_scheduler.eval.scoring import (
    RunManifest,
    Scorecard,
    build_manifest,
    scenario_result_to_jsonl_row,
    score_run,
    write_jsonl,
)
from clinical_scheduler.policy.engine import PolicyEngine, load_baseline
from clinical_scheduler.tools.executor import TOOL_VERSIONS

#: Bumped when the database schema changes in a way that makes artifacts from an
#: earlier version non-comparable. Recorded in every manifest.
SCHEMA_VERSION = "1"


class EvaluationDriver:
    """Runs an evaluation and writes artifacts. One instance per run."""

    def __init__(
        self,
        *,
        work_dir: Path,
        out_dir: Path,
        repo_root: Path,
        policy_loader: Callable[[], PolicyEngine] = load_baseline,
        seed: int | None = 0,
        temperature: float = 0.0,
        max_turns: int = 12,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.out_dir = Path(out_dir)
        self.repo_root = Path(repo_root)
        self.policy_loader = policy_loader
        self.seed = seed
        self.temperature = temperature
        self.max_turns = max_turns

    # -- run -----------------------------------------------------------------

    def run(
        self,
        scenarios: Sequence[Scenario],
        *,
        scenario_set_versions: dict[str, str],
        holdout_included: bool,
        adapter: Any | None = None,
        notes: Sequence[str] = (),
        write: bool = True,
    ) -> tuple[RunManifest, Scorecard, list[ScenarioOutcome], list[OracleResult]]:
        """Execute the scenarios, score them, and optionally write artifacts.

        ``write=False`` runs everything and returns the results without touching
        disk, which is what the tests use so that a test run cannot accidentally
        overwrite a recorded artifact.
        """
        started_at = datetime.now(timezone.utc)
        run_id = f"run_{started_at.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

        policy = self.policy_loader()
        if adapter is None:
            adapter = resolve_adapter_from_env()

        runner = ScenarioRunner(
            adapter=adapter,
            work_dir=self.work_dir / run_id,
            policy_loader=self.policy_loader,
            seed=self.seed,
            temperature=self.temperature,
            max_turns=self.max_turns,
        )

        outcomes: list[ScenarioOutcome] = []
        results: list[OracleResult] = []
        for scenario in scenarios:
            outcome = runner.run_scenario(scenario)
            outcomes.append(outcome)
            results.append(run_oracles(outcome.record))

        finished_at = datetime.now(timezone.utc)

        fixture_hash = next(
            (o.artifacts.fixture_hash for o in outcomes if o.artifacts.fixture_hash), ""
        )

        # Token usage, summed from what the transport reported. ``None`` when it
        # reported nothing: "not measured" is not zero.
        token_usage: dict[str, int | None] = {}
        for key in ("prompt_tokens", "completion_tokens"):
            samples = [
                value
                for outcome in outcomes
                for value in outcome.record.model_metadata.get(key, [])
            ]
            token_usage[key] = sum(samples) if samples else None

        manifest = build_manifest(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            policy_version=policy.bundle.policy_version,
            policy_bundle_hash=policy.bundle_hash(),
            prompt=build_system_prompt(policy),
            tool_versions=dict(TOOL_VERSIONS),
            schema_version=SCHEMA_VERSION,
            fixture_hash=fixture_hash,
            model_provider=type(adapter).__name__,
            model_id=getattr(adapter, "model_id", "unknown"),
            model_parameters={
                "seed": self.seed,
                "temperature": self.temperature,
                "max_turns": self.max_turns,
            },
            scenario_set_versions=scenario_set_versions,
            scenario_ids=[s.scenario_id for s in scenarios],
            holdout_included=holdout_included,
            repo_root=self.repo_root,
            judge_version=JUDGE_VERSION,
            token_usage=token_usage,
            # No retry layer exists, so this is a fact rather than a placeholder.
            retries=0,
            # None because no price is configured. The scorecard reports
            # cost_per_completed_task as unmeasured for the same reason.
            cost=None,
            notes=notes,
        )

        scorecard = score_run(
            run_id=run_id,
            manifest_fingerprint=manifest.fingerprint(),
            outcomes=outcomes,
            oracle_results=results,
        )

        if write:
            self._write_artifacts(manifest, scorecard, outcomes, results)

        return manifest, scorecard, outcomes, results

    # -- artifacts -----------------------------------------------------------

    def _write_artifacts(
        self,
        manifest: RunManifest,
        scorecard: Scorecard,
        outcomes: Sequence[ScenarioOutcome],
        results: Sequence[OracleResult],
    ) -> Path:
        run_dir = self.out_dir / scorecard.manifest_fingerprint
        run_dir.mkdir(parents=True, exist_ok=True)

        rows = [
            scenario_result_to_jsonl_row(outcome, result)
            for outcome, result in zip(outcomes, results)
        ]

        (run_dir / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
        (run_dir / "scorecard.txt").write_text(scorecard.report(), encoding="utf-8")
        # plan.md section 18 names this artifact `scorecard.md`. Written under both
        # names so the contract and the existing readers are each satisfied.
        (run_dir / "scorecard.md").write_text(scorecard.report(), encoding="utf-8")
        (run_dir / "scorecard.json").write_text(
            json.dumps(asdict(scorecard), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        write_jsonl(run_dir / "traces.jsonl", rows)

        # plan.md section 18 also requires the database state before and after, and
        # the diff between them. Each scenario runs against its *own* database, so
        # these two files carry one labelled section per scenario rather than one
        # aggregate: concatenating them unlabelled would make the artifact
        # unreadable and the diff unattributable.
        (run_dir / "db_before.sql").write_text(
            "\n".join(
                f"-- scenario {row['scenario_id']}\n"
                + dump_snapshot_sql(outcome.record.snapshot_before)
                for row, outcome in zip(rows, outcomes)
            ),
            encoding="utf-8",
        )
        (run_dir / "db_after.sql").write_text(
            "\n".join(
                f"-- scenario {row['scenario_id']}\n"
                + dump_snapshot_sql(outcome.record.snapshot_after)
                for row, outcome in zip(rows, outcomes)
            ),
            encoding="utf-8",
        )
        (run_dir / "diff.json").write_text(
            json.dumps(
                {row["scenario_id"]: row["db_delta"] for row in rows},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (run_dir / "results.json").write_text(
            json.dumps(
                [
                    {
                        "scenario_id": row["scenario_id"],
                        "ran": row["ran"],
                        "passed": row["passed"],
                        "severity": row["severity"],
                        "finding_count": len(row["findings"]),
                        "findings": row["findings"],
                    }
                    for row in rows
                ],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (run_dir / "summary.md").write_text(
            _render_summary(manifest, scorecard), encoding="utf-8"
        )
        return run_dir


def _render_summary(manifest: RunManifest, scorecard: Scorecard) -> str:
    """A short prose summary. Written by code, never by a model.

    A model-written summary of an evaluation is a claim about a claim, and it can
    smooth over exactly the detail that matters. This renderer states the facts
    and nothing else.
    """
    gate = "PASS" if scorecard.eligible else "FAIL"
    lines = [
        f"# Evaluation run {scorecard.run_id}",
        "",
        f"- **Eligibility gate:** {gate}",
        f"- **Manifest fingerprint:** `{scorecard.manifest_fingerprint}`",
        f"- **Policy version:** `{manifest.policy_version}` "
        f"(bundle hash `{manifest.policy_bundle_hash[:16]}…`)",
        f"- **Model:** `{manifest.model_provider}` / `{manifest.model_id}`",
        f"- **Fixture hash:** `{manifest.fixture_hash[:16]}…`",
        f"- **Scenarios:** {len(scorecard.scenarios)} "
        f"({sum(1 for s in scorecard.scenarios if s.passed)} passed)",
        f"- **Holdout included:** {manifest.holdout_included}",
    ]
    if manifest.git.get("available"):
        lines.append(
            f"- **Commit:** `{manifest.git.get('commit', '')[:12]}` "
            f"(dirty: {manifest.git.get('dirty')})"
        )
    else:
        lines.append(f"- **Commit:** unavailable ({manifest.git.get('reason', 'unknown')})")
    lines.append("")

    if scorecard.vetoes:
        lines.append("## Vetoes")
        for veto in scorecard.vetoes:
            lines.append(f"- {veto}")
        lines.append("")

    lines.append("## Rates")
    for rate in scorecard.rates.values():
        lines.append(f"- {rate.display()}")
    lines.append("")

    lines.append("## Severity of failures")
    for severity in ("S0", "S1", "S2", "S3"):
        lines.append(f"- {severity}: {scorecard.severity_counts.get(severity, 0)}")
    lines.append("")

    failing = [s for s in scorecard.scenarios if not s.passed]
    if failing:
        lines.append("## Failing scenarios")
        for score in failing:
            lines.append(f"- **{score.scenario_id}** ({score.effective_severity}) — {score.title}")
            lines.append(f"  - Point: {score.key_assertion}")
            if score.execution_error:
                lines.append(f"  - Execution error: {score.execution_error}")
            if score.config_error:
                lines.append(f"  - Configuration error: {score.config_error}")
        lines.append("")

    if manifest.notes:
        lines.append("## Notes")
        for note in manifest.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------


def load_sets(*, include_holdout: bool = False) -> list[ScenarioSet]:
    return load_all_scenarios(SCENARIO_DIR, include_holdout=include_holdout)


def flatten(scenario_sets: Sequence[ScenarioSet]) -> list[Scenario]:
    out: list[Scenario] = []
    for scenario_set in scenario_sets:
        out.extend(scenario_set.scenarios)
    return out


def aggregate_versions(scenario_sets: Sequence[ScenarioSet]) -> dict[str, str]:
    return {s.set_id: s.set_version for s in scenario_sets}


__all__ = [
    "EvaluationDriver",
    "ModelNotConfigured",
    "SCHEMA_VERSION",
    "aggregate_versions",
    "flatten",
    "load_sets",
]
