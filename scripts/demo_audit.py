"""Exercise the mutation path offline, and leave a database with a real audit chain.

Why this exists: ``demo_closed_loop.py`` uses a deliberately non-committal scripted
reply, so **no tool is ever invoked** and no audit row is ever written. That is
right for what it demonstrates -- a routing false negative -- but it means nothing
offline produces the audit evidence this system's whole argument rests on, and the
checkpoint control has nothing to anchor.

This script runs a real tool invocation through the real executor against a real
seeded database: capability broker, session context, kernel, audit chain. What is
substituted is nothing at all -- there is no model in this path, because a tool
invocation does not need one. It is a *driver*, not a simulation.

It writes ``artifacts/audit/scheduler.sqlite`` and prints its path, so
``csched checkpoint create`` has a database to anchor.

Usage::

    python scripts/demo_audit.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clinical_scheduler.db.repository import Repository  # noqa: E402
from clinical_scheduler.db.seed import FIXTURE_NOW, build_fixture_database  # noqa: E402
from clinical_scheduler.domain.states import AuthTier, SessionState  # noqa: E402
from clinical_scheduler.kernel.scheduling import FixedClock, SchedulingKernel  # noqa: E402
from clinical_scheduler.tools.broker import CapabilityBroker, SessionContext  # noqa: E402
from clinical_scheduler.tools.executor import ExecutorContext, ToolExecutor  # noqa: E402

#: A local secret for the capability broker. Not a credential: the broker's HMAC
#: key only has to be consistent within a run.
BROKER_SECRET = b"offline-audit-demonstration-secret"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "audit"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "scheduler.sqlite"
    if db_path.exists():
        db_path.unlink()

    bundle = build_fixture_database(str(db_path))
    repo = Repository(str(db_path))
    try:
        broker = CapabilityBroker(secret=BROKER_SECRET)
        kernel = SchedulingKernel(repo, clock=FixedClock(FIXTURE_NOW))
        executor = ToolExecutor(repo=repo, broker=broker, kernel=kernel, clock=lambda: FIXTURE_NOW)

        session = SessionContext(
            session_id="sess_audit_demo",
            subject_id=bundle.patient_ids[0],
            tier=AuthTier.TIER2_VERIFIED_IDENTITY,
            state=SessionState.OPTIONS_PRESENTED,
            verified_identity=True,
        )
        ctx = ExecutorContext(
            session=session,
            turn_index=0,
            correlation_id="corr_audit_demo",
            policy_version="2026.10-demo.1",
            trace_id="trace_audit_demo",
            session_id_hash="h" * 64,
        )

        # A real invocation: capability minted by the broker, arguments validated
        # against the strict contract, the kernel called, an audit row written.
        capability = broker.mint(session, tool_name="hold_slot")
        result = executor.invoke(
            capability=capability,
            tool_name="hold_slot",
            raw_arguments={"slot_id": bundle.slot_ids[0]},
            ctx=ctx,
        )
        status = result["envelope"]["status"]

        audit_rows = repo.query("SELECT * FROM audit_events")
        chain_ok, chain_detail = repo.verify_audit_chain()
        leases = repo.query("SELECT * FROM slot_leases")
    finally:
        repo.close()

    print(f"tool          : hold_slot -> {status}")
    print(f"leases created: {len(leases)}")
    print(f"audit rows    : {len(audit_rows)}")
    print(f"audit chain   : {'OK' if chain_ok else 'BROKEN'} ({chain_detail})")
    print()
    print(db_path)

    if not audit_rows:
        print(
            "error: no audit row was written, so there is nothing to anchor",
            file=sys.stderr,
        )
        return 1
    if not chain_ok:
        print(f"error: the audit chain does not verify: {chain_detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
