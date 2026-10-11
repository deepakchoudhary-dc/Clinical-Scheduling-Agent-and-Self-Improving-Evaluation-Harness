"""Find a database with a non-empty audit chain, for the checkpoint demonstration.

``csched checkpoint create`` refuses to anchor an empty chain. That refusal is
correct -- an anchor over nothing proves nothing -- but it means the command needs
a database that actually has audit rows, and the per-scenario databases a short
offline run produces often do not: if the agent never invoked a tool, nothing was
audited, and the chain is legitimately empty.

This script scans for the database with the most audit rows and prints it, so the
demonstration does not depend on guessing a path.

Usage::

    python scripts/find_audit_db.py [--root artifacts]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def audit_row_count(path: Path) -> int | None:
    """Rows in ``audit_events``, or ``None`` if this is not a scheduler database.

    A file that is not a SQLite database, or is one without the table, is skipped
    rather than raising: the point of the scan is to find the right file, and a
    candidate that cannot be read is simply not it.
    """
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        return int(connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def best_database(root: Path) -> tuple[Path, int] | None:
    """The database with the most audit rows under ``root``, or ``None``."""
    best: tuple[Path, int] | None = None
    for candidate in sorted(root.rglob("*.sqlite")):
        count = audit_row_count(candidate)
        if count and (best is None or count > best[1]):
            best = (candidate, count)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(PROJECT_ROOT / "artifacts"))
    args = parser.parse_args(argv)

    found = best_database(Path(args.root))
    if found is None:
        print(f"no database with audit rows under {args.root}", file=sys.stderr)
        print(
            "run the closed-loop demonstration first, which exercises tool calls: "
            "python scripts/demo_closed_loop.py --target ADV-11",
            file=sys.stderr,
        )
        return 1

    path, count = found
    # The path goes to stdout alone, so the script composes:
    #   $db = & python scripts/find_audit_db.py
    print(path)
    print(f"# {count} audit row(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
