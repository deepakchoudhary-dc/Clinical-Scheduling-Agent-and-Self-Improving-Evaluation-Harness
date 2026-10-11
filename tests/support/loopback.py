"""Re-export of the loopback transport, which now lives in the package.

The transport moved to ``clinical_scheduler.adapters.loopback`` so that
``scripts/demo_closed_loop.py`` and ``csched demo`` could use it without importing
from the test tree -- a delivered script depending on ``tests/`` is a dependency
no packaging would install. This module is kept as a re-export so existing test
imports keep working, and so there is exactly one implementation.
"""

from clinical_scheduler.adapters.loopback import (
    LoopbackModel,
    completion,
    tool_completion,
)

__all__ = ["LoopbackModel", "completion", "tool_completion"]
