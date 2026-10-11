"""Evaluation harness: scenarios, simulator, oracles, scoring and runner.

Nothing in this package is imported by the runtime. The dependency direction is
runtime <- eval, never the reverse, so that the evaluator cannot influence the
subject it evaluates.
"""
