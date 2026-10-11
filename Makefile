# Convenience wrapper for CI and for Linux/macOS shells.
#
# plan.md section 18 asks for setup / test / demo / eval targets. Each one is a
# thin alias over a command documented in README.md, so this file cannot become
# the only place a task is defined -- and on Windows, where `make` may not be
# installed, the `csched` console script is the supported entry point.
#
#   make gate        ==  csched verify
#   make demo        ==  csched demo
#   make eval        ==  csched eval --config configs/baseline.yaml
#
# The interpreter is the in-folder virtual environment. Override with
# `make test PY=/path/to/python` if yours lives elsewhere.

PY ?= ../.venv/Scripts/python.exe

.PHONY: help setup test gate demo eval

help:
	@echo "setup  install the package and dev dependencies into the venv"
	@echo "test   run the full suite"
	@echo "gate   run the completion gate (the suite is one of its checks)"
	@echo "demo   offline closed-loop demonstration (loopback transport, labelled)"
	@echo "eval   evaluate against a configured model endpoint (see configs/baseline.yaml)"

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	PYTHONDONTWRITEBYTECODE=1 $(PY) -m pytest -q -p no:cacheprovider

gate:
	PYTHONDONTWRITEBYTECODE=1 $(PY) ../.agent/eval/scripts/verify.py

demo:
	PYTHONDONTWRITEBYTECODE=1 $(PY) scripts/demo_closed_loop.py --target ADV-11

eval:
	PYTHONDONTWRITEBYTECODE=1 $(PY) -m clinical_scheduler.cli eval --config configs/baseline.yaml
