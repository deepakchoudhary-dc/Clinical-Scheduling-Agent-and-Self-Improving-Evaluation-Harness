"""The CLI surface: the console script must resolve and refuse honestly.

``pyproject.toml`` declares ``csched = clinical_scheduler.cli:main``. That claim
went unchecked once already -- the module did not exist, and the entry point
raised ``ModuleNotFoundError`` on first use (defect D-030, found by the 2026-10-08
static audit). These tests are the check.

The behaviour worth pinning is the refusal: with no model endpoint configured,
``eval`` must exit non-zero and say so, never fall back to something that would
produce a number nobody could attribute to a model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_scheduler.cli import (
    REPO_ROOT,
    _adapter_for,
    _load_config,
    build_parser,
    main,
)
from clinical_scheduler.eval.runner import ModelNotConfigured


def test_the_declared_console_script_resolves() -> None:
    """The packaging claim, made mechanical."""
    from clinical_scheduler.cli import main as entry_point

    assert callable(entry_point)


def test_version_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.startswith("csched ")


def test_the_parser_exposes_the_documented_subcommands() -> None:
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions, "the parser has no subcommand group"
    choices = set(actions[0].choices or {})
    assert {
        "version",
        "verify",
        "repair",
        "demo",
        "report",
        "metrics",
        "chat",
        "eval",
        "multiseed",
        "review",
        "calibrate",
        "checkpoint",
    } <= choices


def test_eval_without_a_model_endpoint_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No endpoint is a refusal, not a fallback to something unlabelled."""
    monkeypatch.delenv("CLINICAL_SCHEDULER_MODEL_ENDPOINT", raising=False)
    assert main(["eval", "--suite", "core"]) == 2
    assert "no model endpoint configured" in capsys.readouterr().err


def test_a_missing_config_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_config(str(tmp_path / "absent.yaml"))
    assert "config file not found" in str(excinfo.value)


def test_a_config_without_a_model_block_yields_no_model(tmp_path: Path) -> None:
    """The committed config ships with the model block commented out on purpose."""
    config = _load_config(str(REPO_ROOT / "configs" / "baseline.yaml"))
    assert "model" not in config, (
        "configs/baseline.yaml now carries a model block; if that is deliberate, "
        "make sure it names an api_key_env rather than a key"
    )
    assert config["suite"] in {"core", "adversarial", "metamorphic", "holdout", "all"}


def test_api_key_env_is_resolved_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A credential is named, never inlined -- and a missing name is an error."""
    block = {
        "provider": "openai_compatible",
        "base_url": "http://127.0.0.1:9/v1",
        "model_id": "test-model",
        "api_key_env": "CSCHED_TEST_KEY",
    }

    monkeypatch.delenv("CSCHED_TEST_KEY", raising=False)
    with pytest.raises(ModelNotConfigured) as excinfo:
        _adapter_for(block)
    assert "CSCHED_TEST_KEY" in str(excinfo.value)

    monkeypatch.setenv("CSCHED_TEST_KEY", "not-a-real-credential")
    adapter = _adapter_for(block)
    assert type(adapter).__name__ == "OpenAICompatAdapter"
