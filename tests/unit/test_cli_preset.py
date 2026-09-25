"""The CLI --preset layer (explicit flag > preset > engine default)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app
from ubt.core.job_options import sidecar_path
from ubt.core.presets import PRESETS, Preset, resolve_engine_params

runner = CliRunner()


def test_no_preset_and_no_flags_leaves_engine_defaults() -> None:
    """An un-preset invocation must not touch a single engine parameter."""
    assert resolve_engine_params(None, {}) == {}
    assert (
        resolve_engine_params(
            None,
            {
                "exec_mode": None,
                "prompt_strategy": None,
                "formula_enrichment": None,
                "formula_render": None,
                "math_backend": None,
                "render_engine": None,
                "emit_both": None,
            },
        )
        == {}
    )


def test_preset_bundle_fills_only_unset_flags() -> None:
    resolved = resolve_engine_params(Preset.PUBLICATION, {"math_backend": None, "emit_both": None})
    # The render route is deliberately NOT part of any bundle: a quality tier
    # must not silently pick a typesetting engine (see presets.py docstring).
    assert "render_engine" not in resolved
    assert resolved["exec_mode"] == "auto"
    assert resolved["prompt_strategy"] == "rich"
    assert resolved["math_backend"] == "mathjax"
    assert resolved["emit_both"] is False


def test_explicit_flag_beats_the_preset() -> None:
    resolved = resolve_engine_params(
        Preset.PUBLICATION, {"render_engine": "rigid", "exec_mode": "short"}
    )
    assert resolved["render_engine"] == "rigid"
    assert resolved["exec_mode"] == "short"
    # The rest of the bundle still applies.
    assert resolved["prompt_strategy"] == "rich"
    assert resolved["math_backend"] == "mathjax"


def test_every_preset_is_a_valid_bundle() -> None:
    for preset, policy in PRESETS.items():
        resolved = resolve_engine_params(preset, {})
        assert resolved == policy.engine_overrides()
        assert resolved["exec_mode"] in ("auto", "short", "long")


def test_fast_preset_definition_and_resolution() -> None:
    assert Preset.FAST.value == "fast"
    policy = PRESETS[Preset.FAST]
    assert policy.exec_mode == "short"
    assert policy.prompt_strategy == "minimal"
    assert policy.formula_enrichment == "off"
    assert policy.math_backend == "image"
    assert policy.formula_render == "image"

    resolved = resolve_engine_params(Preset.FAST, {})
    assert resolved["exec_mode"] == "short"
    assert resolved["prompt_strategy"] == "minimal"
    assert resolved["formula_enrichment"] == "off"
    assert resolved["math_backend"] == "image"
    assert resolved["formula_render"] == "image"
    assert resolved["emit_both"] is False


def test_preset_bundle_applies_through_validated_config() -> None:
    """Risk: the C2 validated-override path must still apply the complete preset
    bundle — including falsy values such as ``emit_both=False``, which a
    truthiness filter would silently drop (re-rendering a second artifact)."""
    from ubt.cli.main import _build_config

    cfg = _build_config(resolve_engine_params(Preset.PUBLICATION, {}))
    assert cfg.render_engine == "auto"  # preset leaves the route to smart dispatch
    assert cfg.exec_mode == "auto"
    assert cfg.prompt_strategy == "rich"
    assert cfg.math_backend == "mathjax"
    assert cfg.emit_both is False


def test_cli_help_lists_preset() -> None:
    result = runner.invoke(app, ["translate", "--help"])
    assert result.exit_code == 0
    assert "--preset" in result.stdout


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    f = tmp_path / "preset_book.md"
    f.write_text(
        "# Chapter 1: Introduction\n\nWelcome to the testing universe.\n", encoding="utf-8"
    )
    return f


def _dry_run(sample_md: Path, tmp_path: Path, extra: list[str]) -> dict[str, object]:
    out_file = tmp_path / "out.md"
    report = sidecar_path(out_file, "quality_report.json")
    result = runner.invoke(
        app,
        [
            "translate",
            str(sample_md),
            "-o",
            str(out_file),
            "--dry-run",
            "--db-dir",
            str(tmp_path / "ledgers"),
            *extra,
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    snapshot = payload["config_snapshot"]
    assert isinstance(snapshot, dict)
    return snapshot


def test_preview_preset_reaches_the_config(sample_md: Path, tmp_path: Path) -> None:
    snapshot = _dry_run(sample_md, tmp_path, ["--preset", "preview"])
    assert snapshot["prompt_strategy"] == "minimal"


def test_explicit_prompt_strategy_beats_the_preset(sample_md: Path, tmp_path: Path) -> None:
    snapshot = _dry_run(sample_md, tmp_path, ["--preset", "preview", "--prompt-strategy", "rich"])
    assert snapshot["prompt_strategy"] == "rich"


def test_without_preset_the_engine_default_survives(sample_md: Path, tmp_path: Path) -> None:
    snapshot = _dry_run(sample_md, tmp_path, [])
    assert snapshot["prompt_strategy"] == "auto"
