"""Unit tests for CLI Profile-Aware and Engine-Aware adaptive defaults."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ubt.cli import main as cli_main
from ubt.cli.main import app, resolve_cli_adaptive_dual_mode

runner = CliRunner()


@pytest.fixture
def sample_book_md(tmp_path: Path) -> Path:
    f = tmp_path / "paper_test.md"
    f.write_text("# Abstract\n\nTesting spatiotemporal composability.\n", encoding="utf-8")
    return f


def test_resolve_cli_adaptive_dual_mode_matrix() -> None:
    # Explicit user choice always wins
    assert resolve_cli_adaptive_dual_mode("inline", "paper", "auto") == "inline"
    assert resolve_cli_adaptive_dual_mode("facing", "paper", "rigid") == "facing"
    assert resolve_cli_adaptive_dual_mode("monolingual", "general", "reflow") == "monolingual"

    # Smart default for academic papers: monolingual
    assert resolve_cli_adaptive_dual_mode(None, "paper", "auto") == "monolingual"
    assert resolve_cli_adaptive_dual_mode(None, "paper", "reflow") == "monolingual"

    # Smart default for fiction / novels: monolingual
    assert resolve_cli_adaptive_dual_mode(None, "fiction", "auto") == "monolingual"
    assert resolve_cli_adaptive_dual_mode(None, "novel", "reflow") == "monolingual"

    # Smart default for rigid engine: monolingual
    assert resolve_cli_adaptive_dual_mode(None, "general", "rigid") == "monolingual"
    assert resolve_cli_adaptive_dual_mode(None, "textbook", "inplace") == "monolingual"

    # General prose and textbooks on reflow/auto remain unset (follow config / UBT_DUAL_MODE / inline)
    assert resolve_cli_adaptive_dual_mode(None, "general", "auto") is None
    assert resolve_cli_adaptive_dual_mode(None, "textbook", "reflow") is None
    assert resolve_cli_adaptive_dual_mode(None, "humanities", "auto") is None


def test_cli_paper_profile_defaults_to_monolingual_without_warning(
    monkeypatch: pytest.MonkeyPatch, sample_book_md: Path, tmp_path: Path
) -> None:
    built: list[Any] = []
    real_build = cli_main._build_config

    def spy(overrides: dict[str, Any]) -> Any:
        cfg = real_build(overrides)
        built.append(cfg)
        return cfg

    monkeypatch.setattr(cli_main, "_build_config", spy)

    out = tmp_path / "paper_out.md"
    result = runner.invoke(
        app, ["translate", str(sample_book_md), "--profile", "paper", "-o", str(out), "--dry-run"]
    )
    assert result.exit_code == 0
    assert built, "_build_config was not called"
    assert built[-1].dual_mode == "monolingual"
    assert "will be downgraded to 'monolingual'" not in result.stdout


def test_cli_rigid_engine_defaults_to_monolingual_without_warning(
    monkeypatch: pytest.MonkeyPatch, sample_book_md: Path, tmp_path: Path
) -> None:
    built: list[Any] = []
    real_build = cli_main._build_config

    def spy(overrides: dict[str, Any]) -> Any:
        cfg = real_build(overrides)
        built.append(cfg)
        return cfg

    monkeypatch.setattr(cli_main, "_build_config", spy)

    out = tmp_path / "rigid_out.md"
    result = runner.invoke(
        app,
        ["translate", str(sample_book_md), "--render-engine", "rigid", "-o", str(out), "--dry-run"],
    )
    assert result.exit_code == 0
    assert built, "_build_config was not called"
    assert built[-1].dual_mode == "monolingual"
    assert "will be downgraded to 'monolingual'" not in result.stdout


def test_cli_explicit_dual_mode_conflict_with_rigid_warns(
    monkeypatch: pytest.MonkeyPatch, sample_book_md: Path, tmp_path: Path
) -> None:
    out = tmp_path / "rigid_inline.md"
    result = runner.invoke(
        app,
        [
            "translate",
            str(sample_book_md),
            "--render-engine",
            "rigid",
            "--dual-mode",
            "inline",
            "-o",
            str(out),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Warning: --render-engine rigid is monolingual-only:" in result.stdout
    assert "will be downgraded to 'monolingual'" in result.stdout


def test_preflight_panel_formats_adaptive_dual_mode_without_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pre-flight layout tradeoff panel must not display '--dual-mode None' and must reflect adaptive dual-mode."""
    from unittest.mock import MagicMock

    from ubt.core.advisor import DocumentAdvisor

    pdf_file = tmp_path / "dense_math.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%dense\n")

    fake_adv = MagicMock()
    fake_adv.recommended_render_engine = "rigid"
    fake_adv.math_density = "high"
    monkeypatch.setattr(DocumentAdvisor, "analyze", staticmethod(lambda _p: fake_adv))
    monkeypatch.setattr("ubt.cli.commands.translate._is_interactive", lambda: True)

    result = runner.invoke(
        app,
        ["translate", str(pdf_file), "--profile", "paper", "--render-engine", "reflow"],
        input="3\n",
    )
    assert "--dual-mode None" not in result.output
    assert (
        "adaptive dual-mode: monolingual" in result.output
        or "自适应双语模式: monolingual" in result.output
    )


def test_preflight_panel_option1_heals_conflicting_output_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Choosing option [1] (rigid) must heal contradictory output filename containing reflow/bilingual."""
    from unittest.mock import MagicMock

    from ubt.core.advisor import DocumentAdvisor

    pdf_file = tmp_path / "dense_math.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%dense\n")

    fake_adv = MagicMock()
    fake_adv.recommended_render_engine = "rigid"
    fake_adv.math_density = "high"
    monkeypatch.setattr(DocumentAdvisor, "analyze", staticmethod(lambda _p: fake_adv))
    monkeypatch.setattr("ubt.cli.commands.translate._is_interactive", lambda: True)

    out_file = tmp_path / "paper_reflow_bilingual.pdf"
    captured_paths: list[Path | None] = []

    async def fake_run_translation(*args: Any, **kwargs: Any) -> Path:
        captured_paths.append(kwargs.get("output_path"))
        return kwargs.get("output_path") or out_file

    monkeypatch.setattr(
        "ubt.cli.commands.translate._get_run_translation", lambda: fake_run_translation
    )

    result = runner.invoke(
        app,
        [
            "translate",
            str(pdf_file),
            "--profile",
            "paper",
            "--render-engine",
            "reflow",
            "--dual-mode",
            "inline",
            "-o",
            str(out_file),
            "--dry-run",
        ],
        input="1\n",
    )
    assert result.exit_code == 0
    assert captured_paths
    assert captured_paths[-1] is not None
    assert "reflow" not in captured_paths[-1].name
    assert "bilingual" not in captured_paths[-1].name
    assert "rigid" in captured_paths[-1].name


def test_preflight_panel_option2_guarantees_companion_rigid_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Choosing option [2] (keep reflow + companion) must set emit_companion_rigid on config."""
    from unittest.mock import MagicMock

    from ubt.core.advisor import DocumentAdvisor

    pdf_file = tmp_path / "dense_math.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%dense\n")

    fake_adv = MagicMock()
    fake_adv.recommended_render_engine = "rigid"
    fake_adv.math_density = "high"
    monkeypatch.setattr(DocumentAdvisor, "analyze", staticmethod(lambda _p: fake_adv))
    monkeypatch.setattr("ubt.cli.commands.translate._is_interactive", lambda: True)

    captured_kwargs: dict[str, Any] = {}

    async def spy_run_translation(*args: Any, **kwargs: Any) -> Path:
        captured_kwargs.update(kwargs)
        return tmp_path / "out.pdf"

    monkeypatch.setattr(
        "ubt.cli.commands.translate._get_run_translation", lambda: spy_run_translation
    )

    result = runner.invoke(
        app,
        [
            "translate",
            str(pdf_file),
            "--profile",
            "paper",
            "--render-engine",
            "reflow",
            "--dual-mode",
            "inline",
            "--dry-run",
        ],
        input="2\n",
    )
    assert result.exit_code == 0
    assert captured_kwargs.get("emit_companion_rigid") is True

    # Also verify that when passed to overrides, _build_config populates config.emit_companion_rigid
    cfg = cli_main._build_config({"emit_companion_rigid": True})
    assert cfg.emit_companion_rigid is True


def test_auto_render_engine_bypasses_formula_enrichment_on_formula_heavy_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When render_engine is 'auto' and a PDF is formula_heavy, resolve_formula_enrichment must return False."""
    import ubt.core.ports as ports
    from ubt.adapters.pdf.docling_parser import resolve_formula_enrichment

    pdf_file = tmp_path / "formula_doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%math\n")

    # Mock classify_pdf_content: formula_heavy=True
    monkeypatch.setattr(ports, "classify_pdf_content", lambda _path: (False, True))

    # With render_engine='auto' on formula-dense document, VLM is bypassed
    should_enrich = resolve_formula_enrichment(
        formula_enrichment="auto",
        render_engine="auto",
        formula_render="witness",
        has_accelerator=lambda: True,
        path=pdf_file,
    )
    assert should_enrich is False

    # When formula_heavy is False (prose), with GPU available, it is allowed
    monkeypatch.setattr(ports, "classify_pdf_content", lambda _path: (False, False))
    should_enrich_prose = resolve_formula_enrichment(
        formula_enrichment="auto",
        render_engine="auto",
        formula_render="witness",
        has_accelerator=lambda: True,
        path=pdf_file,
    )
    assert should_enrich_prose is True

    # Explicit 'on' always overrides
    should_enrich_forced = resolve_formula_enrichment(
        formula_enrichment="on",
        render_engine="auto",
        formula_render="witness",
        has_accelerator=lambda: True,
        path=pdf_file,
    )
    assert should_enrich_forced is True


@pytest.mark.asyncio
async def test_advisory_stage_downgrades_when_adaptive_policy_resolves_rigid(
    tmp_path: Path,
) -> None:
    """When config.render_engine is 'auto' but adaptive_policy.render_engine is 'rigid',
    advisory stage must downgrade effective_mode to 'monolingual' and record dual_mode_downgraded."""
    from unittest.mock import MagicMock

    from ubt.core.config import UBTConfig
    from ubt.core.engine.stages.advisory import run_mode_advisory_stage
    from ubt.core.ir.models import BlockType, BookManifest, BoundingBox, IRBlock

    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 mock")

    manifest = BookManifest(doc_id="doc1", title="Test", source_path=str(pdf_file))
    manifest.run.bilingual_mode = "inline"

    config = UBTConfig(render_engine="auto", dual_mode="inline")

    ctx = MagicMock()
    ctx.config = config
    ctx.manifest = manifest
    ctx.source_pdf_path = pdf_file
    ctx.input_path = pdf_file
    ctx.profile_name = "paper"
    ctx.job_id = "job_test_rigid_downgrade"
    ctx.enforcement = "advise"

    # Adaptive policy resolved render_engine='rigid'
    adaptive_policy = MagicMock()
    adaptive_policy.render_engine = "rigid"
    ctx.adaptive_policy = adaptive_policy

    blocks = [
        IRBlock(
            id=f"b{i}",
            spine_index=i,
            block_type=BlockType.FORMULA if i % 2 == 0 else BlockType.NARRATIVE,
            source_text="E = mc^2" if i % 2 == 0 else "Energy is conserved.",
            target_text="E = mc^2" if i % 2 == 0 else "能量守恒。",
            skip_translate=(i % 2 == 0),
            bbox=BoundingBox(page=1, x0=50.0, y0=100.0 + i * 20, x1=400.0, y1=115.0 + i * 20),
        )
        for i in range(1, 10)
    ]

    async def _current_blocks(force_refresh: bool = False) -> list[IRBlock]:
        return blocks

    async def _create_event(*args: object, **kwargs: object) -> object:
        mock_ev = MagicMock()
        mock_ev.event_type = "MODE_ADVISED"
        return mock_ev

    ctx.current_blocks = _current_blocks
    ctx.create_event = _create_event

    events = [ev async for ev in run_mode_advisory_stage(ctx)]
    assert len(events) == 1
    assert manifest.run.dual_mode_downgraded == "inline"
    assert manifest.run.effective_dual_mode == "monolingual"
    assert manifest.run.bilingual_mode == "monolingual"
    assert manifest.run.bilingual_advisory is not None
    assert manifest.run.bilingual_advisory["effective"] == "monolingual"
    assert manifest.run.bilingual_advisory["rendered_modes"] == ["monolingual"]


def test_export_document_label_is_adaptive() -> None:
    """Spec of the document-label predicate: monolingual vs bilingual outputs.

    The production label is computed inline in the export stage and the CLI
    summary; this pins the predicate both must implement.
    """
    from ubt.core.ir.models import BookManifest

    manifest_mono = BookManifest(doc_id="m1", title="Mono", source_path="mono.pdf")
    manifest_mono.run.effective_dual_mode = "monolingual"
    manifest_mono.run.bilingual_mode = "monolingual"

    manifest_bilingual = BookManifest(doc_id="b1", title="Bi", source_path="bi.pdf")
    manifest_bilingual.run.effective_dual_mode = "inline"
    manifest_bilingual.run.bilingual_mode = "inline"

    def get_doc_label(m: BookManifest) -> str:
        is_bi = (
            getattr(m.run, "effective_dual_mode", None) not in ("monolingual", None)
            and getattr(m.run, "bilingual_mode", None) != "monolingual"
        )
        return "Bilingual document" if is_bi else "Translated document"

    assert get_doc_label(manifest_mono) == "Translated document"
    assert get_doc_label(manifest_bilingual) == "Bilingual document"
