"""Unit tests for reflow/inline guardrails, VLM lifecycle cleanup, and PDF layout fixes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ubt.adapters.pdf.docling_blocks import resolve_overlapping_formula_blocks
from ubt.adapters.pdf.docling_parser import _cleanup_docling_converter
from ubt.adapters.pdf.typst_fragments import _prose_to_typst, _reference_numbers
from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
from ubt.core.cleaners.lnds_pruner import normalize_academic_pdf_math
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock


@pytest.mark.fast
def test_cleanup_docling_converter_releases_vlm_engine() -> None:
    """_cleanup_docling_converter must call engine.cleanup() and null out stage.engine
    so CodeFormulaVlmModel.__del__ is a no-op during interpreter shutdown."""
    mock_engine = MagicMock()
    mock_stage = MagicMock()
    mock_stage.engine = mock_engine

    mock_pipeline = MagicMock()
    mock_pipeline.enrichment_pipe = [mock_stage]

    mock_converter = MagicMock()
    mock_converter.initialized_pipelines = {"pdf": mock_pipeline}

    _cleanup_docling_converter(mock_converter)

    mock_engine.cleanup.assert_called_once()
    assert mock_stage.engine is None
    assert mock_converter.initialized_pipelines == {}


@pytest.mark.fast
def test_reference_numbers_ignores_toc_references_and_resets_on_body_chapters() -> None:
    """A 'References' entry inside the Table of Contents must NOT trigger [1]..[N]
    reference numbering for bullet lists in subsequent body chapters."""
    blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            block_type=BlockType.HEADING,
            source_text="References",
            target_text="参考文献",
            provenance={"toc_entry": True, "toc_page": "79"},
        ),
        IRBlock(
            id="b2",
            spine_index=2,
            block_type=BlockType.HEADING,
            source_text="1. Introduction",
            target_text="1. 引言",
        ),
        IRBlock(
            id="b3",
            spine_index=3,
            block_type=BlockType.LIST_ITEM,
            source_text="Closure: the sequential composition of two effects is again an effect;",
            target_text="封闭性：两个效应的顺序复合仍是一个效应；",
        ),
        IRBlock(
            id="b4",
            spine_index=4,
            block_type=BlockType.HEADING,
            source_text="References",
            target_text="参考文献",
        ),
        IRBlock(
            id="b5",
            spine_index=5,
            block_type=BlockType.LIST_ITEM,
            source_text="E. Moggi, Notions of computation and monads, 1991.",
            target_text="E. Moggi, Notions of computation and monads, 1991.",
        ),
    ]
    ref_map = _reference_numbers(blocks)
    assert "b3" not in ref_map, "Body chapter bullet list item must not be numbered as a reference"
    assert ref_map.get("b5") == "[1]", "Real terminal bibliography entry must be numbered [1]"


@pytest.mark.fast
def test_toc_entry_renders_with_leader_dots_and_page_number() -> None:
    """TOC entries (provenance.toc_entry=True) must render with leader dots and right-aligned
    toc_page instead of giant chapter headings."""
    reconstructor = TypstReconstructor(target_lang="zh")
    toc_block = IRBlock(
        id="toc_1",
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text="2. Preliminaries",
        target_text="2. 预备知识",
        provenance={"toc_entry": True, "toc_page": "7"},
    )
    lines: list[str] = []
    reconstructor._emit_block(toc_block, lines, bilingual=True)
    joined = "\n".join(lines)
    assert "repeat[.]" in joined
    assert "7" in joined
    assert "== 2. 预备知识" not in joined


@pytest.mark.fast
def test_prose_to_typst_handles_spaced_inline_latex_and_single_digit_math() -> None:
    """Spaced inline LaTeX like '$ \\Gamma $' or '$ \\Gamma \\to \\Gamma $' and '$0$' must
    not leak raw escaped dollar signs ('\\$') into Typst output."""
    text = "其中 $ \\Gamma $ 为上下文类型，栖居于 $ \\Gamma \\to \\Gamma $ 中：$0$ 表示未使用，$1$ 表示线性使用。"
    out = _prose_to_typst(text)
    assert "\\$" not in out, f"Leaked escaped dollar sign in Typst output: {out!r}"
    assert "Gamma" in out


@pytest.mark.fast
def test_resolve_overlapping_formula_blocks_merges_vertical_overlap() -> None:
    """Two consecutive FORMULA blocks on the same page whose bounding boxes overlap
    vertically (e.g. equation + commutative diagram) must merge into a single union block."""
    f1 = IRBlock(
        id="pdf_main#b0148",
        spine_index=148,
        block_type=BlockType.FORMULA,
        flow_id=FlowID.MAIN_STORY,
        source_text=r"\Pr_1 \circ f' = f \circ \Pr_1",
        skip_translate=True,
        bbox=BoundingBox(page=10, x0=200.0, y0=410.0, x1=430.0, y1=490.0),
    )
    f2 = IRBlock(
        id="pdf_main#b0149",
        spine_index=149,
        block_type=BlockType.FORMULA,
        flow_id=FlowID.MAIN_STORY,
        source_text=r"\begin{array}{ccc} \Gamma \xrightarrow{f} \Gamma \end{array}",
        skip_translate=True,
        bbox=BoundingBox(page=10, x0=210.0, y0=360.0, x1=390.0, y1=455.0),
    )
    merged = resolve_overlapping_formula_blocks([f1, f2])
    assert len(merged) == 1
    assert merged[0].bbox == BoundingBox(page=10, x0=200.0, y0=360.0, x1=430.0, y1=490.0)


@pytest.mark.fast
def test_normalize_academic_pdf_math_heals_soft_hyphen_word_splits() -> None:
    """Soft hyphens followed by spaces ('transfor\\xad mation', 'compos\\xad ability')
    must be rejoined into whole words."""
    raw = "spatiotemporal compos\xad ability and transfor\xad mation in orches\xad trate"
    cleaned = normalize_academic_pdf_math(raw)
    assert cleaned == "spatiotemporal composability and transformation in orchestrate"


@pytest.mark.fast
@pytest.mark.asyncio
async def test_forced_reflow_on_formula_dense_pdf_enables_companion_rigid_delivery(
    tmp_path: Path,
) -> None:
    """When a user forces --render-engine reflow --dual-mode inline on a formula-dense PDF
    where auto dispatch would select 'rigid' (and advisory tier is 'discourage'),
    run_mode_advisory_stage must schedule emit_secondary_engine='rigid' so export
    delivers a companion *_rigid.pdf alongside the requested reflow PDF."""
    from ubt.core.config import UBTConfig
    from ubt.core.engine.stages.advisory import run_mode_advisory_stage
    from ubt.core.ir.models import BookManifest

    blocks = [
        IRBlock(
            id=f"b{i}",
            spine_index=i,
            block_type=BlockType.FORMULA if i % 3 == 0 else BlockType.NARRATIVE,
            source_text=r"x^2 + y^2 = z^2" if i % 3 == 0 else "Short fragment",
            target_text=r"x^2 + y^2 = z^2" if i % 3 == 0 else "短片段",
            skip_translate=(i % 3 == 0),
            bbox=BoundingBox(page=1, x0=50.0, y0=100.0 + i * 20, x1=400.0, y1=115.0 + i * 20),
        )
        for i in range(1, 16)
    ]
    manifest = BookManifest(doc_id="doc1", title="Test", source_path=str(tmp_path / "paper.pdf"))
    config = UBTConfig(render_engine="reflow", dual_mode="inline")

    ctx = MagicMock()
    ctx.config = config
    ctx.manifest = manifest
    ctx.source_pdf_path = tmp_path / "paper.pdf"
    ctx.input_path = tmp_path / "paper.pdf"
    ctx.profile_name = "paper"
    ctx.job_id = "job_test"

    async def _current_blocks(force_refresh: bool = False) -> list[IRBlock]:
        return blocks

    async def _create_event(*args: object, **kwargs: object) -> object:
        return MagicMock()

    ctx.current_blocks = _current_blocks
    ctx.create_event = _create_event

    events = [ev async for ev in run_mode_advisory_stage(ctx)]
    assert len(events) == 1
    assert manifest.run.emit_secondary_engine == "rigid"


@pytest.mark.fast
def test_detect_math_density_recognizes_unicode_type_theory_and_greek_math() -> None:
    """detect_math_density must flag Unicode math (Greek letters, turnstile ⊢,
    tensor ⊗, arrows →, Theorem/Definition)."""
    from ubt.core.archetype import MathDensity, detect_math_density

    sample = (
        "1. Introduction\nWe study spatiotemporal composability.\n"
        "Definition 2.1. A context transformation Γ ⊢ M : A ⊗ B → C ⊸ D satisfies "
        "f ∘ g = id_Γ for all α, β ∈ Φ(Γ) and ∀x ∈ Δ, ρ(x) ≤ σ(x)."
    )
    assert detect_math_density(sample) == MathDensity.HIGH


@pytest.mark.fast
def test_cli_preflight_guardrail_warns_and_interrupts_on_forced_reflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When --render-engine reflow is passed on a PDF where DocumentAdvisor recommends rigid,
    the CLI must print the Pre-Flight warning panel and interrupt for confirmation when interactive."""
    from typer.testing import CliRunner

    from ubt.cli.main import app
    from ubt.tui.advisor import DocumentAdvisor

    pdf_file = tmp_path / "dense_paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%fake\n")

    fake_adv = MagicMock()
    fake_adv.recommended_render_engine = "rigid"
    fake_adv.math_density = "high"
    fake_adv.category = "academic_paper"
    fake_adv.check_conflict.return_value = ["Forced reflow on rigid-recommended document"]

    monkeypatch.setattr(DocumentAdvisor, "analyze", staticmethod(lambda _p: fake_adv))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    runner = CliRunner()
    # Simulate user typing '3' (Abort) at the interactive interruption prompt
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
        ],
        input="3\n",
    )
    assert result.exit_code != 0
    assert "排版风险预警" in result.output or "Pre-Flight Layout Tradeoff" in result.output


@pytest.mark.fast
def test_glue_run_and_extract_lines_cache_fast_on_dense_fragments() -> None:
    """_glue_run must precompute dehyph() in O(K) instead of O(K^2) inner-loop regex calls."""
    import time

    from ubt.adapters.pdf.textgeom import LineBox, _glue_run

    run = [
        LineBox(f"fragment_{i}_with_some_text", (float(i * 2), 100.0, float(i * 2 + 10), 110.0))
        for i in range(600)
    ]
    t0 = time.perf_counter()
    glued = _glue_run(run)
    elapsed = time.perf_counter() - t0
    assert glued.text
    assert elapsed < 0.25, f"_glue_run took {elapsed:.3f}s on 600 fragments (expected < 0.25s)"


@pytest.mark.fast
def test_rigid_candidate_page_numbers_scopes_to_block_pages() -> None:
    """_candidate_rigid_pages must only select block pages plus +1/+2 continuation lookahead
    when rendering a small pre-flight sample instead of all pages of a long PDF."""
    from ubt.adapters.pdf.rigid.typesetter import _candidate_rigid_pages

    sample_blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Hello",
            target_text="你好",
            bbox=BoundingBox(page=1, x0=50.0, y0=500.0, x1=300.0, y1=520.0),
        )
    ]
    pages = _candidate_rigid_pages(total_pages=15, blocks=sample_blocks)
    assert pages == [1, 2, 3]


@pytest.mark.fast
def test_partition_render_skips_separates_intentional_preserved_from_fail_closed() -> None:
    """_partition_render_skip_counts must separate intentional preserved elements
    (policy, non_prose, chrome, footer) from true fail-closed skips (spill, no_zone, math_unrenderable)."""
    from ubt.core.engine.stages.export import _partition_render_skip_counts

    checkpoints = (
        [{"block_id": f"p{i}", "error_flags": ["render_skip:policy"]} for i in range(56)]
        + [{"block_id": f"n{i}", "error_flags": ["render_skip:non_prose"]} for i in range(14)]
        + [{"block_id": "c1", "error_flags": ["render_skip:chrome"]}]
        + [{"block_id": "c2", "error_flags": ["render_skip:chrome"]}]
        + [{"block_id": "f1", "error_flags": ["render_skip:footer"]}]
        + [
            {"block_id": "fc1", "error_flags": ["render_skip:no_zone"]},
            {"block_id": "fc2", "error_flags": ["render_skip:math_unrenderable"]},
            {"block_id": "fc3", "error_flags": ["render_skip:spill"]},
        ]
    )
    fail_closed, preserved = _partition_render_skip_counts(checkpoints)
    assert fail_closed == 3
    assert preserved == 73


@pytest.mark.fast
@pytest.mark.asyncio
async def test_openai_responses_transport_caches_reasoning_fallback_after_first_400() -> None:
    """Once a model on /responses returns 400 for reasoning_effort and succeeds on retry,
    subsequent calls for that model must use the working reasoning format on the first request."""
    from ubt.core.router.transports.openai_responses import OpenAIResponsesTransport

    transport = OpenAIResponsesTransport(
        api_key="test",
        base_url="https://opencode.ai/zen/go/v1",
    )
    sent_payloads: list[dict[str, object]] = []

    async def fake_request_json(_client: object, _url: str, payload: dict[str, object]) -> object:
        sent_payloads.append(dict(payload))
        resp = MagicMock()
        if "reasoning_effort" in payload:
            resp.status_code = 400
            resp.text = '{"error": "unrecognized field reasoning_effort"}'
        else:
            resp.status_code = 200
            resp.json.return_value = {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "translated text"}],
                    }
                ]
            }
        return resp

    transport._request_json = fake_request_json  # type: ignore[assignment]

    out1, _ = await transport._generate_responses_meta(
        prompt="hi",
        system_prompt=None,
        target_model="muse-spark-1.3-contributor",
        temperature=0.1,
        max_tokens=100,
        reasoning_effort="low",
    )
    assert out1 == "translated text"
    assert len(sent_payloads) == 2

    out2, _ = await transport._generate_responses_meta(
        prompt="hello",
        system_prompt=None,
        target_model="muse-spark-1.3-contributor",
        temperature=0.1,
        max_tokens=100,
        reasoning_effort="low",
    )
    assert out2 == "translated text"
    assert len(sent_payloads) == 3


@pytest.mark.fast
def test_configure_logging_quiets_pikepdf_and_pdf_oxide_bridge() -> None:
    """pikepdf, pdf_oxide, and tiny_skia C++/Rust logger bridges must be quieted to ERROR level."""
    import logging

    from ubt.core.log_config import setup_logging

    setup_logging()
    assert logging.getLogger("pikepdf").level >= logging.ERROR
    assert logging.getLogger("pikepdf._core").level >= logging.ERROR
    assert logging.getLogger("pdf_oxide").level >= logging.ERROR
    assert logging.getLogger("tiny_skia").level >= logging.ERROR
    assert logging.getLogger("tiny_skia.painter").level >= logging.ERROR
