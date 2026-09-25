"""Regression guards for the 2026-09-18 second review round.

Each test reproduces a defect that was confirmed by running the real code at
HEAD~ (the command is in the docstring), so it fails if the fix is reverted.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ubt.adapters.pdf.font_probe import available_font_families, is_cjk_capable
from ubt.adapters.pdf.overlay_text import typst_escape as overlay_escape
from ubt.adapters.pdf.typst_fragments import _escape_typst_markup, sanitize_font_family
from ubt.adapters.pdf.typst_reconstructor import _prose_to_typst
from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator, derive_job_id
from ubt.core.ir.models import BlockType, ChapterIR, FlowID, IRBlock
from ubt.core.policy.layout_policy import NON_TEXT_BLOCK_TYPES
from ubt.core.policy.verdict import judge_block
from ubt.core.qe.comet_runner import HeuristicQERunner
from ubt.core.qe.defect_taxonomy import (
    CRITICAL_DEFECT_MARKERS,
    STRUCTURAL_DEFECT_MARKERS,
    is_transient_failure,
)
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.provider import MockModelProvider, OpenAICompatibleProvider
from ubt.core.router.rate_limiter import AdaptiveTokenBucket
from ubt.core.router.router import ModelRouter

SRC_EN = "The quick brown fox jumps over the lazy dog near the river bank."

ECHO_REASON = "Target identical to source"
GRID_REASON = "Table grid mismatch"


def _fp(source_lang: str, target_lang: str) -> FastPassFilter:
    return FastPassFilter(source_lang=source_lang, target_lang=target_lang)


# --- 1. an untranslated block auto-passed for every same-script pair ----------


def test_verbatim_echo_is_rejected_for_same_script_pairs() -> None:
    """Before: ``en->fr``/``de->en``/``ja->zh`` echoes returned passed=True
    with "Flawless", and the quality gate released them as MTQE_PASSED because
    the only identity check lived in a scorer that branch never calls."""
    for src_lang, tgt_lang in (("en", "fr"), ("de", "en"), ("en", "en"), ("ja", "zh")):
        decision = _fp(src_lang, tgt_lang).evaluate(SRC_EN, SRC_EN, block_type=BlockType.NARRATIVE)
        assert not decision.passed, f"{src_lang}->{tgt_lang} shipped an echo: {decision.reason}"
        assert ECHO_REASON in decision.reason


def test_real_translation_still_passes_and_echo_class_is_fabricated() -> None:
    ok = "Le renard brun rapide saute par-dessus le chien paresseux pres de la riviere."
    decision = _fp("en", "fr").evaluate(SRC_EN, ok, block_type=BlockType.NARRATIVE)
    assert decision.passed, decision.reason
    # One rule, one band: ``score_pairs`` used to carry a second copy of the
    # identity check with its own length/format exemptions.
    echo = _fp("en", "fr").evaluate(SRC_EN, SRC_EN, block_type=BlockType.NARRATIVE)
    assert HeuristicQERunner.score_from_decision_reason(echo.reason) == pytest.approx(0.15)
    assert asyncio.run(HeuristicQERunner().score_pairs([{"src": SRC_EN, "mt": SRC_EN}])) == [
        pytest.approx(0.15)
    ]


def test_echo_gate_exempts_the_blocks_that_keep_origin_by_contract() -> None:
    """Verbatim ships must not be routed into repair (which ignores
    ``skip_translate`` and used to leave them stale-FAILED)."""
    assert _fp("en", "fr").evaluate(SRC_EN, SRC_EN, skip_translate=True).passed
    assert _fp("en", "fr").evaluate(SRC_EN, SRC_EN, block_type=BlockType.CODE).passed
    assert _fp("en", "fr").evaluate(SRC_EN, SRC_EN, block_type=BlockType.FORMULA).passed
    # Short and wordless blocks legitimately survive the trip unchanged.
    assert _fp("en", "fr").evaluate("Fig. 3", "Fig. 3", block_type=BlockType.NARRATIVE).passed
    assert _fp("en", "fr").evaluate("1234 5678 9", "1234 5678 9", block_type=BlockType.HEADING)


def test_echo_marker_is_fatal_but_never_a_transient_failure() -> None:
    """The marker must survive any QE threshold *and* must not be re-queued on
    resume: ``is_transient_failure`` matches the lowercase ``untranslated:``
    lifecycle prefix, so a near-miss in casing would silently change the
    resume semantics of every echoed block."""
    flag = f"{ECHO_REASON}: the passage was not translated"
    assert flag in STRUCTURAL_DEFECT_MARKERS or any(m in flag for m in STRUCTURAL_DEFECT_MARKERS), (
        "echo must be fatal"
    )
    assert any(m in flag for m in CRITICAL_DEFECT_MARKERS), "an unrepaired echo is Critical"
    assert not is_transient_failure([flag])


# --- 2. "//" silently deleted the rest of a paragraph in the reflow engine ----


_NOTO_CJK_INSTALLED = any(
    "noto" in f.casefold() and is_cjk_capable(f) for f in (available_font_families() or frozenset())
)


@pytest.mark.skipif(
    not _NOTO_CJK_INSTALLED,
    reason=(
        "round-trips CJK through typst embedding + pdftotext extraction, so it measures "
        "the installed face's ToUnicode as much as the emitter; the golden-adjacent "
        "escape logic is asserted without rendering in the two asserts above"
    ),
)
def test_double_slash_survives_the_reflow_emitter(tmp_path: Path) -> None:
    """Before: ``typst compile`` exited 0 and the text after ``//`` was gone --
    Typst read it as a line comment. The anchored path already guarded with a
    zero-width space; the reflow emitter did not."""
    text = "段落前的文字 // 后半句不能消失"
    escaped = _escape_typst_markup(text)
    assert escaped != text, "// reached Typst unescaped"
    assert "//" not in escaped
    document = _prose_to_typst(text)
    typst = tmp_path / "doc.typ"
    pdf = tmp_path / "doc.pdf"
    typst.write_text(document + "\n", encoding="utf-8")
    subprocess.run(
        ["typst", "compile", str(typst), str(pdf)],
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = subprocess.run(
        ["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True
    ).stdout
    assert "后半句不能消失" in rendered.replace("​", "")


def test_double_slash_guard_has_one_owner_in_both_engines() -> None:
    """Both emitters must break the comment token the same way, or the two
    engines disagree about what a ``//`` means."""
    guard = "/​/"
    assert guard in _escape_typst_markup("a//b")
    assert guard in overlay_escape("a//b")
    # URLs stay readable (one invisible character inside the scheme separator).
    assert "https:/​/example.com" in _escape_typst_markup("see https://example.com/x")


# --- 3. --dry-run poisoned the resumable ledger and the shared TM ------------


def test_mock_run_namespaces_the_derived_ledger_id() -> None:
    """Before: a ``--dry-run`` pass wrote ``mtqe_passed``/``[模拟翻译]`` rows into
    ``job_<doc>_<lang>``, and the next real run made zero API calls, exported
    the mock text and printed "completed"."""

    def derive(**kw: object) -> str:
        base: dict[str, object] = {
            "doc_id": "0" * 64,
            "target_lang": "zh",
            "pages": None,
            "start_chapter": 1,
            "max_chapters": None,
        }
        base.update(kw)
        return derive_job_id(**base)  # type: ignore[arg-type]

    derived = derive()
    assert derived == f"job_{'0' * 12}_zh"
    assert derive(mock_run=True) == f"{derived}_mock"
    # The window suffixes and the mock suffix compose in either order.
    assert derive(pages="3-5", mock_run=True) == f"{derived}_p3_5_mock"


def test_orchestrator_detects_a_simulated_run_from_the_provider() -> None:
    """``--dry-run`` keeps a real ``api_key`` in config and swaps only the
    router's provider, so a config-based check would miss every dry run."""
    limiter = AdaptiveTokenBucket(initial_rpm=60, max_rpm=60)
    mock = ModelRouter(provider=MockModelProvider(), rate_limiter=limiter, draft_model="m")
    real = ModelRouter(
        provider=OpenAICompatibleProvider(api_key="sk-real", base_url="http://127.0.0.1:9/v1"),
        rate_limiter=limiter,
        draft_model="m",
    )
    assert PipelineOrchestrator(config=UBTConfig(), router=mock)._is_mock_run is True
    assert PipelineOrchestrator(config=UBTConfig(), router=real)._is_mock_run is False


# --- 4. font_family was written to an object nobody reads --------------------


def test_font_family_reaches_both_render_engines() -> None:
    """Before: the pipeline assigned ``adapter.font_family`` while the render
    strategy and the emitter each kept their own construction-time ``None``."""
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter

    adapter = DoclingPDFAdapter()
    assert adapter.font_family is None
    adapter.font_family = "Noto Serif Test"
    assert adapter._renderer.font_family == "Noto Serif Test"
    assert adapter.reconstructor.font_family == "Noto Serif Test"

    # A family name is interpolated into ``#set text(font: "...")``.
    assert sanitize_font_family('x") #import "evil') is None
    adapter.font_family = 'x") #import "evil'
    assert adapter.font_family is None
    assert adapter._renderer.font_family is None


def test_reflow_preamble_puts_the_override_first_and_keeps_the_fallbacks(
    noto_cjk_installed: None,
) -> None:
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor

    with_override = TypstReconstructor()
    with_override.font_family = "My Body Font"
    default = TypstReconstructor()
    doc = with_override.generate_typst_source([_block()], target_lang="zh")
    plain = default.generate_typst_source([_block()], target_lang="zh")
    line = next(ln for ln in doc.splitlines() if ln.startswith("#set text(font:"))
    assert line.index('"My Body Font"') < line.index('"Noto'), "override must come first"
    assert line.count('"') >= 6, "the language fallback stack was replaced"
    assert "My Body Font" not in plain


# --- 5. a text adapter happily wrote Markdown into a .pdf name ---------------


def test_adapters_declare_the_suffixes_they_write() -> None:
    from ubt.adapters.base import BaseDocumentAdapter, BasePDFEngineAdapter
    from ubt.adapters.epub.adapter import EPUBAdapter
    from ubt.adapters.markdown.adapter import MarkdownAdapter

    assert BaseDocumentAdapter.output_suffixes == frozenset()
    assert BasePDFEngineAdapter.output_suffixes == frozenset({".pdf"})
    assert MarkdownAdapter.output_suffixes == frozenset({".md", ".markdown", ".txt"})
    assert EPUBAdapter.output_suffixes == frozenset({".epub"})


def test_pipeline_refuses_a_contradicting_output_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.exceptions import UnsupportedDocumentFormatError
    from ubt.core.ports import resolve_adapter

    source = tmp_path / "book.md"
    source.write_text("# Chapter One\n\nHello world.\n", encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(
        "ubt.core.engine.pipeline.resolve_adapter",
        lambda *a, **k: resolve_adapter(*a, **k),
    )
    orchestrator = PipelineOrchestrator(
        config=UBTConfig(db_dir=tmp_path / "db"),
        router=_counting_router(calls),
    )
    with pytest.raises(UnsupportedDocumentFormatError) as exc:
        asyncio.run(_drain(orchestrator, source, tmp_path / "book.pdf"))
    assert ".md" in str(exc.value)
    assert calls == [], "the refusal must happen before the first model call"


# --- 6. every table shipped in the source language stamped 1.0 --------------


def test_tables_are_translation_content_not_verbatim_residue() -> None:
    """The parser marks tables ``skip=False`` and the QE layer requires a
    translated table, but the verdict kept them out of the model entirely while
    ingest stamped them MTQE_PASSED/1.0."""
    assert BlockType.TABLE not in NON_TEXT_BLOCK_TYPES
    verdict = judge_block(_irblock(block_type=BlockType.TABLE))
    assert verdict.translate is True
    for keep in (BlockType.FORMULA, BlockType.CODE, BlockType.IMAGE):
        assert judge_block(_irblock(block_type=keep)).translate is False


def test_translated_table_grid_passes_and_a_mangled_one_does_not() -> None:
    src = "| Cell | Alpha | Beta Value |\n| --- | --- | --- |\n| 12 | 34 | 56 |"
    translated = "| 单元格 | 阿尔法 | Beta 值 |\n| --- | --- | --- |\n| 12 | 34 | 56 |"
    dropped_column = "| 单元格 | 阿尔法 |\n| --- | --- |\n| 12 | 34 |"
    fp = _fp("en", "zh")
    assert fp.evaluate(src, translated, block_type=BlockType.TABLE).passed
    decision = fp.evaluate(src, dropped_column, block_type=BlockType.TABLE)
    assert not decision.passed and GRID_REASON in decision.reason
    assert any(m in decision.reason for m in STRUCTURAL_DEFECT_MARKERS)


# --- 7. the context before a block was the wrong language --------------------


def test_preceding_context_carries_the_translation(tmp_path: Path) -> None:
    """The prose before a block is the prose the model is continuing.

    Both directions of the neighbor window read ``source_text``, so an English
    book being rendered into Chinese was handed English context for a paragraph
    it had already translated: register, term choice and sentence rhythm were
    re-decided per block instead of carried forward. The *following* excerpt must
    stay source-side -- that text has no translation yet.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest

    ledger = SQLiteJobLedger(tmp_path / "ctx.sqlite")
    job_id = "job_ctx"
    ledger.init_job_from_manifest(job_id, BookManifest(doc_id="d1", title="T", source_path="x"))
    blocks = [
        IRBlock(
            id="ch01#b1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            source_text="The elf closed the door quietly.",
            target_text="L'elfe ferma la porte en silence.",
        ),
        IRBlock(
            id="ch01#b2",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            source_text="Nobody heard it.",
        ),
    ]
    ledger.append_chapter(
        job_id, ChapterIR(doc_id="d1", chapter_id="ch01", title="c", spine_index=1, blocks=blocks)
    )

    preceding = ledger.get_preceding_text_tail(
        job_id=job_id, flow_id=FlowID.MAIN_STORY, before_spine_index=2
    )
    assert "L'elfe ferma la porte" in preceding
    assert "closed the door" not in preceding

    following = ledger.get_following_text_head(
        job_id=job_id, flow_id=FlowID.MAIN_STORY, after_spine_index=1
    )
    assert following == "Nobody heard it."

    # Untranslated neighbours still provide their source rather than nothing.
    untranslated = ledger.get_preceding_text_tail(
        job_id=job_id, flow_id=FlowID.MAIN_STORY, before_spine_index=3
    )
    assert "Nobody heard it." in untranslated and "L'elfe" in untranslated
    ledger.close()


def test_in_batch_neighbor_window_prefers_the_finished_target() -> None:
    from ubt.core.memory.neighbor_window import NeighborContextBuilder

    builder = NeighborContextBuilder(neighbor_chars=200)
    prev = IRBlock(
        id="n1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="The elf closed the door quietly.",
        target_text="L'elfe ferma la porte en silence.",
    )
    current = IRBlock(
        id="n2",
        flow_id=FlowID.MAIN_STORY,
        spine_index=2,
        source_text="Nobody heard it.",
    )
    ctx = builder.extract_from_blocks(current, [prev, current])
    assert "L'elfe ferma la porte" in ctx
    assert "PRECEDING CONTEXT" in ctx


# --- 8. the visual gate measured the wrong document for a reflow render ------


def test_source_bboxes_only_gate_the_engine_that_keeps_them() -> None:
    """T1 compares source-page rectangles against the output mediabox.

    Meaningful for the anchored overlay (its canvas is the original page) and
    for the alternating zipper (its pages *are* the source pages); the reflow
    engine rebuilds at A4, so a US-Letter full-bleed table is flagged "major"
    on a perfectly rendered book -- and that finding drives a whole-book re-render
    plus quarantine of the blocks it names.
    """
    from ubt.adapters.pdf.visual_gate import blocks_out_of_bounds_findings
    from ubt.core.engine.reflow_loop import ReflowControlLoop

    letter_block = [
        SimpleNamespace(
            id="tbl1", bbox=SimpleNamespace(page=1, x0=6.0, y0=90.0, x1=606.0, y1=300.0)
        )
    ]
    a4 = {1: (595.276, 841.89)}
    letter = {1: (612.0, 792.0)}
    assert [f.code for f in blocks_out_of_bounds_findings(letter_block, a4)] == [
        "block_out_of_bounds"
    ], "the false positive this guard exists for"
    assert blocks_out_of_bounds_findings(letter_block, letter) == []

    def loop_with(metadata: object, run_engine: str | None = None) -> ReflowControlLoop:
        return cast(
            "ReflowControlLoop",
            SimpleNamespace(
                manifest=SimpleNamespace(
                    metadata=metadata,
                    run=SimpleNamespace(render_engine_effective=run_engine),
                )
            ),
        )

    keeps = ReflowControlLoop._output_keeps_source_geometry
    # The typed run field is the source of truth; the metadata copy is the
    # fallback for manifests that predate it (or never had it as a dict).
    assert keeps(loop_with({"render_engine_effective": "publication"})) is False
    assert keeps(loop_with({"render_engine_effective": "rigid"})) is True
    assert keeps(loop_with(None)) is True
    assert keeps(loop_with({})) is True
    # The typed field wins over the metadata copy, and a typed "publication"
    # is honored even when metadata is missing entirely.
    assert keeps(loop_with({"render_engine_effective": "publication"}, run_engine="rigid")) is True
    assert keeps(loop_with(None, run_engine="publication")) is False


# --- 9. the fitter measured a font the page does not use ---------------------


def test_width_metrics_face_follows_the_render_font(monkeypatch: pytest.MonkeyPatch) -> None:
    """``load_width_font`` used to raise unless the TTC held one exact face.

    ``UBT_CJK_FONT`` pointing at a Serif collection -- a path
    ``resolve_cjk_ttc`` itself recommends and ``ubt doctor`` reports as OK --
    aborted the whole anchored render over a *measurement* font, and a
    configured ``font_family`` was measured with Sans widths anyway.
    """
    import fontTools.ttLib as ttlib

    from ubt.adapters.pdf.font_metrics import load_width_font

    def install(*names: str) -> None:
        class _Collection:
            def __init__(self, _path: str) -> None:
                # fontTools fonts are dict-like: the loader reads font["name"].
                self.fonts = [
                    {"name": SimpleNamespace(getDebugName=lambda _i, n=n: n)} for n in names
                ]

        monkeypatch.setattr(ttlib, "TTCollection", _Collection)

    # The requested family is the one the page renders in, so measure it.
    install("Noto Serif CJK SC", "Noto Sans CJK TC")
    assert _face(load_width_font("x.ttc", "Noto Serif CJK SC")) == "Noto Serif CJK SC"
    # Nothing resembles the request: measure the first face instead of failing.
    assert _face(load_width_font("x.ttc", "Kaiti SC")) == "Noto Serif CJK SC"
    assert _face(load_width_font("x.ttc")) == "Noto Serif CJK SC"
    # Packaging renames still resolve ("NotoSerifCJKsc-Regular" is a Serif face).
    install("NotoSerifCJKsc-Regular")
    assert _face(load_width_font("x.ttc", "Noto Serif CJK SC")) == "NotoSerifCJKsc-Regular"
    install()
    with pytest.raises(Exception, match="No font face inside"):
        load_width_font("x.ttc")


def _face(font: Any) -> str:
    return str(font["name"].getDebugName(4))


# --- helpers ----------------------------------------------------------------


def _irblock(**kwargs: object) -> IRBlock:
    base: dict[str, object] = {
        "id": "t1",
        "spine_index": 0,
        "source_text": "| Cell | Alpha | Beta |\n| --- | --- | --- |\n| 1 | 2 | 3 |",
    }
    base.update(kwargs)
    return IRBlock(**base)  # type: ignore[arg-type]


def _block() -> IRBlock:
    return IRBlock(
        id="b1",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="Hello world.",
        target_text="Bonjour le monde.",
    )


def _counting_router(calls: list[object]) -> ModelRouter:
    class _Counting(MockModelProvider):
        async def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> str:
            calls.append(prompt)
            return "[TRANSLATED]"

    return ModelRouter(
        provider=_Counting(),
        rate_limiter=AdaptiveTokenBucket(initial_rpm=60, max_rpm=60),
        draft_model="counting",
    )


async def _drain(orchestrator: PipelineOrchestrator, source: Path, out: Path) -> None:
    async for _ in orchestrator.run(
        input_path=source, output_path=out, target_lang="fr", source_lang="en"
    ):
        pass


# --- 10. a reflow render dropped an image and reported full coverage ------


def test_reflow_records_the_images_it_could_not_stage(tmp_path: Path) -> None:
    """A dropped figure left only a ``//`` comment in the Typst source.

    Invisible in the PDF, so ``render_coverage`` stayed at 100% and ``--strict``
    passed a book whose figures never shipped. The anchored engine has always
    reported these through ``last_render_skips``; the reflow path now feeds the
    same channel.
    """
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
    from ubt.core.ir.models import BoundingBox, FlowID, IRBlock

    reconstructor = TypstReconstructor()
    image_block = IRBlock(
        id="pg1#img9",
        spine_index=1,
        flow_id=FlowID.MAIN_STORY,
        block_type=BlockType.IMAGE,
        source_text="Figure 1: architecture",
        bbox=BoundingBox(page=1, x0=0.0, y0=0.0, x1=100.0, y1=100.0),
    )
    reconstructor.generate_typst_source([image_block], target_lang="zh")
    assert reconstructor.last_image_skips == [("pg1#img9", "missing_asset")]

    # A second render must not inherit the first one's ledger.
    reconstructor.generate_typst_source([image_block], target_lang="zh")
    assert len(reconstructor.last_image_skips) == 1
