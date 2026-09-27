"""Cover policy (deterministic, no heuristics), formula fail-closed rendering,
and prompt-strategy override.

Regression context: chapter-3 (Core model for FinFETs) page 1 — headings plus
body narrative — was misclassified as a book cover by the old block-count
heuristic (<=10 blocks + any heading), pushing page-1 body onto a generated
cover page. Display-math source polluted with Docling ``\\$\\$`` artifacts
passed pandoc through unconverted (exit 0, garbage out) and broke Typst
compilation document-wide.
"""

from __future__ import annotations

import re

from ubt.adapters.pdf import typst_math
from ubt.adapters.pdf.typst_math import (
    _emit_formula_math,
    _has_residual_latex,
    _latex_math_to_typst,
    _latex_math_to_typst_regex,
)
from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor, normalize_cover_mode
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock
from ubt.core.router.capabilities import PromptStrategy
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


def _block(
    spine: int,
    block_type: BlockType,
    source: str,
    target: str | None = None,
    page: int | None = None,
) -> IRBlock:
    bbox = (
        BoundingBox(page=page, x0=0.0, y0=500.0, x1=100.0, y1=520.0) if page is not None else None
    )
    return IRBlock(
        id=f"t#b{spine:04d}",
        spine_index=spine,
        block_type=block_type,
        source_text=source,
        target_text=target if target is not None else source,
        bbox=bbox,
    )


# ---------------------------------------------------------------------------
# Cover policy
# ---------------------------------------------------------------------------


def test_cover_headings_only_page_is_cover() -> None:
    recon = TypstReconstructor()
    page = [
        _block(1, BlockType.HEADING, "CHAPTER", "章"),
        _block(2, BlockType.HEADING, "Core model for FinFETs", "FinFET的核心模型"),
    ]
    assert recon._is_cover_page(page) is True


def test_cover_page_with_body_text_is_interior() -> None:
    """Chapter pages (headings + narrative) must NOT become cover pages."""
    recon = TypstReconstructor()
    page = [
        _block(1, BlockType.HEADING, "CHAPTER", "章"),
        _block(2, BlockType.HEADING, "Core model for FinFETs", "FinFET的核心模型"),
        _block(3, BlockType.NARRATIVE, "In the implementation", "在实现中"),
    ]
    assert recon._is_cover_page(page) is False


def test_cover_empty_page_is_not_cover() -> None:
    assert TypstReconstructor()._is_cover_page([]) is False


def test_cover_empty_body_blocks_do_not_block_cover() -> None:
    recon = TypstReconstructor()
    page = [
        _block(1, BlockType.HEADING, "Title", "标题"),
        _block(2, BlockType.NARRATIVE, "", ""),
    ]
    assert recon._is_cover_page(page) is True


def test_cover_mode_overrides() -> None:
    recon = TypstReconstructor()
    body = [_block(1, BlockType.NARRATIVE, "body", "正文")]
    assert recon._is_cover_page(body, "never") is False
    assert recon._is_cover_page(body, "always") is True
    assert recon._is_cover_page(body, "bogus") is False  # unknown -> auto
    assert normalize_cover_mode("NEVER") == "never"
    assert normalize_cover_mode(None) == "auto"


def test_single_page_doc_with_body_renders_as_content() -> None:
    """A 1-page document must translate as content, not cover-only output."""
    recon = TypstReconstructor()
    blocks = [
        _block(1, BlockType.HEADING, "Brief Note", "简短说明", page=1),
        _block(2, BlockType.NARRATIVE, "Hello world body.", "你好世界正文。", page=1),
    ]
    src = recon.generate_typst_source(blocks, title="Note", page_strict=True)
    assert "#v(3.5cm)" not in src  # no cover page emitted
    assert "你好世界正文。" in src


def test_chapter_page_one_keeps_body_on_page_one() -> None:
    """chapter-3 regression: page-1 body must not be sucked into a cover."""
    recon = TypstReconstructor()
    blocks = [
        _block(1, BlockType.HEADING, "CHAPTER", "第章", page=1),
        _block(2, BlockType.HEADING, "Core model for FinFETs", "FinFET的核心模型", page=1),
        _block(3, BlockType.NARRATIVE, "In the implementation", "在实现中", page=1),
        _block(4, BlockType.NARRATIVE, "Various compact models", "各种紧凑模型", page=2),
    ]
    src = recon.generate_typst_source(blocks, title="第章", page_strict=True)
    assert "#v(3.5cm)" not in src
    assert "在实现中" in src.split("#pagebreak()")[0]


def test_pagebreaks_flag_controls_hard_breaks() -> None:
    """Alternating keeps 1:1 breaks; monolingual reflow must not strand pages."""
    recon = TypstReconstructor()
    blocks = [
        _block(1, BlockType.NARRATIVE, "Page one body.", "第一页正文。", page=1),
        _block(2, BlockType.NARRATIVE, "Page two body.", "第二页正文。", page=2),
    ]
    strict = recon.generate_typst_source(blocks, title="T", page_strict=True)
    assert "#pagebreak()" in strict
    flowing = recon.generate_typst_source(blocks, title="T", page_strict=True, pagebreaks=False)
    assert "#pagebreak()" not in flowing
    assert "第一页正文。" in flowing and "第二页正文。" in flowing


def test_page_strict_pads_a_missing_source_page() -> None:
    """A page with no block must still occupy its page, or every later page shifts.

    Regression: ``_generate_page_strict`` only emitted the pages that carried a
    block, so a blank/image-only page moved every following source page one
    slot earlier and the bilingual alternator paired the wrong pages.
    """
    recon = TypstReconstructor()
    blocks = [
        _block(1, BlockType.NARRATIVE, "Page one body.", "第一页正文。", page=1),
        _block(2, BlockType.NARRATIVE, "Page three body.", "第三页正文。", page=3),
    ]
    strict = recon.generate_typst_source(
        blocks, title="T", page_strict=True, pagebreaks=True, source_page_count=4
    )
    # Page 2 (no block) and page 4 (trailing) are both placeholders.
    assert strict.count("#box(height: 1pt)") == 2
    assert "第一页正文。" in strict and "第三页正文。" in strict
    # Three hard breaks separate the four page slots.
    assert strict.count("#pagebreak()") == 3


def test_page_strict_without_pagebreaks_does_not_pad() -> None:
    """Monolingual reflow (no hard breaks) must not gain blank pages."""
    recon = TypstReconstructor()
    blocks = [
        _block(1, BlockType.NARRATIVE, "Page one body.", "第一页正文。", page=1),
        _block(2, BlockType.NARRATIVE, "Page three body.", "第三页正文。", page=3),
    ]
    flowing = recon.generate_typst_source(
        blocks, title="T", page_strict=True, pagebreaks=False, source_page_count=4
    )
    assert "#box(height: 1pt)" not in flowing


# ---------------------------------------------------------------------------
# Formula fail-closed
# ---------------------------------------------------------------------------


def test_residual_latex_detector() -> None:
    assert _has_residual_latex(r"\frac{a}{b}") is True
    assert _has_residual_latex(r"frac(a, b)") is False
    assert _has_residual_latex("x^2 + y") is False


def test_polluted_display_math_compiles_safe() -> None:
    """Docling ``\\$\\$``-polluted source must not emit raw LaTeX math."""
    polluted = (
        "\\$\\$\\frac { \\partial ^ { 2 } \\psi ( x , y ) } "
        "{ \\partial x ^ { 2 } } = \\frac { q } { \\varepsilon }\\$\\$"
    )
    line = _emit_formula_math(polluted, "b-polluted")
    assert not _has_residual_latex(line), line


def test_pandoc_passthrough_falls_back_to_regex(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pandoc exit-0 passthrough must not poison the primary path."""
    monkeypatch.setattr(
        typst_math,
        "_pandoc_math_to_typst",
        lambda _s: "\\frac{a}{b} \\varepsilon",
    )
    out = _latex_math_to_typst("\\frac{a}{b} \\varepsilon")
    assert out == _latex_math_to_typst_regex("\\frac{a}{b} \\varepsilon")
    assert not _has_residual_latex(out)


def test_unconvertible_formula_falls_back_verbatim() -> None:
    """Unbalanced output must degrade to a verbatim span, never raw math.

    Backslashes inside a backtick span are literal text (compile-safe);
    the residual-LaTeX ban applies only to ``$...$`` math mode.
    """
    line = _emit_formula_math("\\qgzq { \\wobble ((( ", "b-garbage")
    assert line.startswith("`"), line


def test_no_residual_backslash_commands_in_math_mode() -> None:
    line = _emit_formula_math(
        "\\frac { \\partial ^ { 2 } \\psi } { \\partial x ^ { 2 } }", "b-frac"
    )
    assert not re.search(r"\\[a-zA-Z]", line), line


def test_verify_math_lines_keeps_good_degrades_bad() -> None:
    """Batch probe stops at the first error; residue is verified per line."""
    import shutil

    import pytest

    if shutil.which("typst") is None:
        pytest.skip("typst binary unavailable")
    recon = TypstReconstructor()
    lines = [
        "$ frac(a, b) $  // [formula good]",
        "$ cong x $  // [formula bad]",
    ]
    degraded = recon._verify_math_lines(lines, [])
    assert degraded == 1
    assert lines[0].startswith("$ frac(a, b) $")
    assert not lines[1].lstrip().startswith("$")


# ---------------------------------------------------------------------------
# Interior-page hygiene: footnote anchors, page chrome, draft fallback
# ---------------------------------------------------------------------------


def _iblock(
    spine: int,
    block_type: BlockType,
    source: str,
    target: str | None = None,
    draft: str | None = None,
) -> IRBlock:
    return IRBlock(
        id=f"t#b{spine:04d}",
        spine_index=spine,
        block_type=block_type,
        source_text=source,
        target_text=target,
        draft_text=draft,
    )


def test_lone_numeral_body_block_is_dropped() -> None:
    """Chapter numeral '3' as a body paragraph must not strand in output."""
    recon = TypstReconstructor()
    lines: list[str] = []
    recon._emit_block(_iblock(1, BlockType.NARRATIVE, "3", "3"), lines, False)
    assert lines == []
    lines2: list[str] = []
    recon._emit_block(_iblock(1, BlockType.NARRATIVE, "69", "69"), lines2, False)
    assert lines2 == []


def test_heading_numeral_and_real_text_survive() -> None:
    recon = TypstReconstructor()
    lines: list[str] = []
    recon._emit_block(_iblock(1, BlockType.HEADING, "3", "3"), lines, False)
    assert any("3" in ln for ln in lines)
    lines2: list[str] = []
    recon._emit_block(
        _iblock(2, BlockType.NARRATIVE, "3.1 Core model", "3.1 核心模型"), lines2, False
    )
    assert any("3.1 核心模型" in ln for ln in lines2)


def test_failed_block_prefers_draft_over_source() -> None:
    """QE-killed blocks must emit the draft, not raw English source."""
    recon = TypstReconstructor()
    lines: list[str] = []
    recon._emit_block(
        _iblock(
            1,
            BlockType.LIST_ITEM,
            "G.D.J. Smit et al. PSP-based model",
            None,
            draft="G.D.J. Smit 等人基于 PSP 的模型",
        ),
        lines,
        False,
    )
    joined = "\n".join(lines)
    assert "基于 PSP 的模型" in joined
    assert "PSP-based model" not in joined


def test_interior_page_drops_chrome_uses_draft_tags_formula() -> None:
    """Interior path must share _emit_block's fixes (no shadow logic)."""
    recon = TypstReconstructor()
    lines: list[str] = []
    recon._emit_interior_page(
        [
            _iblock(1, BlockType.NARRATIVE, "3", "3"),
            _iblock(
                2,
                BlockType.NARRATIVE,
                "Body in English source.",
                None,
                draft="中文草稿正文。",
            ),
            _iblock(3, BlockType.FORMULA, "\\qgzq { \\wobble ((( ", "\\qgzq { \\wobble ((( "),
        ],
        lines,
        bilingual=False,
        page_num=1,
    )
    joined = "\n".join(lines)
    assert not any(ln.strip() == "3" for ln in joined.splitlines())
    assert "中文草稿正文。" in joined
    assert "Body in English source." not in joined
    # Unconvertible formula degrades to verbatim WITH content, never a
    # silent empty comment (the untagged-shadow-branch defect). The trailing
    # ``// [formula <id>]`` tracking tag is intentional: the render pass uses
    # it to swap in the source graphic (see
    # test_emit_formula_line_tags_verbatim_fallback).
    assert "`" in joined
    assert "qgzq" in joined
    assert "omitted: empty" not in joined


# ---------------------------------------------------------------------------
# Prompt-strategy override
# ---------------------------------------------------------------------------


def test_prompt_override_forces_strategy_but_keeps_extraction() -> None:
    router = ModelRouter(
        provider=MockModelProvider(),
        draft_model="test-mt-model",
        prompt_strategy_override="rich",
    )
    from ubt.core.router.capabilities import ExtractionStrategy, ModelProfile

    router.registry.register(
        ModelProfile(
            model_pattern="test-mt-model",
            prompt_strategy=PromptStrategy.MINIMAL,
            extraction_strategy=ExtractionStrategy.RAW,
            supports_system_prompt=False,
        )
    )
    profile = router._get_profile("test-mt-model")
    assert profile.prompt_strategy == PromptStrategy.RICH
    # Only prompt assembly is overridden; MT output parsing stays RAW.
    assert profile.extraction_strategy == ExtractionStrategy.RAW


def test_prompt_override_auto_keeps_registry() -> None:
    router = ModelRouter(
        provider=MockModelProvider(),
        draft_model="test-mt-model",
        prompt_strategy_override="auto",
    )
    from ubt.core.router.capabilities import ExtractionStrategy, ModelProfile

    router.registry.register(
        ModelProfile(
            model_pattern="test-mt-model",
            prompt_strategy=PromptStrategy.MINIMAL,
            extraction_strategy=ExtractionStrategy.RAW,
            supports_system_prompt=False,
        )
    )
    assert router._get_profile("test-mt-model").prompt_strategy == PromptStrategy.MINIMAL


def test_prompt_override_invalid_is_ignored() -> None:
    router = ModelRouter(
        provider=MockModelProvider(),
        draft_model="test-mt-model",
        prompt_strategy_override="gibberish",
    )
    from ubt.core.router.capabilities import ExtractionStrategy, ModelProfile

    router.registry.register(
        ModelProfile(
            model_pattern="test-mt-model",
            prompt_strategy=PromptStrategy.MINIMAL,
            extraction_strategy=ExtractionStrategy.RAW,
            supports_system_prompt=False,
        )
    )
    assert router.prompt_strategy_override is None
    assert router._get_profile("test-mt-model").prompt_strategy == PromptStrategy.MINIMAL


# ---------------------------------------------------------------------------
# Trailing \text{prose} stripped from formula sources (chapter-3 b0013)
# ---------------------------------------------------------------------------


def test_strip_trailing_text_prose_chapter3() -> None:
    src = (
        r"\frac { q } { \varepsilon _ { \text {ch} } } \left ( N _ { \text {ch} } \right ) "
        r"\\ \text {is the electrostatic potential in the channel } a "
        r"\text { is the magnitude of the }"
    )
    out = typst_math._strip_trailing_text_prose(src)
    assert "electrostatic" not in out and "magnitude" not in out
    assert out.rstrip().endswith(r"\right )")
    assert r"\text {ch}" in out  # subscript tags stay


def test_strip_trailing_text_prose_negatives() -> None:
    assert (
        typst_math._strip_trailing_text_prose(r"\frac{q}{\varepsilon_{\text{ch}}}")
        == r"\frac{q}{\varepsilon_{\text{ch}}}"
    )
    aligned = r"\begin{aligned} a \\ b \end{aligned}"
    assert typst_math._strip_trailing_text_prose(aligned) == aligned
    assert typst_math._strip_trailing_text_prose("y = a x + b") == "y = a x + b"


def test_emit_formula_math_drops_glued_prose() -> None:
    src = r"E = m c^2 \text { where m is the total mass }"
    assert "where" not in typst_math._emit_formula_math(src, "t#b0001")


# ---------------------------------------------------------------------------
# Target polish: FIG labels + ¼ glyph confusion
# ---------------------------------------------------------------------------


def test_polish_fig_label() -> None:
    from ubt.adapters.pdf.typst_reconstructor import _polish_target_text

    assert _polish_target_text("FIG. 3.1 Schematic view.") == "图 3.1 Schematic view."
    assert _polish_target_text("Fig 3.2 Plot.") == "图 3.2 Plot."
    assert _polish_target_text("图 3.1 示意图。") == "图 3.1 示意图。"
    assert _polish_target_text("CONFIGURE the device.") == "CONFIGURE the device."


def test_polish_tilde_mangle_scoped() -> None:
    from ubt.adapters.pdf.typst_reconstructor import _polish_target_text

    assert _polish_target_text("Nch ¼ 15 cm") == "Nch～15 cm"
    assert _polish_target_text("a ¼ share") == "a ¼ share"
    assert _polish_target_text("mix ¼ cup of flour") == "mix ¼ cup of flour"


# ---------------------------------------------------------------------------
# Chapter-number fusion (CHAPTER + "3" -> "CHAPTER 3")
# ---------------------------------------------------------------------------


def test_fuse_chapter_number() -> None:
    from ubt.adapters.pdf.docling_blocks import fuse_chapter_number

    blocks = [
        _block(1, BlockType.HEADING, "CHAPTER"),
        _block(2, BlockType.HEADING, "Core model for FinFETs"),
        _block(3, BlockType.NARRATIVE, "3"),
        _block(4, BlockType.NARRATIVE, "Body text."),
    ]
    out = fuse_chapter_number(blocks)
    assert [b.id for b in out] == ["t#b0001", "t#b0002", "t#b0004"]
    assert out[0].source_text == "CHAPTER 3"


def test_fuse_chapter_number_guards() -> None:
    from ubt.adapters.pdf.docling_blocks import fuse_chapter_number

    no_head = [_block(1, BlockType.NARRATIVE, "Intro"), _block(2, BlockType.NARRATIVE, "3")]
    assert len(fuse_chapter_number(no_head)) == 2
    far_num = [
        _block(1, BlockType.HEADING, "CHAPTER"),
        _block(2, BlockType.NARRATIVE, "a"),
        _block(3, BlockType.NARRATIVE, "b"),
        _block(4, BlockType.NARRATIVE, "c"),
        _block(5, BlockType.NARRATIVE, "3"),
    ]
    assert len(fuse_chapter_number(far_num)) == 5


def test_caption_polish_end_to_end_through_interior_page() -> None:
    """FIG labels must survive the *interior* renderer, not just the helper.

    Regression: polish first landed in ``_emit_block`` (cover path) while
    interior pages render through ``_emit_interior_page`` — captions stayed
    English with all unit tests green.
    """
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor

    recon = TypstReconstructor()
    lines: list[str] = []
    recon._emit_interior_page(
        [_block(1, BlockType.NARRATIVE, "FIG. 3.1", "FIG. 3.1")],
        lines,
        bilingual=False,
        page_num=2,
    )
    assert any("图 3.1" in ln for ln in lines), lines
    assert not any("FIG. 3.1" in ln for ln in lines), lines


def test_polish_arrows_ech_tox() -> None:
    from ubt.adapters.pdf.typst_reconstructor import _polish_target_text

    assert _polish_target_text("In order -> find it") == "In order → find it"
    assert _polish_target_text("x->y in prose") == "x→y in prose"
    assert _polish_target_text("ech 是通道的介电常数") == "εch 是通道的介电常数"
    assert _polish_target_text("the tech exists") == "the tech exists"
    assert _polish_target_text("t ox～1 nm") == "tox～1 nm"
    assert _polish_target_text("at ox carts") == "at ox carts"


def test_strip_leading_text_prose() -> None:
    strip = typst_math._strip_leading_text_prose
    src = r"\text { can be integrated analytically, leading to } I _ { d s } = x"
    assert strip(src) == r"I _ { d s } = x"
    assert strip(r"\text{for } x > 0") == r"\text{for } x > 0"
    assert strip(r"\text{if }x") == r"\text{if }x"
    assert strip("E = m c^2") == "E = m c^2"


def test_verbatim_fallback_shows_cleaned_source() -> None:
    """Fail-loud verbatim must not resurrect already-cleaned debris."""
    src = r"\text { can be integrated analytically, leading to } \frac{a}{b"
    out = typst_math._emit_formula_math(src, "t#b0099")
    assert out.startswith("`")
    assert "can be integrated" not in out
    assert "frac" in out  # the actual conversion-breaker stays visible


def test_degrade_math_line_shows_cleaned_source() -> None:
    """Probe-degraded lines must not resurrect cleaned prose either."""
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor

    recon = TypstReconstructor()
    blk = _block(
        63,
        BlockType.FORMULA,
        r"\text { can be integrated analytically, leading to } \frac{a}{b",
    )
    blk.id = "pdf_main#b0063"
    lines = ["$ frac(a)(b $  // [formula pdf_main#b0063]"]
    recon._degrade_math_line(lines, 0, {"pdf_main#b0063": blk})
    assert lines[0].startswith("`")
    assert "can be integrated" not in lines[0]


def test_verify_math_lines_maps_a_multiline_equation_to_the_right_entry() -> None:
    """A multi-line formula must not shift the probe's error attribution.

    ``_emit_formula_math`` legitimately emits ``\\n`` inside ``$…$`` (author
    ``\\\\`` breaks, split long equations), while the batch probe mapped a
    compiler error's physical line to an entry index assuming one line each.
    With one multi-line entry in front, the *healthy* formula behind the broken
    one was degraded instead — its text replaced by the source graphic.
    """
    import shutil

    import pytest

    if shutil.which("typst") is None:
        pytest.skip("typst binary unavailable")
    recon = TypstReconstructor()
    lines = [
        "$ alpha \\\n  quad beta $  // [formula first]",
        "$ cong x $  // [formula bad]",
        "$ frac(a, b) $  // [formula last]",
    ]
    degraded = recon._verify_math_lines(lines, [])
    assert degraded == 1
    assert lines[0].startswith("$ alpha")
    assert lines[2].startswith("$ frac(a, b) $"), "healthy formula degraded by the offset"
    assert lines[1].lstrip().startswith("`"), "the culprit was not degraded"


def _cover_block(spine: int, block_type: BlockType, source: str, target: str, y0: float) -> IRBlock:
    return IRBlock(
        id=f"cover#b{spine:04d}",
        spine_index=spine,
        block_type=block_type,
        source_text=source,
        target_text=target,
        bbox=BoundingBox(page=1, x0=0.0, y0=y0, x1=100.0, y1=y0 + 20.0),
    )


def test_cover_thresholds_scale_with_source_page_height() -> None:
    """Absolute 300/200pt cuts are A4-specific; on a short page they misclassify.

    A subtitle at y0=250 is 'description' under the absolute cuts but 'subtitle'
    once the cut scales with a 400pt page (0.356*400=142.5).
    """
    recon = TypstReconstructor()
    blocks = [
        _cover_block(1, BlockType.HEADING, "Title", "标题", 380.0),
        _cover_block(2, BlockType.NARRATIVE, "SUB", "副标题", 250.0),
        _cover_block(3, BlockType.NARRATIVE, "DESC", "说明", 150.0),
    ]
    short = recon.generate_typst_source(
        blocks, title="T", page_strict=True, cover_mode="always", source_page_height=400.0
    )
    sub_line = next(ln for ln in short.splitlines() if "副标题" in ln)
    assert "14pt" in sub_line  # subtitle style
    desc_line = next(ln for ln in short.splitlines() if "说明" in ln)
    assert 'style: "italic"' in desc_line  # description style

    # No page height -> A4-absolute fallback keeps 250 as the description.
    fallback = recon.generate_typst_source(blocks, title="T", page_strict=True, cover_mode="always")
    sub_fallback = next(ln for ln in fallback.splitlines() if "副标题" in ln)
    assert 'style: "italic"' in sub_fallback
