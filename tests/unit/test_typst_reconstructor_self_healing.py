"""Unit tests for TypstReconstructor self-healing and layout/math enhancements.

Covers:
- Typst compile deadlock breaker and unclosed delimiter healing.
- Decorative chapter banner detection & suppression (Issue 1).
- Hierarchical equation numbering & explicit tag preservation (Issue 2).
- Linguistic consistency fallback prioritizing Chinese drafts over raw English (Issue 7).
"""

from pathlib import Path

import pytest

from ubt.adapters.pdf.typst_reconstructor import (
    TypstReconstructor,
    _extract_chapter_number,
    _extract_formula_tag,
    _heal_persistent_comment_error,
    _is_decorative_chapter_banner,
    _resolve_content,
)
from ubt.core.ir import BlockStatus, BlockType, BoundingBox, IRBlock
from ubt.core.ir.models import FlowID


def test_heal_persistent_comment_error_finds_unclosed_quote() -> None:
    lines = [
        "= Chapter 1",
        "Some text before.",
        '$ x = "unclosed quote',  # line 3: unclosed quote
        "// [UBT_SYNTAX_FALLBACK] problematic comment line",  # line 4: reported as error
    ]
    modified = _heal_persistent_comment_error(lines, 3)
    assert modified
    # Check that line 3 (index 2) was nullified
    assert "// [UBT_SYNTAX_FALLBACK_REMOVED] #v(0pt)" in lines[2]
    # Check that line 4 was sanitized
    assert "// [UBT_SYNTAX_FALLBACK_REMOVED] #v(0pt)" in lines[3]


def test_heal_persistent_comment_error_finds_unclosed_delimiter() -> None:
    lines = [
        "= Section 2",
        "$ \\frac{a}{b",  # unclosed brace
        "// [UBT_SYNTAX_FALLBACK] closing brace error",
    ]
    modified = _heal_persistent_comment_error(lines, 2)
    assert modified
    assert "// [UBT_SYNTAX_FALLBACK_REMOVED] #v(0pt)" in lines[1]


def test_compile_pdf_self_healing_breaks_deadlock(tmp_path: Path) -> None:
    source = """#set page(paper: "a4")
$ x = \\"left $
// [UBT_SYNTAX_FALLBACK] $ y = 2 $
= Final section
This should compile successfully.
"""
    recon = TypstReconstructor()
    out_pdf = tmp_path / "self_healed.pdf"
    res = recon.compile_pdf(source, out_pdf)
    assert res.exists()
    assert res.stat().st_size > 0


def test_heal_persistent_comment_error_across_section_heading() -> None:
    """Ensure backward scan does not abort at section headings and correctly nullifies the true culprit."""
    lines = [
        "= Chapter 1",
        '$ x = "unclosed string without end',  # line 2 (index 1): culprit
        "Some prose text.",
        "= Chapter 2 Next Section",  # line 4 (index 3): section heading
        "More prose text.",
        "// [UBT_SYNTAX_FALLBACK] line that failed due to unclosed string above",  # line 6 (index 5)
    ]
    modified = _heal_persistent_comment_error(lines, 5)
    assert modified
    assert "// [UBT_SYNTAX_FALLBACK_REMOVED]" in lines[1]
    assert lines[4] == "More prose text."
    assert lines[3] == "= Chapter 2 Next Section"


def test_extract_chapter_number() -> None:
    assert _extract_chapter_number("第 3 章 紧凑建模") == 3
    assert _extract_chapter_number("第三章 DG-FinFET 模型") == 3
    assert _extract_chapter_number("第十二章 总结") == 12
    assert _extract_chapter_number("Chapter 4 Compact Modeling") == 4
    assert _extract_chapter_number("CHAPTER 10") == 10
    assert _extract_chapter_number("5. 泊松方程求解") == 5
    assert _extract_chapter_number("普通的段落文本") is None


def test_extract_formula_tag() -> None:
    assert (
        _extract_formula_tag(r"\frac{d^2\psi}{dx^2} = \frac{q N_{ch}}{\epsilon_{si}} \tag{3.1}")
        == "3.1"
    )
    assert _extract_formula_tag(r"E = mc^2 \tag{1}") == "1"
    assert _extract_formula_tag(r"E = mc^2 (3.1)") == "3.1"
    assert _extract_formula_tag(r"E = mc^2") is None


def test_extract_formula_tag_appendix_alpha_label() -> None:
    # Docling OCR spaces every token; the appendix A.12a label must survive
    # normalization so it can be reprinted instead of the chapter counter.
    assert _extract_formula_tag(r"T _ { a } = \psi _ { pert } \quad ( A . 1 2 a )") == "A.12a"
    assert _extract_formula_tag(r"x = y \quad ( A . 1 0 )") == "A.10"


def test_extract_formula_tag_rejects_spaced_numeric_and_prose() -> None:
    # Spaced pure numerics stay on the chapter counter (body already matches)
    # and OCR body artifacts like "( a . 1 )" must not become appendix tags.
    assert _extract_formula_tag(r"E = mc^2 ( 3 . 1 4 )") is None
    assert _extract_formula_tag(r"x = y ( a . 1 )") is None


def test_cross_ref_appendix_label_maps_and_rewrites() -> None:
    from ubt.adapters.pdf.typst_reconstructor import _apply_cross_refs, _build_formula_map

    block = IRBlock(
        id="pdf_main#b0169",
        spine_index=1,
        block_type=BlockType.FORMULA,
        source_text=r"T _ { a } = \psi _ { pert } / V _ { t m } ^ { 2 } \quad ( A . 1 2 a )",
    )
    f_map = _build_formula_map([block])
    assert f_map["A.12a"] == "eq-pdf_main-b0169"
    rewritten = _apply_cross_refs("其中见式 (A.12a) 与 Eq. (3.11)。", f_map)
    assert "#ref(<eq-pdf_main-b0169>)" in rewritten


def test_is_decorative_chapter_banner_detection() -> None:
    banner = IRBlock(
        id="banner-1",
        spine_index=0,
        block_type=BlockType.IMAGE,
        source_text="CHAPTER",
        bbox=BoundingBox(page=1, x0=50, y0=700, x1=500, y1=740),
    )
    heading = IRBlock(
        id="h-1",
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text="Chapter 3 Compact Modeling",
        target_text="第 3 章 紧凑建模",
        bbox=BoundingBox(page=1, x0=50, y0=650, x1=500, y1=690),
    )
    blocks = [banner, heading]
    assert _is_decorative_chapter_banner(banner, blocks, 0) is True


def test_decorative_banner_skip_is_recorded_for_render_coverage() -> None:
    recon = TypstReconstructor()
    banner = IRBlock(
        id="banner-1",
        spine_index=0,
        block_type=BlockType.IMAGE,
        source_text="CHAPTER",
        target_text="CHAPTER",
        bbox=BoundingBox(page=1, x0=50, y0=700, x1=500, y1=740),
    )
    heading = IRBlock(
        id="h-1",
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text="Chapter 3 Compact Modeling",
        target_text="第 3 章 紧凑建模",
        bbox=BoundingBox(page=1, x0=50, y0=650, x1=500, y1=690),
    )

    lines: list[str] = []
    recon._generate_flowing([banner, heading], lines, bilingual=False)

    assert ("banner-1", "decorative_banner") in recon.last_image_skips


def test_is_decorative_chapter_banner_preserves_real_figures() -> None:
    fig = IRBlock(
        id="fig-1",
        spine_index=1,
        block_type=BlockType.IMAGE,
        source_text="figure_data.png",
        bbox=BoundingBox(page=1, x0=50, y0=400, x1=400, y1=600),
    )
    caption = IRBlock(
        id="cap-1",
        spine_index=2,
        block_type=BlockType.NARRATIVE,
        source_text="Figure 3.1: DG FinFET architecture.",
        target_text="图 3.1：双栅 FinFET 结构。",
        bbox=BoundingBox(page=1, x0=50, y0=380, x1=400, y1=395),
    )
    heading = IRBlock(
        id="h-1",
        spine_index=0,
        block_type=BlockType.HEADING,
        source_text="3.1 Introduction",
        target_text="3.1 引言",
        bbox=BoundingBox(page=1, x0=50, y0=650, x1=500, y1=690),
    )
    blocks = [heading, fig, caption]
    assert _is_decorative_chapter_banner(fig, blocks, 1) is False


def test_is_decorative_chapter_banner_logs_unreadable_asset(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unreadable asset means "content", not "decorative" — say why."""
    import logging

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\nnot a real png")
    img = IRBlock(
        id="img-1",
        spine_index=1,
        block_type=BlockType.IMAGE,
        source_text="scan.png",
    )
    heading = IRBlock(
        id="h-1",
        spine_index=0,
        block_type=BlockType.HEADING,
        source_text="3.1 Introduction",
        target_text="3.1 引言",
        bbox=BoundingBox(page=1, x0=50, y0=650, x1=500, y1=690),
    )
    blocks = [heading, img]
    with caplog.at_level(logging.WARNING, logger="ubt.adapters.pdf.typst_reconstructor"):
        assert _is_decorative_chapter_banner(img, blocks, 1, asset_path=str(broken)) is False
    assert any("aspect probe failed" in rec.message for rec in caplog.records)


def test_resolve_content_preserves_chinese_draft_when_blocked_human() -> None:
    block = IRBlock(
        id="p-1",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        status=BlockStatus.BLOCKED_HUMAN,
        source_text="Complex theoretical physics narrative that failed automated QE review.",
        draft_text="未能通过自动质量检查的复杂理论物理正文叙述。",
        target_text="",
    )
    effective, target, source = _resolve_content(block)
    assert effective == "未能通过自动质量检查的复杂理论物理正文叙述。"
    assert target == ""
    assert source == "Complex theoretical physics narrative that failed automated QE review."


def test_resolve_content_wraps_missing_draft_in_review_marker() -> None:
    block = IRBlock(
        id="p-2",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        status=BlockStatus.BLOCKED_HUMAN,
        source_text="Completely untranslated paragraph.",
        draft_text="",
        target_text="",
    )
    effective, target, source = _resolve_content(block)
    assert effective == "【待审校: Completely untranslated paragraph.】"


def test_resolve_content_exempts_non_prose_blocks() -> None:
    img_block = IRBlock(
        id="img-1",
        spine_index=0,
        block_type=BlockType.IMAGE,
        source_text="/path/to/diagram.png",
        target_text="",
        draft_text="",
    )
    effective, _, _ = _resolve_content(img_block)
    assert effective == "/path/to/diagram.png"

    formula_block = IRBlock(
        id="eq-1",
        spine_index=1,
        block_type=BlockType.FORMULA,
        source_text=r"E = mc^2",
        target_text="",
        draft_text="",
    )
    effective_eq, _, _ = _resolve_content(formula_block)
    assert effective_eq == r"E = mc^2"


def test_reconstructor_equation_tagging_and_chapter_counter() -> None:
    h_blk = IRBlock(
        id="h-ch3",
        spine_index=0,
        block_type=BlockType.HEADING,
        source_text="Chapter 3 Compact Modeling",
        target_text="第 3 章 紧凑建模",
    )
    eq_blk_tagged = IRBlock(
        id="eq-tagged",
        spine_index=1,
        block_type=BlockType.FORMULA,
        source_text=r"\frac{d^2\psi}{dx^2} = \frac{q N_{ch}}{\epsilon_{si}} \tag{3.1}",
        target_text=r"\frac{d^2\psi}{dx^2} = \frac{q N_{ch}}{\epsilon_{si}} \tag{3.1}",
    )
    eq_blk_untagged = IRBlock(
        id="eq-untagged",
        spine_index=2,
        block_type=BlockType.FORMULA,
        source_text=r"V_{tm} = \frac{k_B T}{q}",
        target_text=r"V_{tm} = \frac{k_B T}{q}",
    )
    blocks = [h_blk, eq_blk_tagged, eq_blk_untagged]

    recon = TypstReconstructor(target_lang="zh-cn")
    typst_code = recon.generate_typst_source(blocks, bilingual=False, page_strict=False)

    assert "#counter(heading).update(3)" in typst_code
    assert "#counter(math.equation).update(0)" in typst_code
    assert 'numbering: _ => "(3.1)"' in typst_code
    assert "<eq-eq-tagged>" in typst_code
    assert "<eq-eq-untagged>" in typst_code


def test_section_headings_do_not_restart_equation_numbering() -> None:
    """Numbered sections repeat the chapter digit; only a chapter change resets.

    chapter-3 §3.1/§3.2 both parsed as chapter 3, so the old unconditional
    reset restarted equation numbering mid-chapter — the printed numbers
    drifted to (3.9)-(3.15) while in-text references still said (3.25)-(3.31).
    """
    h_chapter = IRBlock(
        id="h-ch3",
        spine_index=0,
        block_type=BlockType.HEADING,
        source_text="Chapter 3 Compact Modeling",
        target_text="第 3 章 紧凑建模",
    )
    h_section_31 = IRBlock(
        id="h-31",
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text="3.1 Core model for double-gate FinFETs",
        target_text="3.1 双栅 FinFET 的核心模型",
    )
    h_section_32 = IRBlock(
        id="h-32",
        spine_index=2,
        block_type=BlockType.HEADING,
        source_text="3.2 The unified FinFET compact model",
        target_text="3.2 统一 FinFET 紧凑模型",
    )
    eq_blk = IRBlock(
        id="eq-1",
        spine_index=3,
        block_type=BlockType.FORMULA,
        source_text=r"V_{tm} = \frac{k_B T}{q}",
        target_text=r"V_{tm} = \frac{k_B T}{q}",
    )
    recon = TypstReconstructor(target_lang="zh-cn")
    code = recon.generate_typst_source(
        [h_chapter, h_section_31, eq_blk, h_section_32], bilingual=False, page_strict=False
    )
    assert code.count("#counter(math.equation).update(0)") == 1


def test_decorative_banner_dropped_in_flowing_and_strict_rendering() -> None:
    banner = IRBlock(
        id="banner-img",
        spine_index=0,
        block_type=BlockType.IMAGE,
        source_text="chapter_banner.png",
        target_text="chapter_banner.png",
        bbox=BoundingBox(page=1, x0=50, y0=700, x1=500, y1=730),
    )
    h_blk = IRBlock(
        id="h-ch3",
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text="Chapter 3 Compact Modeling",
        target_text="第 3 章 紧凑建模",
        bbox=BoundingBox(page=1, x0=50, y0=640, x1=500, y1=680),
    )
    p_blk = IRBlock(
        id="p-1",
        spine_index=2,
        block_type=BlockType.NARRATIVE,
        source_text="Introduction to FinFET compact modeling.",
        target_text="FinFET 紧凑建模引言。",
        bbox=BoundingBox(page=1, x0=50, y0=580, x1=500, y1=620),
    )
    blocks = [banner, h_blk, p_blk]
    recon = TypstReconstructor(target_lang="zh-cn")

    code_flowing = recon.generate_typst_source(blocks, bilingual=False, page_strict=False)
    assert 'image("chapter_banner.png"' not in code_flowing
    assert "第 3 章 紧凑建模" in code_flowing

    code_strict = recon.generate_typst_source(blocks, bilingual=False, page_strict=True)
    assert 'image("chapter_banner.png"' not in code_strict
    assert "第 3 章 紧凑建模" in code_strict


# ---------------------------------------------------------------------------
# Formula image fallback: a formula that cannot be typeset is shown as its
# ORIGINAL graphic instead of a wall of raw text.
# ---------------------------------------------------------------------------
# chapter-3 b0040 (Eq. 3.9) used to ship as `cal(E)_("x s")= sqrt( frac( ...`
# — complete but unreadable. Docling's LaTeX for it is brace-unbalanced, so
# neither pandoc nor the regex tier can produce valid Typst and the formula
# legitimately fails conversion; the graphic is what the reader can actually
# use, and it is lossless.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAF"
    "BQH/qZ7l4wAAAABJRU5ErkJggg=="
)


def _formula_block(bid: str = "pdf_main#b0040") -> IRBlock:
    return IRBlock(
        id=bid,
        spine_index=40,
        block_type=BlockType.FORMULA,
        source_text=r"\mathcal { E } _ { x s } = \sqrt { \frac { 2 q n _ { i } } { x }",
        bbox=BoundingBox(page=6, x0=171.0, y0=414.0, x1=456.0, y1=447.0),
    )


def _patch_math_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the owner-only math cache (where fallback crops now live) at tmp."""
    import ubt.adapters.pdf.math_renderer as math_renderer

    cache = tmp_path / "math_cache"
    monkeypatch.setattr(math_renderer, "MATH_CACHE_DIR", cache)
    return cache


def test_formula_fallback_uses_source_graphic_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a source PDF, a failed formula becomes an #image, sized from its bbox."""
    import ubt.adapters.pdf.visual_scalpel as scalpel

    monkeypatch.setattr(scalpel, "crop_ir_block_image", lambda *a, **k: _TINY_PNG_B64)
    cache = _patch_math_cache(monkeypatch, tmp_path)

    rec = TypstReconstructor()
    rec.source_pdf = tmp_path / "source.pdf"
    block = _formula_block()
    lines = [f"$ x = y $  // [formula {block.id}]"]
    rec._degrade_math_line(lines, 0, {block.id: block})

    assert lines[0].startswith("#v(0.4em)#align(center)[#image(")
    # 456.0 - 171.0 = 285pt: the source box width, not the generic 85% raster rule.
    assert "width: 285.0pt" in lines[0]
    assert not lines[0].lstrip().startswith("`")
    assert list(cache.glob("fallback-pdf_main-b0040-*.png"))


def test_formula_fallback_paths_are_content_addressed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Names derive from the crop bytes, not a per-run token.

    Same content re-runs overwrite (the temp dir no longer grows one file
    per degraded formula per run), while two renders differing in content
    still get distinct paths, so concurrent jobs sharing a block id cannot
    ship each other's equation.
    """
    import re

    import ubt.adapters.pdf.visual_scalpel as scalpel

    cache = _patch_math_cache(monkeypatch, tmp_path)
    other_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )

    def _render(b64: str) -> str:
        monkeypatch.setattr(scalpel, "crop_ir_block_image", lambda *a, **k: b64)
        rec = TypstReconstructor()
        rec.source_pdf = tmp_path / "source.pdf"
        lines = ["$ x = y $  // [formula pdf_main#b0040]"]
        rec._degrade_math_line(lines, 0, {"pdf_main#b0040": _formula_block()})
        return re.search(r'#image\("([^"]+)"', lines[0]).group(1)  # type: ignore[union-attr]

    first = _render(_TINY_PNG_B64)
    assert _render(_TINY_PNG_B64) == first  # idempotent, no pile-up
    assert len(list(cache.glob("fallback-*.png"))) == 1
    assert _render(other_png) != first  # different crop, different path


def test_formula_fallback_without_source_pdf_keeps_verbatim_text() -> None:
    """No source PDF (non-PDF input, tests) -> behaviour is unchanged."""
    rec = TypstReconstructor()
    assert rec.source_pdf is None
    block = _formula_block()
    lines = [f"$ x = y $  // [formula {block.id}]"]
    rec._degrade_math_line(lines, 0, {block.id: block})
    assert lines[0].startswith("`") and "mathcal" in lines[0]


def test_formula_fallback_without_bbox_keeps_verbatim_text() -> None:
    rec = TypstReconstructor()
    rec.source_pdf = Path("/nonexistent.pdf")
    block = _formula_block().model_copy(update={"bbox": None})
    lines = [f"$ x = y $  // [formula {block.id}]"]
    rec._degrade_math_line(lines, 0, {block.id: block})
    assert lines[0].startswith("`")


def test_formula_fallback_crop_failure_keeps_verbatim_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ubt.adapters.pdf.visual_scalpel as scalpel

    monkeypatch.setattr(scalpel, "crop_ir_block_image", lambda *a, **k: None)
    rec = TypstReconstructor()
    rec.source_pdf = Path("/nonexistent.pdf")
    block = _formula_block()
    lines = [f"$ x = y $  // [formula {block.id}]"]
    rec._degrade_math_line(lines, 0, {block.id: block})
    assert lines[0].startswith("`")


def test_formula_fallback_width_is_clamped() -> None:
    """A pathological bbox must not overflow the text column."""
    from ubt.adapters.pdf.typst_reconstructor import _FORMULA_FALLBACK_MAX_WIDTH_PT

    assert _FORMULA_FALLBACK_MAX_WIDTH_PT <= 493.0  # A4 text column


def test_emit_formula_line_tags_verbatim_fallback() -> None:
    """A Gate-4-rejected formula must still carry its ``// [formula <id>]`` tag.

    Without the tag the render pass cannot map the line back to its block, so
    the source graphic can never be substituted and the reader gets a wall of
    raw text (chapter-3 b0040 shipped exactly that way).
    """
    from ubt.adapters.pdf.typst_reconstructor import _emit_formula_line

    block = _formula_block()
    line = _emit_formula_line(block, block.source_text)
    assert line.lstrip().startswith("`"), "fixture must fail Gate 4"
    assert f"// [formula {block.id}]" in line


def test_generate_typst_source_substitutes_verbatim_formula_with_graphic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """INTEGRATION: drive the real entry point, never a private method.

    The first version of this fallback was verified by calling
    ``_degrade_math_line`` directly — which the real flow never reaches for a
    formula that Gate 4 refused at emission time, because ``_verify_math_lines``
    only scans ``$`` lines. Unit-testing the helper hid that gap; this test
    drives ``generate_typst_source`` and would have caught it.
    """
    import ubt.adapters.pdf.visual_scalpel as scalpel

    monkeypatch.setattr(scalpel, "crop_ir_block_image", lambda *a, **k: _TINY_PNG_B64)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(TypstReconstructor, "is_compiler_available", lambda self: False)

    rec = TypstReconstructor()
    rec.source_pdf = tmp_path / "source.pdf"
    source = rec.generate_typst_source([_formula_block()], title="t", target_lang="zh")
    lines = source.split("\n")

    assert any("#image(" in ln for ln in lines)
    assert any("width: 285.0pt" in ln for ln in lines)
    # The graphic carries the original equation number, so the counter must be
    # stepped by hand or every later formula is numbered one too low.
    assert any("#counter(math.equation).step()" in ln for ln in lines)
    # Nothing is left as a verbatim text fallback.
    assert not [ln for ln in lines if ln.lstrip().startswith("`") and "// [formula" in ln]


def test_graphic_fallback_keeps_equation_counter_aligned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: the graphic must not shift the numbering of later formulas."""
    import ubt.adapters.pdf.visual_scalpel as scalpel

    monkeypatch.setattr(scalpel, "crop_ir_block_image", lambda *a, **k: _TINY_PNG_B64)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    rec = TypstReconstructor()
    rec.source_pdf = tmp_path / "source.pdf"
    lines = ["$ x = y $  // [formula pdf_main#b0040]"]
    rec._degrade_math_line(lines, 0, {"pdf_main#b0040": _formula_block()})
    assert lines[0].count("#counter(math.equation).step()") == 1


def test_shrink_wrap_keeps_label_inside_and_tag_comment() -> None:
    """Equations in the (78, 96]-glyph band shrink to 8.5pt instead of splitting.

    The label must ride INSIDE the ``#block`` (``#text()`` wrapping breaks
    ``@eq-`` references with "cannot reference styled", and a bare ``[...]``
    block prints literal brackets), and the ``// [formula <id>]`` tag comment
    must survive for the compile-probe pass.
    """
    from ubt.adapters.pdf.typst_math import _math_display_len
    from ubt.adapters.pdf.typst_reconstructor import _emit_formula_line

    inner = "$ " + " + ".join(f"alpha_{i} beta_{i}" for i in range(6)) + " $"
    width = _math_display_len(inner[1:-1])
    assert 78 < width <= 96, f"fixture out of shrink band: {width}"
    block = IRBlock(
        id="pdf_main#b0999",
        spine_index=99,
        block_type=BlockType.FORMULA,
        source_text="x",
        target_text=inner[2:-2],
    )
    line = _emit_formula_line(block, block.target_text or "")
    assert line.startswith("#block[#show math.equation: set text(size: 8.5pt); $")
    assert "<eq-pdf_main-b0999>" in line
    assert line.rstrip().endswith("// [formula pdf_main#b0999]")


def test_fit_equation_has_no_wrapper() -> None:
    """Equations within the body-size budget emit plain ``$`` lines."""
    from ubt.adapters.pdf.typst_reconstructor import _emit_formula_line

    block = IRBlock(
        id="pdf_main#b0998",
        spine_index=98,
        block_type=BlockType.FORMULA,
        source_text="x",
        target_text="Q_(inv) = C_(ox)",
    )
    line = _emit_formula_line(block, block.target_text or "")
    assert line.startswith("$")
    assert "#block" not in line


def test_compiler_version_reports_installed_binary() -> None:
    """Real binary (when present) yields a dotted version string."""
    import shutil
    import subprocess

    if shutil.which("typst") is None:
        pytest.skip("typst binary required")
    rec = TypstReconstructor()
    version = rec.compiler_version()
    assert version is not None
    assert version == rec.compiler_version()
    out = subprocess.run(["typst", "--version"], capture_output=True, text=True, timeout=10)
    assert version in out.stdout


def test_compiler_version_none_for_missing_binary() -> None:
    """Missing binary yields None (never raises, cached)."""
    rec = TypstReconstructor(typst_binary="/nonexistent/ubt-typst-probe")
    assert rec.compiler_version() is None
    assert rec.compiler_version() is None


def test_compiler_version_parses_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """`typst 0.15.1 (...)` parses to `0.15.1`; second call reuses the cache."""
    import subprocess

    calls: list[list[str]] = []

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="typst 0.15.1 (unknown commit)\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    rec = TypstReconstructor()
    assert rec.compiler_version() == "0.15.1"
    assert rec.compiler_version() == "0.15.1"
    assert len(calls) == 1


def test_compiler_version_none_on_garbage_or_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garbage stdout / nonzero exit yield None (honest unknown)."""
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="hello world\n", stderr=""
        ),
    )
    assert TypstReconstructor().compiler_version() is None
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            args=argv, returncode=1, stdout="typst 0.15.1\n", stderr="boom"
        ),
    )
    assert TypstReconstructor().compiler_version() is None


def test_page_strict_keeps_column_reading_order() -> None:
    """Publication reflow must not re-sort an interior page by geometry.

    ``spine_index`` is column-aware; sorting each page by ``-y0`` again put the
    right column's first paragraph between the left column's sentences, which on
    a two-column book scrambles every page of the delivered PDF.
    """
    columns = [
        (1, 60.0, 700.0, "L1 left column first paragraph"),
        (2, 60.0, 620.0, "L2 left column second paragraph"),
        (3, 230.0, 700.0, "R1 right column first paragraph"),
        (4, 230.0, 620.0, "R2 right column second paragraph"),
    ]
    blocks = [
        IRBlock(
            id=f"ch01#b{i:03d}",
            spine_index=i,
            block_type=BlockType.NARRATIVE,
            source_text=text,
            target_text=text,
            bbox=BoundingBox(page=2, x0=x0, y0=y0, x1=x0 + 150.0, y1=y0 + 40.0),
        )
        for i, x0, y0, text in columns
    ]

    code = TypstReconstructor(target_lang="zh-cn").generate_typst_source(
        blocks, bilingual=False, page_strict=True
    )

    order = [code.index(text) for _, _, _, text in columns]
    assert order == sorted(order), "interior page emitted out of reading order"


@pytest.mark.parametrize(
    ("target_lang", "expected"),
    [
        ("zh", "表 3.1"),
        ("zh-tw", "表 3.1"),
        ("ja", "表 3.1"),
        ("ko", "표 3.1"),
        ("en", "Table 3.1"),
        ("fr", "Tableau 3.1"),
        ("de", "Tabelle 3.1"),
        ("es", "Tabla 3.1"),
        ("ru", "Таблица 3.1"),
    ],
)
def test_table_label_follows_the_target_language(target_lang: str, expected: str) -> None:
    """``TABLE 3.1`` was rewritten with a hard-coded Chinese literal.

    A French or English target book therefore shipped a Chinese "表" in its
    table captions, while the figure label beside it was already localized from
    the per-language profile.
    """
    from ubt.adapters.pdf.typst_reconstructor import _polish_target_text

    assert _polish_target_text("TABLE 3.1 Devices.", target_lang).startswith(expected)


def _r0918b_block() -> IRBlock:
    return IRBlock(
        id="b1",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="Hello world.",
        target_text="Bonjour le monde.",
    )


def test_reflow_preamble_puts_the_override_first_and_keeps_the_fallbacks(
    noto_cjk_installed: None,
) -> None:
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor

    with_override = TypstReconstructor()
    with_override.font_family = "My Body Font"
    default = TypstReconstructor()
    doc = with_override.generate_typst_source([_r0918b_block()], target_lang="zh")
    plain = default.generate_typst_source([_r0918b_block()], target_lang="zh")
    line = next(ln for ln in doc.splitlines() if ln.startswith("#set text(font:"))
    assert line.index('"My Body Font"') < line.index('"Noto'), "override must come first"
    assert line.count('"') >= 6, "the language fallback stack was replaced"
    assert "My Body Font" not in plain


def test_reflow_records_the_images_it_could_not_stage(tmp_path: Path) -> None:
    """A dropped figure left only a ``//`` comment in the Typst source.

    Invisible in the PDF, so ``render_coverage`` stayed at 100% and ``--strict``
    passed a book whose figures never shipped. The anchored engine has always
    reported these through ``last_render_skips``; the reflow path now feeds the
    same channel.
    """
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
    from ubt.core.ir.models import BoundingBox, IRBlock

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
