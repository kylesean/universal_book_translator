"""Unit tests for render-hygiene fixes: native SVG sizing, chrome-video
dedup, footnote-anchor guard (p3/p33 screenshot defects)."""

import subprocess
from pathlib import Path

import pytest

from tests.block_builder import make_test_block
from ubt.adapters.pdf.font_probe import available_font_families, is_cjk_capable
from ubt.adapters.pdf.overlay_text import typst_escape as overlay_escape
from ubt.adapters.pdf.typst_fragments import _escape_typst_markup, _reference_numbers
from ubt.adapters.pdf.typst_reconstructor import (
    TypstReconstructor,
    _image_size_spec,
    _prose_to_typst,
    _svg_native_size,
)
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, FlowID, IRBlock


def _block(
    idx: int,
    block_type: BlockType,
    source: str,
    target: str,
    *,
    y0: float = 400.0,
    skip: bool = False,
    flow: FlowID = FlowID.MAIN_STORY,
) -> IRBlock:
    return make_test_block(
        id=f"t#b{idx:04d}",
        spine_index=idx,
        block_type=block_type,
        flow_id=flow,
        source_text=source,
        target_text=target,
        skip_translate=skip,
        status=BlockStatus.MTQE_PASSED,
        bbox=BoundingBox(page=33, x0=50.0, y0=y0, x1=460.0, y1=y0 + 20.0),
    )


def test_svg_native_size_reads_viewbox(tmp_path: Path) -> None:
    svg = tmp_path / "chip.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="86 281 66 27"></svg>',
        encoding="utf-8",
    )
    assert _svg_native_size(str(svg)) == (66.0, 27.0)
    assert _svg_native_size(str(tmp_path / "missing.svg")) is None
    bad = tmp_path / "bad.svg"
    bad.write_text("not xml", encoding="utf-8")
    assert _svg_native_size(str(bad)) is None


def test_image_size_spec_uses_native_svg_size(tmp_path: Path) -> None:
    svg = tmp_path / "chip.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="86 281 66 27"></svg>',
        encoding="utf-8",
    )
    assert _image_size_spec(str(svg)) == "width: 66pt"
    wide = tmp_path / "wide.svg"
    wide.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 467 57"></svg>',
        encoding="utf-8",
    )
    assert _image_size_spec(str(wide)) == "width: 440pt"
    assert _image_size_spec(str(tmp_path / "ghost.svg")) == "width: 70%"


def test_image_size_spec_reports_an_unreadable_raster(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt image falls back to the default width — but not silently."""
    import logging

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\nnot really a png")
    with caplog.at_level(logging.WARNING, logger="ubt.adapters.pdf.typst_fragments"):
        assert _image_size_spec(str(broken)) == "width: 70%"
    assert any("image size probe failed" in rec.message for rec in caplog.records)


def test_chrome_key_normalizes_digits() -> None:
    from ubt.adapters.pdf.docling_blocks import chrome_key as _chrome_key

    assert _chrome_key("Handbook 02 10") == _chrome_key("Handbook 02 11")
    assert _chrome_key("Prefill") != _chrome_key("Decode")


def test_repeat_handle_keeps_first_drops_rest() -> None:
    from ubt.adapters.pdf.docling_blocks import is_repeat_handle as _is_repeat_handle

    seen: set[str] = set()
    assert _is_repeat_handle("@techNmak", seen) is False
    assert _is_repeat_handle("@techNmak", seen) is True
    assert _is_repeat_handle("Regular heading", seen) is False


def test_bottom_anchor_skips_list_and_verbatim() -> None:
    recon = TypstReconstructor()
    lines: list[str] = []
    recon._emit_interior_page(
        [
            _block(
                1, BlockType.LIST_ITEM, "[1] A. Vaswani et al.", "[1] A. Vaswani et al.", y0=50.0
            ),
            _block(
                2,
                BlockType.NARRATIVE,
                "A. Vaswani et al. arXiv:1706.03762.",
                "A. Vaswani et al. arXiv:1706.03762.",
                y0=50.0,
                skip=True,
            ),
        ],
        lines,
        bilingual=False,
        page_num=33,
    )
    assert not any("8.5pt" in ln for ln in lines), lines


def test_bottom_anchor_still_catches_plain_footnote() -> None:
    """Footnote flow stays anchored even without bottom geometry."""
    recon = TypstReconstructor()
    lines: list[str] = []
    recon._emit_interior_page(
        [
            _block(
                1,
                BlockType.NARRATIVE,
                "Some note.",
                "一些注。",
                y0=400.0,
                flow=FlowID.FOOTNOTE,
            )
        ],
        lines,
        bilingual=False,
        page_num=3,
    )
    assert any("8.5pt" in ln for ln in lines)


def test_bottom_body_text_is_not_anchored() -> None:
    """Body flowing to the page bottom must stay body (no spring, no slate)."""
    recon = TypstReconstructor()
    lines: list[str] = []
    recon._emit_interior_page(
        [_block(1, BlockType.NARRATIVE, "Body reaches bottom.", "正文流到页底。", y0=50.0)],
        lines,
        bilingual=False,
        page_num=3,
    )
    assert not any("#v(1fr)" in ln for ln in lines), lines
    assert not any("64748b" in ln for ln in lines), lines
    assert any("正文流到页底。" in ln for ln in lines)


def test_footer_uses_translated_title() -> None:
    from ubt.adapters.pdf.typst_reconstructor import _footer_display_title

    blocks = [
        _block(1, BlockType.HEADING, "Understanding KV Cache", "理解 KV 缓存"),
        _block(2, BlockType.NARRATIVE, "Body.", "正文。"),
    ]
    assert _footer_display_title(blocks, "Understanding KV Cache", False) == "理解 KV 缓存"
    assert (
        _footer_display_title(blocks, "Understanding KV Cache", True)
        == "理解 KV 缓存 (Understanding KV Cache)"
    )
    assert _footer_display_title([], "Raw Title", False) == "Raw Title"


def test_reference_numbers_map_section_list_items() -> None:
    from ubt.adapters.pdf.typst_reconstructor import _reference_numbers

    blocks = [
        _block(1, BlockType.HEADING, "Intro", "介绍"),
        _block(2, BlockType.LIST_ITEM, "a", "甲"),
        _block(3, BlockType.HEADING, "References", "参考文献"),
        _block(4, BlockType.LIST_ITEM, "Paper One", "论文一"),
        _block(5, BlockType.LIST_ITEM, "Paper Two", "论文二"),
    ]
    mapping = _reference_numbers(blocks)
    assert mapping == {"t#b0004": "[1]", "t#b0005": "[2]"}


def test_reference_numbers_survive_translated_subheading() -> None:
    """Docling emits REFERENCES + a translated sub-heading before the items.

    Resetting on any non-matching heading un-numbers the whole section
    (three user-visible regressions of bullets on the last page); reference
    sections are terminal so numbering must stay on.
    """
    from ubt.adapters.pdf.typst_reconstructor import _reference_numbers

    blocks = [
        _block(1, BlockType.HEADING, "REFERENCES", "参考文献"),
        _block(
            2,
            BlockType.HEADING,
            "Primary papers and current implementation sources",
            "主要论文与当前实现来源",
        ),
        _block(3, BlockType.LIST_ITEM, "A. Vaswani et al.", "A. Vaswani 等。"),
        _block(4, BlockType.LIST_ITEM, "N. Shazeer.", "N. Shazeer。"),
    ]
    assert _reference_numbers(blocks) == {"t#b0003": "[1]", "t#b0004": "[2]"}


def test_reference_items_render_with_labels() -> None:
    recon = TypstReconstructor()
    lines: list[str] = []
    recon._emit_block(
        _block(4, BlockType.LIST_ITEM, "Paper One", "论文一"),
        lines,
        False,
        ref_numbers={"t#b0004": "[1]"},
    )
    assert any('#strong("[1]")' in ln and "论文一" in ln for ln in lines)
    assert not any(ln.startswith("- ") for ln in lines)


def test_cover_page_never_prints_image_asset_path() -> None:
    """Regression: pdf_main#svg_* IMAGE blocks on p1 leaked their absolute
    asset path as cover author text (vec_p1_0.svg defect)."""
    recon = TypstReconstructor()
    title = _block(1, BlockType.HEADING, "Chapter 1", "第一章")
    img = _block(2, BlockType.IMAGE, "/tmp/ubt_assets/vec_p1_0.svg", "/tmp/ubt_assets/vec_p1_0.svg")
    lines: list[str] = []
    recon._emit_cover_page([title, img], lines, bilingual=False)
    joined = "\n".join(lines)
    assert "vec_p1_0" not in joined
    assert "ubt_assets" not in joined
    assert any("Cover image asset unavailable" in ln for ln in lines)


def test_emit_block_missing_image_never_prints_path() -> None:
    recon = TypstReconstructor()
    img = _block(3, BlockType.IMAGE, "/tmp/ubt_assets/missing.svg", "/tmp/ubt_assets/missing.svg")
    lines: list[str] = []
    recon._emit_block(img, lines, False)
    assert "ubt_assets" not in "\n".join(lines)


def test_normalize_cjk_spacing() -> None:
    from ubt.core.cleaners.cjk_spacing import normalize_cjk_spacing

    assert normalize_cjk_spacing("改进， 而不是彻底变革。") == "改进，而不是彻底变革。"
    assert normalize_cjk_spacing("（ 测试 ）") == "（测试）"
    assert normalize_cjk_spacing("FinFET 和 GAA") == "FinFET 和 GAA"
    assert normalize_cjk_spacing("你好世界") == "你好世界"
    assert normalize_cjk_spacing("") == ""
    # A bare \s+ would also match \n and weld two paragraphs together.
    assert normalize_cjk_spacing("第一行\n第二行") == "第一行\n第二行"
    assert normalize_cjk_spacing("第一行 \n第二行") == "第一行\n第二行"
    assert normalize_cjk_spacing("段一。\n\n段二。") == "段一。\n\n段二。"
    # Ideographic space (U+3000) is still normalized.
    assert normalize_cjk_spacing("改进，　而不是") == "改进，而不是"
    # Consecutive spaced CJK ideographs must all be normalized
    assert normalize_cjk_spacing("中 文 书") == "中文书"
    assert normalize_cjk_spacing("一 二 三 四") == "一二三四"


def test_body_font_stack_prefers_cjk_sans(noto_cjk_installed: None) -> None:
    recon = TypstReconstructor()
    src = recon.generate_typst_source([], title="", bilingual=False)
    assert "Noto Sans CJK SC" in src


def test_html_mark_tags_stripped_in_typst_emission() -> None:
    """Verify that quarantine/error <mark> tags in target_text do not leak raw HTML into Typst."""
    recon = TypstReconstructor()
    quarantined = _block(
        1,
        BlockType.NARRATIVE,
        "Original source text",
        '<mark class="ubt-blocked-human" title="Quarantine">【待人工审校】Original source text</mark>',
    )
    lines: list[str] = []
    recon._emit_block(quarantined, lines, bilingual=False)
    out = "\n".join(lines)
    assert "<mark" not in out
    assert "</mark>" not in out
    assert "【待人工审校】Original source text" in out


def test_strip_html_mark_tags_helper() -> None:
    from ubt.core.cleaners.html_sanitizer import strip_html_mark_tags

    raw = '<mark class="ubt-failed-draft" title="error msg">Clean translated text</mark>'
    assert strip_html_mark_tags(raw) == "Clean translated text"
    assert strip_html_mark_tags("plain text without marks") == "plain text without marks"
    assert strip_html_mark_tags("") == ""


def test_html_mark_tags_stripped_in_overlay_escape() -> None:
    from ubt.adapters.pdf.overlay_text import typst_escape

    raw = '<mark class="ubt-failed-draft">Error text [special]</mark>'
    escaped = typst_escape(raw)
    assert "<mark" not in escaped
    assert "</mark>" not in escaped
    assert "Error text \\[special\\]" in escaped


def test_typst_escape_matches_fragment_markup_set() -> None:
    """Overlay escaping must cover the same Typst markup as the fragment
    emitter, including ``~`` (non-breaking space)."""
    from ubt.adapters.pdf.overlay_text import typst_escape

    assert typst_escape("a~b") == "a\\~b"


def test_line_start_markup_escaped_in_both_engines() -> None:
    """A newline before ``=``/``+``/``-``/``N.`` must not re-open Typst markup.

    Regression: both escapers covered their special set everywhere but never a
    line-leading block marker, so a translated paragraph carrying a newline
    before ``= heading`` (etc.) was parsed as a heading/list and silently
    re-laid-out the page.
    """
    from ubt.adapters.pdf.overlay_text import typst_escape
    from ubt.adapters.pdf.typst_fragments import _escape_typst_markup

    expected = {"= h": "\\= h", "+ h": "\\+ h", "- h": "\\- h", "1. h": "1\\. h"}
    for esc in (_escape_typst_markup, typst_escape):
        for src, want in expected.items():
            assert esc(f"body\n{src}") == f"body\n{want}"
        # A decimal at line start is not an enum marker and must be left alone.
        assert esc("body\n12.5 units") == "body\n12.5 units"


def test_page_strict_books_only_the_images_it_could_not_stage(tmp_path: Path) -> None:
    """A rendered figure must not also be reported as a lost asset.

    The skip append sat at the if/else level, so every shipped image was booked
    as ``missing_asset`` too: the quality report stamped render_skip flags on
    figures that were in the PDF, ``--strict`` failed a clean book, and the
    length policy flipped rendered figures on fit-to-page pages to NEEDS_HUMAN.
    """
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
    from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock

    png = tmp_path / "fig.png"
    png.write_bytes(_tiny_png_bytes())
    missing = tmp_path / "gone.png"

    def image_block(idx: int, path: Path) -> IRBlock:
        return IRBlock(
            id=f"ch01#img{idx}",
            spine_index=idx,
            flow_id=FlowID.MAIN_STORY,
            block_type=BlockType.IMAGE,
            source_text=str(path),
            target_text=str(path),
            bbox=BoundingBox(page=2, x0=50.0, y0=500.0, x1=250.0, y1=620.0),
        )

    blocks = [image_block(1, png), image_block(2, missing)]
    recon = TypstReconstructor()
    code = recon.generate_typst_source(blocks, target_lang="zh", page_strict=True)

    assert "#image(" in code
    assert recon.last_image_skips == [("ch01#img2", "missing_asset")], recon.last_image_skips


_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAF"
    "BQH/qZ7l4wAAAABJRU5ErkJggg=="
)


def _tiny_png_bytes() -> bytes:
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQH/qZ7l4wAAAABJRU5ErkJggg=="
    )


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


def test_overlay_renderer_passes_the_target_language_through() -> None:
    """The anchored/overlay path hardcoded zh, so a Korean book lost its spaces twice."""
    from ubt.adapters.pdf.overlay_text import prepare_overlay_text

    korean = "이것은 테스트 입니다."
    assert prepare_overlay_text(korean, target_lang="ko") == korean
    assert prepare_overlay_text("中 文 书", target_lang="zh") == "中文书"


_r0918b_NOTO_CJK_INSTALLED = any(
    "noto" in f.casefold() and is_cjk_capable(f) for f in (available_font_families() or frozenset())
)


@pytest.mark.skipif(
    not _r0918b_NOTO_CJK_INSTALLED,
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
