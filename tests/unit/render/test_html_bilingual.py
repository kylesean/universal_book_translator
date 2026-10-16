"""Unit tests for the modern scholarly bilingual HTML view."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.render.html_view import compose_html

pytestmark = pytest.mark.fast


def test_bilingual_html_view_generation(tmp_path: Path) -> None:
    el1 = make_element(
        id="b1", spine_index=0, block_type=BlockType.HEADING, source_text="Introduction", level=1
    )
    el2 = make_element(
        id="b2",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="This paper presents the Transformer.",
    )

    blocks = [
        IRBlock(element=el1, target_text="引言"),
        IRBlock(element=el2, target_text="本文提出了 Transformer 架构。"),
    ]

    translations = {"b1": "引言", "b2": "本文提出了 Transformer 架构。"}

    out = tmp_path / "bilingual.html"
    comp = compose_html(blocks, translations, out, title="Attention Paper", bilingual=True)

    text = out.read_text(encoding="utf-8")

    assert comp.output_path == out
    assert '<body class="mode-target">' in text
    assert "ubt-toolbar" in text
    assert "纯译文" in text
    assert "双语堆叠" in text
    assert "左右对照" in text
    assert 'class="bilingual-pair"' in text
    assert '<div class="source-row">' in text
    assert '<div class="target-row">' in text
    assert "This paper presents the Transformer." in text
    assert "本文提出了 Transformer 架构。" in text
    # KaTeX is really integrated: the renderer JS + auto-render contrib load,
    # not just the stylesheet.
    assert "katex.min.js" in text
    assert "auto-render.min.js" in text
    assert "renderMathInElement" in text
    # A theme toggle is present and wired to a data-theme override.
    assert "ubt-theme-btn" in text
    assert "toggleTheme" in text
    assert "data-theme" in text


def test_formula_blocks_emit_katex_markup(tmp_path: Path) -> None:
    """A formula lowers to a ``.math-tex`` span whose text is delimited LaTeX.

    KaTeX auto-render parses the element's text content, so the LaTeX (in
    ``$$...$$``) must be the content and the readable fallback the tooltip.
    """
    el = make_element(
        id="f1",
        spine_index=0,
        block_type=BlockType.FORMULA,
        source_text=r"$E = mc^2$",
    )
    blocks = [IRBlock(element=el, target_text=r"$E = mc^2$")]
    out = tmp_path / "formula.html"
    compose_html(blocks, {"f1": r"$E = mc^2$"}, out, bilingual=True)
    text = out.read_text(encoding="utf-8")
    assert 'class="math-tex"' in text
    # The span content is the delimited LaTeX KaTeX will typeset.
    assert "$$E = mc^2$$" in text
    # The readable fallback rides the title/aria-label.
    assert 'title="E = mc^2"' in text


def test_monolingual_html_view_omits_toolbar_and_pairs(tmp_path: Path) -> None:
    el = make_element(
        id="b1",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="Original English text.",
    )
    blocks = [IRBlock(element=el, target_text="中文翻译文本。")]
    translations = {"b1": "中文翻译文本。"}

    out = tmp_path / "mono.html"
    compose_html(blocks, translations, out, bilingual=False)
    text = out.read_text(encoding="utf-8")

    assert '<div class="ubt-toolbar">' not in text
    assert 'class="bilingual-pair"' not in text
    assert "<p>中文翻译文本。</p>" in text
