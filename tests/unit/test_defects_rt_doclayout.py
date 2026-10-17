"""Regressions for the defects a real paper exposed (PaddleOCR RT-DocLayout).

Each test pins a shape that shipped *wrong* on arXiv 2606.23344 and names the
consequence, so a future tightening of the same rule cannot reopen it:

* two full English paragraphs leaked untranslated because ``looks_like_listing``
  read prose as program syntax;
* a one-off footnote paragraph shipped untranslated because every page footer
  was treated as chrome;
* a Korean word rode inside a Chinese block past every language-identity gate.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# looks_like_listing: prose and equations are not program syntax
# --------------------------------------------------------------------------- #

#: (source, why it is prose) — each was classified as a code listing and so
#: skipped translation entirely, leaving the English paragraph on the page.
_PROSE_NOT_LISTINGS = [
    (
        "Classification and Detection Losses. For category prediction, we adopt "
        "focal loss with 𝛼 = 0.25 and 𝛾 = 2.0 to address class imbalance:",
        "'class imbalance:' matched the loose class-definition form",
    ),
    (
        "where 𝑃 denotes the set of valid upper-triangular pairs (excluding "
        "diagonal), and GCE(𝑧, 𝑡) = (1 - 𝑝𝑞 𝑡 )/𝑞 with 𝑝𝑡 = 𝜎(𝑧) if 𝑡 = 1 else "
        "1 - 𝜎(𝑧), and 𝑞 = 0.7 for robustness.",
        "'else 1' matched the loose statement-with-line-number form",
    ),
]


@pytest.mark.parametrize(("text", "why"), _PROSE_NOT_LISTINGS)
def test_prose_and_equations_are_not_listings(text: str, why: str) -> None:
    from ubt.analyze.structure import looks_like_listing

    assert not looks_like_listing(text), why


#: Real program/algorithm syntax must still be recognized.
_REAL_LISTINGS = [
    "class Foo:",
    "class Foo(Base):",
    "def process(x):",
    "await fetch(url)",
    "from os import path",
    "x ← 3",
    "1  x ← 3\n2  else 5",
]


@pytest.mark.parametrize("text", _REAL_LISTINGS)
def test_real_listings_are_still_recognized(text: str) -> None:
    from ubt.analyze.structure import looks_like_listing

    assert looks_like_listing(text)


# --------------------------------------------------------------------------- #
# Page footer: a one-off footnote paragraph is prose, not chrome
# --------------------------------------------------------------------------- #


def test_a_footnote_paragraph_is_translatable() -> None:
    from ubt.adapters.pdf.docling_parser import _is_prose_footer

    footnote = (
        "RT-DocLayout is released in PaddleOCR under the name PP-DocLayoutV3. "
        "RT-DocLayout is the name used in this academic publication, while "
        "PP-DocLayoutV3 is the name adopted in the PaddleOCR open-source "
        "project. The model weights are publicly available at "
        "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3"
    )
    assert _is_prose_footer(footnote)


@pytest.mark.parametrize(
    "chrome",
    [
        "1",
        "arXiv:2606.23344v1 [cs.CV] 22 Jun 2026",
        "©2022 William Fedus, Barret Zoph and Noam Shazeer.",
        "https://example.com",
        "Submitted 8/21; Revised 3/22; Published 4/22",
        "Handbook of Document Analysis, Second Edition, Springer, 2021, page 42",
    ],
)
def test_page_chrome_is_not_a_footnote(chrome: str) -> None:
    from ubt.adapters.pdf.docling_parser import _is_prose_footer

    assert not _is_prose_footer(chrome)


# --------------------------------------------------------------------------- #
# Foreign-script cross-talk: a wrong-script word in the target
# --------------------------------------------------------------------------- #


def test_a_korean_word_inside_chinese_is_foreign_script() -> None:
    """The measured leak: "查询기반" (query-based) rode in a Chinese block.

    Every density gate passed — the block's CJK share measured 0.82 — so the
    word shipped. Chinese uses no Hangul, so even a single syllable is an error.
    """
    from ubt.core.language_profile import foreign_script_chars

    leaked = "遵循查询기반范式，解码器迭代地优化 𝑁 个对象查询"
    foreign = foreign_script_chars(leaked, target_code="zh")
    assert foreign == "기반"


def test_clean_chinese_is_not_flagged() -> None:
    from ubt.core.language_profile import foreign_script_chars

    clean = "本文提出 RT-DocLayout——一种高度高效的文档布局分析端到端框架"
    assert foreign_script_chars(clean, target_code="zh") == ""


def test_kana_inside_chinese_is_foreign_script() -> None:
    from ubt.core.language_profile import foreign_script_chars

    assert foreign_script_chars("深層学習のモデル", target_code="zh") != ""


def test_korean_target_admits_hangul_but_not_kana() -> None:
    from ubt.core.language_profile import foreign_script_chars

    assert foreign_script_chars("언어 기반 모델", target_code="ko") == ""
    assert foreign_script_chars("언어 기반 モデル", target_code="ko") != ""


def test_the_foreign_script_gate_rejects_the_leak() -> None:
    """End to end through FastPass: the leaked block must fail, not pass."""
    from ubt.core.qe.fast_pass import FastPassFilter

    filt = FastPassFilter()
    source = "Following the query-based paradigm, the decoder iteratively refines N queries."
    target = "遵循查询기반范式，解码器迭代地优化 N 个查询。"
    decision = filt.evaluate(source, target, block_type="narrative")
    assert not decision.passed
    assert "Foreign-script" in decision.reason


# --------------------------------------------------------------------------- #
# Table cell line breaks: a stacked column must not fuse into one cell blob
# --------------------------------------------------------------------------- #


class _Cell:
    """Stand-in for a Docling table cell (the attributes ``table_to_markdown`` reads)."""

    def __init__(self, text: str, row: int, col: int) -> None:
        self.text = text
        self.start_row_offset_idx = row
        self.start_col_offset_idx = col
        self.row_span = 1
        self.col_span = 1


class _Table:
    def __init__(self, grid: list[list[str]]) -> None:
        self.data = type(
            "D",
            (),
            {"grid": [[_Cell(t, r, c) for c, t in enumerate(row)] for r, row in enumerate(grid)]},
        )()


def test_a_stacked_table_cell_keeps_its_line_breaks() -> None:
    """A wrapped cell's lines are a stack, not one sentence.

    Docling emits one text line per wrapped visual row inside a cell. The old
    extraction joined them with a space, so the benchmark table's "Recognition
    Method" column — four model names stacked — read as a single
    "MonkeyOCR [27] MinerU2.5 [5] Dolphin-v2 [28] …" blob and the whole table
    rendered unreadable. ``<br>`` preserves the stack (the renderer splits on it).
    """
    from ubt.adapters.pdf.docling_blocks import table_to_markdown

    table = _Table(
        [
            ["Layout Analysis", "RecognitionMethod"],
            ["DocLayout-YOLO", "MonkeyOCR [27]\nMinerU2.5 [5]\nDolphin-v2 [28]"],
        ]
    )
    md = table_to_markdown(table)
    assert "MonkeyOCR [27]<br>MinerU2.5 [5]<br>Dolphin-v2 [28]" in md
    # The stack is not fused into one space-joined run.
    assert "MonkeyOCR [27] MinerU2.5 [5]" not in md


# --------------------------------------------------------------------------- #
# Figure caption binding and native figure width
# --------------------------------------------------------------------------- #


def test_a_translated_figure_caption_binds_to_its_image() -> None:
    """``图 1 | …`` is a caption, not a standalone paragraph.

    The delimiter regex accepted only ``:``/``.``, so the translator's ``|``
    form missed it; the caption (301 chars) then also cleared the length cap,
    and the image and its caption rendered on separate pages.
    """
    from ubt.render.typst.document_renderer import _CAPTION_RE

    caption = (
        "图 1 | RT-DocLayout 的性能与鲁棒性。我们的方法在文档布局分析领域取得了新的当前最优水平。"
        * 3
    )
    assert _CAPTION_RE.match(caption)
    assert _CAPTION_RE.match("Figure 1 | Performance and Robustness")

    from pathlib import Path

    from ubt.core.ir.models import BlockType, IRBlock, make_element
    from ubt.render.typst.document_renderer import TypstDocumentRenderer

    def img(bid: str) -> IRBlock:
        el = make_element(id=bid, spine_index=1, block_type=BlockType.IMAGE, source_text="logo.png")
        return IRBlock(element=el)

    def para(bid: str, text: str) -> IRBlock:
        el = make_element(id=bid, spine_index=1, block_type=BlockType.NARRATIVE, source_text=text)
        return IRBlock(element=el, target_text=text)

    tmp = Path("/tmp/opencode/ubt_caption_test")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    renderer = TypstDocumentRenderer(assets_root=tmp)
    source, _ = renderer.build_source([img("i1"), para("p1", caption)], tmp / "b", target_lang="zh")
    # The caption is inside the figure, not a stray paragraph after it.
    assert "caption: [" in source


def test_a_small_logo_keeps_its_own_width() -> None:
    """A 121pt logo must not be blown up to 88% of the page (this shipped)."""
    from pathlib import Path

    from ubt.render.typst.document_renderer import TypstDocumentRenderer, _Figure

    tmp = Path("/tmp/opencode/ubt_caption_test")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    renderer = TypstDocumentRenderer(assets_root=tmp)

    small = renderer._figure_markup(_Figure(asset=str(tmp / "logo.png"), width_pt=121.0), tmp)
    assert "width: 121pt" in small
    wide = renderer._figure_markup(_Figure(asset=str(tmp / "logo.png"), width_pt=524.0), tmp)
    assert "width: 88%" in wide


# --------------------------------------------------------------------------- #
# Cross-page span split: a paragraph continuation is not a caption
# --------------------------------------------------------------------------- #


class _Prov:
    def __init__(self, page_no: int, charspan: tuple[int, int]) -> None:
        self.page_no = page_no
        self.charspan = charspan
        self.bbox = None


class _Item:
    def __init__(self, text: str, provs: list[_Prov]) -> None:
        self.text = text
        self.prov = provs


def test_a_page_break_that_mid_sentence_is_not_split() -> None:
    """``…we evaluate RT-`` + ``DocLayout using…`` is one paragraph.

    The tail test (capital + terminal punctuation) passed, so the split fired —
    and the detached fragment then absorbed the neighbouring table's content in
    translation, shipping a table where a sentence belonged (arXiv 2606.23344).
    A head that does not end a sentence keeps the item whole.
    """
    from ubt.adapters.pdf.docling_blocks import split_prov_spans

    text = (
        "Due to the lack of standardized evaluation metrics we evaluate RT-"
        "DocLayout using downstream document parsing benchmarks."
    )
    head_len = len("Due to the lack of standardized evaluation metrics we evaluate RT-")
    item = _Item(text, [_Prov(8, (0, head_len)), _Prov(9, (head_len, len(text)))])
    assert split_prov_spans(item) is None


def test_a_cross_page_caption_is_still_split() -> None:
    """A head that ends a sentence followed by a caption body still splits."""
    from ubt.adapters.pdf.docling_blocks import split_prov_spans

    head = "The dataset is described below."
    tail = "Table 1 shows the comparison across all systems."
    text = head + tail
    item = _Item(text, [_Prov(3, (0, len(head))), _Prov(4, (len(head), len(text)))])
    result = split_prov_spans(item)
    assert result is not None
    got_head, got_tail, _bbox = result
    assert got_head == head
    assert got_tail == tail


# --------------------------------------------------------------------------- #
# Stacked table cell: Docling fuses several visual rows into one cell
# --------------------------------------------------------------------------- #


def test_a_stacked_citation_cell_splits_into_rows() -> None:
    """``MonkeyOCR [27] MinerU2.5 [5]`` is two rows, not one cell."""
    from ubt.adapters.pdf.docling_blocks import _split_stacked_cell

    assert _split_stacked_cell("MonkeyOCR [27] MinerU2.5 [5]") == [
        "MonkeyOCR [27]",
        "MinerU2.5 [5]",
    ]
    assert _split_stacked_cell("0.169 0.177 0.178") == ["0.169", "0.177", "0.178"]
    assert _split_stacked_cell("85.10 85.73") == ["85.10", "85.73"]


def test_prose_with_two_citations_is_not_split() -> None:
    """A sentence citing two works is prose, not a name stack."""
    from ubt.adapters.pdf.docling_blocks import _split_stacked_cell

    assert _split_stacked_cell("SigLIP [12] and CLIP [13]") is None
    assert _split_stacked_cell("Ours [7] and Baseline [8]") is None


def test_a_single_value_cell_is_left_alone() -> None:
    from ubt.adapters.pdf.docling_blocks import _split_stacked_cell

    assert _split_stacked_cell("Dolphin-v2 [28]") is None
    assert _split_stacked_cell("Layout Analysis") is None
    assert _split_stacked_cell("85.10") is None


# --------------------------------------------------------------------------- #
# Reference numbering: Docling strips the [n] off a reference entry
# --------------------------------------------------------------------------- #


def test_a_dropped_reference_number_is_recovered_from_the_page_text() -> None:
    """``[1] Mark Chen, …`` — the number Docling leaves out of ``text``.

    Measured on arXiv 2609.20519: the source page carries 108 markers, every
    ``[n]`` prefix sits immediately before the entry in the page's own text
    layer, and Docling's ``list_item`` arrives with an empty ``marker`` and the
    number already stripped. The delivered bibliography lost its numbering.
    """
    from ubt.adapters.pdf.docling_parser import _recover_reference_marker

    page = "[1] Mark Chen, Jerry Tworek, Heewoo Jun.\n[2] John Yang, Carlos Jimenez."
    assert _recover_reference_marker("Mark Chen, Jerry Tworek, Heewoo Jun.", page) == "[1]"
    assert _recover_reference_marker("John Yang, Carlos Jimenez.", page) == "[2]"


def test_recovery_tolerates_a_re_wrapped_line() -> None:
    """The page text may break a line between words; Docling's text has a space.

    Measured shape: the page layer wraps at the margin, so an entry reads
    "…Michalewski,\nDavid Dohan." while Docling carries the same words separated
    by spaces. Collapsing whitespace on both sides is what made the difference
    between 1/57 and 52/54 recovered on the reference list.
    """
    from ubt.adapters.pdf.docling_parser import _recover_reference_marker

    page = "[7] Maarten Bosma, Henryk Michalewski,\nDavid Dohan."
    assert (
        _recover_reference_marker("Maarten Bosma, Henryk Michalewski, David Dohan.", page) == "[7]"
    )


def test_recovery_returns_none_when_the_entry_is_not_numbered() -> None:
    """A body bullet has no [n]; recovery must not invent one."""
    from ubt.adapters.pdf.docling_parser import _recover_reference_marker

    page = "• Breadth expands hypothesis coverage.\n[1] Mark Chen, Jerry Tworek."
    assert _recover_reference_marker("Breadth expands hypothesis coverage.", page) is None
    assert _recover_reference_marker("Completely unrelated entry text here.", page) is None
