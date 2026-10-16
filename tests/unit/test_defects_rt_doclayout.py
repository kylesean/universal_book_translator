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
