"""merge_row_fragments: row glue must not collapse a page into one line.

pdfium emits one rect per glyph run and merge_row_fragments glues them back
into visual rows. The band-formation pass assumes rows never overlap, but a
fragment far taller than a row (vertical sidebar text, rotated watermarks)
overlaps dozens of rows; if it seeds a band, every row in its y-span joins it
and half the page becomes one glued line — which then steals zones from
neighbouring blocks in the rigid typesetter (arXiv 2609.32391 wiped its
introduction paragraph). These tests pin the seal: super-tall fragments stay
their own line and never absorb normal rows.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.textgeom import LineBox, merge_row_fragments

pytestmark = pytest.mark.fast


def _sidebar() -> LineBox:
    """The real shape: arXiv's rotated identifier column, h = 340.8pt."""
    return LineBox("arXiv:2609.32391v2 [cs.AI] 29 Sep 2026", (24.0, 225.6, 42.2, 566.4))


def test_tall_sidebar_fragment_does_not_swallow_overlapping_rows() -> None:
    rows = [
        LineBox("A continual-learning agent is a system comprising", (70.8, 227.3, 541.3, 235.0)),
        LineBox("habits, and workflow state. Across such a lifecycle", (71.2, 215.4, 540.4, 224.6)),
        _sidebar(),
    ]
    merged = merge_row_fragments(rows)
    assert len(merged) == 3
    by_text = {ln.text: ln.rect for ln in merged}
    assert by_text["arXiv:2609.32391v2 [cs.AI] 29 Sep 2026"] == (24.0, 225.6, 42.2, 566.4)
    assert by_text["A continual-learning agent is a system comprising"] == (
        70.8,
        227.3,
        541.3,
        235.0,
    )


def test_tall_fragment_survives_alongside_a_full_page_of_rows() -> None:
    # A whole middle-of-page column of overlapping-x rows plus the sidebar:
    # nothing may chain into one glued line.
    rows = [_sidebar()]
    y = 540.0
    for i in range(20):
        rows.append(LineBox(f"body text line {i}", (70.8, y - 8.0, 540.0, y)))
        y -= 12.0
    merged = merge_row_fragments(rows)
    assert len(merged) == 21


def test_normal_row_fragments_still_glue() -> None:
    # The regression must not break the pass's purpose: word fragments on one
    # visual row still merge.
    rows = [
        LineBox("Continual-learning", (88.2, 541.1, 180.0, 549.0)),
        LineBox("agents", (184.0, 541.1, 230.0, 549.0)),
        LineBox("are systems of models", (234.0, 541.1, 380.0, 549.0)),
        LineBox("Next row", (88.0, 529.1, 200.0, 536.9)),
    ]
    merged = merge_row_fragments(rows)
    assert len(merged) == 2
    assert merged[0].text == "Continual-learning agents are systems of models"


def test_superscript_attaches_to_its_row_not_a_tall_fragment() -> None:
    rows = [
        _sidebar(),
        LineBox("base", (88.0, 529.1, 120.0, 536.9)),
        LineBox("2", (121.0, 531.0, 125.0, 537.5)),  # small fragment, h < 0.7*med
    ]
    merged = merge_row_fragments(rows)
    texts = [ln.text for ln in merged]
    assert any("base 2" in t or "base2" in t for t in texts)
    assert "arXiv:2609.32391v2 [cs.AI] 29 Sep 2026" in texts


def test_boundary_overread_folds_a_duplicated_subscript() -> None:
    # pdfium's get_text_bounded returns every character whose box *intersects*
    # the query rect, so a run's box reports the subscript glyph on its right
    # edge too, and the subscript's own box reports it again. Gluing both baked
    # "creating 𝑚 𝑚" into the line (arXiv 2608.25512). The duplicate folds away.
    rows = [
        LineBox("writes once the O-Insert creating 𝑚", (235.5, 696.6, 397.7, 704.8)),
        LineBox("𝑚", (397.7, 696.6, 406.7, 701.6)),
        LineBox("has set it empty, and the", (407.0, 693.6, 527.2, 704.8)),
    ]
    merged = merge_row_fragments(rows)
    text = " ".join(ln.text for ln in merged)
    assert "creating 𝑚 𝑚" not in text
    assert "creating 𝑚 has set it empty" in text


def test_adjacent_words_sharing_a_letter_are_not_folded() -> None:
    # "dynamic"|"composition" collide on "c" but are two real words; the fold
    # must not splice them into "dynamicomposition".
    rows = [
        LineBox("increasingly demands dynamic", (71.1, 660.8, 320.4, 672.0)),
        LineBox("composition, where components", (321.9, 660.8, 527.2, 672.0)),
    ]
    merged = merge_row_fragments(rows)
    assert "dynamic composition, where components" in " ".join(ln.text for ln in merged)


def test_boundary_overread_folds_a_partial_word() -> None:
    # The over-read glyph can also be a word's first letter: "It is D" +
    # "Definition 33" (rects overlap by the D) must fold to "It is Definition 33".
    rows = [
        LineBox("are compared at has to keep them. It is D", (71.2, 630.9, 255.1, 639.2)),
        LineBox("Definition 33", (254.6, 630.9, 316.1, 639.2)),
    ]
    merged = merge_row_fragments(rows)
    assert "It is Definition 33" in " ".join(ln.text for ln in merged)


def test_a_longer_word_extension_is_not_folded() -> None:
    # "recover"|"recovers": the right fragment extends (not repeats) the left, so
    # it is not the over-read double-count and must survive.
    rows = [
        LineBox("how recover", (71.1, 660.8, 300.0, 672.0)),
        LineBox("recovers the context", (301.0, 660.8, 527.2, 672.0)),
    ]
    merged = merge_row_fragments(rows)
    assert "recover recovers the context" in " ".join(ln.text for ln in merged)
