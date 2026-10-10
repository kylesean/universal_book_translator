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


def test_a_large_heading_glyph_still_bands_with_its_row() -> None:
    # A display title's caps are ~4x the page's median rect height, not the
    # 48x a rotated sidebar is. A 3x tall cut sealed each title glyph into its
    # own band, so the title arrived as 15 fragments ("D", "i", "eco n") that
    # each translated and drew into overlapping boxes (arXiv 2610.02185's
    # title was destroyed). Glyph rows must band by baseline, not by height.
    title = [
        LineBox("D", (61.1, 687.0, 76.8, 705.6)),
        LineBox("e", (77.5, 686.7, 91.1, 701.1)),
        LineBox("c", (91.1, 686.7, 104.4, 701.1)),
        LineBox("o", (104.4, 686.7, 118.4, 701.1)),
        LineBox("d", (118.4, 686.7, 133.4, 705.6)),
        LineBox("i", (133.4, 687.0, 138.8, 706.5)),
        LineBox("n", (138.8, 687.0, 152.4, 701.1)),
        LineBox("g", (152.4, 682.3, 165.9, 701.1)),
    ]
    merged = merge_row_fragments(title)
    assert len(merged) == 1, [ln.text for ln in merged]
    assert merged[0].text == "Decoding"


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


# --------------------------------------------------------------------------- #
# Per-character rows (one pdfium rect per glyph)
# --------------------------------------------------------------------------- #


def _glyph_row(text: str, *, x: float = 60.0, h: float = 5.0) -> list[LineBox]:
    """One glyph per LineBox, ~1pt apart, with a 6pt gap at each space."""
    out: list[LineBox] = []
    for ch in text:
        if ch == " ":
            x += 6.0
            continue
        out.append(LineBox(ch, (x, 700.0, x + 4.0, 700.0 + h)))
        x += 5.0
    return out


def test_a_glyph_row_glues_letters_into_words() -> None:
    # Some PDFs give pdfium one rect per character. Space-joining those bakes
    # "L O O P C D" into the line; the row must join on the inter-glyph gap
    # instead (arXiv 2610.02185: 99% fragment rate, 4 of 20 local documents).
    merged = merge_row_fragments(_glyph_row("LOOPCD"))
    assert [ln.text for ln in merged] == ["LOOPCD"]


def test_a_glyph_row_keeps_real_spaces() -> None:
    merged = merge_row_fragments(_glyph_row("The Model"))
    assert [ln.text for ln in merged] == ["The Model"]


def test_a_glyph_row_recovers_a_sentence() -> None:
    merged = merge_row_fragments(_glyph_row("Looped Transformers"))
    assert [ln.text for ln in merged] == ["Looped Transformers"]


def test_a_word_row_still_space_separates() -> None:
    # The glyph-row join must not touch ordinary word fragments.
    rows = [
        LineBox("Continual-learning", (88.2, 541.1, 180.0, 549.0)),
        LineBox("agents", (184.0, 541.1, 230.0, 549.0)),
        LineBox("are systems", (234.0, 541.1, 380.0, 549.0)),
    ]
    merged = merge_row_fragments(rows)
    assert merged[0].text == "Continual-learning agents are systems"
