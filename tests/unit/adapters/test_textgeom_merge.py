from __future__ import annotations

import pytest

from ubt.adapters.pdf.textgeom import LineBox, merge_row_fragments

pytestmark = pytest.mark.fast


def _sidebar() -> LineBox:
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
    rows = [_sidebar()]
    y = 540.0
    for i in range(20):
        rows.append(LineBox(f"body text line {i}", (70.8, y - 8.0, 540.0, y)))
        y -= 12.0
    merged = merge_row_fragments(rows)
    assert len(merged) == 21


def test_normal_row_fragments_still_glue() -> None:
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
        LineBox("2", (121.0, 531.0, 125.0, 537.5)),
    ]
    merged = merge_row_fragments(rows)
    texts = [ln.text for ln in merged]
    assert any("base 2" in t or "base2" in t for t in texts)
    assert "arXiv:2609.32391v2 [cs.AI] 29 Sep 2026" in texts


def test_boundary_overread_folds_a_duplicated_subscript() -> None:
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
    rows = [
        LineBox("increasingly demands dynamic", (71.1, 660.8, 320.4, 672.0)),
        LineBox("composition, where components", (321.9, 660.8, 527.2, 672.0)),
    ]
    merged = merge_row_fragments(rows)
    assert "dynamic composition, where components" in " ".join(ln.text for ln in merged)


def test_boundary_overread_folds_a_partial_word() -> None:
    rows = [
        LineBox("are compared at has to keep them. It is D", (71.2, 630.9, 255.1, 639.2)),
        LineBox("Definition 33", (254.6, 630.9, 316.1, 639.2)),
    ]
    merged = merge_row_fragments(rows)
    assert "It is Definition 33" in " ".join(ln.text for ln in merged)


def test_a_longer_word_extension_is_not_folded() -> None:
    rows = [
        LineBox("how recover", (71.1, 660.8, 300.0, 672.0)),
        LineBox("recovers the context", (301.0, 660.8, 527.2, 672.0)),
    ]
    merged = merge_row_fragments(rows)
    assert "recover recovers the context" in " ".join(ln.text for ln in merged)


def _glyph_row(text: str, *, x: float = 60.0, h: float = 5.0) -> list[LineBox]:
    out: list[LineBox] = []
    for ch in text:
        if ch == " ":
            x += 6.0
            continue
        out.append(LineBox(ch, (x, 700.0, x + 4.0, 700.0 + h)))
        x += 5.0
    return out


def test_a_glyph_row_glues_letters_into_words() -> None:
    merged = merge_row_fragments(_glyph_row("LOOPCD"))
    assert [ln.text for ln in merged] == ["LOOPCD"]


def test_a_glyph_row_keeps_real_spaces() -> None:
    merged = merge_row_fragments(_glyph_row("The Model"))
    assert [ln.text for ln in merged] == ["The Model"]


def test_a_glyph_row_recovers_a_sentence() -> None:
    merged = merge_row_fragments(_glyph_row("Looped Transformers"))
    assert [ln.text for ln in merged] == ["Looped Transformers"]


def test_a_word_row_still_space_separates() -> None:
    rows = [
        LineBox("Continual-learning", (88.2, 541.1, 180.0, 549.0)),
        LineBox("agents", (184.0, 541.1, 230.0, 549.0)),
        LineBox("are systems", (234.0, 541.1, 380.0, 549.0)),
    ]
    merged = merge_row_fragments(rows)
    assert merged[0].text == "Continual-learning agents are systems"
