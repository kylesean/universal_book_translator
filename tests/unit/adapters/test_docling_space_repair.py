"""Docling's dropped inter-word spaces repaired from the page's pdfium lines."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.adapters.pdf.docling_crosscheck import (
    join_witness_lines,
    reinsert_missing_spaces,
    repair_missing_spaces_with_lines,
    witness_line_text,
)
from ubt.adapters.pdf.textgeom import LineBox
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    text: str,
    *,
    page: int = 1,
    block_type: BlockType = BlockType.NARRATIVE,
    x0: float = 50.0,
    y0: float = 600.0,
    x1: float = 300.0,
    y1: float = 700.0,
    skip: bool = False,
) -> IRBlock:
    return IRBlock(
        element=make_element(
            block_type=block_type,
            id=block_id,
            source_text=text,
            spine_index=1,
            bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page),
            skip_translate=skip,
        )
    )


def _lines(lines: list[LineBox]) -> tuple[list[LineBox], tuple[float, float]]:
    return lines, (595.0, 842.0)


# -- the witness itself ------------------------------------------------------- #


def test_a_tracked_label_is_not_split_into_two_words() -> None:
    # pdfium splits "Chronus" into two runs 0.3pt apart on a 22pt label; gluing
    # them with a space (as textgeom does) would give "Chr onus".
    line = LineBox(
        "Chr onus",
        (433.7, 391.7, 470.7, 398.8),
        members=(
            LineBox("Chr", (433.7, 391.7, 450.7, 398.8), font_size=22.0),
            LineBox("onus", (451.0, 391.7, 470.7, 396.7), font_size=22.0),
        ),
        font_size=22.0,
    )
    assert witness_line_text(line) == "Chronus"


def test_a_glued_run_with_a_real_gap_keeps_its_space() -> None:
    line = LineBox(
        "Alpha Beta",
        (100.0, 700.0, 140.0, 710.0),
        members=(
            LineBox("Alpha", (100.0, 700.0, 120.0, 710.0), font_size=10.0),
            LineBox("Beta", (126.0, 700.0, 140.0, 710.0), font_size=10.0),
        ),
        font_size=10.0,
    )
    assert witness_line_text(line) == "Alpha Beta"


def test_witness_lines_join_with_a_space_at_a_line_break() -> None:
    lines = [
        LineBox("DeepSeek Elastic Compute (DSec):", (172.1, 709.1, 424.9, 724.7)),
        LineBox("A Sandbox Infrastructure", (70.6, 689.0, 523.8, 704.7)),
    ]
    assert join_witness_lines(lines) == "DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure"


def test_witness_lines_drop_a_line_break_hyphen() -> None:
    lines = [
        LineBox("sand-", (70.6, 148.5, 526.0, 159.6)),
        LineBox("boxes per day", (70.7, 134.2, 524.4, 145.2)),
    ]
    assert join_witness_lines(lines) == "sandboxes per day"


def test_an_unmapped_hyphen_glyph_at_a_line_end_dehyphenates() -> None:
    # A custom-font hyphen reaches pdfium as a control glyph (\x02); it is a
    # line-break hyphen, so the two halves join without a space.
    lines = [
        LineBox("about 3 million sand\x02", (70.6, 148.5, 526.0, 159.6)),
        LineBox("boxes per day", (70.7, 134.2, 524.4, 145.2)),
    ]
    assert join_witness_lines(lines) == "about 3 million sandboxes per day"


# -- the merge ---------------------------------------------------------------- #


def test_a_dropped_space_between_words_is_re_inserted() -> None:
    assert (
        reinsert_missing_spaces(
            "DeepSeek Elastic Compute (DSec): ASandbox Infrastructure",
            "DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure",
        )
        == "DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure"
    )
    assert reinsert_missing_spaces("WeuseAppArmorprofiles", "We use AppArmor profiles") == (
        "We use AppArmor profiles"
    )


def test_pdfium_punctuation_spacing_is_never_imported() -> None:
    # pdfium widens the space around parentheses and commas; only gaps between
    # two word characters may be filled, so the text stays untouched.
    text = "a filesystem (Gao et al., 2019), designed for reads"
    witness = "a filesystem ( Gao et al. ,2019 ), designed for reads"
    assert reinsert_missing_spaces(text, witness) == text


def test_spaces_are_never_removed() -> None:
    assert reinsert_missing_spaces("A single unit", "Asingle unit") == "A single unit"


def test_a_witness_that_disagrees_is_ignored() -> None:
    assert reinsert_missing_spaces("ASandbox", "A SandboX") is None


def test_a_letter_spaced_witness_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    block = _block("b1", "Data Blocks")
    monkeypatch.setattr(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        lambda _path, _page: _lines([LineBox("D a t a B l o c k s", (70.0, 600.0, 300.0, 700.0))]),
    )
    out = repair_missing_spaces_with_lines([block], Path("/does/not/matter.pdf"))
    assert out[0].source_text == "Data Blocks"
    assert "space_repair" not in out[0].provenance


# -- the driver --------------------------------------------------------------- #


def test_the_title_is_repaired_from_the_page_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    block = _block(
        "b1",
        "DeepSeek Elastic Compute (DSec): ASandbox Infrastructure for Effective Agentic Training",
        block_type=BlockType.HEADING,
        x0=70.0,
        y0=689.0,
        x1=524.0,
        y1=725.0,
    )
    monkeypatch.setattr(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        lambda _path, _page: _lines(
            [
                LineBox("DeepSeek Elastic Compute (DSec):", (172.1, 709.1, 424.9, 724.7)),
                LineBox(
                    "A Sandbox Infrastructure for Effective Agentic Training",
                    (70.6, 689.0, 523.8, 704.7),
                ),
            ]
        ),
    )
    out = repair_missing_spaces_with_lines([block], Path("/does/not/matter.pdf"))

    assert out[0].source_text == (
        "DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training"
    )
    assert out[0].provenance["space_repair"] == "pdfium-line-witness"


def test_a_control_glyph_line_break_is_not_read_as_a_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = _block(
        "b1",
        "Asingle production-scale unit of DSec spans about 3 million sandboxes per day",
    )
    monkeypatch.setattr(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        lambda _path, _page: _lines(
            [
                LineBox(
                    "A single production-scale unit of DSec spans about 3 million sand\x02",
                    (70.0, 660.0, 300.0, 672.0),
                ),
                LineBox("boxes per day", (70.0, 645.0, 300.0, 657.0)),
            ]
        ),
    )
    out = repair_missing_spaces_with_lines([block], Path("/does/not/matter.pdf"))

    assert out[0].source_text == (
        "A single production-scale unit of DSec spans about 3 million sandboxes per day"
    )


def test_a_textless_page_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    block = _block("b1", "ASandbox")
    monkeypatch.setattr(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        lambda _path, _page: _lines([]),
    )
    out = repair_missing_spaces_with_lines([block], Path("/does/not/matter.pdf"))
    assert out[0].source_text == "ASandbox"


def test_a_kept_verbatim_block_is_not_touched() -> None:
    block = _block("b1", "ASandbox", skip=True)
    calls: list[int] = []

    def fake_extract_lines(_path: Path, _page: int) -> tuple[list[LineBox], tuple[float, float]]:
        calls.append(_page)
        return _lines([LineBox("A Sandbox", (70.0, 600.0, 300.0, 700.0))])

    with patch("ubt.adapters.pdf.docling_crosscheck.extract_lines", fake_extract_lines):
        out = repair_missing_spaces_with_lines([block], Path("/does/not/matter.pdf"))

    assert calls == []
    assert out[0].source_text == "ASandbox"
