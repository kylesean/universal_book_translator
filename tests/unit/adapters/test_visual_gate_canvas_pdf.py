from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ubt.adapters.pdf.visual_gate import CANVAS_COLLISION_RATIO, run_visual_gate

pytestmark = pytest.mark.fast

_WIDTH = 612.0
_HEIGHT = 792.0
_MARGIN = 54.0
_LEADING = 16.0
_FONT_SIZE = 12.0
_LINES = 6

_RULE_THROUGH_TEXT = 4.0
_RULE_IN_GAP = -5.0


def _write_page(
    path: Path,
    *,
    text: bool,
    rule_offset: float | None = None,
) -> Path:
    writer = PdfWriter()
    page = writer.add_blank_page(width=_WIDTH, height=_HEIGHT)
    ops: list[str] = []
    if rule_offset is not None:
        for index in range(_LINES):
            y = _HEIGHT - _MARGIN - index * _LEADING + rule_offset
            ops.append(f"q 0 0 0 RG 0.8 w {_MARGIN - 10} {y} m {_WIDTH - _MARGIN + 10} {y} l S Q")
    if text:
        ops.extend(
            [
                "BT",
                f"/F1 {_FONT_SIZE} Tf",
                f"1 0 0 1 {_MARGIN} {_HEIGHT - _MARGIN} Tm",
                f"{_LEADING} TL",
            ]
        )
        ops.extend(
            f"(Line {index}: the quick brown fox jumps over the lazy dog) Tj T*"
            for index in range(_LINES)
        )
        ops.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(ops).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)  # noqa: SLF001
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}  # noqa: SLF001
            )
        }
    )
    with path.open("wb") as handle:
        writer.write(handle)
    return path


_CANVAS_SOURCE = (
    "// Universal Book Translator - 1:1 Facsimile Canvas Document\n"
    "#page(width: 612.0pt, height: 792.0pt, margin: 0pt, "
    'background: image("bg_stripped.pdf", page: 1, width: 100%, height: 100%))[\n'
    "]\n"
)

_REFLOW_SOURCE = (
    "#page(width: 612.0pt, height: 792.0pt, "
    "margin: (top: 36pt, bottom: 36pt, left: 36pt, right: 36pt))[\n]\n"
)


def _gate(pdf: Path, background: Path | None, *, typ_text: str | None = _CANVAS_SOURCE) -> Any:
    return asyncio.run(
        run_visual_gate(
            pdf,
            sample_pages=1,
            canvas_background_pdf=background,
            typ_text=typ_text,
        )
    )


def test_a_rule_crossing_the_text_fires_the_canvas_tier(tmp_path: Path) -> None:
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=_RULE_THROUGH_TEXT)

    result = _gate(artifact, background)

    assert result.stats["canvas_pages"] == 1
    assert result.stats["canvas_collision_pages"] == 1
    assert result.stats["canvas_worst_page"] == 1
    assert result.stats["canvas_worst_ratio"] > 5 * CANVAS_COLLISION_RATIO
    assert [f.page for f in result.findings if f.code == "canvas_ink_collision"] == [1]
    assert "canvas" in result.tiers_ran
    assert "canvas" not in result.tiers_skipped


def test_an_artifact_built_without_the_background_rules_is_not_measured(
    tmp_path: Path,
) -> None:
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=None)

    result = _gate(artifact, background)

    assert result.stats["canvas_pages"] == 0, "an unpaired page must not be scored"
    assert [f for f in result.findings if f.code == "canvas_ink_collision"] == []
    assert result.stats["canvas_unmeasured_pages"] == 1


def test_a_matched_pair_is_measured_and_reports_its_identity(tmp_path: Path) -> None:
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=_RULE_THROUGH_TEXT)

    result = _gate(artifact, background)

    assert result.stats["canvas_pages"] == 1
    assert result.stats["canvas_collision_pages"] == 1
    assert result.stats["canvas_unmeasured_pages"] == 0


def test_a_rule_in_the_empty_band_between_lines_does_not_fire(tmp_path: Path) -> None:
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_IN_GAP)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=_RULE_IN_GAP)

    result = _gate(artifact, background)

    assert result.stats["canvas_pages"] == 1, "the tier ran; it is not merely absent"
    assert result.stats["canvas_collision_pages"] == 0
    assert result.stats["canvas_worst_ratio"] < CANVAS_COLLISION_RATIO
    assert [f for f in result.findings if f.code == "canvas_ink_collision"] == []


def test_no_background_reports_the_tier_as_skipped_not_as_clean(tmp_path: Path) -> None:
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=None)

    result = _gate(artifact, None)

    assert result.stats["canvas_pages"] == 0
    assert "canvas" in result.tiers_skipped
    assert "canvas" not in result.tiers_ran
    assert "canvas_background_absent" in result.skip_codes
    assert "canvas ink-collision check skipped" in result.skipped_reason


def test_a_reflowed_page_is_not_scored_against_the_background(tmp_path: Path) -> None:
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=None)

    result = _gate(artifact, background, typ_text=_REFLOW_SOURCE)

    assert result.stats["canvas_declared_pages"] == 0
    assert result.stats["canvas_pages"] == 0
    assert [f for f in result.findings if f.code == "canvas_ink_collision"] == []
    assert "canvas" in result.tiers_skipped
    assert "canvas_pages_unknown" in result.skip_codes


def test_a_source_with_no_canvas_page_reports_the_tier_as_skipped(tmp_path: Path) -> None:
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=_RULE_THROUGH_TEXT)

    result = _gate(artifact, background, typ_text=None)

    assert result.stats["canvas_pages"] == 0
    assert "canvas_pages_unknown" in result.skip_codes
    assert "canvas_background_absent" not in result.skip_codes


def test_a_canvas_page_outside_the_sample_reports_that_rather_than_zero(tmp_path: Path) -> None:
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=_RULE_THROUGH_TEXT)
    source = "".join(
        [
            "#page(width: 612pt, height: 792pt, margin: 0pt, "
            'background: image("bg.pdf", page: 1, width: 100%, height: 100%))[\n]\n',
            "#page(width: 612pt, height: 792pt, margin: 0pt, "
            'background: image("bg.pdf", page: 2, width: 100%, height: 100%))[\n]\n',
        ]
    )

    result = _gate(artifact, background, typ_text=source)

    assert result.stats["canvas_declared_pages"] == 2
    assert result.stats["canvas_sampled_declared"] == 1
    assert result.stats["canvas_pages"] == 1


def test_a_sampled_canvas_page_that_measures_nothing_says_so(tmp_path: Path) -> None:
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_IN_GAP)
    artifact = _write_page(tmp_path / "out.pdf", text=False, rule_offset=_RULE_IN_GAP)

    result = _gate(artifact, background)

    assert result.stats["canvas_declared_pages"] == 1
    assert result.stats["canvas_pages"] == 0
    assert "canvas_nothing_measured" in result.skip_codes
    assert "canvas" in result.tiers_skipped
    assert [f for f in result.findings if f.code == "canvas_ink_collision"] == []


def test_an_unreadable_background_is_reported_rather_than_raised(tmp_path: Path) -> None:
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=None)
    broken = tmp_path / "bg.pdf"
    broken.write_bytes(b"%PDF-1.4\nnot really a pdf\n")

    result = _gate(artifact, broken)

    assert result.stats["canvas_pages"] == 0
    assert "canvas_background_render_failed" in result.skip_codes
    assert [f for f in result.findings if f.code == "canvas_ink_collision"] == []
