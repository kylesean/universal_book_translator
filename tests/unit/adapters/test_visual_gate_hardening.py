"""The visual gate's fail-closed semantics and new deterministic checks.

These pin the behaviours the deep review named as fail-open or blind: a VLM
that answers nonsense or crashes, an offset-overprinted pair the IoU test
cannot see, a faithful delivery that spilled a page, body text in the
furniture band, and a text-layer codepoint no embedded font can draw.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.gate_config import DEFAULT_GATE_THRESHOLDS, GateThresholds, resolve
from ubt.adapters.pdf.visual_gate import (
    _offset_overlap,
    block_overlap_findings,
    blocks_out_of_bounds_findings,
    furniture_collision_findings,
    judge_pages_with_vlm,
    parse_vlm_verdict,
)

pytestmark = pytest.mark.fast


# -- VLM verdict semantics -------------------------------------------------


def test_an_unparseable_verdict_is_not_a_pass() -> None:
    passed, detail = parse_vlm_verdict("I think this page looks mostly fine?")
    assert passed is None
    assert "mostly fine" in detail


def test_documented_verdicts_still_parse() -> None:
    assert parse_vlm_verdict("verdict: PASS\nissues: none")[0] is True
    assert parse_vlm_verdict("verdict: FAIL\nissues: clipped text")[0] is False


@pytest.mark.asyncio
async def test_a_nonsense_verdict_becomes_a_major_finding() -> None:
    async def judge(_images: list[str], _prompt: str) -> str:
        return "the page is fine I guess"

    findings = await judge_pages_with_vlm({3: "b64"}, judge)
    assert [(f.severity, f.code) for f in findings] == [("major", "vlm_verdict_unparseable")]


@pytest.mark.asyncio
async def test_a_crashing_judge_becomes_a_major_finding() -> None:
    async def judge(_images: list[str], _prompt: str) -> str:
        raise RuntimeError("provider 502")

    findings = await judge_pages_with_vlm({5: "b64"}, judge)
    assert [(f.severity, f.code) for f in findings] == [("major", "vlm_judge_error")]


@pytest.mark.asyncio
async def test_a_real_pass_stays_clean() -> None:
    async def judge(_images: list[str], _prompt: str) -> str:
        return "verdict: PASS\nissues: none"

    assert await judge_pages_with_vlm({1: "b64"}, judge) == []


# -- overlap: the offset signature ----------------------------------------


class _Box:
    def __init__(self, page: int, x0: float, y0: float, x1: float, y1: float) -> None:
        self.bbox = type("B", (), {"page": page, "x0": x0, "y0": y0, "x1": x1, "y1": y1})()


def test_offset_overprinted_lines_are_detected() -> None:
    """4pt on a 14pt line is IoU 0.556 — under the threshold, and the common
    real defect. The offset signature catches it."""
    a = (50.0, 700.0, 350.0, 714.0)
    b = (50.0, 696.0, 350.0, 710.0)
    assert _offset_overlap(a, b) is True
    findings = block_overlap_findings([_Box(1, *a), _Box(1, *b)])
    assert [f.code for f in findings] == ["block_overlap"]
    assert "offset" in findings[0].message


def test_adjacent_lines_in_a_column_do_not_trip_the_offset_signature() -> None:
    """Consecutive lines do not intersect, so normal prose never matches."""
    a = (50.0, 700.0, 350.0, 714.0)
    b = (50.0, 684.0, 350.0, 698.0)
    assert _offset_overlap(a, b) is False
    assert block_overlap_findings([_Box(1, *a), _Box(1, *b)]) == []


def test_side_by_side_boxes_do_not_trip_the_offset_signature() -> None:
    a = (50.0, 700.0, 200.0, 714.0)
    b = (300.0, 697.0, 450.0, 711.0)
    assert _offset_overlap(a, b) is False


def test_overlap_truncation_is_reported() -> None:
    boxes = [_Box(1, 50.0, 700.0 - i, 350.0, 714.0 - i) for i in range(5)]
    truncated: list[int] = []
    block_overlap_findings(
        boxes,
        truncated_pages=truncated,
        thresholds=GateThresholds(max_overlap_rects_per_page=3),
    )
    assert truncated == [1]


# -- bounds use the visible (crop) box ------------------------------------


def test_bounds_compare_against_the_given_visible_box() -> None:
    # The CropBox is 100pt narrower than the MediaBox; text at x=1050 is
    # clipped on screen and must be reported.
    box = _Box(1, 1050.0, 100.0, 1080.0, 120.0)
    assert blocks_out_of_bounds_findings([box], {1: (0.0, 0.0, 1000.0, 800.0)}) != []


# -- furniture ------------------------------------------------------------


def test_body_text_wholly_inside_the_header_band_is_a_collision() -> None:
    source = _Box(1, 50.0, 400.0, 350.0, 420.0)  # source had no header
    artifact = _Box(1, 50.0, 782.0, 350.0, 790.0)  # top 8 % of 800pt = 736+
    findings = furniture_collision_findings([artifact], [source], {1: (0.0, 0.0, 612.0, 800.0)})
    assert [f.code for f in findings] == ["furniture_collision"]


def test_a_source_header_makes_the_same_band_expected() -> None:
    source_header = _Box(1, 50.0, 782.0, 350.0, 790.0)
    artifact = _Box(1, 50.0, 782.0, 350.0, 790.0)
    findings = furniture_collision_findings(
        [artifact], [source_header], {1: (0.0, 0.0, 612.0, 800.0)}
    )
    assert findings == []


def test_grazing_the_band_is_not_a_collision() -> None:
    """A translation that grew into the margin from a block below it only
    overlaps the band partially, which is not furniture."""
    source = _Box(1, 50.0, 700.0, 350.0, 730.0)
    artifact = _Box(1, 50.0, 730.0, 350.0, 760.0)  # crosses 736, not wholly inside
    assert furniture_collision_findings([artifact], [source], {1: (0.0, 0.0, 612.0, 800.0)}) == []


# -- thresholds -----------------------------------------------------------


def test_unknown_threshold_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown gate threshold"):
        resolve({"overlap_iov": 0.9})


def test_resolve_applies_overrides_on_the_documented_defaults() -> None:
    thresholds = resolve({"overlap_iou": 0.75})
    assert thresholds.overlap_iou == 0.75
    assert thresholds.min_overlap_area_pt2 == DEFAULT_GATE_THRESHOLDS.min_overlap_area_pt2


def test_page_count_parity_flags_a_faithful_spill(tmp_path: Path) -> None:
    """A faithful delivery that spilled a page is a contract finding."""
    import asyncio

    from pdf_builders import write_text_pdf

    from ubt.adapters.pdf.visual_gate import run_visual_gate

    pdf = write_text_pdf(
        tmp_path / "out.pdf",
        [["Line one of text."], ["Line two of text."]],
    )
    clean = asyncio.run(
        run_visual_gate(pdf, expected_page_count=2, faithful_pages=True, sample_pages=0)
    )
    assert [f.code for f in clean.findings if f.code == "page_count_changed_faithful"] == []

    spilled = asyncio.run(
        run_visual_gate(pdf, expected_page_count=1, faithful_pages=True, sample_pages=0)
    )
    assert [f.code for f in spilled.findings if f.code == "page_count_changed_faithful"] != []


# -- font coverage --------------------------------------------------------


def test_font_codepoints_reads_a_real_font_cmap() -> None:
    """The positive path: a TrueType face's cmap is read and queried."""
    from pathlib import Path as _Path

    from ubt.adapters.pdf.visual_gate import _font_codepoints

    candidates = [
        path
        for path in _Path("/usr/share/fonts").rglob("*.ttf")
        if "Mono" in path.name or "Sans" in path.name
    ][:5]
    covered = None
    for candidate in candidates:
        probed = _font_codepoints(candidate.read_bytes())
        if probed is not None and ord("A") in probed and ord("0") in probed:
            covered = probed
            break
    if covered is None:
        pytest.skip("no Latin TrueType font available in this environment")
    assert ord("A") in covered and ord("0") in covered


def test_a_document_with_uninspectable_fonts_is_unmeasured_not_clean(
    tmp_path: Path,
) -> None:
    """A Standard-14 PDF embeds no font program: coverage is unknown, so the
    check must report nothing *and* count the page as skipped."""
    from pdf_builders import write_text_pdf

    from ubt.adapters.pdf.visual_gate import font_coverage_findings

    pdf = write_text_pdf(tmp_path / "plain.pdf", [["Hello world, this is text."]])
    stats: dict[str, int] = {}
    findings = font_coverage_findings(pdf, [1], coverage_stats=stats)
    assert findings == [], "uninspectable fonts must never be reported as missing glyphs"
    assert stats.get("font_coverage_pages_skipped") == 1
    assert "font_coverage_pages_measured" not in stats


def test_content_codepoint_filter_excludes_punctuation() -> None:
    from ubt.adapters.pdf.visual_gate import _is_content_codepoint

    assert _is_content_codepoint(ord("A"))
    assert _is_content_codepoint(ord("7"))
    assert _is_content_codepoint(ord("汉"))
    assert not _is_content_codepoint(ord("."))
    assert not _is_content_codepoint(ord(" "))
