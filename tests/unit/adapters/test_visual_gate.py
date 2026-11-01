from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ubt.adapters.pdf.visual_gate import (
    Severity,
    VisualFinding,
    _ArtifactBBox,
    _ArtifactBox,
    text_occlusion_findings,
)

pytestmark = pytest.mark.fast

_DPI = 72
_MEDIA = (0.0, 0.0, 200.0, 100.0)
_BOX = _ArtifactBox(id="l1", bbox=_ArtifactBBox(page=1, x0=10.0, y0=10.0, x1=110.0, y1=30.0))


def _page(
    tmp_path: Path,
    *,
    ink: Sequence[tuple[int, int, int, int]] = (),
    name: str = "page.png",
) -> Path:
    img = Image.new("L", (200, 100), 255)
    draw = ImageDraw.Draw(img)
    for rect in ink:
        draw.rectangle(rect, fill=0)
    path = tmp_path / name
    img.save(path)
    return path


def _stray_pixels(count: int) -> tuple[tuple[int, int, int, int], ...]:
    return tuple((20 + i, 80, 20 + i, 80) for i in range(count))


def test_white_on_white_is_reported_as_occluded(tmp_path: Path) -> None:
    findings = text_occlusion_findings({1: _page(tmp_path)}, [_BOX], {1: _MEDIA}, dpi=_DPI)
    assert [f.code for f in findings] == ["text_occluded"]
    assert findings[0].severity == "major"
    assert findings[0].page == 1
    assert "1 text line(s)" in findings[0].message


def test_a_line_carrying_ink_is_not_reported(tmp_path: Path) -> None:
    page = _page(tmp_path, ink=((20, 74, 100, 86),))
    assert text_occlusion_findings({1: page}, [_BOX], {1: _MEDIA}, dpi=_DPI) == []


def test_stray_dark_pixels_suppress_a_near_white_reading(tmp_path: Path) -> None:
    page = _page(tmp_path, ink=_stray_pixels(12))
    assert text_occlusion_findings({1: page}, [_BOX], {1: _MEDIA}, dpi=_DPI) == []
    sparse = _page(tmp_path, ink=_stray_pixels(6), name="sparse.png")
    findings = text_occlusion_findings({1: sparse}, [_BOX], {1: _MEDIA}, dpi=_DPI)
    assert [f.code for f in findings] == ["text_occluded"]


def test_rotated_page_occlusion_findings(tmp_path: Path) -> None:
    img_90 = Image.new("L", (100, 200), 255)
    draw_90 = ImageDraw.Draw(img_90)
    draw_90.rectangle((12, 15, 28, 55), fill=0)
    p90 = tmp_path / "p90.png"
    img_90.save(p90)

    findings_with_rot = text_occlusion_findings(
        {1: p90}, [_BOX], {1: _MEDIA}, dpi=_DPI, rotations={1: 90}
    )
    assert findings_with_rot == []

    findings_without_rot = text_occlusion_findings(
        {1: p90}, [_BOX], {1: _MEDIA}, dpi=_DPI, rotations={1: 0}
    )
    assert [f.code for f in findings_without_rot] == ["text_occluded"]


def test_pdf_struct_page_rotation(tmp_path: Path) -> None:
    import pikepdf

    from ubt.adapters.pdf.pdf_struct import page_rotation, page_rotations

    pdf = pikepdf.new()
    _ = pdf.add_blank_page(page_size=(200, 100))
    p2 = pdf.add_blank_page(page_size=(200, 100))
    p2.Rotate = 90
    out = tmp_path / "test_rot.pdf"
    pdf.save(out)

    with pikepdf.open(out) as opened:
        assert page_rotation(opened.pages[0]) == 0
        assert page_rotation(opened.pages[1]) == 90

    rots = page_rotations(out)
    assert rots == {1: 0, 2: 90}


def _finding(severity: Severity, code: str = "some_defect") -> VisualFinding:
    return VisualFinding(severity=severity, code=code, message="m")


def test_the_blocking_gate_returns_critical_only_when_enabled_and_short() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    findings = [_finding("critical"), _finding("major"), _finding("info")]
    tripped = blocking_gate_tripped(findings, total_pages=10, enabled=True)
    assert [f.severity for f in tripped] == ["critical"]
    assert blocking_gate_tripped(findings, total_pages=10, enabled=False) == []


def test_a_long_document_never_blocks() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped
    from ubt.core.policy.layout_policy import QA_FULL_GATE_MAX_PAGES

    findings = [_finding("critical")]
    assert (
        blocking_gate_tripped(findings, total_pages=QA_FULL_GATE_MAX_PAGES + 1, enabled=True) == []
    )


def test_zero_pages_means_not_measured_and_does_not_block() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    assert blocking_gate_tripped([_finding("critical")], total_pages=0, enabled=True) == []


def test_an_unreadable_artifact_fails_closed_even_when_disabled() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    findings = [_finding("critical"), _finding("major")]
    tripped = blocking_gate_tripped(findings, total_pages=-1, enabled=False)
    assert [f.severity for f in tripped] == ["critical"]


def test_a_missing_target_language_fails_closed_even_when_disabled() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    absent = _finding("critical", code="target_language_absent")
    tripped = blocking_gate_tripped([absent], total_pages=999, enabled=False)
    assert tripped == [absent]
    crashed = _finding("critical", code="visual_gate_crashed")
    assert blocking_gate_tripped([crashed], total_pages=999, enabled=False) == [crashed]


def test_a_blank_page_fails_closed_even_when_disabled() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped
    from ubt.core.policy.layout_policy import QA_FULL_GATE_MAX_PAGES

    blank = _finding("critical", code="blank_page")
    assert blocking_gate_tripped([blank], total_pages=3, enabled=False) == [blank]
    assert blocking_gate_tripped([blank], total_pages=3, enabled=True) == [blank]
    long_book = QA_FULL_GATE_MAX_PAGES + 100
    assert blocking_gate_tripped([blank], total_pages=long_book, enabled=False) == [blank]
    other = _finding("critical", code="some_defect")
    assert blocking_gate_tripped([other], total_pages=3, enabled=False) == []


def test_vlm_fail_is_advisory_by_default() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    fail = _finding("major", code="vlm_visual_fail")
    assert blocking_gate_tripped([fail], total_pages=10, enabled=True) == []
    assert blocking_gate_tripped([fail], total_pages=10, enabled=False) == []


def test_vlm_fail_blocks_a_short_document_when_explicitly_enabled() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    fail = _finding("major", code="vlm_visual_fail")
    assert blocking_gate_tripped(
        [fail], total_pages=10, enabled=False, vlm_blocking_enabled=True
    ) == [fail]


def test_vlm_fail_does_not_block_a_long_book() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped
    from ubt.core.policy.layout_policy import QA_FULL_GATE_MAX_PAGES

    fail = _finding("major", code="vlm_visual_fail")
    assert (
        blocking_gate_tripped(
            [fail],
            total_pages=QA_FULL_GATE_MAX_PAGES + 1,
            enabled=False,
            vlm_blocking_enabled=True,
        )
        == []
    )


def test_a_bare_page_number_is_page_furniture_not_content() -> None:
    from ubt.adapters.pdf.visual_gate import _has_body_text

    assert _has_body_text("2", 2) is False
    assert _has_body_text("128", 128) is False
    assert _has_body_text("- 7 -", 7) is False
    assert _has_body_text("xiv", 14) is False


def test_a_number_that_is_not_this_pages_number_is_content() -> None:
    from ubt.adapters.pdf.visual_gate import _has_body_text

    assert _has_body_text("2019", 7) is True
    assert _has_body_text("2", 3) is True
    assert _has_body_text("iv", 14) is True


def test_real_text_is_content_however_short(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import _has_body_text

    assert _has_body_text("Thanks to the reviewers.\n8", 8) is True
    assert _has_body_text("Figure 1: results\n2", 2) is True
    assert _has_body_text("Chapter One\n1", 1) is True
    assert _has_body_text("", 5) is False


def _page_pdf(
    tmp_path: Path, name: str, *, lines: Sequence[str], paint: Sequence[str] = ()
) -> Path:
    from pdf_builders import PAGE_HEIGHT, PAGE_WIDTH, _attach_font
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    ops = [f"BT /F1 12 Tf 54 {200 - 20 * i} Td ({line}) Tj ET" for i, line in enumerate(lines)]
    ops.extend(paint)
    stream = DecodedStreamObject()
    stream.set_data("\n".join(ops).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)  # noqa: SLF001 - fixture
    _attach_font(writer, page)
    out = tmp_path / name
    writer.write(out)
    return out


def test_a_page_with_nothing_at_all_is_contentless(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import page_has_no_content

    pdf = _page_pdf(tmp_path, "empty.pdf", lines=())
    assert page_has_no_content(pdf, [1]) == {1}


def test_a_page_whose_only_text_is_its_number_is_contentless(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import page_has_no_content

    pdf = _page_pdf(tmp_path, "numbered.pdf", lines=("1",))
    assert page_has_no_content(pdf, [1]) == {1}


def test_a_page_carrying_a_word_is_not_contentless(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import page_has_no_content

    pdf = _page_pdf(tmp_path, "word.pdf", lines=("End",))
    assert page_has_no_content(pdf, [1]) == set()


def test_a_page_carrying_only_a_drawing_is_not_contentless(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import page_has_no_content

    pdf = _page_pdf(
        tmp_path, "drawing.pdf", lines=(), paint=("q 0 0 0 RG 2 w 20 20 m 180 180 l S Q",)
    )
    assert page_has_no_content(pdf, [1]) == set()


def test_the_contentless_probe_never_reports_a_page_it_could_not_read(
    tmp_path: Path,
) -> None:
    from ubt.adapters.pdf.visual_gate import page_has_no_content

    source = Path(__file__).resolve().parents[3] / "corpus" / "documents" / "chapter-3.pdf"
    if not source.exists():  # pragma: no cover - corpus is optional in a bare checkout
        pytest.skip("corpus document not present")
    truncated = tmp_path / "truncated.pdf"
    truncated.write_bytes(source.read_bytes()[: source.stat().st_size // 2])
    assert page_has_no_content(truncated, [1, 2]) == set()


def test_a_file_that_is_not_a_pdf_at_all_reports_nothing(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import page_has_no_content

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    assert page_has_no_content(broken, [1, 2]) == set()


def test_a_sparse_but_legitimate_page_is_not_reported_blank(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    from ubt.adapters.pdf.visual_gate import page_has_no_content, pixel_findings

    img = Image.new("L", (910, 1287), 255)
    ImageDraw.Draw(img).text((80, 120), "Figure 1: results.", fill=0)
    path = tmp_path / "caption.png"
    img.save(path)
    assert [f.code for f in pixel_findings(path, 1)] == ["blank_page"]
    pdf = _page_pdf(tmp_path, "caption.pdf", lines=("Figure 1: results.",))
    assert page_has_no_content(pdf, [1]) == set()


def test_a_sparse_target_language_major_fails_closed() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    sparse = _finding("major", code="target_language_sparse")
    assert blocking_gate_tripped([sparse], total_pages=999, enabled=False) == [sparse]
    other = _finding("major", code="something_else")
    assert blocking_gate_tripped([other], total_pages=999, enabled=False) == []


def test_the_adaptive_budget_floors_short_books_and_caps_long_ones() -> None:
    from ubt.adapters.pdf.visual_gate import adaptive_sample_budget

    assert adaptive_sample_budget(0, 6) == 0
    assert adaptive_sample_budget(10, 0) == 0
    assert adaptive_sample_budget(8, 6) == 8
    assert adaptive_sample_budget(8, 100) == 8
    long_budget = adaptive_sample_budget(900, 6)
    assert long_budget >= 6
    assert long_budget == adaptive_sample_budget(901, 6)


def test_the_sample_puts_flagged_pages_first_and_covers_the_anchors() -> None:
    from ubt.adapters.pdf.visual_gate import select_sample_pages

    assert select_sample_pages(0, [1], 5) == []
    assert select_sample_pages(10, [1], 0) == []
    assert select_sample_pages(10, [3, 7, 99, 0], 2) == [3, 7]
    assert select_sample_pages(10, [5, 1, 9], 2) == [1, 5]
    sample = select_sample_pages(500, [], 5)
    assert {1, 2, 500}.issubset(set(sample))
    assert len(sample) <= 5
    assert all(1 <= page <= 500 for page in sample)


def test_the_report_names_the_tiers_that_ran_and_the_one_that_did_not() -> None:
    from ubt.adapters.pdf.visual_gate import _tier_coverage

    default_run = _tier_coverage(
        (),
        rendered_pages=5,
        sampled_pages=5,
        canvas_available=True,
    )
    assert "geometry" in default_run
    assert "type" in default_run
    assert "pixel" in default_run
    assert "not run: vlm" in default_run


def test_a_run_with_the_vlm_judge_reports_three_tiers() -> None:
    from ubt.adapters.pdf.visual_gate import _tier_coverage

    full = _tier_coverage((1, 2), rendered_pages=5, sampled_pages=5, canvas_available=True)
    assert "vlm" in full
    assert "not run" not in full


def test_the_canvas_tier_is_reported_as_skipped_when_no_background_was_supplied() -> None:
    from ubt.adapters.pdf.visual_gate import _tier_coverage, _tier_lists

    reflow = _tier_coverage((), rendered_pages=5, sampled_pages=5)
    assert "not run: canvas, vlm" in reflow

    _, skipped = _tier_lists((), rendered_pages=5)
    assert "canvas" in skipped

    _, skipped_with_bg = _tier_lists((), rendered_pages=0, canvas_available=True)
    assert "canvas" in skipped_with_bg

    ran, skipped_ok = _tier_lists((), rendered_pages=5, canvas_available=True)
    assert "canvas" in ran
    assert "canvas" not in skipped_ok


def test_a_report_never_claims_a_tier_that_did_not_run() -> None:
    from ubt.adapters.pdf.visual_gate import _tier_coverage

    nothing_rendered = _tier_coverage(
        (),
        rendered_pages=0,
        sampled_pages=5,
    )
    assert "type" not in nothing_rendered.split("not run")[0], (
        "the type tier needs a rendered page and must not be claimed"
    )
    assert "pixel" not in nothing_rendered.split("not run")[0]
    assert "not run: type, pixel, canvas, vlm" in nothing_rendered
    assert "0 of 5 sampled page(s) rendered" in nothing_rendered
    assert "tiers run: geometry" in nothing_rendered


def _typst_pdf(tmp_path: Path, body: str, *, size: str = "10pt") -> Path:
    import shutil
    import subprocess

    typst = shutil.which("typst")
    if typst is None:
        pytest.skip("typst is not installed")
    source = tmp_path / "main.typ"
    pdf = tmp_path / "main.pdf"
    source.write_text(
        f"#set page(width: 420pt, height: 600pt, margin: 20pt)\n#set text(size: {size})\n{body}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [typst, "compile", str(source), str(pdf)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:  # pragma: no cover - environment failure
        pytest.skip(f"typst failed: {result.stderr[:200]}")
    return pdf


def test_a_page_set_too_small_is_reported_as_illegible(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import type_size_findings

    pdf = _typst_pdf(tmp_path, "The quick brown fox jumps over the lazy dog. " * 40, size="0.5pt")
    findings = type_size_findings(pdf, [1])
    assert [f.code for f in findings] == ["type_illegible"]
    assert findings[0].severity == "critical"
    assert findings[0].page == 1


def test_a_normally_set_page_passes(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import type_size_findings

    pdf = _typst_pdf(tmp_path, "The quick brown fox jumps over the lazy dog. " * 40)
    assert type_size_findings(pdf, [1]) == []


def test_a_table_of_contents_leader_dots_do_not_fail_the_page(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import type_size_findings

    body = "\n".join(f"#h(1fr) Chapter {i} #h(1fr) {i}" for i in range(30))
    pdf = _typst_pdf(tmp_path, body)
    assert type_size_findings(pdf, [1]) == []


def test_six_point_type_is_small_but_legible(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import type_size_findings

    pdf = _typst_pdf(tmp_path, "The quick brown fox jumps over the lazy dog. " * 40, size="6pt")
    assert type_size_findings(pdf, [1]) == []


def test_a_matrix_scaled_font_is_measured_by_rendered_height(tmp_path: Path) -> None:
    from ubt.adapters.pdf.visual_gate import type_size_findings

    body = "#scale(x: 90%, y: 90%)[The quick brown fox jumps over the lazy dog. " * 40 + "]"
    pdf = _typst_pdf(tmp_path, body)
    assert type_size_findings(pdf, [1]) == []


def test_the_illegible_code_fails_closed_with_the_blocking_gate_off() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    finding = VisualFinding(
        severity="critical",
        code="type_illegible",
        message="Page 3 is typeset at a median glyph height of 0.49pt",
        page=3,
    )
    for enabled in (False, True):
        tripped = blocking_gate_tripped([finding], total_pages=400, enabled=enabled)
        assert [f.code for f in tripped] == ["type_illegible"]


def test_a_collapsed_region_warns_but_does_not_fail_closed() -> None:
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped

    finding = VisualFinding(
        severity="major",
        code="type_collapsed",
        message="Page 7 has 80% of its glyphs below 4pt",
        page=7,
    )
    assert blocking_gate_tripped([finding], total_pages=400, enabled=False) == []
    assert blocking_gate_tripped([finding], total_pages=400, enabled=True) == []


def test_the_visual_report_carries_tiers_as_data_not_only_prose() -> None:
    from ubt.adapters.pdf.visual_gate import VisualGateResult

    result = VisualGateResult(
        passed=True,
        skipped_reason="vlm judge disabled; tiers run: geometry, type, pixel (not run: vlm)",
        tiers_ran=("geometry", "type", "pixel"),
        tiers_skipped=("vlm",),
        skip_codes=("vlm_judge_disabled",),
    )
    payload = result.report_payload()

    assert payload["tiers_ran"] == ["geometry", "type", "pixel"]
    assert payload["tiers_skipped"] == ["vlm"]
    assert payload["skip_codes"] == ["vlm_judge_disabled"]
    assert "not run: vlm" in payload["skipped_reason"]


def test_an_unknown_skip_note_is_flagged_rather_than_invented() -> None:
    from ubt.adapters.pdf.visual_gate import _skip_code

    assert _skip_code("vlm judge disabled") == "vlm_judge_disabled"
    assert _skip_code("page render failed (pdf_oxide); pixel checks skipped") == (
        "page_render_failed"
    )
    assert _skip_code("some future note nobody mapped") == "unknown"


def test_tier_lists_distinguish_a_render_failure_from_a_judge_miss() -> None:
    from ubt.adapters.pdf.visual_gate import _tier_lists

    ran, skipped = _tier_lists((), rendered_pages=6)
    assert ran == ["geometry", "source", "type", "pixel"]
    assert skipped == ["canvas", "vlm"]

    ran, skipped = _tier_lists((), rendered_pages=0)
    assert ran == ["geometry", "source"]
    assert skipped == ["type", "pixel", "canvas", "vlm"]

    ran, skipped = _tier_lists((), rendered_pages=6, source_available=False)
    assert ran == ["geometry", "type", "pixel"]
    assert skipped == ["source", "canvas", "vlm"]

    ran, skipped = _tier_lists((2,), rendered_pages=6, canvas_available=True)
    assert ran == ["geometry", "source", "type", "pixel", "canvas", "vlm"]
    assert skipped == []


def test_every_skip_note_the_gate_emits_has_a_code_and_a_translation() -> None:
    import ast
    import re
    from pathlib import Path

    from ubt.adapters.pdf.visual_gate import _SKIP_CODES

    repo = Path(__file__).resolve().parents[3]
    source = (repo / "ubt" / "adapters" / "pdf" / "visual_gate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    emitted: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "append"
            and isinstance(func.value, ast.Name)
            and func.value.id == "skip_notes"
        ):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            emitted.add(str(node.args[0].value))

    assert emitted, "no skip_notes.append(...) literal found; update this test"
    missing = sorted(note for note in emitted if note not in _SKIP_CODES)
    assert not missing, f"skip note(s) with no stable code: {missing}"

    codes = {_SKIP_CODES[note] for note in emitted}
    for lang in ("en", "zh"):
        tsv = (repo / "web" / "src" / "i18n" / "translations" / f"{lang}.ts").read_text(
            encoding="utf-8"
        )
        block = re.search(r"visualSkipCodes:\s*\{(.*?)\}", tsv, re.DOTALL)
        assert block, f"{lang}.ts has no visualSkipCodes block"
        for code in sorted(codes | {"unknown"}):
            assert f"{code}:" in block.group(1), f"{lang}.ts is missing visualSkipCodes.{code}"


def test_the_source_tier_reads_the_typst_file_the_renderer_actually_writes(
    tmp_path: Path,
) -> None:
    from ubt.core.engine.visual_verify_loop import _typst_source_path

    out = Path(tmp_path) / "book_mono.pdf"
    out.write_bytes(b"%PDF-1.4\n")
    build = Path(tmp_path) / ".book_mono_typst"
    build.mkdir()
    (build / "main.typ").write_text("#set text()", encoding="utf-8")

    assert _typst_source_path(out) == build / "main.typ"


def test_a_missing_typst_source_is_reported_as_a_skipped_tier(tmp_path: Path) -> None:
    from ubt.core.engine.visual_verify_loop import _typst_source_path

    out = Path(tmp_path) / "book_mono.pdf"
    out.write_bytes(b"%PDF-1.4\n")
    assert _typst_source_path(out) is None
