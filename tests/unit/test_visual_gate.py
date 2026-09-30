"""Unit tests for the post-render visual gate (T0/T1/T2, warn-only)."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ubt.adapters.pdf.visual_gate import (
    adaptive_sample_budget,
    artifact_text_boxes,
    blank_page_candidates,
    block_overlap_findings,
    blocking_gate_tripped,
    blocks_out_of_bounds_findings,
    chart_pages_from_blocks,
    parse_vlm_verdict,
    pdf_page_count,
    run_visual_gate,
    scan_typ_source,
    select_sample_pages,
)
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock


def _write_pdf(path: Path, pages: int) -> Path:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595.0, height=842.0)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def _block(block_id: str, page: int, x0: float, y0: float, x1: float, y1: float) -> IRBlock:
    return IRBlock(
        id=block_id,
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="source",
        target_text="译文",
        bbox=BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1),
    )


def test_scan_typ_source_flags_u2011() -> None:
    assert scan_typ_source("a\u2011b") != []  # U+2011 non-breaking hyphen
    assert scan_typ_source("a-b") == []  # ASCII hyphen-minus
    assert scan_typ_source("plain text") == []


def test_select_sample_pages_flagged_first() -> None:
    sampled = select_sample_pages(100, [90, 5], 4)
    assert {5, 90} <= set(sampled)
    assert len(sampled) == 4
    assert select_sample_pages(0, [1], 4) == []
    assert select_sample_pages(3, [], 6) == [1, 2, 3]


def test_adaptive_sample_budget_tiers() -> None:
    # <=20 full inspection, 20-100 ~20% (floor 6, cap 10), >100 fixed 10.
    assert adaptive_sample_budget(3, 2) == 3
    assert adaptive_sample_budget(13, 6) == 13
    assert adaptive_sample_budget(50, 6) == 10
    assert adaptive_sample_budget(100, 6) == 10
    assert adaptive_sample_budget(300, 6) == 10
    assert adaptive_sample_budget(300, 3) == 10
    assert adaptive_sample_budget(10, 0) == 0
    assert adaptive_sample_budget(0, 6) == 0


def test_select_sample_pages_long_book_anchors() -> None:
    sampled = select_sample_pages(300, [150], 10)
    assert {1, 2, 150, 300} <= set(sampled)
    assert len(sampled) == 10


def test_blocking_gate_tripped() -> None:
    """Opt-in, short docs, CRITICAL-only."""
    from ubt.adapters.pdf.visual_gate import VisualFinding

    crit = VisualFinding(severity="critical", code="blank_page", message="blank", page=3)
    major = VisualFinding(severity="major", code="block_overlap", message="overlap", page=2)
    assert blocking_gate_tripped([crit, major], 10, True) == [crit]
    assert blocking_gate_tripped([major], 10, True) == []
    assert blocking_gate_tripped([crit], 10, False) == []
    assert blocking_gate_tripped([crit], 21, True) == []
    assert blocking_gate_tripped([crit], 0, True) == []
    assert blocking_gate_tripped([], 10, True) == []
    # An unreadable PDF (pdf_page_count sentinel -1) fails closed,
    # even when the opt-in visual gate is off.
    assert blocking_gate_tripped([crit, major], -1, True) == [crit]
    assert blocking_gate_tripped([crit], -1, False) == [crit]
    assert blocking_gate_tripped([major], -1, True) == []


def test_absent_target_language_fails_closed_regardless_of_gate() -> None:
    """T0.5 parity: an artifact with none of the target language is unshippable
    — even long docs and a disabled gate cannot waive it."""
    from ubt.adapters.pdf.visual_gate import VisualFinding

    absent = VisualFinding(severity="critical", code="target_language_absent", message="zh=0%")
    unrelated = VisualFinding(severity="critical", code="blank_page", message="blank", page=2)
    assert blocking_gate_tripped([absent, unrelated], 500, False) == [absent]


def test_delivery_breaking_parity_majors_fail_closed() -> None:
    """A gate that says passed=False on a geometry/asset parity major must
    refuse export (arXiv 2609.20519 forced-reflow post-mortem: page/image
    count parity majors and a sparse-target-language artifact all shipped).
    Other majors stay warn-only."""
    from ubt.adapters.pdf.visual_gate import VisualFinding

    pages = VisualFinding(severity="major", code="page_count_changed", message="15->16")
    images = VisualFinding(severity="major", code="image_count_changed", message="1743->2")
    sparse = VisualFinding(severity="major", code="target_language_sparse", message="zh=3%")
    cosmetic = VisualFinding(severity="major", code="block_overlap", message="overlap", page=2)

    assert blocking_gate_tripped([pages, cosmetic], 15, False) == [pages]
    assert blocking_gate_tripped([images], 500, False) == [images]
    assert blocking_gate_tripped([sparse], 500, True) == [sparse]
    # non-parity majors keep the warn-only contract
    assert blocking_gate_tripped([cosmetic], 15, False) == []
    # the parity codes only block at major severity via this rung; info stays quiet
    info = VisualFinding(severity="info", code="page_count_changed", message="x")
    assert blocking_gate_tripped([info], 15, False) == []


@pytest.mark.asyncio
async def test_unreadable_pdf_trips_blocking_gate(tmp_path: Path) -> None:
    """A rendered PDF nobody can read must refuse export, not fail open.

    run_visual_gate reports ``passed=False`` with a CRITICAL ``unreadable_pdf``
    finding, but used to leave ``stats["total_pages"] = 0`` — which
    blocking_gate_tripped() reads as "no document measured", returned [], and
    let export.py ship the corrupt artifact despite the explicit gate failure.
    """
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 not really a pdf")

    res = await run_visual_gate(broken, blocks=[], sample_pages=0)

    assert res.passed is False
    assert [f.code for f in res.findings] == ["unreadable_pdf"]
    total_pages = int(res.stats["total_pages"])
    assert total_pages < 0
    tripped = blocking_gate_tripped(res.findings, total_pages, True)
    assert [f.code for f in tripped] == ["unreadable_pdf"]
    # Structural failure, not a quality warning: the refusal holds even with
    # the opt-in visual gate disabled.
    assert blocking_gate_tripped(res.findings, total_pages, False) == list(res.findings)


def test_chart_pages_from_blocks() -> None:
    plain = _block("a", 1, 10.0, 10.0, 60.0, 30.0)
    chart = IRBlock(
        id="b",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="fig",
        bbox=BoundingBox(page=2, x0=10.0, y0=10.0, x1=60.0, y1=30.0),
        provenance={"page_kind": "vector_heavy"},
    )
    assert chart_pages_from_blocks([plain, chart]) == {2}
    assert chart_pages_from_blocks([plain]) == set()


def test_parse_vlm_verdict() -> None:
    passed, _ = parse_vlm_verdict("verdict: PASS\nissues: none")
    assert passed is True
    failed, detail = parse_vlm_verdict("verdict: FAIL\nissues: black squares")
    assert failed is False
    assert "black squares" in detail


def test_parse_vlm_verdict_requires_documented_prefix() -> None:
    """Free-form prose mentioning 'pass' is not an explicit PASS verdict."""
    passed, detail = parse_vlm_verdict("the page passed inspection, no issues")
    assert passed is True
    assert "unparseable" in detail
    passed, detail = parse_vlm_verdict("")
    assert passed is True
    assert "unparseable" in detail


def test_render_pages_to_png_cleans_own_tmpdir_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-work_dir path must not leak its temp dir when nothing rendered."""
    import tempfile

    from ubt.adapters.pdf import oxide_render
    from ubt.adapters.pdf import visual_gate as vg

    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def fake_mkdtemp(*args: str, **kwargs: str) -> str:
        name = str(real_mkdtemp(*args, **kwargs))
        created.append(Path(name))
        return name

    # Patch the real modules: mypy --strict forbids reaching modules through
    # the adapter's namespace (no implicit re-export).
    monkeypatch.setattr(tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(oxide_render, "write_page_png", lambda *a, **k: None)

    result = vg.render_pages_to_png(tmp_path / "x.pdf", [1])
    assert result == {}
    assert created and not created[0].exists()


def test_block_overlap_and_oob() -> None:
    big_a = _block("a", 1, 50.0, 50.0, 450.0, 300.0)
    big_b = _block("b", 1, 60.0, 60.0, 440.0, 290.0)
    findings = block_overlap_findings([big_a, big_b])
    assert any(f.code == "block_overlap" for f in findings)
    # Do not hide a second pair merely because the page already had one finding.
    big_c = _block("c2", 1, 70.0, 70.0, 430.0, 280.0)
    assert len(block_overlap_findings([big_a, big_b, big_c])) == 3
    small = _block("c", 2, 10.0, 10.0, 60.0, 30.0)
    assert block_overlap_findings([small]) == []
    oob = _block("d", 1, -50.0, 10.0, 60.0, 30.0)
    bounds = {1: (0.0, 0.0, 595.0, 842.0)}
    assert any(
        f.code == "block_out_of_bounds" for f in blocks_out_of_bounds_findings([oob], bounds)
    )


def test_blank_page_candidates_text_level(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "blank.pdf", 2)
    assert pdf_page_count(pdf) == 2
    assert blank_page_candidates(pdf) == [1, 2]


@pytest.mark.asyncio
async def test_run_visual_gate_flags_truly_blank_pdf(tmp_path: Path) -> None:
    # add_blank_page PDFs render as genuinely blank: pixel check must fire.
    # 3-page books are fully inspected, so sample_pages=2 adapts to 3.
    pdf = _write_pdf(tmp_path / "out.pdf", 3)
    res = await run_visual_gate(pdf, blocks=[], typ_text="a-b", sample_pages=2)
    assert res.stats["total_pages"] == 3
    assert len(res.sampled_pages) == 3
    assert res.vlm_pages == ()
    assert any(f.code == "blank_page" for f in res.findings)
    assert res.passed is False


@pytest.mark.asyncio
async def test_run_visual_gate_text_only_green_path(tmp_path: Path) -> None:
    # sample_pages=0 skips rendering: info-grade candidates alone stay green.
    pdf = _write_pdf(tmp_path / "out.pdf", 2)
    res = await run_visual_gate(pdf, blocks=[], typ_text="a-b", sample_pages=0)
    assert res.sampled_pages == ()
    assert any(f.code == "blank_candidate" for f in res.findings)
    assert res.passed is True


@pytest.mark.asyncio
async def test_run_visual_gate_vlm_fail_is_major(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _write_pdf(tmp_path / "out.pdf", 1)

    async def _fail_judge(images: list[str], prompt: str) -> str:
        assert len(images) == 1
        assert "verdict" in prompt.lower()
        return "verdict: FAIL\nissues: overlapping text"

    # Bypass real pdf_oxide rendering in CI with a stub PNG.
    png = tmp_path / "p1-1.png"
    png.write_bytes(b"fakepng")
    monkeypatch.setattr(
        "ubt.adapters.pdf.visual_gate.render_pages_to_png", lambda *a, **k: {1: png}
    )
    res = await run_visual_gate(pdf, sample_pages=1, max_vlm_pages=1, vlm_judge=_fail_judge)
    assert res.vlm_pages == (1,)
    assert any(f.code == "vlm_visual_fail" for f in res.findings)
    assert res.passed is False


@pytest.mark.asyncio
async def test_vlm_prefers_flagged_chart_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flagged+chart > flagged > chart > rest for VLM slots."""
    pdf = _write_pdf(tmp_path / "out.pdf", 3)
    png = tmp_path / "p1-1.png"
    png.write_bytes(b"fakepng")
    pngs = {1: png, 2: png, 3: png}
    monkeypatch.setattr(
        "ubt.adapters.pdf.visual_gate.render_pages_to_png", lambda *a, **k: dict(pngs)
    )
    # Blank pages make every page flagged; page 2 is the chart page.
    chart_block = IRBlock(
        id="c",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="fig",
        bbox=BoundingBox(page=2, x0=10.0, y0=10.0, x1=60.0, y1=30.0),
        provenance={"page_kind": "mixed_complex"},
    )

    async def _pass_judge(images: list[str], prompt: str) -> str:
        return "verdict: PASS\nissues: none"

    res = await run_visual_gate(
        pdf, blocks=[chart_block], sample_pages=3, max_vlm_pages=1, vlm_judge=_pass_judge
    )
    assert res.vlm_pages == (2,)


@pytest.mark.asyncio
async def test_mock_provider_vision_records_call() -> None:
    from ubt.core.router.provider import MockModelProvider

    provider = MockModelProvider(default_response="verdict: PASS\nissues: none")
    out = await provider.generate_with_images("look", ["aGk="], model="m")
    assert "PASS" in out
    assert provider.call_history[0]["image_count"] == 1


async def test_visual_gate_offloads_blocking_render_to_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Render_pages_to_png rasterizes up to 6 pages in-process via pdf_oxide —
    it must run off the event loop via asyncio.to_thread so a
    concurrent SSE stream / lease renewal is never frozen. We confirm the render
    callback executes on a worker thread, not the event-loop (main) thread.
    """
    import threading

    pdf = _write_pdf(tmp_path / "out.pdf", 2)
    offloaded = {"worker": False}

    def fake_render(*args: object, **kwargs: object) -> dict[int, Path]:
        offloaded["worker"] = threading.current_thread() is not threading.main_thread()
        return {}

    monkeypatch.setattr("ubt.adapters.pdf.visual_gate.render_pages_to_png", fake_render)
    await run_visual_gate(pdf, blocks=[], sample_pages=2)
    assert offloaded["worker"] is True


async def test_visual_gate_facing_spread_flyleaf_exempt(tmp_path: Path) -> None:
    """Verify that page 1 intentional blank flyleaf in facing spread mode is not flagged as defect."""
    pdf = _write_pdf(tmp_path / "facing.pdf", 2)

    # Without facing_spread, page 1 blank is flagged
    res_normal = await run_visual_gate(pdf, blocks=[], facing_spread=False)
    # With facing_spread, page 1 blank is exempted
    res_facing = await run_visual_gate(pdf, blocks=[], facing_spread=True)

    normal_blank_pages = [
        f.page for f in res_normal.findings if f.code in ("blank_page", "blank_candidate")
    ]
    facing_blank_pages = [
        f.page for f in res_facing.findings if f.code in ("blank_page", "blank_candidate")
    ]

    assert 1 in normal_blank_pages
    assert 1 not in facing_blank_pages


async def test_visual_gate_suppresses_declared_padding_pages(tmp_path: Path) -> None:
    """Declared interleave-padding pages must not be blank-page defects.

    Facing/alternating interleaving pads the shorter side with intentional
    blanks; the gate only exempted the page-1 flyleaf, so every real padding
    page came back CRITICAL ``blank_page`` — flipping a correct artifact to
    ``passed=False`` and, under the opt-in blocking gate on a short doc,
    refusing to ship it. The alternator now declares exactly which pages it
    filled, and the gate must honour that list.
    """
    pdf = _write_pdf(tmp_path / "padded.pdf", 3)

    # No declaration: every blank page is still a defect (baseline behaviour).
    undeclared = await run_visual_gate(pdf, blocks=[], sample_pages=3)
    undeclared_blank = {
        f.page for f in undeclared.findings if f.code in ("blank_page", "blank_candidate")
    }
    assert {1, 2, 3} <= undeclared_blank

    # Declared padding pages are exempt; an undeclared blank page still fires.
    declared = await run_visual_gate(pdf, blocks=[], sample_pages=3, padding_pages=(2, 3))
    declared_blank = {
        f.page for f in declared.findings if f.code in ("blank_page", "blank_candidate")
    }
    assert 1 in declared_blank
    assert 2 not in declared_blank
    assert 3 not in declared_blank


def test_source_bboxes_only_gate_the_engine_that_keeps_them() -> None:
    """T1 compares source-page rectangles against the output mediabox.

    Meaningful for the anchored overlay (its canvas is the original page) and
    for the alternating zipper (its pages *are* the source pages); the reflow
    engine rebuilds at A4, so a US-Letter full-bleed table is flagged "major"
    on a perfectly rendered book -- and that finding drives a whole-book re-render
    plus quarantine of the blocks it names.
    """
    from ubt.adapters.pdf.visual_gate import blocks_out_of_bounds_findings
    from ubt.core.engine.reflow_loop import ReflowControlLoop

    letter_block = [
        SimpleNamespace(
            id="tbl1", bbox=SimpleNamespace(page=1, x0=6.0, y0=90.0, x1=606.0, y1=300.0)
        )
    ]
    a4 = {1: (0.0, 0.0, 595.276, 841.89)}
    letter = {1: (0.0, 0.0, 612.0, 792.0)}
    assert [f.code for f in blocks_out_of_bounds_findings(letter_block, a4)] == [
        "block_out_of_bounds"
    ], "the false positive this guard exists for"
    assert blocks_out_of_bounds_findings(letter_block, letter) == []

    def loop_with(metadata: object, run_engine: str | None = None) -> ReflowControlLoop:
        return cast(
            "ReflowControlLoop",
            SimpleNamespace(
                manifest=SimpleNamespace(
                    metadata=metadata,
                    run=SimpleNamespace(render_engine_effective=run_engine),
                )
            ),
        )

    keeps = ReflowControlLoop._output_keeps_source_geometry
    # The typed run field is the source of truth; the metadata copy is the
    # fallback for manifests that predate it (or never had it as a dict).
    assert keeps(loop_with({"render_engine_effective": "publication"})) is False
    assert keeps(loop_with({"render_engine_effective": "rigid"})) is True
    assert keeps(loop_with(None)) is True
    assert keeps(loop_with({})) is True
    # The typed field wins over the metadata copy, and a typed "publication"
    # is honored even when metadata is missing entirely.
    assert keeps(loop_with({"render_engine_effective": "publication"}, run_engine="rigid")) is True
    assert keeps(loop_with(None, run_engine="publication")) is False


def test_visual_gate_measures_the_artifact_not_the_ir(tmp_path: Path) -> None:
    """Geometry is read from the delivered PDF, not the IR's source boxes.

    Two text runs rendered on top of each other are invisible to the IR (whose
    bboxes are fine) but obvious in the artifact. The gate must read the
    artifact's own pdfium rects, so the overprint is flagged.
    """
    import shutil
    import subprocess

    if shutil.which("typst") is None:
        pytest.skip("typst binary unavailable")

    typ = (
        "#set page(width: 300pt, height: 200pt, margin: 0pt)\n"
        "#place(top + left, dx: 20pt, dy: 20pt)[#text(size: 12pt)[overlapping line one]]\n"
        "#place(top + left, dx: 20pt, dy: 20pt)[#text(size: 12pt)[overlapping line two]]\n"
    )
    src = tmp_path / "over.typ"
    out = tmp_path / "over.pdf"
    src.write_text(typ, encoding="utf-8")
    subprocess.run(["typst", "compile", str(src), str(out)], check=True)

    boxes = artifact_text_boxes(out, [1])
    assert len(boxes) >= 2
    assert any(f.code == "block_overlap" for f in block_overlap_findings(boxes))


@pytest.mark.asyncio
async def test_visual_gate_reports_artifact_unverified_when_no_text(tmp_path: Path) -> None:
    """No artifact text means the layout was not verified — say so, don't pass clean."""
    pdf = _write_pdf(tmp_path / "blank.pdf", 1)
    res = await run_visual_gate(pdf, blocks=[], sample_pages=0)
    assert any(f.code == "artifact_unverified" for f in res.findings)


@pytest.mark.fast
def test_oob_uses_the_mediabox_origin_not_just_its_size() -> None:
    """A visible box on a non-zero-origin MediaBox is not out of bounds.

    T1 compared against the MediaBox *size* only, so on a page whose box is
    ``[10 10 610 810]`` a run ending at x=608.9 (inside the visible page, which
    reaches 610) was flagged 'major' because 608.9 > width(600) + tol.
    """
    from ubt.adapters.pdf.visual_gate import blocks_out_of_bounds_findings

    box = {1: (10.0, 10.0, 610.0, 810.0)}
    visible = [
        SimpleNamespace(
            id="p1", bbox=SimpleNamespace(page=1, x0=20.0, y0=400.0, x1=608.9, y1=410.0)
        )
    ]
    assert blocks_out_of_bounds_findings(visible, box) == []
    # A box past the true right edge is still flagged.
    outside = [
        SimpleNamespace(
            id="p2", bbox=SimpleNamespace(page=1, x0=20.0, y0=400.0, x1=625.0, y1=410.0)
        )
    ]
    assert [f.code for f in blocks_out_of_bounds_findings(outside, box)] == ["block_out_of_bounds"]


@pytest.mark.fast
def test_oob_reads_a_real_offset_mediabox_artifact(tmp_path: Path) -> None:
    """End-to-end: a real PDF with MediaBox ``[10 10 610 810]`` whose text is
    fully visible must not trip the out-of-bounds check."""
    import pikepdf
    from pikepdf import Array, Dictionary, Name

    from ubt.adapters.pdf.visual_gate import page_bounds

    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    page.obj["/MediaBox"] = Array([10, 10, 610, 810])
    page.obj["/Contents"] = pdf.make_stream(b"BT /F1 12 Tf 582 400 Td (Hello) Tj ET")
    page.obj["/Resources"] = Dictionary(
        Font=Dictionary(F1=Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica))
    )
    out = tmp_path / "offset.pdf"
    pdf.save(str(out))
    pdf.close()

    boxes = artifact_text_boxes(out, [1])
    assert boxes, "pdfium must read the text"
    assert blocks_out_of_bounds_findings(boxes, page_bounds(out)) == []


@pytest.mark.fast
def test_geometry_sampling_always_includes_the_final_page() -> None:
    """The even artifact-geometry sample must never drop the last page.

    The pixel/VLM sampler anchors the tail (``select_sample_pages``), but the
    geometry sample used ``round(1 + i*step)`` and above ~180 pages its last
    probe landed before ``total`` -- so a long book's final pages were never
    checked for overlap / out-of-bounds.
    """
    from ubt.adapters.pdf.visual_gate import MAX_ARTIFACT_GEOMETRY_PAGES, _geometry_pages

    for total in (121, 240, 1000):
        pages = _geometry_pages(total)
        assert pages[0] == 1
        assert pages[-1] == total, f"last page missing from geometry sample (total={total})"
        assert len(pages) <= MAX_ARTIFACT_GEOMETRY_PAGES + 1
        assert pages == sorted(set(pages))


@pytest.mark.fast
def test_render_pages_to_png_registers_atexit_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ubt.adapters.pdf import visual_gate

    registered: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        "atexit.register",
        lambda fn, *args, **kwargs: registered.append((fn, args, kwargs)),
    )
    monkeypatch.setattr(
        "ubt.adapters.pdf.oxide_render.write_page_png",
        lambda pdf_path, page, dpi, tmp: tmp / f"page_{page}.png",
    )

    pdf_dummy = tmp_path / "dummy.pdf"
    pdf_dummy.write_bytes(b"%PDF-dummy")

    result = visual_gate.render_pages_to_png(pdf_dummy, [1], work_dir=None)
    assert 1 in result
    # Check that atexit registered rmtree for the temporary directory
    import shutil

    assert any(
        fn is shutil.rmtree and len(args) > 0 and args[0] == result[1].parent
        for fn, args, _ in registered
    )


def test_blocking_gate_trips_on_a_crashed_gate_even_when_disabled() -> None:
    # A gate that crashed means "visual state unknown". That refuses export on
    # the same fail-closed reasoning as an unreadable PDF, regardless of the
    # opt-in flag or the page window — a crash used to read as "no gate" and
    # skipped every rejection layer.
    from ubt.adapters.pdf.visual_gate import VisualFinding

    finding = VisualFinding(
        severity="critical",
        code="visual_gate_crashed",
        message="visual gate did not complete: RuntimeError: boom",
    )
    tripped = blocking_gate_tripped((finding,), total_pages=0, enabled=False)
    assert [f.code for f in tripped] == ["visual_gate_crashed"]


@pytest.mark.asyncio
async def test_run_visual_gate_flags_text_painted_over(tmp_path: Path) -> None:
    """Text the text layer reports but the raster does not show is occluded.

    A near-white paint over a paragraph is invisible to the geometry checks
    (the boxes are still there) and to the blank-page heuristic (the page is
    not blank overall) — the classic scalpel white-out shipped undetected.
    """
    import pikepdf

    pdf = tmp_path / "whitebox.pdf"
    content = (
        b"BT /F1 12 Tf 72 700 Td (Hidden paragraph text here) Tj ET\n"
        b"1 1 1 rg\n72 690 300 20 re f\n"
        b"0 0 0 rg\n72 100 200 50 re f\n"
        b"BT /F1 12 Tf 72 670 Td (Visible line stays readable) Tj ET\n"
    )
    with pikepdf.new() as doc:
        page = doc.add_blank_page(page_size=(595, 842))
        page.Contents = doc.make_stream(content)
        page.Resources = pikepdf.Dictionary(
            Font=pikepdf.Dictionary(
                F1=pikepdf.Dictionary(
                    Type=pikepdf.Name("/Font"),
                    Subtype=pikepdf.Name("/Type1"),
                    BaseFont=pikepdf.Name("/Helvetica"),
                )
            )
        )
        doc.save(pdf)

    res = await run_visual_gate(pdf, blocks=[], typ_text="a-b", sample_pages=1)
    assert any(f.code == "text_occluded" for f in res.findings)
    assert res.passed is False
