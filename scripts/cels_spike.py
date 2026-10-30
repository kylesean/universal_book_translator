"""G3 CELS spike: content-stream feasibility evidence on real pages.

Runs the two questions G3 has to answer before G4 builds a solver
(``docs/guides/ADAPTIVE_DELIVERY_IMPLEMENTATION_PLAN.md`` §4.3):

1. **Stage 1 geometry** — does the page split into the columns a human sees,
   and is every near-miss gutter refused with a reason? (``ubt.core.layout``.)
2. **Stage 5 mechanism** — do ``Tz`` / ``Tc`` / ``TL`` survive text extraction,
   colours, clipping and ``q``/``Q``, and does a compressed line stay inside its
   column? (``ubt.adapters.pdf.cels_inject``.)

Usage::

    uv run python scripts/cels_spike.py \
        --pdf corpus/documents/chapter-3.pdf --pages 9 12 \
        --out-dir output/adaptive_delivery/g3

Outputs, per page: ``p<N>_injected.pdf`` (tracking arm) and
``p<N>_leading.pdf`` (``TL``/``T*`` twin), plus ``evidence.json`` and a summary
on stdout. The raster diff compares the raw grayscale buffers row by row, so
the spike runs with no optional dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pikepdf
import pypdfium2 as pdfium

from ubt.adapters.pdf.cels_inject import (
    DEFAULT_MAX_TRACKING_EM,
    FontMetrics,
    InjectionUnsupported,
    LineSite,
    advance_width,
    encode_probe,
    inject_line,
    replace_stream,
    scan_lines,
    solve_horizontal,
)
from ubt.adapters.pdf.strip_pdf_text import cast_instructions
from ubt.core.layout import (
    Box,
    ColumnTrack,
    ElasticFlowSegment,
    assign_lines,
    detect_columns,
    pin_from,
    track_overruns,
)

#: Words the probe is grown from. Plain ASCII on purpose: the corpus fonts are
#: single-byte Type1 without ``/ToUnicode``, so a non-Latin probe cannot ride
#: the source font at all (that is the spike's font-coverage arm, not a test).
_PROBE_WORDS: tuple[str, ...] = (
    "adaptive",
    "delivery",
    "probe",
    "keeps",
    "its",
    "baseline",
    "inside",
    "the",
    "column",
    "while",
    "tracking",
    "and",
    "scaling",
    "are",
    "measured",
)

#: Arm A aims at this multiple of the source line's width: a ~6% expansion is
#: inside the tracking budget, so the fit must spend tracking only.
TRACKING_ARM_RATIO: float = 1.06

#: Arm B forces ``Tz``: the tracking budget is cut to 0.02 em so tracking
#: cannot pay for the expansion, and the expansion ratio is searched upward
#: until the fit needs a scale. The *mildest* scaling ratio is reported, not the
#: most dramatic one: a probe pinned to the search ceiling proves that the knob
#: moved, but says nothing about whether a small scale is honoured.
_TZ_ARM_TRACKING_EM: float = 0.02
_TZ_ARM_RATIO_RANGE: tuple[float, ...] = tuple(1.05 + 0.01 * step for step in range(56))

#: Graphics-state operators that must be preserved byte-for-byte in count when
#: a text line is rewritten. A text replacement that changes any of these has
#: touched colour, clipping, transparency or an embedded form.
_GRAPHICS_OPS: tuple[str, ...] = (
    "q",
    "Q",
    "cm",
    "re",
    "W",
    "n",
    "rg",
    "RG",
    "g",
    "G",
    "k",
    "K",
    "cs",
    "CS",
    "sc",
    "SC",
    "scn",
    "SCN",
    "gs",
    "Do",
    "BI",
)

#: The last glyph's right side bearing paints past the advance the fit
#: reserves. Measured 0.59pt on p12's scaling arm (456.92 painted against an
#: ``x_right`` of 456.33 at a 334.777pt advance), so the spike budgets 1.0pt of
#: ink pad and Stage 3A must do the same.
_TRAILING_INK_PAD_PT: float = 1.0

_RENDER_SCALE: float = 3.0
_PIXEL_PAD_PT: float = 1.0


@dataclass(frozen=True)
class Verification:
    """What the delivered PDF says about the injected line.

    ``measured_width_pt`` is ink width and ``expected_width_pt`` is the advance
    the fit reserved; they differ by the trailing side bearing, which is the
    reason the fit budget carries :data:`_TRAILING_INK_PAD_PT`.
    """

    probe_found: bool
    measured_width_pt: float
    expected_width_pt: float
    natural_width_pt: float
    right_edge: float
    left_edge: float
    inside_track: bool
    inside_source_band: bool
    band_overflow_pt: float
    compression_ratio: float


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--pages", nargs="+", type=int, required=True, help="1-indexed pages")
    parser.add_argument("--out-dir", type=Path, default=Path("output/adaptive_delivery/g3"))
    args = parser.parse_args(argv)

    pdf_path: Path = args.pdf
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    evidence: dict[str, Any] = {
        "spike": "G3 CELS feasibility",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pdf": str(pdf_path),
        "pdf_sha256": digest,
        "pages": {},
    }

    for page_number in args.pages:
        page_evidence, failures = _page_spike(pdf_path, page_number, out_dir)
        evidence["pages"][str(page_number)] = page_evidence
        _print_page(page_number, page_evidence)
        if failures:
            evidence.setdefault("failures", []).extend(
                f"p{page_number}: {failure}" for failure in failures
            )

    evidence["font_coverage"] = _font_coverage(pdf_path, args.pages)
    evidence["verdict"] = "no-go" if evidence.get("failures") else "go"
    evidence_path = out_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(f"\nevidence: {evidence_path}")
    print(f"verdict: {evidence['verdict']}")
    return 0 if evidence["verdict"] == "go" else 1


def _page_spike(
    pdf_path: Path, page_number: int, out_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    with pikepdf.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        lines = scan_lines(page)
        page_width = float(page.MediaBox[2]) - float(page.MediaBox[0])
        page_height = float(page.MediaBox[3]) - float(page.MediaBox[1])

    boxes = [line.box() for line in lines if line.width > 0.0 and line.font_size > 0.0]
    detection = detect_columns(boxes, page_width=page_width, page_height=page_height)
    assignments, leftover = assign_lines(boxes, detection.tracks)
    overruns = track_overruns(detection.tracks, boxes)
    pins = [pin_from(box) for box in leftover]
    segments = [
        ElasticFlowSegment(
            track_index=index,
            y_top=max(box.y1 for box in items),
            y_bottom=min(box.y0 for box in items),
            source=items,
        )
        for index, items in assignments.items()
        if items
    ]

    columns = {
        "page_size": [round(page_width, 2), round(page_height, 2)],
        "tracks": [asdict(track) for track in detection.tracks],
        "ink_extent": [round(value, 2) for value in detection.ink_extent],
        "accepted_gutters": [asdict(gap) for gap in detection.accepted_gutters],
        "rejected_gutters": [asdict(gap) for gap in detection.rejected_gutters],
        "full_width_pins": [
            {"box": asdict(pin.box), "kind": pin.kind, "flow_beside": pin.flow_beside}
            for pin in pins
        ],
        "segments": [
            {
                "track_index": segment.track_index,
                "y_top": round(segment.y_top, 2),
                "y_bottom": round(segment.y_bottom, 2),
                "height_pt": round(segment.height, 2),
                "source_lines": len(segment.source),
            }
            for segment in segments
        ],
        "lines_assigned": {str(index): len(items) for index, items in assignments.items()},
        "lines_leftover": len(leftover),
        "source_overruns": len(overruns),
        "line_count": len(lines),
    }
    if overruns:
        failures.append(f"{len(overruns)} source lines already paint past their track x_right")

    target = _pick_target(lines, detection.tracks)
    if target is None:
        return {"columns": columns, "error": "no usable target line"}, ["no usable target line"]

    track = _owning_track(target, detection.tracks)
    available_pt = track.x1 - target.x0
    metrics = _metrics_for_line(pdf_path, page_number, target.font_key)

    font = {
        "resource": target.font_key,
        "subtype": _font_field(pdf_path, page_number, target.font_key, "/Subtype"),
        "encoding": _font_field(pdf_path, page_number, target.font_key, "/Encoding"),
        "to_unicode": _font_field(pdf_path, page_number, target.font_key, "/ToUnicode") is not None,
        "size_pt": round(target.font_size, 3),
    }

    injection, arm_failures = _injection_arm(
        pdf_path,
        page_number,
        target,
        track,
        metrics,
        available_pt,
        ratio=TRACKING_ARM_RATIO,
        out_path=out_dir / f"p{page_number}_injected.pdf",
        label="tracking",
    )
    failures.extend(arm_failures)

    tz_ratio = _ratio_for_scale(target, metrics, available_pt)
    tz_arm, tz_failures = _injection_arm(
        pdf_path,
        page_number,
        target,
        track,
        metrics,
        available_pt,
        ratio=tz_ratio,
        max_tracking_em=_TZ_ARM_TRACKING_EM,
        out_path=out_dir / f"p{page_number}_scaled.pdf",
        label="tz",
    )
    # A tz arm that cannot fit is evidence for Stage 4, not a G3 failure.
    failures.extend(failure for failure in tz_failures if "deficit" not in failure)

    leading, leading_failures = _leading_probe(
        pdf_path,
        page_number,
        lines,
        target,
        track,
        metrics,
        available_pt,
        out_path=out_dir / f"p{page_number}_leading.pdf",
    )
    failures.extend(leading_failures)

    result = {
        "columns": columns,
        "font": font,
        "target_line": {
            "text": target.text[:80],
            "x0": round(target.x0, 2),
            "x1": round(target.x1, 2),
            "baseline": round(target.baseline, 2),
            "width_pt": round(target.width, 2),
            "available_pt": round(available_pt, 2),
            "track_index": track.index,
        },
        "injection": injection,
        "tz": tz_arm,
        "leading_probe": leading,
        "graphics_state": _graphics_state(pdf_path, page_number, out_dir),
    }
    return result, failures


def _injection_arm(
    pdf_path: Path,
    page_number: int,
    target: LineSite,
    track: ColumnTrack,
    metrics: FontMetrics,
    available_pt: float,
    *,
    ratio: float,
    out_path: Path,
    label: str,
    max_tracking_em: float = DEFAULT_MAX_TRACKING_EM,
) -> tuple[dict[str, Any], list[str]]:
    probe = _build_probe(metrics, target.font_size, target.width * ratio)
    budget = available_pt - _TRAILING_INK_PAD_PT
    fit = solve_horizontal(
        probe, metrics, target.font_size, budget, max_tracking_em=max_tracking_em
    )

    with pikepdf.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        fresh = _find_line(scan_lines(page), target)
        stream = inject_line(
            page,
            fresh,
            probe,
            horizontal_scale=fit.horizontal_scale,
            char_spacing=fit.char_spacing,
        )
        replace_stream(page, stream, pdf)
        pdf.save(out_path)

    original = _render_gray(pdf_path, page_number)
    mutated = _render_gray(out_path, page_number)
    band = Box(
        x0=target.x0 - _PIXEL_PAD_PT,
        y0=target.baseline - 0.30 * target.font_size,
        x1=max(target.x1, target.x0 + fit.width_pt) + _PIXEL_PAD_PT,
        y1=target.baseline + 0.90 * target.font_size,
    )
    diff = _raster_diff(original, mutated, _page_height(pdf_path, page_number), band)

    verification = _verify_probe(out_path, page_number, probe, track, target, fit)
    failures: list[str] = []
    if not verification.probe_found:
        failures.append(f"{label} arm: probe text is not extractable (not copy/search safe)")
    if not verification.inside_track:
        failures.append(
            f"{label} arm: probe paints past x_right ({verification.right_edge:.2f} > {track.x1:.2f})"
        )
    if not verification.inside_source_band:
        failures.append(
            f"{label} arm: probe ink left the source baseline band by "
            f"{verification.band_overflow_pt:.2f}pt (>0.5)"
        )
    if diff.get("outside_pad_pixels"):
        failures.append(
            f"{label} arm: {diff['outside_pad_pixels']} pixels changed outside the line band"
        )

    return (
        {
            "probe_text": probe,
            "probe_chars": len(probe),
            "probe_ratio": round(ratio, 3),
            "natural_width_pt": round(advance_width(probe, metrics, target.font_size), 3),
            "advance_budget_pt": round(budget, 3),
            "ink_pad_pt": _TRAILING_INK_PAD_PT,
            "fit": asdict(fit),
            "verification": asdict(verification),
            "raster_diff": diff,
            "output": str(out_path),
        },
        failures,
    )


def _leading_probe(
    pdf_path: Path,
    page_number: int,
    lines: Sequence[LineSite],
    target: LineSite,
    track: ColumnTrack,
    metrics: FontMetrics,
    available_pt: float,
    *,
    out_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    """``TL`` + ``T*`` twin: the same string twice, one leading apart."""
    probe = _build_probe(metrics, target.font_size, target.width * TRACKING_ARM_RATIO)
    fit = solve_horizontal(probe, metrics, target.font_size, available_pt - _TRAILING_INK_PAD_PT)
    leading = _dominant_leading(lines)

    with pikepdf.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        fresh = _find_line(scan_lines(page), target)
        stream = inject_line(
            page,
            fresh,
            probe,
            horizontal_scale=fit.horizontal_scale,
            char_spacing=fit.char_spacing,
            leading_probe=(probe, leading),
        )
        replace_stream(page, stream, pdf)
        pdf.save(out_path)

    runs = _find_runs(out_path, page_number, probe)
    failures: list[str] = []
    if len(runs) != 2:
        return (
            {
                "probe_text": probe,
                "leading_pt": round(leading, 3),
                "runs_found": len(runs),
                "output": str(out_path),
            },
            [f"leading arm: expected 2 identical runs, extracted {len(runs)}"],
        )
    first, second = runs
    dy = first["y0"] - second["y0"]
    dx = abs(first["x0"] - second["x0"])
    deviation = abs(dy - leading)
    if deviation > 0.5:
        failures.append(f"leading arm: T* moved {dy:.2f}pt, TL said {leading:.2f}pt")
    if dx > 0.5:
        failures.append(f"leading arm: T* shifted x by {dx:.2f}pt (it must not)")
    return (
        {
            "probe_text": probe,
            "leading_pt": round(leading, 3),
            "runs_found": 2,
            "measured_dy_pt": round(dy, 3),
            "baseline_deviation_pt": round(deviation, 3),
            "measured_dx_pt": round(dx, 3),
            "first_run": first,
            "second_run": second,
            "output": str(out_path),
        },
        failures,
    )


def _verify_probe(
    pdf_path: Path,
    page_number: int,
    probe: str,
    track: ColumnTrack,
    target: LineSite,
    fit: Any,
) -> Verification:
    runs = _find_runs(pdf_path, page_number, probe)
    if len(runs) != 1:
        return Verification(False, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0, 0.0)
    run = runs[0]
    natural = advance_width(
        probe, _metrics_for_line(pdf_path, page_number, target.font_key), target.font_size
    )
    band = (target.baseline - 0.30 * target.font_size, target.baseline + 0.90 * target.font_size)
    overflow = max(0.0, band[0] - run["y0"], run["y1"] - band[1])
    measured = run["x1"] - run["x0"]
    return Verification(
        probe_found=True,
        measured_width_pt=round(measured, 3),
        expected_width_pt=round(fit.width_pt, 3),
        natural_width_pt=round(natural, 3),
        right_edge=round(run["x1"], 3),
        left_edge=round(run["x0"], 3),
        inside_track=run["x1"] <= track.x1 + 0.5,
        inside_source_band=overflow <= 0.5,
        band_overflow_pt=round(overflow, 3),
        compression_ratio=round(measured / natural, 4) if natural else 0.0,
    )


def _ratio_for_scale(target: LineSite, metrics: FontMetrics, available_pt: float) -> float:
    """The mildest expansion ratio whose fit actually needs ``Tz``.

    The probe grows a word at a time (~25pt steps), so a target scale cannot be
    hit exactly; picking the largest scale strictly below 1.0 keeps the arm's
    claim narrow — "this much scaling was applied and the page still measured
    correctly" — instead of pinning to a bound.
    """
    budget = available_pt - _TRAILING_INK_PAD_PT
    best: tuple[float, float] | None = None
    for ratio in _TZ_ARM_RATIO_RANGE:
        probe = _build_probe(metrics, target.font_size, target.width * ratio)
        fit = solve_horizontal(
            probe, metrics, target.font_size, budget, max_tracking_em=_TZ_ARM_TRACKING_EM
        )
        if not fit.fits or fit.horizontal_scale >= 1.0:
            continue
        if best is None or fit.horizontal_scale > best[1]:
            best = (ratio, fit.horizontal_scale)
    return best[0] if best is not None else _TZ_ARM_RATIO_RANGE[-1]


def _build_probe(metrics: FontMetrics, size_pt: float, target_pt: float) -> str:
    text = "PROBE"
    index = 0
    while True:
        candidate = f"{text} {_PROBE_WORDS[index % len(_PROBE_WORDS)]}"
        if advance_width(candidate, metrics, size_pt) > target_pt:
            return text
        text = candidate
        index += 1


def _pick_target(lines: Sequence[LineSite], tracks: Sequence[ColumnTrack]) -> LineSite | None:
    candidates = [
        line
        for line in lines
        if line.width > 60.0
        and line.font_size >= 6.0
        and _owning_track_or_none(line, tracks) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda line: line.width)


def _owning_track(line: LineSite, tracks: Sequence[ColumnTrack]) -> ColumnTrack:
    track = _owning_track_or_none(line, tracks)
    if track is None:
        raise InjectionUnsupported(f"line at x={line.x0:.1f} belongs to no track")
    return track


def _owning_track_or_none(line: LineSite, tracks: Sequence[ColumnTrack]) -> ColumnTrack | None:
    for track in tracks:
        if track.contains_horizontally(line.box()):
            return track
    return None


def _find_line(lines: Sequence[LineSite], target: LineSite) -> LineSite:
    for line in lines:
        if abs(line.x0 - target.x0) < 0.01 and abs(line.baseline - target.baseline) < 0.01:
            return line
    raise InjectionUnsupported(f"target line at x={target.x0:.2f} vanished after reparse")


def _dominant_leading(lines: Sequence[LineSite]) -> float:
    gaps = sorted(
        round(lines[i].baseline - lines[i + 1].baseline, 2) for i in range(len(lines) - 1)
    )
    gaps = [gap for gap in gaps if 8.0 <= gap <= 30.0]
    return statistics.median(gaps) if gaps else 12.0


def _metrics_for_line(pdf_path: Path, page_number: int, font_key: str) -> FontMetrics:
    with pikepdf.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        fonts = page.Resources["/Font"]
        return FontMetrics.from_font(fonts[font_key])


def _object(value: Any) -> Any:
    """Dereference an indirect object; a direct dictionary is already one."""
    getter = getattr(value, "get_object", None)
    return getter() if callable(getter) else value


def _font_field(pdf_path: Path, page_number: int, font_key: str, field: str) -> Any:
    with pikepdf.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        font = _object(page.Resources["/Font"][font_key])
        value = font.get(field)
        return None if value is None else str(value)


def _graphics_state(pdf_path: Path, page_number: int, out_dir: Path) -> dict[str, Any]:
    """Count graphics operators before and after a rewrite must be identical."""
    result: dict[str, Any] = {}
    with pikepdf.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        before = Counter(
            str(inst[1]) for inst in cast_instructions(pikepdf.parse_content_stream(page))
        )
    mutated = out_dir / f"p{page_number}_injected.pdf"
    if not mutated.exists():
        return {"status": "unavailable"}
    with pikepdf.open(mutated) as pdf:
        page = pdf.pages[page_number - 1]
        after = Counter(
            str(inst[1]) for inst in cast_instructions(pikepdf.parse_content_stream(page))
        )
    mismatched: dict[str, list[int]] = {}
    for op in _GRAPHICS_OPS:
        if before.get(op, 0) != after.get(op, 0):
            mismatched[op] = [before.get(op, 0), after.get(op, 0)]
    if mismatched:
        result["mismatched"] = mismatched
    result["preserved"] = not mismatched
    result["ops_checked"] = list(_GRAPHICS_OPS)
    return result


def _font_coverage(pdf_path: Path, pages: Sequence[int]) -> dict[str, Any]:
    """The known G4 blocker: the source fonts cannot carry the target script."""
    findings: dict[str, Any] = {"target_script_probe": "自适应文本测试"}
    try:
        encode_probe("自适应文本测试")
    except InjectionUnsupported as exc:
        findings["status"] = "unsupported"
        findings["reason"] = str(exc)
    else:
        findings["status"] = "unexpectedly-encodable"

    per_page: dict[str, Any] = {}
    with pikepdf.open(pdf_path) as pdf:
        for page_number in pages:
            page = pdf.pages[page_number - 1]
            fonts = page.Resources.get("/Font")
            entries: dict[str, Any] = {}
            for key in fonts or {}:
                font = _object(fonts[key])
                entries[str(key)] = {
                    "subtype": str(font.get("/Subtype", "")),
                    "encoding": _short(font.get("/Encoding", "")),
                    "to_unicode": font.get("/ToUnicode") is not None,
                    "embedded": any(
                        name in font.get("/FontDescriptor", {})
                        for name in ("/FontFile", "/FontFile2", "/FontFile3")
                    ),
                }
            per_page[str(page_number)] = entries
    findings["fonts"] = per_page
    findings["implication"] = (
        "A non-Latin translation needs an embedded, subsetted target font with its own "
        "/Encoding and /ToUnicode; the source Type1 fonts cannot encode it. That is G4 "
        "Stage 5 work, not a G3 blocker: the mechanism above is proven with the source font."
    )
    return findings


@dataclass(frozen=True)
class GrayBitmap:
    """A rendered page as raw grayscale bytes (no numpy dependency)."""

    width: int
    height: int
    stride: int
    data: bytes


def _render_gray(pdf_path: Path, page_number: int) -> GrayBitmap | None:
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        page = document[page_number - 1]
        bitmap = page.render(scale=_RENDER_SCALE, grayscale=True)
        try:
            return GrayBitmap(
                width=bitmap.width,
                height=bitmap.height,
                stride=bitmap.stride,
                data=bytes(bitmap.buffer),
            )
        finally:
            bitmap.close()
    finally:
        document.close()


def _raster_diff(
    original: GrayBitmap | None,
    mutated: GrayBitmap | None,
    page_height: float,
    allowed: Box,
) -> dict[str, Any]:
    """Changed pixels must all fall inside the rewritten line's band."""
    if original is None or mutated is None:
        return {"status": "skipped", "reason": "render failed"}
    if (original.width, original.height) != (mutated.width, mutated.height):
        return {
            "status": "size-mismatch",
            "original": [original.width, original.height],
            "mutated": [mutated.width, mutated.height],
        }

    x_low = allowed.x0 * _RENDER_SCALE
    x_high = allowed.x1 * _RENDER_SCALE
    row_low = (page_height - allowed.y1) * _RENDER_SCALE
    row_high = (page_height - allowed.y0) * _RENDER_SCALE
    changed_pixels = 0
    outside = 0
    bbox: list[int] | None = None
    for row in range(original.height):
        a = original.data[row * original.stride : row * original.stride + original.width]
        b = mutated.data[row * mutated.stride : row * mutated.stride + mutated.width]
        if a == b:
            continue
        for col in range(original.width):
            if a[col] == b[col]:
                continue
            changed_pixels += 1
            bbox = (
                [col, row, col, row]
                if bbox is None
                else [
                    min(bbox[0], col),
                    min(bbox[1], row),
                    max(bbox[2], col),
                    max(bbox[3], row),
                ]
            )
            if not (x_low <= col <= x_high and row_low <= row <= row_high):
                outside += 1
    if changed_pixels == 0:
        return {"status": "identical", "changed_pixels": 0, "outside_pad_pixels": 0}
    return {
        "status": "measured",
        "changed_pixels": changed_pixels,
        "outside_pad_pixels": outside,
        "changed_bbox_px": bbox,
        "allowed_px": [round(x_low, 1), round(row_low, 1), round(x_high, 1), round(row_high, 1)],
    }


def _page_height(pdf_path: Path, page_number: int) -> float:
    with pikepdf.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        return float(page.MediaBox[3]) - float(page.MediaBox[1])


def _find_runs(pdf_path: Path, page_number: int, needle: str) -> list[dict[str, float]]:
    """Locate every extracted run of ``needle`` and return its ink box."""
    chars: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        page = document[page_number - 1]
        textpage = page.get_textpage()
        for index in range(textpage.count_chars()):
            chars.append(textpage.get_text_range(index, 1))
            boxes.append(tuple(textpage.get_charbox(index)))
    finally:
        document.close()

    compact = "".join(chars).replace("\r", "").replace("\n", "").replace(" ", "")
    target = needle.replace(" ", "")
    starts: list[int] = []
    cursor = compact.find(target)
    while cursor != -1:
        starts.append(cursor)
        cursor = compact.find(target, cursor + 1)
    if not starts:
        return []

    # Map normalized offsets back to char indices.
    normalized_at: list[int | None] = []
    offset = 0
    for char in chars:
        if char in {"\r", "\n", " "}:
            normalized_at.append(None)
        else:
            normalized_at.append(offset)
            offset += 1
    runs: list[dict[str, float]] = []
    for start in starts:
        indices = [
            index
            for index, position in enumerate(normalized_at)
            if position is not None and start <= position < start + len(target)
        ]
        if not indices:
            continue
        run_boxes = [boxes[index] for index in indices]
        runs.append(
            {
                "x0": round(min(box[0] for box in run_boxes), 3),
                "y0": round(min(box[1] for box in run_boxes), 3),
                "x1": round(max(box[2] for box in run_boxes), 3),
                "y1": round(max(box[3] for box in run_boxes), 3),
                "chars": len(indices),
            }
        )
    return runs


def _short(value: Any) -> str:
    """A compact rendering of a PDF value for evidence JSON."""
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _print_page(page_number: int, result: dict[str, Any]) -> None:
    print(f"\n=== page {page_number} ===")
    columns = result.get("columns", {})
    print(f"  tracks: {len(columns.get('tracks', []))}  lines: {columns.get('line_count')}")
    for gap in columns.get("rejected_gutters", []):
        print(f"  refused gutter {gap['x0']:.1f}–{gap['x1']:.1f}: {gap['reason']}")
    if "target_line" not in result:
        print(f"  error: {result.get('error')}")
        return
    target = result["target_line"]
    print(f"  target: x={target['x0']} width={target['width_pt']} avail={target['available_pt']}")
    font = result["font"]
    print(f"  font: {font['resource']} {font['subtype']} toUnicode={font['to_unicode']}")
    for arm in ("injection", "tz"):
        entry = result[arm]
        fit = entry["fit"]
        verification = entry["verification"]
        print(
            f"  {arm} arm: ratio={entry['probe_ratio']} scale={fit['horizontal_scale']} "
            f"tc={fit['char_spacing']} fits={fit['fits']} width={fit['width_pt']}"
        )
        print(
            f"    found={verification['probe_found']} right={verification['right_edge']} "
            f"inside={verification['inside_track']} "
            f"measured/expected={verification['measured_width_pt']}/{verification['expected_width_pt']} "
            f"compression={verification['compression_ratio']}"
        )
        raster = entry["raster_diff"]
        print(f"    raster: {raster.get('status')} outside={raster.get('outside_pad_pixels')}")
    leading = result["leading_probe"]
    print(
        f"  leading probe: TL={leading.get('leading_pt')} dy={leading.get('measured_dy_pt')} "
        f"deviation={leading.get('baseline_deviation_pt')}"
    )
    print(f"  graphics state preserved: {result['graphics_state'].get('preserved')}")


if __name__ == "__main__":
    sys.exit(main())
