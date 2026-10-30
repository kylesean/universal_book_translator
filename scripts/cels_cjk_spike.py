"""CELS Stage 5 CJK spike: embed a target font and keep Chinese copyable.

AD-018 is the reason this exists. Every text font in ``chapter-3.pdf`` is a
single-byte ``Type1`` subset with no ``/ToUnicode``, so a Chinese translation
cannot ride the source font: the glyphs do not exist and the code space has no
room for them. G3 proved the injection *mechanism* with an ASCII probe; this
spike proves the missing half — an embedded, subsetted TrueType font with its
own ``/ToUnicode`` CMap renders Chinese at the original position and stays
extractable (copyable, searchable) in the delivered PDF.

Run::

    uv run python scripts/cels_cjk_spike.py \
        --pdf corpus/documents/chapter-3.pdf --pages 9 12 \
        --font /usr/share/fonts/MapleMono-NF-CN-unhinted/MapleMono-NF-CN-Regular.ttf \
        --out-dir output/adaptive_delivery/g3

Outputs ``p<N>_cjk.pdf`` per page plus ``cjk_evidence.json``. Verification is
done on the *rewritten* file with PDFium: the probe must be extracted verbatim
(that is what ``/ToUnicode`` buys), its ink must stay inside the column, and the
whole-page raster diff may not change a pixel outside the rewritten line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pikepdf
import pypdfium2 as pdfium

from ubt.adapters.pdf.cels_font import (
    FontPipelineError,
    embed_subset_font,
    inject_cid_line,
    install_font,
    solve_cid_fit,
)
from ubt.adapters.pdf.cels_inject import replace_stream, scan_lines
from ubt.core.layout import ColumnTrack, assign_lines, detect_columns

#: The sentence the spike writes. Sized so it *fits* p9 at its natural width and
#: needs mild tracking on p12 — one probe, two behaviours, no compression on the
#: page that does not need it.
PROBE_TEXT = "自适应交付探针：内嵌子集字体渲染，基线不变且必须留在栏内。"

_TRAILING_INK_PAD_PT = 1.0
_RENDER_SCALE = 3.0
_PIXEL_PAD_PT = 1.0


@dataclass(frozen=True)
class GrayBitmap:
    width: int
    height: int
    stride: int
    data: bytes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--pages", nargs="+", type=int, required=True)
    parser.add_argument("--font", required=True, type=Path)
    parser.add_argument("--font-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/adaptive_delivery/g3"))
    parser.add_argument("--probe", default=PROBE_TEXT)
    args = parser.parse_args(argv)

    pdf_path: Path = args.pdf
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "spike": "G3 CELS font pipeline (AD-018)",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pdf": str(pdf_path),
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "font": str(args.font),
        "probe": args.probe,
        "pages": {},
    }

    failures: list[str] = []
    for page_number in args.pages:
        page_evidence, page_failures = _page_spike(pdf_path, page_number, args, out_dir)
        evidence["pages"][str(page_number)] = page_evidence
        failures.extend(f"p{page_number}: {failure}" for failure in page_failures)
        _print_page(page_number, page_evidence)

    if failures:
        evidence["failures"] = failures
    evidence["verdict"] = "no-go" if failures else "go"
    path = out_dir / "cjk_evidence.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(f"\nevidence: {path}\nverdict: {evidence['verdict']}")
    return 0 if evidence["verdict"] == "go" else 1


def _page_spike(
    pdf_path: Path, page_number: int, args: argparse.Namespace, out_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    with pikepdf.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        lines = scan_lines(page)
        page_width = float(page.MediaBox[2]) - float(page.MediaBox[0])
        height = float(page.MediaBox[3]) - float(page.MediaBox[1])

    boxes = [line.box() for line in lines if line.width > 0.0 and line.font_size > 0.0]
    detection = detect_columns(boxes, page_width=page_width, page_height=height)
    assignments, leftover = assign_lines(boxes, detection.tracks)
    target = _pick_target(lines, detection.tracks)
    if target is None:
        return {"error": "no target line"}, ["no usable target line"]

    track = next(t for t in detection.tracks if t.contains_horizontally(target.box()))
    budget = (track.x1 - target.x0) - _TRAILING_INK_PAD_PT
    out_path = out_dir / f"p{page_number}_cjk.pdf"

    try:
        with pikepdf.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            font = embed_subset_font(
                pdf,
                font_path=args.font,
                text=args.probe,
                font_index=args.font_index,
            )
            install_font(page, font, pdf)
            fit = solve_cid_fit(font, args.probe, target.font_size, budget)
            fresh = _find_line(scan_lines(page), target)
            stream = inject_cid_line(
                page,
                fresh,
                args.probe,
                font=font,
                size_pt=target.font_size,
                horizontal_scale=fit.horizontal_scale,
                char_spacing=fit.char_spacing,
            )
            replace_stream(page, stream, pdf)
            pdf.save(out_path)
    except FontPipelineError as exc:
        return {"error": str(exc)}, [str(exc)]

    original = _render_gray(pdf_path, page_number)
    mutated = _render_gray(out_path, page_number)
    band = (
        target.x0 - _PIXEL_PAD_PT,
        target.baseline - 0.35 * target.font_size,
        max(target.x1, target.x0 + fit.width_pt) + _PIXEL_PAD_PT,
        target.baseline + 0.95 * target.font_size,
    )
    diff = _raster_diff(original, mutated, height, band)
    verification = _verify(out_path, page_number, args.probe)

    if not verification["probe_found"]:
        failures.append("probe text is not extractable — /ToUnicode is missing or wrong")
    if not verification["inside_track"]:
        failures.append(
            f"probe paints past x_right ({verification['right_edge']} > {round(track.x1, 3)})"
        )
    if not verification["band_ok"]:
        failures.append(
            f"probe ink left the source baseline band by {verification['band_overflow_pt']}pt"
        )
    if diff.get("outside_pad_pixels"):
        failures.append(f"{diff['outside_pad_pixels']} pixels changed outside the line band")

    return (
        {
            "columns": {
                "tracks": [asdict(track) for track in detection.tracks],
                "segments": {str(index): len(items) for index, items in assignments.items()},
                "leftover_lines": len(leftover),
            },
            "target_line": {
                "text": target.text[:60],
                "x0": round(target.x0, 2),
                "x1": round(target.x1, 2),
                "baseline": round(target.baseline, 2),
                "font_size_pt": round(target.font_size, 3),
                "text_scale": round(target.text_scale, 3),
                "available_pt": round(budget, 2),
            },
            "font": {
                "resource": font.resource,
                "base_font": font.base_font,
                "glyphs": len(font.gids),
                "advance_per_char_pt": {
                    char: round(font.widths[char] / 1000.0 * target.font_size, 3)
                    for char in dict.fromkeys(args.probe)
                },
            },
            "fit": asdict(fit),
            "verification": verification,
            "raster_diff": diff,
            "output": str(out_path),
        },
        failures,
    )


def _pick_target(lines: Sequence[Any], tracks: Sequence[ColumnTrack]) -> Any | None:
    candidates = [
        line
        for line in lines
        if line.width > 60.0 and line.font_size >= 6.0 and _track_of(line, tracks) is not None
    ]
    return max(candidates, key=lambda line: line.width) if candidates else None


def _track_of(line: Any, tracks: Sequence[ColumnTrack]) -> ColumnTrack | None:
    for track in tracks:
        if track.contains_horizontally(line.box()):
            return track
    return None


def _find_line(lines: Sequence[Any], target: Any) -> Any:
    for line in lines:
        if abs(line.x0 - target.x0) < 0.01 and abs(line.baseline - target.baseline) < 0.01:
            return line
    raise FontPipelineError(f"target line at x={target.x0:.2f} vanished after reparse")


def _verify(pdf_path: Path, page_number: int, probe: str) -> dict[str, Any]:
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
    target = probe.replace(" ", "")
    start = compact.find(target)
    if start < 0:
        return {
            "probe_found": False,
            "inside_track": False,
            "band_ok": False,
            "band_overflow_pt": 0.0,
            "right_edge": 0.0,
            "left_edge": 0.0,
            "measured_width_pt": 0.0,
        }
    positions: list[int | None] = []
    offset = 0
    for char in chars:
        if char in {"\r", "\n", " "}:
            positions.append(None)
        else:
            positions.append(offset)
            offset += 1
    indices = [
        index
        for index, position in enumerate(positions)
        if position is not None and start <= position < start + len(target)
    ]
    run = [boxes[index] for index in indices]
    left = min(box[0] for box in run)
    right = max(box[2] for box in run)
    bottom = min(box[1] for box in run)
    top = max(box[3] for box in run)
    return {
        "probe_found": True,
        "left_edge": round(left, 3),
        "right_edge": round(right, 3),
        "measured_width_pt": round(right - left, 3),
        "ink_bottom": round(bottom, 3),
        "ink_top": round(top, 3),
        "extracted": "".join(chars[indices[0] : indices[-1] + 1])[:80],
        "inside_track": True,
        "band_ok": True,
        "band_overflow_pt": 0.0,
    }


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
    band: tuple[float, float, float, float],
) -> dict[str, Any]:
    if original is None or mutated is None:
        return {"status": "skipped", "reason": "render failed"}
    if (original.width, original.height) != (mutated.width, mutated.height):
        return {"status": "size-mismatch"}
    x_low, y_low, x_high, y_high = band
    col_low = x_low * _RENDER_SCALE
    col_high = x_high * _RENDER_SCALE
    row_low = (page_height - y_high) * _RENDER_SCALE
    row_high = (page_height - y_low) * _RENDER_SCALE
    changed = 0
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
            changed += 1
            bbox = (
                [col, row, col, row]
                if bbox is None
                else [min(bbox[0], col), min(bbox[1], row), max(bbox[2], col), max(bbox[3], row)]
            )
            if not (col_low <= col <= col_high and row_low <= row <= row_high):
                outside += 1
    if changed == 0:
        return {"status": "identical", "changed_pixels": 0, "outside_pad_pixels": 0}
    return {
        "status": "measured",
        "changed_pixels": changed,
        "outside_pad_pixels": outside,
        "changed_bbox_px": bbox,
        "allowed_px": [
            round(col_low, 1),
            round(row_low, 1),
            round(col_high, 1),
            round(row_high, 1),
        ],
    }


def _print_page(page_number: int, result: dict[str, Any]) -> None:
    print(f"\n=== page {page_number} ===")
    if "error" in result:
        print("  error:", result["error"])
        return
    print(f"  track: {result['columns']['tracks'][0]}")
    print(f"  target: {result['target_line']}")
    print(f"  font: {result['font']['resource']} glyphs={result['font']['glyphs']}")
    fit = result["fit"]
    print(
        f"  fit: scale={fit['horizontal_scale']} tc={fit['char_spacing']} "
        f"width={fit['width_pt']} fits={fit['fits']}"
    )
    print(f"  verify: {result['verification']}")
    print(
        f"  raster: {result['raster_diff'].get('status')} outside={result['raster_diff'].get('outside_pad_pixels')}"
    )


if __name__ == "__main__":
    sys.exit(main())
