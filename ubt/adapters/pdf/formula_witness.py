"""Deterministic visual witness for display formulas.

The LaTeX->Typst converter can emit output that compiles and passes every
text-level gate yet still differs from the source equation (wrong glyph,
dropped term, changed subscript). The only ground truth is the rendered page:
this module rasterizes the emitted formula, crops the same region from the
source PDF and compares *structure* — ink density, bounding-box aspect,
connected-component magnitude and baseline position. Pixel identity is never
compared: the Typst fonts legitimately differ from the book's fonts, and
measuring pixels would report those differences as errors.

The witness is zero-token and fail-open by design: any measurement problem
returns ``unwitnessable`` and the caller keeps whatever it had. A formula that
fails a structural check is swapped for its source graphic by the caller
(``TypstReconstructor._witness_math_lines``), which is lossless.

Calibration on chapter-3's 68 rendered display formulas (tight source crops
with glue-bled neighbour lines trimmed): the surviving flags are
``pdf_main#b0176`` (source is only ``sqrt(T_1)`` but the emitted formula added
two extra terms) and ``pdf_main#b0178`` (a one-line source reflowed into two
stacked rows). The source box's composition (equation number at the margin,
the book's lighter font weight) makes ink density and baseline position too
noisy to use as failure criteria on this corpus, so they are not checked. The
witness therefore catches layout-level corruption (a one-line source returning
as stacked rows, a merged/split block, a big term going missing); it does not
catch single-glyph substitution such as ``V_fb`` -> ``V_h``, which needs the
gray-zone VLM tier that is not implemented here.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK
from ubt.core.env import subprocess_env

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from ubt.core.ir.models import IRBlock

logger = logging.getLogger(__name__)

# Raster scale for both sides of the comparison. 150 dpi matches the visual
# scalpel default: enough glyph structure, cheap enough to run per formula.
WITNESS_DPI = 150
# A grayscale pixel at or below this luma counts as ink.
INK_LUMA_MAX = 160
# Pure-Python component analysis scales with pixel count; downscale so one
# wide equation cannot dominate a book render.
MAX_METRIC_WIDTH = 480
# Below this many ink pixels the crop is blank/noise and cannot be judged.
MIN_INK_PIXELS = 24
# Structural tolerances calibrated on chapter-3's 68 rendered display
# formulas (tight source crops): unflagged pairs cluster at
# 0.8-1.4x; the two confirmed defects sit at 0.26x (a one-line source the
# reflow stacks into two rows) and 4.2x with 5.1x components (an OCR block
# whose source is only sqrt(T_1) but which emitted two extra terms). Single
# lines whose source and render differ only in glyph depth (book font vs
# Typst metrics, 0.4x, components 1.0x) are NOT corruption and must pass,
# so the band is asymmetric-wide on that side.
MIN_ASPECT_RATIO = 0.30
MAX_ASPECT_RATIO = 3.0
MIN_COMPONENT_RATIO = 0.30
MAX_COMPONENT_RATIO = 3.0
# Glue-bled neighbour lines inside a formula bbox crop: Docling merges a
# trailing/leading prose line into the formula item, so the crop can carry a
# clipped sliver of that line. It shows up as a short ink band touching the
# crop edge and separated from the formula body by an inter-line gap; a band
# is dropped only when all three signals agree, so rows *inside* a formula
# (aligned environments, cases) are never touched.
_TRIM_MIN_GAP_PT = 4.0
_TRIM_MAX_EDGE_HEIGHT_FRAC = 0.4
_TRIM_EDGE_SLACK_PX = 3


def _extract_math_source(math_line: str) -> str:
    """Reduce an emitted line to the bare ``$ ... $`` math for probing.

    Wrapper forms (``#math.equation(...)``, ``#block[#show ...]``) carry their
    own numbering, which would print an equation tag next to the probe; the
    witness compares formula bodies only, so it always re-wraps the raw math.
    """
    match = re.search(r"\$(.+)\$", math_line, re.DOTALL)
    if match is None:
        return math_line
    return f"$ {match.group(1).strip()} $"


@dataclass
class WitnessResult:
    """Outcome of one formula witness comparison."""

    status: str  # "pass" | "fail" | "unwitnessable"
    findings: list[str] = field(default_factory=list)


@dataclass
class _Metrics:
    density: float
    aspect: float
    components: int
    baseline: float


def _ink_grid(img: PILImage.Image) -> tuple[list[list[bool]], int, int] | None:
    """Binarize an image into an ink grid, downscaled for fast analysis."""
    from PIL import Image

    gray = img.convert("L")
    if gray.width > MAX_METRIC_WIDTH:
        ratio = MAX_METRIC_WIDTH / float(gray.width)
        gray = gray.resize(
            (MAX_METRIC_WIDTH, max(1, int(gray.height * ratio))),
            Image.Resampling.BILINEAR,
        )
    width, height = gray.size
    if width <= 0 or height <= 0:
        return None
    pixels = gray.tobytes()
    grid = [[pixels[y * width + x] <= INK_LUMA_MAX for x in range(width)] for y in range(height)]
    return grid, width, height


def _ink_pixels(grid: list[list[bool]], width: int, height: int) -> list[tuple[int, int]]:
    return [(x, y) for y in range(height) for x in range(width) if grid[y][x]]


def _strip_trailing_number(dark: set[tuple[int, int]], width: int) -> set[tuple[int, int]]:
    """Drop a trailing ink island separated by a wide gap.

    In the source page the equation number sits at the right margin, far from
    the formula body; in the rendered probe there is no number at all ($ ...
    $ is probed without numbering). Without this trim the source metrics span
    formula-to-number (a 25:1 aspect) while the rendered ones span the formula
    alone, and every numbered equation reports a false mismatch.
    """
    if not dark:
        return dark
    columns = sorted({x for x, _ in dark})
    gap_threshold = max(12, width // 12)
    runs: list[tuple[int, int]] = []
    start = prev = columns[0]
    for column in columns[1:]:
        if column - prev > gap_threshold:
            runs.append((start, prev))
            start = column
        prev = column
    runs.append((start, prev))
    if len(runs) < 2:
        return dark
    last_start, last_end = runs[-1]
    before = sum(1 for x, _ in dark if x < last_start)
    after = len(dark) - before
    span = runs[-1][1] - runs[0][0] + 1
    island_width = last_end - last_start + 1
    if after <= 0.5 * len(dark) or island_width <= 0.3 * span:
        return {(x, y) for x, y in dark if x < last_start}
    return dark


def trim_formula_crop(img: PILImage.Image, dpi: int = WITNESS_DPI) -> PILImage.Image:
    """Drop glue-bled neighbour lines from a formula crop (conservative).

    Returns the crop unchanged when no edge band matches every signal, so
    this is a display-only cleanup, never a formula-content edit.
    """
    gray = img.convert("L")
    width, height = gray.size
    if width <= 1 or height <= 1:
        return img
    pixels = gray.tobytes()
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y in range(height):
        row = pixels[y * width : (y + 1) * width]
        ink = any(value <= INK_LUMA_MAX for value in row)
        if ink and start is None:
            start = y
        elif not ink and start is not None:
            bands.append((start, y - 1))
            start = None
    if start is not None:
        bands.append((start, height - 1))
    if len(bands) < 2:
        return img

    gap_px = max(4, round(_TRIM_MIN_GAP_PT * dpi / 72.0))
    core_height = max(end - begin + 1 for begin, end in bands)

    def _edge_junk(band: tuple[int, int], neighbour: tuple[int, int], *, bottom: bool) -> bool:
        begin, end = band
        gap = begin - neighbour[1] - 1 if bottom else neighbour[0] - end - 1
        if gap < gap_px:
            return False
        if end - begin + 1 > core_height * _TRIM_MAX_EDGE_HEIGHT_FRAC:
            return False
        if bottom:
            return end >= height - 1 - _TRIM_EDGE_SLACK_PX
        return begin <= _TRIM_EDGE_SLACK_PX

    kept = list(bands)
    while len(kept) >= 2 and _edge_junk(kept[-1], kept[-2], bottom=True):
        kept.pop()
    while len(kept) >= 2 and _edge_junk(kept[0], kept[1], bottom=False):
        kept.pop(0)
    if len(kept) == len(bands):
        return img
    top = max(0, kept[0][0] - 2)
    bottom = min(height, kept[-1][1] + 3)
    return img.crop((0, top, width, bottom))


def _component_count(dark: set[tuple[int, int]]) -> int:
    """Connected components (8-connectivity), ignoring single-pixel specks."""
    seen: set[tuple[int, int]] = set()
    components = 0
    for start in dark:
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        size = 0
        while queue:
            x, y = queue.popleft()
            size += 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nxt = (x + dx, y + dy)
                    if nxt in dark and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
        if size >= 2:
            components += 1
    return components


def _metrics(img: PILImage.Image) -> _Metrics | None:
    """Structural metrics of one raster, or None when nothing is measurable."""
    parsed = _ink_grid(img)
    if parsed is None:
        return None
    grid, width, height = parsed
    dark_pixels = _ink_pixels(grid, width, height)
    if len(dark_pixels) < MIN_INK_PIXELS:
        return None
    dark = _strip_trailing_number(set(dark_pixels), width)
    if len(dark) < MIN_INK_PIXELS:
        return None
    xs = [x for x, _ in dark]
    ys = [y for _, y in dark]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    box_w = max_x - min_x + 1
    box_h = max_y - min_y + 1
    density = len(dark) / float(box_w * box_h)
    aspect = box_w / float(box_h)
    row_counts = [0] * height
    for _, y in dark:
        row_counts[y] += 1
    baseline_row = max(range(height), key=lambda y: row_counts[y])
    baseline = (baseline_row - min_y) / float(box_h)
    return _Metrics(
        density=density,
        aspect=aspect,
        components=_component_count(set(dark)),
        baseline=baseline,
    )


def compare_structure(
    rendered: PILImage.Image,
    source: PILImage.Image,
    *,
    min_aspect: float = MIN_ASPECT_RATIO,
    max_aspect: float = MAX_ASPECT_RATIO,
    min_component: float = MIN_COMPONENT_RATIO,
    max_component: float = MAX_COMPONENT_RATIO,
) -> list[str]:
    """Return human-readable findings; an empty list means the pair matches.

    Only gross structural deviations are reported — this is a corruption
    witness, not a typesetting critic. Returns ``["unwitnessable"]`` when the
    source crop cannot be measured, so the caller can distinguish "nothing to
    compare" from "compared and matched".

    The tolerances are keyword-configurable because the same metrics judge
    formulas and tables: a formula's band is calibrated tight, a table's is
    wider (a Typst booktabs grid legitimately differs in rules and fonts).
    """
    rendered_metrics = _metrics(rendered)
    source_metrics = _metrics(source)
    if source_metrics is None:
        return ["unwitnessable"]
    if rendered_metrics is None:
        return ["rendered asset has no measurable ink"]
    findings: list[str] = []
    aspect_ratio = rendered_metrics.aspect / source_metrics.aspect
    if not min_aspect <= aspect_ratio <= max_aspect:
        findings.append(f"aspect {aspect_ratio:.2f}x of source")
    component_ratio = rendered_metrics.components / float(max(source_metrics.components, 1))
    if not min_component <= component_ratio <= max_component:
        findings.append(f"components {rendered_metrics.components} vs {source_metrics.components}")
    return findings


def rasterize_typst(
    body: str,
    typst_binary: str,
    dpi: int = WITNESS_DPI,
    page_width_pt: float | None = None,
) -> PILImage.Image | None:
    """Compile one Typst body into a standalone auto-sized page and rasterize it.

    Shared by the formula and table witnesses: every witness rasterizes the
    *emitted* markup in isolation, so the comparison is against the artifact the
    reader will actually see. None when the rasterizer is unavailable (the
    fail-open contract callers depend on).

    ``page_width_pt`` fixes the probe page width. A table's columns are
    percentage-based, so on a ``width: auto`` page they collapse and the grid
    rasterizes tall and narrow; passing the source table's width makes the
    percentages resolve exactly as they do on the delivered page. Left None for
    formulas, whose probe geometry is already calibrated.
    """
    if page_width_pt is not None and page_width_pt > 0:
        page_setup = (
            f"#set page(width: {page_width_pt:.1f}pt, height: auto, margin: 0pt)\n"
            "#set math.equation(numbering: none)\n\n"
        )
    else:
        page_setup = "#set page(width: auto, height: auto)\n#set math.equation(numbering: none)\n\n"
    try:
        with tempfile.TemporaryDirectory(prefix="ubt-witness-") as tmp:
            probe = Path(tmp) / "witness.typ"
            probe.write_text(page_setup + body + "\n", encoding="utf-8")
            pdf_path = Path(tmp) / "witness.pdf"
            proc = subprocess.run(
                [typst_binary, "compile", str(probe), str(pdf_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=120,
                env=subprocess_env(),
            )
            if proc.returncode != 0 or not pdf_path.exists():
                logger.debug("Witness rasterization failed to compile: %s", proc.stderr[:200])
                return None
            import pypdfium2 as pdfium

            with PDFIUM_LOCK:
                pdf = pdfium.PdfDocument(str(pdf_path))
                try:
                    page = pdf[0]
                    try:
                        from typing import cast

                        bitmap = page.render(scale=dpi / 72.0)
                        return cast("PILImage.Image", bitmap.to_pil().convert("L"))
                    finally:
                        page.close()
                finally:
                    pdf.close()
    except (OSError, subprocess.SubprocessError, ImportError) as exc:
        logger.debug("Witness rasterization unavailable: %s", exc)
        return None


def rasterize_math(
    math_line: str,
    typst_binary: str,
    dpi: int = WITNESS_DPI,
) -> PILImage.Image | None:
    """Compile one emitted math line and rasterize it; None when unavailable."""
    return rasterize_typst(_extract_math_source(math_line), typst_binary, dpi=dpi)


def witness_formula(
    math_line: str,
    block: IRBlock,
    source_pdf: Path | str,
    typst_binary: str,
    dpi: int = WITNESS_DPI,
) -> WitnessResult:
    """Compare one emitted formula against its source crop (L1, zero-token).

    Never raises: every measurement failure — missing bbox, unavailable
    crop, a rasterizer crash (e.g. a PDF-backend error ``rasterize_math``
    does not model), unmeasurable pixels — yields ``unwitnessable`` and
    the caller keeps its native math. The witness must never break a
    render.
    """
    if block.bbox is None or block.bbox.page <= 0:
        return WitnessResult("unwitnessable", ["no source bounding box"])
    try:
        from ubt.adapters.pdf.visual_scalpel import crop_block_pil

        source_img = crop_block_pil(source_pdf, block.bbox.page, block.bbox, dpi=dpi, bleed_pt=0.0)
        source_img = trim_formula_crop(source_img, dpi=dpi)
    except Exception as exc:  # witness must never break a render
        logger.debug("Witness source crop unavailable for %s: %s", block.id, exc)
        return WitnessResult("unwitnessable", ["source crop unavailable"])
    try:
        rendered_img = rasterize_math(math_line, typst_binary, dpi=dpi)
        if rendered_img is None:
            return WitnessResult("unwitnessable", ["emitted formula did not rasterize"])
        findings = compare_structure(rendered_img, source_img)
    except Exception as exc:  # witness must never break a render
        logger.debug("Witness rasterize/compare failed for %s: %s", block.id, exc)
        return WitnessResult("unwitnessable", ["rasterize/compare unavailable"])
    if not findings:
        return WitnessResult("pass")
    if findings == ["unwitnessable"]:
        return WitnessResult("unwitnessable", findings)
    return WitnessResult("fail", findings)
