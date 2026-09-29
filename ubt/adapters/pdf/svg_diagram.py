"""Vector diagram extraction with in-diagram label backfill (SVG route).

Replaces the legacy low-resolution PNG harvesting for PDF diagrams
(``pic_p*.png`` at ~1-2x scale, typically a few hundred pixels wide and
visibly blurry when enlarged on A4 pages or HiDPI screens).

Pipeline per diagram (all vector, infinitely sharp at any zoom):

1. ``pdftocairo -svg`` renders the source page to SVG. Poppler outlines
   every glyph to ``<path>`` (no font embedding issues), so the drawing
   itself is resolution-independent by construction.
2. The SVG ``viewBox`` is cropped to the diagram bbox (Docling provenance,
   PDF points) with a small padding.
3. English labels inside the bbox (located via ``pdftotext -bbox`` in
   :class:`DiagramLocalizer`, translated through its glossary) are covered
   with white ``<rect>`` patches and re-emitted as real ``<text>`` nodes —
   sharp *and* selectable/searchable in the final PDF.
4. Typst embeds the ``.svg`` directly via ``#image()`` (verified: Typst
   0.15 renders SVG ``<text>`` including CJK through system fonts).

External binaries (poppler): ``pdftocairo`` + ``pdftotext``. The pipeline
already shells out to ``pdftotext`` in :mod:`diagram_localizer`, so this adds
no new dependency class. When the backend is missing, callers must fall back
to the legacy PNG path (never hard-fail).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.adapters.pdf import oxide_render, pdf_struct
from ubt.adapters.pdf.stream_strip import mul_matrix
from ubt.core.env import subprocess_env
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BoundingBox

if TYPE_CHECKING:
    from PIL import Image

    from ubt.adapters.pdf.diagram_localizer import DiagramLocalizer

logger = logging.getLogger(__name__)

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"


def dereference_svg_uses(root: ET.Element, max_passes: int = 5) -> int:
    """Inline ``<use href="#id">`` references as real geometry.

    Typst/usvg drops poppler-style glyph ``<use>`` references (thousands of
    ``<use xlink:href="#glyph-N" x y>`` per diagram), rendering the whole
    figure as nothing — zero drawing ops, no error (the phantom-blank-page
    defect). Rewriting replaces each ``<use>`` with a translated ``<g>``
    clone of the referenced subtree, so the staged SVG contains only plain
    paths/text that every renderer honors. ``<defs>`` are left in place
    (clipPath/url() references keep working). Nested uses resolve over
    repeated passes (bounded). Returns the replaced count.
    """
    replaced = 0
    for _ in range(max_passes):
        id_map: dict[str, ET.Element] = {}
        for el in root.iter():
            el_id = el.get("id")
            if el_id and el_id not in id_map:
                id_map[el_id] = el
        progress = 0
        for parent in root.iter():
            for child in list(parent):
                if child.tag != f"{{{_SVG_NS}}}use":
                    continue
                ref = child.get("href") or child.get(f"{{{_XLINK_NS}}}href")
                target = id_map.get((ref or "").lstrip("#")) if ref else None
                if target is None:
                    continue
                x = child.get("x", "0")
                y = child.get("y", "0")
                graft = ET.Element(f"{{{_SVG_NS}}}g")
                own_tf = (child.get("transform") or "").strip()
                move = f"translate({x},{y})"
                graft.set("transform", f"{move} {own_tf}".strip())
                for sub in target:
                    graft.append(_deep_copy(sub))
                parent.remove(child)
                parent.append(graft)
                progress += 1
        replaced += progress
        if progress == 0:
            break
    return replaced


def _deep_copy(el: ET.Element) -> ET.Element:
    """Detached deep copy of an ElementTree subtree."""
    clone = ET.Element(el.tag, dict(el.attrib))
    clone.text = el.text
    clone.tail = None
    for sub in el:
        clone.append(_deep_copy(sub))
    return clone


# Padding around the diagram bbox when cropping the page SVG (PDF points).
_CROP_PAD_PT = 2.0
# Padding around a label span for the white-out patch (PDF points).
_LABEL_PAD_PT = 1.0
# Minimum diagram edge length; smaller regions are picture noise, not diagrams.
_MIN_DIAGRAM_EDGE_PT = 20.0


def _estimate_text_width_em(text: str) -> float:
    """Rough advance-width sum in em, without a font-metrics dependency.

    CJK glyphs are full-width, Latin letters/digits near half-width, spaces
    narrower. Used only to decide whether a translated label needs shrinking.
    """
    total = 0.0
    for ch in text:
        if ch.isspace():
            total += 0.28
        elif ord(ch) > 0x2E80:  # CJK ideographs / full-width forms
            total += 1.0
        elif ch.isalnum():
            total += 0.55
        else:
            total += 0.5
    return total or 1.0


def _fit_font_size(text: str, span_w: float, span_h: float) -> float:
    """Largest font size <= ``span_h`` whose estimated text width fits ``span_w``.

    A translated label is often longer than its source, so sizing at the span
    height alone let it overflow into neighbouring diagram elements.
    """
    estimated_w = _estimate_text_width_em(text) * span_h
    if estimated_w <= span_w:
        return span_h
    return span_h * (span_w / estimated_w)


# IMAGE block ids minted directly by the Docling extraction branch
# (``pdf_main#img_0007``). The ``_weave_harvested_images`` fallback mints
# ``pdf_main#img_pic_pN_M`` with a *fabricated* bbox — those must never be
# cropped, the rect would point at the wrong region.
_DOCLING_IMAGE_ID_RE = re.compile(r"pdf_main#img_\d+$")


@dataclass(frozen=True)
class LocalizedSpan:
    """A diagram label with its translation and top-down bbox (PDF points)."""

    text: str
    translated: str
    x0: float
    y0: float
    x1: float
    y1: float


def is_svg_backend_available() -> bool:
    """Whether the poppler SVG backend (pdftocairo + pdftotext) is on PATH."""
    return shutil.which("pdftocairo") is not None and shutil.which("pdftotext") is not None


def get_diagram_backend_status() -> dict[str, Any]:
    """Inspect local system capabilities for diagram extraction & rendering.

    Returns diagnostic status for Poppler utilities and Typst vector capabilities,
    indicating whether the preferred pipeline is 'vector_svg', 'raster_png', or 'none'.
    """
    has_pdftocairo = shutil.which("pdftocairo") is not None
    has_pdftotext = shutil.which("pdftotext") is not None
    has_oxide = oxide_render.is_available()
    typst_svg_ok = is_svg_rendering_supported() if shutil.which("typst") else False

    if has_pdftocairo and has_pdftotext and typst_svg_ok:
        preferred = "vector_svg"
    elif has_oxide:
        preferred = "raster_png"
    else:
        preferred = "none"

    return {
        "pdftocairo": has_pdftocairo,
        "pdftotext": has_pdftotext,
        "oxide": has_oxide,
        "raster_backend": "oxide" if has_oxide else None,
        "typst_svg_ok": typst_svg_ok,
        "preferred_mode": preferred,
    }


_PROBE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100pt" height="60pt"'
    ' viewBox="0 0 100 60">'
    '<rect x="10" y="10" width="40" height="20" fill="red"/></svg>'
)
_PROBE_TYP = '#set page(width: 120pt, height: 80pt)\n\n#image("/probe.svg", width: 80pt)\n'
_PROBE_TIMEOUT_S = 90

# Raster fallback resolution: 300 DPI keeps diagram text sharp on A4/HiDPI.
_RASTER_DPI = 300


def _page_raster(pdf_path: Path | str, page_no: int) -> Image.Image | None:
    """Full-page raster at ``_RASTER_DPI`` as an eagerly-loaded PIL image.

    In-process ``pdf_oxide`` (base dependency; the poppler ``pdftoppm``
    subprocess leg was retired after the stage-2 A/B). pdf_oxide has no
    subprocess timeout by design; its 16 MP ``max_output_pixels`` render
    cap is the in-process backstop.
    """
    data = oxide_render.render_page_png(Path(pdf_path), page_no, _RASTER_DPI)
    if data is None:
        return None
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except OSError:
        return None


def _sample_diagram_background(
    pdf_path: Path | str,
    page_no: int,
    bbox_topdown: tuple[float, float, float, float],
) -> str:
    """Modal colour of the diagram region, as a CSS ``rgb()`` string.

    A hard-coded white white-out patch left a white block over a dark figure;
    the modal colour of the region is its background in practice.
    """
    im = _page_raster(pdf_path, page_no)
    if im is None:
        return "white"
    scale = _RASTER_DPI / 72.0
    x0, y0, x1, y1 = bbox_topdown
    box = (
        max(0, int(x0 * scale)),
        max(0, int(y0 * scale)),
        min(im.width, int(x1 * scale)),
        min(im.height, int(y1 * scale)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return "white"
    colors = im.convert("RGB").crop(box).getcolors(maxcolors=1_000_000)
    if not colors:
        return "white"
    rgb: Any = max(colors, key=lambda c: c[0])[1]
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def render_diagram_png(
    pdf_path: Path | str,
    *,
    page_no: int,
    bbox_bottomup: tuple[float, float, float, float],
    page_height: float,
    out_path: Path | str,
) -> Path | None:
    """Rasterize one diagram region to high-res PNG.

    Fallback for environments where Typst cannot draw SVG shapes (shapes
    vanish with zero drawing ops while ``<text>`` survives — the
    phantom-blank-page defect). The full page comes from ``_page_raster``
    (pdf_oxide), then PIL crops the diagram rect (PDF
    bottom-up bbox mapped to raster top-down pixels). Any failure returns
    None so callers skip the figure instead of emitting dead asset
    references.
    """
    x0, y0, x1, y1 = bbox_bottomup
    if x1 - x0 < _MIN_DIAGRAM_EDGE_PT or y1 - y0 < _MIN_DIAGRAM_EDGE_PT:
        return None
    im = _page_raster(pdf_path, page_no)
    if im is None:
        return None
    try:
        scale = _RASTER_DPI / 72.0
        w, h = im.size
        left = max(0, int(x0 * scale))
        upper = max(0, int((page_height - y1) * scale))
        right = min(w, int(x1 * scale))
        lower = min(h, int((page_height - y0) * scale))
        if right - left < 20 or lower - upper < 20:
            return None
        crop = im.crop((left, upper, right, lower))
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        crop.save(str(out))
        return out
    except OSError:
        return None
    finally:
        im.close()


@lru_cache(maxsize=8)
def _probe_rendering_cached(binary: str, mtime_ns: int) -> bool:
    """Compile a red-rect probe; True iff rectangle ops reach the PDF."""
    del mtime_ns  # cache key only (re-probe after binary upgrades)
    try:
        with tempfile.TemporaryDirectory(prefix="ubt-svg-probe-") as tmp:
            svg = Path(tmp) / "probe.svg"
            svg.write_text(_PROBE_SVG, encoding="utf-8")
            doc = Path(tmp) / "probe.typ"
            doc.write_text(_PROBE_TYP, encoding="utf-8")
            proc = subprocess.run(
                [binary, "compile", str(doc), str(Path(tmp) / "probe.pdf")],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=_PROBE_TIMEOUT_S,
                # Probe compile must mirror typst_compile (same env policy).
                env=subprocess_env(),
            )
            if proc.returncode != 0:
                return False
            with pdf_struct.open_pdf(Path(tmp) / "probe.pdf") as probe_pdf:
                ops = pdf_struct.content_stream_bytes(probe_pdf.pages[0])
            return re.search(rb"\bre\b", ops) is not None
    except (OSError, subprocess.SubprocessError):
        # Uncertain (no binary, wedged backend): preserve current behavior
        # and let the later compile surface real errors.
        return True


def _probe_disk_path(binary: str) -> Path | None:
    """Cache file for the SVG probe result (survives across pytest processes)."""
    try:
        root = Path(os.environ.get("UBT_CACHE_DIR", Path.home() / ".cache" / "ubt"))
        digest = hashlib.sha1(binary.encode("utf-8")).hexdigest()[:16]
        return root / f"typst-svg-probe-{digest}.json"
    except Exception:  # cache must never break probing
        return None


def _probe_disk_hit(binary: str, mtime_ns: int, path: Path) -> bool | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("binary") == binary and raw.get("mtime_ns") == mtime_ns:
            return bool(raw.get("result"))
    except Exception:  # stale/unreadable cache just misses
        pass
    return None


def _probe_disk_store(binary: str, mtime_ns: int, path: Path, result: bool) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"binary": binary, "mtime_ns": mtime_ns, "result": result}),
            encoding="utf-8",
        )
    except Exception:  # cache must never break probing
        pass


def is_svg_rendering_supported(typst_binary: str = "typst") -> bool:
    """Whether the Typst binary actually rasterizes SVG shapes.

    Some builds render SVG ``<text>`` but drop every vector shape (zero
    drawing ops, no error) — every vectorized diagram then vanishes while
    its backfilled labels linger as stray glyphs. Probe once per binary
    (in-process LRU + on-disk cache keyed by binary mtime); only hard
    evidence disables the SVG route in favor of the legacy PNG figures.
    """
    resolved = shutil.which(typst_binary) or typst_binary
    try:
        mtime_ns = Path(resolved).stat().st_mtime_ns if Path(resolved).exists() else 0
    except OSError:
        return True
    disk = _probe_disk_path(resolved)
    if disk is not None:
        hit = _probe_disk_hit(resolved, mtime_ns, disk)
        if hit is not None:
            supported = hit
        else:
            supported = _probe_rendering_cached(resolved, mtime_ns)
            _probe_disk_store(resolved, mtime_ns, disk, supported)
    else:
        supported = _probe_rendering_cached(resolved, mtime_ns)
    if not supported:
        logger.warning(
            "Typst %s does not render SVG shapes; vector diagrams disabled (PNG fallback)",
            resolved,
        )
    return supported


def is_docling_direct_image(block_id: str) -> bool:
    """Whether the IMAGE block carries a real Docling provenance bbox."""
    return _DOCLING_IMAGE_ID_RE.fullmatch(block_id) is not None


def export_page_svg(pdf_path: Path | str, page_no: int, work_dir: Path | str) -> Path | None:
    """Render one PDF page (1-based) to SVG. Returns the path or None."""
    out = Path(work_dir) / f"page_{page_no}.svg"
    try:
        proc = subprocess.run(
            [
                "pdftocairo",
                "-svg",
                "-f",
                str(page_no),
                "-l",
                str(page_no),
                str(pdf_path),
                str(out),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            # Same environment policy as every other external binary here.
            env=subprocess_env(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("pdftocairo unavailable/failed for p%d: %s", page_no, exc)
        return None
    if proc.returncode != 0 or not out.exists():
        logger.debug("pdftocairo failed for p%d: %s", page_no, proc.stderr[:200])
        return None
    return out


def localize_diagram_svg(
    page_svg: Path | str,
    *,
    bbox_topdown: tuple[float, float, float, float],
    spans: list[LocalizedSpan],
    out_path: Path | str,
    background: str = "white",
) -> Path:
    """Crop a page SVG to a diagram rect and backfill translated labels.

    ``bbox_topdown`` is ``(x0, y0, x1, y1)`` in PDF points with a top-left
    origin (matching ``pdftotext -bbox`` / the SVG user space). Only spans
    whose translation differs from the source are patched.
    """
    x0, y0, x1, y1 = bbox_topdown
    if x1 - x0 < _MIN_DIAGRAM_EDGE_PT or y1 - y0 < _MIN_DIAGRAM_EDGE_PT:
        raise DocumentParseError(f"Diagram bbox too small to vectorize: {bbox_topdown!r}")

    ET.register_namespace("", _SVG_NS)
    tree = ET.parse(str(page_svg))
    root = tree.getroot()

    # Glyph-use dereference (phantom-figure defect): inline poppler's
    # <use> references as plain geometry before writing.
    _ = dereference_svg_uses(root)

    vx = x0 - _CROP_PAD_PT
    vy = y0 - _CROP_PAD_PT
    vw = (x1 - x0) + 2 * _CROP_PAD_PT
    vh = (y1 - y0) + 2 * _CROP_PAD_PT
    root.set("viewBox", f"{vx:g} {vy:g} {vw:g} {vh:g}")
    root.set("width", f"{vw:g}pt")
    root.set("height", f"{vh:g}pt")

    overlay = ET.SubElement(root, f"{{{_SVG_NS}}}g", {"id": "ubt-label-backfill"})
    backfilled = [s for s in spans if s.translated and s.translated != s.text]
    # Every white-out patch first, then every label: interleaving them let a
    # later span's patch paint over an earlier span's already-drawn translation
    # whenever the source spans overlap.
    for span in backfilled:
        ET.SubElement(
            overlay,
            f"{{{_SVG_NS}}}rect",
            {
                "x": f"{span.x0 - _LABEL_PAD_PT:g}",
                "y": f"{span.y0 - _LABEL_PAD_PT:g}",
                "width": f"{span.x1 - span.x0 + 2 * _LABEL_PAD_PT:g}",
                "height": f"{span.y1 - span.y0 + 2 * _LABEL_PAD_PT:g}",
                "fill": background,
            },
        )
    patched = 0
    for span in backfilled:
        span_h = max(span.y1 - span.y0, 1.0)
        span_w = max(span.x1 - span.x0, 1.0)
        font_size = _fit_font_size(span.translated, span_w, span_h)
        # alphabetic baseline from the span bottom (descent allowance ~18%).
        baseline = span.y1 - 0.18 * span_h
        text_el = ET.SubElement(
            overlay,
            f"{{{_SVG_NS}}}text",
            {
                "x": f"{span.x0:g}",
                "y": f"{baseline:g}",
                "font-family": "sans-serif",
                "font-size": f"{font_size:g}",
                "xml:space": "preserve",
            },
        )
        text_el.text = span.translated
        patched += 1

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out), encoding="utf-8", xml_declaration=True)
    logger.info("Vectorized diagram -> %s (%d labels backfilled)", out.name, patched)
    return out


def render_diagram_svg(
    pdf_path: Path | str,
    *,
    page_no: int,
    bbox_bottomup: tuple[float, float, float, float],
    page_height: float,
    localizer: DiagramLocalizer,
    target_lang: str = "zh",
    external_translator: Callable[[str], str] | None = None,
    work_dir: Path | str = "/tmp/ubt_svg",
    out_path: Path | str | None = None,
) -> Path | None:
    """End-to-end vectorization of one diagram. Returns the .svg path or None.

    ``bbox_bottomup`` is the Docling provenance rect ``(x0, y0, x1, y1)`` in
    PDF points with a bottom-left origin. Any failure (missing backend,
    no translatable labels, conversion error) returns None so callers fall
    back to the legacy PNG asset.
    """
    if not is_svg_backend_available():
        return None
    bx0, by0, bx1, by1 = bbox_bottomup
    if bx1 - bx0 < _MIN_DIAGRAM_EDGE_PT or by1 - by0 < _MIN_DIAGRAM_EDGE_PT:
        return None
    try:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        page_svg = export_page_svg(pdf_path, page_no, work)
        if page_svg is None:
            return None

        spans = localizer.extract_text_spans(
            pdf_path,
            page_no,
            BoundingBox(page=page_no, x0=bx0, y0=by0, x1=bx1, y1=by1),
            page_height=page_height,
        )
        localized: list[LocalizedSpan] = []
        for span in spans:
            translated = localizer.translate_phrase(span.text, target_lang, external_translator)
            if translated and translated != span.text:
                localized.append(
                    LocalizedSpan(
                        text=span.text,
                        translated=translated,
                        x0=span.x0,
                        y0=span.y0,
                        x1=span.x1,
                        y1=span.y1,
                    )
                )
        # Crop to top-down SVG user space.
        rect = (bx0, page_height - by1, bx1, page_height - by0)
        dest = Path(out_path) if out_path else (work / f"diagram_p{page_no}.svg")
        background = _sample_diagram_background(pdf_path, page_no, rect)
        return localize_diagram_svg(
            page_svg, bbox_topdown=rect, spans=localized, out_path=dest, background=background
        )
    except Exception as exc:
        logger.debug("SVG diagram vectorization failed (p%d): %s", page_no, exc)
        return None


def detect_figure_pages(
    pdf_path: Path | str,
    blocks: list[Any] | None = None,
) -> set[int]:
    """Return 1-based page numbers carrying vector-diagram regions.

    Used by the Phase-1 bilingual advisor (pre-draft, zero-LLM). TABLE
    block bboxes found in ``blocks`` are excluded (tables have their own
    pipeline). Never raises: any read failure yields an empty set.
    """
    from ubt.core.ir.models import BlockType as _BlockType

    try:
        page_count = pdf_struct.page_count(pdf_path)
    except Exception as exc:
        logger.debug("Figure-page detection cannot read '%s': %s", pdf_path, exc)
        return set()
    tables_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for block in blocks or []:
        bbox = getattr(block, "bbox", None)
        if (
            getattr(block, "block_type", None) == _BlockType.TABLE
            and bbox is not None
            and getattr(bbox, "page", 0) > 0
            and bbox.x1 > bbox.x0
            and bbox.y1 > bbox.y0
        ):
            tables_by_page.setdefault(bbox.page, []).append((bbox.x0, bbox.y0, bbox.x1, bbox.y1))
    pages: set[int] = set()
    for page_no in range(1, page_count + 1):
        try:
            if detect_diagram_regions(pdf_path, page_no, exclude=tables_by_page.get(page_no)):
                pages.add(page_no)
        except Exception as exc:
            logger.debug("Figure-page detection failed on p%d: %s", page_no, exc)
    return pages


# ---------------------------------------------------------------------------
# Vector-diagram region detection (inline content-stream parsing)
# ---------------------------------------------------------------------------

# Paint operators that commit the current path (``n`` discards it).
_PAINT_OPS = frozenset({"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"})
# Tokenizer for the small operator subset we track.
_NUM_RE = re.compile(rb"^[+\-.]?[\d.]+$")


def _parse_content_drawings(data: bytes) -> list[tuple[float, float, float, float]]:
    """Collect painted-path bboxes from a raw content stream (user space).

    Tracks the CTM through ``q``/``Q``/``cm`` and approximates curves by
    their control/end points. Text (``BT``…``ET``) contributes no drawing
    bbox — labels are located separately via ``pdftotext -bbox``.
    """
    ctm: tuple[float, float, float, float, float, float] = (1, 0, 0, 1, 0, 0)
    stack: list[tuple[float, float, float, float, float, float]] = []
    operand: list[float] = []
    current: list[tuple[float, float]] = []
    committed: list[tuple[float, float, float, float]] = []

    def _xf(pt: tuple[float, float]) -> tuple[float, float]:
        a, b, c, d, e, f = ctm
        x, y = pt
        return (a * x + c * y + e, b * x + d * y + f)

    def _commit() -> None:
        if current:
            pts = [_xf(p) for p in current]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            committed.append((min(xs), min(ys), max(xs), max(ys)))
            current.clear()

    # Strip string literals (…) and hex strings <…> so their bytes are not
    # mistaken for operators; dictionary <<…>> blocks are operator-free.
    cleaned = re.sub(rb"\((?:\\.|[^\\()])*\)", b" ", data)
    cleaned = re.sub(rb"<[0-9a-fA-F\s]+>", b" ", cleaned)
    cleaned = re.sub(rb"<<.*?>>", b" ", cleaned, flags=re.S)

    for tok in re.findall(rb"[^\s]+", cleaned):
        if _NUM_RE.match(tok):
            try:
                operand.append(float(tok))
            except ValueError:
                operand = []
            continue
        op = tok.decode("latin-1")
        if op == "q":
            stack.append(ctm)
        elif op == "Q":
            if stack:
                ctm = stack.pop()
        elif op == "cm" and len(operand) >= 6:
            a2, b2, c2, d2, e2, f2 = operand[-6:]
            ctm = mul_matrix((a2, b2, c2, d2, e2, f2), ctm)
        elif op == "m" and len(operand) >= 2:
            _commit()
            current.append((operand[-2], operand[-1]))
        elif op == "l" and len(operand) >= 2:
            current.append((operand[-2], operand[-1]))
        elif op in ("c", "v", "y"):
            n = 6 if op == "c" else 4
            if len(operand) >= n:
                pts = operand[-n:]
                for i in range(0, len(pts), 2):
                    current.append((pts[i], pts[i + 1]))
        elif op == "re" and len(operand) >= 4:
            _commit()
            x, y, w, h = operand[-4:]
            current.extend([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            _commit()
        elif op in _PAINT_OPS:
            _commit()
        elif op == "n":
            current.clear()
        operand = []
    return committed


def _merge_rects(
    rects: list[tuple[float, float, float, float]], gap: float = 8.0
) -> list[tuple[float, float, float, float]]:
    """Union rects that overlap or come within ``gap`` points of each other."""
    boxes = list(rects)
    merged: list[tuple[float, float, float, float]] = []
    while boxes:
        base = boxes.pop()
        changed = True
        while changed:
            changed = False
            rest = []
            for other in boxes:
                if (
                    other[0] <= base[2] + gap
                    and other[2] >= base[0] - gap
                    and other[1] <= base[3] + gap
                    and other[3] >= base[1] - gap
                ):
                    base = (
                        min(base[0], other[0]),
                        min(base[1], other[1]),
                        max(base[2], other[2]),
                        max(base[3], other[3]),
                    )
                    changed = True
                else:
                    rest.append(other)
            boxes = rest
        merged.append(base)
    return merged


def _merge_row_bands(
    rects: list[tuple[float, float, float, float]],
    max_x_gap: float = 100.0,
    min_y_overlap: float = 0.6,
) -> list[tuple[float, float, float, float]]:
    """Merge chips sharing one visual row (flow-chart boxes + arrows).

    Structural merging (``_merge_rects``) leaves same-row boxes split when
    arrow gaps exceed the gap tolerance; rendering each chip as its own
    full-width figure then destroys the row. Two regions merge only when
    their vertical overlap covers >= ``min_y_overlap`` of the smaller height
    (vertically stacked boxes, e.g. Layer diagrams, are untouched) and the
    horizontal gap is small (arrows, not separate figures).
    """
    boxes = list(rects)
    merged: list[tuple[float, float, float, float]] = []
    while boxes:
        base = boxes.pop()
        changed = True
        while changed:
            changed = False
            rest = []
            for other in boxes:
                y_top = max(base[1], other[1])
                y_bot = min(base[3], other[3])
                overlap = max(0.0, y_bot - y_top)
                min_h = min(base[3] - base[1], other[3] - other[1])
                x_gap = max(other[0] - base[2], base[0] - other[2], 0.0)
                if min_h > 0 and overlap / min_h >= min_y_overlap and x_gap <= max_x_gap:
                    base = (
                        min(base[0], other[0]),
                        min(base[1], other[1]),
                        max(base[2], other[2]),
                        max(base[3], other[3]),
                    )
                    changed = True
                else:
                    rest.append(other)
            boxes = rest
        merged.append(base)
    return merged


def is_text_panel(
    paths_in_region: int,
    words: list[tuple[float, float, float, float, str]],
    *,
    max_paths: int = 2,
    row_tol: float = 3.0,
    min_words_per_row: int = 3,
    min_rows: int = 2,
) -> bool:
    """Whether a candidate region is formatted prose, not a figure.

    Pure function of path count + word boxes (top-down points). A region
    with trivial vector ink (title background panels, hyperlink rules)
    that contains running-prose rows is decorated text: vectorizing it
    duplicates body content as outlined glyphs and invites glossary
    backfill inside bibliography text (the p33 references defect).
    True diagrams either carry real ink (``paths_in_region`` above
    ``max_paths``) or only short labels (single-row chips, symbol
    annotations), never two multi-word prose rows.
    """
    if paths_in_region > max_paths:
        return False
    rows: list[list[tuple[float, float, float, float, str]]] = []
    for word in sorted(words, key=lambda w: (w[1], w[0])):
        placed = False
        for row in rows:
            if abs(word[1] - row[0][1]) <= row_tol:
                row.append(word)
                placed = True
                break
        if not placed:
            rows.append([word])
    full_rows = sum(1 for row in rows if len(row) >= min_words_per_row)
    return full_rows >= min_rows


_WORD_BOX_RE = re.compile(
    r'<word xMin="([0-9\.]+)" yMin="([0-9\.]+)" xMax="([0-9\.]+)" yMax="([0-9\.]+)">([^<]+)</word>'
)


def _page_word_boxes(
    pdf_path: Path | str, page_no: int
) -> list[tuple[float, float, float, float, str]]:
    """All raw pdftotext word boxes on a page (top-down points, no grouping)."""
    if shutil.which("pdftotext") is None:
        return []
    try:
        proc = subprocess.run(
            ["pdftotext", "-bbox", "-f", str(page_no), "-l", str(page_no), str(pdf_path), "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=subprocess_env(),
        )
        if proc.returncode != 0:
            return []
        words = []
        for m in _WORD_BOX_RE.finditer(proc.stdout):
            wx0, wy0, wx1, wy1 = (float(m.group(i)) for i in range(1, 5))
            text = m.group(5).strip()
            if text:
                words.append((wx0, wy0, wx1, wy1, text))
        return words
    except (OSError, subprocess.SubprocessError):
        return []


def _rects_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection area over the smaller rect (0..1)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def detect_diagram_regions(
    pdf_path: Path | str,
    page_no: int,
    *,
    exclude: list[tuple[float, float, float, float]] | None = None,
    min_paths: int = 3,
    gap: float = 14.0,
) -> list[tuple[float, float, float, float]]:
    """Find vector-diagram regions on a page (bottom-up PDF points).

    Parses the page content stream for painted paths, merges nearby ones,
    and drops full-bleed hairlines plus anything overlapping ``exclude``
    rects (e.g. Docling TABLE bboxes, which have their own pipeline).
    Form XObjects are not descended into (limitation, documented).
    """
    try:
        with pdf_struct.open_pdf(pdf_path) as pdf:
            if not 1 <= page_no <= len(pdf.pages):
                return []
            page = pdf.pages[page_no - 1]
            width, height = pdf_struct.page_size(page)
            stream = pdf_struct.content_stream_bytes(page)
    except Exception as exc:
        logger.debug("Diagram detection failed to read p%d: %s", page_no, exc)
        return []
    if not stream:
        return []

    paths = _parse_content_drawings(stream)
    # Pre-filter furniture paths before spatial clustering; otherwise full-sheet
    # background fills (e.g. 0 0 W H re f on Elsevier/Springer pages) swallow
    # all legitimate diagram paths on the page and cause them to be dropped.
    filtered_paths = [
        p
        for p in paths
        if not ((p[2] - p[0]) * (p[3] - p[1]) > 0.75 * width * height)
        and not ((p[2] - p[0]) > 0.6 * width and (p[3] - p[1]) < 3.0)
    ]
    if len(filtered_paths) < min_paths:
        return []
    # One pdftotext pass per page for the text-panel gate (empty when
    # poppler is missing — the gate then fail-opens toward rendering).
    page_words = _page_word_boxes(pdf_path, page_no)
    regions: list[tuple[float, float, float, float]] = []
    for rect in _merge_row_bands(_merge_rects(filtered_paths, gap=gap)):
        # Clamp to the mediabox; drop spillover artefacts from exotic CTMs.
        cx0, cy0 = max(rect[0], 0.0), max(rect[1], 0.0)
        cx1, cy1 = min(rect[2], width), min(rect[3], height)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < 0.5 * (rect[2] - rect[0]) * (rect[3] - rect[1]):
            continue
        w, h = cx1 - cx0, cy1 - cy0
        if w < _MIN_DIAGRAM_EDGE_PT or h < _MIN_DIAGRAM_EDGE_PT:
            continue
        if w * h > 0.75 * width * height:
            # Page-background fills (e.g. ``0 0 W H re f`` on Elsevier pages)
            # span the whole sheet; they are furniture, never diagrams.
            continue
        if w > 0.8 * width and h < 3.0:
            continue  # full-bleed hairline (header/footer rule)
        rect = (cx0, cy0, cx1, cy1)
        if exclude and any(_rects_overlap(rect, ex) > 0.3 for ex in exclude):
            continue
        # Text-panel gate: trivial ink + running prose rows = decorated
        # text (title panels, hyperlink rules), never vectorize it.
        ink = sum(1 for box in filtered_paths if _rects_overlap(box, rect) > 0.2)
        tx0, ty0, tx1, ty1 = cx0, height - cy1, cx1, height - cy0
        in_region = [
            w for w in page_words if tx0 <= w[0] and w[2] <= tx1 and ty0 <= w[1] and w[3] <= ty1
        ]
        if is_text_panel(ink, in_region):
            logger.debug("Diagram region on p%d rejected as text panel: %r", page_no, rect)
            continue
        regions.append(rect)
    return regions
