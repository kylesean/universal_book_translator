"""Page profiler: fact layer + decision layer.

Facts (cheap, cached) describe *what is on the page*; the decision layer
maps facts to a :class:`PageKind` that drives parse + render routing:

- ``EDITABLE_TEXT`` → fast operator parse, Typst publication render;
- ``VECTOR_HEAVY`` → Docling parse (figures must be represented);
- ``MIXED_COMPLEX`` → Docling parse, per-block render decision;
- ``SCAN_IMAGE`` → VLM parser, then the same publication render.

Only permissive-license libraries are used: pypdfium2 (Apache-2.0) for the
text layer, pikepdf for content-stream operator counts. No PyMuPDF.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ubt.adapters.pdf import pdf_struct
from ubt.adapters.pdf.pdfium_gate import pdfium_serialized
from ubt.cache.store import DiskCacheStore, step_key
from ubt.core.policy.layout_policy import (
    PDF_PATH_OPS,
    POSTER_MAX_CHARS,
    POSTER_MAX_FONTS,
    PROBE_COLUMN_EDGE_RATIO,
    PROBE_COLUMN_GAP_RATIO,
    PROBE_COLUMN_SHARE,
    PROBE_FORMULA_SHARE,
    PROBE_MIN_CHARS,
    PROFILE_VECTOR_PATH_OPS,
    PROFILE_VECTOR_TEXT_CHARS,
    RESUME_MIN_CHARS,
    formula_debris_share,
)

logger = logging.getLogger(__name__)

Rect = tuple[float, float, float, float]


def _group_rows(rects: list[Rect]) -> list[list[Rect]]:
    """Group pdfium rect fragments into text rows by vertical overlap."""
    rows: list[list[Rect]] = []
    for rect in rects:
        left, bottom, _right, top = rect
        height = top - bottom
        for row in rows:
            row_top = max(r[3] for r in row)
            row_bottom = min(r[1] for r in row)
            overlap = min(top, row_top) - max(bottom, row_bottom)
            if overlap > 0.5 * min(height, row_top - row_bottom):
                row.append(rect)
                break
        else:
            rows.append([rect])
    return rows


#: A rotated margin stamp (an arXiv/journal watermark running vertically down
#: the page edge) reads as one glyph per rect: a near-square fragment at most a
#: few glyphs wide. A real text line is far wider. Fragments at or below this
#: width (points) are treated as margin furniture and never counted as a column
#: segment — without this every single-column page with a side stamp scored as
#: multi-column, and ``multicolumn_page_share`` reported 0.225 on a paper whose
#: body is plainly one column (the arXiv Switch Transformer paper).
_FURNITURE_MAX_FRAGMENT_PT = 24.0


def column_right_share(rects: list[Rect], width: float) -> float:
    """Share of text rows whose right column carries a genuine line of text.

    ``count_rects`` fragments a justified line every few glyphs, so judging
    each fragment's left edge flags every dense single-column page as
    multi-column. Fragments merge back into rows by vertical overlap, rows
    split at column-gutter-sized gaps, and only whole segments — the units a
    layout actually positions — are judged.

    Two guards keep page furniture out of the count, because a false "right
    column" here propagates all the way to a wrongly two-column delivery:

    * **Narrow fragments are dropped.** A rotated side stamp (arXiv/journal
      watermark, running vertically down the margin) is emitted as one glyph
      per rect — near-square, a few points wide. Such a fragment is never a
      column segment; only fragments wider than
      :data:`_FURNITURE_MAX_FRAGMENT_PT` are eligible. Stamps without this
      guard made single-column papers score right_row_share ≈ 0.44.
    * **A right-started segment only counts when the same row also carries
      left-column text.** Real two-column text sets a left line and a right
      line at the same vertical band; a page whose only off-centre content is
      marginalia has no such pairing. Requiring both halves on the row is the
      structural difference between a gutter and a stray glyph.
    """
    if not rects or width <= 0:
        return 0.0
    gap = PROBE_COLUMN_GAP_RATIO * width
    edge_x = PROBE_COLUMN_EDGE_RATIO * width
    segments = 0
    right_started = 0
    for row in _group_rows(sorted(rects, key=lambda r: (-r[3], r[0]))):
        ordered = sorted(row, key=lambda r: r[0])
        # Split the row into gutter-separated segments; near-square furniture
        # fragments are excluded before any of this so a stamp cannot seed a
        # segment (nor pad an existing one).
        segs: list[tuple[float, float]] = []
        left, edge = ordered[0][0], ordered[0][2]
        for frag_left, _bottom, frag_right, _top in ordered[1:]:
            if frag_left - edge > gap:
                segs.append((left, edge))
                left, edge = frag_left, frag_right
            else:
                edge = max(edge, frag_right)
        segs.append((left, edge))

        wide = [(lo, hi) for lo, hi in segs if hi - lo > _FURNITURE_MAX_FRAGMENT_PT]
        if not wide:
            continue
        has_left = any(lo <= edge_x for lo, _hi in wide)
        for lo, _hi in wide:
            segments += 1
            right_started += bool(lo > edge_x and has_left)
    return right_started / segments if segments else 0.0


class PageKind(StrEnum):
    """Page classification driving parse + render routing."""

    EDITABLE_TEXT = "editable_text"
    VECTOR_HEAVY = "vector_heavy"
    MIXED_COMPLEX = "mixed_complex"
    SCAN_IMAGE = "scan_image"
    # Short-doc品类 (HYPOTHESIS): poster = image-led single
    # design page; resume = dense single-column prose page. Both need
    # vision-grade ingest to be represented at all (see pdf_profiler).
    POSTER_FIXED = "poster_fixed"
    RESUME_DENSE = "resume_dense"


@dataclass
class PageFacts:
    """Observable facts about one page (no decisions)."""

    page: int  # 1-based
    width_pt: float
    height_pt: float
    n_chars: int
    #: Raw ``count_rects`` fragments. Text extraction pays one
    #: ``get_text_bounded`` + ``_probe_rect_font_style`` per fragment, each of
    #: which walks the page's characters, so ``n_rect_rows * n_chars`` is the
    #: page's extraction cost (see ``assess.PATHOLOGICAL_EXTRACT_COST``).
    n_rect_rows: int
    right_row_share: float
    formula_density: float
    n_path_ops: int
    n_images: int
    n_fonts: int


@dataclass
class PageProfile:
    """Facts plus routing decision for one page."""

    facts: PageFacts
    kind: PageKind


def classify_page(facts: PageFacts) -> PageKind:
    """Map page facts to a routing kind (pure function, fully tested)."""
    # Poster: image-led design page (title + bleed art, almost
    # no body text). Checked first, but never steals vector-heavy drawings:
    # path-heavy pages stay VECTOR_HEAVY even with an image on them. Zero-
    # char image pages stay SCAN_IMAGE (fail-closed: a poster with no text
    # layer is indistinguishable from a scan, and vision handles both).
    if (
        facts.n_images >= 1
        and PROBE_MIN_CHARS <= facts.n_chars < POSTER_MAX_CHARS
        and facts.n_fonts <= POSTER_MAX_FONTS
        and facts.n_path_ops < PROFILE_VECTOR_PATH_OPS
    ):
        return PageKind.POSTER_FIXED
    if facts.n_chars < PROBE_MIN_CHARS:
        if facts.n_path_ops >= PROFILE_VECTOR_PATH_OPS:
            return PageKind.VECTOR_HEAVY
        return PageKind.SCAN_IMAGE
    if facts.n_chars < PROFILE_VECTOR_TEXT_CHARS and facts.n_path_ops >= PROFILE_VECTOR_PATH_OPS:
        return PageKind.VECTOR_HEAVY
    if (
        facts.right_row_share > PROBE_COLUMN_SHARE
        or facts.formula_density >= PROBE_FORMULA_SHARE
        or facts.n_images >= 1
        or facts.n_path_ops >= PROFILE_VECTOR_PATH_OPS
    ):
        return PageKind.MIXED_COMPLEX
    # Resume: dense single-column prose (strictly above the
    # floor so the 2000-char prose baseline stays EDITABLE_TEXT).
    if facts.n_chars > RESUME_MIN_CHARS:
        return PageKind.RESUME_DENSE
    return PageKind.EDITABLE_TEXT


#: Page kinds whose content a reflow re-typeset cannot rebuild faithfully:
#: multi-column bodies, vector drawings, raster/scan pages and image-led
#: posters. ``RESUME_DENSE`` and ``EDITABLE_TEXT`` are plain prose and reflow
#: fine, so they are deliberately excluded.
_STRUCTURAL_KINDS: frozenset[PageKind] = frozenset(
    {
        PageKind.MIXED_COMPLEX,
        PageKind.VECTOR_HEAVY,
        PageKind.SCAN_IMAGE,
        PageKind.POSTER_FIXED,
    }
)


def structural_page_shares(facts: Sequence[PageFacts]) -> tuple[float, float]:
    """Return ``(multicolumn_page_share, structural_page_share)`` over all pages.

    Page-level, unlike an IR block-count share: a page does not multiply when
    the parser splits its paragraphs into more blocks, so the routing signal no
    longer drifts with chunk granularity.

    - ``multicolumn_page_share`` is the share of pages laid out in more than
      one column (``right_row_share`` past the calibrated gutter). Multi-column
      extraction ordering is what a reflow re-typeset mangles, so this is the
      primary tell that a page's reading order will need care.
    - ``structural_page_share`` is the broader share of pages carrying content
      reflow cannot rebuild faithfully (see :data:`_STRUCTURAL_KINDS`) — the
      knob for the zero-cost fidelity-companion decision.
    """
    n = len(facts)
    if n == 0:
        return 0.0, 0.0
    multicolumn = sum(1 for f in facts if f.right_row_share > PROBE_COLUMN_SHARE)
    structural = sum(1 for f in facts if classify_page(f) in _STRUCTURAL_KINDS)
    return multicolumn / n, structural / n


def majority_flags(kinds: Sequence[PageKind]) -> tuple[bool, bool]:
    """``(has_scan, formula_heavy)`` by >=50% page share of the given kinds."""
    n = len(kinds)
    has_scan = n > 0 and sum(1 for k in kinds if k == PageKind.SCAN_IMAGE) * 2 >= n
    formula_heavy = n > 0 and sum(1 for k in kinds if k == PageKind.MIXED_COMPLEX) * 2 >= n
    return has_scan, formula_heavy


@dataclass(frozen=True, slots=True)
class PageProbe:
    """One pdfium page's text-layer read, shared by the profiler and the sampler."""

    width_pt: float
    height_pt: float
    #: The page's *unrotated* MediaBox width. ``get_rect`` returns user-space
    #: rects that a ``/Rotate`` does not transform, while ``get_width`` returns
    #: the rotated display size, so any x-axis comparison must use this one. On
    #: a 90/270-degree portrait page the two differ by a whole dimension: the
    #: gutter probe then looked for the right column past the page's edge and
    #: scored every rotated two-column page as single-column prose.
    unrotated_width_pt: float
    n_chars: int
    text: str
    n_rect_rows: int
    rects: tuple[Rect, ...]


def probe_pdfium_page(page: Any) -> PageProbe:
    """Read one pypdfium2 page: counts, extracted text, and text rectangles.

    The single place that owns the pdfium read sequence (textpage lifecycle,
    the guarded text extraction, the rect sweep); the full-document profiler
    and the engine sampler both consume it, so their per-page facts cannot
    disagree about how a page is read.
    """
    width = float(page.get_width())
    height = float(page.get_height())
    # The frame the rects are in (see ``PageProbe.unrotated_width_pt``).
    try:
        mediabox = page.get_mediabox()
        unrotated_width = float(mediabox[2]) - float(mediabox[0])
    except Exception:  # a page with no readable MediaBox keeps the display size
        unrotated_width = width
    textpage = page.get_textpage()
    try:
        n_chars = int(textpage.count_chars())
        try:
            text = textpage.get_text_range(0, -1)
        except Exception:
            text = ""
        n_rect_rows = int(textpage.count_rects(0, -1))
        rects = tuple(textpage.get_rect(i) for i in range(n_rect_rows))
    finally:
        textpage.close()
    return PageProbe(width, height, unrotated_width, n_chars, text, n_rect_rows, rects)


@pdfium_serialized
def _pdfium_facts(pdf_path: Path) -> list[dict[str, Any]]:
    """Text-layer facts per page via pypdfium2 (empty list on failure)."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return []
    out: list[dict[str, Any]] = []
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:
        logger.debug("profiler: cannot open '%s': %s", pdf_path.name, exc)
        return []
    try:
        for idx in range(len(pdf)):
            page = pdf[idx]
            try:
                probe = probe_pdfium_page(page)
                out.append(
                    {
                        "width_pt": probe.width_pt,
                        "height_pt": probe.height_pt,
                        "n_chars": probe.n_chars,
                        "n_rect_rows": probe.n_rect_rows,
                        "right_row_share": column_right_share(
                            list(probe.rects), probe.unrotated_width_pt
                        ),
                        "formula_density": formula_debris_share(probe.text),
                    }
                )
            finally:
                page.close()
    finally:
        pdf.close()
    return out


def _struct_facts(pdf_path: Path, n_pages: int) -> list[dict[str, Any]]:
    """Content-stream operator counts + resource census per page via pikepdf.

    ``pdf_struct.count_ops`` matches the raw page-stream counts the previous
    backend produced exactly (verified 147/147 pages), so
    ``PROFILE_VECTOR_PATH_OPS`` keeps its calibration; pikepdf is Rust-managed
    and needs no gc-disabling memory workaround.
    """
    try:
        pdf = pdf_struct.open_pdf(pdf_path)
    except Exception as exc:
        logger.debug("profiler: pikepdf cannot read '%s': %s", pdf_path.name, exc)
        return [{} for _ in range(n_pages)]
    out: list[dict[str, Any]] = []
    try:
        for page in list(pdf.pages)[:n_pages]:
            try:
                facts = {
                    "n_path_ops": pdf_struct.count_ops(page, PDF_PATH_OPS),
                    "n_images": pdf_struct.resource_image_count(page),
                    "n_fonts": pdf_struct.resource_font_count(page),
                }
            except Exception:  # one broken page must not lose the book
                facts = {"n_path_ops": 0, "n_images": 0, "n_fonts": 0}
            out.append(facts)
    finally:
        with contextlib.suppress(Exception):
            pdf.close()
    while len(out) < n_pages:
        out.append({"n_path_ops": 0, "n_images": 0, "n_fonts": 0})
    return out


def collect_page_facts(pdf_path: Path) -> list[PageFacts]:
    """Collect fact-layer observations for every page (no decisions)."""
    pdfium_rows = _pdfium_facts(pdf_path)
    if not pdfium_rows:
        return []
    pdf_rows = _struct_facts(pdf_path, len(pdfium_rows))
    facts: list[PageFacts] = []
    for idx, (trow, prow) in enumerate(zip(pdfium_rows, pdf_rows, strict=True)):
        facts.append(
            PageFacts(
                page=idx + 1,
                width_pt=float(trow["width_pt"]),
                height_pt=float(trow["height_pt"]),
                n_chars=int(trow["n_chars"]),
                n_rect_rows=int(trow["n_rect_rows"]),
                right_row_share=float(trow["right_row_share"]),
                formula_density=float(trow["formula_density"]),
                n_path_ops=int(prow.get("n_path_ops", 0)),
                n_images=int(prow.get("n_images", 0)),
                n_fonts=int(prow.get("n_fonts", 0)),
            )
        )
    return facts


# Bump when ``PageFacts`` fields or ``classify_page`` thresholds change. The
# The key must cover the profiler *logic*, not just the PDF bytes: an
# mtime-only key silently reused profiles from older rules for any PDF that
# had not been edited. Version 3 names the sharded DiskCacheStore layout;
# version 5 drops ``n_text_rows``, which the gate no longer reads (a v4 payload
# would still parse, but the field it carries has no reader left).
_PROFILE_CACHE_VERSION = 5


def _cache_key(pdf_path: Path) -> str:
    inputs = [str(pdf_path.resolve())]
    try:
        st = pdf_path.stat()
        inputs.append(f"{st.st_size}:{st.st_mtime_ns}")
    except OSError:
        pass
    return step_key("pdf_page_profile", inputs, {"version": _PROFILE_CACHE_VERSION})


def profile_pdf(pdf_path: Path, cache_dir: Path | None = None) -> list[PageProfile]:
    """Profile every page; results are cached under ``cache_dir``.

    Cache key covers the profiler version + path + size + mtime. The version
    prefix is what makes edited-*code* re-profile: without it, changing
    ``PageFacts`` or ``classify_page`` silently reused profiles produced by the
    old rules for every PDF whose bytes and mtime were untouched (an edited
    PDF still re-profiles via size/mtime).
    """
    pdf_path = Path(pdf_path)
    store: DiskCacheStore | None = None
    if cache_dir is not None:
        store = DiskCacheStore(cache_dir)
        raw = store.get(_cache_key(pdf_path))
        if raw is not None:
            try:
                return [
                    PageProfile(facts=PageFacts(**item["facts"]), kind=PageKind(item["kind"]))
                    for item in json.loads(raw)
                ]
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("profiler: ignoring corrupt profile cache: %s", exc)
    profiles = [PageProfile(facts=f, kind=classify_page(f)) for f in collect_page_facts(pdf_path)]
    # An empty profile means the probe failed (pdfium could not open the
    # document); caching it would serve one transient failure as the PDF's
    # permanent profile — the key covers only path/size/mtime.
    if store is not None and profiles:
        payload = [{"facts": asdict(p.facts), "kind": p.kind.value} for p in profiles]
        store.put(_cache_key(pdf_path), json.dumps(payload))
    return profiles
