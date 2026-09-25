"""Page profiler: fact layer + decision layer.

Facts (cheap, cached) describe *what is on the page*; the decision layer
maps facts to a :class:`PageKind` that drives parse + render routing:

- ``EDITABLE_TEXT`` → fast operator parse, Typst reflow render;
- ``VECTOR_HEAVY`` → operator parse, overlay render preferred;
- ``MIXED_COMPLEX`` → Docling parse, per-block render decision;
- ``SCAN_IMAGE`` → VLM parser, overlay on the source page.

Only permissive-license libraries are used: pypdfium2 (Apache-2.0) for the
text layer, pikepdf for content-stream operator counts. No PyMuPDF.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ubt.adapters.pdf import pdf_struct
from ubt.adapters.pdf.pdfium_gate import pdfium_serialized
from ubt.core.fs_perms import restrict_dir_to_owner, restrict_file_to_owner
from ubt.core.policy.layout_policy import (
    PDF_PATH_OPS,
    PDF_TEXT_OPS,
    POSTER_MAX_CHARS,
    POSTER_MAX_FONTS,
    PROBE_COLUMN_EDGE_RATIO,
    PROBE_COLUMN_SHARE,
    PROBE_FORMULA_SHARE,
    PROBE_MIN_CHARS,
    PROFILE_VECTOR_PATH_OPS,
    PROFILE_VECTOR_TEXT_CHARS,
    RESUME_MIN_CHARS,
    formula_debris_share,
)

logger = logging.getLogger(__name__)


class PageKind(StrEnum):
    """Page classification driving parse + render routing."""

    EDITABLE_TEXT = "editable_text"
    VECTOR_HEAVY = "vector_heavy"
    MIXED_COMPLEX = "mixed_complex"
    SCAN_IMAGE = "scan_image"
    # Short-doc品类 (HYPOTHESIS): poster = image-led single
    # design page; resume = dense single-column prose page. Both route to
    # the rigid overlay chain (see overlay_preferred + engine_selector).
    POSTER_FIXED = "poster_fixed"
    RESUME_DENSE = "resume_dense"


@dataclass
class PageFacts:
    """Observable facts about one page (no decisions)."""

    page: int  # 1-based
    width_pt: float
    height_pt: float
    n_chars: int
    n_rect_rows: int
    right_row_share: float
    formula_density: float
    n_text_ops: int
    n_path_ops: int
    n_images: int
    n_fonts: int


@dataclass
class PageProfile:
    """Facts plus routing decision for one page."""

    facts: PageFacts
    kind: PageKind

    @property
    def needs_vision(self) -> bool:
        """True when the page must go through the VLM parser."""
        return self.kind == PageKind.SCAN_IMAGE

    @property
    def overlay_preferred(self) -> bool:
        """True when overlay render beats reflow for this page."""
        return self.kind in {
            PageKind.VECTOR_HEAVY,
            PageKind.SCAN_IMAGE,
            PageKind.POSTER_FIXED,
            PageKind.RESUME_DENSE,
        }


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
                width = float(page.get_width())
                height = float(page.get_height())
                textpage = page.get_textpage()
                try:
                    n_chars = int(textpage.count_chars())
                    try:
                        text = textpage.get_text_range(0, -1)
                    except Exception:
                        text = ""
                    n_rects = int(textpage.count_rects(0, -1))
                    right = sum(
                        1
                        for i in range(n_rects)
                        if textpage.get_rect(i)[0] > PROBE_COLUMN_EDGE_RATIO * width
                    )
                    out.append(
                        {
                            "width_pt": width,
                            "height_pt": height,
                            "n_chars": n_chars,
                            "n_rect_rows": n_rects,
                            "right_row_share": (right / n_rects) if n_rects else 0.0,
                            "formula_density": formula_debris_share(text),
                        }
                    )
                finally:
                    textpage.close()
            finally:
                page.close()
    finally:
        pdf.close()
    return out


def _struct_facts(pdf_path: Path, n_pages: int) -> list[dict[str, Any]]:
    """Content-stream operator counts + resource census per page via pikepdf.

    ``pdf_struct.count_ops`` matches the retired pypdf raw page-stream counts
    exactly (verified 147/147 pages), so ``PROFILE_VECTOR_PATH_OPS`` keeps
    its calibration; the gc-disabling dance was a pypdf memory-workaround
    and pikepdf (Rust-managed) does not need it.
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
                    "n_text_ops": pdf_struct.count_ops(page, PDF_TEXT_OPS),
                    "n_path_ops": pdf_struct.count_ops(page, PDF_PATH_OPS),
                    "n_images": pdf_struct.resource_image_count(page),
                    "n_fonts": pdf_struct.resource_font_count(page),
                }
            except Exception:  # one broken page must not lose the book
                facts = {"n_text_ops": 0, "n_path_ops": 0, "n_images": 0, "n_fonts": 0}
            out.append(facts)
    finally:
        with contextlib.suppress(Exception):
            pdf.close()
    while len(out) < n_pages:
        out.append({"n_text_ops": 0, "n_path_ops": 0, "n_images": 0, "n_fonts": 0})
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
                n_text_ops=int(prow.get("n_text_ops", 0)),
                n_path_ops=int(prow.get("n_path_ops", 0)),
                n_images=int(prow.get("n_images", 0)),
                n_fonts=int(prow.get("n_fonts", 0)),
            )
        )
    return facts


# Bump when ``PageFacts`` fields or ``classify_page`` thresholds change. The
# cache key must cover the profiler *logic*, not just the PDF bytes: an
# mtime-only key silently reused profiles from older rules for any PDF that had
# not been edited (see ``_cache_key``).
_PROFILE_CACHE_VERSION = 1


def _cache_key(pdf_path: Path) -> str:
    h = hashlib.sha256()
    h.update(f"v{_PROFILE_CACHE_VERSION}".encode())
    h.update(str(pdf_path.resolve()).encode())
    try:
        st = pdf_path.stat()
        h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    except OSError:
        pass
    return h.hexdigest()[:32]


def profile_pdf(pdf_path: Path, cache_dir: Path | None = None) -> list[PageProfile]:
    """Profile every page; results are cached under ``cache_dir``.

    Cache key covers the profiler version + path + size + mtime. The version
    prefix is what makes edited-*code* re-profile: without it, changing
    ``PageFacts`` or ``classify_page`` silently reused profiles produced by the
    old rules for every PDF whose bytes and mtime were untouched (an edited
    PDF still re-profiles via size/mtime).
    """
    pdf_path = Path(pdf_path)
    cache_file: Path | None = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{_cache_key(pdf_path)}.json"
        if cache_file.exists():
            try:
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                return [
                    PageProfile(facts=PageFacts(**item["facts"]), kind=PageKind(item["kind"]))
                    for item in raw
                ]
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("profiler: ignoring corrupt cache %s: %s", cache_file, exc)
    profiles = [PageProfile(facts=f, kind=classify_page(f)) for f in collect_page_facts(pdf_path)]
    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            restrict_dir_to_owner(cache_file.parent)
            payload = [{"facts": asdict(p.facts), "kind": p.kind.value} for p in profiles]
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
            restrict_file_to_owner(cache_file)
        except OSError as exc:
            logger.debug("profiler: cannot write cache %s: %s", cache_file, exc)
    return profiles
