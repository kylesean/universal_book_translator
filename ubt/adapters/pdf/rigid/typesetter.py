"""Rigid typesetter: region-rigid adaptive typesetting for PDFs.

The paradigm (BabelDOC-class, self-implemented, Zero-AGPL): keep the source
page untouched as the canvas, give every narrative block a rectangular
region (:mod:`ubt.adapters.pdf.rigid.zones`), delete the source text
inside it and typeset the translation into that region with an adaptive
font size — shrink (bounded by ``min_font_pt``) until the whole block fits,
paginate leftover text across the block's continuation zones at clause
boundaries, and fail closed (paint nothing, report) when it still does not
fit. Figures, equations and every other vector stay exactly where they are;
no per-line pairing, no coverage gates, no merge windows.

Compared with the in-place engine this drops the whole per-line machinery;
compared with full reflow it keeps the original page geometry and figures.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pikepdf

from ubt.adapters.pdf.font_probe import resolve_font_stack
from ubt.adapters.pdf.overlay_text import (
    prepare_overlay_text,
    render_overlay_line,
    split_math_spans,
    typstify_math,
)
from ubt.adapters.pdf.rigid.extract import extract_pages
from ubt.adapters.pdf.rigid.gate import skip_reason
from ubt.adapters.pdf.rigid.rows import split_clauses
from ubt.adapters.pdf.rigid.zones import CONTENT_BOTTOM_PT, PageFacts, Zone, build_zones
from ubt.adapters.pdf.text_fit import FlowFitter
from ubt.adapters.pdf.textgeom import dehyph
from ubt.adapters.pdf.typst_fragments import sanitize_lang_tag
from ubt.adapters.pdf.typst_math_probe import TypstMathProbe
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, BookManifest, FlowID, IRBlock, LayoutRole
from ubt.core.policy.layout_policy import (
    FIT_PRECISION_PT,
    PROSE_BLOCK_TYPES,
    RIGID_MIN_FONT_PT,
    rigid_min_font_pt_for,
)
from ubt.model.fidelity import Fidelity

logger = logging.getLogger(__name__)

_PAGE_SETUP = "#set page(width: {w:.2f}pt, height: {h:.2f}pt, margin: 0pt, fill: none)\n"
# Line pitch the fitter assumes. Typst's rendered pitch measures ~1.375x;
# staying slightly above it keeps the fit conservative. The painted block is
# clipped to the zone as the last line of defence.
PITCH_FACTOR = 1.40
UPSCALE_MAX = 1.0
FIT_STEP_PT = 0.2
MAX_SIZE_PT = 24.0
ZONE_INSET_TOP_PT = 1.5
# How far an otherwise-spilling block may reclaim confirmed-blank margin below
# its last zone before it fails closed. Only ever consulted for a block whose
# text did not fit (it would otherwise ship untranslated), so a positive value
# strictly raises coverage and never re-lays an already-painted block. Bounded
# by the nearest occupied rect (block bbox or image) below, so it cannot paint
# over a neighbour. Recovers single-line captions/notes whose Chinese needs one
# more line than the thin source band holds (arXiv 2609.20519 p6 "Note.").
RIGID_MARGIN_RECLAIM_DEFAULT_PT = 24.0
# Axiom B last resort: when margin reclaim is still not enough, try one more
# fit below the region floor, down to this absolute minimum, before failing
# closed (leaving source visible). A slightly small target beats an untranslated
# paragraph for the reader; the degraded blocks are counted so the quality
# report can show the tradeoff.
RIGID_DEGRADED_FIT_FLOOR_PT = 5.0
# Minimum blank band (points) kept between a reclaimed zone's new bottom and the
# nearest occupied rect below it, so reclaimed paint never grazes a neighbour.
RECLAIM_GAP_PT = 5.0
# Longest source that still counts as a page-repeating running head (chars).
# A genuine paragraph continuation is far longer and never repeats verbatim.
REPEAT_CHROME_MAX_SRC = 120

Rect = tuple[float, float, float, float]

# Docling labels a bullet line LIST_ITEM but the ``•`` glyph lives in a
# separate drawing/text run the extractor drops, so a translated list item
# reaches the overlay with no marker (arXiv 2609.20519 intro bullets). The
# original number of an ordered list is unrecoverable at this layer, so an
# unordered bullet is restored for every markerless LIST_ITEM; a text that
# already opens with a marker (model reproduced it) is left untouched.
_LEADING_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"[•⁃◦▪●*\-]"  # unordered bullet
    r"|\(?\d{1,3}[.)、]"  # 1. 1) (1) 1、
    r"|\(?[a-zA-Z][.)]"  # a. a) (a)
    r"|[一二三四五六七八九十百]+[、.)]"  # 一、 二）
    r")\s*"
)


@dataclass
class RigidPageReport:
    page: int
    painted_zones: int = 0
    painted_blocks: int = 0
    stripped_ops: int = 0


@dataclass
class RigidReport:
    pages: list[RigidPageReport] = field(default_factory=list)
    rendered_blocks: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    # Blocks that only fit after margin reclaim — counted so the quality
    # report can show how much coverage reclaim bought.
    reclaimed_blocks: list[str] = field(default_factory=list)
    # Blocks that only fit after shrinking below the region floor (Axiom B
    # last resort). Counted so the quality report can show the tradeoff.
    degraded_blocks: list[str] = field(default_factory=list)
    # Blocks that only fit after the last zone was extended to the page bottom
    # (bounded overflow). Translation completeness outranks pixel layout, so
    # these paint past their source box; counted so the tradeoff is visible.
    overflow_blocks: list[str] = field(default_factory=list)
    # page number -> block ids planned for that page. A page-level overlay
    # compile failure must demote these blocks to render skips so the loss is
    # block-scoped and visible in the quality report instead of silently
    # shipping the untranslated source page.
    blocks_by_page: dict[int, list[str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ZonePlan:
    """One zone's final typesetting: chosen size, fitted lines, and the raw text.

    ``text`` is the exact text fitted into this zone. The line strings in
    ``lines`` are the fitter's capacity evidence — its breaks drop the spaces
    at break points, so it cannot be re-joined — while the emitter hands
    ``text`` to Typst unbroken and lets Typst own the line breaking.
    """

    zone: Zone
    size: float
    lines: list[str]
    text: str
    #: Per-region font floor for the emitter's ``#ubt-fit`` min-sz. Captions and
    #: footnotes may go below the body floor, so the emitter must use this (not
    #: the instance-global ``RigidTypesetter.min_font_pt``), or ``min-sz`` can
    #: exceed ``size`` and Typst's fit loop never shrinks.
    min_font_pt: float = RIGID_MIN_FONT_PT


# Skip reasons that mean "the decoder gave us no geometry to paint into". Any
# of them, with zero paintable zones overall, is a decode failure rather than a
# document that simply had nothing to translate.
_GEOMETRY_SKIP_REASONS = frozenset({"no_zone", "no_page_height", "no_bbox"})


def _assert_paintable(paints: dict[int, list[ZonePlan]], report: RigidReport) -> None:
    """Fail closed when nothing is paintable and the decoder gave no geometry.

    Zero paintable zones with only geometry skips means the decode path
    produced no usable geometry: a fallback engine writing all-zero boxes
    (``no_zone``), an all-rotated document whose pages yield no ``PageFacts``
    (``no_page_height``), or blocks that never got a bbox (``no_bbox``).
    Proceeding would copy the source PDF and present it as the translation.
    A document whose only blocks were skipped for a content reason
    (``non_prose``, ``empty_target``) is legitimately empty and does not raise.
    """
    if paints:
        return
    geometry_blocks = [bid for bid, reason in report.skipped if reason in _GEOMETRY_SKIP_REASONS]
    if geometry_blocks:
        raise DocumentParseError(
            "Rigid render produced no paintable zones "
            f"({len(geometry_blocks)} block(s) had no usable page geometry). "
            "Refusing to emit the untranslated source as the translation; use "
            "render_engine='publication' or a PDF engine that reports real bounds."
        )


def _demote_page_blocks(report: RigidReport, page_no: int) -> None:
    """Turn a failed page overlay into block-scoped render skips."""
    for bid in report.blocks_by_page.get(page_no, []):
        if bid in report.rendered_blocks:
            report.rendered_blocks.remove(bid)
        report.skipped.append((bid, "overlay_compile"))


def _erase_rects_by_page(
    zone_map: dict[str, tuple[Zone, ...]],
    rendered_ids: set[str],
    overlaid_pages: set[int] | None = None,
) -> dict[int, list[Rect]]:
    """Erase EVERY zone a rendered block owns on a page, not just painted ones.

    CJK is more compact than the Latin source, so a paragraph whose translation
    fits entirely in its first zone leaves the continuation zone unpainted — but
    that zone's source line still has to be removed, or it survives as an
    untranslated ghost (arXiv 2609.20519 p4 "software changes…"). Only rendered
    blocks qualify: a spilled block keeps its whole source (fail-closed), so
    erasing its zones would punch a hole in the page.

    ``overlaid_pages`` restricts erasure to pages whose overlay actually
    compiled. A multi-page block demoted on ONE page must keep erasing the
    other pages it did render on, or that page's drawn translation sits on top
    of un-erased source text. ``None`` means "every page" (the pre-existing
    behaviour).
    """
    erase_rects: dict[int, list[Rect]] = {}
    for bid, block_zones in zone_map.items():
        if bid not in rendered_ids:
            continue
        for zone in block_zones:
            if overlaid_pages is not None and zone.page not in overlaid_pages:
                continue
            erase_rects.setdefault(zone.page, []).append(zone.rect)
    return erase_rects


def _strip_and_merge_page(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    page_no: int,
    overlay_path: str,
    strip_rects: list[Rect],
    protected_rects: list[Rect],
    shared_forms: set[tuple[int, int]],
) -> tuple[int, bool]:
    """Strip one page's source text and paint its translation overlay.

    The overlay is prepared before any stripping and a failed paint rolls the
    strip back, so a ``False`` return always means the page still carries its
    source text — never a stripped page with no translation painted. Returns
    ``(strip ops dropped, overlay painted)``.
    """
    from ubt.adapters.pdf.stream_strip import strip_page_text_pikepdf

    try:
        with pikepdf.open(overlay_path) as overlay:
            if not overlay.pages:
                logger.warning(
                    "rigid overlay for page %d has no pages; keeping source page", page_no
                )
                return 0, False
            form = pdf.copy_foreign(overlay.pages[0].as_form_xobject())
    except Exception as exc:
        logger.warning(
            "rigid overlay load failed on page %d: %s; keeping source page", page_no, exc
        )
        return 0, False

    stats = strip_page_text_pikepdf(
        page,
        strip_rects,
        protected_rects=protected_rects,
        page_no=page_no,
        shared_forms=shared_forms,
    )
    if stats.shared_forms_skipped:
        # Forms shared with other pages are left intact (their text would
        # vanish everywhere otherwise), so the source text under the erase
        # rects survives. Painting the overlay on top would double it,
        # exactly like an abort — treat it as one and keep the source page.
        logger.warning(
            "rigid strip on page %d left %d page-shared Form "
            "XObject(s) intact; skipping the overlay to avoid "
            "doubling the surviving source text",
            page_no,
            stats.shared_forms_skipped,
        )
        return stats.dropped_ops, False
    if stats.aborted:
        # The source text could not be removed (the strip never committed);
        # drawing the overlay on top would double/overlap the text. Keep the
        # source page and record the loss as a skip.
        logger.warning(
            "rigid strip aborted on page %d (%s); keeping source text",
            page_no,
            stats.aborted,
        )
        return stats.dropped_ops, False
    try:
        # ``add_overlay(form, None)`` places the form in the page's TrimBox
        # (pikepdf's default), which scales and re-anchors the whole
        # translation layer whenever CropBox/TrimBox < MediaBox. The overlay is
        # authored in the MediaBox frame (see rigid/extract.py), so place it
        # there explicitly for a 1:1, unscaled result.
        media = page.mediabox
        page.add_overlay(
            form,
            pikepdf.Rectangle(
                float(media[0]),
                float(media[1]),
                float(media[2]),
                float(media[3]),
            ),
        )
    except Exception as exc:
        if stats.rollback is not None:
            stats.rollback()
        logger.warning(
            "rigid overlay paint failed on page %d: %s; source text restored", page_no, exc
        )
        return stats.dropped_ops, False
    return stats.dropped_ops, True


_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_SUPERSCRIPT_TO_ASCII = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_FOOTNOTE_NUM_RE = re.compile(r"^(\d{1,2})\s+\S")
_LEADING_SUPER_RE = re.compile(r"^([⁰¹²³⁴⁵⁶⁷⁸⁹]+)\s*")


_ORDERED_ROW_MARKER_RE = re.compile(r"^\s*(\d{1,2}\.|\([a-zA-Z0-9]{1,3}\)|[a-zA-Z0-9]\))\s+")


def _with_list_marker(block: IRBlock, text: str, zone_rows: tuple[str, ...] = ()) -> str:
    """Restore dropped list bullets/numbers and format leading footnote markers as superscripts."""
    if not text:
        return text
    if block.flow_id == FlowID.FOOTNOTE or block.layout_role == LayoutRole.FOOTNOTE:
        src_m = _FOOTNOTE_NUM_RE.match((block.source_text or "").strip())
        if src_m:
            fn_num = src_m.group(1)
            tgt_m = re.match(rf"^\s*{re.escape(fn_num)}(?:\s+|(?=\d))(.*)$", text, re.DOTALL)
            if tgt_m:
                sup = fn_num.translate(_SUPERSCRIPT_DIGITS)
                return f"{sup} {tgt_m.group(1).lstrip()}"
    if block.block_type != BlockType.LIST_ITEM:
        return text
    if _LEADING_MARKER_RE.match(text):
        return text
    if zone_rows:
        m = _ORDERED_ROW_MARKER_RE.match(zone_rows[0] or "")
        if m:
            return f"{m.group(1)} {text}"
    return f"• {text}"


def _reclaim_zone_down(
    zone: Zone,
    page_rects: list[tuple[str, Rect]],
    min_y: float,
    max_reclaim_pt: float,
    gap_pt: float = RECLAIM_GAP_PT,
) -> Zone:
    """Extend a zone downward into adjacent confirmed-blank space.

    The overlay fails closed when a block's translated text will not fit its
    zones even at the font floor — most often a short paragraph with extra
    vertical margin below it on the page. This grows ONLY the bottom edge
    (``y0``) of the given zone toward the page bottom, stopping at whichever is
    higher: the nearest occupied rectangle below it that overlaps its horizontal
    span (plus ``gap_pt``), the page ``min_y`` footer clamp, or the block's own
    ``max_reclaim_pt`` budget. It never grows sideways or upward, so it can never
    collide with a neighbouring column or a figure guard that build_zones already
    excluded. Returns the (possibly unchanged) zone; a no-op zone is returned
    unchanged so the caller can cheaply tell "nothing reclaimed".
    """
    if max_reclaim_pt <= 0:
        return zone
    floor = max(min_y, zone.y0 - max_reclaim_pt)
    for other_id, (ox0, other_bottom, ox1, other_top) in page_rects:
        if other_id == zone.block_id:
            continue
        if ox1 <= zone.x0 or ox0 >= zone.x1:  # no horizontal overlap → different column
            continue
        if other_bottom < zone.y0:  # neighbour reaches our bottom line or below
            floor = max(floor, other_top + gap_pt)
    new_y0 = max(floor, min_y)
    if new_y0 >= zone.y0 - 0.5:
        return zone
    return zone.with_y0(new_y0)


def _overflow_zone_down(zone: Zone, min_y: float) -> Zone:
    """Extend a zone down to the page content bottom, ignoring neighbours.

    Bounded overflow (Axiom B): when a translation cannot fit its region even at
    the degraded floor, prefer painting it below the source box — down to the
    page's content bottom — over leaving the source visible. It is bounded by
    the *page*, not by neighbours, so the text may run into blank space or, at
    worst, over a neighbour: the agreed priority order puts translation
    completeness above pixel layout. A no-op zone is returned unchanged.
    """
    if min_y >= zone.y0 - 0.5:
        return zone
    return zone.with_y0(min_y)


def _boxes_for(zone: Zone, size_pt: float) -> list[Rect]:
    """Top-down line boxes inside the zone for one candidate size."""
    usable_h = max(0.0, zone.height - ZONE_INSET_TOP_PT)
    pitch = size_pt * PITCH_FACTOR
    # For a single line, height only needs to accommodate the glyph (size_pt * 0.75);
    # for n lines, total height required is (n - 1) * pitch + size_pt * 0.75.
    count = int((usable_h - size_pt * 0.75) // pitch) + 1 if usable_h >= size_pt * 0.75 else 0
    if count <= 0:
        return []
    boxes: list[Rect] = []
    top = zone.y1 - ZONE_INSET_TOP_PT
    for _ in range(count):
        boxes.append((zone.x0, top - pitch * 0.9, zone.x1, top))
        top -= pitch
    return boxes


def _candidate_rigid_pages(total_pages: int, blocks: Sequence[IRBlock]) -> list[int]:
    """Return 1-based page numbers needed to plan ``blocks`` (+1/+2 continuation window)."""
    pages: set[int] = set()
    for b in blocks:
        if b.bbox is not None and b.bbox.page >= 1:
            for p in (b.bbox.page, b.bbox.page + 1, b.bbox.page + 2):
                if 1 <= p <= total_pages:
                    pages.add(p)
    if not pages:
        return list(range(1, total_pages + 1))
    return sorted(pages)


class RigidTypesetter:
    """Region-rigid adaptive typesetting over the source PDF."""

    def __init__(
        self,
        font_family: str | None = None,
        target_lang: str = "zh",
        min_font_pt: float = RIGID_MIN_FONT_PT,
        backfill_captions: bool = True,
        translate_chrome: bool = False,
        typst_binary: str | None = None,
        margin_reclaim_pt: float | None = None,
        realization_plan: Mapping[str, Fidelity] | None = None,
    ) -> None:
        from ubt.core.language_profile import resolve_font_config

        self.typst_binary = typst_binary or os.environ.get("UBT_TYPST_BINARY", "typst")
        self.target_lang = target_lang
        # The decision plan (ADR-0001 Phase 3): element id -> committed fidelity.
        # A block the plan kept (PRESERVED_OPAQUE) is left in the source rather
        # than painted over with its translation.
        self.realization_plan = realization_plan
        if font_family is not None and font_family.strip():
            self.font_family = font_family.strip()
        else:
            cfg = resolve_font_config(target_lang)
            self.font_family = cfg.typst_fonts[0] if cfg.typst_fonts else "Noto Serif CJK SC"
        self._font_tuple_cache: str | None = None
        self.min_font_pt = min_font_pt
        self.backfill_captions = backfill_captions
        self.translate_chrome = translate_chrome
        # Margin reclaim: how far an overflowing block may extend downward
        # into adjacent confirmed-blank space before it fails closed. On by
        # default (only consulted for blocks that would otherwise spill, so it
        # only ever adds coverage); set UBT_RIGID_MARGIN_RECLAIM_PT=0 to disable
        # it and restore byte-for-byte the pre-reclaim render.
        if margin_reclaim_pt is None:
            env = os.environ.get("UBT_RIGID_MARGIN_RECLAIM_PT")
            margin_reclaim_pt = (
                float(env) if env not in (None, "") else RIGID_MARGIN_RECLAIM_DEFAULT_PT
            )
        self.margin_reclaim_pt = float(margin_reclaim_pt)
        self.probe = TypstMathProbe(self.typst_binary)
        self._fitter: FlowFitter | None = None
        self._width_font: Any | None = None

    # -- fitting kernel ---------------------------------------------------
    def _fitter_obj(self) -> FlowFitter:
        if self._fitter is None:
            from ubt.adapters.pdf.font_metrics import load_width_font, resolve_cjk_ttc
            from ubt.adapters.pdf.font_metrics import text_width_pt as _measure

            if self._width_font is None:
                self._width_font = load_width_font(resolve_cjk_ttc(), self.font_family)
            font = self._width_font
            self._fitter = FlowFitter(
                measure=lambda t, s: _measure(font, t, s),
                min_font_pt=self.min_font_pt,
                precision_pt=FIT_PRECISION_PT,
            )
        return self._fitter

    def _flow(self, text: str, boxes: list[Rect], size_pt: float) -> list[str] | None:
        return self._fitter_obj().flow_paragraph(text, boxes, size_pt)

    def _flow_prefix(
        self, text: str, boxes: list[Rect], size_pt: float
    ) -> tuple[list[str], str, str] | None:
        """Longest clause prefix that fits, as (lines, consumed, remainder).

        ``consumed`` is the text the boxes were proofed against. Deriving it by
        length arithmetic on the original string instead drifts by every
        separator ``split_clauses`` consumed: the zone then paints (and the
        fidelity guard covers) a few characters that were never measured, and
        ``clip: true`` truncates the overflow.
        """
        clauses = split_clauses(text)
        if not clauses:
            return None
        lo, hi = 0, len(clauses)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._flow("".join(clauses[:mid]), boxes, size_pt) is not None:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            return None
        lines = self._flow("".join(clauses[:lo]), boxes, size_pt)
        if lines is None:
            return None
        return lines, "".join(clauses[:lo]), "".join(clauses[lo:])

    def _plan_at_size(
        self, text: str, zones: tuple[Zone, ...], size: float
    ) -> list[ZonePlan] | None:
        """Fill every zone at one fixed size; None if the text does not fit.

        Shared by :meth:`_paginate` (its size search evaluates this at each
        candidate) and the list-size unification in :meth:`_plan_blocks`
        (which pins a whole list to one size).
        """
        remaining = text
        pending: list[ZonePlan] = []
        for zone in zones:
            boxes = _boxes_for(zone, size)
            if not boxes:
                continue
            result = self._flow_prefix(remaining, boxes, size)
            if result is None:
                continue
            lines, consumed, remaining = result
            if any(ln.strip() for ln in lines):
                pending.append(ZonePlan(zone=zone, size=size, lines=lines, text=consumed))
            if not remaining:
                return pending
        return None

    def _paginate(
        self, text: str, zones: tuple[Zone, ...], min_font_pt: float | None = None
    ) -> list[ZonePlan] | None:
        """Fill every zone at one common size (largest that holds the text).

        A block split across zones (page break, figure page in between) is
        typeset at a single size so the halves read as one paragraph; the
        search is monotonic, so it is a plain bisection over candidate
        sizes. None = the text cannot fit the block's regions at the floor.
        """

        def plan(size: float) -> list[ZonePlan] | None:
            return self._plan_at_size(text, zones, size)

        floor = self.min_font_pt if min_font_pt is None else min_font_pt

        def _stamp(plans: list[ZonePlan] | None) -> list[ZonePlan] | None:
            # Carry the region floor onto every plan so the emitter's ``#ubt-fit``
            # min-sz matches what was planned.
            if plans is None:
                return None
            return [replace(p, min_font_pt=floor) for p in plans]

        s_hi = min(zones[0].base_size * UPSCALE_MAX, MAX_SIZE_PT)
        s_lo = floor
        whole = plan(s_hi)
        if whole is not None:
            return _stamp(whole)
        lo, hi = s_lo, s_hi
        best: list[ZonePlan] | None = None
        while hi - lo > FIT_STEP_PT:
            mid = (lo + hi) / 2.0
            candidate = plan(mid)
            if candidate is not None:
                best, lo = candidate, mid
            else:
                hi = mid
        if best is None:
            # Line capacity is discrete (``height // pitch``): the floor can
            # add the line that no mid-grid size reaches.
            best = plan(s_lo)
        return _stamp(best)

    # -- planning ---------------------------------------------------------
    def _plan_blocks(
        self,
        blocks: list[IRBlock],
        zones: dict[str, tuple[Zone, ...]],
        page_heights: dict[int, float],
        page_images: dict[int, list[Rect]] | None = None,
    ) -> tuple[dict[int, list[ZonePlan]], RigidReport]:
        report = RigidReport()
        paints: dict[int, list[ZonePlan]] = {}
        # Per-page occupancy (block_id, rect). Reclaim must not grow over a
        # neighbour — including figure/non-prose/image guards that never get a
        # zone — so occupancy is every block's real bbox plus every image rect,
        # not just planned zones.
        page_zone_rects: dict[int, list[tuple[str, Rect]]] = {}
        for block in blocks:
            bb = block.bbox
            if bb is None:
                continue
            page_zone_rects.setdefault(bb.page, []).append((block.id, (bb.x0, bb.y0, bb.x1, bb.y1)))
        for pno, imgs in (page_images or {}).items():
            for rect in imgs:
                page_zone_rects.setdefault(pno, []).append(("__image__", rect))
        planned: list[tuple[IRBlock, str, tuple[Zone, ...], list[ZonePlan]]] = []
        for block in blocks:
            if block.bbox is None:
                # No geometry to paint into: record the skip so render_coverage
                # does not count it as rendered (and the audit stays complete).
                report.skipped.append((block.id, "no_bbox"))
                continue
            height = page_heights.get(block.bbox.page)
            if height is None:
                report.skipped.append((block.id, "no_page_height"))
                continue
            reason = skip_reason(
                block,
                height,
                backfill_captions=self.backfill_captions,
                translate_chrome=self.translate_chrome,
            )
            if reason is not None:
                # Chrome / policy / non-prose / verbatim: source-visible by
                # design, recorded so the audit stays complete.
                report.skipped.append((block.id, reason))
                continue
            if (
                self.realization_plan is not None
                and self.realization_plan.get(block.id) is Fidelity.PRESERVED_OPAQUE
            ):
                # The decision plan kept this element in the source (ADR-0001
                # Phase 3): leave the source box intact rather than painting the
                # translation over it.
                report.skipped.append((block.id, "plan:preserved"))
                continue
            floor = rigid_min_font_pt_for(block.layout_role, default=self.min_font_pt)
            text = prepare_overlay_text(
                (block.target_text or "").strip(), target_lang=self.target_lang
            )
            block_zones = zones.get(block.id, ())
            zone_rows = block_zones[0].rows if block_zones else ()
            text = _with_list_marker(block, text, zone_rows=zone_rows)
            if not text:
                report.skipped.append((block.id, "empty_target"))
                continue
            unrenderable_math = False
            for is_math, content in split_math_spans(text):
                if not is_math or len(content) < 2:
                    continue
                converted = typstify_math(content[1:-1])
                if converted is None or not self.probe.check(converted):
                    unrenderable_math = True
                    break
            if unrenderable_math:
                # Do not erase the source and paint escaped ``\\Gamma``/``\\to``
                # as if it were a translation. Preserve the readable source box
                # and make the loss visible to render coverage.
                report.skipped.append((block.id, "math_unrenderable"))
                continue
            if not block_zones:
                report.skipped.append((block.id, "no_zone"))
                continue
            if self._is_repeating_chrome(block, block_zones):
                # A running head repeats the SAME line on every page, and the
                # block may also be page 1's (larger) title. Its zones are
                # parallel — each holds the whole source — so stamp the full
                # translation on every zone, each sized to fit that zone's own
                # band. A single uniform size (the old sequential _paginate
                # split) either clipped the small header bands or left all but
                # one page showing the untranslated source head.
                repeat: list[ZonePlan] = []
                for zone in block_zones:
                    zplan = self._paginate(text, (zone,), floor)
                    if zplan is None:
                        repeat = []
                        break
                    repeat.append(zplan[0])
                if repeat:
                    planned.append((block, text, block_zones, repeat))
                    continue
            pending = self._paginate(text, block_zones, floor)
            eff_zones = block_zones
            if pending is None and self.margin_reclaim_pt > 0:
                # The text overflows its zones; try reclaiming the
                # confirmed-blank margin directly below the last zone before
                # failing closed. build_zones already excludes figures/other
                # blocks, and _reclaim_zone_down stops at the nearest occupied
                # rect below and the footer, so reclaim never paints over ink.
                last = block_zones[-1]
                reclaimed_last = _reclaim_zone_down(
                    last,
                    page_zone_rects.get(last.page, []),
                    CONTENT_BOTTOM_PT,
                    self.margin_reclaim_pt,
                )
                if reclaimed_last is not last:
                    trial = self._paginate(text, (*block_zones[:-1], reclaimed_last), floor)
                    if trial is not None:
                        pending = trial
                        eff_zones = (*block_zones[:-1], reclaimed_last)
                        report.reclaimed_blocks.append(block.id)
            if pending is None and floor > RIGID_DEGRADED_FIT_FLOOR_PT:
                # Axiom B last resort: shrink below the region floor rather than
                # leave the source visible. A small but translated paragraph is
                # what the reader needs; the degradation is recorded.
                degraded = self._paginate(text, eff_zones, RIGID_DEGRADED_FIT_FLOOR_PT)
                if degraded is not None:
                    pending = degraded
                    report.degraded_blocks.append(block.id)
            if pending is None and eff_zones:
                # Bounded overflow: even the floor does not fit, so extend the
                # last zone to the page's content bottom and retry. Translation
                # completeness outranks pixel layout, so the text may paint past
                # its source box rather than leave the source visible.
                overflow_last = _overflow_zone_down(eff_zones[-1], CONTENT_BOTTOM_PT)
                if overflow_last is not eff_zones[-1]:
                    trial = self._paginate(
                        text, (*eff_zones[:-1], overflow_last), RIGID_DEGRADED_FIT_FLOOR_PT
                    )
                    if trial is not None:
                        pending = trial
                        eff_zones = (*eff_zones[:-1], overflow_last)
                        report.overflow_blocks.append(block.id)
            if pending is None:
                # Still nothing fits a whole page: keep the source visible and
                # record it (an explicit keep, not a silent drop).
                report.skipped.append((block.id, "spill"))
                continue
            planned.append((block, text, eff_zones, pending))
        self._unify_list_sizes(planned)
        for block, _text, _eff_zones, pending in planned:
            for entry in pending:
                paints.setdefault(entry.zone.page, []).append(entry)
                report.blocks_by_page.setdefault(entry.zone.page, []).append(block.id)
            report.rendered_blocks.append(block.id)
        return paints, report

    def _unify_list_sizes(
        self, planned: list[tuple[IRBlock, str, tuple[Zone, ...], list[ZonePlan]]]
    ) -> None:
        """Pin every item of one list to the smallest size its items need.

        Each block is otherwise fitted to the largest size that holds it, so
        sibling bullets in one flow land at different point sizes and the list
        reads as ragged (arXiv 2609.20519 intro bullets). Group LIST_ITEM blocks
        by flow and, when their chosen sizes differ, re-plan every item at the
        common minimum: a smaller size always still fits text that fit at a
        larger one, so no item is dropped and the fail-closed spill behaviour is
        unchanged.
        """
        runs: list[list[int]] = []
        current_run: list[int] = []
        prev_flow: str | None = None
        prev_page: int | None = None
        prev_spine: int | None = None
        for i, (block, _text, zones, pending) in enumerate(planned):
            if block.block_type != BlockType.LIST_ITEM or not pending:
                if current_run:
                    runs.append(current_run)
                    current_run = []
                prev_flow = None
                prev_page = None
                prev_spine = None
                continue
            curr_flow = str(block.flow_id)
            curr_page = zones[0].page if zones else (block.bbox.page if block.bbox else 0)
            curr_spine = block.spine_index
            same_list = (
                bool(current_run)
                and curr_flow == prev_flow
                and (prev_page is None or abs(curr_page - prev_page) <= 1)
                and (
                    prev_spine is None
                    or curr_spine <= 0
                    or prev_spine <= 0
                    or (curr_spine - prev_spine) <= 1
                )
            )
            if not same_list and current_run:
                runs.append(current_run)
                current_run = []
            current_run.append(i)
            prev_flow = curr_flow
            prev_page = curr_page
            prev_spine = curr_spine
        if current_run:
            runs.append(current_run)

        for idxs in runs:
            if len(idxs) < 2:
                continue
            sizes = [planned[i][3][0].size for i in idxs]
            common = min(sizes)
            if max(sizes) - common < FIT_STEP_PT:
                continue  # already uniform within the fit grid
            for i in idxs:
                block, text, eff_zones, pending = planned[i]
                if abs(pending[0].size - common) < 1e-6:
                    continue
                replanned = self._plan_at_size(text, eff_zones, common)
                if replanned is not None:
                    planned[i] = (block, text, eff_zones, replanned)

    def _is_repeating_chrome(self, block: IRBlock, block_zones: tuple[Zone, ...]) -> bool:
        """True when every zone holds the whole source (a page-repeating head).

        A running head is extracted once but recurs on every page, so
        ``build_zones`` hands it one continuation zone per page, each carrying
        the SAME full line. A genuine paragraph split instead gives zones that
        each hold a different fragment. Detect the former so the head is stamped
        identically on every page rather than paginated across them (which left
        all but the first page showing the untranslated source head).
        """
        if len(block_zones) < 2:
            return False
        src = dehyph(block.source_text or "").strip()
        if not src or len(src) > REPEAT_CHROME_MAX_SRC:
            return False
        for zone in block_zones:
            row_text = dehyph(" ".join(zone.rows)).strip()
            if not row_text or src not in row_text:
                return False
        return True

    # -- Typst emission ---------------------------------------------------
    def _zone_typst(self, entry: ZonePlan, page_h: float) -> str:
        """Emit one zone: Typst owns line breaking and self-measures height.

        ``#ubt-fit`` uses Typst's own ``context { measure(...) }`` to verify
        that the shaped block fits within ``zone.height`` and steps down by
        0.4pt down to ``min_font_pt`` if HarfBuzz wraps an extra line. A block
        that still overflows at the floor is clipped to the zone's 2.5pt slack
        (within the 3.5pt inter-zone gap) instead of overprinting the block
        below.
        """
        from ubt.adapters.pdf.docling_blocks import parse_toc_entry_line
        from ubt.adapters.pdf.overlay_text import restore_zone_superscripts, typst_escape

        zone = entry.zone
        raw_text = entry.text.strip()
        if not raw_text:
            return ""

        toc_page: str | None = None
        for r in zone.rows:
            parsed_row = parse_toc_entry_line(r)
            if parsed_row is not None:
                toc_page = parsed_row[1]
                break
        if toc_page is not None:
            parsed_target = parse_toc_entry_line(raw_text)
            if parsed_target is not None:
                raw_text = parsed_target[0]

        restored_text = restore_zone_superscripts(raw_text, zone.rows)
        body = render_overlay_line(restored_text, self.probe.check, target_lang=self.target_lang)
        if not body:
            return ""
        sup_m = _LEADING_SUPER_RE.match(body)
        if sup_m:
            ascii_digits = sup_m.group(1).translate(_SUPERSCRIPT_TO_ASCII)
            body = f"#super[{ascii_digits}]#h(0.18em){body[sup_m.end() :]}"
        if toc_page is not None:
            escaped_page = typst_escape(toc_page)
            body = (
                f"#box(width: 100%)[{body}#h(4pt)"
                f"#box(width: 1fr, repeat(gap: 3.5pt)[.])"
                f"#h(4pt){escaped_page}]"
            )
        text_args = [f"size: {entry.size:.1f}pt"]
        if zone.bold:
            text_args.append('weight: "bold"')
        if zone.italic:
            text_args.append('style: "italic"')
        text_set = f"#set text({', '.join(text_args)})"
        if zone.align == "center":
            align_set = "#set align(center)\n#set par(justify: false)"
        elif zone.align == "justify":
            align_set = "#set par(justify: true)"
        else:
            align_set = "#set par(justify: false)"
        return (
            f"#place(top + left, dx: {zone.x0:.2f}pt, dy: {page_h - zone.y1:.2f}pt)"
            f"[{text_set}\n{align_set}\n"
            f"#ubt-fit({zone.width:.2f}pt, {zone.height:.2f}pt, {entry.size:.1f}pt, "
            f"{entry.min_font_pt:.1f}pt, {ZONE_INSET_TOP_PT:.1f}pt)[{body}]]"
        )

    def _zone_cover(self, zone: Zone, facts: PageFacts) -> str:
        """Opaque background rect for a scanned page (raster glyphs remain)."""
        return (
            f"#place(top + left, dx: {zone.x0:.2f}pt, dy: {facts.height - zone.y1:.2f}pt)"
            f"[#rect(width: {zone.width:.2f}pt, height: {zone.height:.2f}pt,"
            f' fill: rgb("{facts.bg}"))]'
        )

    def _font_tuple(self) -> str:
        """Emitted font list, resolved once against the compiler's own font view.

        ``_page_overlay`` runs per page, so the stack is built here rather than
        inline: pruning per page would re-warn and re-probe for one answer.
        """
        if self._font_tuple_cache is None:
            from ubt.core.language_profile import resolve_font_config

            cfg = resolve_font_config(self.target_lang)
            fonts = [self.font_family]
            for f in cfg.typst_fonts:
                if f not in fonts:
                    fonts.append(f)
            resolved = resolve_font_stack(
                fonts,
                target_lang=self.target_lang,
                typst_binary=self.typst_binary,
                # ``self.font_family`` is either the user's explicit override or
                # the profile's first name; either way it leads the stack, so
                # protect it from pruning exactly as the reflow path does.
                protected=(self.font_family,),
            )
            if not resolved.script_available:
                logger.warning(
                    "No target-script font installed for %r; the rigid "
                    "overlay will render its text as tofu.",
                    self.target_lang,
                )
            self._font_tuple_cache = resolved.as_typst_tuple()
        return self._font_tuple_cache

    @staticmethod
    def _typst_fit_preamble() -> str:
        return (
            "#let ubt-fit(w, h, base-sz, min-sz, inset-top, body) = context {\n"
            "  let sz = base-sz\n"
            "  let b = block(width: w, inset: (top: inset-top, bottom: 0pt, x: 0pt), text(size: sz, body))\n"
            "  while sz > min-sz and measure(b).height > h + 2.5pt {\n"
            "    sz = calc.max(min-sz, sz - 0.4pt)\n"
            "    b = block(width: w, inset: (top: inset-top, bottom: 0pt, x: 0pt), text(size: sz, body))\n"
            "  }\n"
            "  // A fitted block may overhang by up to 2.5pt (within the 3.5pt\n"
            "  // inter-zone gap). At the floor it can still exceed the zone, so clip\n"
            "  // to that slack instead of overprinting the block below.\n"
            "  let over = measure(b).height > h + 2.5pt\n"
            "  block(width: w, height: if over { h + 2.5pt } else { auto }, inset: (top: inset-top, bottom: 0pt, x: 0pt), clip: over, text(size: sz, body))\n"
            "}"
        )

    def _page_overlay(self, facts: PageFacts, entries: list[ZonePlan]) -> str:
        font_tuple_str = self._font_tuple()
        safe_lang = sanitize_lang_tag(self.target_lang)
        lang_setting = f', lang: "{safe_lang}"' if safe_lang else ""
        parts = [
            self._typst_fit_preamble(),
            _PAGE_SETUP.format(w=facts.width, h=facts.height).rstrip("\n"),
            # ``cjk-latin-spacing: none`` is a precondition of the width model,
            # not a style choice: ``font_metrics.text_width_pt`` sums glyph
            # advances and is only conservative when Typst does not add its own
            # CJK/Latin gap (see the ``text_fit``/``font_metrics`` docstrings).
            # Without this the model under-measures every mixed-script line and
            # ``clip: true`` below silently truncates the overflow.
            f"#set text(font: {font_tuple_str}{lang_setting}, cjk-latin-spacing: none)",
            "#set par(justify: false, leading: 0.52em)",
        ]
        for entry in entries:
            if facts.bg:
                parts.append(self._zone_cover(entry.zone, facts))
            snippet = self._zone_typst(entry, facts.height)
            if snippet:
                parts.append(snippet)
        return "\n".join(parts) + "\n"

    def _batch_page_overlay(
        self,
        pages: dict[int, PageFacts],
        paints: dict[int, list[ZonePlan]],
    ) -> tuple[str, list[int]]:
        """Assemble a single multi-page Typst document for all planned pages."""
        font_tuple_str = self._font_tuple()
        safe_lang = sanitize_lang_tag(self.target_lang)
        lang_setting = f', lang: "{safe_lang}"' if safe_lang else ""

        doc_parts = [
            self._typst_fit_preamble(),
            f"#set text(font: {font_tuple_str}{lang_setting}, cjk-latin-spacing: none)",
            "#set par(justify: false, leading: 0.52em)",
        ]

        compiled_pages: list[int] = []

        for page_no, entries in sorted(paints.items()):
            facts = pages.get(page_no)
            if facts is None or not entries:
                continue

            compiled_pages.append(page_no)

            doc_parts.append(_PAGE_SETUP.format(w=facts.width, h=facts.height).rstrip("\n"))
            for entry in entries:
                if facts.bg:
                    doc_parts.append(self._zone_cover(entry.zone, facts))
                snippet = self._zone_typst(entry, facts.height)
                if snippet:
                    doc_parts.append(snippet)

        return "\n".join(doc_parts) + "\n", compiled_pages

    # -- public entry -----------------------------------------------------
    async def render(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
    ) -> tuple[Path, RigidReport]:
        """Typeset translated blocks into their rigid regions."""
        from ubt.adapters.pdf.typst_compile import typst_available, typst_compile

        # Honour the call-site language: fonts and escaping read
        # ``self.target_lang`` throughout, so a caller that passed a different
        # value used to be silently ignored.
        self.target_lang = target_lang
        if not typst_available(self.typst_binary):
            raise DocumentParseError(
                f"Typst compiler binary '{self.typst_binary}' not found on system "
                "PATH; the rigid engine cannot render. Install Typst or use "
                "render_engine='publication'."
            )

        source_pdf = Path(manifest.source_path)
        if not source_pdf.exists():
            raise DocumentParseError(f"Source PDF not found for rigid render: {source_pdf}")
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _ = target_lang

        import tempfile

        def _extract_all_pages() -> dict[int, Any]:
            with pikepdf.open(str(source_pdf)) as probe_pdf:
                total = len(probe_pdf.pages)
            target_pages = _candidate_rigid_pages(total, blocks)
            return extract_pages(source_pdf, target_pages, list(blocks))

        pages = await asyncio.to_thread(_extract_all_pages)
        zone_map = build_zones(pages, blocks, allow_chrome_bands=self.translate_chrome)
        paints, report = await asyncio.to_thread(
            self._plan_blocks,
            blocks,
            zone_map,
            {p: f.height for p, f in pages.items()},
            {p: list(f.images) for p, f in pages.items()},
        )

        _assert_paintable(paints, report)

        page_reports = {
            pno: RigidPageReport(page=pno, painted_zones=len(entries))
            for pno, entries in paints.items()
        }
        report.pages = [page_reports[p] for p in sorted(page_reports)]

        # Snapshot the fully-planned render set BEFORE any page-overlay demotion:
        # erasure is keyed off what was planned to render, restricted to the
        # pages whose overlay actually compiled (see ``_erase_rects_by_page``).
        planned_rendered_ids = set(report.rendered_blocks)

        overlays: dict[int, str] = {}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # Fast single-pass batch compilation across all pages
            batch_typ, compiled_pages = self._batch_page_overlay(pages, paints)
            batch_succeeded = False
            if compiled_pages:
                batch_typ_path = tmp_dir / "rigid_batch.typ"
                batch_pdf_path = tmp_dir / "rigid_batch.pdf"
                batch_typ_path.write_text(batch_typ, encoding="utf-8")
                ok, err = await asyncio.to_thread(
                    typst_compile, str(batch_typ_path), str(batch_pdf_path), self.typst_binary
                )
                if ok and batch_pdf_path.exists():
                    try:
                        with pikepdf.open(str(batch_pdf_path)) as batch_pdf:
                            if len(batch_pdf.pages) == len(compiled_pages):
                                for idx, pno in enumerate(compiled_pages):
                                    single_pdf = tmp_dir / f"rigid_{pno}.pdf"
                                    with pikepdf.new() as dst:
                                        dst.pages.append(batch_pdf.pages[idx])
                                        dst.save(str(single_pdf))
                                    overlays[pno] = str(single_pdf)
                                batch_succeeded = True
                            else:
                                logger.warning(
                                    "Batch overlay page count mismatch (%d != %d); falling back to per-page",
                                    len(batch_pdf.pages),
                                    len(compiled_pages),
                                )
                    except Exception as exc:
                        logger.warning(
                            "Failed to split batch overlay PDF (%s); falling back to per-page compile",
                            exc,
                        )
                        overlays.clear()
                else:
                    logger.debug(
                        "Batch overlay compilation did not succeed (%s); falling back to per-page",
                        err,
                    )

            # Fall back to page-by-page compilation if batch failed
            if not batch_succeeded:
                for page_no, entries in sorted(paints.items()):
                    facts = pages.get(page_no)
                    if facts is None or not entries:
                        continue
                    typ = self._page_overlay(facts, entries)
                    typ_path = tmp_dir / f"rigid_{page_no}.typ"
                    typ_path.write_text(typ, encoding="utf-8")
                    pdf_path = tmp_dir / f"rigid_{page_no}.pdf"
                    ok, err = await asyncio.to_thread(
                        typst_compile, str(typ_path), str(pdf_path), self.typst_binary
                    )
                    if not ok:
                        logger.warning("rigid overlay failed on page %d: %s", page_no, err)
                        # Demote every block planned for this page to a block-scoped
                        # render skip: the page keeps its source text, and the loss
                        # must reach the quality report instead of vanishing behind
                        # a page-level entry.
                        _demote_page_blocks(report, page_no)
                        page_reports[page_no].painted_zones = 0
                        continue
                    overlays[page_no] = str(pdf_path)

            aborted_pages: list[int] = []

            def _merge_sync() -> None:
                from ubt.adapters.pdf.stream_strip import shared_form_objgens

                erase_rects = _erase_rects_by_page(zone_map, planned_rendered_ids, set(overlays))

                with pikepdf.open(str(source_pdf)) as pdf:
                    # Forms drawn by several pages must not be rewritten while
                    # stripping one of them (their text would vanish everywhere).
                    shared_forms = shared_form_objgens(pdf)
                    for page_no, overlay_path in sorted(overlays.items()):
                        page = pdf.pages[page_no - 1]
                        guards: list[Rect] = []
                        for block in blocks:
                            if block.bbox is None or block.bbox.page != page_no:
                                continue
                            if block.block_type not in PROSE_BLOCK_TYPES:
                                guards.append(
                                    (block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)
                                )
                        dropped, painted = _strip_and_merge_page(
                            pdf,
                            page,
                            page_no,
                            overlay_path,
                            erase_rects.get(page_no, []),
                            guards,
                            shared_forms,
                        )
                        page_reports[page_no].stripped_ops += dropped
                        if not painted:
                            aborted_pages.append(page_no)
                    pdf.save(str(out_path))

            await asyncio.to_thread(_merge_sync)

            for page_no in aborted_pages:
                _demote_page_blocks(report, page_no)
                page_reports[page_no].painted_zones = 0

        preserved_count = sum(
            1
            for _, reason in report.skipped
            if reason
            in (
                "non_prose",
                "chrome",
                "header",
                "footer",
                "header_band",
                "footer_band",
                "page_number",
                "verbatim",
                "symbol",
                "policy",
                "caption",
                "no_bbox",
                "no_page_height",
            )
        )
        anomalous_skips = len(report.skipped) - preserved_count
        if anomalous_skips > 0:
            logger.info(
                "Rigid typesetting: %d block(s) rendered, %d source element(s) preserved intact (%d fail-closed to prevent overlap), %d page(s) painted",
                len(report.rendered_blocks),
                preserved_count,
                anomalous_skips,
                len(paints),
            )
        else:
            logger.info(
                "Rigid typesetting: %d block(s) rendered, %d source element(s) preserved intact, %d page(s) painted",
                len(report.rendered_blocks),
                preserved_count,
                len(paints),
            )
        if report.degraded_blocks:
            logger.info(
                "Rigid typesetting: %d block(s) shrunk below the region floor to avoid leaving source visible",
                len(report.degraded_blocks),
            )
        if report.overflow_blocks:
            logger.info(
                "Rigid typesetting: %d block(s) painted past their box (bounded overflow) to avoid leaving source visible",
                len(report.overflow_blocks),
            )
        return out_path, report
