"""Anchoring: pdfium text layer votes first, driver boxes last.

Two modes, decided per page by EVIDENCE (never by config):
- proofread: pdfium lines exist. Geometry and text stay pdfium; the driver
  transcript only audits (agreement stats out, corrections nowhere — v0
  never rewrites pdfium text from a weaker witness).
- recognition: no usable pdfium text (true scan). Measured driver boxes
  become block geometry (XY order = driver reading order). Unmeasured
  (LLM-VLM) drivers fail CLOSED here — a page-spanning guess block would
  poison every downstream stage.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ubt.adapters.pdf.coordinate_resolver import rotate_rect_clockwise, undo_page_rotation
from ubt.adapters.pdf.vlm.types import PageTranscript

_WS_RE = re.compile(r"\s+")
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "‐": "-", "-": "-"})


def _norm(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text or "").translate(_QUOTES)
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Cc")
    return _WS_RE.sub("", folded)


@dataclass(frozen=True)
class AnchoredLine:
    """One page line with settled geometry + provenance."""

    text: str
    box: tuple[float, float, float, float]  # PDF points, bottom-left origin
    provenance: str  # "pdfium" | "pdfium+proofread" | "vlm-measured"
    needs_review: bool = False


@dataclass(frozen=True)
class AnchorStats:
    matched: int = 0
    vlm_only: int = 0
    pdfium_only: int = 0


def _best_match(nline: str, pdfium_norms: list[str]) -> int | None:
    """Index of the pdfium line containing (or contained in) the VLM line."""
    for i, pnorm in enumerate(pdfium_norms):
        if nline and (nline in pnorm or pnorm in nline):
            return i
    return None


def anchor_transcript(
    pdfium_lines: list[tuple[str, tuple[float, float, float, float]]],
    transcript: PageTranscript,
    page_size_pt: tuple[float, float],
    rotation: int = 0,
) -> tuple[list[AnchoredLine], AnchorStats]:
    """Settle per-line text+geometry. Pure function (unit-testable).

    ``page_size_pt`` is the displayed page's size and ``rotation`` its
    ``/Rotate``; the driver has already mapped the boxes it measured back to
    unrotated user space. ``rotation`` is needed here only for the fallback box
    of a line the driver could not place -- that box is the whole page, and
    "the whole page" is display-shaped on a rotated page.
    """
    usable_pdfium = [(t, b) for t, b in pdfium_lines if _norm(t)]
    pnorms = [_norm(t) for t, _ in usable_pdfium]

    # Recognition mode needs measured geometry; hallucinated will not do.
    if not usable_pdfium:
        if not transcript.measured_boxes:
            raise ValueError(
                f"driver {transcript.engine!r} has no measured boxes: "
                "refusing to guess page geometry (fail closed)"
            )
        width, height = page_size_pt
        # Rotation lives in ONE place -- the driver's box conversion -- and the
        # caller passes the same value here, so this is the same frame mapping
        # applied to the one box the driver never produced.
        whole_page = rotate_rect_clockwise(
            (0.0, 0.0, width, height), undo_page_rotation(rotation), width, height
        )
        out = [
            AnchoredLine(
                text=ln.text.strip(),
                box=ln.measured_box or whole_page,
                provenance="vlm-measured",
                needs_review=ln.measured_box is None,
            )
            for ln in transcript.lines
            if ln.text.strip()
        ]
        return out, AnchorStats(vlm_only=len(out))

    claimed: set[int] = set()
    out_lines: list[AnchoredLine] = []
    stats = AnchorStats()
    for ln in transcript.lines:
        hit = _best_match(_norm(ln.text), pnorms)
        if hit is not None:
            # Agreement counts even on duplicates (table cells repeat):
            # output is every pdfium line regardless; only telemetry and
            # the proofread tag ride on first-claim.
            claimed.add(hit)
            stats = AnchorStats(stats.matched + 1, stats.vlm_only, stats.pdfium_only)
        else:
            stats = AnchorStats(stats.matched, stats.vlm_only + 1, stats.pdfium_only)
        # Unmatched VLM lines are DROPPED in proofread mode: pdfium is the
        # stronger witness and VLM insertions are usually headers/footers the
        # ledger models separately, or hallucinations.
    for i, (text, box) in enumerate(usable_pdfium):
        if i in claimed:
            out_lines.append(AnchoredLine(text=text, box=box, provenance="pdfium+proofread"))
        else:
            out_lines.append(AnchoredLine(text=text, box=box, provenance="pdfium"))
            stats = AnchorStats(stats.matched, stats.vlm_only, stats.pdfium_only + 1)
    # Reading order = pdfium geometry order (top-to-bottom, left-to-right);
    # the VLM's order is discarded wherever pdfium has a vote.
    out_lines.sort(key=lambda ln: (-ln.box[3], ln.box[0]))
    return out_lines, stats
