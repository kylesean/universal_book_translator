"""The visual gate's thresholds, in one typed place with their calibration notes.

Every number here was tuned against a measured corpus, and the comment beside
each names the evidence: the point of collecting them is that a future operator
can see *what* was tuned, on *what*, rather than reverse-engineering magic
numbers out of a 2,500-line module. Defaults are exactly the values the gate
shipped with, so moving them here changes no behaviour.

Override through ``UBTConfig``'s ``[gate]`` section (see
``ubt.core.config.GateConfig``); :func:`resolve` merges that section onto the
defaults and rejects unknown keys instead of silently ignoring a typo.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateThresholds:
    """Measured constants the visual gate decides with."""

    # -- geometry (T1) ------------------------------------------------------
    #: Two artifact rects this alike are the same text placed twice. Calibrated
    #: on the ATC'25 replay: legitimate adjacent lines never reach it.
    overlap_iou: float = 0.6
    #: Rects smaller than this area are skipped: punctuation and leader dots
    #: collide harmlessly at that size.
    min_overlap_area_pt2: float = 500.0
    #: Bound on the per-page O(n^2) pair loop. A page above it is measured on
    #: its first N rects and the truncation is reported.
    max_overlap_rects_per_page: int = 200
    #: Slack allowed before a block counts as clipped. Absorbs the sub-point
    #: differences between the MediaBox origin and the extractor's frame.
    out_of_bounds_tolerance_pt: float = 5.0
    #: Share of page height at each edge that belongs to running heads, folios
    #: and rules rather than body flow (same bands ``source_metrics`` measures).
    furniture_band_share: float = 0.08

    # -- type size ----------------------------------------------------------
    #: A median content glyph under this size is physically unreadable. No
    #: legible convention sets body text below 5pt; the corpus's healthy
    #: minimum reads 5.35pt and the broken file 0.49pt.
    type_illegible_pt: float = 4.0
    #: Share of a page's content glyphs under the floor that makes the page
    #: collapsed. Highest healthy reading is 17 %, the broken file 98 %.
    type_collapsed_fraction: float = 0.5
    #: Fewer content characters than this and the page has too little text to
    #: judge (a figure page with one caption word).
    type_audit_min_glyphs: int = 20

    # -- pixel heuristics ---------------------------------------------------
    #: Raster "almost no ink" bands, measured at 110 dpi on A4: a folio reads
    #: mean 255.0 / std 0.9, one 9pt caption 254.9 / 3.4.
    blank_mean_threshold: float = 250.0
    blank_std_threshold: float = 6.0
    #: A near-black pixel for the missing-glyph-square heuristic.
    black_pixel_value: int = 12
    #: Share of near-black area that nominates a page as a tofu candidate. It
    #: only nominates: a handful of squares per page stays below it, which is
    #: why the deterministic font-coverage check exists beside it.
    black_fraction_threshold: float = 0.05
    #: Share of a line's box that must carry ink before it counts as drawn.
    occlusion_ink_floor: float = 0.02

    # -- canvas ink collision (faithful pages) ------------------------------
    #: Page-global ink ratio over the stripped background that counts as a
    #: collision. Calibrated on the ATC'25 replay and chapter-3.pdf.
    canvas_collision_ratio: float = 0.03
    #: Per-placed-block ratio, the localized counterpart of the above.
    canvas_region_ratio: float = 0.3
    #: Pixel difference above which two rasters count as different.
    canvas_pixel_diff: int = 128
    #: Minimum differing-pixel count before a region is considered at all.
    canvas_min_diff_pixels: int = 40
    #: Darkest luminance that still counts as ink.
    canvas_ink_luminance: int = 200

    # -- sampling -----------------------------------------------------------
    #: Cap on the adaptive zero-token sample for long books.
    adaptive_sample_cap: int = 12
    #: Cap on artifact geometry pages (the even sample over a long book).
    max_artifact_geometry_pages: int = 120
    #: Cap on VLM-judged pages. The gate judges 10 % of the artifact, never
    #: fewer than one page, and never more than this.
    max_vlm_pages: int = 20

    # -- parity -------------------------------------------------------------
    #: Share of the artifact below which the target script is "sparse".
    #: Calibrated on the bilingual zipper layout.
    sparse_target_ratio: float = 0.08

    def merged(self, overrides: Mapping[str, Any] | None) -> GateThresholds:
        """Return a copy with ``overrides`` applied, rejecting unknown keys."""
        if not overrides:
            return self
        known = set(self.__dataclass_fields__)
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(
                f"unknown gate threshold(s): {', '.join(unknown)}; "
                f"known: {', '.join(sorted(known))}"
            )
        merged = {name: overrides.get(name, getattr(self, name)) for name in known}
        return GateThresholds(**merged)


DEFAULT_GATE_THRESHOLDS = GateThresholds()


def resolve(overrides: Mapping[str, Any] | None = None) -> GateThresholds:
    """The effective thresholds: the defaults with any configured override."""
    return DEFAULT_GATE_THRESHOLDS.merged(overrides)


__all__ = ["DEFAULT_GATE_THRESHOLDS", "GateThresholds", "resolve"]
