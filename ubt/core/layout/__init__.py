"""Layout geometry for adaptive delivery (CELS).

Only the Stage 1–2 half lives here: measured boxes in, column tracks /
pinned obstacles / elastic segments out. It is deliberately pure (no PDF, no
measurement service, no constraint solver) so geometry can be unit-tested
against synthetic pages and reused by any render backend.

Target design: ``docs/design/adaptive_delivery_prd.md`` §7. Staged plan:
``docs/guides/ADAPTIVE_DELIVERY_TECH_DESIGN.md`` §7.1.
"""

from __future__ import annotations

from ubt.core.layout.cels import (
    Box,
    ColumnDetection,
    ColumnTrack,
    ElasticFlowSegment,
    GutterCandidate,
    PinnedObstacle,
    TrackOverrun,
    assign_lines,
    detect_columns,
    pin_from,
    track_overruns,
)

__all__ = [
    "Box",
    "ColumnDetection",
    "ColumnTrack",
    "ElasticFlowSegment",
    "GutterCandidate",
    "PinnedObstacle",
    "TrackOverrun",
    "assign_lines",
    "detect_columns",
    "pin_from",
    "track_overruns",
]
