"""Layout geometry and solving for adaptive delivery (CELS).

Stage 1–2 turns measured boxes into column tracks / pinned obstacles /
elastic segments (:mod:`ubt.core.layout.cels`); Stage 3 solves the style group
and the vertical flow (:mod:`ubt.core.layout.solver`). Both are deliberately
pure: no PDF, no render backend; measurement and the Cassowary solver arrive
through injected delegates, so the modules stay unit-testable against
synthetic pages and importable without the ``[cels]`` extra.

Target design: ``docs/design/adaptive_delivery_prd.md`` §7–§8. Staged plan:
``docs/guides/ADAPTIVE_DELIVERY_TECH_DESIGN.md`` §7.1.
"""

from __future__ import annotations

from ubt.core.layout.cels import (
    Box,
    ColumnDetection,
    ColumnTrack,
    ElasticFlowSegment,
    GutterCandidate,
    InlineRigidBox,
    PinnedObstacle,
    TrackOverrun,
    assign_lines,
    detect_columns,
    pin_from,
    track_overruns,
)
from ubt.core.layout.coordinator import (
    CELSPageLayoutSolution,
    PageLayoutBlock,
    Stage4CondensationItem,
    TrackLayoutSolution,
    solve_page_cels,
)
from ubt.core.layout.distortion import (
    DistortionSubscores,
    PageDistortionScore,
    compute_page_distortion,
)
from ubt.core.layout.solver import (
    SolvedBlockSpec,
    StyleBlockInput,
    StyleFit,
    StyleGroupSolution,
    VerticalBlockInput,
    VerticalTrack,
    condensation_char_budget,
    solve_style_group,
    solve_vertical,
)

__all__ = [
    "Box",
    "CELSPageLayoutSolution",
    "ColumnDetection",
    "ColumnTrack",
    "DistortionSubscores",
    "ElasticFlowSegment",
    "GutterCandidate",
    "InlineRigidBox",
    "PageDistortionScore",
    "PageLayoutBlock",
    "PinnedObstacle",
    "SolvedBlockSpec",
    "Stage4CondensationItem",
    "StyleBlockInput",
    "StyleFit",
    "StyleGroupSolution",
    "TrackLayoutSolution",
    "TrackOverrun",
    "VerticalBlockInput",
    "VerticalTrack",
    "assign_lines",
    "compute_page_distortion",
    "condensation_char_budget",
    "detect_columns",
    "pin_from",
    "solve_page_cels",
    "solve_style_group",
    "solve_vertical",
    "track_overruns",
]
