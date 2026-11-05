"""Layout geometry and solving for adaptive delivery (CELS).

Stage 1–2 turns measured boxes into column tracks / pinned obstacles /
elastic segments (:mod:`ubt.core.layout.cels`); Stage 3B solves the vertical
flow (:mod:`ubt.core.layout.solver`). Both are deliberately pure: no PDF, no
render backend; measured inputs and the Cassowary solver arrive through
injected values, so the modules stay unit-testable against synthetic pages and
importable without the ``[cels]`` extra.
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
    VerticalBlockInput,
    VerticalTrack,
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
    "TrackLayoutSolution",
    "TrackOverrun",
    "VerticalBlockInput",
    "VerticalTrack",
    "assign_lines",
    "compute_page_distortion",
    "detect_columns",
    "pin_from",
    "solve_page_cels",
    "solve_vertical",
    "track_overruns",
]
