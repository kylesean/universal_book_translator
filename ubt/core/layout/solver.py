"""CELS Stage 3B: the vertical solve.

The math form is a Cassowary linear system. Like the rest of
``ubt.core.layout`` this module is deliberately pure: inputs in, solved
geometry out. It opens no PDF and owns no renderer, and the Cassowary solver
arrives through ``kiwisolver`` (the ``[cels]`` extra), imported lazily so a
default install can still import this module.

Only the continuous vertical degrees of freedom live here. Line counts are a
``ceil()`` over wrapping -- a piecewise constant that changes discontinuously
with size, tracking and width -- so a Cassowary system must never be asked to
derive them; the caller fixes each block's required height before the solve.

Phase 3B -- ``solve_vertical``
    kiwisolver over ``y_top/y_bottom`` and ``deficit`` per block. The strength
    assignment follows the design, with one deliberate hardening: the fit
    inequality ``(height + deficit ≥ H_req)`` is **required**, not strong. As
    the strong term the design names, it can conflict with the equally-strong
    ``deficit == 0`` on an over-full track, and which one Cassowary then breaks
    is unspecified. Required-fit keeps the semantics deterministic: track bounds
    and gaps can force a height below its requirement, the strong
    ``deficit == 0`` is sacrificed, and the resulting positive deficit is the
    "unmet height" that triggers Stage 4 -- never a shrunk font, never a
    silently dropped block.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ubt.core.exceptions import UBTError

__all__ = [
    "DEFICIT_TRIGGER_PT",
    "DEFAULT_MIN_GAP_PT",
    "SolvedBlockSpec",
    "VerticalBlockInput",
    "VerticalTrack",
    "solve_vertical",
]

#: Stage 4 trigger: a deficit above this is a real shortfall.
DEFICIT_TRIGGER_PT = 1.5
#: Default minimum gap between consecutive blocks in one track.
DEFAULT_MIN_GAP_PT = 2.0


class CELSSolverUnavailableError(UBTError):
    """``kiwisolver`` (the ``[cels]`` extra) is not installed."""


@dataclass(frozen=True, slots=True)
class VerticalBlockInput:
    """One block entering the 3B vertical solve.

    ``required_height_pt`` is the height the caller chose for the block's
    style; the solver never derives it. ``source_top_pt`` is the source block's
    top edge in PDF user space (y up), and inputs are ordered top-down, so
    block 1 is the highest on the page.
    """

    index: int
    required_height_pt: float
    source_height_pt: float
    source_top_pt: float


@dataclass(frozen=True, slots=True)
class VerticalTrack:
    """The vertical extent one track's flow may occupy (PDF y up)."""

    y_min_pt: float
    y_max_pt: float

    @property
    def height(self) -> float:
        return max(0.0, self.y_max_pt - self.y_min_pt)


@dataclass(frozen=True, slots=True)
class SolvedBlockSpec:
    """3B's output per block: where the block goes, and what it still owes.

    ``deficit_pt > 0`` means the track could not grant the block its measured
    height even after every medium/weak preference was sacrificed. That is a
    Stage 4 trigger (LLM condensation against a char budget), never a reason
    to shrink the font here.
    """

    index: int
    y_top_pt: float
    y_bottom_pt: float
    height_pt: float
    deficit_pt: float
    gap_to_next_pt: float | None
    needs_stage4: bool


# ---------------------------------------------------------------------------
# Stage 3B
# ---------------------------------------------------------------------------


def _load_kiwisolver() -> Any:
    try:
        import kiwisolver
    except ImportError as exc:  # pragma: no cover - install-shape dependent
        raise CELSSolverUnavailableError(
            "CELS Stage 3B needs kiwisolver: install the [cels] extra "
            "(pip install 'universal-book-translator[cels]')"
        ) from exc
    return kiwisolver


def solve_vertical(
    blocks: Sequence[VerticalBlockInput],
    *,
    track: VerticalTrack,
    min_gap_pt: float = DEFAULT_MIN_GAP_PT,
) -> tuple[SolvedBlockSpec, ...]:
    """Distribute blocks vertically inside one track: Phase 3B.

    Strengths, and what each one buys:

    - **required** -- every block stays inside the track, consecutive blocks
      keep at least ``min_gap_pt``, heights stay non-negative, and
      ``height + deficit >= required_height``. The fit inequality carries the
      deficit slack, so it can always hold: when the track genuinely cannot
      grant a block its measured height, the height shrinks and the deficit
      records the shortfall instead of the solve failing or the layout
      overlapping.
    - **strong** -- ``deficit == 0``: prefer granting every block its full
      measured height. Sacrificed first, per block, exactly where the track
      runs out.
    - **medium** -- keep each block's source height and source gap.
    - **weak** -- keep each block's top near its source position.

    Inputs are ordered top-down (descending ``source_top_pt``); the returned
    specs carry the input's ``index``. A positive ``deficit_pt`` beyond
    :data:`DEFICIT_TRIGGER_PT` flags the block for Stage 4 (condensation), and
    is never resolved here by shrinking the group's font.
    """
    if not blocks:
        return ()
    ordered = sorted(blocks, key=lambda block: -block.source_top_pt)
    for block in ordered:
        if block.required_height_pt < 0 or block.source_height_pt < 0:
            raise ValueError(f"block {block.index} has a negative height")

    # Clamp top values to track bounds so boundary jitter does not fail the solve
    clamped_tops = [
        max(track.y_min_pt, min(track.y_max_pt, block.source_top_pt)) for block in ordered
    ]

    ks = _load_kiwisolver()
    solver = ks.Solver()
    tops = [ks.Variable(f"y_top_{block.index}") for block in ordered]
    bottoms = [ks.Variable(f"y_bot_{block.index}") for block in ordered]
    deficits = [ks.Variable(f"deficit_{block.index}") for block in ordered]
    required = ks.strength.required
    strong = ks.strength.strong
    medium = ks.strength.medium
    weak = ks.strength.weak

    for i, block in enumerate(ordered):
        top, bottom, deficit = tops[i], bottoms[i], deficits[i]
        solver.addConstraint((bottom >= track.y_min_pt) | required)
        solver.addConstraint((top <= track.y_max_pt) | required)
        solver.addConstraint((top - bottom >= 0) | required)
        solver.addConstraint((deficit >= 0) | required)
        # The fit inequality with its deficit slack: always satisfiable, so a
        # genuinely over-full track surfaces as deficit, never as an exception.
        solver.addConstraint((top - bottom + deficit >= block.required_height_pt) | required)
        solver.addConstraint((deficit == 0) | strong)
        solver.addConstraint((top - bottom == block.source_height_pt) | medium)
        solver.addConstraint((top == clamped_tops[i]) | weak)
        if i > 0:
            previous_bottom = bottoms[i - 1]
            gap = previous_bottom - top
            solver.addConstraint((gap >= min_gap_pt) | required)
            source_gap = max(
                (clamped_tops[i - 1] - ordered[i - 1].source_height_pt) - clamped_tops[i],
                min_gap_pt,
            )
            solver.addConstraint((gap == source_gap) | medium)
    solver.updateVariables()

    specs: list[SolvedBlockSpec] = []
    for i, block in enumerate(ordered):
        y_top = float(tops[i].value())
        y_bottom = float(bottoms[i].value())
        deficit = float(deficits[i].value())
        gap = None if i + 1 >= len(ordered) else float(bottoms[i].value() - tops[i + 1].value())
        specs.append(
            SolvedBlockSpec(
                index=block.index,
                y_top_pt=round(y_top, 4),
                y_bottom_pt=round(y_bottom, 4),
                height_pt=round(y_top - y_bottom, 4),
                deficit_pt=round(max(0.0, deficit), 4),
                gap_to_next_pt=None if gap is None else round(gap, 4),
                needs_stage4=deficit > DEFICIT_TRIGGER_PT,
            )
        )
    if math.isclose(sum(spec.height_pt for spec in specs), 0.0, abs_tol=1e-9) and any(
        block.required_height_pt > 0 for block in ordered
    ):
        raise ValueError("vertical solve collapsed every block to zero height")
    return tuple(specs)
