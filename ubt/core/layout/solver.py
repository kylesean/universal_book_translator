"""CELS Stage 3: the two-phase style and vertical solve.

Design: ``docs/design/adaptive_delivery_prd.md`` §8.2.1 (the math form) and
§7.4.3 of the tech design (the two phases). Like the rest of
``ubt.core.layout`` this module is deliberately pure: measured inputs in,
solved geometry out. It opens no PDF and owns no renderer; the Typst-backed
measurement arrives as an injected :data:`BlockMeasureFn`, and the Cassowary
solver arrives through ``kiwisolver`` (the ``[cels]`` extra), imported lazily so
a default install can still import this module.

The two phases, and why they must not merge
-------------------------------------------
``N_i`` (a block's line count) is a ``ceil()`` over wrapping -- a piecewise
constant that changes discontinuously with size, tracking and width. Cassowary
only solves continuous systems, so handing it ``N_i`` as something to derive is
the "pseudo-linear, fake-feasible" failure the tech design forbids. Phase 3A
therefore fixes the style (and with it every ``N_i``, via real measurement),
and Phase 3B solves only the continuous vertical degrees of freedom.

Phase 3A -- ``solve_style_group``
    One size per style group (PRD §8.2: "字号差异只能来自 StyleRole"). The
    search walks the knobs in the PRD's *preference* order -- tracking, then
    leading, then right-borrow, and size reduction last (P10: 缩字号是最后手段)
    -- and accepts the first candidate whose measured heights fit the track.

    One documented deviation from §7.4.3: the search order lists ``γ_x`` first,
    but the only measurement backend that exists (the Typst probe) **refuses
    non-unity horizontal scale** -- Typst's static-font ``stretch`` does not
    change advance widths, so a γ_x swept here would be measured on a lie. 3A
    therefore pins γ_x = 1.0 and leaves horizontal compression to Stage 5's
    line-level fitting (``solve_horizontal``), which reads widths from the
    PDF's own ``/Widths``. When a shaping-aware measure path (HarfBuzz) lands,
    γ_x joins the search; until then sweeping it here would violate the
    project's "代理估算只能做预筛" rule at the moment it matters most.

Phase 3B -- ``solve_vertical``
    kiwisolver over ``y_top/y_bottom`` and ``deficit`` per block. The strength
    assignment is the PRD's, with one deliberate hardening: the fit inequality
    ``(height + deficit ≥ H_req)`` is **required**, not strong. As the PRD
    labels it (strong) it can conflict with the equally-strong ``deficit == 0``
    on an over-full track, and which one Cassowary then breaks is unspecified.
    Required-fit keeps the semantics deterministic: track bounds and gaps can
    force a height below its requirement, the strong ``deficit == 0`` is
    sacrificed, and the resulting positive deficit is exactly the PRD's
    "unmet height" that triggers Stage 4 -- never a shrunk font, never a
    silently dropped block.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ubt.core.exceptions import UBTError

__all__ = [
    "BODY_FLOOR_PT",
    "BLOCKED_BY_GEOMETRY",
    "DEFICIT_TRIGGER_PT",
    "DEFAULT_MIN_GAP_PT",
    "MAX_EXPANSION_RATIO",
    "NON_BODY_FLOOR_PT",
    "SIZE_STEP_PT",
    "StyleBlockInput",
    "StyleFit",
    "StyleGroupSolution",
    "SolvedBlockSpec",
    "VerticalBlockInput",
    "VerticalTrack",
    "condensation_char_budget",
    "solve_style_group",
    "solve_vertical",
]

#: Absolute body floor (PRD §7.4.3): a body group never solves below this.
BODY_FLOOR_PT = 7.5
#: Non-body roles (captions, table cells) may go lower, but must warn.
NON_BODY_FLOOR_PT = 5.5
#: Candidate size step for the discrete 3A search (PRD: 0.25pt).
SIZE_STEP_PT = 0.25
#: faithful may enlarge a group to at most 1.10x its source size (PRD §8.2).
MAX_EXPANSION_RATIO = 1.10
#: Stage 4 trigger: a deficit above this is a real shortfall (PRD §7.4.4).
DEFICIT_TRIGGER_PT = 1.5
#: Default minimum gap between consecutive blocks in one track.
DEFAULT_MIN_GAP_PT = 2.0
#: Micro-typography bounds (PRD §7.4.3): tracking −0.03..+0.06em, leading 1.08..1.55.
TRACKING_MIN_EM = -0.03
TRACKING_STEP_EM = 0.005
LEADING_MIN_RATIO = 1.08
LEADING_MAX_RATIO = 1.55
LEADING_STEP = 0.05
#: Right-borrow fractions tried before any size reduction.
BORROW_FRACTIONS = (0.0, 0.5, 1.0)
#: Reason codes on :attr:`StyleGroupSolution.reason`.
SOURCE_STYLE_FITS = "source_style_fits"
ADJUSTED_TO_FIT = "adjusted_to_fit"
FLOOR_EXHAUSTED = "floor_exhausted"
BLOCKED_BY_GEOMETRY = "blocked_by_geometry"


class CELSSolverUnavailableError(UBTError):
    """``kiwisolver`` (the ``[cels]`` extra) is not installed."""


class MeasuredLike(Protocol):
    """The slice of a measurement result the solver reads.

    Structural on purpose: :class:`ubt.delivery.measurement.MeasuredBlock`
    satisfies it, and importing that type here would point ``ubt.core.layout``
    at ``ubt.delivery`` -- a package whose own errors module already reaches
    back into ``ubt.core``. The solver must not close that cycle.
    """

    line_count: int
    height_pt: float


#: ``(text, size_pt, tracking_em, leading_pt, width_pt) -> measurement``.
#:
#: The caller closes over its own style template (font family, version,
#: language, justify) and builds the backend's style object per call, so the
#: solver never touches a style type it cannot import. A real delegate wraps
#: :class:`ubt.delivery.measurement.MeasurementService`; tests pass a pure
#: function, which is what keeps this module's tier fast and offline.
BlockMeasureFn = Callable[[str, float, float, float, float], MeasuredLike]


@dataclass(frozen=True, slots=True)
class StyleBlockInput:
    """One block of a style group, as 3A sees it.

    ``borrowable_pt`` is the room between the block's current right edge and
    the track's right limit (``X_col_right``); the caller computes it from the
    geometry it already holds, so the solver stays box-free.
    """

    index: int
    text: str
    role: Literal["body", "caption", "table_cell"]
    source_size_pt: float
    source_leading_pt: float
    source_width_pt: float
    source_height_pt: float
    borrowable_pt: float = 0.0


@dataclass(frozen=True, slots=True)
class StyleFit:
    """One block's measured outcome under the group's solved style."""

    index: int
    width_pt: float
    line_count: int
    height_pt: float
    #: Set only when the group does not fit: the character budget Stage 4 may
    #: condense this block to, measured (not estimated) at the floor style.
    char_budget: int | None = None


@dataclass(frozen=True, slots=True)
class StyleGroupSolution:
    """The 3A result: one style for the whole group, per-block line facts."""

    size_pt: float
    tracking_em: float
    leading_ratio: float
    leading_pt: float
    fits: bool
    blocks: tuple[StyleFit, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class VerticalBlockInput:
    """One block entering the 3B vertical solve.

    ``required_height_pt`` is 3A's *measured* height at the solved style; the
    solver never derives it. ``source_top_pt`` is the source block's top edge
    in PDF user space (y up), and inputs are ordered top-down as the PRD's
    ``B_1..B_n``.
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
# Stage 3A
# ---------------------------------------------------------------------------


def _floor_for(role: Literal["body", "caption", "table_cell"]) -> float:
    return BODY_FLOOR_PT if role == "body" else NON_BODY_FLOOR_PT


def _leading_candidates(source_ratio: float) -> tuple[float, ...]:
    """Source leading first, then tighter, down to the floor. Never above 1.55.

    The floor itself is always a candidate even when the 0.05 step lands past
    it (1.30, 1.25, … 1.10, 1.08): a style that only fits at the exact floor
    must be reachable, or the search would walk the font size past a leading
    solution that exists.
    """
    start = min(max(source_ratio, LEADING_MIN_RATIO), LEADING_MAX_RATIO)
    out: list[float] = []
    ratio = start
    while ratio >= LEADING_MIN_RATIO - 1e-9:
        out.append(round(ratio, 4))
        ratio -= LEADING_STEP
    if out[-1] > LEADING_MIN_RATIO + 1e-9:
        out.append(LEADING_MIN_RATIO)
    return tuple(out)


def _tracking_candidates() -> tuple[float, ...]:
    """Zero first (no distortion), then tighter, down to −0.03em."""
    out: list[float] = []
    value = 0.0
    while value >= TRACKING_MIN_EM - 1e-9:
        out.append(round(value, 4))
        value -= TRACKING_STEP_EM
    return tuple(out)


def _measure_group(
    blocks: Sequence[StyleBlockInput],
    *,
    measure: BlockMeasureFn,
    size_pt: float,
    tracking_em: float,
    leading_pt: float,
    borrow_fraction: float,
) -> tuple[StyleFit, ...]:
    fits: list[StyleFit] = []
    for block in blocks:
        width = block.source_width_pt + borrow_fraction * block.borrowable_pt
        if not block.text.strip():
            fits.append(StyleFit(index=block.index, width_pt=width, line_count=0, height_pt=0.0))
            continue
        measured = measure(block.text, size_pt, tracking_em, leading_pt, width)
        fits.append(
            StyleFit(
                index=block.index,
                width_pt=width,
                line_count=measured.line_count,
                height_pt=measured.height_pt,
            )
        )
    return tuple(fits)


def _group_fits(fits: Sequence[StyleFit], *, available_height_pt: float, min_gap_pt: float) -> bool:
    total = sum(fit.height_pt for fit in fits)
    gaps = max(0, len(fits) - 1) * min_gap_pt
    return total + gaps <= available_height_pt + 1e-9


def solve_style_group(
    blocks: Sequence[StyleBlockInput],
    *,
    available_height_pt: float,
    measure: BlockMeasureFn,
    min_gap_pt: float = DEFAULT_MIN_GAP_PT,
    size_step_pt: float = SIZE_STEP_PT,
) -> StyleGroupSolution:
    """Solve one style group's shared style (PRD §8.2.1 phase 3A).

    Walks the knobs in the PRD's preference order -- tracking, leading,
    right-borrow, and size last -- accepting the first candidate whose
    *measured* heights (never estimates) fit ``available_height_pt``. Size
    descends from the group's source size toward the role floor in
    ``size_step_pt`` steps; the floor is ``max(0.80 x source, 7.5pt)`` for body
    text and 5.5pt for captions/table cells.

    When even the floor style does not fit, the returned solution carries
    ``fits=False`` and a measured :attr:`StyleFit.char_budget` per block --
    Stage 4's condensation budget -- instead of pretending a style works.
    """
    if not blocks:
        raise ValueError("solve_style_group needs at least one block")
    role = blocks[0].role
    if any(block.role != role for block in blocks):
        raise ValueError("a style group must share one role; split by role first")

    base_size = max(block.source_size_pt for block in blocks)
    densest = max(blocks, key=lambda block: block.source_size_pt)
    source_ratio = densest.source_leading_pt / densest.source_size_pt
    floor = _floor_for(role)
    size_floor = max(floor, 0.80 * base_size)
    size_ceiling = MAX_EXPANSION_RATIO * base_size
    sizes: list[float] = []
    size = min(base_size, size_ceiling)
    while size >= size_floor - 1e-9:
        sizes.append(round(size, 4))
        size -= size_step_pt
    if not sizes:
        # A source below the role floor (a 7pt body block): the floor is the
        # only candidate, and the artifact QA's type tier will judge it.
        sizes.append(round(size_floor, 4))

    taus = _tracking_candidates()
    lambdas = _leading_candidates(source_ratio)

    def accept(
        size_pt: float,
        tracking_em: float,
        leading_ratio: float,
        borrow_fraction: float,
        fits: tuple[StyleFit, ...],
        reason: str,
    ) -> StyleGroupSolution:
        return StyleGroupSolution(
            size_pt=size_pt,
            tracking_em=tracking_em,
            leading_ratio=leading_ratio,
            leading_pt=round(leading_ratio * size_pt, 4),
            fits=True,
            blocks=fits,
            reason=reason,
        )

    last_candidate: tuple[float, float, float, float, tuple[StyleFit, ...]] | None = None
    for size_pt in sizes:
        for tracking_em in taus:
            for leading_ratio in lambdas:
                leading_pt = leading_ratio * size_pt
                for borrow_fraction in BORROW_FRACTIONS:
                    fits = _measure_group(
                        blocks,
                        measure=measure,
                        size_pt=size_pt,
                        tracking_em=tracking_em,
                        leading_pt=leading_pt,
                        borrow_fraction=borrow_fraction,
                    )
                    if any(
                        fit.line_count < 1
                        for fit, block in zip(fits, blocks, strict=True)
                        if block.text.strip()
                    ):
                        continue
                    last_candidate = (size_pt, tracking_em, leading_ratio, borrow_fraction, fits)
                    if _group_fits(
                        fits, available_height_pt=available_height_pt, min_gap_pt=min_gap_pt
                    ):
                        reduced = size_pt < base_size - 1e-9
                        adjusted = (
                            reduced
                            or tracking_em < 0
                            or leading_ratio < source_ratio - 1e-9
                            or borrow_fraction > 0
                        )
                        return accept(
                            size_pt,
                            tracking_em,
                            leading_ratio,
                            borrow_fraction,
                            fits,
                            ADJUSTED_TO_FIT if adjusted else SOURCE_STYLE_FITS,
                        )

    # Even the floor style does not fit: report the floor candidate and hand
    # Stage 4 a measured char budget per block. Heights are shared out
    # proportionally to source heights (the gaps come off the total first).
    floor_size = sizes[-1]
    floor_tau = taus[-1]
    floor_lam = lambdas[-1]
    fits = _measure_group(
        blocks,
        measure=measure,
        size_pt=floor_size,
        tracking_em=floor_tau,
        leading_pt=floor_lam * floor_size,
        borrow_fraction=1.0,
    )
    total_source = sum(block.source_height_pt for block in blocks) or 1.0
    shareable = max(0.0, available_height_pt - max(0, len(blocks) - 1) * min_gap_pt)
    budgeted: list[StyleFit] = []
    for block, fit in zip(blocks, fits, strict=True):
        share = shareable * (block.source_height_pt / total_source)
        budget = condensation_char_budget(
            block.text,
            measure,
            size_pt=floor_size,
            tracking_em=floor_tau,
            leading_pt=floor_lam * floor_size,
            width_pt=fit.width_pt,
            available_height_pt=share,
        )
        budgeted.append(
            StyleFit(
                index=block.index,
                width_pt=fit.width_pt,
                line_count=fit.line_count,
                height_pt=fit.height_pt,
                char_budget=budget,
            )
        )
    return StyleGroupSolution(
        size_pt=floor_size,
        tracking_em=floor_tau,
        leading_ratio=floor_lam,
        leading_pt=round(floor_lam * floor_size, 4),
        fits=False,
        blocks=tuple(budgeted),
        reason=BLOCKED_BY_GEOMETRY if last_candidate is None else FLOOR_EXHAUSTED,
    )


def condensation_char_budget(
    text: str,
    measure: BlockMeasureFn,
    *,
    size_pt: float,
    tracking_em: float,
    leading_pt: float,
    width_pt: float,
    available_height_pt: float,
) -> int:
    """The largest prefix of ``text`` whose *measured* height fits.

    Stage 4 hands an LLM a character budget, and the only honest way to size
    one is to measure: binary-search the prefix length against the real
    backend. A prefix cut can split a word or an atom; the budget is guidance
    for the rewrite, not the rewrite itself -- the delivered text is re-measured
    by the artifact QA either way.
    """
    if not text.strip() or available_height_pt <= 0:
        return 0

    def fits_prefix(n: int) -> bool:
        measured = measure(text[:n], size_pt, tracking_em, leading_pt, width_pt)
        return measured.height_pt <= available_height_pt + 1e-9

    if fits_prefix(len(text)):
        return len(text)
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if fits_prefix(mid):
            low = mid
        else:
            high = mid - 1
    return low


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
    """Distribute blocks vertically inside one track (PRD §8.2.1 phase 3B).

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
