"""Document-level style groups: the G4 Stage 3A decision.

`ubt/render/layout_solver.solve_page_2d` searches one uniform body size **per
page**, in a band derived from that page's own measured source size. Two
directions of drift come out of that, and both are visible in the 26-page
BSIM-CMG ledger: roomy pages climb to the ceiling (9.79pt on p1/p5/p17) while
tight pages are pushed to the floor (7.5pt on p3/p15), giving a 2.29pt p5-p95
spread against a 0.5pt contract. The source page size itself is measured with
tenths of a point of jitter, so the group is already lost before the solver
runs.

This module makes the decision the other way round: **one** size for the whole
document body group, chosen so that every member page can still hold its text.
The group size is a short-plank minimum, not an average — an average would
shrink the tight pages and leave the roomy ones inflated, which is the
"ransom-note" typography the PRD forbids (P10 / §8.2).

Order of preference follows PRD §7.4.3, with the levers that actually exist on
the current rendering path:

    tracking (Tc)  ->  leading pitch (TL)  ->  right-side borrow (Δw)  ->  size

``γ_x`` (``Tz`` horizontal scaling) is deliberately absent: G2 measured that
Typst does not change advance width for a static face via ``text(stretch:)``,
so the only real implementation is the CELS content-stream path (Stage 5). A
ladder rung that cannot be applied is not listed here — a rung that silently
did nothing would let the planner believe it had bought space it never bought.

Two integrity rules are load-bearing:

* **Membership comes from the plan, the number comes from the artifact.** The
  contract reads body sizes out of the delivered PDF (the truth for *how big*),
  but which pages belong to a group and which are excused comes from this plan,
  written before rendering. Deriving group membership from delivered sizes
  would let every drift excuse itself.
* **An exception is declared, never inferred.** A page that cannot hold the
  group size at the floor is recorded here with its deficit and handed to
  Stage 4 / the L2 escape hatch. The plan is not allowed to quietly pick a
  different size for it.

The module is pure: it takes measured page metrics and a feasibility callback,
and returns the decision. It imports no renderer, which keeps it testable as
data and stops it from growing a second layout engine.

Boundary with :mod:`ubt.core.layout.solver` (CELS Stage 3A)
-----------------------------------------------------------
``solver.solve_style_group`` is the *target* engine: it searches tracking,
leading, borrow and size against real measured heights for a group of blocks on
one track, and feeds Stage 3B's Cassowary solve. This module answers the
narrower question the legacy facsimile channel can act on today — one size for
the document body group, decided from cheap geometry before a single page is
drawn, plus the exception policy and the record the delivery contract reads.

They share the same PRD limits (7.5pt body floor, 0.80/1.10 band, 0.25pt step,
tracking −0.03em, leading 1.08) and the constants are mirrored here rather than
imported, because the facsimile path must not start failing when the CELS engine
is refactored. When the CELS renderer replaces the facsimile one, the plan this
module produces becomes the *input* to ``solve_style_group``: same group, same
size, measured instead of estimated.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: The only role that participates in the body drift check today. Headings,
#: captions and table cells have their own legitimate sizes and are excluded on
#: purpose: including them would make every book fail its style check.
BODY_ROLE = "body"

#: Kept in lockstep with ``delivery_contract._STYLE_DRIFT_PT``. Duplicating the
#: constant is intentional — the planner must not be able to widen the contract's
#: tolerance by editing one number here.
DRIFT_TOLERANCE_PT = 0.5

#: PRD §7.4.3 body limits: an absolute floor plus a relative one, whichever is
#: higher. The relative floor is what stops a book measured at 9.8pt from
#: shipping a 7.5pt body just because one page was full.
BODY_FLOOR_PT = 7.5
RELATIVE_FLOOR = 0.80

#: PRD §8.2: ``faithful`` may grow the body up to 1.10x the source. The ceiling
#: bounds the *search*, it is never a target — starting there is exactly the
#: inflation half of the measured drift.
CEILING_RATIO = 1.10
MAX_BODY_PT = 12.0

#: Discrete size step. The PRD asks for 0.25pt; stepping rather than bisecting
#: keeps the probe count bounded and the result reproducible, and the feasibility
#: of ``H_req = N_i * s * lambda`` is monotone in ``s`` only while ``N_i`` is
#: fixed, which a step ladder sidesteps entirely.
SIZE_STEP_PT = 0.25

#: Exception budget. At or below these shares an exception is a documented page;
#: above them the group is not a group, and the contract escalates accordingly.
EXCEPTION_PAGE_SHARE = 0.10
EXCEPTION_BLOCK_SHARE = 0.05

#: Lowest line-height ratio the ladder is allowed to reach (PRD §7.4.3).
PITCH_FLOOR = 1.08

#: Widest tracking compression the ladder is allowed to reach (PRD §7.4.3).
TRACKING_FLOOR_EM = -0.030

#: Most Δw the ladder may borrow from the same column's right whitespace. It is
#: a *same-column* borrow: Stage 1's track boundary, not the page margin, is the
#: real limit, and the width probe that supplies it is responsible for the
#: trailing-ink pad (AD-020).
WIDTH_BORROW_MAX_PT = 4.0


@dataclass(frozen=True, slots=True)
class MicroTypography:
    """One rung of the ladder, expressed as what the renderer must apply.

    ``pitch_drop`` is relative to the document's own measured pitch so the same
    ladder serves a 1.45 book and a 1.25 book without a second table. ``None``
    means "leave the document pitch alone".
    """

    label: str
    tracking_em: float = 0.0
    pitch_drop: float = 0.0
    width_borrow_pt: float = 0.0

    def apply(self, source_pitch: float) -> float:
        """The line-height ratio this rung asks for, given the document's pitch."""
        return max(PITCH_FLOOR, source_pitch - self.pitch_drop)


#: The ladder, in preference order. Each rung is a *prefix* of the previous ones:
#: the planner only advances a rung when the whole group still fails at the
#: previous one, so no page is compressed more than it had to be.
LADDER: tuple[MicroTypography, ...] = (
    MicroTypography("plain"),
    MicroTypography("tracking_-0.015", tracking_em=-0.015),
    MicroTypography("tracking_-0.030", tracking_em=TRACKING_FLOOR_EM),
    MicroTypography("leading_-0.10", tracking_em=TRACKING_FLOOR_EM, pitch_drop=0.10),
    MicroTypography("leading_-0.26", tracking_em=TRACKING_FLOOR_EM, pitch_drop=0.26),
    MicroTypography(
        "borrow_4pt",
        tracking_em=TRACKING_FLOOR_EM,
        pitch_drop=0.26,
        width_borrow_pt=WIDTH_BORROW_MAX_PT,
    ),
)


@dataclass(frozen=True, slots=True)
class PageFit:
    """One page's feasibility verdict at a candidate size and rung."""

    page: int
    fits: bool
    deficit_pt: float = 0.0
    block_ids: tuple[str, ...] = ()


#: ``probe(page, size_pt, rung, pitch) -> PageFit``. The caller owns the
#: measurement; this module owns only the decision — including the resolved
#: line-height ratio, so the probe never has to re-derive the document pitch and
#: cannot disagree with the plan about it.
FeasibilityProbe = Callable[[int, float, "MicroTypography", float], PageFit]


@dataclass(frozen=True, slots=True)
class StyleGroup:
    """The document's body group: one size and one set of micro-typographic knobs."""

    group_id: str
    role: str
    size_pt: float
    pitch: float
    tracking_em: float
    width_borrow_pt: float
    source_size_pt: float
    source_pitch: float
    pages: tuple[int, ...]
    floor_pt: float
    ceiling_pt: float
    rung: str

    @property
    def leading_em(self) -> float:
        """Extra leading over the font's own line box, in em (solver contract)."""
        return round(max(0.0, self.pitch - 1.0), 4)

    def to_payload(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "role": self.role,
            "size_pt": round(self.size_pt, 3),
            "pitch": round(self.pitch, 4),
            "leading_em": self.leading_em,
            "tracking_em": round(self.tracking_em, 4),
            "width_borrow_pt": round(self.width_borrow_pt, 3),
            "source_size_pt": round(self.source_size_pt, 3),
            "source_pitch": round(self.source_pitch, 4),
            "pages": list(self.pages),
            "floor_pt": round(self.floor_pt, 3),
            "ceiling_pt": round(self.ceiling_pt, 3),
            "rung": self.rung,
        }


@dataclass(frozen=True, slots=True)
class StyleGroupException:
    """A page the group size could not hold, with the deficit that proves it."""

    page: int
    block_ids: tuple[str, ...]
    deficit_pt: float
    rung: str
    #: ``does_not_fit`` (L1 overflow), ``l2_shrink`` (the page-locked reflow had
    #: to go below the group size), or ``unmodeled`` (no text to fit at all).
    reason: str = "does_not_fit"

    def to_payload(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "block_ids": list(self.block_ids),
            "deficit_pt": round(self.deficit_pt, 3),
            "rung": self.rung,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class StyleGroupPlan:
    """The Stage 3A decision plus the evidence of how it was reached."""

    group: StyleGroup
    exceptions: tuple[StyleGroupException, ...] = ()
    rounds: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    page_source_sizes: dict[int, float] = field(default_factory=dict)
    #: Pages whose fixed boxes overlap in the source: their fit is decided by the
    #: page-locked reflow at the same group size, so they are neither judged here
    #: nor allowed to influence the size.
    deferred_pages: tuple[int, ...] = ()
    policy: str = "source_first"

    @property
    def exception_pages(self) -> tuple[int, ...]:
        return tuple(sorted(exception.page for exception in self.exceptions))

    def replacing_exceptions(self, exceptions: Sequence[StyleGroupException]) -> StyleGroupPlan:
        """Swap the *predicted* exceptions for the ones the renderer actually hit.

        The planner works from a fast approximation of each page's text records;
        the renderer works from the real ones (multi-region flow, obstacles,
        heading boxes). When those disagree the renderer wins, because it is the
        side that placed the glyphs — but the decision itself (size, rung) stands,
        and the difference stays visible in ``rounds``.
        """
        return StyleGroupPlan(
            group=self.group,
            exceptions=tuple(sorted(exceptions, key=lambda exception: exception.page)),
            rounds=self.rounds,
            page_source_sizes=dict(self.page_source_sizes),
            deferred_pages=self.deferred_pages,
            policy=self.policy,
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize for ``manifest.metadata["document_style_group"]``."""
        return {
            "group": self.group.to_payload(),
            "exceptions": [exception.to_payload() for exception in self.exceptions],
            "exception_pages": list(self.exception_pages),
            "deferred_pages": list(self.deferred_pages),
            "policy": self.policy,
            "rounds": [dict(entry) for entry in self.rounds],
            "page_source_sizes": {
                str(page): round(size, 3) for page, size in sorted(self.page_source_sizes.items())
            },
        }


def _weighted_mode(values: Sequence[tuple[float, int]], quantum: float) -> float | None:
    """The most common value, weighted by how much text sits behind each sample.

    A page with two body lines (a part title, a mostly-figure page) must not
    outvote a page with forty. Rounding to a ``quantum`` first keeps 8.86 and
    8.94 from splitting one mode into two.
    """
    if not values:
        return None
    counts: Counter[float] = Counter()
    for value, weight in values:
        counts[round(value / quantum) * quantum] += max(1, int(weight))
    try:
        mode, _ = counts.most_common(1)[0]
    except IndexError:  # pragma: no cover - guarded by the empty check above
        return None
    return float(mode)


def derive_document_body_metrics(
    page_metrics: Mapping[int, tuple[float, float, int]],
) -> tuple[float, float]:
    """The document's body size and pitch, from per-page ``(size, pitch, weight)``.

    Uses a text-weighted mode rather than a median: the 26-page corpus measures
    8.9 on most pages but 7.9/8.0 on a handful, and a median over *pages* would
    happily settle between the two and belong to neither. Falls back to the
    median of the available samples when no mode is well defined.
    """
    sizes = [(size, weight) for size, _pitch, weight in page_metrics.values() if size > 0]
    pitches = [
        (pitch, weight)
        for _size, pitch, weight in page_metrics.values()
        if pitch > 0 and pitch <= 3.0
    ]
    mode_size = _weighted_mode(sizes, 0.1)
    if mode_size is None:
        mode_size = statistics.median([size for size, _ in sizes]) if sizes else 9.0
    mode_pitch = _weighted_mode(pitches, 0.05)
    if mode_pitch is None:
        mode_pitch = statistics.median([pitch for pitch, _ in pitches]) if pitches else 1.348
    return round(float(mode_size), 2), round(float(mode_pitch), 3)


def body_band(source_size_pt: float) -> tuple[float, float]:
    """The ``(floor, ceiling)`` a body group may occupy for this document."""
    floor = max(BODY_FLOOR_PT, round(RELATIVE_FLOOR * source_size_pt, 3))
    ceiling = min(MAX_BODY_PT, round(CEILING_RATIO * source_size_pt, 3))
    if ceiling < floor:  # a mis-measured source: the floor still wins (see solver)
        ceiling = floor
    return floor, ceiling


def plan_body_group(
    page_metrics: Mapping[int, tuple[float, float, int]],
    probe: FeasibilityProbe,
    *,
    group_id: str = "body:document",
    ladder: Sequence[MicroTypography] = LADDER,
    size_step_pt: float = SIZE_STEP_PT,
    block_count: int = 0,
    deferred_pages: Sequence[int] = (),
) -> StyleGroupPlan:
    """Choose the document body size, then declare whatever it cannot hold.

    ``page_metrics`` maps page -> ``(source_size_pt, source_pitch, text_weight)``.
    ``probe`` answers feasibility at a candidate size and rung; every judged page
    is asked at every candidate, because the group is exactly the promise that
    *all* of them were considered.

    Policy (confirmed with the product owner, in this order):

    1. **Source size first.** The search starts at the document's own measured
       body size, not at the ceiling: PRD P10 puts micro-typography before size,
       and §8.2 clones the source. The ladder is walked only when a page still
       does not fit.
    2. **Least compression that helps.** At any size the rung with the *smallest*
       failing set wins, and among rungs tied on that set the mildest one wins.
       Compressing the whole book to rescue pages it does not rescue is a
       legibility cost with nothing bought.
    3. **Exceptions are budgeted, never hidden.** If the failing set at the
       source size exceeds the exception budget the whole group steps down by
       ``size_step_pt`` and tries again; it stops at the first size inside the
       budget, or at the floor. Whatever remains failing there is declared, and
       the contract turns it into a blocking finding.
    4. **Deferred pages do not vote.** A page whose boxes overlap in the source
       cannot be judged by a fixed-box probe (it will be re-typeset by the
       page-locked reflow); its fit is decided later, at the same group size, so
       it must not drag the size down for pages that can actually be judged.
    """
    source_size, source_pitch = derive_document_body_metrics(page_metrics)
    floor, ceiling = body_band(source_size)
    start = round(min(ceiling, max(floor, source_size)), 3)
    pages = tuple(sorted(page_metrics))
    deferred = tuple(sorted({page for page in deferred_pages if page in page_metrics}))
    judged = tuple(page for page in pages if page not in deferred)
    rounds: list[dict[str, Any]] = []

    def _ask(size: float, rung: MicroTypography) -> list[PageFit]:
        pitch = rung.apply(source_pitch)
        results = [
            PageFit(page=page, fits=True) if page in deferred else probe(page, size, rung, pitch)
            for page in pages
        ]
        rounds.append(
            {
                "size_pt": round(size, 3),
                "rung": rung.label,
                "fits": all(result.fits for result in results),
                "failing_pages": [result.page for result in results if not result.fits],
                "worst_deficit_pt": round(
                    max((result.deficit_pt for result in results), default=0.0), 3
                ),
            }
        )
        return results

    def _group(size: float, rung: MicroTypography) -> StyleGroup:
        return StyleGroup(
            group_id=group_id,
            role=BODY_ROLE,
            size_pt=round(size, 3),
            pitch=round(rung.apply(source_pitch), 4),
            tracking_em=rung.tracking_em,
            width_borrow_pt=rung.width_borrow_pt,
            source_size_pt=source_size,
            source_pitch=source_pitch,
            pages=pages,
            floor_pt=floor,
            ceiling_pt=ceiling,
            rung=rung.label,
        )

    def _within_budget(results: Sequence[PageFit]) -> bool:
        failing = [result for result in results if not result.fits]
        if not failing:
            return True
        page_share = round(len(failing) / max(1, len(judged)), 4)
        if page_share > EXCEPTION_PAGE_SHARE:
            return False
        if block_count > 0:
            block_share = round(
                sum(len(result.block_ids) for result in failing) / max(1, block_count), 4
            )
            if block_share > EXCEPTION_BLOCK_SHARE:
                return False
        return True

    def _mildest(
        results_by_rung: Sequence[tuple[MicroTypography, list[PageFit]]],
    ) -> tuple[MicroTypography, list[PageFit]]:
        fewest = min(
            sum(1 for result in results if not result.fits) for _rung, results in results_by_rung
        )
        for rung, results in results_by_rung:
            if sum(1 for result in results if not result.fits) == fewest:
                return rung, results
        return results_by_rung[-1]  # pragma: no cover - the loop above always returns

    evaluations: list[tuple[MicroTypography, list[PageFit]]] = []
    for rung in ladder:
        results = _ask(start, rung)
        evaluations.append((rung, results))
        if all(result.fits for result in results):
            break

    size = start
    rung, results = _mildest(evaluations)
    if not all(result.fits for result in results):
        while not _within_budget(results) and size > floor + 1e-9:
            size = round(max(floor, size - size_step_pt), 3)
            evaluations = [(candidate, _ask(size, candidate)) for candidate in ladder]
            rung, results = _mildest(evaluations)
            if all(result.fits for result in results):
                break

    exceptions = tuple(
        StyleGroupException(
            page=result.page,
            block_ids=result.block_ids,
            deficit_pt=round(result.deficit_pt, 3),
            rung=rung.label,
            reason="deferred_fit_unresolved" if result.page in deferred else "does_not_fit",
        )
        for result in sorted(results, key=lambda item: item.page)
        if not result.fits
    )
    return StyleGroupPlan(
        group=_group(size, rung),
        exceptions=exceptions,
        rounds=tuple(rounds),
        page_source_sizes={page: page_metrics[page][0] for page in pages},
        deferred_pages=deferred,
        policy="source_first",
    )


def exception_shares(plan: StyleGroupPlan, *, block_count: int) -> tuple[float, float]:
    """``(page share, block share)`` of the declared exceptions.

    ``block_count`` of 0 means "the caller could not count the document's
    blocks", and the block share is then reported as 0.0 so an unknown count
    cannot on its own trip the flood gate.
    """
    pages = len(plan.group.pages) or 1
    exception_blocks = sum(len(exception.block_ids) for exception in plan.exceptions)
    if block_count <= 0:
        return round(len(plan.exceptions) / pages, 4), 0.0
    return (
        round(len(plan.exceptions) / pages, 4),
        round(exception_blocks / block_count, 4),
    )


def plan_from_metadata(payload: Mapping[str, Any] | None) -> StyleGroupPlan | None:
    """Rebuild a plan from ``manifest.metadata`` (the contract's side)."""
    if not isinstance(payload, Mapping):
        return None
    group_payload = payload.get("group")
    if not isinstance(group_payload, Mapping):
        return None
    try:
        group = StyleGroup(
            group_id=str(group_payload.get("group_id") or "body:document"),
            role=str(group_payload.get("role") or BODY_ROLE),
            size_pt=float(group_payload.get("size_pt") or 0.0),
            pitch=float(group_payload.get("pitch") or 1.348),
            tracking_em=float(group_payload.get("tracking_em") or 0.0),
            width_borrow_pt=float(group_payload.get("width_borrow_pt") or 0.0),
            source_size_pt=float(group_payload.get("source_size_pt") or 0.0),
            source_pitch=float(group_payload.get("source_pitch") or 1.348),
            pages=tuple(int(page) for page in group_payload.get("pages") or ()),
            floor_pt=float(group_payload.get("floor_pt") or BODY_FLOOR_PT),
            ceiling_pt=float(group_payload.get("ceiling_pt") or MAX_BODY_PT),
            rung=str(group_payload.get("rung") or "plain"),
        )
    except (TypeError, ValueError):
        return None
    exceptions: list[StyleGroupException] = []
    for entry in payload.get("exceptions") or ():
        if not isinstance(entry, Mapping):
            continue
        try:
            exceptions.append(
                StyleGroupException(
                    page=int(entry.get("page") or 0),
                    block_ids=tuple(str(item) for item in entry.get("block_ids") or ()),
                    deficit_pt=float(entry.get("deficit_pt") or 0.0),
                    rung=str(entry.get("rung") or group.rung),
                    reason=str(entry.get("reason") or "does_not_fit"),
                )
            )
        except (TypeError, ValueError):
            continue
    page_sizes: dict[int, float] = {}
    for key, value in (payload.get("page_source_sizes") or {}).items():
        try:
            page_sizes[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return StyleGroupPlan(
        group=group,
        exceptions=tuple(sorted(exceptions, key=lambda exception: exception.page)),
        rounds=(),
        page_source_sizes=page_sizes,
        deferred_pages=tuple(int(page) for page in payload.get("deferred_pages") or ()),
        policy=str(payload.get("policy") or "source_first"),
    )


__all__ = [
    "BODY_FLOOR_PT",
    "BODY_ROLE",
    "CEILING_RATIO",
    "DRIFT_TOLERANCE_PT",
    "EXCEPTION_BLOCK_SHARE",
    "EXCEPTION_PAGE_SHARE",
    "LADDER",
    "MAX_BODY_PT",
    "PITCH_FLOOR",
    "RELATIVE_FLOOR",
    "SIZE_STEP_PT",
    "FeasibilityProbe",
    "MicroTypography",
    "PageFit",
    "StyleGroup",
    "StyleGroupException",
    "StyleGroupPlan",
    "body_band",
    "derive_document_body_metrics",
    "exception_shares",
    "plan_body_group",
    "plan_from_metadata",
]
