"""BreakageSolver: flow one element's text across a chain of physical boxes.

A semantic element (a paragraph crossing a page boundary, a caption continuing in
a second column) can own several physical boxes. The element is translated as a
whole -- one call, so long-range context and pronouns stay coherent -- and this
solver decides how that one target stream breaks across the boxes: it walks the
boxes in reading order, binary-searches the largest prefix of the remaining text
that fits each box (measuring with the real typesetter), and lets the tail flow
into the next box. Breaking is *punctuation-preferring*: candidate cuts are the
positions after sentence punctuation (and before opening brackets), so a break
lands between clauses rather than mid-word.

The measurement is injected (``measure(text, width_pt) -> height_pt``), so the
solver is pure and testable without a typesetter; the compositor wires it to the
Typst micro-core. The last box receives the remainder whatever its capacity --
overflow there is the caller's to handle, not something to silently drop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ubt.model.span import PhysicalBox

#: ``measure(text, width_pt) -> height_pt``; larger than any box means "does not fit".
Measure = Callable[[str, float], float]

#: A break is preferred *after* one of these (sentence/clause enders, closers).
#: The colons are here because a subtitle break belongs *after* the colon -- the
#: source title's own break -- and a CJK target that broke before it would lead
#: the second line with "：".
_BREAK_AFTER = "。！？；：…!?;:.、，,）)]】》」』\"'”’»"
#: ...and *before* one of these (openers), so the bracket travels to the next box.
_BREAK_BEFORE = "（([{【《「『“‘«"


@dataclass(frozen=True, slots=True)
class FlowPlacement:
    """The slice of the element's target text placed into one physical box."""

    box: PhysicalBox
    text: str


def _candidate_cuts(text: str) -> list[int]:
    """Character offsets a break may land on, in ascending order.

    Sentence punctuation and word spaces; positions 0 and ``len(text)`` are not
    cuts (they are "all" or "nothing").
    """
    cuts: set[int] = set()
    for index, char in enumerate(text):
        if char in _BREAK_AFTER:
            cuts.add(index + 1)
        elif char in _BREAK_BEFORE or char.isspace():
            cuts.add(index)
    return sorted(cut for cut in cuts if 0 < cut < len(text))


def _fits(text: str, box: PhysicalBox, measure: Measure) -> bool:
    return bool(text) and measure(text, box.available_width) <= box.available_height


def _largest_fitting_cut(text: str, box: PhysicalBox, measure: Measure) -> int:
    """The offset of the longest break candidate whose prefix fits ``box`` (0 if none)."""
    if _fits(text, box, measure):
        return len(text)
    cuts = _candidate_cuts(text)
    if not cuts:
        return 0
    # ``fits(prefix)`` is monotone in the cut position for a fixed box, so a
    # binary search over the candidate list finds the largest prefix that fits.
    low, high = 0, len(cuts)
    while low < high:
        mid = (low + high + 1) // 2
        if _fits(text[: cuts[mid - 1]].rstrip(), box, measure):
            low = mid
        else:
            high = mid - 1
    return cuts[low - 1] if low > 0 else 0


def solve_flow(
    text: str, boxes: Sequence[PhysicalBox], measure: Measure
) -> tuple[FlowPlacement, ...]:
    """Break ``text`` across ``boxes`` in reading order, punctuation-preferring.

    Every box gets a placement (possibly empty when nothing fits); the final box
    receives the remaining text in full.
    """
    if not boxes:
        return ()
    remaining = text.strip()
    placements: list[FlowPlacement] = []
    for box in boxes[:-1]:
        cut = _largest_fitting_cut(remaining, box, measure)
        if cut <= 0:
            placements.append(FlowPlacement(box, ""))
            continue
        placements.append(FlowPlacement(box, remaining[:cut].rstrip()))
        remaining = remaining[cut:].lstrip()
    placements.append(FlowPlacement(boxes[-1], remaining))
    return tuple(placements)


__all__ = ["FlowPlacement", "Measure", "solve_flow"]
