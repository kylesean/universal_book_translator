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

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ubt.core.cjk_ranges import is_cjk_char
from ubt.model.span import PhysicalBox

#: ``measure(text, width_pt) -> height_pt``; larger than any box means "does not fit".
Measure = Callable[[str, float], float]

#: Inline math (``$...$`` / ``$$...$$``) and pipeline bracket markers (``⟦...⟧``)
#: are atomic tokens: splitting inside one leaves both boxes with unbalanced
#: delimiters and corrupts the formula or emphasis span at render time.
_ATOMIC_SPAN_RE = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+?\$|⟦[^⟧\n]+⟧")

#: A break is preferred *after* one of these (sentence/clause enders, closers).
#: The colons are here because a subtitle break belongs *after* the colon -- the
#: source title's own break -- and a CJK target that broke before it would lead
#: the second line with "：".
_BREAK_AFTER = "。！？；：…!?;:.、，,）)]】》」』\"'”’»"
#: ...and *before* one of these (openers), so the bracket travels to the next box.
_BREAK_BEFORE = "（([{【《「『“‘«"

#: Characters that may never *start* a line (避头尾: "no leading" — closing
#: punctuation, sentence enders, trailing marks). A character-boundary cut must
#: not land immediately before one of these.
_NO_LINE_START = "。！？；：、，,．.）)]】》」』\"'”’»…—～!?;:"
#: Characters that may never *end* a line (opening brackets/quotes): a cut must
#: not land immediately after one of these.
_NO_LINE_END = "（([{【《「『“‘«"


@dataclass(frozen=True, slots=True)
class FlowPlacement:
    """The slice of the element's target text placed into one physical box."""

    box: PhysicalBox
    text: str


def _filter_atomic_cuts(text: str, cuts: set[int]) -> list[int]:
    """Filter out cut offsets that fall strictly inside an atomic math or marker span."""
    spans = [(m.start(), m.end()) for m in _ATOMIC_SPAN_RE.finditer(text)]
    return sorted(
        cut
        for cut in cuts
        if 0 < cut < len(text) and not any(start < cut < end for start, end in spans)
    )


def _candidate_cuts(text: str) -> list[int]:
    """Character offsets a break may land on, in ascending order.

    Sentence punctuation and word spaces; positions 0 and ``len(text)`` are not
    cuts (they are "all" or "nothing"), and offsets inside ``$...$`` or ``⟦...⟧``
    spans are excluded so an inline formula is never split across two boxes.
    """
    cuts: set[int] = set()
    for index, char in enumerate(text):
        if char in _BREAK_AFTER:
            cuts.add(index + 1)
        elif char in _BREAK_BEFORE or char.isspace():
            cuts.add(index)
    return _filter_atomic_cuts(text, cuts)


def _cjk_candidate_cuts(text: str) -> list[int]:
    """Every 避头尾-legal CJK inter-character cut, for text with no punctuation cuts.

    A CJK run carries no spaces, so a long unpunctuated sentence ("... 的 ... 的
    ...") has *no* punctuation cut at all and the punctuation-preferring search
    finds nothing to fit — the first box is left empty and the whole paragraph
    overflows the last box and descends to source. CJK may break between any two
    characters, so this returns every offset subject to the kinsoku rules: a cut
    may not fall immediately before a no-line-start character or immediately
    after a no-line-end one.

    A cut is offered only where at least one adjacent character is CJK, so a
    Latin run inside the text is never broken mid-word (its spaces already gave
    ``_candidate_cuts`` the cuts it needs). Used only as a fallback, so
    punctuation still wins when it is available.
    """
    cuts: set[int] = set()
    for index in range(1, len(text)):
        before, after = text[index - 1], text[index]
        # Not before a no-start char, and not after a no-end char.
        if after in _NO_LINE_START or before in _NO_LINE_END:
            continue
        # Only between characters where a CJK char is involved; never split a
        # Latin word.
        if not (is_cjk_char(before) or is_cjk_char(after)):
            continue
        cuts.add(index)
    return _filter_atomic_cuts(text, cuts)


def _fits(text: str, box: PhysicalBox, measure: Measure) -> bool:
    return bool(text) and measure(text, box.available_width) <= box.available_height


def _largest_fitting_cut(text: str, box: PhysicalBox, measure: Measure) -> int:
    """The offset of the longest break candidate whose prefix fits ``box`` (0 if none).

    Punctuation cuts are preferred. When none of them fits even the first box —
    a long unpunctuated CJK run has no punctuation cut at all — the search
    falls back to 避头尾-legal inter-character cuts so the box still receives a
    full line instead of being left empty.
    """
    if _fits(text, box, measure):
        return len(text)
    for cuts in (_candidate_cuts(text), _cjk_candidate_cuts(text)):
        if not cuts:
            continue
        cut = _largest_fitting(text, cuts, box, measure)
        if cut > 0:
            return cut
    return 0


def _largest_fitting(text: str, cuts: list[int], box: PhysicalBox, measure: Measure) -> int:
    """Largest candidate cut whose prefix fits, via binary search (0 if none).

    ``fits(prefix)`` is monotone in the cut position for a fixed box, so a
    binary search over the candidate list finds the largest prefix that fits.
    """
    low, high = 0, len(cuts)
    while low < high:
        mid = (low + high + 1) // 2
        if _fits(text[: cuts[mid - 1]].rstrip(), box, measure):
            low = mid
        else:
            high = mid - 1
    return cuts[low - 1] if low > 0 else 0


def solve_flow(
    text: str,
    boxes: Sequence[PhysicalBox],
    measure: Measure,
    *,
    fallback_measure: Measure | None = None,
) -> tuple[FlowPlacement, ...]:
    """Break ``text`` across ``boxes`` in reading order, punctuation-preferring.

    Every box gets a placement (possibly empty when nothing fits); the final box
    receives the remaining text in full.

    ``fallback_measure`` is consulted for a box the primary measure cannot fit
    even one break into (``None`` leaves the box empty). Its capacity is the
    *draw* box's -- the ink box plus the compositor's line slack -- so a box that
    can print a line is not left blank: the source line it replaces is masked
    either way, and an empty box therefore loses that line from the page.
    """
    if not boxes:
        return ()
    remaining = text.strip()
    placements: list[FlowPlacement] = []
    for box in boxes[:-1]:
        cut = _largest_fitting_cut(remaining, box, measure)
        if cut <= 0 and fallback_measure is not None:
            cut = _largest_fitting_cut(remaining, box, fallback_measure)
        if cut <= 0:
            placements.append(FlowPlacement(box, ""))
            continue
        placements.append(FlowPlacement(box, remaining[:cut].rstrip()))
        remaining = remaining[cut:].lstrip()
    placements.append(FlowPlacement(boxes[-1], remaining))
    return tuple(placements)


def approximate_measure(
    font_size_pt: float,
    leading_pt: float,
    *,
    cjk_aspect: float = 1.0,
) -> Measure:
    """Return an approximate geometric typesetter measurement closure.

    Useful for fast pre-flight flow decisions or pure tests.
    """

    def _measure(text: str, width_pt: float) -> float:
        if not text or width_pt <= 0:
            return 0.0
        eff_glyph_w = font_size_pt * cjk_aspect
        per_line = max(1.0, width_pt / eff_glyph_w)
        lines = max(1, math.ceil(len(text) / per_line))
        return float(lines) * leading_pt

    return _measure


__all__ = ["FlowPlacement", "Measure", "approximate_measure", "solve_flow"]
