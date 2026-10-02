"""Text fitting: the width model and the font-size/capacity kernel.

The overlay emitter hands the paragraph to Typst, which owns line breaking.
This module keeps the two jobs Python still does before that: measure whether
the text can fit a zone's box chain at a candidate size (capacity), and bisect
to the largest size that fits. Pure box geometry over an injected width
measure, so it is testable without fonts, PDFs, or Typst.

- :class:`EmWidth` — conservative font-free fallback (CJK 1.0 / ASCII 0.55 /
  punctuation 0.62 / space 0.32) used when fontTools metrics are unavailable;
  shaped Typst output only ever shrinks below it *given* the overlay disables
  CJK-Latin auto-spacing (``cjk-latin-spacing: none``): with Typst's default
  spacing, mixed-script lines lay out ~0.25em wider per script boundary
  (GAA find: 175.3pt measured vs 193.9pt laid out). Production injects
  :func:`ubt.adapters.pdf.font_metrics.text_width_pt` instead.
- :class:`FlowFitter` — greedy capacity check + ``fit_chars`` (kinsoku-aware
  cut) over a box chain; ``flow_paragraph`` returns the break evidence, and
  the rigid typesetter bisects on it for the font size. The line strings
  are capacity evidence only: their breaks drop the spaces at break points,
  so the emitter passes the raw zone text and Typst re-breaks it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ubt.adapters.pdf.overlay_text import math_span_extents
from ubt.core.cjk_ranges import (
    CJK_EXT_A,
    CJK_PUNCTUATION,
    CJK_UNIFIED,
    FULLWIDTH_FORMS,
    GENERAL_PUNCTUATION,
)
from ubt.core.policy.layout_policy import (
    CJK_CLOSE_PUNCT,
    CJK_OPEN_PUNCT,
    CJK_PUNCT_CHARS,
    EM_ASCII_ADV,
    EM_CJK_ADV,
    EM_PUNCT_ADV,
    EM_SPACE_ADV,
    FIT_MIN_FONT_PT,
    FIT_PRECISION_PT,
    PUNCT_SQUEEZE_CAP,
    PUNCT_SQUEEZE_PER_PUNCT,
)

Rect = tuple[float, float, float, float]


class WidthMeasure(Protocol):
    """Advance-sum width of ``text`` at ``size_pt``."""

    def __call__(self, text: str, size_pt: float) -> float: ...


class EmWidth:
    """Font-free conservative width model (fallback, no fontTools needed).

    Production uses :func:`ubt.adapters.pdf.font_metrics.text_width_pt` (the
    width authority) injected through ``RigidTypesetter._fitter_obj``; this
    model covers tests and font-less environments only.
    """

    def __call__(self, text: str, size_pt: float) -> float:
        total = 0.0
        for ch in text:
            if ch == " ":
                total += EM_SPACE_ADV
            elif ch.isascii() and (ch.isalnum() or ch in "'\"-_/\\|@#$%^&*+=~`"):
                total += EM_ASCII_ADV
            elif ch in CJK_PUNCT_CHARS or (not ch.isascii() and not _is_cjk(ch)):
                total += EM_PUNCT_ADV
            else:
                total += EM_CJK_ADV
        return total * size_pt


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(
        lo <= o <= hi
        for lo, hi in (
            CJK_UNIFIED,
            CJK_EXT_A,
            CJK_PUNCTUATION,
            FULLWIDTH_FORMS,
            GENERAL_PUNCTUATION,
        )
    )


class FlowFitter:
    """Greedy-flow + bisection fitter over injected width metrics."""

    def __init__(
        self,
        measure: WidthMeasure,
        min_font_pt: float = FIT_MIN_FONT_PT,
        precision_pt: float = FIT_PRECISION_PT,
        punct_squeeze_cap: float = PUNCT_SQUEEZE_CAP,
        punct_squeeze_per_punct: float = PUNCT_SQUEEZE_PER_PUNCT,
    ) -> None:
        self.measure = measure
        self.min_font_pt = min_font_pt
        self.precision_pt = precision_pt
        self.punct_squeeze_cap = punct_squeeze_cap
        self.punct_squeeze_per_punct = punct_squeeze_per_punct

    def squeezed_width(self, text: str, size_pt: float) -> float:
        """Advance-sum minus the punctuation-squeeze discount.

        Typst compresses CJK punctuation, so the raw advance-sum overstates
        shaped width wherever punctuation is dense. The discount scales with
        punctuation density and caps out — pure-ASCII text measures exactly
        the raw sum (no slack claimed where none provably exists). The
        discount is density-dependent, so prefix widths are not strictly
        monotone — bisection stays safe (every returned cut rechecks) but
        may stop one char short of maximal inside punct clusters.
        """
        raw = self.measure(text, size_pt)
        if not text or self.punct_squeeze_cap <= 0:
            return raw
        n_punct = sum(1 for ch in text if ch in CJK_PUNCT_CHARS)
        if not n_punct:
            return raw
        return raw * (
            1.0 - min(self.punct_squeeze_cap, n_punct * self.punct_squeeze_per_punct / len(text))
        )

    def fit_chars(self, text: str, width_pt: float, size_pt: float) -> int:
        """Chars of text fitting width_pt (CJK anywhere, ASCII on spaces).

        Kinsoku: the cut never leaves an opener dangling at the line end
        (pushed down while it does), and a closer starting the remainder is
        pulled up while it still fits. Residual closer-starts are possible
        inside dense closer clusters — strictly better than raw cuts.
        """
        if self.squeezed_width(text, size_pt) <= width_pt:
            return len(text)
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.squeezed_width(text[:mid], size_pt) <= width_pt:
                lo = mid
            else:
                hi = mid - 1
        cut = lo
        while 0 < cut < len(text) and text[cut - 1] in CJK_OPEN_PUNCT:
            cut -= 1
        while 0 < cut < len(text) and text[cut] in CJK_CLOSE_PUNCT:
            if self.squeezed_width(text[: cut + 1], size_pt) <= width_pt:
                cut += 1
            else:
                break
        if 0 < cut < len(text):
            ahead, behind = text[cut : cut + 1], text[cut - 1 : cut]
            if ahead.isascii() and ahead.isalnum() and behind.isascii() and behind.isalnum():
                sp = text.rfind(" ", 0, cut)
                if sp > 0 and cut - sp < 20:
                    cut = sp + 1
        return cut

    def flow_paragraph(self, text: str, boxes: Sequence[Rect], size_pt: float) -> list[str] | None:
        """Flow text through the box chain; None when it cannot fit.

        Alignment invariant: the returned list ALWAYS has exactly
        ``len(boxes)`` entries — trailing boxes the text never reaches keep
        ``""`` so the caller still paints+strips them (condensed translations
        must not leave source English naked).

        Span-atomic flow: a ``$...$`` math span never straddles two lines.
        A cut inside a span is pulled back to the span's opening ``$`` so
        the whole span moves down one box; a span wider than the box fails
        closed through the existing ``n <= 0`` overflow path. Without this,
        straddling spans reach the overlay with unbalanced ``$`` per line
        and degrade to literal backslash text (chapter-1 ``$T_{\\text{si}}``
        class) — strictly worse than clean source-visible overflow.
        """
        rest = text.replace("\n", " ").strip()
        spans = math_span_extents(rest)
        out: list[str] = []
        consumed = 0
        for left, _bottom, right, _top in boxes:
            w = right - left
            if not rest:
                out.append("")
                continue
            n = self.fit_chars(rest, w, size_pt)
            cut = consumed + n
            for start, end in spans:
                if start < cut < end and start >= consumed:
                    n = start - consumed
                    break

            # Kinsoku line-start guard: next line must never start with closer punctuation
            closers = CJK_CLOSE_PUNCT | frozenset("”’\"',.:;!?)]}")
            openers = CJK_OPEN_PUNCT | frozenset("“‘\"'([{<")

            # Phase 1: Try hanging tolerance forward for any closers
            while 0 < n < len(rest) and rest[n] in closers:
                if self.squeezed_width(rest[: n + 1], size_pt) <= w + 0.6 * size_pt:
                    n += 1
                else:
                    break

            # Phase 2: If next line would still start with a closer (hanging failed or was partial),
            # strictly retreat until next line does not start with a closer and line doesn't end on opener.
            while 0 < n < len(rest) and rest[n] in closers:
                n -= 1
                while n > 0 and rest[n - 1] in openers:
                    n -= 1

            if n <= 0:
                return None
            out.append(rest[:n])
            rest = rest[n:]
            consumed += n
            stripped = rest.lstrip()
            consumed += len(rest) - len(stripped)
            rest = stripped
        if rest:
            return None

        # Orphan character prevention (JIS X 4051 CJK orphan line balance):
        # If the last non-empty line has only 1 or 2 characters (e.g. '件', '中，', '各'):
        # 1) Try to absorb it into the previous line if width permits with squeeze tolerance.
        # 2) Otherwise, borrow 2 characters from previous line to avoid solitary single-glyph line.
        non_empty_indices = [i for i, ln in enumerate(out) if ln.strip()]
        if len(non_empty_indices) >= 2:
            last_idx = non_empty_indices[-1]
            prev_idx = non_empty_indices[-2]
            last_line = out[last_idx].strip()
            prev_line = out[prev_idx].strip()
            prev_w = boxes[prev_idx][2] - boxes[prev_idx][0]
            if 0 < len(last_line) <= 2 and len(prev_line) >= 5:
                if self.squeezed_width(prev_line + last_line, size_pt) <= prev_w + 0.8 * size_pt:
                    out[prev_idx] = prev_line + last_line
                    out[last_idx] = ""
                else:
                    borrow = 2
                    while borrow < len(prev_line) - 2 and (
                        prev_line[-borrow] in closers or prev_line[-borrow - 1] in openers
                    ):
                        borrow += 1
                    if borrow < len(prev_line) - 2:
                        cand_prev = prev_line[:-borrow].rstrip()
                        cand_last = prev_line[-borrow:] + last_line
                        last_w = boxes[last_idx][2] - boxes[last_idx][0]
                        if self.squeezed_width(cand_last, size_pt) <= last_w:
                            out[prev_idx] = cand_prev
                            out[last_idx] = cand_last

        return out


__all__ = [
    "EmWidth",
    "FlowFitter",
    "Rect",
    "WidthMeasure",
]
