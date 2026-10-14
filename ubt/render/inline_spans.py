"""Locating styled spans (:class:`InlineRun`) in delivered text, and splitting around them.

One owner of the tolerant matching, shared by every consumer that has to re-apply
a source-recorded style to text the pipeline delivered: the Typst renderer wraps
a located span in ``#strong``/``rgb(...)`` markup, and the HTML/EPUB views wrap
it in ``<strong>``/``<span style>``. A run's ``text`` is its literal characters
(not offsets), so both map it onto the delivered text by substring -- a span the
translation rewrote finds no match and is drawn plain, which is the documented
contract of :class:`~ubt.core.ir.models.InlineRun`.

The matching is tolerant because the recorded text and the delivered text drift
apart in presentation-only ways: the target's CJK publishing pass inserts a
space between Latin and CJK that draft-time run text does not carry, and
full-width ASCII, the ideographic space, and the unicode dash/quote variants
fold to their ASCII forms before the substring search.
"""

from __future__ import annotations

from collections.abc import Sequence

from ubt.core.ir.models import InlineRun, IRBlock

#: A located span: ``(start, end, bold, italic, superscript, color_hex)``.
SpanStyle = tuple[int, int, bool, bool, bool, str | None]


def fold_char(ch: str) -> str:
    """Fold one char for tolerant matching.

    Full-width ASCII becomes half-width, the ideographic space a plain space,
    and the unicode dash/quote variants their ASCII forms -- so a run text the
    extractor and the target spell differently still matches.
    """
    code = ord(ch)
    if code == 0x3000:
        return " "
    if 0xFF01 <= code <= 0xFF5E:
        return chr(code - 0xFEE0)
    if code == 0x2212 or 0x2010 <= code <= 0x2015:
        return "-"
    if code in (0x2018, 0x2019, 0x02BC, 0x0060, 0x00B4):
        return "'"
    if code in (0x201C, 0x201D):
        return '"'
    return ch


def fold_for_match(text: str) -> tuple[str, list[int], list[int]]:
    """Fold ``text`` for tolerant matching, with a map back to original indices.

    Drops all whitespace and folds full-width ASCII to half-width, so a run text
    recorded before the target's publishing normalization still matches it: the
    CJK pass inserts a space between Latin and CJK that the draft-time run text
    does not carry.
    """
    folded: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    length = len(text)
    while index < length:
        ch = text[index]
        if ch.isspace():
            index += 1
            continue
        folded.append(fold_char(ch))
        starts.append(index)
        ends.append(index + 1)
        index += 1
    return "".join(folded), starts, ends


def folded_find(text: str, needle: str, from_index: int = 0) -> tuple[int, int] | None:
    """Find ``needle`` in ``text`` ignoring whitespace/full-width differences."""
    folded_needle = fold_for_match(needle)[0].strip()
    if not folded_needle:
        return None
    folded_text, starts, ends = fold_for_match(text)
    position = 0
    if from_index > 0:
        while position < len(starts) and starts[position] < from_index:
            position += 1
    at = folded_text.find(folded_needle, position)
    if at < 0:
        return None
    return starts[at], ends[at + len(folded_needle) - 1]


def locate_runs(text: str, runs: Sequence[InlineRun]) -> list[SpanStyle]:
    """Locate each run's literal span in ``text``, in order.

    A run whose text is not present (a span the translation rewrote) is skipped;
    the sequential cursor keeps repeated tokens aligned to their occurrence
    rather than the first one. A span that survives only after the target's CJK
    normalization is found by the tolerant fallback.
    """
    spans: list[SpanStyle] = []
    cursor = 0
    for run in runs:
        if not run.text:
            continue
        index = text.find(run.text, cursor)
        if index < 0:
            index = text.find(run.text)
        if index >= 0:
            spans.append(
                (
                    index,
                    index + len(run.text),
                    run.bold,
                    run.italic,
                    run.superscript,
                    run.color_hex,
                )
            )
            cursor = index + len(run.text)
            continue
        located = folded_find(text, run.text, cursor)
        if located is None:
            located = folded_find(text, run.text, 0)
        if located is None:
            continue
        start, end = located
        spans.append((start, end, run.bold, run.italic, run.superscript, run.color_hex))
        cursor = end
    return spans


def chunks_for_spans(text: str, spans: Sequence[SpanStyle]) -> list[tuple[str, SpanStyle | None]]:
    """Split ``text`` at the spans' boundaries; each chunk carries the style enclosing it.

    A chunk inside two spans takes the *first* enclosing span: the spans are
    located in order and a caller applies them non-nested (an outer wrapper
    would need nesting semantics the source probe does not model). Callers that
    must exclude a span (the renderer drops runs straddling a math delimiter)
    filter ``spans`` before calling.
    """
    if not spans:
        return [(text, None)]
    boundaries: set[int] = {0, len(text)}
    for start, end, *_style in spans:
        boundaries.update((start, end))
    out: list[tuple[str, SpanStyle | None]] = []
    for start, end in zip(sorted(boundaries), sorted(boundaries)[1:], strict=False):
        if end <= start:
            continue
        chunk = text[start:end]
        style = next((s for s in spans if s[0] <= start and end <= s[1]), None)
        out.append((chunk, style))
    return out


def styled_chunks(text: str, runs: Sequence[InlineRun]) -> list[tuple[str, SpanStyle | None]]:
    """Locate ``runs`` in ``text`` and split at their boundaries."""
    return chunks_for_spans(text, locate_runs(text, runs) if runs else [])


def block_runs(block: IRBlock) -> tuple[InlineRun, ...]:
    """The block's source- and target-derived inline runs, as renderable runs.

    Source-derived runs (``inline_runs``) come first: they are usually the more
    specific span (a blue citation inside a bold sentence), and a consumer's
    first-enclosing-wins lookup lets the earlier span win. Target-derived runs
    (``target_runs``, the model's bold markers) follow.
    """
    style = block.style
    if style is None:
        return ()
    return tuple(run for run in (*style.inline_runs, *style.target_runs) if run.text)


__all__ = [
    "SpanStyle",
    "block_runs",
    "chunks_for_spans",
    "fold_char",
    "fold_for_match",
    "folded_find",
    "locate_runs",
    "styled_chunks",
]
