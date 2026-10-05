"""Target-side emphasis markup: the model's ``⟦B⟧ … ⟦/B⟧`` bold spans.

The pdfium probe finds *source* styled runs, but a run the translation rewrote
has no source substring to locate in the target. For those the draft prompt asks
the model to bracket the emphasized target text with paired sentinels; a run
that survives verbatim is still handled by the source-derived mechanism. The
sentinels belong to the project's ``⟦…⟧`` token family, so the mask restore
never mistakes them for a placeholder, and they never collide with the ``*``
significance markers common in academic text.

Nothing here imports the pipeline: it is a pure text codec over
:class:`~ubt.core.ir.models.InlineRun`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ubt.core.ir.models import InlineRun

#: Paired sentinels the draft prompt asks the model to preserve.
BOLD_OPEN = "⟦B⟧"
BOLD_CLOSE = "⟦/B⟧"

#: Tolerant marker match: spacing and case inside the brackets are accepted so
#: a model that writes ``⟦ B ⟧`` or ``⟦/b⟧`` is still understood. The ``⟦⟧``
#: delimiters make a false positive on ordinary text effectively impossible.
_MARKER_RE = re.compile(r"⟦\s*(/)?\s*[Bb]\s*⟧")

#: A pdfium char probe and the docling text can disagree on a dash or quote
#: (``agent–environment`` vs ``agent-environment``), so a source span located
#: verbatim in the run text may not be found in the block's source text. These
#: classes make the flexible characters equivalent during the fallback search.
_DASH_CLASS = r"[\u2010-\u2015\u2212-]"
_APOSTROPHE_CLASS = "['\u2018\u2019\u02bc`\u00b4]"
_QUOTE_CLASS = '["\u201c\u201d]'
_DASHES = frozenset("-") | {chr(code) for code in range(0x2010, 0x2016)} | {"\u2212"}
_APOSTROPHES = frozenset({"'"})
_APOSTROPHES |= {"\u2018", "\u2019", "\u02bc", "`", "\u00b4"}
_QUOTES = frozenset({'"', "\u201c", "\u201d"})


def _flexible_pattern(needle: str) -> str:
    """A regex matching ``needle`` with dash/quote/whitespace variants folded."""
    parts: list[str] = []
    index = 0
    length = len(needle)
    while index < length:
        ch = needle[index]
        if ch.isspace():
            stop = index
            while stop < length and needle[stop].isspace():
                stop += 1
            parts.append(r"\s+")
            index = stop
            continue
        if ch in _DASHES:
            parts.append(_DASH_CLASS)
        elif ch in _APOSTROPHES:
            parts.append(_APOSTROPHE_CLASS)
        elif ch in _QUOTES:
            parts.append(_QUOTE_CLASS)
        else:
            parts.append(re.escape(ch))
        index += 1
    return "".join(parts)


def _locate_span(text: str, needle: str, cursor: int) -> tuple[int, int] | None:
    """Locate ``needle`` in ``text`` from ``cursor``, tolerant of dash/quote."""
    index = text.find(needle, cursor)
    if index >= 0:
        return index, index + len(needle)
    match = re.search(_flexible_pattern(needle), text[cursor:])
    if match is None:
        return None
    return cursor + match.start(), cursor + match.end()


def mark_bold_spans(text: str, spans: Sequence[str]) -> str:
    """Wrap each source span's occurrence in ``text`` with the bold sentinels.

    Spans are located left-to-right with a moving cursor, so a repeated token
    maps to its own occurrence; a span not found from the cursor (or empty) is
    skipped. Because the cursor advances past each wrapped span, wrapped regions
    never overlap or nest. A dash/quote the extractor and the reader spell
    differently still matches.
    """
    if not spans:
        return text
    marked: list[str] = []
    cursor = 0
    for span in spans:
        needle = span.strip()
        if not needle:
            continue
        located = _locate_span(text, needle, cursor)
        if located is None:
            continue
        start, end = located
        marked.append(text[cursor:start])
        marked.append(BOLD_OPEN)
        marked.append(text[start:end])
        marked.append(BOLD_CLOSE)
        cursor = end
    marked.append(text[cursor:])
    return "".join(marked)


def parse_bold(text: str) -> tuple[str, tuple[InlineRun, ...]]:
    """Split ``text`` into clean target text and its bold runs.

    Returns the text with every marker stripped and one ``bold=True`` run per
    contiguous emphasized region. Unbalanced markers are tolerated: a stray
    close is dropped, an unterminated open emphasizes to the end, and a marker
    is never left behind in the returned text.
    """
    clean_chars: list[str] = []
    bold_flags: list[bool] = []
    last = 0
    depth = 0
    for match in _MARKER_RE.finditer(text):
        chunk = text[last : match.start()]
        clean_chars.append(chunk)
        bold_flags.extend([depth > 0] * len(chunk))
        if match.group(1):  # closing marker
            if depth > 0:
                depth -= 1
        else:  # opening marker
            depth += 1
        last = match.end()
    tail = text[last:]
    clean_chars.append(tail)
    bold_flags.extend([depth > 0] * len(tail))

    clean_text = "".join(clean_chars)
    runs: list[InlineRun] = []
    index = 0
    length = len(bold_flags)
    while index < length:
        if not bold_flags[index]:
            index += 1
            continue
        stop = index
        while stop < length and bold_flags[stop]:
            stop += 1
        segment = clean_text[index:stop].strip()
        if segment:
            runs.append(InlineRun(text=segment, bold=True))
        index = stop
    return clean_text, tuple(runs)


def strip_emphasis_markers(text: str) -> str:
    """``text`` with every bold marker removed (for display/length checks)."""
    return _MARKER_RE.sub("", text)


def runs_to_json(runs: Sequence[InlineRun]) -> str:
    """Serialize target-side runs for storage (the shared translation memory)."""
    return json.dumps([run.model_dump() for run in runs], ensure_ascii=False)


def runs_from_json(text: str) -> tuple[InlineRun, ...]:
    """Parse runs stored by :func:`runs_to_json`; empty on any malformed value."""
    if not text:
        return ()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return ()
    if not isinstance(data, list):
        return ()
    runs: list[InlineRun] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            runs.append(InlineRun.model_validate(item))
        except ValueError:
            continue
    return tuple(runs)


__all__ = [
    "BOLD_CLOSE",
    "BOLD_OPEN",
    "mark_bold_spans",
    "parse_bold",
    "runs_from_json",
    "runs_to_json",
    "strip_emphasis_markers",
]
