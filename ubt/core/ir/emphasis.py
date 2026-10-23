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
ITALIC_OPEN = "⟦I⟧"
ITALIC_CLOSE = "⟦/I⟧"

#: Tolerant marker match. A model reading the sentinels as mathematical brackets
#: routinely transcribes ``⟦B⟧`` into LaTeX (``\llbracket B \rrbracket``), and
#: ``⟦/B⟧`` into ``\llbracket /B \rrbracket`` or ``\llbracket / B \rrbracket``.
#: Those are the *same* marker: leaving them in the text put a stray ``\llbracket``
#: and a bare ``/`` inside a math span, and Typst failed with ``unknown variable:
#: lbracket`` (mamba ``pdf_main#b0426``) — losing the document. Spacing and case
#: inside the brackets are accepted too, so ``⟦ B ⟧`` and ``⟦/b⟧`` match.
_MARKER_RE = re.compile(r"(?:\u27e6|\\llbracket)\s*(/)?\s*([BbIi])\s*(?:\u27e7|\\rrbracket)")

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


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch == "_")


def _has_valid_word_boundary(text: str, start: int, end: int) -> bool:
    if start < end:
        if _is_ascii_word_char(text[start]) and start > 0 and _is_ascii_word_char(text[start - 1]):
            return False
        if (
            _is_ascii_word_char(text[end - 1])
            and end < len(text)
            and _is_ascii_word_char(text[end])
        ):
            return False
    return True


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
    """Locate ``needle`` in ``text`` from ``cursor``, tolerant of dash/quote and word boundaries."""
    search_pos = cursor
    needle_len = len(needle)
    while search_pos < len(text):
        index = text.find(needle, search_pos)
        if index < 0:
            break
        end = index + needle_len
        if _has_valid_word_boundary(text, index, end):
            return index, end
        search_pos = index + 1

    pattern = re.compile(_flexible_pattern(needle))
    for match in pattern.finditer(text, cursor):
        start, end = match.start(), match.end()
        if _has_valid_word_boundary(text, start, end):
            return start, end
    return None


def mark_emphasis_spans(
    text: str,
    *,
    bold_spans: Sequence[str] = (),
    italic_spans: Sequence[str] = (),
) -> str:
    """Wrap each source span's occurrence in ``text`` with bold/italic sentinels."""
    if not bold_spans and not italic_spans:
        return text

    items: list[tuple[int, int, str, str]] = []
    cursor = 0
    for span in bold_spans:
        needle = span.strip()
        if not needle:
            continue
        loc = _locate_span(text, needle, cursor)
        if loc is not None:
            items.append((loc[0], loc[1], BOLD_OPEN, BOLD_CLOSE))
            cursor = loc[1]

    cursor = 0
    for span in italic_spans:
        needle = span.strip()
        if not needle:
            continue
        loc = _locate_span(text, needle, cursor)
        if loc is not None and not any(
            b_start <= loc[0] < b_end or b_start < loc[1] <= b_end for b_start, b_end, _, _ in items
        ):
            items.append((loc[0], loc[1], ITALIC_OPEN, ITALIC_CLOSE))
            cursor = loc[1]

    if not items:
        return text

    items.sort(key=lambda x: x[0])
    marked: list[str] = []
    idx = 0
    for start, end, open_tag, close_tag in items:
        if start < idx:
            continue
        marked.append(text[idx:start])
        marked.append(open_tag)
        marked.append(text[start:end])
        marked.append(close_tag)
        idx = end
    marked.append(text[idx:])
    return "".join(marked)


def mark_bold_spans(text: str, spans: Sequence[str]) -> str:
    """Wrap each source span's occurrence in ``text`` with the bold sentinels.

    Spans are located left-to-right with a moving cursor, so a repeated token
    maps to its own occurrence; a span not found from the cursor (or empty) is
    skipped. Because the cursor advances past each wrapped span, wrapped regions
    never overlap or nest. A dash/quote the extractor and the reader spell
    differently still matches.
    """
    return mark_emphasis_spans(text, bold_spans=spans)


def parse_emphasis(text: str) -> tuple[str, tuple[InlineRun, ...]]:
    """Split ``text`` into clean target text and its bold/italic runs.

    Returns the text with every marker stripped and one ``InlineRun`` per
    contiguous emphasized region. Unbalanced markers are tolerated.
    """
    clean_chars: list[str] = []
    style_flags: list[tuple[bool, bool]] = []
    last = 0
    bold_depth = 0
    italic_depth = 0
    for match in _MARKER_RE.finditer(text):
        chunk = text[last : match.start()]
        clean_chars.append(chunk)
        style_flags.extend([(bold_depth > 0, italic_depth > 0)] * len(chunk))
        is_close = bool(match.group(1))
        tag_type = (match.group(2) or "b").lower()
        if tag_type == "i":
            if is_close:
                if italic_depth > 0:
                    italic_depth -= 1
            else:
                italic_depth += 1
        else:
            if is_close:
                if bold_depth > 0:
                    bold_depth -= 1
            else:
                bold_depth += 1
        last = match.end()
    tail = text[last:]
    clean_chars.append(tail)
    style_flags.extend([(bold_depth > 0, italic_depth > 0)] * len(tail))
    clean_text = "".join(clean_chars)

    runs: list[InlineRun] = []
    index = 0
    length = len(style_flags)
    while index < length:
        b, it = style_flags[index]
        if not b and not it:
            index += 1
            continue
        stop = index
        while stop < length and style_flags[stop] == (b, it):
            stop += 1
        segment = clean_text[index:stop].strip()
        if segment:
            runs.append(InlineRun(text=segment, bold=b, italic=it))
        index = stop
    return clean_text, tuple(runs)


def parse_bold(text: str) -> tuple[str, tuple[InlineRun, ...]]:
    """Split ``text`` into clean target text and its bold runs."""
    return parse_emphasis(text)


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
    "ITALIC_CLOSE",
    "ITALIC_OPEN",
    "mark_bold_spans",
    "mark_emphasis_spans",
    "parse_bold",
    "parse_emphasis",
    "runs_from_json",
    "runs_to_json",
    "strip_emphasis_markers",
]
