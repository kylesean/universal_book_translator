r"""Typst *markup* escaping (the non-math half of the render boundary).

The renderer and ``ubt/render/table_compiler.py`` both need
identical escaping, so it lives here rather than in the PDF adapter.

Two functions:

``escape_line_start_markup``
    Escape the markers Typst treats specially **only at line start** (``=``,
    ``+``, ``-``, ``/``, ``N.``). They are deliberately absent from the mid-line
    escape set, because escaping them everywhere would litter ordinary prose.
``typst_escape``
    Escape a whole text run for Typst markup: the structural characters, then
    the line-start markers, then the ``//`` comment opener.
"""

from __future__ import annotations

import re

from ubt.core.cleaners.html_sanitizer import strip_html_mark_tags

_LINE_START_MARKUP_RE = re.compile(r"(?m)^([ \t]*)(=|\+|-|/|\d+\.)(?=\s|$)")

#: A maximal run of two or more slashes. Escaped as ``\/`` between them, so
#: ``//`` can never survive to open a Typst comment (see :func:`typst_escape`).
#: A single slash is left alone.
_SLASH_RUN_RE = re.compile(r"/{2,}")


def _escape_slash_run(m: re.Match[str]) -> str:
    r"""Escape every slash in a run: ``//`` -> ``\/\/``.

    Typst reads ``//`` as a line comment even inside ``[#text[...]]``, swallowing
    the closing brackets and failing the compile; ``///`` lost a whole doc
    comment, and inside a table cell the leftover broke the document.

    The previous fix joined the run's slashes with a zero-width space. That
    breaks the comment token, but a ZWSP is a real character in the PDF's text
    layer: every ``https://`` in a delivered file came out as ``https:/\u200b/``,
    so the URL could not be found by search or pasted into a browser. Measured on
    a 22-page paper: 77 such URLs, zero of them searchable.

    A backslash escape is invisible in the text layer *and* blocks the comment.
    Escaping each slash individually -- not joining them with an inserted
    character -- is what keeps the round-trip exact: ``//`` must become
    ``\/\/`` (four characters for two), never ``/\//`` (which still contains a
    literal ``//`` and unescapes to three slashes).
    """
    return "".join("\\/" for _ in m.group(0))


def escape_line_start_markup(text: str) -> str:
    """Escape Typst block markers that are special only at the start of a line.

    ``=``/``+``/``-``/``/``/``N.`` open a heading/list/enum/term-list in Typst
    *only* at line start, so they are deliberately absent from the mid-line
    escape sets (escaping them everywhere would litter ordinary prose and
    maths). A translated paragraph can still carry a newline immediately before
    such a marker, which silently re-lays-out the page as a heading or list (or
    fails the whole compile, as a ``/`` term marker does); escape the marker
    (and, for ``N.``, its dot) so it renders as literal text.
    """

    # The marker must be followed by whitespace or end-of-line: ``12.5`` is a
    # decimal, not an enumerated-list item, and must be left untouched.
    def _repl(m: re.Match[str]) -> str:
        lead, marker = m.group(1), m.group(2)
        if marker.endswith("."):
            return f"{lead}{marker[:-1]}\\."
        return f"{lead}\\{marker}"

    return _LINE_START_MARKUP_RE.sub(_repl, text)


def typst_escape(text: str) -> str:
    text = strip_html_mark_tags(text)
    out: list[str] = []
    for ch in text:
        # `[` / `]` are structural (they open/close content blocks): a chunk
        # split mid-span (`[26]` -> `[2` | `6]`, chapter-3 p2/p17) would
        # otherwise unbalance the statement, swallowing every following
        # statement on the page ~200pt away (A.1 garble, eq-3.1 intrusion).
        if ch in ("\\", "{", "}", "$", "#", "@", "<", ">", "`", "*", "_", "[", "]", "~"):
            out.append("\\" + ch)
        elif ch == '"':
            out.append('\\"')
        else:
            out.append(ch)
    # `//` opens a Typst line comment even inside [#text[...]] content,
    # swallowing the closing brackets and failing the whole compile. Escape it.
    #
    # A maximal-run replace, not ``str.replace``: ``str.replace`` is
    # non-overlapping and single-pass, so a run of 3 slashes kept a literal
    # ``//`` and Typst still read it as a comment. Escaping *every* slash in the
    # run means no ``//`` survives whatever the run length.
    escaped = escape_line_start_markup("".join(out))
    return _SLASH_RUN_RE.sub(_escape_slash_run, escaped)
