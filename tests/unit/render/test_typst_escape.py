"""``typst_escape`` must leave no ``//`` that Typst reads as a line comment.

A literal ``//`` in translated text opens a Typst line comment even inside a
``[content]`` block, silently discarding everything after it on the line — a
``/// <summary>`` doc comment vanished from the page, and inside a table cell
the leftover made the whole document fail to compile. The original fix used a
single non-overlapping ``str.replace("//", "/​/")``, which missed odd-length
slash runs: ``a///b`` became ``a/​//b`` and still contained ``//``. These tests
pin the run-length cases the ``str.replace`` form missed.
"""

from __future__ import annotations

import re

import pytest

from ubt.render.typst.escape import typst_escape

pytestmark = pytest.mark.fast


def _unescape_slashes(text: str) -> str:
    r"""Turn ``\/`` back into ``/``, scanning left to right.

    A plain ``replace(chr(92)+"/", "/")`` is wrong here: in ``/\//\/`` the
    backslash before the second slash is itself the escape for the next one, so
    a naive pass collapses two escapes into one and the round-trip loses a
    character.
    """
    out: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == chr(92) and index + 1 < len(text):
            out.append(text[index + 1])
            index += 2
            continue
        out.append(text[index])
        index += 1
    return "".join(out)


@pytest.mark.parametrize(
    "raw",
    [
        "a//b",
        "a///b",
        "a////b",
        "a/////b",
        "x//////y",
        "/// <summary>",
        "file:///tmp/x",
        "see http://x.com/p//q",
        "https://x///y",
    ],
)
def test_no_unescaped_double_slash_survives_any_run_length(raw: str) -> None:
    r"""Every slash in a run is escaped, so Typst cannot read a comment.

    The assertion is about the *unescaped* form: ``\/`` renders as ``/`` and
    blocks the comment token, while escaping only alternate slashes left a
    literal ``//`` that swallowed the rest of the line.
    """
    escaped = typst_escape(raw)
    # A slash is "raw" when no backslash precedes it; two raw slashes in a row
    # would open a Typst comment.
    raw_slashes = re.findall(r"(?<!\\)/", escaped)
    assert "//" not in "".join(raw_slashes), f"{raw!r} -> {escaped!r} still opens a comment"


def test_single_slash_is_left_alone() -> None:
    # A lone ``/`` is not a comment opener; escaping it would litter every path.
    assert typst_escape("a/b") == "a/b"
    assert typst_escape("and/or") == "and/or"


def test_escaping_leaves_no_invisible_character_in_the_text() -> None:
    """A backslash escape is invisible; a ZWSP is not.

    The earlier fix joined slash runs with zero-width spaces. It blocked the
    comment, but a ZWSP IS a character in the PDF's text layer: every
    ``https://`` in a delivered paper came out as ``https:/\u200b/``, so 77 URLs
    could not be searched for or pasted into a browser.
    """
    raw = "see https://example.com/a///b for more"
    escaped = typst_escape(raw)
    assert "\u200b" not in escaped
    # Stripping the escapes recovers the original text exactly.
    assert _unescape_slashes(escaped) == raw


def test_an_url_stays_searchable_after_escaping() -> None:
    """The property the ZWSP version broke, stated directly."""
    url = "https://github.com/modelcontextprotocol/servers"
    escaped = typst_escape(url)
    assert _unescape_slashes(escaped) == url
    assert "\u200b" not in escaped
