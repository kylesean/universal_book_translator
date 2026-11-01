from __future__ import annotations

import re

import pytest

from ubt.render.typst.escape import typst_escape

pytestmark = pytest.mark.fast


def _unescape_slashes(text: str) -> str:
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
    escaped = typst_escape(raw)
    raw_slashes = re.findall(r"(?<!\\)/", escaped)
    assert "//" not in "".join(raw_slashes), f"{raw!r} -> {escaped!r} still opens a comment"


def test_single_slash_is_left_alone() -> None:
    assert typst_escape("a/b") == "a/b"
    assert typst_escape("and/or") == "and/or"


def test_escaping_leaves_no_invisible_character_in_the_text() -> None:
    raw = "see https://example.com/a///b for more"
    escaped = typst_escape(raw)
    assert "\u200b" not in escaped
    assert _unescape_slashes(escaped) == raw


def test_an_url_stays_searchable_after_escaping() -> None:
    url = "https://github.com/modelcontextprotocol/servers"
    escaped = typst_escape(url)
    assert _unescape_slashes(escaped) == url
    assert "\u200b" not in escaped
