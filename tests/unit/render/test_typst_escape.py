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

import pytest

from ubt.render.typst.escape import typst_escape

pytestmark = pytest.mark.fast


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
def test_no_double_slash_survives_any_run_length(raw: str) -> None:
    escaped = typst_escape(raw)
    assert "//" not in escaped, f"{raw!r} -> {escaped!r} still opens a comment"


def test_single_slash_is_left_alone() -> None:
    # A lone ``/`` is not a comment opener; escaping it would litter every path.
    assert typst_escape("a/b") == "a/b"
    assert typst_escape("and/or") == "and/or"


_ZWSP = "​"


def test_zero_width_space_is_the_only_insertion() -> None:
    # The break must be invisible: stripping the ZWSP recovers the input.
    raw = "see https://example.com/a///b for more"
    assert typst_escape(raw).replace(_ZWSP, "") == raw
