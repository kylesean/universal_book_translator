"""The styled-run terminator is a rule, not a list (slow: compiles Typst).

Kept apart from ``test_document_renderer`` because that module is fast-tier and
this one shells out to the compiler; mixing tiers in one module trips the
marker-hygiene gate.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ubt.core.ir.models import InlineRun
from ubt.render.typst.document_renderer import TypstDocumentRenderer, render_prose

pytestmark = pytest.mark.slow


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_every_ascii_follower_after_a_styled_run_compiles(tmp_path: Path) -> None:
    """A styled run must never make the *next character* a parse failure.

    The old guard was a two-entry blacklist (``(`` and ``.`` + identifier) whose
    correctness depended on an undocumented invariant: that
    ``typst_escape`` had already backslash-escaped every other follower. That
    invariant holds today — an exhaustive sweep of these 32 characters found no
    compile failure even before the terminator became unconditional — which is
    exactly why a blacklist was unsafe: nothing in the code asserted it, so a
    later change to the escaper or to what may follow a run would break
    rendering with no test failing.

    Now every styled run is followed by ``/**/``, so the question is closed by
    construction. This test compiles the whole class of cases anyway, because
    the terminator itself could be wrong: an empty comment must terminate the
    expression *and* change nothing on the page.
    """
    import subprocess

    followers = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    preamble = TypstDocumentRenderer().preamble("zh")
    # Use a distinct sentinel after each follower so a dropped tail is visible.
    for ch in followers:
        body = render_prose(f"甲{ch}尾标", (InlineRun(text="甲", bold=True),))
        source = preamble + "\n" + body + "\n"
        typ_path = tmp_path / "t.typ"
        typ_path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            ["typst", "compile", str(typ_path), str(tmp_path / "t.pdf"), "--font-path", "fonts"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"a styled run followed by {ch!r} failed to compile: "
            f"{result.stderr.strip().splitlines()[0] if result.stderr else ''}"
        )
        assert "尾标" in body, f"the text after {ch!r} was dropped from the source"
