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
    import subprocess

    followers = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    preamble = TypstDocumentRenderer().preamble("zh")
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
