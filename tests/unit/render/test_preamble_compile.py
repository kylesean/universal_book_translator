"""The preamble must COMPILE, not merely look right (W3-5 regression).

The string assertions in ``test_preamble_fonts.py`` could not catch the bug
this file exists for: rewriting the heading blocks' font argument left an
extra ``)`` (``text(..., font: (...)))[#it.body]``). Every heading broke and
every real render failed at compile with "missing argument: body" — but the
fast tier never compiles a document with a heading, so it stayed green until
the slow tier ran.

Slow tier: it shells out to ``typst``. The preamble's language and direction
lines are part of the same emission path, so an RTL target is covered too.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.slow


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_a_document_with_a_heading_compiles_for_every_direction(tmp_path: Path) -> None:
    for lang in ("zh", "ja", "ar"):
        source = TypstDocumentRenderer().preamble(lang) + "\n\n= Heading\n\nA paragraph.\n"
        typ = tmp_path / f"pre-{lang}.typ"
        typ.write_text(source, encoding="utf-8")
        proc = subprocess.run(
            ["typst", "compile", str(typ), str(tmp_path / f"pre-{lang}.pdf")],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"{lang}: typst rejected the preamble:\n{proc.stderr}"
