"""The emitted raw code block must survive the real Typst compiler.

A markup newline folds into a space, so only the compiled page proves the
listing's line structure held; the string assertions in
``test_code_block_rendering.py`` cannot. Separate slow-tier file: the fast
module above carries module-level ``fast`` and must not grow a ``slow`` test
(it would drag the pre-push tier).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.slow


def _code_block(text: str) -> IRBlock:
    return IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.CODE,
            source_text=text,
            skip_translate=True,
        ),
        target_text=text,
    )


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_the_emitted_listing_compiles_and_keeps_its_lines(tmp_path: Path) -> None:
    listing = 'def load_balance_loss(p):\n    return p  # keep "quotes" & #hash'
    build = tmp_path / "build"
    body = TypstDocumentRenderer().build_source([_code_block(listing)], build)[0]
    source = TypstDocumentRenderer().preamble("zh") + "\n\n" + body
    typ = tmp_path / "code.typ"
    typ.write_text(source, encoding="utf-8")
    proc = subprocess.run(
        ["typst", "compile", str(typ), str(tmp_path / "code.pdf")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"typst rejected the listing:\n{proc.stderr}"

    from pypdf import PdfReader

    text = PdfReader(tmp_path / "code.pdf").pages[0].extract_text()
    # Both lines present and the metacharacters literal — a folded newline or
    # an entered code mode would fail one of these two.
    assert "def load_balance_loss(p):" in text
    assert 'return p  # keep "quotes" & #hash' in text
