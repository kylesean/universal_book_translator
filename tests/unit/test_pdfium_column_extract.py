"""Tests for column-aware geometric extraction in PDFiumAdapter."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ubt.adapters.pdf.pdfium_adapter import extract_blocks_with_pdfium


def _deps_available() -> bool:
    if shutil.which("typst") is None:
        return False
    try:
        import pypdfium2  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _deps_available(), reason="typst + pypdfium2 required")

TWO_COLUMN_TYP = """
#set page(width: 500pt, height: 400pt, margin: 30pt, columns: 2)
#set text(size: 10pt, font: "Liberation Serif")

Column One Paragraph One: This is the very first paragraph in the left column.

Column One Paragraph Two: This is the second paragraph in the left column.

#colbreak()

Column Two Paragraph One: This is the first paragraph in the right column.

Column Two Paragraph Two: This is the second paragraph in the right column.
"""


def _build_twocol_pdf(tmp_path: Path) -> Path:
    typ = tmp_path / "twocol.typ"
    pdf = tmp_path / "twocol.pdf"
    typ.write_text(TWO_COLUMN_TYP, encoding="utf-8")
    subprocess.run(["typst", "compile", str(typ), str(pdf)], check=True)
    return pdf


def test_pdfium_extracts_multicolumn_in_reading_order_with_valid_bboxes(tmp_path: Path) -> None:
    pdf = _build_twocol_pdf(tmp_path)
    blocks = extract_blocks_with_pdfium(pdf)

    assert len(blocks) >= 4
    # All blocks must have valid, finite, positive-area BBoxes
    for b in blocks:
        assert b.bbox is not None
        assert b.bbox.page == 1
        assert b.bbox.x1 > b.bbox.x0
        assert b.bbox.y1 > b.bbox.y0
        assert not b.validate_contract()

    # Reading order check: Left column paragraphs must precede Right column paragraphs
    texts = [b.source_text for b in blocks]
    col1_idx = next(i for i, t in enumerate(texts) if "Column One Paragraph One" in t)
    col2_idx = next(i for i, t in enumerate(texts) if "Column Two Paragraph One" in t)
    assert col1_idx < col2_idx, (
        f"Left column (idx={col1_idx}) should precede right column (idx={col2_idx})"
    )
