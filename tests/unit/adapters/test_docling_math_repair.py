"""Unit tests for Docling math symbol corruption repair via PDFium line witness."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.adapters.pdf.docling_crosscheck import (
    reinsert_missing_spaces,
    repair_math_symbol_corruptions,
    repair_math_symbols_with_lines,
)
from ubt.adapters.pdf.textgeom import LineBox
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    text: str,
    *,
    page: int = 1,
    block_type: BlockType = BlockType.NARRATIVE,
    x0: float = 50.0,
    y0: float = 100.0,
    x1: float = 400.0,
    y1: float = 150.0,
    skip: bool = False,
) -> IRBlock:
    return IRBlock(
        element=make_element(
            block_type=block_type,
            id=block_id,
            source_text=text,
            spine_index=1,
            bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page),
            skip_translate=skip,
        )
    )


def test_repair_math_symbol_corruptions_basic() -> None:
    doc = "We denote by x mod m the remainder. For example, 17 mod 5 ∅ 2, because 17 ∅ 3 ′ 5 ⊕ 2."
    wit = "We denote by x mod m the remainder. For example, 17 mod 5=2, because 17=3·5+2."
    repaired = repair_math_symbol_corruptions(doc, wit)
    assert "17 mod 5 = 2, because 17 = 3 · 5 + 2." in repaired
    assert "∅" not in repaired
    assert "⊕" not in repaired
    assert "′" not in repaired


def test_repair_math_symbol_corruptions_upsilon_and_phi() -> None:
    doc = "it is both 1 2 ϕ s 2 ϒ s 1 ϕ d and ( s 1 ϒ p )"
    wit = "it is both 1 2 | s 2 − s 1 | d and ( s 1 − p )"
    repaired = repair_math_symbol_corruptions(doc, wit)
    assert "1 2 | s 2 − s 1 | d" in repaired
    assert "( s 1 − p )" in repaired
    assert "ϒ" not in repaired
    assert "ϕ" not in repaired


def test_repair_math_symbol_corruptions_unrelated_text_untouched() -> None:
    plain = "This is a regular paragraph with no corrupted symbols."
    assert repair_math_symbol_corruptions(plain, plain) == plain


def test_repair_math_symbols_with_lines_end_to_end() -> None:
    doc_text = "17 mod 5 ∅ 2, because 17 ∅ 3 ′ 5 ⊕ 2."
    block = _block("b001", doc_text, page=16, y0=110.0, y1=140.0)
    pdf_lines = [
        LineBox(
            "17 mod 5 = 2, because 17 = 3 · 5 + 2.",
            (50.0, 115.0, 400.0, 135.0),
        )
    ]

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        return_value=(pdf_lines, (600.0, 800.0)),
    ):
        blocks = repair_math_symbols_with_lines([block], Path("/dummy/book.pdf"))

    assert blocks[0].source_text == "17 mod 5 = 2, because 17 = 3 · 5 + 2."
    assert blocks[0].provenance.math_symbol_repair == "pdfium-line-witness"


def test_math_symbol_repair_unlocks_space_repair() -> None:
    # When symbols are corrupted ('∅'), text_chars != witness_chars so space repair fails.
    # Repairing math symbols first makes the character streams agree and allows space repair!
    doc_text = "ASandboxwith 17 mod 5 ∅ 2"
    wit_text = "A Sandbox with 17 mod 5 = 2"

    # Direct space repair fails because '∅' != '='
    assert reinsert_missing_spaces(doc_text, wit_text) is None

    # Step 1: Repair math symbols
    symbol_repaired = repair_math_symbol_corruptions(doc_text, wit_text)
    assert symbol_repaired == "ASandboxwith 17 mod 5 = 2"

    # Step 2: Space repair now succeeds!
    space_repaired = reinsert_missing_spaces(symbol_repaired, wit_text)
    assert space_repaired == "A Sandbox with 17 mod 5 = 2"
