"""P2-12: the ports bridge and the visual-scalpel provider must agree.

The L4 visual-repair path reaches :func:`ubt.adapters.pdf.visual_scalpel.
is_visual_scalpel_applicable` through the SPI bridge in
:func:`ubt.core.ports.is_visual_scalpel_applicable`. The two names had drifted:
the bridge passed ``vision_threshold=`` while the provider declared
``qe_threshold_for_vision``, so every real call raised ``TypeError`` and the L4
path never actually ran — the exception was swallowed by a broad ``except`` in
the repair loop and only logged at debug level.

Pinned here: the bridge is callable (keyword names match the provider), it
returns the provider's verdict, and the provider still makes the decisions it is
supposed to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.core.ports import is_visual_scalpel_applicable

pytestmark = pytest.mark.fast


def _block(
    *,
    block_type: BlockType = BlockType.NARRATIVE,
    mtqe_score: float | None = None,
    page: int = 1,
) -> IRBlock:
    return IRBlock(
        element=make_element(
            block_type=block_type,
            id="b1",
            source_text="text",
            spine_index=1,
            bbox=BoundingBox(x0=50.0, y0=100.0, x1=400.0, y1=150.0, page=page),
        ),
        mtqe_score=mtqe_score,
    )


def test_the_ports_bridge_does_not_raise_on_the_threshold_keyword(tmp_path: Path) -> None:
    """The bridge's ``vision_threshold=`` keyword must be accepted by the provider."""
    pdf = tmp_path / "src.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    # A formula block qualifies regardless of score; the call must not raise a
    # TypeError over the drifted keyword name.
    assert is_visual_scalpel_applicable(
        _block(block_type=BlockType.FORMULA), source_pdf_path=pdf, vision_threshold=0.6
    )


def test_the_threshold_still_gates_a_low_scoring_block(tmp_path: Path) -> None:
    """A narrative block below the threshold qualifies; one above does not."""
    pdf = tmp_path / "src.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    assert is_visual_scalpel_applicable(
        _block(mtqe_score=0.3), source_pdf_path=pdf, vision_threshold=0.6
    )
    assert not is_visual_scalpel_applicable(
        _block(mtqe_score=0.9), source_pdf_path=pdf, vision_threshold=0.6
    )


def test_a_non_pdf_source_is_never_applicable(tmp_path: Path) -> None:
    """The scalpel crops from a real PDF; a non-PDF path is refused."""
    doc = tmp_path / "src.docx"
    doc.write_bytes(b"PK\x03\x04")

    assert not is_visual_scalpel_applicable(
        _block(block_type=BlockType.FORMULA), source_pdf_path=doc, vision_threshold=0.6
    )
