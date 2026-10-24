"""The pdf-oxide binding contract, pinned so a dependency bump fails here first.

pdf-oxide is young (0.3.x, near-weekly releases) and its 0.3.x bindings have
drifted against its own README before (``page.text`` exists in the README and
not in 0.3.78). UBT touches exactly four symbols — ``PdfDocument``,
``page_count``, ``render_page(idx, dpi=)`` and ``extract_text(idx)`` — all
through :mod:`ubt.adapters.pdf.oxide_render`, whose entry points never raise
(a binding drift therefore degrades the visual gates *silently* at runtime).
This contract exercises the four symbols through a real document, so a bump
that renames, re-signatures or re-semantics one of them is a red test at
review time instead. When the pin in ``pyproject.toml`` is raised, this file
is the re-probe the old comment demanded.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from pdf_builders import write_text_pdf

from ubt.adapters.pdf.oxide_render import extract_page_texts, render_page_png

pytestmark = pytest.mark.fast


def _png_dimensions(png: bytes) -> tuple[int, int]:
    """(width, height) from the PNG IHDR chunk."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    width, height = struct.unpack(">II", png[16:24])
    return width, height


def test_the_four_symbols_behave_as_oxide_render_assumes(tmp_path: Path) -> None:
    from pdf_oxide import PdfDocument

    pdf = tmp_path / "contract.pdf"
    write_text_pdf(pdf, ["The four symbols the engine relies on."])
    doc = PdfDocument(str(pdf))
    assert doc.page_count == 1

    rendered_72 = bytes(doc.render_page(0, dpi=72))
    assert rendered_72[:4] == b"\x89PNG"
    rendered_144 = bytes(doc.render_page(0, dpi=144))
    # DPI parity: "dpi" must mean dots per 72pt inch, the assumption the
    # svg_diagram crop math and the visual gate's pixel thresholds sit on.
    w72, h72 = _png_dimensions(rendered_72)
    w144, h144 = _png_dimensions(rendered_144)
    assert (w144, h144) == (w72 * 2, h72 * 2)

    # 0-based page indexing and a str return are what extract_page_texts maps.
    # The text SHAPE is part of the contract: for this fixture's simple font
    # pdf-oxide emits one glyph per line ("T\nh\ne\n …") — the blank-page
    # detectors in the visual gate count characters, not words, so the shape
    # only has to stay non-empty and lossless, never word-wise.
    text = doc.extract_text(0)
    assert isinstance(text, str)
    assert "symbols" in "".join(text.split())


def test_oxide_render_legs_map_the_conventions(tmp_path: Path) -> None:
    """The wrapper's 1-based pages and never-raise contract, on the real engine."""
    pdf = tmp_path / "legs.pdf"
    write_text_pdf(pdf, ["First page body.", "Second page body."])
    png = render_page_png(pdf, page=1, dpi=72)
    assert png is not None and png[:4] == b"\x89PNG"
    # 1-based: page 2 exists on a two-page document, page 3 does not.
    assert render_page_png(pdf, page=2, dpi=72) is not None
    assert render_page_png(pdf, page=3, dpi=72) is None
    pages = extract_page_texts(pdf)
    assert len(pages) == 2
    assert "Second" in "".join(pages[1].split())
