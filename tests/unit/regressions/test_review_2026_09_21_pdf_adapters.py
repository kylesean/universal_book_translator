"""Regression guards for the 2026-09-21 review round: PDF adapter fixes.

Each test reproduces a defect found by the 2026-09-21 review, so it fails if
the fix is reverted.
"""

from __future__ import annotations

from pathlib import Path

from ubt.adapters.pdf.asset_extractor import extract_pdf_figures


def _caption_pdf(path: Path, pages: int = 2) -> Path:
    """Minimal born-digital PDF: every page shows the same "FIG. 3.2" caption.

    Hand-written bytes (the shape ``tests/pdf_builders.text_pdf`` uses) so the
    extraction test does not also test a writer library.
    """
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(pages))
    objs: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
    }
    for i in range(pages):
        text = "FIG. 3.2 Shared caption body text here."
        content = f"BT /F1 12 Tf 72 100 Td ({text}) Tj ET".encode("latin-1")
        objs[3 + i * 2] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            + f"/Contents {4 + i * 2} 0 R /Resources << /Font << /F1 99 0 R >> >> >>".encode()
        )
        objs[4 + i * 2] = (
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        )
    objs[99] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    size = max(offsets) + 1
    xref_pos = len(out)
    out += f"xref\n0 {size}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, size):
        out += f"{offsets.get(num, 0):010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    path.write_bytes(bytes(out))
    return path


def test_figures_with_the_same_number_on_different_pages_stay_distinct(
    tmp_path: Path,
) -> None:
    """A book repeats "Figure 3.2" in more than one chapter.

    Keying by figure number alone made the later page overwrite the earlier
    PNG and dict entry, so the first figure's translation was lost and both
    blocks pointed at one image.
    """
    pdf = _caption_pdf(tmp_path / "caps.pdf", pages=2)

    figures = extract_pdf_figures(pdf, tmp_path / "assets", dpi=72)

    assert len(figures) == 2, "each page's figure must survive as its own entry"
    assert {fig.page for fig in figures.values()} == {1, 2}
    paths = {fig.image_path for fig in figures.values()}
    assert len(paths) == 2 and all(p.exists() for p in paths)
