"""Builders for the synthetic PDFs the suite feeds its parsers.

Two shapes, because the tests need two different things out of a fake PDF:

* :func:`text_pdf` — a born-digital page whose text a parser can actually pull
  out of the content stream. Hand-built bytes rather than a writer library so the
  extraction tests are not also testing pypdf.
* :func:`blank_pdf` — pages with *no* text at all, i.e. a scan. The fast-lane
  probe and the scan fail-closed guard both key off "extractable chars == 0", so
  a born-digital page cannot stand in for this case.

``_text_pdf`` was copy-pasted byte-identically into three test files and the
blank-page helper existed four times under two names with two different A4
roundings, which is how a test ends up asserting on a page size nobody chose.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter

#: A4 in points. ``test_bilingual_alternator_interleaving`` asserts this exact
#: rounding, so it is the canonical value here rather than 595 x 842.
A4_PORTRAIT = (595.276, 841.890)


def blank_pdf(path: Path, pages: int = 1) -> Path:
    """``pages`` textless A4 pages — the scan/image-only PDF a probe must reject."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=A4_PORTRAIT[0], height=A4_PORTRAIT[1])
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def text_pdf(path: Path, pages: int, chars_per_page: int = 300) -> Path:
    """Minimal born-digital PDF: ``pages`` pages of extractable ASCII text.

    Built as raw bytes with a hand-written xref so nothing but the format itself
    is under test: pypdf's writer emits an object stream the naive parsers in
    these tests do not claim to read.
    """
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(pages))
    objs: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
    }
    for i in range(pages):
        text = ("lorem ipsum dolor sit amet " * ((chars_per_page // 27) + 1))[:chars_per_page]
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
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
