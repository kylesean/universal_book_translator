"""The source-text strip's font-metric parsing (CID /W arrays).

The strip sizes its erase rects from real glyph advances when a font's widths
resolve. A CID /W array whose range entries use the *flat* spelling
(``first last w`` as three consecutive numbers — the form the PDF spec's own
example uses) registered nothing, so every CID it covered fell back to /DW:
at the default 1000 em the measured line drifts well past the erase rect and a
trailing colored citation survives the strip.

The show operand reaches those widths as *bytes*, never as decoded text: the
bytes are the glyph codes the /Widths table is indexed by. Decoding them through
PDFDocEncoding and re-encoding to latin-1 replaced every byte the round trip
could not represent — the whole 0x80-0x9F range, where an Identity-H font's CJK
codes live — with ``?``, so the measured advance was wrong by every CJK glyph on
the line.
"""

from __future__ import annotations

import pikepdf
import pytest

from ubt.adapters.pdf.stream_strip import (
    FontAdvance,
    _advance_from_widths,
    _as_bytes,
    _parse_cid_widths,
    _show_glyph_bytes,
)

pytestmark = pytest.mark.fast


def _widths(*entries: object) -> dict[int, float]:
    parsed: dict[int, float] = {}
    _parse_cid_widths(pikepdf.Array(list(entries)), parsed)
    return parsed


def test_a_flat_range_entry_registers_every_cid() -> None:
    # PDF 32000-1 §9.7.4.3's example spelling: 709 711 500.
    assert _widths(120, pikepdf.Array([400]), 709, 711, 500) == {
        120: 400.0,
        709: 500.0,
        710: 500.0,
        711: 500.0,
    }


def test_a_nested_range_entry_still_registers() -> None:
    # The other spec spelling: [first last w] as one nested array.
    assert _widths(pikepdf.Array([20, 22, 600])) == {20: 600.0, 21: 600.0, 22: 600.0}


def test_run_and_flat_entries_mix() -> None:
    assert _widths(1, pikepdf.Array([200]), 5, 7, 500) == {
        1: 200.0,
        5: 500.0,
        6: 500.0,
        7: 500.0,
    }


def test_a_flat_range_measures_the_run_it_covers() -> None:
    # The point of parsing /W at all: two glyphs at 250 em each advance 0.5 em,
    # not the 2.0 em the /DW=1000 fallback would report.
    fa = FontAdvance(code_bytes=2, widths=_widths(9, 10, 250), default=1000.0)
    assert _advance_from_widths(b"\x00\x09\x00\x0a", fa) == pytest.approx(0.5)
    # Without the flat parse the same bytes read as 2.0 em (fallback to /DW).
    fallback = FontAdvance(code_bytes=2, widths={}, default=1000.0)
    assert _advance_from_widths(b"\x00\x09\x00\x0a", fallback) == pytest.approx(2.0)


def test_a_pdf_string_operand_keeps_its_bytes() -> None:
    # Every one of these bytes decoded to a distinct latin-1/PDFDocEncoding
    # character, which ``str()`` then could not re-encode to latin-1: the old
    # ``str(val).encode("latin-1", "replace")`` returned 32 ``?`` bytes here.
    raw = bytes(range(0x80, 0xA0))
    assert _as_bytes(pikepdf.String(raw)) == raw
    assert _as_bytes(pikepdf.String(b"\x00\x41\x80")) == b"\x00\x41\x80"
    # A plain Python str has no raw bytes to preserve; the latin-1 round trip is
    # still the best available and stays for that case.
    assert _as_bytes("ab") == b"ab"


def test_the_measured_advance_uses_the_cid_codes_not_question_marks() -> None:
    # The regression in one assertion: a CJK line whose widths are only known
    # for the real codes. Decoded-and-replaced, every code became 0x3F and the
    # whole line fell back to /DW.
    fa = FontAdvance(code_bytes=2, widths={0x8140: 500.0, 0x8141: 500.0}, default=1000.0)
    show = pikepdf.Array([pikepdf.String(b"\x81\x40\x81\x41")])
    assert _show_glyph_bytes([show]) == b"\x81\x40\x81\x41"
    assert _advance_from_widths(_show_glyph_bytes([show]), fa) == pytest.approx(1.0)
