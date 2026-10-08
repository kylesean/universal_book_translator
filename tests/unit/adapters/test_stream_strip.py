"""The source-text strip's font-metric parsing (CID /W arrays).

The strip sizes its erase rects from real glyph advances when a font's widths
resolve. A CID /W array whose range entries use the *flat* spelling
(``first last w`` as three consecutive numbers — the form the PDF spec's own
example uses) registered nothing, so every CID it covered fell back to /DW:
at the default 1000 em the measured line drifts well past the erase rect and a
trailing colored citation survives the strip.
"""

from __future__ import annotations

import pikepdf
import pytest

from ubt.adapters.pdf.stream_strip import FontAdvance, _advance_from_widths, _parse_cid_widths

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
