"""CELS Stage 5 mechanism tests: metrics, fit search, and stream surgery.

These run against a synthetic page built here, so they pin the *arithmetic* and
the *operator rewriting* without a corpus. The end-to-end geometry claims
(extraction, raster diff, TL twin) are the spike script's job — see
``output/adaptive_delivery/g3`` — because they need a real producer's fonts.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pikepdf
import pytest

from ubt.adapters.pdf.cels_inject import (
    FontMetrics,
    HorizontalFit,
    InjectionUnsupported,
    advance_width,
    encode_probe,
    inject_line,
    replace_stream,
    scan_lines,
    solve_horizontal,
)

pytestmark = pytest.mark.fast

#: Every code in the synthetic font advances 0.6 em, so an expected width is
#: ``0.6 · size · glyph_count`` and a wrong formula cannot hide behind real
#: font metrics.
_SYNTHETIC_ADVANCE_EM: float = 0.6


def _font(widths: list[float] | None = None) -> pikepdf.Dictionary:
    entries: dict[str, Any] = {
        "/Type": pikepdf.Name("/Font"),
        "/Subtype": pikepdf.Name("/Type1"),
        "/BaseFont": pikepdf.Name("/Helvetica"),
        "/FirstChar": 32,
        "/LastChar": 126,
    }
    if widths is not None:
        entries["/Widths"] = pikepdf.Array(widths)
    return pikepdf.Dictionary(entries)


def _build_page(
    instructions: list[tuple[list[Any], str]] | None = None,
    *,
    font: pikepdf.Dictionary | None = None,
) -> tuple[pikepdf.Pdf, pikepdf.Page]:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(400, 300))
    chosen = font if font is not None else _font([_SYNTHETIC_ADVANCE_EM * 1000.0] * 95)
    page.Resources = pikepdf.Dictionary({"/Font": pikepdf.Dictionary({"/F1": chosen})})
    body = instructions if instructions is not None else _two_line_body()
    stream = pikepdf.unparse_content_stream(
        [(operands, pikepdf.Operator(op)) for operands, op in body]
    )
    page.Contents = pdf.make_stream(stream)
    return pdf, page


def _two_line_body() -> list[tuple[list[Any], str]]:
    return [
        ([], "BT"),
        (["/F1", 10], "Tf"),
        ([1, 0, 0, 1, 50, 200], "Tm"),
        ([pikepdf.String(b"Hello")], "Tj"),
        ([pikepdf.String(b"World")], "Tj"),
        ([10], "TL"),
        ([], "T*"),
        ([pikepdf.String(b"Second")], "Tj"),
        ([], "ET"),
    ]


def _op_names(pdf: pikepdf.Pdf, data: bytes) -> list[str]:
    return [str(operator) for _operands, operator in _parse(pdf, data)]


def _numeric_operands(pdf: pikepdf.Pdf, data: bytes, names: tuple[str, ...]) -> list[float]:
    return [
        float(operands[0]) for operands, operator in _parse(pdf, data) if str(operator) in names
    ]


def _parse(pdf: pikepdf.Pdf, data: bytes) -> list[Any]:
    """Parse rewritten stream bytes back into instructions."""
    return list(pikepdf.parse_content_stream(pdf.make_stream(data)))


def test_advance_width_follows_the_pdf_displacement_formula() -> None:
    metrics = FontMetrics(first_char=32, widths=(600.0,) * 95)

    plain = advance_width("AAAAA", metrics, 10.0)
    tracked = advance_width("AAAAA", metrics, 10.0, char_spacing=2.0, word_spacing=3.0)
    scaled = advance_width("A A", metrics, 10.0, horizontal_scale=0.9, word_spacing=3.0)

    assert plain == pytest.approx(30.0)
    # Tc applies to every glyph; Tw only to the space, and there is none here.
    assert tracked == pytest.approx(30.0 + 5 * 2.0)
    # Tz multiplies both the glyph advances and the word spacing (spec Th).
    assert scaled == pytest.approx((18.0 + 3.0) * 0.9)


def test_scan_lines_groups_shows_sharing_a_baseline() -> None:
    _pdf, page = _build_page()

    lines = scan_lines(page)

    assert len(lines) == 2, "T* must start a second line"
    first = lines[0]
    assert first.text == "HelloWorld"
    assert first.x0 == pytest.approx(50.0)
    assert first.baseline == pytest.approx(200.0)
    assert first.width == pytest.approx(60.0), "10 glyphs × 0.6 em × 10pt"
    assert first.font_size == pytest.approx(10.0)
    second = lines[1]
    assert second.text == "Second"
    assert second.baseline == pytest.approx(190.0), "T* moved down by TL=10"


def test_scan_lines_carries_tz_and_tc_for_the_rewrite_to_restore() -> None:
    body: list[tuple[list[Any], str]] = [
        ([], "BT"),
        (["/F1", 10], "Tf"),
        ([90], "Tz"),
        ([1.5], "Tc"),
        ([1, 0, 0, 1, 50, 200], "Tm"),
        ([pikepdf.String(b"Hello")], "Tj"),
        ([], "ET"),
    ]
    _pdf, page = _build_page(body)

    line = scan_lines(page)[0]

    assert line.horizontal_scale == pytest.approx(0.9)
    assert line.char_spacing == pytest.approx(1.5)
    # width = 0.9 · (5 · 6 + 5 · 1.5) = 33.75
    assert line.width == pytest.approx(33.75)


def test_tj_kerning_adjustments_are_in_the_measured_width() -> None:
    body: list[tuple[list[Any], str]] = [
        ([], "BT"),
        (["/F1", 10], "Tf"),
        ([1, 0, 0, 1, 50, 200], "Tm"),
        ([pikepdf.Array([pikepdf.String(b"AB"), 500, pikepdf.String(b"CD")])], "TJ"),
        ([], "ET"),
    ]
    _pdf, page = _build_page(body)

    line = scan_lines(page)[0]

    # 4 glyphs × 6pt; a positive TJ number moves the pen back (tightens).
    assert line.width == pytest.approx(24.0 - 5.0)


def test_solve_horizontal_spends_tracking_before_scaling() -> None:
    metrics = FontMetrics(first_char=32, widths=(600.0,) * 95)

    fit = solve_horizontal("AAAAA", metrics, 10.0, available_pt=27.0)

    assert fit.fits
    assert fit.horizontal_scale == pytest.approx(1.0), "Tz must not be spent while Tc can pay"
    assert fit.char_spacing == pytest.approx((27.0 - 30.0) / 5.0)
    assert fit.width_pt <= 27.0


def test_solve_horizontal_reaches_for_scaling_after_the_tracking_budget() -> None:
    metrics = FontMetrics(first_char=32, widths=(600.0,) * 95)

    fit = solve_horizontal("AAAAA", metrics, 10.0, available_pt=25.0, max_tracking_em=0.04)

    assert fit.fits
    assert fit.char_spacing >= -0.4 - 1e-9, "tracking may not go past its 0.04 em floor"
    assert fit.char_spacing < 0.0, "the line did need compression"
    assert fit.horizontal_scale < 1.0
    assert fit.width_pt <= 25.0


def test_solve_horizontal_reports_a_deficit_instead_of_truncating() -> None:
    metrics = FontMetrics(first_char=32, widths=(600.0,) * 95)

    fit = solve_horizontal("AAAAAAAAAA", metrics, 10.0, available_pt=40.0)

    assert not fit.fits, "60pt of text cannot fit 40pt inside the bounds"
    assert fit.deficit_pt > 0.0
    assert fit.width_pt > 40.0, "the reported width must be the honest one"


def test_inject_line_replaces_the_run_and_restores_the_metrics() -> None:
    pdf, page = _build_page()
    line = scan_lines(page)[0]

    stream = inject_line(page, line, "Replacement", horizontal_scale=0.96, char_spacing=-0.3)

    assert _op_names(pdf, stream) == [
        "BT",
        "Tf",
        "Tm",
        "Tz",
        "Tc",
        "Tj",
        "Tz",
        "Tc",
        "TL",
        "T*",
        "Tj",
        "ET",
    ]
    assert _numeric_operands(pdf, stream, ("Tz",)) == pytest.approx([96.0, 100.0])
    assert _numeric_operands(pdf, stream, ("Tc",)) == pytest.approx([-0.3, 0.0])


def test_inject_line_restores_the_values_that_were_in_force() -> None:
    body: list[tuple[list[Any], str]] = [
        ([], "BT"),
        (["/F1", 10], "Tf"),
        ([85], "Tz"),
        ([0.5], "Tc"),
        ([1, 0, 0, 1, 50, 200], "Tm"),
        ([pikepdf.String(b"Hello")], "Tj"),
        ([], "ET"),
    ]
    pdf, page = _build_page(body)
    line = scan_lines(page)[0]

    stream = inject_line(page, line, "Replacement", horizontal_scale=1.0, char_spacing=0.0)

    assert _numeric_operands(pdf, stream, ("Tz",)) == pytest.approx([85.0, 100.0, 85.0])
    assert _numeric_operands(pdf, stream, ("Tc",)) == pytest.approx([0.5, 0.0, 0.5])


def test_inject_line_emits_the_leading_probe_after_the_show() -> None:
    pdf, page = _build_page()
    line = scan_lines(page)[0]

    stream = inject_line(page, line, "Probe", leading_probe=("Probe", 12.0))

    names = _op_names(pdf, stream)
    first_show = names.index("Tj")
    assert names[first_show + 1 : first_show + 4] == ["TL", "T*", "Tj"]
    assert names.count("Tj") == 3, "the probe twin plus the untouched second line"
    assert 12.0 in _numeric_operands(pdf, stream, ("TL",))


def test_replace_stream_installs_the_rewritten_bytes() -> None:
    pdf, page = _build_page()
    line = scan_lines(page)[0]

    replace_stream(page, inject_line(page, line, "Replaced"), pdf)

    installed = bytes(pdf.pages[0].Contents.read_bytes())
    assert b"Replaced" in installed
    assert _op_names(pdf, installed)[0] == "BT"


def test_composite_fonts_are_refused_rather_than_guessed() -> None:
    composite = pikepdf.Dictionary(
        {
            "/Type": pikepdf.Name("/Font"),
            "/Subtype": pikepdf.Name("/Type0"),
            "/BaseFont": pikepdf.Name("/SomeCJK"),
        }
    )

    with pytest.raises(InjectionUnsupported, match="Type0"):
        FontMetrics.from_font(composite)


def test_font_without_widths_is_refused() -> None:
    with pytest.raises(InjectionUnsupported, match="/Widths"):
        FontMetrics.from_font(_font())


def test_cjk_probe_is_refused_with_the_coverage_reason() -> None:
    with pytest.raises(InjectionUnsupported, match="embedded font"):
        encode_probe("自适应文本")


def test_non_printable_ascii_is_refused() -> None:
    with pytest.raises(InjectionUnsupported, match="printable ASCII"):
        encode_probe("tab\there")


def test_scan_reports_a_missing_font_resource() -> None:
    body: list[tuple[list[Any], str]] = [
        ([], "BT"),
        (["/Missing", 10], "Tf"),
        ([1, 0, 0, 1, 50, 200], "Tm"),
        ([pikepdf.String(b"Hello")], "Tj"),
        ([], "ET"),
    ]
    _pdf, page = _build_page(body)

    with pytest.raises(InjectionUnsupported, match="not found"):
        scan_lines(page)


def test_text_space_operands_are_converted_to_device_space() -> None:
    """``Tf 1`` + ``Tm 9.96``: Tc and TL operands are text-space, not device.

    The corpus writes the real size into ``Tm`` and leaves ``Tf`` at 1, so a
    ``Tc`` computed in device points has to be divided by the matrix scale or
    the line silently compresses (or explodes) by 9.96×. This is the defect G3
    found; the conversion is what fixes it.
    """
    body: list[tuple[list[Any], str]] = [
        ([], "BT"),
        (["/F1", 1], "Tf"),
        ([9.96, 0, 0, 9.96, 121.55, 428.32], "Tm"),
        ([pikepdf.String(b"Hello")], "Tj"),
        ([], "ET"),
    ]
    pdf, page = _build_page(body)
    line = scan_lines(page)[0]

    assert line.text_scale == pytest.approx(9.96)
    assert line.font_size == pytest.approx(9.96)

    stream = inject_line(
        page,
        line,
        "Compressed",
        char_spacing=-0.996,
        leading_probe=("Compressed", 11.952),
    )

    assert _numeric_operands(pdf, stream, ("Tc",))[0] == pytest.approx(-0.1)
    leading = _numeric_operands(pdf, stream, ("TL",))[0]
    assert leading == pytest.approx(1.2), "11.952pt device leading at a 9.96× scale"


def test_fit_evidence_stays_asdict_able() -> None:
    """The spike serialises ``HorizontalFit`` straight into ``evidence.json``."""
    assert asdict(HorizontalFit(1.0, 0.0, 10.0, True, 0.0)) == {
        "horizontal_scale": 1.0,
        "char_spacing": 0.0,
        "width_pt": 10.0,
        "fits": True,
        "deficit_pt": 0.0,
    }
