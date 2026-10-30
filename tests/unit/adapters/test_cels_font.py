"""The Stage 5 font pipeline: Type0 + Identity-H + a real /ToUnicode.

AD-018 is why these exist: the corpus fonts are single-byte Type1 with no
``/ToUnicode``, so a CJK translation has to bring its own subsetted font. The
tests use whatever CJK TrueType face the machine has (the pipeline refuses CFF),
and skip when there is none — the structural assertions are what pin the PDF
contract, and the extraction round-trip lives in ``scripts/cels_cjk_spike.py``
because it needs the whole render stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pikepdf
import pytest

from ubt.adapters.pdf.cels_font import (
    CidFont,
    FontPipelineError,
    embed_subset_font,
    inject_cid_line,
    install_font,
    solve_cid_fit,
)
from ubt.adapters.pdf.cels_inject import scan_lines

pytestmark = pytest.mark.fast

_CJK_CANDIDATES = (
    "/usr/share/fonts/MapleMono-NF-CN-unhinted/MapleMono-NF-CN-Regular.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)


def _cjk_font() -> Path:
    for candidate in _CJK_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    pytest.skip("no CJK TrueType font installed; the pipeline refuses CFF outlines")


def _page(pdf: pikepdf.Pdf) -> pikepdf.Page:
    page = pdf.add_blank_page(page_size=(300, 200))
    page.Resources = pikepdf.Dictionary({"/Font": pikepdf.Dictionary({})})
    return page


def test_embed_subset_font_builds_a_type0_with_tounicode(tmp_path: Path) -> None:
    text = "自适应交付探针"
    path = tmp_path / "cjk.pdf"
    with pikepdf.Pdf.new() as pdf:
        page = _page(pdf)
        font = embed_subset_font(pdf, font_path=_cjk_font(), text=text)
        install_font(page, font, pdf)
        pdf.save(path)

    assert isinstance(font, CidFont)
    assert set(font.gids) == set(text)
    assert all(0 <= gid <= 0xFFFF for gid in font.gids.values())
    encoded = font.encode(text)
    assert len(encoded) == 2 * len(text), "Identity-H writes two bytes per glyph"
    assert encoded[:2] == font.gids[text[0]].to_bytes(2, "big")

    with pikepdf.open(path) as pdf:
        installed = pdf.pages[0].Resources["/Font"][font.resource]
        assert installed.is_indirect, "a font holding streams must be an indirect object"
        assert str(installed.get("/Subtype")) == "/Type0"
        assert str(installed.get("/Encoding")) == "/Identity-H"
        assert installed.get("/ToUnicode") is not None
        descendant = installed["/DescendantFonts"][0]
        assert str(descendant.get("/Subtype")) == "/CIDFontType2"
        assert str(descendant.get("/CIDToGIDMap")) == "/Identity"
        assert descendant["/FontDescriptor"].get("/FontFile2") is not None
        cmap = bytes(installed["/ToUnicode"].read_bytes()).decode("ascii")
        assert "beginbfchar" in cmap
        # The CMap maps CIDs to Unicode, which is what makes copy/search work.
        assert f"<{font.gids[text[0]]:04X}>" in cmap


def test_encode_refuses_characters_the_subset_does_not_hold() -> None:
    with pikepdf.Pdf.new() as pdf:
        font = embed_subset_font(pdf, font_path=_cjk_font(), text="自适应")

        assert not font.supports("测试")
        assert font.missing("测试自") == ["测", "试"]
        assert font.advance_pt("自适应", 10.0) > 0
        with pytest.raises(FontPipelineError, match="no glyph"):
            font.encode("测试")


def test_solve_cid_fit_spends_tracking_before_scaling() -> None:
    with pikepdf.Pdf.new() as pdf:
        font = embed_subset_font(pdf, font_path=_cjk_font(), text="自适应")
    natural = font.advance_pt("自适应", 10.0)

    fit = solve_cid_fit(font, "自适应", 10.0, natural * 0.95)

    assert fit.fits
    assert fit.horizontal_scale == pytest.approx(1.0)
    assert fit.char_spacing < 0.0
    assert fit.width_pt <= natural * 0.95 + 1e-6


def test_inject_cid_line_switches_font_and_restores_it() -> None:
    text = "自适应"
    with pikepdf.Pdf.new() as pdf:
        page = _page(pdf)
        font = embed_subset_font(pdf, font_path=_cjk_font(), text=text)
        install_font(page, font, pdf)
        page.Resources["/Font"][pikepdf.Name("/F1")] = pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name("/Helvetica"),
                "/Encoding": pikepdf.Name("/WinAnsiEncoding"),
                "/FirstChar": 32,
                "/LastChar": 126,
                "/Widths": pikepdf.Array([600.0] * 95),
            }
        )
        page.Contents = pdf.make_stream(b"BT /F1 9.9626 Tf 1 0 0 1 50 200 Tm (Hello) Tj ET")

        line = scan_lines(page)[0]
        merged = inject_cid_line(page, line, text, font=font, size_pt=line.font_size)
        names = [
            str(instruction[1])
            for instruction in cast(Any, pikepdf.parse_content_stream(pdf.make_stream(merged)))
        ]

    assert names == ["BT", "Tf", "Tm", "Tz", "Tc", "Tf", "Tj", "Tf", "Tz", "Tc", "ET"]
    assert names.count("Tf") == 3, "switch to the embedded font, restore, and the body's own"
