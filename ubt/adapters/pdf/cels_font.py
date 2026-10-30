"""CELS Stage 5 font pipeline: embed a subsetted target font with /ToUnicode.

Why this exists (AD-018)
------------------------
Measured on ``chapter-3.pdf``: p9 carries 17 font resources of which 12 have no
``/ToUnicode``, p12 carries 10 of which 7 have none, and every one is a
single-byte ``Type1`` subset with a ``MacRomanEncoding``/``/Differences``
encoding. A CJK translation cannot be written with those fonts at all — they
hold Latin glyphs and their code space has no room for CJK. CELS therefore has
to *bring* the font: a subsetted TrueType/OpenType with

* ``/Type0`` + ``/Identity-H`` encoding, so a 2-byte code is a CID,
* ``/CIDToGIDMap /Identity``, so the CID doubles as the glyph index,
* ``/W`` widths taken from the font's own ``hmtx``, and
* a generated ``/ToUnicode`` CMap, without which the text is neither searchable
  nor copyable — which the delivery contract treats as a defect, not a nicety.

Scope: one font, one subset per document, upright horizontal text. Vertical
writing, ``GSUB`` shaping (ligatures, Indic reordering) and CID-keyed CFF are
out of scope for the G3 spike; a font that needs them is refused rather than
embedded in a way that silently drops the mapping.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pikepdf

from ubt.adapters.pdf.cels_inject import (
    DEFAULT_MAX_SCALE_LOSS,
    DEFAULT_MAX_TRACKING_EM,
    HorizontalFit,
    LineSite,
)
from ubt.adapters.pdf.strip_pdf_text import cast_instructions

#: Glyph space of a PDF font: 1/1000 of an em.
_EM = 1000.0


class FontPipelineError(RuntimeError):
    """The target font cannot be embedded with this pipeline's scope."""


@dataclass(frozen=True, slots=True)
class CidFont:
    """A font embedded in this PDF plus the mapping the content stream needs."""

    #: ``/Font`` resource name (``/UBTSUB+CelsCJK``), already installed.
    resource: str
    base_font: str
    #: character → glyph index. ``/CIDToGIDMap /Identity`` makes CID == GID.
    gids: dict[str, int]
    #: character → advance width in 1/1000 em (the same units ``/W`` uses).
    widths: dict[str, int]

    def supports(self, text: str) -> bool:
        return all(char in self.gids for char in text)

    def missing(self, text: str) -> list[str]:
        return [char for char in text if char not in self.gids]

    def encode(self, text: str) -> bytes:
        """The Identity-H string: two bytes per glyph, big-endian."""
        if not self.supports(text):
            raise FontPipelineError(
                f"font {self.base_font} has no glyph for {self.missing(text)!r}; "
                "the subset was built from a different text"
            )
        return b"".join(self.gids[char].to_bytes(2, "big") for char in text)

    def advance_pt(self, text: str, size_pt: float) -> float:
        """Advance width of ``text`` at ``size_pt``, in points."""
        return sum(self.widths[char] for char in text) / _EM * size_pt


def embed_subset_font(
    pdf: pikepdf.Pdf,
    *,
    font_path: Path,
    text: str,
    resource_name: str = "/UBTSUB+CelsCJK",
    font_index: int = 0,
) -> CidFont:
    """Subset ``font_path`` to ``text`` and embed it in ``pdf``.

    Returns the mapping needed to write the string; the caller still has to
    attach the font to a page's ``/Resources /Font`` dictionary
    (:func:`install_font`), because a document may share one subset across many
    pages and only the caller knows which.
    """
    try:
        # fontTools ships no py.typed marker; the subset API used here is stable.
        from fontTools import subset  # type: ignore[import-untyped]
        from fontTools.ttLib import TTFont  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - the [cels] extra
        raise FontPipelineError(
            "fontTools is not installed; the font pipeline lives in the [cels] extra"
        ) from exc

    if not text:
        raise FontPipelineError("cannot subset a font to an empty string")
    if not font_path.exists():
        raise FontPipelineError(f"font file not found: {font_path}")

    try:
        font = TTFont(str(font_path), fontNumber=font_index, lazy=False)
    except Exception as exc:
        raise FontPipelineError(f"cannot read font {font_path}: {exc}") from exc
    if "glyf" not in font:
        raise FontPipelineError(
            f"{font_path.name} has no 'glyf' table (CFF outlines): the G3 spike "
            "embeds TrueType subsets only"
        )

    cmap = font.getBestCmap()
    missing = sorted({char for char in text if ord(char) not in cmap})
    if missing:
        raise FontPipelineError(
            f"{font_path.name} has no glyph for {missing!r}; pick a font that covers "
            "the target script"
        )

    options = subset.Options()
    # No shaping tables: Identity-H with one code per glyph is a straight
    # left-to-right run, and keeping GSUB/GPOS would only bloat the subset.
    options.layout_features = []
    options.drop_tables += ["EBDT", "EBLC", "EBSC", "SVG ", "COLR", "CPAL"]
    options.notdef_outline = True
    options.recalc_bounds = True
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(unicodes=[ord(char) for char in text])
    subsetter.subset(font)

    upem = float(font["head"].unitsPerEm)
    subset_cmap = font.getBestCmap()
    glyph_ids: dict[str, int] = {}
    widths: dict[str, int] = {}
    for char in dict.fromkeys(text):
        glyph_name = subset_cmap.get(ord(char))
        if glyph_name is None:  # pragma: no cover - guarded by the cmap check above
            raise FontPipelineError(f"subset lost the glyph for {char!r}")
        gid = int(font.getGlyphID(glyph_name))
        glyph_ids[char] = gid
        widths[char] = int(round(font["hmtx"][glyph_name][0] * _EM / upem))

    buffer = io.BytesIO()
    font.save(buffer)

    base_font = resource_name.lstrip("/")
    if not base_font:
        raise FontPipelineError("resource_name must name a font")
    head = font["head"]
    bbox = [
        int(round(head.xMin * _EM / upem)),
        int(round(head.yMin * _EM / upem)),
        int(round(head.xMax * _EM / upem)),
        int(round(head.yMax * _EM / upem)),
    ]
    ascent = int(round(font["hhea"].ascent * _EM / upem))
    descent = int(round(font["hhea"].descent * _EM / upem))
    cap_height = ascent
    if "OS/2" in font:
        cap_height = int(round(font["OS/2"].sCapHeight * _EM / upem)) or ascent

    descriptor = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/FontDescriptor"),
                "/FontName": pikepdf.Name(f"/{base_font}"),
                # Symbolic: the font's own cmap is irrelevant to the reader; the
                # /ToUnicode below is what makes the text extractable.
                "/Flags": 4,
                "/FontBBox": pikepdf.Array(bbox),
                "/ItalicAngle": 0,
                "/Ascent": ascent,
                "/Descent": descent,
                "/CapHeight": cap_height,
                "/StemV": 80,
                "/FontFile2": pikepdf.Stream(pdf, buffer.getvalue()),
            }
        )
    )
    widths_array = pikepdf.Array()
    for char in dict.fromkeys(text):
        widths_array.append(glyph_ids[char])
        widths_array.append(pikepdf.Array([widths[char]]))
    cid_font = pikepdf.Dictionary(
        {
            "/Type": pikepdf.Name("/Font"),
            "/Subtype": pikepdf.Name("/CIDFontType2"),
            "/BaseFont": pikepdf.Name(f"/{base_font}"),
            "/CIDSystemInfo": pikepdf.Dictionary(
                {
                    "/Registry": pikepdf.String("Adobe"),
                    "/Ordering": pikepdf.String("Identity"),
                    "/Supplement": 0,
                }
            ),
            "/FontDescriptor": descriptor,
            "/DW": _EM,
            "/W": widths_array,
            "/CIDToGIDMap": pikepdf.Name("/Identity"),
        }
    )
    # A font dictionary that carries streams (``/FontFile2``, ``/ToUnicode``)
    # must be an *indirect* object: a direct dictionary holding a stream is
    # invalid PDF, and PDFium stopped resolving the CMap from one — the glyphs
    # were drawn but the text layer came back empty (measured on this spike's
    # first run).
    type0 = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type0"),
                "/BaseFont": pikepdf.Name(f"/{base_font}"),
                "/Encoding": pikepdf.Name("/Identity-H"),
                "/DescendantFonts": pikepdf.Array([cid_font]),
                "/ToUnicode": pikepdf.Stream(pdf, _to_unicode_cmap(glyph_ids).encode("ascii")),
            }
        )
    )
    registry = pdf.Root.get("/UBTEmbeddedFonts")
    if registry is None:
        registry = pikepdf.Dictionary()
        pdf.Root[pikepdf.Name("/UBTEmbeddedFonts")] = registry
    registry[pikepdf.Name(f"/{base_font}")] = type0
    return CidFont(
        resource=f"/{base_font}",
        base_font=base_font,
        gids=glyph_ids,
        widths=widths,
    )


def install_font(page: pikepdf.Page, font: CidFont, pdf: pikepdf.Pdf) -> None:
    """Attach the embedded font to ``page``'s ``/Resources /Font`` dictionary.

    An indirect copy is used (``pdf.copy_foreign`` is unnecessary here because
    the font lives in the same document) so a shared subset is written once.
    """
    registry = pdf.Root.get("/UBTEmbeddedFonts")
    if registry is None:  # pragma: no cover - embed_subset_font creates it
        raise FontPipelineError(f"font {font.base_font} was never embedded in this document")
    stored = registry.get(f"/{font.base_font}")
    if stored is None:  # pragma: no cover - the registry key is written above
        raise FontPipelineError(f"font {font.base_font} is missing from the font registry")
    resources = page.Resources
    if resources is None:  # pragma: no cover - pages always carry one here
        resources = pikepdf.Dictionary()
        page.Resources = resources
    fonts = resources.get("/Font")
    if fonts is None:
        fonts = pikepdf.Dictionary()
        resources[pikepdf.Name("/Font")] = fonts
    fonts[pikepdf.Name(f"/{font.base_font}")] = stored


def solve_cid_fit(
    font: CidFont,
    text: str,
    size_pt: float,
    available_pt: float,
    *,
    max_tracking_em: float = DEFAULT_MAX_TRACKING_EM,
    max_scale_loss: float = DEFAULT_MAX_SCALE_LOSS,
) -> HorizontalFit:
    """Fit ``text`` into ``available_pt`` by tracking first, then scaling.

    The same rule as :func:`ubt.adapters.pdf.cels_inject.solve_horizontal`, but
    the advances come from the embedded font's ``/W`` rather than a single-byte
    ``/Widths`` table: a CJK glyph is a 2-byte code with its own width.
    """
    natural = font.advance_pt(text, size_pt)
    count = len(text)
    if count == 0:
        return HorizontalFit(1.0, 0.0, 0.0, True, 0.0)
    if natural <= available_pt:
        return HorizontalFit(1.0, 0.0, natural, True, 0.0)
    tracking_floor = -max_tracking_em * size_pt
    steps = max(1, int(round(max_scale_loss / 0.005)))
    for step in range(steps + 1):
        scale = 1.0 - step * 0.005
        needed = (available_pt / scale - natural) / count
        if needed < tracking_floor:
            continue
        width = scale * (natural + count * needed)
        return HorizontalFit(round(scale, 4), round(needed, 4), round(width, 3), True)
    scale = 1.0 - max_scale_loss
    width = scale * (natural + count * tracking_floor)
    return HorizontalFit(
        horizontal_scale=round(scale, 4),
        char_spacing=round(tracking_floor, 4),
        width_pt=round(width, 3),
        fits=False,
        deficit_pt=round(width - available_pt, 3),
    )


def inject_cid_line(
    page: pikepdf.Page,
    line: LineSite,
    text: str,
    *,
    font: CidFont,
    size_pt: float,
    horizontal_scale: float = 1.0,
    char_spacing: float = 0.0,
) -> bytes:
    """Replace ``line``'s glyph run with ``text`` set in the embedded font.

    ``Tf`` switches to the embedded resource for the run and back to the page's
    own font afterwards, so a following line is unaffected. ``Tz``/``Tc`` are
    the device-space values ``solve_cid_fit`` produced; ``Tc`` is written in
    text space (divided by the matrix scale) because ``chapter-3.pdf`` carries
    its size in ``Tm`` and sets ``Tf`` to 1.
    """
    if not font.supports(text):
        raise FontPipelineError(f"font cannot encode {font.missing(text)!r}")
    instructions = cast_instructions(pikepdf.parse_content_stream(page))
    scale = line.text_scale if line.text_scale > 0.0 else 1.0
    previous_size = round(line.font_size / scale, 4) if line.font_size else 1.0
    replacement: list[Any] = [
        ([round(horizontal_scale * 100.0, 4)], pikepdf.Operator("Tz")),
        ([round(char_spacing / scale, 4)], pikepdf.Operator("Tc")),
        ([pikepdf.Name(font.resource), round(size_pt / scale, 4)], pikepdf.Operator("Tf")),
        ([pikepdf.String(font.encode(text))], pikepdf.Operator("Tj")),
        ([pikepdf.Name(line.font_key), previous_size], pikepdf.Operator("Tf")),
        ([round(line.horizontal_scale * 100.0, 4)], pikepdf.Operator("Tz")),
        ([round(line.char_spacing, 4)], pikepdf.Operator("Tc")),
    ]
    merged = instructions[: line.first_index] + replacement + instructions[line.last_index + 1 :]
    return pikepdf.unparse_content_stream(merged)


def _to_unicode_cmap(glyph_ids: dict[str, int]) -> str:
    """A ``bfchar`` CMap from glyph index to Unicode.

    ``/Identity-H`` alone gives a PDF that *draws* the right glyphs but whose
    text layer is garbage (each 2-byte code is a private glyph id). The CMap is
    what makes copy/search work, which is why the delivery contract fails an
    artifact without it.
    """
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<0000> <FFFF>",
        "endcodespacerange",
    ]
    entries = sorted(glyph_ids.items(), key=lambda item: item[1])
    for start in range(0, len(entries), 100):
        chunk = entries[start : start + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        for char, gid in chunk:
            code = (
                f"{ord(char):04X}"
                if len(char) == 1
                else "".join(f"{ord(part):04X}" for part in char)
            )
            lines.append(f"<{gid:04X}> <{code}>")
        lines.append("endbfchar")
    lines.extend(
        [
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "CidFont",
    "FontPipelineError",
    "embed_subset_font",
    "inject_cid_line",
    "install_font",
    "solve_cid_fit",
]
