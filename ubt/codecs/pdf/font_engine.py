"""Dual-format (TrueType + OpenType CFF) font subsetting & HarfBuzz shaping engine.

Upgrades the CELS TrueType-only spike into a production font pipeline for
in-place PDF rewriting:
- Supports both TrueType (``glyf`` -> ``/CIDFontType2`` + ``/FontFile2``) and
  OpenType CFF (``CFF `` -> ``/CIDFontType0`` + ``/FontFile3``), so bundled
  ``NotoSerifCJKsc-*.otf`` / ``NotoSansCJKsc-*.otf`` work directly.
- Shapes text runs with ``uharfbuzz`` for exact advance widths, kerning, and
  multi-script fallback (CJK + Latin + symbols) with zero tofu.
- Emits chunked UTF-16BE ``/ToUnicode`` CMaps (<=100 entries per ``beginbfchar``
  block per PDF 1.7 spec) so every rendered PDF is 100% searchable and copyable.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pikepdf
import uharfbuzz as hb
from fontTools import subset
from fontTools.ttLib import TTFont, TTLibFileIsCollectionError


def _open_ttfont(source: Any, *, lazy: bool) -> TTFont:
    """Open a standalone TTF/OTF or face 0 of a TTC/OTC font collection."""
    try:
        return TTFont(source, lazy=lazy)
    except TTLibFileIsCollectionError:
        if hasattr(source, "seek"):
            source.seek(0)
        return TTFont(source, lazy=lazy, fontNumber=0)


logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

_EM = 1000.0
_BFCHAR_CHUNK = 100


class FontEngineError(RuntimeError):
    """Raised when a required font cannot be loaded or subsetted."""


class FontSlot(StrEnum):
    """Logical typography slots mapped to concrete font files per target language."""

    SERIF_REGULAR = "serif_regular"
    SERIF_BOLD = "serif_bold"
    SANS_REGULAR = "sans_regular"
    SANS_BOLD = "sans_bold"
    LATIN_FALLBACK = "latin_fallback"
    #: CJK-capable face used only when the target language's own slots have no
    #: glyph for a character (a Chinese name or quotation in a Spanish book, a
    #: rehearsal marker in any target). Optional: omitted when no CJK font is
    #: installed, so a Latin-only environment keeps working instead of failing.
    CJK_FALLBACK = "cjk_fallback"


@dataclass(frozen=True, slots=True)
class ShapedGlyph:
    """One shaped glyph positioned by HarfBuzz in PDF points."""

    codepoint: int  # GID in the source font
    char: str  # Original Unicode substring for this cluster
    x_advance_pt: float
    x_offset_pt: float
    y_offset_pt: float
    font_slot: FontSlot


@dataclass(frozen=True, slots=True)
class EmbeddedFont:
    """A subsetted PDF Type0/Identity-H font installed in a document."""

    slot: FontSlot
    resource_name: str  # e.g. "/UBT_F0"
    base_font: str
    pdf_font_obj: pikepdf.Object
    char_to_cid: dict[str, int]
    cid_widths: dict[int, int]
    ascent_em: float
    descent_em: float

    def encode_text(self, text: str) -> bytes:
        out = bytearray()
        for ch in text:
            cid = self.char_to_cid.get(ch, 0)
            out.extend(cid.to_bytes(2, "big"))
        return bytes(out)


class _LoadedFace:
    """In-memory HarfBuzz + cmap handle for fast shaping across pages."""

    __slots__ = (
        "path",
        "raw_bytes",
        "hb_face",
        "hb_font",
        "cmap",
        "upem",
        "ascent_em",
        "descent_em",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.raw_bytes = path.read_bytes()
        self.hb_face = hb.Face(self.raw_bytes)
        self.hb_font = hb.Font(self.hb_face)
        tt = _open_ttfont(io.BytesIO(self.raw_bytes), lazy=True)
        try:
            self.cmap: dict[int, str] = dict(tt.getBestCmap() or {})
            self.upem = float(tt["head"].unitsPerEm)
            # Prefer OpenType OS/2 typographic metrics over hhea when available:
            # CJK fonts (Noto Serif/Sans CJK) inflate hhea to usWinAscent (~1.15/-0.29)
            # for legacy line spacing while their real em-box is sTypoAscender/Descender (0.88/-0.12).
            os2 = tt.get("OS/2", None)
            if (
                os2 is not None
                and getattr(os2, "sTypoAscender", 0) > 0
                and getattr(os2, "sTypoDescender", 0) <= 0
            ):
                self.ascent_em = float(os2.sTypoAscender) / self.upem
                self.descent_em = float(os2.sTypoDescender) / self.upem
            else:
                self.ascent_em = float(tt["hhea"].ascent) / self.upem
                self.descent_em = float(tt["hhea"].descent) / self.upem
        finally:
            tt.close()

    def supports_ord(self, cp: int) -> bool:
        return cp in self.cmap


def _default_fonts_dir() -> Path:
    """Locate bundled ``fonts/`` directory relative to the repo root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "fonts"
        if candidate.is_dir():
            return candidate
    return Path("fonts")


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


def resolve_slot_paths(
    target_lang: str,
    fonts_dir: Path | None = None,
) -> dict[FontSlot, Path]:
    """Select concrete TTF/OTF paths for each :class:`FontSlot`."""
    fdir = fonts_dir or _default_fonts_dir()
    lang = (target_lang or "zh").lower().split("-")[0]

    sys_dirs = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".local/share/fonts",
    ]

    def _find(names: list[str]) -> Path | None:
        direct = _first_existing([fdir / n for n in names])
        if direct is not None:
            return direct
        for sdir in sys_dirs:
            if not sdir.is_dir():
                continue
            for name in names:
                found = next(sdir.rglob(name), None)
                if found is not None and found.is_file():
                    return found
        return None

    if lang == "ja":
        serif_reg = _find(
            [
                "NotoSerifCJKjp-Regular-subset.otf",
                "NotoSerifCJKjp-Regular.otf",
                "NotoSerifCJK-Regular.ttc",
            ]
        )
        serif_bold = _find(
            [
                "NotoSerifCJKsc-Bold-subset.otf",
                "NotoSerifCJKjp-Bold.otf",
                "NotoSansCJKsc-Bold-subset.otf",
            ]
        )
        sans_reg = _find(["NotoSansCJKsc-Regular-subset.otf", "NotoSansCJKjp-Regular.otf"])
        sans_bold = _find(["NotoSansCJKsc-Bold-subset.otf", "NotoSansCJKjp-Bold.otf"])
    elif lang == "ko":
        serif_reg = _find(
            [
                "NotoSerifCJKkr-Regular-subset.otf",
                "NotoSerifCJKkr-Regular.otf",
                "NotoSerifCJK-Regular.ttc",
            ]
        )
        serif_bold = _find(["NotoSerifCJKsc-Bold-subset.otf", "NotoSansCJKsc-Bold-subset.otf"])
        sans_reg = _find(["NotoSansCJKsc-Regular-subset.otf", "NotoSansCJKkr-Regular.otf"])
        sans_bold = _find(["NotoSansCJKsc-Bold-subset.otf", "NotoSansCJKkr-Bold.otf"])
    elif lang == "ar":
        serif_reg = _find(["NotoNaskhArabic-Regular.ttf", "NotoSerif-Regular.ttf"])
        serif_bold = serif_reg
        sans_reg = serif_reg
        sans_bold = serif_reg
    elif lang == "th":
        serif_reg = _find(["NotoSerifThai-Regular.ttf", "NotoSerif-Regular.ttf"])
        serif_bold = serif_reg
        sans_reg = serif_reg
        sans_bold = serif_reg
    elif lang in {"hi", "mr", "ne"}:
        serif_reg = _find(["NotoSerifDevanagari-Regular.ttf", "NotoSerif-Regular.ttf"])
        serif_bold = serif_reg
        sans_reg = serif_reg
        sans_bold = serif_reg
    elif lang == "he":
        serif_reg = _find(["NotoSerifHebrew-Regular.ttf", "NotoSerif-Regular.ttf"])
        serif_bold = serif_reg
        sans_reg = serif_reg
        sans_bold = serif_reg
    elif lang in {"en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru"}:
        serif_reg = _find(
            ["NotoSerif-Regular.ttf", "LiberationSerif-Regular.ttf", "DejaVuSerif.ttf"]
        )
        serif_bold = _find(
            ["NotoSerif-Bold.ttf", "LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf"]
        )
        sans_reg = _find(["NotoSans-Regular.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"])
        sans_bold = _find(["NotoSans-Bold.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"])
    else:
        # Default: Simplified Chinese (zh) + full CJK coverage
        serif_reg = _find(
            [
                "NotoSerifCJKsc-Regular-subset.otf",
                "NotoSerifCJKsc-Regular.otf",
                "NotoSansCJKsc-Regular-subset.otf",
                "NotoSerif-Regular.ttf",
            ]
        )
        serif_bold = _find(
            [
                "NotoSerifCJKsc-Bold-subset.otf",
                "NotoSansCJKsc-Bold-subset.otf",
                "NotoSerifCJKsc-Regular-subset.otf",
            ]
        )
        sans_reg = _find(
            [
                "NotoSansCJKsc-Regular-subset.otf",
                "NotoSerifCJKsc-Regular-subset.otf",
                "NotoSerif-Regular.ttf",
            ]
        )
        sans_bold = _find(
            [
                "NotoSansCJKsc-Bold-subset.otf",
                "NotoSerifCJKsc-Bold-subset.otf",
                "NotoSansCJKsc-Regular-subset.otf",
            ]
        )

    latin_fb = _find(
        [
            "NotoSerif-Regular.ttf",
            "LiberationSerif-Regular.ttf",
            "DejaVuSerif.ttf",
            "NotoSerifCJKsc-Regular-subset.otf",
        ]
    )
    # Always resolve a CJK-capable face: translated prose is not restricted to
    # the target language's script (proper nouns, quotations, mixed-script
    # technical text), and dropping those glyphs silently loses characters.
    # Prefer the simplified-Chinese subset (widest Han coverage in the bundled
    # fonts); fall back to any installed CJK face.
    cjk_fb = _find(
        [
            "NotoSerifCJKsc-Regular-subset.otf",
            "NotoSansCJKsc-Regular-subset.otf",
            "NotoSerifCJK-Regular.ttc",
            "NotoSansCJK-Regular.ttc",
            "NotoSansCJKsc-Regular.otf",
            "NotoSerifCJKsc-Regular.otf",
        ]
    )

    fallback = serif_reg or sans_reg or latin_fb or cjk_fb
    if fallback is None:
        raise FontEngineError(f"No usable font found in {fdir} or system font directories")

    slots = {
        FontSlot.SERIF_REGULAR: serif_reg or fallback,
        FontSlot.SERIF_BOLD: serif_bold or serif_reg or fallback,
        FontSlot.SANS_REGULAR: sans_reg or serif_reg or fallback,
        FontSlot.SANS_BOLD: sans_bold or sans_reg or serif_bold or fallback,
        FontSlot.LATIN_FALLBACK: latin_fb or fallback,
    }
    if cjk_fb is not None:
        slots[FontSlot.CJK_FALLBACK] = cjk_fb
    return slots


def _to_utf16_hex(cp: int) -> str:
    return chr(cp).encode("utf-16-be").hex().upper()


def _build_tounicode_cmap(cid_to_uni: dict[int, int]) -> bytes:
    """Build a PDF 1.7 compliant /ToUnicode CMap (chunked <= 100 bfchar entries)."""
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
    items = sorted(cid_to_uni.items())
    for start in range(0, len(items), _BFCHAR_CHUNK):
        chunk = items[start : start + _BFCHAR_CHUNK]
        lines.append(f"{len(chunk)} beginbfchar")
        for cid, uni in chunk:
            lines.append(f"<{cid:04X}> <{_to_utf16_hex(uni)}>")
        lines.append("endbfchar")
    lines.extend(
        [
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
    )
    return "\n".join(lines).encode("ascii")


def _extract_cid_mapping(font: TTFont) -> tuple[dict[str, int], bool]:
    """Return ``(glyph_name -> cid, is_cff)`` for a subsetted TTFont."""
    is_cff = "CFF " in font and "glyf" not in font
    glyph_order = font.getGlyphOrder()
    mapping: dict[str, int] = {}
    if is_cff:
        top = font["CFF "].cff.topDictIndex[0]
        is_cid_keyed = hasattr(top, "ROS") and top.ROS is not None
        for gid, gname in enumerate(glyph_order):
            if is_cid_keyed and gname.startswith("cid") and gname[3:].isdigit():
                mapping[gname] = int(gname[3:])
            elif is_cid_keyed and gname == ".notdef":
                mapping[gname] = 0
            else:
                mapping[gname] = gid
    else:
        for gid, gname in enumerate(glyph_order):
            mapping[gname] = gid
    return mapping, is_cff


class FontEngine:
    """Per-document font resolver, HarfBuzz measurer, and PDF Type0 subsetter."""

    def __init__(
        self,
        target_lang: str = "zh",
        *,
        fonts_dir: Path | None = None,
        slot_paths: dict[FontSlot, Path] | None = None,
    ) -> None:
        self.target_lang = target_lang
        self._slot_paths = slot_paths or resolve_slot_paths(target_lang, fonts_dir=fonts_dir)
        self._faces: dict[Path, _LoadedFace] = {}
        self._used_chars: dict[FontSlot, set[str]] = {slot: set() for slot in FontSlot}

    def _face_for(self, slot: FontSlot) -> _LoadedFace:
        path = self._slot_paths[slot]
        face = self._faces.get(path)
        if face is None:
            face = _LoadedFace(path)
            self._faces[path] = face
        return face

    def pick_slot(self, *, is_serif: bool = True, is_bold: bool = False) -> FontSlot:
        if is_serif:
            return FontSlot.SERIF_BOLD if is_bold else FontSlot.SERIF_REGULAR
        return FontSlot.SANS_BOLD if is_bold else FontSlot.SANS_REGULAR

    def metrics(self, slot: FontSlot) -> tuple[float, float]:
        """Return ``(ascent_em, descent_em)`` for ``slot``."""
        face = self._face_for(slot)
        return face.ascent_em, face.descent_em

    def resolve_char_slot(self, ch: str, preferred: FontSlot) -> FontSlot:
        """Choose the best slot that actually contains a glyph for ``ch``."""
        cp = ord(ch)
        if self._face_for(preferred).supports_ord(cp):
            return preferred
        for fb in (
            FontSlot.LATIN_FALLBACK,
            FontSlot.CJK_FALLBACK,
            FontSlot.SERIF_REGULAR,
            FontSlot.SANS_REGULAR,
        ):
            # The CJK fallback is optional (absent on a Latin-only host), so a
            # missing slot is skipped rather than raising KeyError.
            if fb == preferred or fb not in self._slot_paths:
                continue
            if self._face_for(fb).supports_ord(cp):
                return fb
        return preferred

    def measure_text_pt(self, text: str, slot: FontSlot, font_size_pt: float) -> float:
        """Return exact advance width of ``text`` in PDF points matching Type0 ``/W``."""
        if not text:
            return 0.0
        total = 0.0
        for run_slot, run_text in self.split_by_fallback(text, slot):
            face = self._face_for(run_slot)
            buf = hb.Buffer()
            buf.add_str(run_text)
            buf.guess_segment_properties()
            # PDF Type0 CID ``Tj`` advances strictly by nominal ``/W`` (hmtx) widths without
            # OpenType GPOS kerning or GSUB ligatures; disable them so measured width matches
            # actual rendered glyph placement 1:1 (avoiding Latin-to-CJK collisions like
            # "OpenWebText上").
            hb.shape(face.hb_font, buf, {"kern": False, "liga": False})
            scale = font_size_pt / face.upem
            for pos in buf.glyph_positions:
                total += pos.x_advance * scale
        return total

    def split_by_fallback(self, text: str, preferred: FontSlot) -> list[tuple[FontSlot, str]]:
        """Partition ``text`` into contiguous runs sharing one supported font slot."""
        if not text:
            return []
        runs: list[tuple[FontSlot, str]] = []
        cur_slot = self.resolve_char_slot(text[0], preferred)
        buf = [text[0]]
        for ch in text[1:]:
            s = self.resolve_char_slot(ch, preferred)
            if s == cur_slot:
                buf.append(ch)
            else:
                runs.append((cur_slot, "".join(buf)))
                cur_slot = s
                buf = [ch]
        runs.append((cur_slot, "".join(buf)))
        return runs

    def record_usage(self, text: str, preferred: FontSlot) -> list[tuple[FontSlot, str]]:
        """Split ``text`` by fallback slot and register every used character for subsetting."""
        runs = self.split_by_fallback(text, preferred)
        for slot, seg in runs:
            self._used_chars[slot].update(seg)
        return runs

    def embed_all_used(self, pdf: pikepdf.Pdf) -> dict[FontSlot, EmbeddedFont]:
        """Subset and embed every font slot that recorded characters into ``pdf``."""
        embedded: dict[FontSlot, EmbeddedFont] = {}
        for idx, slot in enumerate(FontSlot):
            chars = self._used_chars.get(slot)
            if not chars:
                continue
            resource_name = f"/UBT_F{idx}"
            embedded[slot] = self._embed_single(
                pdf,
                slot=slot,
                font_path=self._slot_paths[slot],
                chars=chars,
                resource_name=resource_name,
            )
        return embedded

    def _embed_single(
        self,
        pdf: pikepdf.Pdf,
        *,
        slot: FontSlot,
        font_path: Path,
        chars: set[str],
        resource_name: str,
    ) -> EmbeddedFont:
        font = _open_ttfont(str(font_path), lazy=False)
        try:
            cmap = font.getBestCmap() or {}
            supported_cps = sorted({ord(ch) for ch in chars if ord(ch) in cmap})
            if not supported_cps:
                supported_cps = [0x20] if 0x20 in cmap else [next(iter(cmap))]

            opts = subset.Options()
            opts.layout_features = []
            opts.drop_tables += ["EBDT", "EBLC", "EBSC", "SVG ", "COLR", "CPAL"]
            opts.notdef_outline = True
            opts.recalc_bounds = True
            opts.retain_gids = False
            sub = subset.Subsetter(options=opts)
            sub.populate(unicodes=supported_cps)
            sub.subset(font)

            upem = float(font["head"].unitsPerEm)
            gname_to_cid, is_cff = _extract_cid_mapping(font)
            sub_cmap = font.getBestCmap() or {}
            hmtx = font["hmtx"]

            char_to_cid: dict[str, int] = {}
            cid_widths: dict[int, int] = {}
            cid_to_uni: dict[int, int] = {}
            for cp in supported_cps:
                gname = sub_cmap.get(cp)
                if gname is None:
                    continue
                cid = gname_to_cid[gname]
                char_to_cid[chr(cp)] = cid
                cid_widths[cid] = int(round(hmtx[gname][0] * _EM / upem))
                cid_to_uni[cid] = cp

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

            base_font = f"UBTSUB+{slot.value}"
            desc_dict: dict[str, Any] = {
                "/Type": pikepdf.Name("/FontDescriptor"),
                "/FontName": pikepdf.Name(f"/{base_font}"),
                "/Flags": 4,
                "/FontBBox": pikepdf.Array(bbox),
                "/ItalicAngle": 0,
                "/Ascent": ascent,
                "/Descent": descent,
                "/CapHeight": cap_height,
                "/StemV": 80,
            }

            if is_cff:
                top = font["CFF "].cff.topDictIndex[0]
                is_cid_keyed = hasattr(top, "ROS") and top.ROS is not None
                if is_cid_keyed:
                    cff_bytes = font.getTableData("CFF ")
                    ff3 = pikepdf.Stream(pdf, cff_bytes)
                    ff3["/Subtype"] = pikepdf.Name("/CIDFontType0C")
                else:
                    buf = io.BytesIO()
                    font.save(buf)
                    ff3 = pikepdf.Stream(pdf, buf.getvalue())
                    ff3["/Subtype"] = pikepdf.Name("/OpenType")
                desc_dict["/FontFile3"] = ff3
                cid_subtype = "/CIDFontType0"
            else:
                buf = io.BytesIO()
                font.save(buf)
                desc_dict["/FontFile2"] = pikepdf.Stream(pdf, buf.getvalue())
                cid_subtype = "/CIDFontType2"

            descriptor = pdf.make_indirect(pikepdf.Dictionary(desc_dict))

            w_array = pikepdf.Array()
            for cid, w in sorted(cid_widths.items()):
                w_array.append(cid)
                w_array.append(pikepdf.Array([w]))

            cid_dict_data: dict[str, Any] = {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name(cid_subtype),
                "/BaseFont": pikepdf.Name(f"/{base_font}"),
                "/CIDSystemInfo": pikepdf.Dictionary(
                    {
                        "/Registry": pikepdf.String("Adobe"),
                        "/Ordering": pikepdf.String("Identity"),
                        "/Supplement": 0,
                    }
                ),
                "/FontDescriptor": descriptor,
                "/DW": int(_EM),
                "/W": w_array,
            }
            if not is_cff:
                cid_dict_data["/CIDToGIDMap"] = pikepdf.Name("/Identity")

            cid_font = pdf.make_indirect(pikepdf.Dictionary(cid_dict_data))
            tounicode = pikepdf.Stream(pdf, _build_tounicode_cmap(cid_to_uni))

            type0 = pdf.make_indirect(
                pikepdf.Dictionary(
                    {
                        "/Type": pikepdf.Name("/Font"),
                        "/Subtype": pikepdf.Name("/Type0"),
                        "/BaseFont": pikepdf.Name(f"/{base_font}"),
                        "/Encoding": pikepdf.Name("/Identity-H"),
                        "/DescendantFonts": pikepdf.Array([cid_font]),
                        "/ToUnicode": tounicode,
                    }
                )
            )
            return EmbeddedFont(
                slot=slot,
                resource_name=resource_name,
                base_font=base_font,
                pdf_font_obj=type0,
                char_to_cid=char_to_cid,
                cid_widths=cid_widths,
                ascent_em=ascent / _EM,
                descent_em=descent / _EM,
            )
        finally:
            font.close()
