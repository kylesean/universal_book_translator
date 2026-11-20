"""Direct lxml OOXML surgical codec for DOCX documents.

Operates directly on ``word/document.xml`` (plus footnotes/endnotes) inside the
DOCX ZIP archive with zero loss of native Word structures:
- Preserves ``styles.xml``, headers/footers, themes, tables, borders, comments.
- Coalesces adjacent ``<w:r>`` sharing identical ``<w:rPr>`` into ``<rN>...</rN>``.
- Preserves Word OMML equations (``<m:oMath>`` / ``<m:oMathPara>``), drawings,
  and footnote references as rigid ``<xN/>`` tokens that travel cleanly with
  translated word order.
- Partitions chapters naturally at Heading 1 boundaries and supports both
  ``MONO`` (in-place paragraph rewrite) and ``BILINGUAL`` (cloned sibling
  paragraph insertion).
"""

from __future__ import annotations

import copy
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from ubt.codecs.protocol import (
    ApplyReport,
    DeliveryMode,
    DocumentProfile,
    InlineToken,
    TranslationUnit,
    UnitRole,
)
from ubt.core.exceptions import DocumentParseError
from ubt.core.zip_safety import ZipReadBudget, read_member


def _read_zip_bytes(zf: zipfile.ZipFile, name: str, budget: ZipReadBudget) -> bytes:
    data = read_member(zf, name, budget=budget)
    if data is None:
        raise DocumentParseError(f"ZIP decompression budget exceeded while reading {name!r}")
    return data


_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_XML_NS = "http://www.w3.org/XML/1998/namespace"

_W_P = f"{{{_W_NS}}}p"
_W_PPR = f"{{{_W_NS}}}pPr"
_W_PSTYLE = f"{{{_W_NS}}}pStyle"
_W_VAL = f"{{{_W_NS}}}val"
_W_R = f"{{{_W_NS}}}r"
_W_RPR = f"{{{_W_NS}}}rPr"
_W_RFONTS = f"{{{_W_NS}}}rFonts"
_W_EASTASIA = f"{{{_W_NS}}}eastAsia"
_W_T = f"{{{_W_NS}}}t"
_W_HYPERLINK = f"{{{_W_NS}}}hyperlink"
_W_TC = f"{{{_W_NS}}}tc"
_XML_SPACE = f"{{{_XML_NS}}}space"

_RIGID_P_CHILDREN: frozenset[str] = frozenset(
    {
        f"{{{_M_NS}}}oMath",
        f"{{{_M_NS}}}oMathPara",
        f"{{{_W_NS}}}bookmarkStart",
        f"{{{_W_NS}}}bookmarkEnd",
        f"{{{_W_NS}}}fldSimple",
    }
)
_RIGID_R_CHILDREN: frozenset[str] = frozenset(
    {
        f"{{{_W_NS}}}drawing",
        f"{{{_W_NS}}}pict",
        f"{{{_W_NS}}}object",
        f"{{{_W_NS}}}footnoteReference",
        f"{{{_W_NS}}}endnoteReference",
        f"{{{_W_NS}}}br",
        f"{{{_W_NS}}}tab",
        f"{{{_W_NS}}}sym",
    }
)

_EAST_ASIA_FONT_MAP: dict[str, str] = {
    "zh": "Noto Serif CJK SC",
    "ja": "Noto Serif CJK JP",
    "ko": "Noto Serif CJK KR",
}

_TAG_PARSE_RE = re.compile(r"<(/?)([a-z]\d+)(\s*/)?\s*>")
_XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _parse_xml_safe(raw: bytes) -> etree._Element:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    return etree.fromstring(raw, parser=parser)


def _rpr_key(rpr: etree._Element | None) -> bytes:
    return etree.tostring(rpr) if rpr is not None else b""


def _paragraph_style_id(p: etree._Element) -> str:
    ppr = p.find(_W_PPR)
    if ppr is None:
        return ""
    pstyle = ppr.find(_W_PSTYLE)
    if pstyle is None:
        return ""
    return str(pstyle.get(_W_VAL) or "")


def _is_heading_style(style_id: str) -> tuple[bool, int]:
    low = style_id.lower()
    m = re.match(r"heading\s*(\d+)", low)
    if m:
        return True, int(m.group(1))
    if low in {"title", "subtitle"}:
        return True, 1
    return False, 0


def _in_table_cell(p: etree._Element) -> bool:
    cur = p.getparent()
    while cur is not None:
        if cur.tag == _W_TC:
            return True
        cur = cur.getparent()
    return False


def _encode_paragraph(p: etree._Element) -> tuple[str, str, list[InlineToken], bytes]:
    """Encode ``<w:p>`` children into ``(marked, plain, tokens, dominant_rpr_xml)``."""
    # First pass: determine dominant rPr by character count.
    rpr_counts: dict[bytes, int] = {}
    for r in p.findall(f".//{_W_R}"):
        txt = "".join(t.text or "" for t in r.findall(_W_T))
        if txt:
            k = _rpr_key(r.find(_W_RPR))
            rpr_counts[k] = rpr_counts.get(k, 0) + len(txt)
    dom_rpr = max(rpr_counts, key=lambda k: rpr_counts[k]) if rpr_counts else b""

    marked_parts: list[str] = []
    plain_parts: list[str] = []
    tokens: list[InlineToken] = []
    seq = 0

    def _emit_run(run_el: etree._Element) -> None:
        nonlocal seq
        rpr_xml = _rpr_key(run_el.find(_W_RPR))
        buf_text: list[str] = []

        def _flush_buf() -> None:
            nonlocal seq
            if not buf_text:
                return
            seg = "".join(buf_text)
            buf_text.clear()
            plain_parts.append(seg)
            if rpr_xml != dom_rpr and seg.strip():
                seq += 1
                tid = f"r{seq}"
                tokens.append(
                    InlineToken(
                        tag_id=tid,
                        kind="pair",
                        payload={"kind": "rpr", "xml": rpr_xml.decode("utf-8")},
                    )
                )
                marked_parts.append(f"<{tid}>{seg}</{tid}>")
            else:
                marked_parts.append(seg)

        for ch in run_el:
            if ch.tag == _W_RPR:
                continue
            if ch.tag == _W_T:
                buf_text.append(ch.text or "")
            elif ch.tag in _RIGID_R_CHILDREN:
                _flush_buf()
                seq += 1
                tid = f"x{seq}"
                # Wrap rigid run child inside its own <w:r> with its rPr so it replays standalone.
                wrapper = etree.Element(_W_R)
                if rpr_xml:
                    wrapper.append(_parse_xml_safe(rpr_xml))
                wrapper.append(copy.deepcopy(ch))
                tokens.append(
                    InlineToken(
                        tag_id=tid,
                        kind="rigid",
                        payload={"xml": etree.tostring(wrapper).decode("utf-8")},
                    )
                )
                marked_parts.append(f"<{tid}/>")
        _flush_buf()

    for child in p:
        if child.tag == _W_PPR:
            continue
        if child.tag in _RIGID_P_CHILDREN:
            seq += 1
            tid = f"x{seq}"
            tokens.append(
                InlineToken(
                    tag_id=tid,
                    kind="rigid",
                    payload={"xml": etree.tostring(child).decode("utf-8")},
                )
            )
            marked_parts.append(f"<{tid}/>")
        elif child.tag == _W_R:
            _emit_run(child)
        elif child.tag == _W_HYPERLINK:
            link_text = "".join(t.text or "" for t in child.findall(f".//{_W_T}"))
            if not link_text.strip():
                seq += 1
                tid = f"x{seq}"
                tokens.append(
                    InlineToken(
                        tag_id=tid,
                        kind="rigid",
                        payload={"xml": etree.tostring(child).decode("utf-8")},
                    )
                )
                marked_parts.append(f"<{tid}/>")
            else:
                seq += 1
                tid = f"a{seq}"
                tokens.append(
                    InlineToken(
                        tag_id=tid,
                        kind="pair",
                        payload={"kind": "hyperlink", "attrs": dict(child.attrib)},
                    )
                )
                marked_parts.append(f"<{tid}>")
                for sub_r in child.findall(_W_R):
                    _emit_run(sub_r)
                marked_parts.append(f"</{tid}>")

    return "".join(marked_parts), "".join(plain_parts), tokens, dom_rpr


def _make_run_element(text: str, rpr_xml: bytes, east_asia_font: str | None) -> etree._Element:
    clean = _XML_ILLEGAL_RE.sub("", text)
    r = etree.Element(_W_R)
    rpr = _parse_xml_safe(rpr_xml) if rpr_xml else etree.Element(_W_RPR)
    if east_asia_font:
        rfonts = rpr.find(_W_RFONTS)
        if rfonts is None:
            rfonts = etree.SubElement(rpr, _W_RFONTS)
        if not rfonts.get(_W_EASTASIA):
            rfonts.set(_W_EASTASIA, east_asia_font)
    if len(rpr) > 0:
        r.append(rpr)
    t = etree.SubElement(r, _W_T)
    if clean.startswith(" ") or clean.endswith(" "):
        t.set(_XML_SPACE, "preserve")
    t.text = clean
    return r


def _rebuild_paragraph_xml(
    p: etree._Element,
    marked_text: str,
    tokens: tuple[InlineToken, ...],
    dom_rpr_xml: bytes,
    east_asia_font: str | None,
) -> None:
    """Rewrite ``<w:p>`` children in-place preserving ``<w:pPr>``, ``<w:rPr>``, and OMML."""
    token_map = {t.tag_id: t for t in tokens}
    ppr = p.find(_W_PPR)
    ppr_copy = copy.deepcopy(ppr) if ppr is not None else None
    p.clear()
    if ppr_copy is not None:
        p.append(ppr_copy)

    rpr_stack: list[bytes] = [dom_rpr_xml]
    container_stack: list[tuple[str | None, etree._Element]] = [(None, p)]

    def _append_text(s: str) -> None:
        if s:
            container_stack[-1][1].append(_make_run_element(s, rpr_stack[-1], east_asia_font))

    pos = 0
    for m in _TAG_PARSE_RE.finditer(marked_text):
        _append_text(marked_text[pos : m.start()])
        pos = m.end()

        is_close = bool(m.group(1))
        tag_id = m.group(2)
        is_self = bool(m.group(3))
        tok = token_map.get(tag_id)
        if tok is None:
            continue

        if is_self or tok.kind == "rigid":
            raw_xml = str(tok.payload.get("xml", ""))
            if raw_xml:
                container_stack[-1][1].append(_parse_xml_safe(raw_xml.encode("utf-8")))
            continue

        kind = tok.payload.get("kind")
        if not is_close:
            if kind == "hyperlink":
                hl = etree.Element(_W_HYPERLINK)
                for k, v in dict(tok.payload.get("attrs", {})).items():
                    hl.set(k, str(v))
                container_stack[-1][1].append(hl)
                container_stack.append((tag_id, hl))
            else:
                xml_str = str(tok.payload.get("xml", ""))
                rpr_stack.append(xml_str.encode("utf-8"))
                container_stack.append((tag_id, container_stack[-1][1]))
        else:
            if len(container_stack) > 1 and container_stack[-1][0] == tag_id:
                _, elem = container_stack.pop()
                if elem.tag != _W_HYPERLINK and len(rpr_stack) > 1:
                    rpr_stack.pop()

    _append_text(marked_text[pos:])


@dataclass(slots=True)
class DOCXCodec:
    """Surgical direct-OOXML codec for ``.docx`` documents."""

    supported_suffixes: frozenset[str] = frozenset({".docx"})
    supports_page_selection: bool = False

    def inspect(self, source_path: Path) -> DocumentProfile:
        with zipfile.ZipFile(source_path, "r") as zf:
            budget = ZipReadBudget()
            raw = _read_zip_bytes(zf, "word/document.xml", budget)
        root = _parse_xml_safe(raw)
        chars = sum(len(t.text or "") for t in root.findall(f".//{_W_T}"))
        has_math = root.find(f".//{{{_M_NS}}}oMath") is not None
        return DocumentProfile(
            format_name="docx",
            page_count=None,
            estimated_chars=chars,
            has_math=has_math,
            recommended_mode=DeliveryMode.MONO,
        )

    def _translatable_parts(self, zf: zipfile.ZipFile) -> list[str]:
        names = zf.namelist()
        parts: list[str] = []
        if "word/document.xml" in names:
            parts.append("word/document.xml")
        for extra in ("word/footnotes.xml", "word/endnotes.xml"):
            if extra in names:
                parts.append(extra)
        return parts

    def extract(
        self,
        source_path: Path,
        *,
        pages: set[int] | None = None,
    ) -> tuple[DocumentProfile, list[TranslationUnit]]:
        del pages
        profile = self.inspect(source_path)
        units: list[TranslationUnit] = []
        with zipfile.ZipFile(source_path, "r") as zf:
            budget = ZipReadBudget()
            for part_name in self._translatable_parts(zf):
                part_key = Path(part_name).stem
                root = _parse_xml_safe(_read_zip_bytes(zf, part_name, budget))
                chapter_idx = 1
                for p_idx, p in enumerate(root.findall(f".//{_W_P}"), start=1):
                    marked, plain, tokens, dom_rpr = _encode_paragraph(p)
                    if not plain.strip():
                        continue
                    is_h, level = _is_heading_style(_paragraph_style_id(p))
                    if part_name == "word/document.xml" and is_h and level == 1 and units:
                        chapter_idx += 1
                    role = (
                        UnitRole.HEADING
                        if is_h
                        else (
                            UnitRole.FOOTNOTE
                            if part_key in {"footnotes", "endnotes"}
                            else (UnitRole.TABLE_CELL if _in_table_cell(p) else UnitRole.PARAGRAPH)
                        )
                    )
                    uid = f"{part_key}_p{p_idx:05d}"
                    units.append(
                        TranslationUnit(
                            unit_id=uid,
                            chapter_id=f"{part_key}_ch{chapter_idx:03d}",
                            role=role,
                            source_marked=marked.strip(),
                            source_plain=plain.strip(),
                            tokens=tuple(tokens),
                            translate=True,
                            codec_meta={
                                "part": part_name,
                                "p_index": p_idx,
                                "dom_rpr": dom_rpr.decode("utf-8"),
                                "heading_level": level if is_h else 0,
                            },
                        )
                    )
        return profile, units

    def apply(
        self,
        source_path: Path,
        units: list[TranslationUnit],
        translations: dict[str, str],
        *,
        target_lang: str,
        mode: DeliveryMode = DeliveryMode.MONO,
        output_path: Path,
        pages: set[int] | None = None,
    ) -> ApplyReport:
        del pages
        lang_base = (target_lang or "zh").lower().split("-")[0]
        ea_font = _EAST_ASIA_FONT_MAP.get(lang_base)
        units_by_id = {u.unit_id: u for u in units}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_all = 0
        patched_all = 0

        with zipfile.ZipFile(source_path, "r") as src_zf:
            target_parts = set(self._translatable_parts(src_zf))
            budget = ZipReadBudget()
            with zipfile.ZipFile(output_path, "w") as dst_zf:
                for info in src_zf.infolist():
                    data = _read_zip_bytes(src_zf, info.filename, budget)
                    if info.filename in target_parts:
                        part_key = Path(info.filename).stem
                        root = _parse_xml_safe(data)
                        # Snapshot original paragraphs before inserting bilingual clones.
                        orig_ps = list(root.findall(f".//{_W_P}"))
                        for p_idx, p in enumerate(orig_ps, start=1):
                            uid = f"{part_key}_p{p_idx:05d}"
                            unit = units_by_id.get(uid)
                            if unit is None:
                                continue
                            total_all += 1
                            translated = translations.get(uid)
                            if translated is None or not translated.strip():
                                continue
                            dom_rpr = str(unit.codec_meta.get("dom_rpr", "")).encode("utf-8")
                            if mode == DeliveryMode.BILINGUAL:
                                clone = copy.deepcopy(p)
                                _rebuild_paragraph_xml(
                                    clone, translated, unit.tokens, dom_rpr, ea_font
                                )
                                parent = p.getparent()
                                if parent is not None:
                                    parent.insert(list(parent).index(p) + 1, clone)
                            else:
                                _rebuild_paragraph_xml(p, translated, unit.tokens, dom_rpr, ea_font)
                            patched_all += 1
                        data = etree.tostring(
                            root, xml_declaration=True, encoding="UTF-8", standalone=True
                        )
                    dst_zf.writestr(info, data)

        return ApplyReport(
            output_path=output_path,
            mode=mode,
            total_units=total_all,
            patched_units=patched_all,
        )

    def render_page_preview(
        self,
        source_path: Path,
        page_no: int,
        units: list[TranslationUnit],
        translations: dict[str, str],
        *,
        target_lang: str,
        dpi: int = 144,
    ) -> bytes:
        raise NotImplementedError("DOCXCodec outputs native OOXML; open in Word/LibreOffice.")
