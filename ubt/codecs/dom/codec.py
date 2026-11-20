"""Unified surgical DOM codec for EPUB and HTML/XHTML documents.

Extracts leaf block elements using the UBT v2 lightweight inline anchor
protocol (``<b1>...</b1>``, ``<i2>...</i2>``, ``<a3>...</a3>``, ``<s4>...</s4>``,
``<c5>...</c5>``, ``<x1/>``) so bold, italic, links, superscripts, images,
SVGs, and MathML round-trip with 100% fidelity in both ``MONO`` and
``BILINGUAL`` delivery modes.
"""

from __future__ import annotations

import copy
import posixpath
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

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


BILINGUAL_TARGET_CLASS = "ubt-bilingual-target"

_BLOCK_TAGS: frozenset[str] = frozenset(
    {
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "td",
        "th",
        "blockquote",
        "figcaption",
        "dt",
        "dd",
        "title",
    }
)
_RIGID_INLINE_TAGS: frozenset[str] = frozenset(
    {"img", "svg", "math", "picture", "video", "audio", "br", "hr", "object", "iframe", "canvas"}
)
_PAIR_PREFIX: dict[str, str] = {
    "strong": "b",
    "b": "b",
    "em": "i",
    "i": "i",
    "a": "a",
    "sup": "s",
    "sub": "s",
    "code": "c",
    "span": "c",
    "mark": "c",
    "u": "c",
    "del": "c",
    "abbr": "c",
    "cite": "i",
    "q": "i",
}
_TAG_PARSE_RE = re.compile(r"<(/?)([a-z]\d+)(\s*/)?\s*>")


def _is_safe_member_name(name: str) -> bool:
    if name in {"", ".", ".."} or "\\" in name:
        return False
    norm = posixpath.normpath(name)
    if norm in {"", ".", ".."} or norm.startswith("../") or norm.startswith("/"):
        return False
    return ":" not in norm.split("/", 1)[0]


def _role_for_tag(name: str) -> UnitRole:
    if name in {"h1", "h2", "h3", "h4", "h5", "h6", "title"}:
        return UnitRole.HEADING
    if name == "li":
        return UnitRole.LIST_ITEM
    if name in {"td", "th"}:
        return UnitRole.TABLE_CELL
    if name == "figcaption":
        return UnitRole.CAPTION
    return UnitRole.PARAGRAPH


def _is_leaf_block(tag: Tag) -> bool:
    """True when ``tag`` is a translatable block containing no nested block children."""
    if tag.name not in _BLOCK_TAGS:
        return False
    if tag.find_parent(["script", "style", "pre", "code", "svg", "math"]):
        return False
    # If this block contains another block tag, translate the inner leaves instead.
    for desc in tag.descendants:
        if isinstance(desc, Tag) and desc.name in _BLOCK_TAGS and desc is not tag:
            return False
    return True


def _encode_inline_children(
    node: Tag,
) -> tuple[str, str, list[InlineToken]]:
    """Serialize children of ``node`` into ``(marked, plain, tokens)``."""
    marked_parts: list[str] = []
    plain_parts: list[str] = []
    tokens: list[InlineToken] = []
    seq = 0

    def _walk(parent: Tag) -> None:
        nonlocal seq
        for child in parent.children:
            if isinstance(child, NavigableString):
                text = str(child)
                marked_parts.append(text)
                plain_parts.append(text)
                continue
            if not isinstance(child, Tag):
                continue

            cname = (child.name or "").lower()
            if cname in _RIGID_INLINE_TAGS or (
                cname == "a"
                and not child.get_text(strip=True)
                and (child.get("id") or child.get("name"))
            ):
                seq += 1
                tag_id = f"x{seq}"
                tokens.append(
                    InlineToken(
                        tag_id=tag_id,
                        kind="rigid",
                        payload={"html": str(child)},
                    )
                )
                marked_parts.append(f"<{tag_id}/>")
                continue

            if cname in _PAIR_PREFIX:
                inner_txt = child.get_text()
                if not inner_txt.strip():
                    # Preserve empty formatting/anchor tags as rigid tokens.
                    seq += 1
                    tag_id = f"x{seq}"
                    tokens.append(
                        InlineToken(tag_id=tag_id, kind="rigid", payload={"html": str(child)})
                    )
                    marked_parts.append(f"<{tag_id}/>")
                    continue
                seq += 1
                tag_id = f"{_PAIR_PREFIX[cname]}{seq}"
                attrs = {str(k): v for k, v in child.attrs.items()}
                tokens.append(
                    InlineToken(
                        tag_id=tag_id,
                        kind="pair",
                        payload={"tag": cname, "attrs": attrs},
                    )
                )
                marked_parts.append(f"<{tag_id}>")
                _walk(child)
                marked_parts.append(f"</{tag_id}>")
                continue

            # Unknown inline wrapper: walk children transparently.
            _walk(child)

    _walk(node)
    marked = "".join(marked_parts)
    plain = "".join(plain_parts)
    return marked, plain, tokens


def _rebuild_dom_children(
    soup: BeautifulSoup,
    target_node: Tag,
    marked_text: str,
    tokens: tuple[InlineToken, ...],
) -> None:
    """Replace ``target_node`` children from ``marked_text`` using ``tokens``."""
    token_map = {t.tag_id: t for t in tokens}
    target_node.clear()
    stack: list[tuple[str | None, Tag]] = [(None, target_node)]

    def _append_text(s: str) -> None:
        if s:
            stack[-1][1].append(NavigableString(s))

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
            raw_html = str(tok.payload.get("html", ""))
            if raw_html:
                frag = BeautifulSoup(raw_html, "html.parser")
                for elem in list(frag.contents):
                    stack[-1][1].append(elem)
            continue

        if not is_close:
            tag_name = str(tok.payload.get("tag", "span"))
            attrs = dict(tok.payload.get("attrs", {}))
            new_el = soup.new_tag(tag_name)
            for k, v in attrs.items():
                new_el[k] = v
            stack[-1][1].append(new_el)
            stack.append((tag_id, new_el))
        else:
            if len(stack) > 1 and stack[-1][0] == tag_id:
                stack.pop()

    _append_text(marked_text[pos:])


def _strip_duplicate_ids(node: Tag) -> None:
    if node.has_attr("id"):
        del node["id"]
    for desc in list(node.descendants):
        if isinstance(desc, Tag) and desc.has_attr("id"):
            del desc["id"]


def _patch_html_document(
    raw_html: str,
    *,
    chapter_id: str,
    translations: dict[str, str],
    units_by_id: dict[str, TranslationUnit],
    mode: DeliveryMode,
    parser: str = "html.parser",
) -> tuple[str, int, int]:
    soup = BeautifulSoup(raw_html, parser)
    leaves = [tag for tag in soup.find_all(list(_BLOCK_TAGS)) if _is_leaf_block(tag)]
    total = 0
    patched = 0
    for idx, leaf in enumerate(leaves, start=1):
        uid = f"{chapter_id}_b{idx:04d}"
        unit = units_by_id.get(uid)
        if unit is None:
            continue
        total += 1
        translated = translations.get(uid)
        if translated is None or not translated.strip():
            continue

        if mode == DeliveryMode.BILINGUAL and leaf.name != "title":
            clone = copy.copy(leaf)
            _strip_duplicate_ids(clone)
            existing_cls = clone.get("class")
            cls_list = (
                list(existing_cls)
                if isinstance(existing_cls, list)
                else (str(existing_cls).split() if existing_cls else [])
            )
            if BILINGUAL_TARGET_CLASS not in cls_list:
                cls_list.append(BILINGUAL_TARGET_CLASS)
            clone["class"] = " ".join(cls_list)
            _rebuild_dom_children(soup, clone, translated, unit.tokens)
            leaf.insert_after(clone)
        else:
            _rebuild_dom_children(soup, leaf, translated, unit.tokens)
        patched += 1

    return str(soup), total, patched


def _epub_spine_members(zf: zipfile.ZipFile) -> list[str]:
    """Return ordered XHTML/HTML content member paths from EPUB OPF spine."""
    names = [n for n in zf.namelist() if _is_safe_member_name(n)]
    budget = ZipReadBudget()

    opf_path: str | None = None
    if "META-INF/container.xml" in names:
        cxml = _read_zip_bytes(zf, "META-INF/container.xml", budget).decode(
            "utf-8", errors="ignore"
        )
        csoup = BeautifulSoup(cxml, "xml")
        rf = csoup.find("rootfile")
        if isinstance(rf, Tag) and rf.get("full-path"):
            cand = str(rf["full-path"])
            if _is_safe_member_name(cand) and cand in names:
                opf_path = cand

    if opf_path is None:
        opf_path = next((n for n in names if n.lower().endswith(".opf")), None)

    if opf_path is not None:
        opf_dir = posixpath.dirname(opf_path)
        opf_xml = _read_zip_bytes(zf, opf_path, budget).decode("utf-8", errors="ignore")
        osoup = BeautifulSoup(opf_xml, "xml")
        id_to_href: dict[str, str] = {}
        for item in osoup.find_all("item"):
            if not isinstance(item, Tag):
                continue
            iid = str(item.get("id") or "")
            href = str(item.get("href") or "")
            mtype = str(item.get("media-type") or "")
            if (
                iid
                and href
                and ("html" in mtype or href.lower().endswith((".xhtml", ".html", ".htm")))
            ):
                full = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href
                if _is_safe_member_name(full) and full in names:
                    id_to_href[iid] = full
        ordered: list[str] = []
        for itemref in osoup.find_all("itemref"):
            if isinstance(itemref, Tag):
                ref = str(itemref.get("idref") or "")
                if ref in id_to_href:
                    ordered.append(id_to_href[ref])
        if ordered:
            return ordered

    return sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))


@dataclass(slots=True)
class DOMCodec:
    """Surgical in-place DOM codec for ``.epub``, ``.html``, ``.htm``, and ``.xhtml``."""

    supported_suffixes: frozenset[str] = frozenset({".epub", ".html", ".htm", ".xhtml"})
    supports_page_selection: bool = False

    def inspect(self, source_path: Path) -> DocumentProfile:
        suffix = source_path.suffix.lower()
        if suffix == ".epub":
            with zipfile.ZipFile(source_path, "r") as zf:
                members = _epub_spine_members(zf)
                budget = ZipReadBudget()
                chars = 0
                has_math = False
                for m in members[:5]:
                    txt = _read_zip_bytes(zf, m, budget).decode("utf-8", errors="replace")
                    if "<math" in txt:
                        has_math = True
                    chars += len(BeautifulSoup(txt, "html.parser").get_text())
                est = int(chars / max(1, min(5, len(members))) * len(members)) if members else 0
            return DocumentProfile(
                format_name="epub",
                page_count=len(members),
                estimated_chars=est,
                has_math=has_math,
                recommended_mode=DeliveryMode.MONO,
                metadata={"spine_files": len(members)},
            )

        raw = source_path.read_text(encoding="utf-8", errors="replace")
        return DocumentProfile(
            format_name="html",
            page_count=1,
            estimated_chars=len(BeautifulSoup(raw, "html.parser").get_text()),
            has_math="<math" in raw,
            recommended_mode=DeliveryMode.MONO,
        )

    def _extract_html_units(
        self, raw_html: str, chapter_id: str, member_name: str | None = None
    ) -> list[TranslationUnit]:
        soup = BeautifulSoup(raw_html, "html.parser")
        leaves = [tag for tag in soup.find_all(list(_BLOCK_TAGS)) if _is_leaf_block(tag)]
        units: list[TranslationUnit] = []
        for idx, leaf in enumerate(leaves, start=1):
            marked, plain, tokens = _encode_inline_children(leaf)
            if not plain.strip():
                continue
            uid = f"{chapter_id}_b{idx:04d}"
            tag_name = (leaf.name or "p").lower()
            heading_level = (
                int(tag_name[1])
                if len(tag_name) == 2 and tag_name[0] == "h" and tag_name[1].isdigit()
                else 0
            )
            meta: dict[str, Any] = {
                "tag": leaf.name,
                "leaf_index": idx,
                "heading_level": heading_level,
            }
            if member_name is not None:
                meta["member"] = member_name
            units.append(
                TranslationUnit(
                    unit_id=uid,
                    chapter_id=chapter_id,
                    role=_role_for_tag(tag_name),
                    source_marked=marked.strip(),
                    source_plain=plain.strip(),
                    tokens=tuple(tokens),
                    translate=True,
                    codec_meta=meta,
                )
            )
        return units

    def extract(
        self,
        source_path: Path,
        *,
        pages: set[int] | None = None,
    ) -> tuple[DocumentProfile, list[TranslationUnit]]:
        del pages
        profile = self.inspect(source_path)
        units: list[TranslationUnit] = []
        if source_path.suffix.lower() == ".epub":
            with zipfile.ZipFile(source_path, "r") as zf:
                budget = ZipReadBudget()
                for c_idx, member in enumerate(_epub_spine_members(zf), start=1):
                    raw = _read_zip_bytes(zf, member, budget).decode("utf-8", errors="replace")
                    units.extend(
                        self._extract_html_units(raw, f"ch{c_idx:03d}", member_name=member)
                    )
        else:
            raw = source_path.read_text(encoding="utf-8", errors="replace")
            units.extend(self._extract_html_units(raw, "doc"))
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
        del target_lang, pages
        units_by_id = {u.unit_id: u for u in units}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_all = 0
        patched_all = 0

        if source_path.suffix.lower() == ".epub":
            with zipfile.ZipFile(source_path, "r") as src_zf:
                spine = _epub_spine_members(src_zf)
                member_to_chap = {m: f"ch{i:03d}" for i, m in enumerate(spine, start=1)}
                budget = ZipReadBudget()
                with zipfile.ZipFile(output_path, "w") as dst_zf:
                    for info in src_zf.infolist():
                        if not _is_safe_member_name(info.filename):
                            continue
                        data = _read_zip_bytes(src_zf, info.filename, budget)
                        if info.filename in member_to_chap:
                            raw = data.decode("utf-8", errors="replace")
                            patched_html, t_cnt, p_cnt = _patch_html_document(
                                raw,
                                chapter_id=member_to_chap[info.filename],
                                translations=translations,
                                units_by_id=units_by_id,
                                mode=mode,
                            )
                            total_all += t_cnt
                            patched_all += p_cnt
                            data = patched_html.encode("utf-8")
                        dst_zf.writestr(info, data)
        else:
            raw = source_path.read_text(encoding="utf-8", errors="replace")
            patched_html, total_all, patched_all = _patch_html_document(
                raw,
                chapter_id="doc",
                translations=translations,
                units_by_id=units_by_id,
                mode=mode,
            )
            output_path.write_text(patched_html, encoding="utf-8")

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
        raise NotImplementedError(
            "DOMCodec outputs reflowable HTML/EPUB; preview directly in browser."
        )
