"""Native EPUB reader: EPUB -> typed ``Document`` (native multi-format reader seam).

An EPUB is a zip with an OPF manifest/spine; the reader walks the spine in order
and turns each XHTML document into elements with the shared markup walker
(:mod:`ubt.analyze.reader_html`). ``Span.page`` is the 1-based spine position, so
an element's provenance is "which chapter document it came from"; ``Span.chars``
indexes the concatenated canonical text across the whole book.

Figure assets are resolved to their *in-archive* path (the OPF directory is the
base), so an ``asset_id`` names a real member of the zip rather than a relative
href.
"""

from __future__ import annotations

import functools
import posixpath
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup

from ubt.analyze._identity import file_digest
from ubt.analyze.assemble import assemble, number
from ubt.analyze.reader_html import decode_bytes, elements_from_markup
from ubt.core.xml_safety import parse_xml
from ubt.core.zip_safety import ZipReadBudget, read_member
from ubt.model.ast import Document

_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
_OPF_NS = "http://www.idpf.org/2007/opf"


def _opf_path(zf: zipfile.ZipFile, budget: ZipReadBudget | None = None) -> str:
    """The OPF package path from ``META-INF/container.xml``."""
    container = read_member(zf, "META-INF/container.xml", budget)
    if container is None:
        raise ValueError("EPUB container.xml is missing or oversized")
    root = parse_xml(container)
    rootfile = root.find(f".//{{{_CONTAINER_NS}}}rootfile")
    if rootfile is None:
        raise ValueError("EPUB container.xml has no rootfile")
    return str(rootfile.get("full-path"))


#: Spine members that are reading matter. An OPF manifest may also list stylesheets,
#: fonts and the navigation document; only these are chapters.
_SPINE_MEMBER_SUFFIXES = (".xhtml", ".html", ".htm", ".xml")


def nav_item_ids(root: object) -> set[str]:
    """The manifest ids of the package's own navigation documents.

    EPUB3 declares ``properties="nav"``; EPUB2 lists the NCX by media type. They
    are metadata, not reading matter: mining ``nav.xhtml`` turned every
    table-of-contents entry into a block that was sent to the model (paid
    twice) and then re-injected as a dead ``<li>`` bullet beside the real,
    already-translated link.
    """
    ids: set[str] = set()
    for item in root.findall(f".//{{{_OPF_NS}}}manifest/{{{_OPF_NS}}}item"):  # type: ignore[attr-defined]
        item_id = item.get("id")
        if not item_id:
            continue
        properties = str(item.get("properties") or "").split()
        media_type = str(item.get("media-type") or "")
        if "nav" in properties or media_type in {
            "application/x-dtbncx+xml",
            "application/x-dtbook+xml",
        }:
            ids.add(str(item_id))
    return ids


def spine_chapters(
    zf: zipfile.ZipFile,
    opf_path: str,
    budget: ZipReadBudget | None = None,
) -> list[str]:
    """The reading-order XHTML member paths of the spine.

    The single implementation of the spine rule, shared with the EPUB adapter.
    The two used to be separate: this one had none of the adapter's
    navigation-document filtering or duplicate collapsing, so the same minimal
    EPUB produced three pages through ``read_epub`` (the nav document counted as
    prose) and two chapters through the adapter.

    Rules, each of which was a fixed bug in one of the two copies:

    * OPF hrefs are percent-encoded (``text%20ch1.xhtml``); joining the raw value
      makes the archive lookup fail and the chapter vanish silently;
    * a relative href (``../OEBPS/x.xhtml``) is resolved against the OPF's own
      directory;
    * the navigation document is skipped (:func:`nav_item_ids`);
    * a spine that repeats a member contributes it once — a second chapter for
      the same file clones identical blocks (paid twice) that the renderer can
      never both inject;
    * with no usable spine at all, fall back to the archive's HTML members in
      natural order, still skipping anything whose name says "toc".
    """
    base = posixpath.dirname(opf_path)
    opf = read_member(zf, opf_path, budget)
    if opf is None:
        raise ValueError(f"EPUB package {opf_path!r} is missing or oversized")
    root = parse_xml(opf)
    manifest: dict[str, str] = {}
    for item in root.findall(f".//{{{_OPF_NS}}}manifest/{{{_OPF_NS}}}item"):
        item_id, href = item.get("id"), item.get("href")
        if item_id and href:
            manifest[str(item_id)] = posixpath.normpath(
                posixpath.join(base, unquote(str(href))) if base else unquote(str(href))
            )
    skip = nav_item_ids(root)

    spine: list[str] = []
    seen: set[str] = set()
    for itemref in root.findall(f".//{{{_OPF_NS}}}spine/{{{_OPF_NS}}}itemref"):
        idref = str(itemref.get("idref") or "")
        if not idref or idref in skip:
            continue
        member = manifest.get(idref)
        if not member or not member.lower().endswith(_SPINE_MEMBER_SUFFIXES):
            continue
        if member in seen:
            continue
        seen.add(member)
        spine.append(member)
    if spine:
        return spine

    # No usable spine: the archive's own HTML members, in natural order.
    return sorted(
        (
            name
            for name in zf.namelist()
            if name.lower().endswith((".xhtml", ".html", ".htm")) and "toc" not in name.lower()
        ),
        key=_natural_sort_key,
    )


def _natural_sort_key(name: str) -> tuple[object, ...]:
    """Sort ``ch2`` before ``ch10``: split digit runs out as integers."""
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name))


def _resolve_asset(src: str, item_path: str) -> str:
    """An image ``src`` as its member path inside the archive."""
    clean = src.split("#", 1)[0].split("?", 1)[0]
    if not clean:
        return src
    if clean.startswith("/"):
        clean = clean.lstrip("/")
        return posixpath.normpath(clean)
    return posixpath.normpath(posixpath.join(posixpath.dirname(item_path), clean))


def read_epub(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Read an EPUB into a typed :class:`Document` with real spans."""
    epub_path = Path(path)
    out = []
    with zipfile.ZipFile(epub_path) as zf:
        budget = ZipReadBudget()
        spine = spine_chapters(zf, _opf_path(zf, budget), budget)
        for index, item_path in enumerate(spine):
            raw = read_member(zf, item_path, budget)
            if raw is None:
                continue
            soup = BeautifulSoup(decode_bytes(raw), "html.parser")
            out.extend(
                elements_from_markup(
                    soup,
                    page=index + 1,
                    resolve_asset=functools.partial(_resolve_asset, item_path=item_path),
                )
            )
    elements = number(out, "epub")
    return assemble(elements, doc_id=doc_id or file_digest(epub_path), path=str(epub_path))


__all__ = ["read_epub"]
