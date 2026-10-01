"""Native EPUB reader: EPUB -> typed ``Document`` (ADR-0001 §12 Q5).

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
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup
from lxml import etree

from ubt.analyze._identity import file_digest
from ubt.analyze.assemble import assemble, number
from ubt.analyze.reader_html import decode_bytes, elements_from_markup
from ubt.model.ast import Document

_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
_OPF_NS = "http://www.idpf.org/2007/opf"


def _opf_path(zf: zipfile.ZipFile) -> str:
    """The OPF package path from ``META-INF/container.xml``."""
    root = etree.fromstring(zf.read("META-INF/container.xml"))
    rootfile = root.find(f".//{{{_CONTAINER_NS}}}rootfile")
    if rootfile is None:
        raise ValueError("EPUB container.xml has no rootfile")
    return str(rootfile.get("full-path"))


def _spine(zf: zipfile.ZipFile, opf_path: str) -> list[str]:
    """The XHTML member paths of the spine, in reading order."""
    base = posixpath.dirname(opf_path)
    root = etree.fromstring(zf.read(opf_path))
    manifest: dict[str, str] = {}
    for item in root.findall(f".//{{{_OPF_NS}}}manifest/{{{_OPF_NS}}}item"):
        item_id, href = item.get("id"), item.get("href")
        if item_id and href:
            manifest[item_id] = href
    spine: list[str] = []
    for itemref in root.findall(f".//{{{_OPF_NS}}}spine/{{{_OPF_NS}}}itemref"):
        href = manifest.get(itemref.get("idref") or "")
        if href:
            spine.append(posixpath.normpath(posixpath.join(base, href)))
    return spine


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
        spine = _spine(zf, _opf_path(zf))
        for index, item_path in enumerate(spine):
            try:
                raw = zf.read(item_path)
            except KeyError:
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
