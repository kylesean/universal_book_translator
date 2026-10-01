"""EPUB lowering: the Document as an EPUB 3 package (ADR-0001 L6, §12 Q5).

The second non-PDF view. It reuses :func:`ubt.render.html_view.render_fragment`
for the XHTML body, so the HTML and EPUB views lower elements identically, and
wraps it in a minimal, valid EPUB 3 container (``mimetype`` stored first, OPF
metadata, a nav document, one content document). As with the HTML view it is a
*plain* view: the AST does not model inline structure yet, so paragraphs are
text-only. Same coverage rule as every lowering -- a missing attestation is a
refusal, not a silent gap.
"""

from __future__ import annotations

import html
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from ubt.model.ast import Document
from ubt.model.fidelity import Attestation
from ubt.render.html_view import render_fragment
from ubt.render.outputs import Composition

_CONTAINER = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
    "  <rootfiles>\n"
    '    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>\n'
    "  </rootfiles>\n"
    "</container>\n"
)

_XHTML_HEAD = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    "<!DOCTYPE html>\n"
    '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" '
    'xml:lang="{lang}" lang="{lang}">\n'
    '<head><meta charset="utf-8"/><title>{title}</title></head>\n'
    "<body>\n"
)
_XHTML_TAIL = "\n</body>\n</html>\n"


def _opf(identifier: str, title: str, lang: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        f'    <dc:identifier id="bookid">{html.escape(identifier)}</dc:identifier>\n'
        f"    <dc:title>{html.escape(title)}</dc:title>\n"
        f"    <dc:language>{html.escape(lang)}</dc:language>\n"
        "  </metadata>\n"
        "  <manifest>\n"
        '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>\n'
        '    <item id="text" href="text.xhtml" media-type="application/xhtml+xml"/>\n'
        "  </manifest>\n"
        "  <spine>\n"
        '    <itemref idref="text"/>\n'
        "  </spine>\n"
        "</package>\n"
    )


def _nav(title: str, lang: str) -> str:
    head = _XHTML_HEAD.format(lang=lang, title=html.escape(title))
    entry = f'<nav epub:type="toc"><ol><li><a href="text.xhtml">{html.escape(title)}</a></li></ol></nav>'
    return f"{head}{entry}{_XHTML_TAIL}"


def compose_epub(
    document: Document,
    attestations: Sequence[Attestation],
    delivered: Mapping[str, str],
    output_path: str | Path,
    *,
    title: str = "UBT translation",
    lang: str = "en",
) -> Composition:
    """Lower a realized document to an EPUB 3 package, recording every placement."""
    fragment, placements = render_fragment(document, attestations, delivered)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    identifier = f"urn:ubt:{document.source.doc_id or 'document'}"

    with zipfile.ZipFile(output, "w") as archive:
        # The OCF spec requires ``mimetype`` first and uncompressed.
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, "application/epub+zip")
        archive.writestr("META-INF/container.xml", _CONTAINER)
        archive.writestr("OEBPS/content.opf", _opf(identifier, title, lang))
        archive.writestr("OEBPS/nav.xhtml", _nav(title, lang))
        xhtml = _XHTML_HEAD.format(lang=lang, title=html.escape(title)) + fragment + _XHTML_TAIL
        archive.writestr("OEBPS/text.xhtml", xhtml)
    return Composition(output_path=output, placements=placements)


__all__ = ["compose_epub"]
