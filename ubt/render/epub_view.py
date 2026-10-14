"""EPUB lowering: the delivered blocks as an EPUB 3 package.

It reuses :func:`ubt.render.html_view.render_fragment` for the XHTML body, so the
HTML and EPUB views lower blocks identically, and wraps it in a minimal, valid
EPUB 3 container (``mimetype`` stored first, OPF metadata, a nav document, one
content document). As with the HTML view it shares the block lowering
(:func:`ubt.render.html_view.render_fragment`), so styled runs ship as XHTML
``<strong>``/``<em>``/``<sup>`` and colour spans in both views.
"""

from __future__ import annotations

import html
import mimetypes
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from ubt.core.atomic import atomic_write_path
from ubt.core.ir.models import IRBlock
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
    'xml:lang="{lang}" lang="{lang}"{dir_attr}>\n'
    '<head><meta charset="utf-8"/><title>{title}</title></head>\n'
    "<body>\n"
)
_XHTML_TAIL = "\n</body>\n</html>\n"


def _dir_attr(direction: str) -> str:
    """`` dir="rtl"`` for an RTL target, empty for LTR (unchanged bytes)."""
    return f' dir="{html.escape(direction)}"' if direction and direction != "ltr" else ""


def _opf(
    identifier: str,
    title: str,
    lang: str,
    direction: str = "ltr",
    images: tuple[str, ...] = (),
) -> str:
    progression = "rtl" if direction == "rtl" else "ltr"
    image_items = "".join(
        f'    <item id="img{index}" href="{html.escape(href)}" '
        f'media-type="{html.escape(mimetypes.guess_type(href)[0] or "image/png")}"/>\n'
        for index, href in enumerate(images, start=1)
    )
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
        f"{image_items}"
        "  </manifest>\n"
        f'  <spine page-progression-direction="{progression}">\n'
        '    <itemref idref="text"/>\n'
        "  </spine>\n"
        "</package>\n"
    )


def _nav(title: str, lang: str, direction: str = "ltr") -> str:
    head = _XHTML_HEAD.format(lang=lang, title=html.escape(title), dir_attr=_dir_attr(direction))
    entry = f'<nav epub:type="toc"><ol><li><a href="text.xhtml">{html.escape(title)}</a></li></ol></nav>'
    return f"{head}{entry}{_XHTML_TAIL}"


def compose_epub(
    blocks: Sequence[IRBlock],
    translations: Mapping[str, str],
    output_path: str | Path,
    *,
    title: str = "UBT translation",
    lang: str = "en",
    direction: str = "ltr",
    doc_id: str = "",
) -> Composition:
    """Lower the delivered blocks to an EPUB 3 package, recording every placement.

    ``lang``/``direction`` describe the target language; an RTL target sets
    ``dir="rtl"`` on the content documents and ``page-progression-direction`` on
    the spine, so a reading system turns pages right-to-left.

    Figures are packaged: each resolvable asset is copied into ``OEBPS/images/``
    (deduplicated by content), listed in the OPF manifest, and referenced by the
    content document's ``<img src>``. The old package dropped them entirely —
    the reading system showed an empty box where every figure belonged.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    identifier = f"urn:ubt:{doc_id or 'document'}"

    # Resolve + collect assets during lowering; the resolver also assigns the
    # in-package href, so render_fragment stays storage-agnostic.
    packaged: dict[str, bytes] = {}  # href -> bytes (insertion order = dedupe order)

    def _resolve_asset(asset: str) -> str | None:
        if not asset:
            return None
        candidate = Path(asset)
        if not candidate.is_absolute() or not candidate.exists():
            return None
        data = candidate.read_bytes()
        for href, existing in packaged.items():
            if existing == data:
                return href
        href = f"images/{len(packaged) + 1:04d}{candidate.suffix.lower() or '.png'}"
        packaged[href] = data
        return href

    fragment, placements = render_fragment(blocks, translations, asset_resolver=_resolve_asset)

    with atomic_write_path(output) as tmp_path, zipfile.ZipFile(tmp_path, "w") as archive:
        # The OCF spec requires ``mimetype`` first and uncompressed.
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, "application/epub+zip")
        archive.writestr("META-INF/container.xml", _CONTAINER)
        archive.writestr(
            "OEBPS/content.opf", _opf(identifier, title, lang, direction, tuple(packaged))
        )
        archive.writestr("OEBPS/nav.xhtml", _nav(title, lang, direction))
        for href, data in packaged.items():
            archive.writestr(f"OEBPS/{href}", data)
        xhtml = (
            _XHTML_HEAD.format(lang=lang, title=html.escape(title), dir_attr=_dir_attr(direction))
            + fragment
            + _XHTML_TAIL
        )
        archive.writestr("OEBPS/text.xhtml", xhtml)
    return Composition(output_path=output, placements=placements)


__all__ = ["compose_epub"]
