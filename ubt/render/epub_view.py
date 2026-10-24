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

#: ``dcterms:modified`` when the caller supplies none. EPUB 3 requires the
#: element, and a wall-clock value made two renders of one document differ.
REPRODUCIBLE_MODIFIED = "2000-01-01T00:00:00Z"
#: ZIP entry timestamp (the earliest a ZIP can express). ``ZipInfo`` otherwise
#: defaults to ``time.localtime()``, stamping the current time into every entry.
REPRODUCIBLE_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def _entry(name: str) -> zipfile.ZipInfo:
    """A ``ZipInfo`` for ``name`` with a fixed timestamp."""
    info = zipfile.ZipInfo(name, date_time=REPRODUCIBLE_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


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
    '<head><meta charset="utf-8"/><title>{title}</title>{style}</head>\n'
    "<body>\n"
)
_XHTML_TAIL = "\n</body>\n</html>\n"

#: Minimal reading stylesheet, linked from every content document.
#:
#: The package previously shipped with no CSS at all: every reading system fell
#: back to its own default, which on a phone means an unindented wall of text at
#: one fixed size. This is intentionally small — a body size, a measure, line
#: height, paragraph spacing and a caption size — because the delivered file is
#: *reflowable*: the reading system owns the viewport and the user owns the font.
_EPUB_CSS = """\
body { margin: 0 5%; line-height: 1.6; }
p { margin: 0 0 0.8em; text-align: justify; }
h1, h2, h3, h4, h5, h6 { line-height: 1.25; margin: 1.2em 0 0.5em; page-break-after: avoid; }
h1 { font-size: 1.5em; }
h2 { font-size: 1.3em; }
h3 { font-size: 1.15em; }
figure { margin: 1em 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
figcaption { font-size: 0.9em; opacity: 0.8; text-align: left; margin-top: 0.4em; }
table { border-collapse: collapse; width: 100%; font-size: 0.9em; margin: 0.8em 0; }
th, td { border: 1px solid #999; padding: 0.25em 0.4em; vertical-align: top; }
pre { overflow-x: auto; font-size: 0.85em; line-height: 1.4; }
blockquote { margin: 0.8em 0 0.8em 1.5em; }
"""


def _dir_attr(direction: str) -> str:
    """`` dir="rtl"`` for an RTL target, empty for LTR (unchanged bytes)."""
    return f' dir="{html.escape(direction)}"' if direction and direction != "ltr" else ""


def _opf(
    identifier: str,
    title: str,
    lang: str,
    direction: str = "ltr",
    images: tuple[str, ...] = (),
    modified: str | None = None,
) -> str:
    """The package document.

    ``dcterms:modified`` is **required** by EPUB 3 (it is how a reading system
    tells two releases of the same book apart) and its absence is an epubcheck
    ERROR. The package never wrote it, so every delivered EPUB was technically
    invalid. ``modified`` is an ISO-8601 UTC timestamp supplied by the caller
    (the composition time); the fallback exists only for direct library callers.
    """
    progression = "rtl" if direction == "rtl" else "ltr"
    modified = modified or REPRODUCIBLE_MODIFIED
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
        f'    <meta property="dcterms:modified">{modified}</meta>\n'
        "  </metadata>\n"
        "  <manifest>\n"
        '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>\n'
        '    <item id="text" href="text.xhtml" media-type="application/xhtml+xml"/>\n'
        f'    <item id="css" href="style.css" media-type="text/css"/>\n'
        f"{image_items}"
        "  </manifest>\n"
        f'  <spine page-progression-direction="{progression}">\n'
        '    <itemref idref="text"/>\n'
        "  </spine>\n"
        "</package>\n"
    )


def _nav(title: str, lang: str, direction: str = "ltr") -> str:
    """The EPUB 3 navigation document.

    A single entry pointing at the one body document: the companion view
    renders the whole book into one ``text.xhtml``, so there are no per-chapter
    files to link. (A richer TOC would need heading anchors written into the
    body and one entry per anchor — not implemented; the docstring that
    claimed it described the plan, not the file.)
    """
    head = _XHTML_HEAD.format(
        lang=lang,
        title=html.escape(title),
        dir_attr=_dir_attr(direction),
        style='<link rel="stylesheet" type="text/css" href="style.css"/>',
    )
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
    modified: str | None = None,
) -> Composition:
    """Lower the delivered blocks to an EPUB 3 package, recording every placement.

    ``lang``/``direction`` describe the target language; an RTL target sets
    ``dir="rtl"`` on the content documents and ``page-progression-direction`` on
    the spine, so a reading system turns pages right-to-left.

    Figures are packaged: each resolvable asset is copied into ``OEBPS/images/``
    (deduplicated by content), listed in the OPF manifest, and referenced by the
    content document's ``<img src>``. The old package dropped them entirely —
    the reading system showed an empty box where every figure belonged.

    **The package is byte-reproducible.** ``modified`` is the ``dcterms:modified``
    timestamp (ISO-8601 UTC); ``None`` means :data:`REPRODUCIBLE_MODIFIED`. Every
    ZIP entry is stamped :data:`REPRODUCIBLE_ZIP_DATE` instead of the current
    time, because the default wrote the wall clock into both the OPF and the
    archive and made two renders of one document differ in bytes.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    identifier = f"urn:ubt:{doc_id or 'document'}"
    stamp = modified or REPRODUCIBLE_MODIFIED

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

    fragment, placements = render_fragment(
        blocks, translations, asset_resolver=_resolve_asset, math_renderer="unicode"
    )

    with atomic_write_path(output) as tmp_path, zipfile.ZipFile(tmp_path, "w") as archive:
        # The OCF spec requires ``mimetype`` first and uncompressed.
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, "application/epub+zip")
        archive.writestr(_entry("META-INF/container.xml"), _CONTAINER)
        archive.writestr(
            _entry("OEBPS/content.opf"),
            _opf(identifier, title, lang, direction, tuple(packaged), modified=stamp),
        )
        archive.writestr(_entry("OEBPS/nav.xhtml"), _nav(title, lang, direction))
        # Without a stylesheet the reading system applies its own defaults, and
        # a reflowable book then arrives as one size of unindented text.
        archive.writestr(_entry("OEBPS/style.css"), _EPUB_CSS)
        for href, data in packaged.items():
            archive.writestr(_entry(f"OEBPS/{href}"), data)
        xhtml = (
            _XHTML_HEAD.format(
                lang=lang,
                title=html.escape(title),
                dir_attr=_dir_attr(direction),
                style='<link rel="stylesheet" type="text/css" href="style.css"/>',
            )
            + fragment
            + _XHTML_TAIL
        )
        archive.writestr(_entry("OEBPS/text.xhtml"), xhtml)
    return Composition(output_path=output, placements=placements)


__all__ = ["compose_epub"]
