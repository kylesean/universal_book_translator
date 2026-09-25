"""Builders for minimal but standards-compliant EPUB 3 packages.

Every EPUB test used to paste the same ``container.xml``, the same OPF frame and
the same seven-line ``ZipFile`` block — six copies in ``test_epub_adapter.py``
alone, which is why that file averaged 66 lines per test. What actually differs
between those tests is the *manifest and spine*: the hrefs, the itemrefs, whether
a nav/ncx exists. So the frame is shared and the interesting part stays visible
at the call site.

``mimetype`` is written first and ``ZIP_STORED`` because that is what the EPUB
spec requires, not an accident of the old helper.
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>"""

XHTML_NS = "http://www.w3.org/1999/xhtml"

_OPF_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="pub-id">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="pub-id">{pub_id}</dc:identifier>
        <dc:title>{title}</dc:title>{creator}
        <dc:language>{language}</dc:language>
    </metadata>
    <manifest>
{items}
    </manifest>
    <spine{toc_attr}>
{spine}
    </spine>
</package>"""


def item(
    id: str, href: str, *, media_type: str = "application/xhtml+xml", properties: str = ""
) -> str:
    """One ``<item>`` line for :func:`opf`'s ``items``."""
    extra = f' properties="{properties}"' if properties else ""
    return f'        <item id="{id}" href="{href}" media-type="{media_type}"{extra}/>'


def opf(
    *,
    pub_id: str,
    title: str,
    items: Sequence[str],
    spine: Sequence[str] | None = None,
    creator: str | None = None,
    language: str = "en",
    toc: str | None = None,
) -> str:
    """A complete ``content.opf``. ``spine`` defaults to the declared ``items``.

    ``toc`` names the spine's NCX item; pass an explicit ``spine`` to exercise
    dangling or out-of-order ``itemref``s, which is half of what these tests do.
    """
    refs = spine if spine is not None else [i.split('"')[1] for i in items]
    dc_creator = f"\n        <dc:creator>{creator}</dc:creator>" if creator else ""
    return _OPF_HEAD.format(
        pub_id=pub_id,
        title=title,
        creator=dc_creator,
        language=language,
        toc_attr=f' toc="{toc}"' if toc else "",
        items="\n".join(items),
        spine="\n".join(f'        <itemref idref="{ref}"/>' for ref in refs),
    )


def page(body: str, *, title: str = "t") -> str:
    """XHTML chapter document wrapping ``body``."""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<html xmlns="{XHTML_NS}">\n'
        f"<head><title>{title}</title></head>\n"
        f"<body>\n{body}\n</body>\n"
        "</html>"
    )


def write_epub(target_path: Path, *, opf_xml: str, parts: dict[str, str]) -> Path:
    """Zip ``parts`` (arcname -> body) around ``opf_xml`` as a valid EPUB.

    Arcnames are taken verbatim, so a test can place a chapter outside ``OEBPS/``
    or give it a percent-encoded name — the cases the adapter must resolve.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target_path, "w") as zf:
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            b"application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", opf_xml)
        for arcname, body in parts.items():
            zf.writestr(arcname, body)
    return target_path
