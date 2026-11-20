"""DOCX inspection adapter: typed AST extraction via ``reader_docx``.

Translation and OOXML delivery are owned by :class:`ubt.codecs.docx.DOCXCodec`;
this adapter is only reached by the explicit inspection path.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from docx import Document

from ubt.adapters.base import BaseDocumentAdapter
from ubt.analyze.assemble import number
from ubt.analyze.reader_docx import walk_docx_elements
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BookManifest,
    ChapterIR,
    ChapterMeta,
    IRBlock,
)
from ubt.core.ir.serializer import compute_file_sha256_cached
from ubt.core.zip_safety import ZipReadBudget

logger = logging.getLogger(__name__)

#: Dublin Core namespace on the ``docProps/core.xml`` title element.
_DC_TITLE = "{http://purl.org/dc/elements/1.1/}title"

#: Members a DOCX body parse must materialise. ``python-docx`` opens the package
#: with a plain ``zipfile`` and reads every part it needs with no cap of its own,
#: so these are the declared sizes that decide whether the body parse is safe.
#: ``word/document.xml`` is the payload (a 500 KB member can declare 10 GB); the
#: rest are the package scaffolding python-docx also unmarshals.
_DOCX_BODY_MEMBERS = (
    "word/document.xml",
    "word/styles.xml",
    "word/numbering.xml",
    "word/settings.xml",
)


def _guard_package_size(path: Path) -> None:
    """Refuse a DOCX whose body part declares a bomb-scale size.

    ``python-docx`` decompresses whatever the central directory declares, so a
    crafted ``word/document.xml`` takes the process down in ``Document(...)``
    before any UBT code sees a block — the kernel OOM-kills it, and there is no
    traceback to diagnose from. The zip budget in :mod:`ubt.core.zip_safety`
    already guards every *member read* UBT performs itself (EPUB, TMX, XLIFF,
    the core-properties title); this closes the one path where the read happens
    inside a third-party loader.

    The check is on the *declared* size, so it costs one central-directory
    parse: no bytes are decompressed to make the decision. An unreadable or
    non-zip file is left to ``python-docx`` to report, so its own error message
    (which names the file) is not replaced by a zip error.
    """
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            budget = ZipReadBudget()
            for name in _DOCX_BODY_MEMBERS:
                try:
                    info = zf.getinfo(name)
                except KeyError:
                    continue
                if not budget.allow(name, info.file_size):
                    raise DocumentParseError(
                        f"Refusing to parse {path.name}: its {name} declares "
                        f"{info.file_size} bytes, over the "
                        f"{budget.max_member_bytes}-byte decompression cap for one "
                        "archive member. A legitimate book body never reaches it; "
                        "this is the shape of a zip bomb."
                    )
    except zipfile.BadZipFile:
        return


#: Suffixes of OOXML package parts the DTD scanner inspects. The XML parts and
#: the relationship parts cover every member python-docx unmarshals into its
#: object tree; binary members (media, fonts) carry no XML prolog to scan.
_DOCX_XML_PART_SUFFIXES = (".xml", ".rels")


def _reject_dtd_parts(path: Path) -> None:
    """Refuse a DOCX whose XML parts declare a DOCTYPE or xml-stylesheet PI.

    ``python-docx`` parses package parts with plain lxml, which resolves
    internal entities and honours whatever the interpreter build allows — a
    default the project refuses to depend on (see ``ubt.core.xml_safety``).
    OOXML legitimately has no DTD in any part, so there is no harmless subset
    to negotiate: any part carrying a DOCTYPE (billion-laughs internal
    entities, ``SYSTEM "file:///..."`` external entities) or a stylesheet PI
    fails closed *before* ``Document(...)`` is called, with the offending
    member named in the error.

    The scan reads members through the zip budget, so a bomb-scale part is
    refused as unreadable rather than decompressed for scanning.
    """
    import zipfile

    from ubt.core.xml_safety import UnsafeXMLError, assert_no_doctype
    from ubt.core.zip_safety import ZipReadBudget, read_member

    try:
        with zipfile.ZipFile(path) as zf:
            budget = ZipReadBudget()
            for name in zf.namelist():
                if not name.lower().endswith(_DOCX_XML_PART_SUFFIXES):
                    continue
                raw = read_member(zf, name, budget)
                if raw is None:
                    continue
                try:
                    assert_no_doctype(raw, context=f"{path.name}:{name}")
                except UnsafeXMLError as err:
                    raise DocumentParseError(str(err)) from err
    except zipfile.BadZipFile:
        # Not a zip: python-docx reports its own (file-naming) error below, so
        # the guard stays quiet rather than replacing that message.
        return


def _core_title(path: Path) -> str:
    """The document's core title, or ``""`` when absent/unreadable.

    Reads only the small ``docProps/core.xml`` member rather than loading the
    whole package with ``python-docx``: ``extract_manifest`` needs the title
    alone, and ``parse_stream`` immediately loads the document again for the
    body, so the old path parsed every paragraph of the book twice (a second
    full DOM) just to read one metadata field.
    """
    import zipfile

    from ubt.core.xml_safety import UnsafeXMLError, parse_xml
    from ubt.core.zip_safety import ZipReadBudget, read_member

    try:
        with zipfile.ZipFile(path) as zf:
            raw = read_member(zf, "docProps/core.xml", ZipReadBudget())
    except (OSError, zipfile.BadZipFile) as err:
        raise DocumentParseError(f"Failed to open DOCX document {path}: {err}") from err
    if raw is None:
        return ""
    try:
        root = parse_xml(raw)
    except UnsafeXMLError:
        # A malformed core.xml is metadata trouble, not a document that cannot
        # be translated: fall back to the filename-derived title.
        return ""
    node = root.find(_DC_TITLE)
    return (node.text or "").strip() if node is not None else ""


class DOCXInspectionAdapter(BaseDocumentAdapter):
    """Adapter for Word documents (.docx) with in-place bilingual injection."""

    output_suffixes = frozenset({".docx"})

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Extract a single-chapter manifest titled from core properties."""
        if not input_path.exists():
            raise DocumentParseError(f"DOCX file not found: {input_path}")

        doc_id = await asyncio.to_thread(compute_file_sha256_cached, input_path)
        title = await asyncio.to_thread(_core_title, input_path) or input_path.stem

        chapter = ChapterMeta(
            chapter_id="docx_main",
            title=title,
            spine_index=1,
            source_file=input_path.name,
        )
        return BookManifest(
            doc_id=doc_id,
            title=title,
            source_path=str(input_path),
            chapters=[chapter],
        )

    async def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        """Stream the document as one ChapterIR (body walk off the event loop)."""
        manifest = await self.extract_manifest(input_path)
        chapter = manifest.chapters[0]

        loop = asyncio.get_running_loop()
        blocks = await loop.run_in_executor(None, self._parse_blocks_sync, input_path, chapter)

        yield ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id=chapter.chapter_id,
            title=chapter.title,
            spine_index=chapter.spine_index,
            blocks=blocks,
        )

    def _parse_blocks_sync(self, path: Path, chapter: ChapterMeta) -> list[IRBlock]:
        """Build the typed blocks from the DOCX document structure."""
        _guard_package_size(path)
        _reject_dtd_parts(path)
        doc = Document(str(path))
        pairs = walk_docx_elements(doc)
        elements = [p[0] for p in pairs]
        numbered = number(elements, chapter.chapter_id)
        return [IRBlock(element=elem) for elem in numbered]
