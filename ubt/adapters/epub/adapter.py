"""EPUB inspection adapter: ZIP/OPF/spine parsing for read-only audits.

Translation and delivery are owned by :class:`ubt.codecs.dom.DOMCodec`. This
adapter keeps the typed-reader projection the table/math/figure audits need;
it does not inject or repack a deliverable.
"""

import asyncio
import dataclasses
import html
import logging
import posixpath
import re
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from ubt.adapters.base import (
    BaseDocumentAdapter,
    decode_markup,
)
from ubt.analyze.assemble import number
from ubt.analyze.reader_epub import spine_chapters
from ubt.analyze.reader_html import walk_markup_elements
from ubt.core.cleaners.dynamic_boilerplate import (
    BoilerplateFingerprint,
    DynamicBoilerplateHarvester,
)
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BlockProvenance,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    IRBlock,
)
from ubt.core.ir.serializer import compute_file_sha256_cached
from ubt.core.xml_safety import strip_doctype_declarations
from ubt.core.zip_safety import ZipReadBudget, read_member
from ubt.model.ast import TextElement

logger = logging.getLogger(__name__)


def _natural_sort_key(name: str) -> list[object]:
    """Sort key ordering embedded integers numerically (``ch2`` before ``ch10``).

    Used only by the no-spine fallback below. A plain ``sorted()`` is
    lexicographic, so a spineless EPUB of ``ch1.xhtml``/``ch2.xhtml``/``ch10.xhtml``
    would read ``ch1, ch10, ch2`` -- whole chapters out of order. Splitting digit
    runs out and comparing them as ints restores the author's order.
    """
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _read_epub_member(
    zf: zipfile.ZipFile, name: str, budget: ZipReadBudget | None = None
) -> bytes | None:
    """Read one member unless it is missing or bomb-scale (see ``zip_safety``).

    Returns ``None`` (after logging) for a missing or over-budget member instead
    of raising, so the caller can skip it and still deliver the rest of the book.
    The guard caps both a single member (a crafted enormous declaration) and the
    archive total (a book made of many just-under-cap members).
    """
    return read_member(zf, name, budget)


_NAMED_HTML_ENTITY_RE = re.compile(r"&([a-zA-Z][a-zA-Z0-9]*);")
_EXEMPT_XML_ENTITIES = frozenset({"amp", "lt", "gt", "quot", "apos"})


def _resolve_named_entities(raw_html: str) -> str:
    """Pre-resolve HTML named entities (&nbsp;, &mdash;, etc.) to UTF-8 characters.

    Standard XML parsers (lxml with recover=True) discard undefined named
    entities when no DTD is bound, turning "Hello&nbsp;world" into "Helloworld".
    The 5 standard XML predefined entities (&amp;, &lt;, &gt;, &quot;, &apos;)
    are left intact.
    """

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in _EXEMPT_XML_ENTITIES:
            return m.group(0)
        decoded = html.unescape(m.group(0))
        return decoded if decoded != m.group(0) else m.group(0)

    return _NAMED_HTML_ENTITY_RE.sub(_sub, raw_html)


def _parse_xhtml(raw_html: str) -> BeautifulSoup:
    """Parse EPUB chapter markup as XML first, HTML as fallback.

    EPUB chapters are XHTML: the ``xml`` parser preserves self-closing tags,
    namespace prefixes (``epub:type``) and CDATA — ``html.parser`` mangles all
    of them. Real-world books still ship sloppy chapters (undefined entities
    like ``&nbsp;``, unclosed tags), so any XML parse failure degrades to the
    HTML parser instead of failing the book.

    The DOCTYPE — and any xml-stylesheet PI — is stripped *before* either
    parser runs (:func:`ubt.core.xml_safety.strip_doctype_declarations`): the
    lxml backend is not defusedxml, and a chapter carrying an internal or
    external entity declaration would be expanded per the interpreter build's
    defaults. XHTML needs no DOCTYPE to parse, so dropping the declaration
    neutralizes billion-laughs and XXE payloads instead of trusting defaults.
    """
    cleaned = _resolve_named_entities(strip_doctype_declarations(raw_html))
    try:
        return BeautifulSoup(cleaned, "xml")
    except Exception:
        return BeautifulSoup(cleaned, "html.parser")


BLOCK_TAGS = [
    "p",
    "blockquote",
    "li",
    "dd",
    "dt",
    "figcaption",
    # EPUBs can wrap prose in bare <div>s, put text in table
    # cells, and carry code in <pre>.
    # <pre> maps to BlockType.CODE (skip_translate) in the block classifier.
    "div",
    "td",
    "th",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
]


def wrap_nested_direct_blocks(soup: BeautifulSoup) -> None:
    """Wrap direct inline runs in container elements that also have block children into <p> tags.

    This ensures that e.g. nested lists (<li>Item text <ul><li>Subitem</li></ul></li>)
    do not lose the parent item's direct text node when scanning for leaf blocks.
    """
    block_tag_names = (
        set(BLOCK_TAGS)
        | {"ul", "ol", "table", "tbody", "thead", "tfoot", "tr"}
        # Content-model block containers: folding one of these into a
        # synthesized <p> emits <p><dl>…</dl></p>, which is invalid XHTML and
        # coalesces figure/definition structure into a translated paragraph.
        | {"dl", "figure", "hr", "details"}
    )
    for tag in soup.find_all(
        ["li", "blockquote", "div", "dd", "td", "th", "section", "article", "main", "body"]
    ):
        has_block_child = any(
            isinstance(c, Tag) and c.name in block_tag_names for c in tag.children
        )
        if not has_block_child:
            continue

        current_run: list[Tag | NavigableString] = []
        children = list(tag.contents)
        for c in children:
            if isinstance(c, Tag) and c.name in block_tag_names:
                if any(
                    isinstance(node, Tag) or (isinstance(node, NavigableString) and node.strip())
                    for node in current_run
                ):
                    p = soup.new_tag("p")
                    c.insert_before(p)
                    for node in current_run:
                        p.append(node.extract())
                current_run = []
            else:
                # Every PageElement is either a Tag or a NavigableString
                # (Comment/CData/etc. subclass NavigableString), so this
                # narrowing is exhaustive and behaviour-preserving.
                if isinstance(c, (Tag, NavigableString)):
                    current_run.append(c)
        if any(
            isinstance(node, Tag) or (isinstance(node, NavigableString) and node.strip())
            for node in current_run
        ):
            p = soup.new_tag("p")
            tag.append(p)
            for node in current_run:
                p.append(node.extract())


def _parse_chapter_blocks(
    zf: zipfile.ZipFile,
    chapter: ChapterMeta,
    fp: BoilerplateFingerprint,
    is_page_slice: bool,
    global_spine: int,
    budget: ZipReadBudget | None = None,
) -> list[IRBlock]:
    """Parse one EPUB spine item into IR blocks (synchronous).

    Runs on a worker thread via ``asyncio.to_thread`` from ``parse_stream``: the
    zip read and BeautifulSoup parse are CPU + IO and must not block the event
    loop. ``global_spine`` is the starting spine index; the caller advances it by
    ``len(result)``, so this returns the blocks only.
    """
    source_file = chapter.source_file
    if source_file is None:
        # Callers only enqueue a chapter whose spine member resolved (the
        # parse_stream loop skips a falsy/absent source_file), so this is
        # defensive. It is an explicit raise rather than an ``assert`` because
        # ``python -O`` strips asserts, which would turn a None into a later
        # ``AttributeError`` on ``None`` instead of this actionable message.
        raise DocumentParseError(f"EPUB chapter {chapter.chapter_id!r} has no source file to parse")
    raw_bytes = _read_epub_member(zf, source_file, budget)
    if raw_bytes is None:
        # Missing or zip-bomb-sized member: yield no blocks rather than read it.
        return []
    soup = _parse_xhtml(decode_markup(raw_bytes))
    wrap_nested_direct_blocks(soup)
    body = soup.body or soup

    pairs = walk_markup_elements(body)
    elements = [p[0] for p in pairs]
    numbered = number(elements, chapter.chapter_id)
    chapter_blocks: list[IRBlock] = []

    for elem, (_orig_elem, dom_node) in zip(numbered, pairs, strict=True):
        text = elem.text if isinstance(elem, TextElement) else ""
        if not text and not elem.is_asset:
            continue
        if (is_page_slice or fp.footer_disclaimers) and text:
            cleaned = fp.clean(text)
            if not cleaned:
                continue

        prov: dict[str, Any] = {}
        if isinstance(dom_node, Tag):
            leaf_id: Any = dom_node.get("id")
            if not leaf_id:
                anchor = dom_node.find(["a", "span"], attrs={"id": True})
                if anchor is not None:
                    leaf_id = anchor.get("id")
            if not leaf_id:
                named = dom_node.find("a", attrs={"name": True})
                if named is not None:
                    leaf_id = named.get("name")
            if leaf_id:
                prov["html_id"] = str(leaf_id)

        chapter_blocks.append(
            IRBlock(
                element=dataclasses.replace(elem, spine_index=global_spine),
                provenance=BlockProvenance(**prov),
            )
        )
        global_spine += 1

    return chapter_blocks


class EPUBAdapter(BaseDocumentAdapter):
    """EPUB inspection adapter: reads spine members into typed IR blocks."""

    output_suffixes = frozenset({".epub"})

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Parse EPUB container and OPF package to extract manifest metadata."""
        if not input_path.exists():
            raise DocumentParseError(f"EPUB file not found: {input_path}")

        # Off-loop: compute doc_id sha in thread pool to avoid blocking the event loop on multi-megabyte EPUBs.
        doc_id = await asyncio.to_thread(compute_file_sha256_cached, input_path)

        try:
            with zipfile.ZipFile(input_path) as zf:
                # One read budget for this archive: the OPF, the container and
                # the boilerplate samples share it, so a book of many
                # just-under-cap members cannot add up to an exhaustion.
                budget = ZipReadBudget()
                opf_path = self._locate_opf(zf, budget)
                opf_bytes = _read_epub_member(zf, opf_path, budget)
                if opf_bytes is None:
                    raise DocumentParseError(
                        f"Invalid EPUB: OPF package manifest {opf_path!r} could not be read "
                        "(missing or oversized member)."
                    )
                # Same guard as the chapters: the OPF is untrusted, and the
                # lxml-based parse must never see a DOCTYPE it could expand.
                opf_soup = BeautifulSoup(
                    strip_doctype_declarations(decode_markup(opf_bytes)), "xml"
                )

                title_tag = opf_soup.find("dc:title") or opf_soup.find("title")
                title = title_tag.get_text().strip() if title_tag else input_path.stem

                creator_tag = opf_soup.find("dc:creator") or opf_soup.find("creator")
                author = creator_tag.get_text().strip() if creator_tag else "Unknown"

                # Reading-order chapters, from the ONE spine implementation
                # (``ubt.analyze.reader_epub.spine_chapters``). This used to be a
                # second copy of the rule with navigation filtering and
                # duplicate collapsing; the native reader's copy had neither, so
                # the same minimal EPUB produced three pages through
                # ``read_epub`` (nav document counted as prose) and two chapters
                # here.
                spine_files = spine_chapters(zf, opf_path, budget)
                chapters: list[ChapterMeta] = [
                    ChapterMeta(
                        chapter_id=f"ch_{idx:03d}_{posixpath.basename(member)}",
                        title=posixpath.basename(member),
                        spine_index=idx,
                        source_file=member,
                    )
                    for idx, member in enumerate(spine_files, 1)
                ]

                metadata: dict[str, object] = {"author": author, "opf_path": opf_path}

                # Detect if this is an OCR / scanned page-slice EPUB
                is_page_slice = len(chapters) >= 20 and any(
                    bool(c.source_file and "page_" in c.source_file.lower())
                    for c in chapters[: min(10, len(chapters))]
                )
                if is_page_slice or len(chapters) >= 50:
                    metadata["is_page_slice_epub"] = is_page_slice
                    sample_texts: list[str] = []
                    sample_names = set(zf.namelist())
                    sample_range = chapters[min(5, len(chapters) - 1) : min(35, len(chapters))]
                    for c in sample_range:
                        if c.source_file and c.source_file in sample_names:
                            sample_raw = _read_epub_member(zf, c.source_file, budget)
                            if sample_raw is None:
                                continue
                            raw = decode_markup(sample_raw)
                            soup = BeautifulSoup(raw, "html.parser")
                            txt = soup.get_text(" ", strip=True)
                            if len(txt) >= 150:
                                sample_texts.append(txt)
                    if sample_texts:
                        harvester = DynamicBoilerplateHarvester()
                        fp = harvester.harvest(sample_texts)
                        if fp.footer_disclaimers:
                            metadata["boilerplate_footers"] = list(fp.footer_disclaimers)

                return BookManifest(
                    doc_id=doc_id,
                    title=title,
                    source_path=str(input_path),
                    metadata=metadata,
                    chapters=chapters,
                )

        except Exception as err:
            raise DocumentParseError(
                f"Failed to parse EPUB manifest from {input_path}: {err}",
                doc_id=doc_id,
                details={"error": str(err)},
            ) from err

    async def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        """Stream EPUB documents partitioned by chapter spine items."""
        manifest = await self.extract_manifest(input_path)
        global_spine = 1

        footers = manifest.metadata.get("boilerplate_footers")
        fp = (
            BoilerplateFingerprint(footer_disclaimers=tuple(str(f) for f in footers))
            if footers and isinstance(footers, list)
            else BoilerplateFingerprint()
        )
        is_page_slice = bool(manifest.metadata.get("is_page_slice_epub", False))

        with zipfile.ZipFile(input_path) as zf:
            # One central-directory read for the whole stream: per-chapter
            # namelist() calls rebuilt the full member list every iteration.
            member_names = set(zf.namelist())
            # One read budget for the whole stream, so the chapters together
            # cannot exceed the archive cap even if each stays under the member
            # cap.
            budget = ZipReadBudget()
            for chapter in manifest.chapters:
                if not chapter.source_file or chapter.source_file not in member_names:
                    continue

                # Per-chapter zip read + BeautifulSoup parse + block build is
                # synchronous CPU + IO; offload it so a concurrent task is not
                # stalled once per chapter.
                try:
                    chapter_blocks = await asyncio.to_thread(
                        _parse_chapter_blocks,
                        zf,
                        chapter,
                        fp,
                        is_page_slice,
                        global_spine,
                        budget,
                    )
                except DocumentParseError as err:
                    # Name the chapter. A single unreadable member used to abort
                    # the whole book with an error that pointed at no particular
                    # file; silently skipping is worse (a delivered book would be
                    # missing a chapter), so fail with the location.
                    raise DocumentParseError(
                        f"EPUB chapter {chapter.chapter_id!r} ({chapter.source_file}) "
                        f"could not be parsed: {err}"
                    ) from err
                global_spine += len(chapter_blocks)

                if chapter_blocks:
                    yield ChapterIR(
                        doc_id=manifest.doc_id,
                        chapter_id=chapter.chapter_id,
                        title=chapter.title,
                        spine_index=chapter.spine_index,
                        blocks=chapter_blocks,
                    )

    def _locate_opf(self, zf: zipfile.ZipFile, budget: ZipReadBudget | None = None) -> str:
        """Locate the root OPF file path via META-INF/container.xml."""
        container_bytes = _read_epub_member(zf, "META-INF/container.xml", budget)
        if container_bytes is not None:
            container_xml = decode_markup(container_bytes)
            m = re.search(r'full-path=["\']([^"\']+)["\']', container_xml)
            if m:
                return m.group(1)

        # Fallback to finding any .opf file in the archive
        for name in zf.namelist():
            if name.lower().endswith(".opf"):
                return name

        raise DocumentParseError("Invalid EPUB: OPF package manifest could not be located.")
