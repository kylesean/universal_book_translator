"""EPUB Native DOM Tree bilingual injection adapter using zipfile and BeautifulSoup."""

import asyncio
import html
import logging
import posixpath
import re
import shutil
import tempfile
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from bs4 import BeautifulSoup, Tag
from bs4.element import AttributeValueList, NavigableString

from ubt.adapters.base import BILINGUAL_TARGET_CLASS, BaseDocumentAdapter, decode_markup
from ubt.adapters.unresolved import failure_note, is_unresolved
from ubt.core.cleaners.dynamic_boilerplate import (
    BoilerplateFingerprint,
    DynamicBoilerplateHarvester,
)
from ubt.core.cleaners.html_sanitizer import (
    sanitize_html_fragment,
    scrub_css_text,
    scrub_source_document,
    strip_html_mark_tags,
)
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BlockType,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.ir.serializer import compute_file_sha256_cached

logger = logging.getLogger(__name__)


def _is_safe_epub_member_name(name: str) -> bool:
    """False for an archive member name that escapes the package.

    ``render_output`` preserves each source member's ``ZipInfo`` verbatim, so a
    name like ``../evil.xhtml`` or ``/etc/passwd`` would propagate into the
    delivered ``.epub`` and could escape a downstream naive extractor.
    """
    if name in ("", ".", "..") or "\\" in name:
        return False
    normalized = posixpath.normpath(name)
    if normalized in ("", ".", "..") or normalized.startswith("../"):
        return False
    if normalized.startswith("/"):
        return False
    # A drive/URN-looking first segment ("C:") is absolute on Windows.
    return ":" not in normalized.split("/", 1)[0]


#: Refuse to materialise a single member beyond this. A "zip bomb" member
#: declares an enormous uncompressed size; ``zipfile`` stops at the declared
#: size, so capping it before ``read`` is what keeps one crafted member from
#: exhausting memory (a lying declaration is stopped and CRC-failed inside
#: ``zipfile`` anyway).
_MAX_EPUB_MEMBER_BYTES = 300 * 1024 * 1024


def _read_epub_member(zf: zipfile.ZipFile, name: str) -> bytes | None:
    """Read one member unless its declared size is bomb-scale.

    Returns ``None`` (after logging) for a missing or oversized member instead
    of raising, so the caller can skip it and still deliver the rest of the book.
    """
    try:
        info = zf.getinfo(name)
    except KeyError:
        return None
    if info.file_size > _MAX_EPUB_MEMBER_BYTES:
        logger.warning("EPUB: skipping oversized member %r (%d bytes)", name, info.file_size)
        return None
    return zf.read(name)


_XML_DECL_ENCODING_RE = re.compile(
    r"""(<\?xml[^>]*?\bencoding\s*=\s*["'])([A-Za-z0-9._-]+)(["'])""", re.IGNORECASE
)


def _force_utf8_declaration(text: str) -> str:
    """Rewrite a non-UTF-8 XML declaration to ``utf-8`` after a re-encode.

    A member decoded with its declared encoding is written back as UTF-8; if the
    declaration still named the old encoding the reader would mis-decode it.
    """
    return _XML_DECL_ENCODING_RE.sub(r"\1utf-8\3", text, count=1)


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
    namespace prefixes (``epub:type``), DOCTYPE, and CDATA — ``html.parser``
    mangles all of them. Real-world books still ship sloppy chapters
    (undefined entities like ``&nbsp;``, unclosed tags), so any XML parse
    failure degrades to the HTML parser instead of failing the book.
    """
    cleaned = _resolve_named_entities(raw_html)
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

BILINGUAL_CSS_NAME = "ubt-bilingual.css"

BILINGUAL_CSS = """
/* Universal Book Translator bilingual styling */
.ubt-bilingual-target {
    color: #4a5568;
    margin-top: 0.35em !important;
    margin-bottom: 0.6em !important;
    font-size: 0.96em;
    line-height: 1.55;
}
"""


def is_leaf_block(tag: Tag, block_names: set[str]) -> bool:
    """True if tag does not contain any nested sub-blocks.

    Uses the same ``block_names`` set as the caller so a wrapper tag is not
    mined twice (e.g. a ``<div>`` around ``<pre>`` as both CODE and NARRATIVE).
    """
    return tag.find(list(block_names)) is None


#: Inline children a monolingual rewrite must not lose. Replacing a leaf's text
#: with the translation used to ``clear()`` the element, silently deleting inline
#: images, footnote/cross-reference anchors, ``<br>`` breaks and embedded math;
#: the delivered book then lost every figure and broke every footnote link.
#: DOCX already preserves graphic runs and hyperlinks; these are the HTML/EPUB
#: counterparts.
_PRESERVED_INLINE_TAGS = frozenset(
    {"img", "svg", "picture", "source", "video", "audio", "br", "hr", "math", "object", "a"}
)


def take_preserved_inline_children(leaf: Tag) -> list[Tag]:
    """Detach and return inline descendants a monolingual rewrite must keep.

    Collected from the whole subtree (a footnote reference is the standard
    ``<sup><a href="#fn1">1</a></sup>`` shape — a direct-children scan drops it
    with its href/id when the leaf is cleared). Only the outermost preserved
    tags are taken: their subtrees ride along, so nothing is duplicated.
    The nodes are detached (kept alive) so the caller can ``clear()`` the leaf
    and re-append them after the translation.
    """
    preserved = [
        c
        for c in leaf.find_all(list(_PRESERVED_INLINE_TAGS))
        if not any(p.name in _PRESERVED_INLINE_TAGS for p in c.parents)
    ]
    for node in preserved:
        node.extract()
    return preserved


def wrap_nested_direct_blocks(soup: BeautifulSoup) -> None:
    """Wrap direct inline runs in container elements that also have block children into <p> tags.

    This ensures that e.g. nested lists (<li>Item text <ul><li>Subitem</li></ul></li>)
    do not lose the parent item's direct text node when scanning for leaf blocks.
    """
    block_tag_names = set(BLOCK_TAGS) | {"ul", "ol", "table", "tbody", "thead", "tfoot", "tr"}
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


def _get_class_str(tag: Tag) -> str:
    """Safely extract class attribute from a Tag as a lowercase space-delimited string."""
    raw = tag.get("class")
    if isinstance(raw, list):
        return " ".join(str(c) for c in raw).lower()
    return str(raw).lower() if raw else ""


def determine_flow_id(tag: Tag) -> FlowID:
    """Heuristically assign semantic FlowID based on parent hierarchy and class names."""
    parent_classes = " ".join([_get_class_str(p) for p in tag.parents if isinstance(p, Tag)])

    tag_class = _get_class_str(tag)
    combined_classes = f"{parent_classes} {tag_class}"

    if any(k in combined_classes for k in ("footnote", "endnote", "fn-")):
        return FlowID.FOOTNOTE
    if any(k in combined_classes for k in ("sidebar", "aside", "callout", "box")):
        return FlowID.SIDEBAR_ASIDE
    if tag.name == "figcaption" or "caption" in combined_classes:
        return FlowID.CAPTION
    if tag.name in ("td", "th") or any(p.name in ("td", "th") for p in tag.parents):
        return FlowID.TABLE_GRID

    return FlowID.MAIN_STORY


def _toc_label_text(text: str) -> str:
    """Visible text of a translated heading, safe to place in a TOC label.

    TOC labels are text nodes: embedding raw markup in one serializes as
    literal escaped tags, and the reader sees the *source* of the quarantine
    placeholder — E2E found ``<mark class="ubt-blocked-human" ...>【待人工
    审校…】</mark>`` spelled out inside toc.xhtml, which is exactly the
    "grammar tag leakage" the README forbids.

    Order matters: strip the real ``<mark>`` wrappers first, parse away any
    other real markup, and only then decode entities — that way the source
    quote the placeholder carries (stored as ``&lt;…&gt;``) re-emerges as
    *visible text* instead of being parsed as tags and swallowed. The
    entity-escaped spelling of the placeholder (the double-escaped form seen
    downstream) is stripped after decoding.
    """
    if not text:
        return text
    cleaned = strip_html_mark_tags(text)
    if "<" in cleaned:
        # Any other markup (e.g. the ubt-failed-draft wrapper): keep the text.
        cleaned = BeautifulSoup(cleaned, "html.parser").get_text(" ", strip=True)
    decoded = html.unescape(cleaned)
    if "<mark" in decoded:
        decoded = strip_html_mark_tags(decoded)
    return decoded.strip()


def _parse_chapter_blocks(
    zf: zipfile.ZipFile,
    chapter: ChapterMeta,
    block_names: set[str],
    fp: BoilerplateFingerprint,
    is_page_slice: bool,
    global_spine: int,
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
    raw_bytes = _read_epub_member(zf, source_file)
    if raw_bytes is None:
        # Missing or zip-bomb-sized member: yield no blocks rather than read it.
        return []
    soup = _parse_xhtml(decode_markup(raw_bytes))
    wrap_nested_direct_blocks(soup)
    body = soup.body or soup

    leaves = [t for t in body.find_all(BLOCK_TAGS) if is_leaf_block(t, block_names)]
    chapter_blocks: list[IRBlock] = []

    for leaf_idx, leaf in enumerate(leaves):
        text = leaf.get_text(" ", strip=True)
        if not text:
            continue

        # Dynamic boilerplate cleaning on page-slice or detected books
        if is_page_slice or fp.footer_disclaimers:
            text = fp.clean(text)
            if not text:
                continue

        # Classify block type
        if leaf.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            b_type = BlockType.HEADING
        elif leaf.name in ("pre", "code", "tt"):
            # See html adapter: inline <code> inside a narrative
            # paragraph must not make the whole block CODE.
            b_type = BlockType.CODE
        else:
            b_type = BlockType.NARRATIVE

        flow_id = determine_flow_id(leaf)
        block_id = f"{chapter.chapter_id}#p{leaf_idx:04d}"
        prov: dict[str, Any] = {}
        leaf_id: Any = leaf.get("id")
        if not leaf_id:
            anchor = leaf.find(["a", "span"], attrs={"id": True})
            if anchor is not None:
                leaf_id = anchor.get("id")
        if not leaf_id:
            named = leaf.find("a", attrs={"name": True})
            if named is not None:
                leaf_id = named.get("name")
        if leaf_id:
            prov["html_id"] = str(leaf_id)

        chapter_blocks.append(
            IRBlock(
                id=block_id,
                flow_id=flow_id,
                spine_index=global_spine,
                block_type=b_type,
                source_text=text,
                skip_translate=bool(b_type == BlockType.CODE),
                provenance=prov,
            )
        )
        global_spine += 1

    return chapter_blocks


class EPUBAdapter(BaseDocumentAdapter):
    """Adapter for EPUB books with zero-corruption DOM injection."""

    output_suffixes = frozenset({".epub"})

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Parse EPUB container and OPF package to extract manifest metadata."""
        if not input_path.exists():
            raise DocumentParseError(f"EPUB file not found: {input_path}")

        # Off-loop: compute doc_id sha in thread pool to avoid blocking the event loop on multi-megabyte EPUBs.
        doc_id = await asyncio.to_thread(compute_file_sha256_cached, input_path)

        try:
            with zipfile.ZipFile(input_path) as zf:
                opf_path = self._locate_opf(zf)
                opf_bytes = _read_epub_member(zf, opf_path)
                if opf_bytes is None:
                    raise DocumentParseError(
                        f"Invalid EPUB: OPF package manifest {opf_path!r} could not be read "
                        "(missing or oversized member)."
                    )
                opf_soup = BeautifulSoup(decode_markup(opf_bytes), "xml")

                # Extract title and author
                title_tag = opf_soup.find("dc:title") or opf_soup.find("title")
                title = title_tag.get_text().strip() if title_tag else input_path.stem

                creator_tag = opf_soup.find("dc:creator") or opf_soup.find("creator")
                author = creator_tag.get_text().strip() if creator_tag else "Unknown"

                # Extract spine items (reading order)
                manifest_items: dict[str, str] = {}
                #: ids of the package's own navigation documents (EPUB3
                #: ``properties="nav"``, EPUB2 NCX). They are metadata, not
                #: reading matter: mining nav.xhtml turned every table-of-contents
                #: entry into a block that was sent to the model (paid twice) and
                #: then re-injected as a dead ``<li>`` bullet next to the real,
                #: already-bilingualized link.
                nav_item_ids: set[str] = set()
                for item in opf_soup.find_all("item"):
                    item_id = item.get("id")
                    href = item.get("href")
                    properties = str(item.get("properties") or "").split()
                    media_type = str(item.get("media-type") or "")
                    if item_id and (
                        "nav" in properties
                        or media_type in {"application/x-dtbncx+xml", "application/x-dtbook+xml"}
                    ):
                        nav_item_ids.add(str(item_id))
                    if item_id and href:
                        item_id_str = str(item_id)
                        # OPF hrefs are URL references: percent-escapes must be
                        # decoded and relative segments resolved before the
                        # path is used as a zip member name, or encoded/spelling
                        # like "text%20ch1.xhtml" / "../OEBPS/x.xhtml" silently
                        # drops whole chapters.
                        href_str = unquote(str(href))
                        # Normalize relative path to OPF directory
                        base_dir = posixpath.dirname(opf_path)
                        full_href = (
                            posixpath.normpath(posixpath.join(base_dir, href_str))
                            if base_dir
                            else posixpath.normpath(href_str)
                        )
                        manifest_items[item_id_str] = full_href

                chapters: list[ChapterMeta] = []
                spine_tags = opf_soup.find_all("itemref")
                spine_idx = 1
                seen_source_files: set[str] = set()

                for itemref in spine_tags:
                    idref = itemref.get("idref")
                    if not idref:
                        continue
                    idref_str = str(idref)
                    if idref_str in nav_item_ids:
                        continue
                    file_href = manifest_items.get(idref_str)
                    if file_href and file_href.lower().endswith(
                        (".xhtml", ".html", ".htm", ".xml")
                    ):
                        # Anthologies legitimately repeat a spine item, but a
                        # second ChapterMeta for the same file clones identical
                        # blocks (paid twice to translate) that the renderer can
                        # never both inject: the zip member is visited once and
                        # the chapter lookup matches only the first clone.
                        if file_href in seen_source_files:
                            logger.info(
                                "EPUB manifest: spine repeats %s; keeping the first occurrence only",
                                file_href,
                            )
                            continue
                        seen_source_files.add(file_href)
                        chapters.append(
                            ChapterMeta(
                                chapter_id=f"ch_{spine_idx:03d}_{idref}",
                                title=posixpath.basename(file_href),
                                spine_index=spine_idx,
                                source_file=file_href,
                            )
                        )
                        spine_idx += 1

                # If no spine items found, fallback to sorting html files directly
                if not chapters:
                    html_files = sorted(
                        n
                        for n in zf.namelist()
                        if n.lower().endswith((".xhtml", ".html", ".htm"))
                        and "toc" not in n.lower()
                    )
                    for idx, hf in enumerate(html_files, 1):
                        chapters.append(
                            ChapterMeta(
                                chapter_id=f"ch_{idx:03d}",
                                title=posixpath.basename(hf),
                                spine_index=idx,
                                source_file=hf,
                            )
                        )

                metadata: dict[str, object] = {"author": author, "opf_path": opf_path}

                # Detect if this is an OCR / scanned page-slice EPUB
                is_page_slice = len(chapters) >= 20 and any(
                    bool(c.source_file and "page_" in c.source_file.lower())
                    for c in chapters[: min(10, len(chapters))]
                )
                if is_page_slice or len(chapters) >= 50:
                    metadata["is_page_slice_epub"] = is_page_slice
                    sample_texts: list[str] = []
                    sample_range = chapters[min(5, len(chapters) - 1) : min(35, len(chapters))]
                    for c in sample_range:
                        if c.source_file and c.source_file in zf.namelist():
                            sample_raw = _read_epub_member(zf, c.source_file)
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
        block_names = set(BLOCK_TAGS)
        global_spine = 1

        footers = manifest.metadata.get("boilerplate_footers")
        fp = (
            BoilerplateFingerprint(footer_disclaimers=tuple(str(f) for f in footers))
            if footers and isinstance(footers, list)
            else BoilerplateFingerprint()
        )
        is_page_slice = bool(manifest.metadata.get("is_page_slice_epub", False))

        with zipfile.ZipFile(input_path) as zf:
            for chapter in manifest.chapters:
                if not chapter.source_file or chapter.source_file not in zf.namelist():
                    continue

                # Per-chapter zip read + BeautifulSoup parse + block build is
                # synchronous CPU + IO; offload it so a concurrent task is not
                # stalled once per chapter.
                chapter_blocks = await asyncio.to_thread(
                    _parse_chapter_blocks,
                    zf,
                    chapter,
                    block_names,
                    fp,
                    is_page_slice,
                    global_spine,
                )
                global_spine += len(chapter_blocks)

                if chapter_blocks:
                    yield ChapterIR(
                        doc_id=manifest.doc_id,
                        chapter_id=chapter.chapter_id,
                        title=chapter.title,
                        spine_index=chapter.spine_index,
                        blocks=chapter_blocks,
                    )

    async def render_blocks(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        render_engine: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Inject bilingual translation nodes into original DOM (ledger-free).

        DOM parsing and the zip rewrite are synchronous CPU + file IO; run them
        off the event loop so a concurrent job/task is not stalled for the whole
        document.
        """
        return await asyncio.to_thread(
            self._render_blocks_sync,
            manifest,
            blocks,
            target_lang,
            output_path,
            bilingual_mode,
            render_engine,
            **kwargs,
        )

    def _render_blocks_sync(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        render_engine: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Inject bilingual translation nodes into original DOM (ledger-free)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_path = Path(manifest.source_path)
        block_names = set(BLOCK_TAGS)

        # Build in-memory lookup table of translated blocks: block_id -> target_text
        all_blocks = blocks
        # A skip_translate block's target_text equals its source_text (ingest.py
        # stamps them so the ledger holds the verbatim text). Injecting them
        # again would duplicate every <pre>/<code>/kept block in the shipped
        # EPUB as a ubt-bilingual-target node; the other three adapters filter
        # the same way.
        translation_map = {
            b.id: b.target_text for b in all_blocks if b.target_text and not b.skip_translate
        }
        # Blocks whose draft never passed the quality gates stay labelled on the
        # page (ubt.adapters.unresolved); EPUB used to inject the bare machine
        # draft indistinguishably from an approved translation.
        unresolved_notes = {
            b.id: failure_note(b.status)
            for b in all_blocks
            if is_unresolved(b.status) and b.target_text and not b.skip_translate
        }
        source_map = {b.id: b.source_text for b in all_blocks if b.source_text}

        if bilingual_mode is None and manifest and manifest.run:
            bilingual_mode = manifest.run.bilingual_mode

        total_injected = 0
        with tempfile.TemporaryDirectory(prefix="ubt_epub_render_") as tmpdir:
            temp_epub = Path(tmpdir) / "output.epub"

            with (
                zipfile.ZipFile(input_path) as zin,
                zipfile.ZipFile(temp_epub, "w") as zout,
            ):
                # Resolve the package file and its TOC documents so
                # they can be rewritten (language, CSS item, bilingual labels).
                opf_path = str(manifest.metadata.get("opf_path") or "")
                opf_dir = posixpath.dirname(opf_path)
                toc_paths: set[str] = set()
                opf_bytes = (
                    _read_epub_member(zin, opf_path)
                    if opf_path and opf_path in zin.namelist()
                    else None
                )
                if opf_bytes is not None:
                    opf_soup = BeautifulSoup(decode_markup(opf_bytes), "xml")
                    for item in opf_soup.find_all("item"):
                        href = str(item.get("href") or "")
                        if not href:
                            continue
                        media_type = str(item.get("media-type") or "")
                        properties = str(item.get("properties") or "").split()
                        if media_type == "application/x-dtbncx+xml" or "nav" in properties:
                            full = posixpath.join(opf_dir, href) if opf_dir else href
                            toc_paths.add(posixpath.normpath(full))
                title_map = self._translated_titles(manifest, all_blocks)
                css_entry = posixpath.join(opf_dir, BILINGUAL_CSS_NAME) if opf_path else ""

                # EPUB requirement: 'mimetype' must be the first file and ZIP_STORED (uncompressed)
                for info in zin.infolist():
                    if not _is_safe_epub_member_name(info.filename):
                        # A traversal-shaped or absolute name is copied verbatim
                        # into the deliverable by ``writestr``; a downstream
                        # naive extract could then escape its target directory.
                        logger.warning("EPUB: dropping member with unsafe name %r", info.filename)
                        continue
                    if css_entry and info.filename == css_entry:
                        # Existing bilingual stylesheet will be re-written once at the end
                        continue

                    data = _read_epub_member(zin, info.filename)
                    if data is None:
                        # Missing or zip-bomb-sized member: skip it rather than
                        # materialise it.
                        continue

                    if info.filename == "mimetype":
                        zout.writestr(
                            zipfile.ZipInfo("mimetype"),
                            data,
                            compress_type=zipfile.ZIP_STORED,
                        )
                        continue

                    # Package + TOC documents get bilingual updates.
                    if opf_path and info.filename == opf_path:
                        data = self._update_opf(data, target_lang)
                    elif info.filename in toc_paths:
                        doc_dir = posixpath.dirname(info.filename)
                        if info.filename.lower().endswith(".ncx"):
                            data = self._update_ncx(
                                data, doc_dir, title_map, bilingual_mode=bilingual_mode
                            )
                        else:
                            data = self._update_nav_xhtml(
                                data, doc_dir, title_map, bilingual_mode=bilingual_mode
                            )

                    # Check if file is a chapter XHTML document
                    matching_chapter = next(
                        (c for c in manifest.chapters if c.source_file == info.filename),
                        None,
                    )

                    if matching_chapter and info.filename.lower().endswith(
                        (".xhtml", ".html", ".htm", ".xml")
                    ):
                        # <link> the package-wide stylesheet instead
                        # of duplicating inline CSS into every chapter.
                        stylesheet_href: str | None = None
                        if opf_path:
                            css_path = posixpath.join(opf_dir, BILINGUAL_CSS_NAME)
                            stylesheet_href = posixpath.relpath(
                                css_path, posixpath.dirname(info.filename) or "."
                            )
                        data, chapter_injected = self._inject_bilingual_dom(
                            raw_html=data,
                            chapter_id=matching_chapter.chapter_id,
                            translation_map=translation_map,
                            block_names=block_names,
                            source_map=source_map,
                            unresolved_notes=unresolved_notes,
                            stylesheet_href=stylesheet_href,
                            bilingual_mode=bilingual_mode,
                        )
                        total_injected += chapter_injected

                    # Source markup is copied into the deliverable, so a
                    # malicious source could ship <script>/onload=/javascript:
                    # to the reader. Both the rewritten chapter and the members
                    # passed through untouched go through the same scrub; the
                    # PDF/OPF/NCX members below are not markup a reader executes.
                    if info.filename.lower().endswith((".xhtml", ".html", ".htm", ".xml", ".svg")):
                        data = _force_utf8_declaration(
                            scrub_source_document(decode_markup(data))
                        ).encode("utf-8")
                    elif info.filename.lower().endswith(".css"):
                        # An external stylesheet ships inside the deliverable, so
                        # a malicious source could smuggle a remote @import or a
                        # remote url() past the chapter scrub (tracking, and CSS
                        # attribute-selector exfiltration). Gate it exactly like
                        # an inline <style> body.
                        data = scrub_css_text(decode_markup(data)).encode("utf-8")

                    # Pass the original ZipInfo object so date_time,
                    # external_attr (unix permissions), and compress metadata
                    # survive the rewrite (writestr overrides only the size /
                    # CRC / compress_type it recomputes).
                    zout.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)

                # Ship the bilingual stylesheet as a real package item.
                if opf_path and opf_path in zin.namelist():
                    css_entry = posixpath.join(opf_dir, BILINGUAL_CSS_NAME)
                    zout.writestr(css_entry, BILINGUAL_CSS, compress_type=zipfile.ZIP_DEFLATED)

            # Loud failure instead of a silent untranslated book: if there were
            # translations to inject but none landed, the block ids did not
            # match the extracted chapters and the deliverable would ship in
            # the source language while the run reports success.
            if translation_map and total_injected == 0:
                raise DocumentParseError(
                    f"EPUB render injected 0 of {len(translation_map)} translations; "
                    "block ids did not match the extracted chapters, so the book "
                    "would ship untranslated. Refusing to emit a false success."
                )

            # Move temp epub to output_path using shutil.move (supports cross-device mounts)
            shutil.move(temp_epub, output_path)

        return output_path

    def _locate_opf(self, zf: zipfile.ZipFile) -> str:
        """Locate the root OPF file path via META-INF/container.xml."""
        container_bytes = _read_epub_member(zf, "META-INF/container.xml")
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

    def _inject_bilingual_dom(
        self,
        raw_html: bytes,
        chapter_id: str,
        translation_map: dict[str, str],
        block_names: set[str],
        source_map: dict[str, str] | None = None,
        unresolved_notes: dict[str, str] | None = None,
        stylesheet_href: str | None = None,
        bilingual_mode: str | None = None,
    ) -> tuple[bytes, int]:
        """Inject target translation tags as siblings directly after original source tags (or replace in monolingual mode).

        Returns the rewritten XHTML and the number of target nodes injected, so
        the caller can detect a whole chapter that silently stayed untranslated
        (block-id mismatch) instead of shipping a source-language book and
        reporting success.
        """
        soup = _parse_xhtml(decode_markup(raw_html))
        wrap_nested_direct_blocks(soup)
        body = soup.body or soup
        leaves = [t for t in body.find_all(BLOCK_TAGS) if is_leaf_block(t, block_names)]
        is_monolingual = bilingual_mode in ("target", "monolingual")

        injected_count = 0
        for leaf_idx, leaf in enumerate(leaves):
            block_id = f"{chapter_id}#p{leaf_idx:04d}"
            target_text = translation_map.get(block_id)
            if (
                source_map
                and target_text  # Only touch source when we also have a
                # translation — otherwise the cleaner-stripped boilerplate would
                # silently vanish from the *source* text too, losing the original.
                and block_id in source_map
                and source_map[block_id]
                and source_map[block_id] != leaf.get_text(" ", strip=True)
                and not leaf.find_all(True)
            ):
                leaf.string = source_map[block_id]
            if not target_text:
                continue

            # Labelled unresolved draft, never mistaken for an approved
            # translation (ubt.adapters.unresolved).
            note = (unresolved_notes or {}).get(block_id)
            if note:
                target_text = f"{note}\n\n{target_text}"

            if is_monolingual:
                preserved_inline = take_preserved_inline_children(leaf)
                if "\n\n" in target_text:
                    paras = [p.strip() for p in target_text.split("\n\n") if p.strip()]
                    leaf.clear()
                    # Sanitize every paragraph exactly like the
                    # single-paragraph path below, so LLM inline markup is
                    # parsed consistently instead of rendering literally.
                    in_cell = leaf.name in ("td", "th", "li")
                    last_node: Tag = leaf
                    for i, p_text in enumerate(paras):
                        sanitized = sanitize_html_fragment(p_text)
                        if i == 0:
                            container: Tag = leaf
                        elif in_cell:
                            # A monolingual cell keeps its own <td>; extra
                            # paragraphs nest as <div> *inside* it. A sibling
                            # <td> would double the column count and wreck
                            # the table.
                            container = soup.new_tag("div")
                        else:
                            container = soup.new_tag(leaf.name)
                        parsed_fragment = BeautifulSoup(sanitized, "html.parser")
                        for child in list(parsed_fragment.contents):
                            container.append(child)
                        if i > 0:
                            if in_cell:
                                leaf.append(container)
                            else:
                                last_node.insert_after(container)
                        last_node = container
                    injected_count += len(paras)
                else:
                    sanitized_target = sanitize_html_fragment(target_text)
                    leaf.clear()
                    parsed_fragment = BeautifulSoup(sanitized_target, "html.parser")
                    for child in list(parsed_fragment.contents):
                        leaf.append(child)
                    injected_count += 1
                # Restore the inline media/anchors the rewrite must not lose.
                for node in preserved_inline:
                    leaf.append(node)
                continue

            # If translation contains multiple paragraphs separated by \n\n,
            # inject each cleanly. The allowlist sanitizer runs first — the
            # same contract as the single-paragraph branches above and below:
            # raw LLM markup must never reach the stored EPUB, and escaping
            # happens in the sanitizer, so inline <em>/<strong> still renders
            # as formatting instead of literal text.
            raw_classes = leaf.get("class")
            if isinstance(raw_classes, list):
                target_classes = list(raw_classes) + [BILINGUAL_TARGET_CLASS]
            elif raw_classes:
                target_classes = str(raw_classes).split() + [BILINGUAL_TARGET_CLASS]
            else:
                target_classes = [BILINGUAL_TARGET_CLASS]

            if "\n\n" in target_text:
                paras = [p.strip() for p in target_text.split("\n\n") if p.strip()]
                is_internal_child = leaf.name in ("td", "th", "li")
                if is_internal_child:
                    # Inside a table cell or ordered list item: append inside the element,
                    # never as a sibling (a sibling cell doubles column count, sibling li
                    # doubles ordered list item counters).
                    for p_text in paras:
                        parsed_para = BeautifulSoup(sanitize_html_fragment(p_text), "html.parser")
                        new_tag = soup.new_tag("div")
                        new_tag["class"] = AttributeValueList(target_classes)
                        for child in list(parsed_para.contents):
                            new_tag.append(child)
                        leaf.append(new_tag)
                        injected_count += 1
                else:
                    last_node = leaf
                    for p_text in paras:
                        parsed_para = BeautifulSoup(sanitize_html_fragment(p_text), "html.parser")
                        new_tag = soup.new_tag("p")
                        new_tag["class"] = AttributeValueList(target_classes)
                        for child in list(parsed_para.contents):
                            new_tag.append(child)
                        last_node.insert_after(new_tag)
                        last_node = new_tag
                        injected_count += 1
            else:
                # LLM output is untrusted — run the allowlist sanitizer
                # before any HTML parsing so <script>/handlers/javascript:
                # URLs can never reach the stored EPUB.
                sanitized_target = sanitize_html_fragment(target_text)
                is_internal_child = leaf.name in ("td", "th", "li")
                # Inside a table cell or ordered list: append a <div> *inside* the element so the
                # row/column structure and list numbering are preserved; otherwise match leaf tag.
                new_tag = soup.new_tag("div") if is_internal_child else soup.new_tag(leaf.name)
                new_tag["class"] = AttributeValueList(target_classes)
                parsed_fragment = BeautifulSoup(sanitized_target, "html.parser")
                for child in list(parsed_fragment.contents):
                    new_tag.append(child)

                if is_internal_child:
                    leaf.append(new_tag)
                else:
                    leaf.insert_after(new_tag)
                injected_count += 1

        if injected_count > 0:
            # Inject styling into <head>. The stylesheet ships once
            # as an OPF manifest item and is <link>ed (EPUB best practice,
            # no duplicated inline CSS per chapter); callers that don't
            # provide a package-wide href fall back to the inline <style>.
            head = soup.head
            if head is None and soup.html:
                head = soup.new_tag("head")
                soup.html.insert(0, head)
            if head is not None:
                if stylesheet_href:
                    link_tag = soup.new_tag("link")
                    link_tag["rel"] = "stylesheet"
                    link_tag["type"] = "text/css"
                    link_tag["href"] = stylesheet_href
                    head.append(link_tag)
                else:
                    style_tag = soup.new_tag("style")
                    style_tag.string = BILINGUAL_CSS
                    head.append(style_tag)

        return str(soup).encode("utf-8"), injected_count

    # ------------------------------------------------------------------
    # OPF / NCX / nav package updates
    # ------------------------------------------------------------------

    def _update_opf(self, raw_opf: bytes, target_lang: str) -> bytes:
        """Point the package at the bilingual output: dc:language + CSS item."""
        soup = BeautifulSoup(decode_markup(raw_opf), "xml")
        lang = soup.find("dc:language") or soup.find("language")
        if lang is not None:
            lang.string = target_lang
        manifest_tag = soup.find("manifest")
        if manifest_tag is not None and not manifest_tag.find(
            "item", attrs={"href": BILINGUAL_CSS_NAME}
        ):
            css_item = soup.new_tag("item")
            css_item["id"] = "ubt-bilingual-css"
            css_item["href"] = BILINGUAL_CSS_NAME
            css_item["media-type"] = "text/css"
            manifest_tag.append(css_item)
        return str(soup).encode("utf-8")

    def _translated_titles(
        self, manifest: BookManifest, all_blocks: list[IRBlock]
    ) -> dict[str, str]:
        """Map chapter source_file (and source_file#anchor) -> translated heading text.

        The earliest heading translation per chapter (by spine order) stands
        in for the chapter title. Subheadings with anchor IDs or matching source
        text are also mapped so nested TOC navPoints resolve accurately.
        """
        chapter_by_id = {c.chapter_id: c for c in manifest.chapters}
        candidates: dict[str, list[tuple[int, str]]] = {}
        result: dict[str, str] = {}

        for b in all_blocks:
            if not b.target_text or b.skip_translate or b.block_type != BlockType.HEADING:
                continue
            cleaned = _toc_label_text(b.target_text)
            if not cleaned:
                continue
            if b.source_text:
                result[f"__text__:{b.source_text.strip()}"] = cleaned

            chapter_id = b.id.split("#", 1)[0]
            chapter = chapter_by_id.get(chapter_id)
            if chapter is None or not chapter.source_file:
                continue

            candidates.setdefault(chapter.source_file, []).append((b.spine_index, cleaned))

            html_id = (b.provenance or {}).get("html_id")
            if html_id:
                result[f"{chapter.source_file}#{html_id}"] = cleaned

        for source_file, pairs in candidates.items():
            result[source_file] = min(pairs)[1]

        return result

    def _update_ncx(
        self,
        raw_ncx: bytes,
        ncx_dir: str,
        title_map: dict[str, str],
        bilingual_mode: str | None = None,
    ) -> bytes:
        """Append bilingual labels (or monolingual translated labels) to NCX navPoints."""
        soup = BeautifulSoup(decode_markup(raw_ncx), "xml")
        is_monolingual = bilingual_mode in ("target", "monolingual")
        for nav_point in soup.find_all("navPoint"):
            content = nav_point.find("content")
            label = nav_point.find("navLabel")
            if content is None or label is None:
                continue
            src_raw = str(content.get("src") or "")
            if not src_raw:
                continue
            src_raw = unquote(src_raw)
            if "#" in src_raw:
                src_file, anchor_id = src_raw.split("#", 1)
            else:
                src_file, anchor_id = src_raw, ""

            full_file = (
                posixpath.normpath(posixpath.join(ncx_dir, src_file)) if ncx_dir else src_file
            )
            full_with_anchor = f"{full_file}#{anchor_id}" if anchor_id else full_file

            text_tag = label.find("text") or label
            original = text_tag.get_text(strip=True)

            translated = None
            if anchor_id:
                translated = title_map.get(full_with_anchor)
            if not translated and original:
                translated = title_map.get(f"__text__:{original}")
            if not translated and not anchor_id:
                translated = title_map.get(full_file)

            if not translated:
                continue
            if translated in original:
                continue
            if is_monolingual:
                text_tag.string = translated
            else:
                text_tag.string = f"{original} / {translated}" if original else translated
        return str(soup).encode("utf-8")

    def _update_nav_xhtml(
        self,
        raw_nav: bytes,
        nav_dir: str,
        title_map: dict[str, str],
        bilingual_mode: str | None = None,
    ) -> bytes:
        """Append bilingual labels (or monolingual translated labels) to EPUB3 nav document links."""
        soup = _parse_xhtml(decode_markup(raw_nav))
        is_monolingual = bilingual_mode in ("target", "monolingual")
        for anchor in soup.find_all("a"):
            href_raw = str(anchor.get("href") or "")
            if not href_raw:
                continue
            href_raw = unquote(href_raw)
            if "#" in href_raw:
                href_file, anchor_id = href_raw.split("#", 1)
            else:
                href_file, anchor_id = href_raw, ""

            full_file = (
                posixpath.normpath(posixpath.join(nav_dir, href_file)) if nav_dir else href_file
            )
            full_with_anchor = f"{full_file}#{anchor_id}" if anchor_id else full_file

            original = anchor.get_text(strip=True)

            translated = None
            if anchor_id:
                translated = title_map.get(full_with_anchor)
            if not translated and original:
                translated = title_map.get(f"__text__:{original}")
            if not translated and not anchor_id:
                translated = title_map.get(full_file)

            if not translated:
                continue
            if translated in original:
                continue
            if is_monolingual:
                anchor.string = translated
            else:
                anchor.string = f"{original} / {translated}" if original else translated
        return str(soup).encode("utf-8")
