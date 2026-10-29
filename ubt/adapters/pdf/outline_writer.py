"""Write translated chapter bookmarks back into the delivered PDF.

The PDF family already *reads* the source's bookmark tree
(:func:`ubt.adapters.pdf.docling_outline.read_pdf_outline_entries`) to recover
the heading levels Docling's TOC pass drops. That read half has an obvious
counterpart that did not exist: the delivered artifact had **no** outline at
all, so a 400-page translated book opened in a reader showed a flat wall of
pages and the translated chapter titles were reachable only by scrolling. The
source book had a working sidebar; the translation lost it. That is a
regression a reader feels immediately and cannot work around.

Two facts make this cheap rather than speculative:

* The facsimile and page-locked channels preserve the source **page count 1:1**
  (``_render_facsimile`` iterates ``range(1, total_pages + 1)`` and emits exactly
  one page per source page), so a heading's measured ``bbox.page`` *is* its page
  in the delivered artifact. No dest-to-dest remapping is needed — which is
  precisely why this is safe here and would not be under the reflow channel,
  whose page count changes with the target language's length.
* :class:`pikepdf.Outline` exposes ``add``/``root`` (verified on
  ``pikepdf 10.13.0.post1``), so writing an outline is one library call, not a
  hand-built ``/Outlines`` dictionary tree.

The entries are built from **IR blocks, not from the source outline**, because
the deliverable needs the *translated* titles: re-using the source's titles
would ship a Chinese book whose sidebar reads English. ``element.level``
supplies the real nesting (the PDF adapters merge the source outline into the
blocks at ingest, see ``assign_heading_levels_from_outline``), so the hierarchy
reconstructed here matches the book's own structure rather than a guessed one.

Deliberately **not** applied to the reflow channel: its page count is not the
source's, so ``bbox.page`` names a page that no longer exists and every bookmark
would point at the wrong place. Silently mislabelled navigation is worse than
none, so the caller gates on the delivery mode and this module reports
``skipped`` instead of guessing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ubt.core.ir.models import IRBlock

logger = logging.getLogger(__name__)

#: Deepest heading level written as a bookmark. A book's ``####`` runs are
#: usually inline emphasis rather than navigable structure, and a sidebar with
#: 600 sub-sub-entries is less usable than one with 60. Levels deeper than this
#: are dropped, not flattened into their parent (flattening would emit a
#: duplicate title on every deeper heading).
MAX_OUTLINE_LEVEL = 3

#: Cap on the number of bookmarks. ``/Outlines`` is cheap per entry, but a
#: heading-dense 900-page proceedings volume can produce thousands, and every
#: reader pays for parsing them on open. Far above any real book's chapter count.
MAX_OUTLINE_ENTRIES = 2000


@dataclass(frozen=True, slots=True)
class OutlineEntry:
    """One bookmark: its translated title, its 1-based page, its 1-based depth."""

    title: str
    page: int
    level: int


def build_outline_entries(
    blocks: Sequence[IRBlock],
    *,
    max_page: int,
    max_level: int = MAX_OUTLINE_LEVEL,
    max_entries: int = MAX_OUTLINE_ENTRIES,
) -> list[OutlineEntry]:
    """Translated heading blocks -> ordered bookmarks.

    ``max_page`` is the delivered artifact's page count. It is not decoration:
    a heading whose measured page exceeds it (a stale block from a resume, a
    page range that was narrowed) must be dropped rather than clamped, because a
    bookmark pointing at a page that does not exist makes readers error on open.

    A block with no target text is skipped: a bookmark has to say something, and
    falling back to the source text would mix languages in the sidebar that the
    rest of the book was translated out of.
    """
    from ubt.core.ir.models import BlockType
    from ubt.render.delivered_text import delivered_text

    entries: list[OutlineEntry] = []
    seen: set[tuple[str, int]] = set()
    for block in blocks:
        if block.block_type is not BlockType.HEADING:
            continue
        bbox = block.bbox
        if bbox is None or bbox.page <= 0:
            continue
        page = int(bbox.page)
        if page > max_page:
            logger.debug(
                "Outline: dropping heading on page %d beyond artifact end %d", page, max_page
            )
            continue
        # ``delivered_text`` unwraps the ``<mark>`` quarantine wrappers, which
        # must not reach a bookmark title as literal markup.
        title = " ".join((delivered_text(block) or "").split())
        if not title:
            continue
        level = _effective_level(block)
        if level > max_level:
            continue
        # A heading split across a page break arrives as two blocks with the
        # same title on adjacent pages; the sidebar should list it once.
        key = (title, page)
        if key in seen:
            continue
        seen.add(key)
        entries.append(OutlineEntry(title=title, page=page, level=level))
        if len(entries) >= max_entries:
            break
    return entries


def _effective_level(block: IRBlock) -> int:
    """The heading's declared depth, or 1 when the extractor recorded none."""
    level = getattr(block.element, "level", None)
    if isinstance(level, int) and level > 0:
        return level
    return 1


def write_outline(
    pdf_path: Path,
    entries: Sequence[OutlineEntry],
    *,
    page_mode: bool = True,
) -> int:
    """Replace the PDF's bookmark tree with ``entries``; return how many landed.

    Returns ``0`` and changes nothing when there are no entries or the file
    cannot be opened — an outline is navigation, not content, so a failure here
    must never fail a render that already produced a good artifact.

    The existing tree is cleared rather than appended to: the delivered PDF is a
    fresh Typst compile, so anything already in it is either absent or the
    source's untranslated titles leaking through an embedded layer.

    ``page_mode`` sets ``/PageMode /UseOutlines`` so a reader opens with the
    sidebar visible. Without it the bookmarks exist but every reader's default
    hides them, and the feature reads as unimplemented to the person holding the
    file.
    """
    if not entries:
        return 0
    try:
        import pikepdf
    except Exception as exc:  # pragma: no cover - import failure
        logger.debug("pikepdf unavailable; PDF outline not written: %s", exc)
        return 0

    try:
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                return 0
            usable = [
                OutlineEntry(title=e.title, page=e.page, level=max(1, e.level))
                for e in entries
                if 1 <= e.page <= total_pages
            ]
            if not usable:
                return 0
            with pdf.open_outline() as outline:
                outline.root.clear()
                # ``Any`` rather than ``object``: ``OutlineItem.children`` is a
                # live proxy into the PDF's object graph, and pikepdf's stubs
                # describe the container, not the items it yields. Annotating it
                # as ``object`` made the parent-append below a mypy error on a
                # call that is correct at runtime (verified by the round-trip
                # tests, which read the nesting back out of a saved file).
                stack: list[tuple[int, Any]] = []
                for entry in usable:
                    while stack and stack[-1][0] >= entry.level:
                        stack.pop()
                    item = pikepdf.OutlineItem(entry.title, entry.page - 1)
                    if stack:
                        stack[-1][1].children.append(item)
                    else:
                        outline.root.append(item)
                    stack.append((entry.level, item))
            if page_mode:
                pdf.Root.PageMode = pikepdf.Name.UseOutlines
            pdf.save()
            return len(usable)
    except Exception as exc:
        logger.warning("Could not write the PDF outline to %s: %s", pdf_path, exc)
        return 0


def write_outline_from_blocks(
    pdf_path: Path,
    blocks: Sequence[IRBlock],
    *,
    expected_pages: int | None = None,
) -> int:
    """Build entries from ``blocks`` and write them, holding the page invariant.

    ``expected_pages`` is the source page count the render was built from. The
    caller passes it for the page-preserving channels so the 1:1 assumption is
    *checked* rather than trusted: if the artifact's page count differs, the
    bookmark pages no longer name the pages the reader will see, and writing
    them would mislabel the navigation. In that case nothing is written and the
    reason is logged.
    """
    try:
        import pikepdf

        with pikepdf.open(pdf_path) as pdf:
            artifact_pages = len(pdf.pages)
    except Exception as exc:
        logger.debug("Cannot read artifact page count for outline: %s", exc)
        return 0
    if expected_pages is not None and artifact_pages != expected_pages:
        logger.warning(
            "PDF outline not written: artifact has %d pages but the render was "
            "built from %d source pages, so block page numbers do not name the "
            "delivered pages.",
            artifact_pages,
            expected_pages,
        )
        return 0
    entries = build_outline_entries(blocks, max_page=artifact_pages)
    return write_outline(pdf_path, entries)


__all__ = [
    "MAX_OUTLINE_ENTRIES",
    "MAX_OUTLINE_LEVEL",
    "OutlineEntry",
    "build_outline_entries",
    "write_outline",
    "write_outline_from_blocks",
]
