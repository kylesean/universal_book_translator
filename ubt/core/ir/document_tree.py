"""Section tree: a *view* over IRBlocks, not a second document model.

RFC-2026-02 §3 sketched a ``DocumentTree`` with its own ``Node`` dataclass. That
shape was written before this module's author read ``ubt/core/ir/models.py``,
which already owns the document model:

* ``IRBlock`` carries a typed ``element`` (single source of truth for
  structure — including a heading's ``level``), ``spine_index``, provenance and
  execution state;
* ``ContentGraph`` already states the delivery contract over those blocks.

Adding a parallel ``Node``/``DocumentTree`` would create a second source of
truth for the same facts — the exact failure mode this rearchitecture exists to
remove. So this module provides what was actually *missing*: a **grouping** of
the existing blocks into sections, so translation can run at section scope
(§2.3) instead of one block at a time.

Nothing here is stored. The section list is derived on demand from a block
sequence and holds no state of its own; the ledger keeps the block-level record
that resume, billing and PE export depend on (RFC §7.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ubt.core.ir.models import IRBlock

#: Block types that open a new section. Anything else joins the section that is
#: already open, so a heading-less preamble (title page, abstract) still gets a
#: unit to translate.
_SECTION_OPENERS = frozenset({"heading", "title"})


@dataclass(slots=True)
class Section:
    """A contiguous run of blocks sharing one heading context.

    ``heading`` is the block that opened the section (``None`` for the
    preamble). ``blocks`` holds every block in reading order, including the
    heading itself, so a caller that concatenates their text gets the heading
    back and the numbering stays visible to the model.
    """

    index: int
    heading: IRBlock | None
    blocks: list[IRBlock] = field(default_factory=list)
    level: int = 0
    #: Heading path from the document root, e.g. ``["2", "2.1"]``. Derived from
    #: the heading text's own numbering, not from a counter, so it stays correct
    #: when the source skips a level.
    path: tuple[str, ...] = ()
    #: The same ancestry as :attr:`path` but by heading *title*, self last, e.g.
    #: ``("2 Switch Transformer", "2.1 Simplifying Sparse Routing")``. This is
    #: what :func:`heading_trail` renders: a model that sees only the leaf title
    #: "2.1 Simplifying" cannot tell which chapter it sits in, which is the
    #: antecedent a back-reference needs (RFC-2026-02 §2.3).
    trail: tuple[str, ...] = ()

    @property
    def title(self) -> str:
        return (self.heading.source_text if self.heading is not None else "").strip()

    @property
    def text(self) -> str:
        """Concatenated source text of every block in the section."""
        return "\n\n".join((b.source_text or "").strip() for b in self.blocks if b.source_text)

    @property
    def translatable(self) -> list[IRBlock]:
        """Blocks in this section that the pipeline decided to translate."""
        return [b for b in self.blocks if not _skips_translation(b)]

    def __len__(self) -> int:
        return len(self.blocks)


def _skips_translation(block: IRBlock) -> bool:
    """Whether the pipeline decided to keep this block in the source language."""
    if block.skip_translate:
        return True
    # The policy verdict is authoritative when it exists; ``skip_translate`` is
    # only the extractor's default (see IRBlock's own docstring).
    return block.policy_translate is False


def _heading_number(text: str) -> str:
    """Leading section number of a heading ('2.1 Simplifying' -> '2.1')."""
    stripped = (text or "").strip()
    for token in stripped.split():
        candidate = token.rstrip(".:")
        if candidate and all(c.isdigit() or c == "." for c in candidate):
            return candidate
        break
    return ""


def _effective_level(block: IRBlock, fallback: int) -> int:
    """Heading level of a block, preferring a real level over the default.

    The level lives on the typed element (``block.element.level``), not on
    ``IRBlock`` — that model carries execution state, and structure belongs to
    ``element``. Reading ``block.level`` (which does not exist) made every
    heading fall back to 1, so every heading opened a *new* section and no
    sub-section ever nested: a document's ``1.1`` read as a sibling of ``1``
    instead of a child, and the prompt's heading trail lost the ancestor
    (RFC-2026-02 §2.3 is the whole reason this grouping exists). Found by
    replacing the duck-typed stand-in in this module's tests with a real
    ``IRBlock``, which is the only way a rename in the IR can be caught.

    A PDF adapter that has not merged the source outline still writes ``1`` for
    every heading (Docling emits ``#`` regardless of depth), so the fallback
    stays; a level above ``max_level`` is handled by the caller.
    """
    level = getattr(block.element, "level", None)
    if isinstance(level, int) and level > 0:
        return level
    return fallback


def build_section_tree(blocks: Sequence[IRBlock], *, max_level: int = 2) -> list[Section]:
    """Group ``blocks`` into sections in reading order.

    A heading at or above ``max_level`` opens a new section; a deeper heading
    (``###`` inside a ``##`` section) stays inside the open one, so a section's
    text covers a coherent span rather than splitting on every sub-sub-heading.

    Returns at least one section whenever ``blocks`` is non-empty: a document
    with no headings yields a single preamble section, which is what makes the
    section-scope translation path work for un-headed input (a plain article,
    an EPUB chapter).
    """
    sections: list[Section] = []
    current: Section | None = None
    #: Ancestors as ``(level, path_token, title_token)``. The level travels with
    #: the token because the depth of a heading is not the size of the stack: a
    #: source that skips a level (``1 Intro`` then ``1.0.1 Sub``) leaves two
    #: entries for a level-3 heading, so trimming by ``len(stack)`` never pops
    #: the previous sibling and each one nests under the last.
    stack: list[tuple[int, str, str]] = []

    for block in blocks:
        block_type = str(getattr(block.block_type, "value", block.block_type) or "")
        is_opener = block_type in _SECTION_OPENERS and _effective_level(block, 1) <= max_level

        if is_opener or current is None:
            if is_opener:
                level = _effective_level(block, 1)
                number = _heading_number(block.source_text)
                title = block.source_text.strip()
                # Pop every ancestor at or below this heading's level, comparing
                # *levels* rather than stack size (see ``stack`` above).
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, number or title[:40], title[:80]))
            current = Section(
                index=len(sections),
                heading=block if is_opener else None,
                level=_effective_level(block, 0) if is_opener else 0,
                path=tuple(entry[1] for entry in stack),
                trail=tuple(entry[2] for entry in stack),
            )
            sections.append(current)

        current.blocks.append(block)

    return sections


def section_texts(sections: Sequence[Section]) -> list[str]:
    """The translation units: one source-text string per section."""
    return [s.text for s in sections if s.text.strip()]


def section_index(blocks: Sequence[IRBlock], *, max_level: int = 2) -> dict[str, Section]:
    """Map every block id to the section that contains it.

    Built once per batch and read per chunk. ``chunk_by_section`` guarantees a
    chunk never straddles a heading, but a chunk that does not itself contain
    the heading still needs to know which section it is in: the model otherwise
    sees "this approach" with no antecedent (RFC-2026-02 §2.3).
    """
    mapping: dict[str, Section] = {}
    for section in build_section_tree(blocks, max_level=max_level):
        for block in section.blocks:
            mapping[block.id] = section
    return mapping


def heading_trail(section: Section) -> str:
    """The heading ancestry of one section: ``"2 Switch Transformer > 2.1 …"``.

    Ancestors come from :func:`build_section_tree`, which trims the trail to the
    heading's own depth, so a skipped level in the source cannot leave a stale
    parent here either. The leaf title alone is not enough: a chunk under "2.1
    Simplifying" whose own heading is in an earlier chunk reaches the model with
    no idea which chapter it is in, and a back-reference loses its antecedent
    (RFC-2026-02 §2.3).
    """
    if section.heading is None:
        return ""
    return " > ".join(section.trail or (section.title,))


def section_context_for(
    blocks: Sequence[IRBlock],
    chunk: Sequence[IRBlock],
    *,
    max_level: int = 2,
) -> str:
    """The heading trail for a chunk, from the section that contains it.

    Returns ``""`` when the chunk has no heading context (a heading-less
    document), so the caller can pass the result straight through.

    A chunk that already contains its own heading gets the trail too, and that
    is deliberate: the prompt is then identical for every chunk of a section,
    so the content-addressed cache key does not change halfway through.
    """
    if not chunk:
        return ""
    section = section_index(blocks, max_level=max_level).get(chunk[0].id)
    if section is None:
        return ""
    return heading_trail(section)
