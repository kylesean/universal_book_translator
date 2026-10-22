"""Delivered-artifact presence check + text extraction (PDF forensics).

Lives beside the other artifact forensics (:mod:`ubt.adapters.pdf.artifact_parity`):
the extractor is PDF-adapter machinery (pdfium textpages with a textgeom line
fallback). The compiler layer's old copy reached into the adapters from the
artifact module; the layering acceptance tool forbids that edge, so the check
moved here beside its extraction machinery.

:func:`check_artifact` asks "does the delivered PDF actually carry each text
block's realization?". The report/element types moved with it; the path
bookkeeping half (``DeliveredArtifact``, in the compiler package) stays in
the compiler layer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.qe.fast_pass import strip_rehearsal_marker

#: Shortest delivered text worth probing for. Below this a match is noise -- a
#: bullet, a lone glyph -- and the check says nothing either way.
_MIN_PROBE = 4
#: Share of an element's tokens that must appear in the artifact. A reflowing
#: backend re-breaks lines (and may hyphenate a long word), so an exact substring
#: match would report the very elements it placed; token overlap measures the
#: realization without mistaking layout for loss.
_MIN_OVERLAP = 0.6

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[^\s\W_]+(?:-[^\s\W_]+)*")

__all__ = ["ElementCheck", "ArtifactReport", "check_artifact", "artifact_tokens"]


def _tokenize(text: str) -> list[str]:
    """Tokenize text into probe units: words in spaced scripts, individual CJK glyphs."""
    return _TOKEN_RE.findall(text)


@dataclass(frozen=True, slots=True)
class ElementCheck:
    """One text block's claim, and whether the artifact carries it."""

    element_id: str
    page: int
    present: bool


@dataclass(frozen=True, slots=True)
class ArtifactReport:
    """The artifact's agreement with the delivered text, block by block."""

    total: int
    checks: tuple[ElementCheck, ...]

    @property
    def missing(self) -> tuple[ElementCheck, ...]:
        """Text elements whose realization the artifact does not carry."""
        return tuple(check for check in self.checks if not check.present)

    @property
    def passed(self) -> bool:
        return not self.missing

    def summary_line(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] artifact carries {self.total - len(self.missing)}/{self.total} "
            f"text realization(s)"
        )


def _read_page_texts(artifact: Path) -> list[str]:
    """Each page's text, in page order, via pdfium textpages behind the C gate."""
    from ubt.adapters.pdf.pdfium_gate import open_document, pdfium_serialized

    @pdfium_serialized
    def _read(p: Path) -> list[str]:
        chunks: list[str] = []
        with open_document(p) as pdf:
            for idx in range(len(pdf)):
                page = pdf[idx]
                try:
                    tp = page.get_textpage()
                    try:
                        chunks.append(tp.get_text_range())
                    finally:
                        tp.close()
                finally:
                    page.close()
        return chunks

    return _read(artifact)


def _read_all_text_pages(artifact: Path) -> str:
    """All pages' text via pdfium textpages, serialized behind the C gate."""
    return " ".join(_read_page_texts(artifact))


def page_tokens(artifact: Path) -> list[frozenset[str]]:
    """Per-page token sets (index 0 = page 1), for a page-scoped presence probe.

    The whole-book token set is what made the audit blind in one direction and
    noisy in the other. Blind: a paragraph lost from a 300-page book still had
    every one of its common CJK glyphs somewhere in the remaining 299 pages, so
    the probe reported it present. Noisy: a wrapper's tokens (``mark``,
    ``class``, ``Repair``, ``500``) never appear on any page, so a block that
    *was* placed failed.

    Falls back to line extraction when the textpage read raises, matching
    :func:`artifact_tokens`.
    """
    from ubt.adapters.pdf import pdf_struct, textgeom

    try:
        pages = _read_page_texts(artifact)
    except Exception:
        with pdf_struct.open_pdf(artifact) as pdf:
            page_count = len(pdf.pages)
        pages = []
        for page_no in range(1, page_count + 1):
            lines, _ = textgeom.extract_lines(artifact, page_no)
            pages.append(" ".join(line.text for line in lines))
    return [frozenset(_tokenize(text)) for text in pages]


def artifact_tokens(artifact: Path) -> frozenset[str]:
    """Every token in the artifact (pages joined).

    Joined across pages on purpose: a bilingual artifact interleaves source and
    target pages, so a realization need not sit on its source page number.
    Extracts native textpage ranges via pdfium (covering rotated margin/header
    stamps as well as standard horizontal lines), falling back to line extraction
    if needed.
    """
    from ubt.adapters.pdf import pdf_struct, textgeom

    try:
        raw = _read_all_text_pages(artifact)
    except Exception:
        with pdf_struct.open_pdf(artifact) as pdf:
            pages = len(pdf.pages)
        chunks: list[str] = []
        for page_no in range(1, pages + 1):
            lines, _ = textgeom.extract_lines(artifact, page_no)
            chunks.append(" ".join(line.text for line in lines))
        raw = " ".join(chunks)

    return frozenset(_tokenize(raw))


def _normalize(text: str) -> str:
    """Whitespace-collapsed text, so a reflowed line break is not a mismatch."""
    return " ".join(text.split())


def _present(expected: str, tokens: frozenset[str]) -> bool:
    """Whether the artifact carries enough of ``expected`` to call it placed."""
    wanted = _tokenize(expected)
    if len(expected) < _MIN_PROBE or not wanted:
        return True
    hits = sum(1 for token in wanted if token in tokens)
    return hits / len(wanted) >= _MIN_OVERLAP


def _expected(block: IRBlock, translations: Mapping[str, str]) -> str:
    """The text the artifact should carry for this block.

    The map value is used as given — the export stage builds it with
    :func:`ubt.render.delivered_text.delivered_text`, which already strips the
    pipeline's ``<mark>`` wrappers. The *source* fallback goes through the same
    rule for the same reason: a quarantined block's ``target_text`` is a
    ``<mark class="ubt-blocked-human">`` wrapper around the source, and probing
    its markup tokens (``mark``, ``class``, ``title``, …) asked the artifact for
    text no renderer ever draws, reporting a placed block as missing.

    A leading rehearsal prefix is stripped before comparison: a rehearsal may
    render the marker or drop it, and the audit must not fail on that difference
    (see :func:`ubt.core.qe.fast_pass.strip_rehearsal_marker`).
    """
    target = (translations.get(block.id) or "").strip()
    if target:
        # The map is normally plain text (the export stage builds it with
        # ``delivered_text``), but a ledger row written before that fix, or a
        # caller passing ``target_text`` directly, can still carry a wrapper.
        # Probe the text the renderer draws, never the markup.
        from ubt.render.delivered_text import is_any_mark, unwrap_mark

        if is_any_mark(target):
            target = unwrap_mark(target).strip()
        return strip_rehearsal_marker(target)
    from ubt.render.delivered_text import delivered_text

    return strip_rehearsal_marker(delivered_text(block))


def check_artifact(
    blocks: Sequence[IRBlock],
    translations: Mapping[str, str],
    artifact_pdf: str | Path,
) -> ArtifactReport:
    """Check every delivered *text* block against the artifact.

    ``translations`` is the run's ``element id -> placed target`` map; a text
    block with a placed target is expected to show it, one kept in the source its
    source. Assets are not text claims and are skipped.

    A heading the renderer deliberately drops as running chrome (an arXiv or
    journal stamp, a submission date) is skipped too: the audit must not demand a
    realization the renderer is right not to place. The drop decision is asked of
    its single owner (:func:`ubt.render.delivered_text.dropped_page_furniture`),
    imported lazily so the adapters→render edge stays off the runtime import graph.
    """
    from ubt.render.delivered_text import dropped_page_furniture

    pages = page_tokens(Path(artifact_pdf))
    # A realization need not sit on the *same* page number as its source: a
    # bilingual artifact interleaves source and target pages, and a reflowing
    # layout moves text across page boundaries. Probing the whole book is the
    # safe direction for that, but it makes the probe blind — every common glyph
    # of a lost paragraph is somewhere in a long book. So probe the block's own
    # page first, then its neighbours, and only then the whole book: a block
    # that really was placed is found in one of the first two, and one that was
    # lost is not found in any of them.
    whole_book = frozenset().union(*pages) if pages else frozenset()
    checks: list[ElementCheck] = []
    for block in blocks:
        if block.block_type in (BlockType.FORMULA, BlockType.TABLE, BlockType.IMAGE):
            continue
        if dropped_page_furniture(block):
            continue
        expected = _normalize(_expected(block, translations))
        page_number = block.bbox.page if block.bbox is not None else 0
        checks.append(
            ElementCheck(
                block.id,
                page_number,
                _present_on_pages(expected, pages, page_number, whole_book),
            )
        )
    return ArtifactReport(total=len(checks), checks=tuple(checks))


def _present_on_pages(
    expected: str,
    pages: list[frozenset[str]],
    page_number: int,
    whole_book: frozenset[str],
) -> bool:
    """Whether ``expected`` is placed, looking near its source page first.

    The page window is what closes the false-pass: a lost paragraph shares its
    glyphs with the rest of the book, but not with the two pages around its own
    position. The whole-book fallback is kept last so a block the layout moved
    far from its source page is still reported present rather than as a false
    failure.
    """
    if not pages:
        return _present(expected, whole_book)
    if 1 <= page_number <= len(pages):
        for candidate in (page_number, page_number + 1, page_number - 1):
            if 1 <= candidate <= len(pages) and _present(expected, pages[candidate - 1]):
                return True
        return False
    return _present(expected, whole_book)
