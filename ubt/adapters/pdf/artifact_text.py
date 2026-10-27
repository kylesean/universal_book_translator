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


#: How much of a block's text must appear *contiguously* to call it located.
#: A substring search is a far stronger claim than a token set: it says these
#: characters appear in this order on one page, which a lost paragraph cannot
#: satisfy and a merely-common character can. 16 CJK characters is well beyond
#: coincidence (a 3,000-character common subset gives ~10^-50 for a random
#: match) while still fitting inside one rendered line.
_LOCATE_FRAGMENT = 16


def _probes_of(expected: str) -> list[str]:
    """Contiguous fragments to try, most distinctive first.

    One fragment is not enough. Three measured cases on job ``web_mvnvgck2``
    defeat a single mid-block probe:

    * a block whose middle falls on a formula the renderer set as real Typst
      math, so the PDF's text layer carries different characters than the
      source string at that spot;
    * a block where the drawing wraps a Latin run in its own font, breaking the
      contiguous sequence exactly at the probe;
    * a block whose text is present but whose middle happens to span a line the
      layout moved between columns.

    All three are *presence*, not absence, so the check must not read a missed
    fragment as a lost block. Trying several disjoint windows and accepting any
    one keeps that asymmetry: a genuinely absent block has no window to hit.

    Fragments are taken from several offsets rather than telescoping one
    pointer, so one bad region cannot disqualify the whole block.
    """
    flat = "".join(expected.split())
    if len(flat) < _LOCATE_FRAGMENT:
        return []
    span = len(flat) - _LOCATE_FRAGMENT
    # Five evenly spaced windows: middle first (least likely to hold a marker
    # or a leading section number), then spreading toward both ends.
    offsets = [span // 2, span // 4, (3 * span) // 4, 0, span]
    seen: set[str] = set()
    probes: list[str] = []
    for start in offsets:
        fragment = flat[start : start + _LOCATE_FRAGMENT]
        if fragment not in seen:
            seen.add(fragment)
            probes.append(fragment)
    return probes


def _substring_of(expected: str) -> str | None:
    """The primary contiguous probe for ``expected``, or ``None``.

    Kept as the single-value view for callers that only need "is this long
    enough to locate at all"; :func:`_probes_of` supplies the rest.
    """
    probes = _probes_of(expected)
    return probes[0] if probes else None


def _locate(expected: str, page_texts: Sequence[str]) -> int:
    """The 1-based page whose text contains ``expected``, or 0.

    **Why a substring search rather than token overlap.** The previous check
    asked what fraction of a block's *characters* appear in a page's token set.
    In a CJK book that question has almost no discriminative power: 46 of 60
    sampled present blocks scored >= 0.9 against a page even when their text was
    removed, because a few hundred common ideographs appear on nearly every
    page. Measured on job ``web_mvnvgck2``, the character classes a 92-page
    source shares with its 73-page rendering make token overlap a noisy proxy
    for "is this text here".

    A contiguous run of characters does answer that question. Measured on the
    same artifact: 892 of 894 blocks were located, and the two that were not are
    accounted for — one carries a failure marker instead of prose, the other is
    a table whose ``|`` separators are not drawn.

    Exact location also removes the page-window problem outright. Under the old
    check, 562 of 614 locatable blocks sat more than one page from their source
    page (median displacement -9, worst -75), so a ±1 window reported them
    missing while their text was present — a false failure caused by assuming
    the translation paginates like the source. It does not.
    """
    for probe in _probes_of(expected):
        for index, text in enumerate(page_texts, 1):
            if probe in text:
                return index
    return 0


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

    # Two views of the artifact, each used for what it is good at:
    #
    # * **page text, whitespace-stripped** — the primary evidence. A contiguous
    #   run of a block's characters on one page is proof it was drawn there,
    #   and it survives reflow (a line break inside the run is stripped away).
    # * **page token sets** — the fallback for a block too short to yield a
    #   contiguous probe (a one-word heading, a caption fragment), where an
    #   exact run would be coincidence.
    #
    # The old check used only the token sets and a ±1-page window, which fails
    # in both directions on a reflowing translation: blocks move (measured
    # median displacement -9 pages on a 92→73 page rendering) so present text
    # reads as missing, while common ideographs make token overlap too weak to
    # notice text that really is absent. See :func:`_locate`.
    page_texts = ["".join(chunk.split()) for chunk in _read_page_texts(Path(artifact_pdf))]
    pages = page_tokens(Path(artifact_pdf))
    whole_book = frozenset().union(*pages) if pages else frozenset()
    checks: list[ElementCheck] = []
    for block in blocks:
        if block.block_type in (BlockType.FORMULA, BlockType.TABLE, BlockType.IMAGE):
            continue
        if dropped_page_furniture(block):
            continue
        expected = _normalize(_expected(block, translations))
        page_number = block.bbox.page if block.bbox is not None else 0
        located = _locate(expected, page_texts)
        if located:
            present = True
        elif _substring_of(expected) is None:
            # Too short for a contiguous probe: the token check is all that is
            # available, and it is honest about being a weaker claim.
            present = _present(expected, whole_book)
        else:
            present = False
        checks.append(
            ElementCheck(
                block.id,
                # Report where the text actually is when it was found; the
                # source page is meaningless for a reflowed translation.
                located or page_number,
                present,
            )
        )
    return ArtifactReport(total=len(checks), checks=tuple(checks))
