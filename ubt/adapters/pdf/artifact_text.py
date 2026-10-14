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


def _read_all_text_pages(artifact: Path) -> str:
    """All pages' text via pdfium textpages, serialized behind the C gate."""
    from ubt.adapters.pdf.pdfium_gate import open_document, pdfium_serialized

    @pdfium_serialized
    def _read(p: Path) -> str:
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
        return " ".join(chunks)

    return _read(artifact)


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

    A leading rehearsal prefix is stripped to match the renderer, which never
    places it (see :func:`ubt.core.qe.fast_pass.strip_rehearsal_marker`); without
    this the audit demanded a marker the artifact is right not to contain.
    """
    target = (translations.get(block.id) or "").strip()
    if target:
        return strip_rehearsal_marker(target)
    return strip_rehearsal_marker(block.source_text or "")


def check_artifact(
    blocks: Sequence[IRBlock],
    translations: Mapping[str, str],
    artifact_pdf: str | Path,
) -> ArtifactReport:
    """Check every delivered *text* block against the artifact.

    ``translations`` is the run's ``element id -> placed target`` map; a text
    block with a placed target is expected to show it, one kept in the source its
    source. Assets are not text claims and are skipped.
    """
    text = artifact_tokens(Path(artifact_pdf))
    checks: list[ElementCheck] = []
    for block in blocks:
        if block.block_type in (BlockType.FORMULA, BlockType.TABLE, BlockType.IMAGE):
            continue
        expected = _normalize(_expected(block, translations))
        checks.append(
            ElementCheck(
                block.id,
                block.bbox.page if block.bbox is not None else 0,
                _present(expected, text),
            )
        )
    return ArtifactReport(total=len(checks), checks=tuple(checks))
