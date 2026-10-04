"""Artifact-level attestation check (artifact-level attestation check).

The attestation layer says, per element, what was realized -- a verified
translation, a reconstructed asset, or the source kept whole. This closes the
loop on the *delivered artifact*: does the file a reader gets actually carry the
realization the attestation claims?

It is deliberately narrow. Only *text* elements are checked, and only by their
delivered text: an asset's realization is markup (a formula's LaTeX, a table's
grid), which is drawn rather than spelled out, so its fidelity is a pixel claim
the witnesses make -- not something a text probe can confirm. A text element
attested above the floor must show its translation in the artifact; one kept at
the floor must show its source.

During the migration this measures the "output fidelity ≥ legacy baseline" criterion on every real
delivery instead of only in a corpus run. It is a *report*: a missing element is
surfaced, never silently tolerated, and never allowed to sink the artifact.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ubt.core.cjk_ranges import CJK_BMP_CLASS
from ubt.core.job_options import SidecarKind, companion_path, sidecar_path
from ubt.core.qe.fast_pass import strip_rehearsal_marker
from ubt.model.ast import Document, Element
from ubt.model.fidelity import Attestation, Fidelity
from ubt.model.span import CanonicalSource
from ubt.render.overlay_backend import source_slice

#: Shortest delivered text worth probing for. Below this a match is noise -- a
#: bullet, a lone glyph -- and the check says nothing either way.
_MIN_PROBE = 4
#: Share of an element's tokens that must appear in the artifact. A reflowing
#: backend re-breaks lines (and may hyphenate a long word), so an exact substring
#: match would report the very elements it placed; token overlap measures the
#: realization without mistaking layout for loss.
_MIN_OVERLAP = 0.6

_CJK_RANGE = CJK_BMP_CLASS
_TOKEN_RE = re.compile(rf"[{_CJK_RANGE}]|[^\s\W_]+(?:-[^\s\W_]+)*")


def _tokenize(text: str) -> list[str]:
    """Tokenize text into probe units: words in spaced scripts, individual CJK glyphs."""
    return _TOKEN_RE.findall(text)


@dataclass(frozen=True, slots=True)
class ElementCheck:
    """One text element's claim, and whether the artifact carries it."""

    element_id: str
    fidelity: Fidelity
    page: int
    present: bool


@dataclass(frozen=True, slots=True)
class ArtifactReport:
    """The artifact's agreement with the attestations, element by element."""

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


def _normalize(text: str) -> str:
    """Whitespace-collapsed text, so a reflowed line break is not a mismatch."""
    return " ".join(text.split())


def _artifact_tokens(artifact: Path) -> frozenset[str]:
    """Every token in the artifact (pages joined).

    Joined across pages on purpose: a bilingual artifact interleaves source and
    target pages, so a realization need not sit on its source page number.
    Extracts native textpage ranges via pdfium (covering rotated margin/header
    stamps as well as standard horizontal lines), falling back to line extraction
    if needed.
    """
    from ubt.adapters.pdf import pdf_struct, textgeom
    from ubt.adapters.pdf.pdfium_gate import open_document, pdfium_serialized

    @pdfium_serialized
    def _read_all_text(p: Path) -> str:
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

    try:
        raw = _read_all_text(artifact)
    except Exception:
        with pdf_struct.open_pdf(artifact) as pdf:
            pages = len(pdf.pages)
        chunks: list[str] = []
        for page_no in range(1, pages + 1):
            lines, _ = textgeom.extract_lines(artifact, page_no)
            chunks.append(" ".join(line.text for line in lines))
        raw = " ".join(chunks)

    return frozenset(_tokenize(raw))


def _present(expected: str, tokens: frozenset[str]) -> bool:
    """Whether the artifact carries enough of ``expected`` to call it placed."""
    wanted = _tokenize(expected)
    if len(expected) < _MIN_PROBE or not wanted:
        return True
    hits = sum(1 for token in wanted if token in tokens)
    return hits / len(wanted) >= _MIN_OVERLAP


def _expected(
    element: Element,
    fidelity: Fidelity,
    source: CanonicalSource,
    delivered: Mapping[str, str],
) -> str:
    """The text the attestation says the artifact carries for this element.

    A leading rehearsal prefix is stripped to match the renderer, which never
    places it (see :func:`ubt.core.qe.fast_pass.strip_rehearsal_marker`); without
    this the audit demanded a marker the artifact is right not to contain.
    """
    if fidelity > Fidelity.PRESERVED_OPAQUE:
        return strip_rehearsal_marker(delivered.get(element.id, ""))
    return strip_rehearsal_marker(source_slice(element, source))


def check_artifact(
    document: Document,
    attestations: Sequence[Attestation],
    delivered: Mapping[str, str],
    artifact_pdf: str | Path,
) -> ArtifactReport:
    """Check every attested *text* element against the delivered artifact.

    ``delivered`` is the run's ``element id -> target`` map (the same one the
    backend is built with); a text element attested above the floor is expected to
    show its target, one at the floor its source. Assets are not text claims and
    are skipped.
    """
    text = _artifact_tokens(Path(artifact_pdf))
    by_id = {attestation.element_id: attestation for attestation in attestations}
    checks: list[ElementCheck] = []
    for element in document.elements:
        if not element.is_text:
            continue
        attestation = by_id.get(element.id)
        if attestation is None:
            continue
        expected = _normalize(_expected(element, attestation.fidelity, document.source, delivered))
        checks.append(
            ElementCheck(
                element.id,
                attestation.fidelity,
                element.span.page,
                _present(expected, text),
            )
        )
    return ArtifactReport(total=len(checks), checks=tuple(checks))


@dataclass(frozen=True, slots=True)
class DeliveredArtifact:
    """One delivered artifact and the files that hang off it.

    Two artifact identities coexist in the export stage: ``target_output`` (the
    path the primary render was *asked* for, which the PE queue keys on) and
    ``rendered_path`` (the file the adapter actually *returned*, which the visual
    gate may rewrite and which every sidecar and companion keys on). Keeping both
    in one value means the stage never has to remember which helper takes which,
    and the sibling/companion naming has a single owner instead of ad-hoc
    ``with_name`` calls spread through the stage.
    """

    rendered_path: Path
    target_output: Path

    def sidecar(self, kind: SidecarKind) -> Path:
        """The derived report belonging to the returned artifact."""
        return sidecar_path(self.rendered_path, kind)

    def companion(self, suffix: str) -> Path:
        """A companion document beside the returned artifact (e.g. ``.xliff``)."""
        return companion_path(self.rendered_path, suffix)

    def sibling(self, suffix: str) -> Path:
        """A second artifact beside the *requested* target (e.g. ``_rigid.pdf``)."""
        out = self.target_output
        return out.with_name(f"{out.stem}{suffix}{out.suffix}")


def delivered_artifact(rendered_path: str | Path, target_output: str | Path) -> DeliveredArtifact:
    """Name the two artifact identities once, where the render returns."""
    return DeliveredArtifact(Path(rendered_path), Path(target_output))


__all__ = [
    "ArtifactReport",
    "DeliveredArtifact",
    "ElementCheck",
    "check_artifact",
    "delivered_artifact",
]
