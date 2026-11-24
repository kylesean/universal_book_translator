"""Deterministic post-render artifact gate: searchability and blank pages.

The old gate was a 3,000-line pixel/geometry/VLM stack built for the Typst
facsimile renderer. That renderer is retired. The native codec now reports
exact geometry through its delivery contract, and artifact parity (T0.5)
compares the delivered text layer against the source. What remains specific to
"a PDF was produced" is cheap and deterministic:

* the file opens and has pages;
* every page that carries content has extractable text (``/ToUnicode`` works);
* no page is blank when the source page was not;
* a faithful delivery still has the source's page count.

No rasterisation, no VLM, no Typst source parsing, no gate thresholds. A
finding is still warn-only unless it says the artifact is unreadable or has
nothing to read, in which case export refuses it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ubt.adapters.pdf.pdfium_gate import open_document, pdfium_serialized
from ubt.core.policy.layout_policy import QA_FULL_GATE_MAX_PAGES

logger = logging.getLogger(__name__)

Severity = Literal["info", "major", "critical"]

#: Codes that refuse export regardless of the opt-in blocking gate: the
#: artifact is not readable at all, or there is no text for a reader to consume.
_FAIL_CLOSED = frozenset(
    {
        "blank_page",
        "font_missing_tounicode",
        "replacement_character",
        "target_language_absent",
        "target_language_sparse",
        "text_not_searchable",
        "unreadable_pdf",
        "artifact_gate_crashed",
    }
)

#: A page with fewer extractable characters than this is a blank candidate.
#: The check is about *no* text, so one caption or page number is enough to
#: clear it.
BLANK_TEXT_THRESHOLD_CHARS = 20


@dataclass(frozen=True)
class ArtifactFinding:
    """One artifact observation, scoped to a page or a block when known."""

    severity: Severity
    code: str
    message: str
    page: int | None = None
    block_id: str | None = None


@dataclass(frozen=True)
class ArtifactGateResult:
    """Outcome of one artifact-gate run over a rendered PDF."""

    passed: bool
    findings: tuple[ArtifactFinding, ...] = ()
    sampled_pages: tuple[int, ...] = ()
    skipped_reason: str | None = None
    stats: dict[str, str | int | float] = field(default_factory=dict)

    def report_payload(self) -> dict[str, object]:
        """Gate-owned half of the ``artifact_report.json`` schema."""
        return {
            "passed": self.passed,
            "skipped_reason": self.skipped_reason,
            "stats": dict(self.stats),
            "sampled_pages": list(self.sampled_pages),
            "findings": [
                {
                    "severity": getattr(finding, "severity", ""),
                    "code": getattr(finding, "code", ""),
                    "message": getattr(finding, "message", ""),
                    "page": getattr(finding, "page", None),
                    "block_id": getattr(finding, "block_id", None),
                }
                for finding in self.findings
            ],
        }


def _page_texts(pdf_path: Path) -> list[str]:
    """Each page's extractable text, in page order."""

    @pdfium_serialized
    def _read(path: Path) -> list[str]:
        chunks: list[str] = []
        with open_document(path) as pdf:
            for index in range(len(pdf)):
                page = pdf[index]
                try:
                    textpage = page.get_textpage()
                    try:
                        chunks.append(textpage.get_text_range())
                    finally:
                        textpage.close()
                finally:
                    page.close()
        return chunks

    return _read(pdf_path)


def _count_pages(pdf_path: Path) -> int:
    """Page count, or ``-1`` when the artifact cannot be opened."""

    @pdfium_serialized
    def _count(path: Path) -> int:
        with open_document(path) as pdf:
            return len(pdf)

    try:
        return _count(pdf_path)
    except Exception as exc:  # noqa: BLE001 - an unreadable artifact is a finding
        logger.debug("Artifact gate could not open %s: %s", pdf_path, exc)
        return -1


_NON_TEXT_PAINT_OPS: frozenset[str] = frozenset(
    {"Do", "S", "s", "f", "F", "f*", "B", "B*", "b", "b*"}
)


def _inspect_pdf_resources(pdf_path: Path) -> tuple[int, set[int], set[int]]:
    """Inspect font ``/ToUnicode`` coverage and non-text graphical pages in one pass.

    Returns ``(missing_tounicode_fonts, non_text_pages, unmapped_font_pages)``.
    """
    try:
        import pikepdf
    except Exception:  # noqa: BLE001 - pikepdf is a base dependency, but never kill a gate
        return 0, set(), set()
    missing = 0
    non_text_pages: set[int] = set()
    unmapped_font_pages: set[int] = set()
    try:
        with pikepdf.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                resources = page.get("/Resources")
                has_xobjects = False
                if resources is not None:
                    xobjs = resources.get("/XObject")
                    if isinstance(xobjs, pikepdf.Dictionary) and len(xobjs) > 0:
                        has_xobjects = True
                    fonts = resources.get("/Font")
                    if fonts is not None:
                        page_fonts = 0
                        page_missing = 0
                        for _, font in fonts.items():
                            if not isinstance(font, pikepdf.Dictionary):
                                continue
                            page_fonts += 1
                            if "/ToUnicode" not in font:
                                missing += 1
                                page_missing += 1
                        if page_fonts > 0 and page_missing == page_fonts:
                            unmapped_font_pages.add(page_no)
                if has_xobjects:
                    non_text_pages.add(page_no)
                    continue
                try:
                    for inst in pikepdf.parse_content_stream(page):
                        if str(inst.operator) in _NON_TEXT_PAINT_OPS:
                            non_text_pages.add(page_no)
                            break
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("Artifact gate could not inspect fonts in %s: %s", pdf_path, exc)
        return 0, set(), set()
    return missing, non_text_pages, unmapped_font_pages


def run_artifact_gate(
    pdf_path: Path,
    *,
    expected_page_count: int | None = None,
    faithful_pages: bool = False,
    target_lang: str | None = None,
) -> ArtifactGateResult:
    """Run the deterministic artifact checks on ``pdf_path``."""
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return ArtifactGateResult(passed=True, skipped_reason="not a rendered pdf")

    total = _count_pages(pdf_path)
    if total <= 0:
        return ArtifactGateResult(
            passed=False,
            findings=(
                ArtifactFinding(
                    severity="critical",
                    code="unreadable_pdf",
                    message=f"Cannot read rendered PDF: {pdf_path.name}",
                ),
            ),
            stats={"total_pages": -1},
        )

    findings: list[ArtifactFinding] = []
    if faithful_pages and expected_page_count is not None and total != expected_page_count:
        findings.append(
            ArtifactFinding(
                severity="major",
                code="page_count_changed_faithful",
                message=(
                    f"Faithful delivery has {total} page(s) but the source has "
                    f"{expected_page_count}"
                ),
            )
        )

    try:
        page_texts = _page_texts(pdf_path)
    except Exception as exc:  # noqa: BLE001
        return ArtifactGateResult(
            passed=False,
            findings=(
                ArtifactFinding(
                    severity="critical",
                    code="text_not_searchable",
                    message=f"PDF text layer could not be extracted: {type(exc).__name__}: {exc}",
                ),
            ),
            stats={"total_pages": total},
        )

    missing_tounicode, non_text_pages, unmapped_font_pages = _inspect_pdf_resources(pdf_path)

    blank_pages = [
        page_no
        for page_no, text in enumerate(page_texts, start=1)
        if len("".join(text.split())) < BLANK_TEXT_THRESHOLD_CHARS and page_no not in non_text_pages
    ]
    for page_no in blank_pages:
        findings.append(
            ArtifactFinding(
                severity="critical",
                code="blank_page",
                message=f"Page {page_no} has no extractable text",
                page=page_no,
            )
        )

    joined = "".join(page_texts)
    if not joined.strip() and len(non_text_pages) < total:
        findings.append(
            ArtifactFinding(
                severity="critical",
                code="text_not_searchable",
                message="The artifact carries no extractable text on any page",
            )
        )
    elif joined.strip():
        replacement = joined.count("\ufffd")
        if replacement:
            findings.append(
                ArtifactFinding(
                    severity="major",
                    code="replacement_character",
                    message=f"The text layer contains {replacement} U+FFFD replacement character(s)",
                )
            )
        if target_lang:
            from ubt.adapters.pdf.artifact_parity import _script_ranges, target_script_counts

            if _script_ranges(target_lang):
                hits, _total_chars = target_script_counts(joined, target_lang)
                if hits == 0:
                    findings.append(
                        ArtifactFinding(
                            severity="major",
                            code="target_language_absent",
                            message=f"The artifact's text layer carries no {target_lang} script characters",
                        )
                    )

    if missing_tounicode:
        unsearchable_font_pages = [
            page_no
            for page_no in sorted(unmapped_font_pages)
            if 1 <= page_no <= len(page_texts)
            and len("".join(page_texts[page_no - 1].split())) < BLANK_TEXT_THRESHOLD_CHARS
        ]
        if not joined.strip():
            findings.append(
                ArtifactFinding(
                    severity="major",
                    code="font_missing_tounicode",
                    message=f"{missing_tounicode} font(s) lack /ToUnicode and no text is extractable",
                )
            )
        elif unsearchable_font_pages:
            for page_no in unsearchable_font_pages:
                findings.append(
                    ArtifactFinding(
                        severity="major",
                        code="font_missing_tounicode",
                        message=f"Page {page_no} uses font(s) without /ToUnicode and yielded no searchable text",
                        page=page_no,
                    )
                )

    passed = not any(finding.severity in ("major", "critical") for finding in findings)
    return ArtifactGateResult(
        passed=passed,
        findings=tuple(findings),
        sampled_pages=tuple(range(1, total + 1)),
        stats={
            "total_pages": total,
            "pages_with_text": sum(1 for text in page_texts if text.strip()),
            "blank_pages": len(blank_pages),
            "fonts_without_tounicode": missing_tounicode,
        },
    )


def blocking_gate_tripped(
    findings: Sequence[ArtifactFinding],
    total_pages: int,
    enabled: bool,
) -> list[ArtifactFinding]:
    """Critical findings that refuse export, plus the always-fail-closed ones."""
    if total_pages < 0:
        return [finding for finding in findings if finding.severity == "critical"]
    always_fail = [
        finding
        for finding in findings
        if finding.code in _FAIL_CLOSED and finding.severity in ("major", "critical")
    ]
    if always_fail:
        return always_fail
    if not enabled or total_pages == 0 or total_pages > QA_FULL_GATE_MAX_PAGES:
        return []
    return [finding for finding in findings if finding.severity == "critical"]


__all__ = [
    "BLANK_TEXT_THRESHOLD_CHARS",
    "ArtifactFinding",
    "ArtifactGateResult",
    "blocking_gate_tripped",
    "run_artifact_gate",
]
