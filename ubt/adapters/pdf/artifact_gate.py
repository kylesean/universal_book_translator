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
        "visual_gate_crashed",
    }
)

#: A page with fewer extractable characters than this is a blank candidate.
#: The check is about *no* text, so one caption or page number is enough to
#: clear it.
BLANK_TEXT_THRESHOLD_CHARS = 20


@dataclass(frozen=True)
class VisualFinding:
    """One artifact observation, scoped to a page or a block when known."""

    severity: Severity
    code: str
    message: str
    page: int | None = None
    block_id: str | None = None


@dataclass(frozen=True)
class VisualGateResult:
    """Outcome of one artifact-gate run over a rendered PDF."""

    passed: bool
    findings: tuple[VisualFinding, ...] = ()
    sampled_pages: tuple[int, ...] = ()
    vlm_pages: tuple[int, ...] = ()
    skipped_reason: str | None = None
    stats: dict[str, str | int | float] = field(default_factory=dict)
    tiers_ran: tuple[str, ...] = ()
    tiers_skipped: tuple[str, ...] = ()
    skip_codes: tuple[str, ...] = ()

    def report_payload(self) -> dict[str, object]:
        """Gate-owned half of the ``visual_report.json`` schema."""
        return {
            "passed": self.passed,
            "skipped_reason": self.skipped_reason,
            "tiers_ran": list(self.tiers_ran),
            "tiers_skipped": list(self.tiers_skipped),
            "skip_codes": list(self.skip_codes),
            "stats": dict(self.stats),
            "sampled_pages": list(self.sampled_pages),
            "vlm_pages": list(self.vlm_pages),
            "findings": [
                {
                    "severity": getattr(finding, "severity", ""),
                    "code": getattr(finding, "code", ""),
                    "message": getattr(finding, "message", ""),
                    "page": getattr(finding, "page", None),
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


def _fonts_without_tounicode(pdf_path: Path) -> int:
    """Count font dictionaries the artifact uses without a ``/ToUnicode`` map."""
    try:
        import pikepdf
    except Exception:  # noqa: BLE001 - pikepdf is a base dependency, but never kill a gate
        return 0
    missing = 0
    try:
        with pikepdf.open(pdf_path) as pdf:
            for page in pdf.pages:
                resources = page.get("/Resources")
                if resources is None:
                    continue
                fonts = resources.get("/Font")
                if fonts is None:
                    continue
                for _, font in fonts.items():
                    if not isinstance(font, pikepdf.Dictionary):
                        continue
                    if "/ToUnicode" not in font:
                        missing += 1
    except Exception as exc:  # noqa: BLE001
        logger.debug("Artifact gate could not inspect fonts in %s: %s", pdf_path, exc)
        return 0
    return missing


def run_visual_gate(
    pdf_path: Path,
    *,
    expected_page_count: int | None = None,
    faithful_pages: bool = False,
    target_lang: str | None = None,
) -> VisualGateResult:
    """Run the deterministic artifact checks on ``pdf_path``."""
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return VisualGateResult(passed=True, skipped_reason="not a rendered pdf")

    total = _count_pages(pdf_path)
    if total <= 0:
        return VisualGateResult(
            passed=False,
            findings=(
                VisualFinding(
                    severity="critical",
                    code="unreadable_pdf",
                    message=f"Cannot read rendered PDF: {pdf_path.name}",
                ),
            ),
            stats={"total_pages": -1},
            tiers_ran=("open",),
            tiers_skipped=("searchability", "blank_pages"),
        )

    findings: list[VisualFinding] = []
    if faithful_pages and expected_page_count is not None and total != expected_page_count:
        findings.append(
            VisualFinding(
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
        return VisualGateResult(
            passed=False,
            findings=(
                VisualFinding(
                    severity="critical",
                    code="text_not_searchable",
                    message=f"PDF text layer could not be extracted: {type(exc).__name__}: {exc}",
                ),
            ),
            stats={"total_pages": total},
            tiers_ran=("open",),
            tiers_skipped=("blank_pages",),
        )

    blank_pages = [
        page_no
        for page_no, text in enumerate(page_texts, start=1)
        if len("".join(text.split())) < BLANK_TEXT_THRESHOLD_CHARS
    ]
    for page_no in blank_pages:
        findings.append(
            VisualFinding(
                severity="critical",
                code="blank_page",
                message=f"Page {page_no} has no extractable text",
                page=page_no,
            )
        )

    joined = "".join(page_texts)
    if not joined.strip():
        findings.append(
            VisualFinding(
                severity="critical",
                code="text_not_searchable",
                message="The artifact carries no extractable text on any page",
            )
        )
    else:
        replacement = joined.count("\ufffd")
        if replacement:
            findings.append(
                VisualFinding(
                    severity="major",
                    code="replacement_character",
                    message=f"The text layer contains {replacement} U+FFFD replacement character(s)",
                )
            )
        if target_lang and target_lang.lower().startswith(("zh", "ja", "ko")):
            cjk = sum(1 for char in joined if "\u4e00" <= char <= "\u9fff")
            if cjk == 0:
                findings.append(
                    VisualFinding(
                        severity="major",
                        code="target_language_absent",
                        message="The artifact's text layer carries no CJK characters",
                    )
                )

    missing_tounicode = _fonts_without_tounicode(pdf_path)
    if missing_tounicode and not joined.strip():
        findings.append(
            VisualFinding(
                severity="major",
                code="font_missing_tounicode",
                message=f"{missing_tounicode} font(s) lack /ToUnicode and no text is extractable",
            )
        )

    passed = not any(finding.severity in ("major", "critical") for finding in findings)
    return VisualGateResult(
        passed=passed,
        findings=tuple(findings),
        sampled_pages=tuple(range(1, total + 1)),
        stats={
            "total_pages": total,
            "pages_with_text": sum(1 for text in page_texts if text.strip()),
            "blank_pages": len(blank_pages),
            "fonts_without_tounicode": missing_tounicode,
        },
        tiers_ran=("open", "searchability", "blank_pages"),
        tiers_skipped=("geometry", "pixel", "vlm"),
        skip_codes=("tiers_retired_with_typst",),
    )


def blocking_gate_tripped(
    findings: Sequence[VisualFinding],
    total_pages: int,
    enabled: bool,
) -> list[VisualFinding]:
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
    "VisualFinding",
    "VisualGateResult",
    "blocking_gate_tripped",
    "run_visual_gate",
]
