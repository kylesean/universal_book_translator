"""Pure document-archetype detection, shared by the assess quote and the
pipeline's routing policy.

Lives in ``core`` because neither consumer may reach into the other's layer:
``ubt.core.assess`` consumes it machine-readably and the pipeline's policy
reads it for routing.
Detection is zero-token and, for PDF, only ever touches pdfium through the
serialized gate reached via ``ubt.core.ports`` (this module names no adapter
module and no heavy PDF dependency directly).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class MathDensity(StrEnum):
    """Density classification of mathematical equations and formulas in a document."""

    NONE = "none"
    LOW = "low"
    HIGH = "high"


class DocCategory(StrEnum):
    """High-level document categorization."""

    ACADEMIC_PAPER = "academic_paper"
    TECHNICAL_BOOK = "technical_book"
    LITERATURE = "literature"
    GENERAL = "general"


# Keyword registries for automatic domain detection
DOMAIN_PATTERNS: dict[str, list[str]] = {
    "semiconductor": [
        r"\bfinfet\b",
        r"\bmosfet\b",
        r"\bsubthreshold\b",
        r"\bchannel\b",
        r"\bgate\b",
        r"\bdrain\b",
        r"\bsource\b",
        r"\bpoisson\b",
        r"\bt_?fin\b",
        r"\bt_?ox\b",
        r"\bv_?ch\b",
        r"\bv_?ds\b",
        r"\bv_?gs\b",
        r"\bdoping\b",
        r"\bsemiconductor\b",
        r"\bcapacitance\b",
        r"\bbandgap\b",
        r"\bquasifermi\b",
    ],
    "biomedicine": [
        r"\bprotein\b",
        r"\bgenome\b",
        r"\bantibody\b",
        r"\benzyme\b",
        r"\bcellular\b",
        r"\bpathway\b",
        r"\breceptor\b",
        r"\bassay\b",
        r"\bclinical\b",
        r"\bgene\b",
        r"\brna\b",
        r"\bdna\b",
        r"\bin vitro\b",
        r"\bin vivo\b",
    ],
    "finance": [
        r"\bequity\b",
        r"\bportfolio\b",
        r"\bdividend\b",
        r"\basset\b",
        r"\bvolatility\b",
        r"\barbitrage\b",
        r"\bbalance sheet\b",
        r"\bmacroeconomic\b",
        r"\binflation\b",
        r"\bfiscal\b",
        r"\bliquidity\b",
    ],
    "computer_science": [
        r"\bneural network\b",
        r"\btransformer\b",
        r"\bbackpropagation\b",
        r"\blatency\b",
        r"\bthroughput\b",
        r"\bcompiler\b",
        r"\bcuda\b",
        r"\balgorithm\b",
        r"\bgradient\b",
        r"\bconvolutional\b",
    ],
}

# Patterns indicating mathematical equations, differential equations, or dense symbols
MATH_PATTERNS = [
    r"\\frac\{",
    r"\\partial",
    r"\\int",
    r"\\sum",
    r"\\sqrt",
    r"\b\d+\s*[-–]\s*10\s*\d+\b",  # flattened scientific notation: 1 - 10 15
    r"¼",  # Elsevier CMap font bug for =
    r"\b[A-Za-z]{1,3}\s*=\s*[-+]?\d",  # 'x = 5' math assignment, not 'age = 25' prose
    r"[∂∇∫∑∏√∞±×÷≤≥≠≈≡∈∉⊂⊆∪∩∀∃⊢⊨⊗⊕⊸→←↔↦∘⟨⟩⟦⟧]",
    r"[ΓΔΘΛΞΠΣΦΨΩα-ω]",
    r"\b(?:Theorem|Lemma|Proposition|Corollary|Definition)\s+\d+(?:\.\d+)*",
    r"\b\w+_\{\w+\}",
    r"\b[A-Za-z]_[A-Za-z0-9]\b",
    r"\bcm\s*[-–]?\s*[23]\b",
    r"\b[A-Z]ch\b",
    r"\bTFIN\b",
]


@dataclass(frozen=True)
class Archetype:
    """What a document *is*, decided from a local sample without spending a token."""

    format_ext: str
    page_or_ch_count: int
    is_scanned: bool
    math_density: MathDensity
    detected_domain: str
    domain_confidence: float
    category: DocCategory
    sample_chars: int


def sample_document(path: Path, ext: str) -> tuple[int, bool, str]:
    """Sample the document to extract page/chapter count and text preview."""
    if ext == "pdf":
        # Reached only through the ports bridge, so this module names neither
        # an adapter module nor a heavy PDF dependency directly (pdfium gate +
        # pypdf fallback live in ubt.core.ports.sample_pdf_pages).
        from ubt.core.ports import sample_pdf_pages

        return sample_pdf_pages(path)
    elif ext in ("md", "markdown", "txt"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            ch_count = max(1, len(re.findall(r"^#+\s+", text, re.MULTILINE)))
            return ch_count, False, text[:15000]
        except Exception as exc:
            # The probe never raises (analyze_archetype's contract), but a
            # corrupt file degrading to "empty document" must leave a trace.
            logger.warning("Document probe failed for %s (%s): %s", path, ext, exc)
            return 1, False, ""
    elif ext in ("html", "htm"):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(raw, "html.parser")
                text = soup.get_text()
                headings = len(soup.find_all(re.compile(r"^h[1-6]$", re.I)))
            except Exception:
                text = re.sub(r"<[^>]+>", " ", raw)
                headings = len(re.findall(r"<h[1-6][\s>]", raw, re.IGNORECASE))
            ch_count = max(1, headings)
            return ch_count, False, text[:15000]
        except Exception as exc:
            # The probe never raises (analyze_archetype's contract), but a
            # corrupt file degrading to "empty document" must leave a trace.
            logger.warning("Document probe failed for %s (%s): %s", path, ext, exc)
            return 1, False, ""
    elif ext == "docx":
        try:
            import docx

            doc = docx.Document(str(path))
            text_parts = [p.text for p in doc.paragraphs[:100]]
            sample = "\n".join(text_parts)
            ch_count = max(
                1,
                len([p for p in doc.paragraphs if p.style and "heading" in p.style.name.lower()]),
            )
            return ch_count, False, sample
        except Exception as exc:
            # The probe never raises (analyze_archetype's contract), but a
            # corrupt file degrading to "empty document" must leave a trace.
            logger.warning("Document probe failed for %s (%s): %s", path, ext, exc)
            return 1, False, ""
    elif ext == "epub":
        try:
            import zipfile

            from bs4 import BeautifulSoup

            from ubt.core.zip_safety import ZipReadBudget, read_member

            epub_parts: list[str] = []
            with zipfile.ZipFile(str(path), "r") as z:
                budget = ZipReadBudget()
                html_files = [
                    n for n in z.namelist() if n.lower().endswith((".html", ".xhtml", ".htm"))
                ]
                for name in html_files[:5]:
                    data = read_member(z, name, budget)
                    if data is None:
                        continue
                    soup = BeautifulSoup(data, "html.parser")
                    epub_parts.append(soup.get_text())
            sample = "\n".join(epub_parts)
            return len(html_files), False, sample
        except Exception as exc:
            # The probe never raises (analyze_archetype's contract), but a
            # corrupt file degrading to "empty document" must leave a trace.
            logger.warning("Document probe failed for %s (%s): %s", path, ext, exc)
            return 1, False, ""

    return 1, False, ""


def detect_math_density(text: str) -> MathDensity:
    """Inspect sampled text for equation and mathematical formula density."""
    if not text:
        return MathDensity.NONE

    hits = 0
    for pat in MATH_PATTERNS:
        found = len(re.findall(pat, text, re.IGNORECASE))
        hits += found

    # Density is matches per 1000 sampled characters. An absolute hit count
    # alone is not enough to call a book math-heavy: a long technical
    # document easily accumulates six incidental subscript-shaped tokens, so
    # the hits>=6 shortcut must still clear a minimum density. Otherwise the
    # advisor would pre-select the heavy publication preset for a prose book.
    density = (hits / max(len(text), 1)) * 1000
    if density >= 2.0 or (hits >= 6 and density >= 0.5):
        return MathDensity.HIGH
    elif density >= 0.5 or hits >= 2:
        return MathDensity.LOW
    return MathDensity.NONE


def detect_domain(text: str) -> tuple[str, float]:
    """Detect document subject domain via keyword density scoring."""
    if not text:
        return "general", 0.0

    scores: dict[str, int] = {}
    for domain, patterns in DOMAIN_PATTERNS.items():
        domain_hits = 0
        for pat in patterns:
            domain_hits += len(re.findall(pat, text, re.IGNORECASE))
        scores[domain] = domain_hits

    best_domain = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_hits = scores[best_domain]

    if best_hits >= 4:
        confidence = min(1.0, 0.4 + (best_hits * 0.05))
        return best_domain, round(confidence, 2)
    return "general", 0.1


_ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
_ARXIV_TEXT_RE = re.compile(r"\barxiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)


def classify_category(
    ext: str,
    math_density: MathDensity,
    domain: str,
    pages: int,
    path: Path | None = None,
    text: str = "",
) -> DocCategory:
    """Classify into high-level category for user presentation."""
    # Explicit arXiv naming or citation in text guarantees academic paper
    stem = path.stem.lower() if path is not None else ""
    if ext == "pdf":
        if (
            _ARXIV_RE.search(stem) or "arxiv" in stem or _ARXIV_TEXT_RE.search(text)
        ) and pages <= 120:
            return DocCategory.ACADEMIC_PAPER

        if math_density == MathDensity.HIGH or domain in ("semiconductor", "computer_science"):
            if pages <= 80:
                return DocCategory.ACADEMIC_PAPER
            return DocCategory.TECHNICAL_BOOK

    if (
        ext in ("epub", "md", "markdown", "html", "htm", "txt")
        and domain == "general"
        and math_density == MathDensity.NONE
    ):
        return DocCategory.LITERATURE
    return DocCategory.GENERAL


def infer_profile_from_archetype(arch: Archetype, path: Path | None = None) -> str:
    """Infer the optimal domain profile (e.g. 'paper', 'semiconductor', 'general') from an archetype."""
    if arch.category == DocCategory.ACADEMIC_PAPER:
        return "paper"
    if path is not None and (_ARXIV_RE.search(path.stem) or "arxiv" in path.stem.lower()):
        return "paper"
    if arch.detected_domain in ("semiconductor", "humanities"):
        return arch.detected_domain
    if arch.category == DocCategory.LITERATURE:
        return "fiction"
    return "general"


def analyze_archetype(path: Path) -> Archetype:
    """Sample a cold file and return its archetype facts. Never raises on probe failure."""
    ext = path.suffix.lower().lstrip(".")
    count, is_scanned, sample_text = sample_document(path, ext)
    math_density = detect_math_density(sample_text)
    detected_domain, domain_confidence = detect_domain(sample_text)
    category = classify_category(
        ext, math_density, detected_domain, count, path=path, text=sample_text
    )
    return Archetype(
        format_ext=ext,
        page_or_ch_count=count,
        is_scanned=is_scanned,
        math_density=math_density,
        detected_domain=detected_domain,
        domain_confidence=domain_confidence,
        category=category,
        sample_chars=len(sample_text),
    )
