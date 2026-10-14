"""Artifact parity: compare the delivered PDF against the source as physical evidence.

The T0/T1 checks in ``visual_gate`` prove the pipeline agrees with itself —
page counts, compile status, IR block geometry. That is self-attestation. The
failure that motivated this gate is
exactly the kind self-attestation misses: an rigid render whose Chinese layer
sits *under* the white boxes passes page size, page count and compile status
while most of the book is untranslated.

So every check here reads only the two files — the source and the artifact —
with poppler subprocesses, and never imports pipeline modules. Findings use the
visual gate's ``(severity, code, message)`` shape so the visual-gate loop can merge
them; ``target_language_absent`` is the one fail-closed code: an artifact
containing none of the language it was commissioned in must not ship
regardless of gate settings.

Geometry and image-count parity are deliberately not checked: they only made
sense for a render that *promised* to keep the source page and then altered it,
and the renderer rebuilds every page by design, so both checks fired on
every correct run. What is checked is script coverage, which is
engine-independent.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ubt.core.env import subprocess_env

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_S = 30.0

# CJK/latin target-script coverage below which the artifact is suspect even
# though it contains *some* target glyphs. Comfortably below what any
# correctly translated monolingual or bilingual book reaches: even the
# alternating zipper keeps the target share above half of a text page.
_SPARSE_TARGET_RATIO = 0.08


@dataclass(frozen=True, slots=True)
class ParityFinding:
    severity: str  # "info" | "major" | "critical"
    code: str
    message: str


def _script_ranges(target_lang: str) -> tuple[range, ...]:
    """Codepoint ranges that count as the target script, by language family."""
    lang = (target_lang or "").strip().lower().replace("_", "-")
    base = lang.split("-")[0]
    han = range(0x4E00, 0x9FFF)
    if base == "zh":
        return (han, range(0x3400, 0x4DBF))
    if base == "ja":
        return (han, range(0x3040, 0x30FF))
    if base == "ko":
        return (range(0xAC00, 0xD7A3), range(0x1100, 0x11FF))
    if base == "ar":
        return (
            range(0x0600, 0x06FF),
            range(0x0750, 0x077F),
            range(0x08A0, 0x08FF),
            range(0xFB50, 0xFDFF),
            range(0xFE70, 0xFEFF),
        )
    if base == "he":
        return (range(0x0590, 0x05FF), range(0xFB1D, 0xFB4F))
    return ()


def target_script_counts(text: str, target_lang: str) -> tuple[int, int]:
    """``(target_script_chars, total_non_whitespace_chars)`` for ``text``."""
    ranges = _script_ranges(target_lang)
    total = sum(1 for ch in text if not ch.isspace())
    if not ranges:
        return 0, total
    # ord() — `str in range` is value-comparison against ints and never matches.
    hits = sum(1 for ch in text if any(ord(ch) in r for r in ranges))
    return hits, total


def target_script_ratio(text: str, target_lang: str) -> float:
    hits, total = target_script_counts(text, target_lang)
    return (hits / total) if total else 0.0


def _run(cmd: list[str]) -> str | None:
    """Total subprocess text: stdout, or None when the binary is missing/fails."""
    binary = shutil.which(cmd[0])
    if not binary:
        return None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
            env=subprocess_env(),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _page_text(pdf_path: Path, pages: Sequence[int] | None = None) -> str | None:
    if not pages:
        return _run(["pdftotext", str(pdf_path), "-"])
    sorted_pages = sorted(set(pages))
    first, last = sorted_pages[0], sorted_pages[-1]
    if sorted_pages == list(range(first, last + 1)):
        return _run(["pdftotext", "-f", str(first), "-l", str(last), str(pdf_path), "-"])
    chunks: list[str] = []
    for p in sorted_pages:
        txt = _run(["pdftotext", "-f", str(p), "-l", str(p), str(pdf_path), "-"])
        if txt is not None:
            chunks.append(txt)
    return "\n".join(chunks) if chunks else None


def check_artifact_parity(
    *,
    artifact_pdf: Path,
    target_lang: str,
    selected_pages: Sequence[int] | None = None,
) -> list[ParityFinding]:
    """Physical-evidence diff between the delivered PDF and its source.

    Degrades to ``[]`` (with an info note) when poppler cannot answer — a
    missing probe is not evidence of a defect.
    """
    findings: list[ParityFinding] = []
    artifact_text = _page_text(artifact_pdf, pages=selected_pages)
    if artifact_text is None:
        return [
            ParityFinding(
                "info",
                "parity_probe_unavailable",
                "pdftotext missing or failed; artifact parity not measured",
            )
        ]

    hits, total = target_script_counts(artifact_text, target_lang)
    if not _script_ranges(target_lang):
        # Latin-script targets have no measurable range here. Saying so is the
        # honest form of the gate: silently skipping would let an artifact that
        # carries none of its commissioned language pass unexamined.
        findings.append(
            ParityFinding(
                "info",
                "target_script_unmeasurable",
                f"Target-language presence could not be measured: no script "
                f"ranges are defined for {target_lang!r}",
            )
        )
    elif total > 50:
        ratio = hits / total
        if hits == 0:
            findings.append(
                ParityFinding(
                    "critical",
                    "target_language_absent",
                    f"Delivered PDF text layer contains none of the target script "
                    f"({target_lang!r}) in {total} characters — untranslated or "
                    "covered content is shipping",
                )
            )
        elif ratio < _SPARSE_TARGET_RATIO:
            findings.append(
                ParityFinding(
                    "major",
                    "target_language_sparse",
                    f"Only {ratio:.1%} of the delivered PDF's characters are the "
                    f"target script ({target_lang!r}); expect a far higher share",
                )
            )

    if "\ufffd" in artifact_text:
        findings.append(
            ParityFinding(
                "major",
                "replacement_chars_in_text_layer",
                f"{artifact_text.count(chr(0xFFFD))} U+FFFD replacement characters "
                "in the delivered text layer (broken font mapping)",
            )
        )
    if "(cid:" in artifact_text:
        findings.append(
            ParityFinding(
                "info",
                "cid_in_text_layer",
                "Delivered text layer contains unmapped (cid:) glyphs; copy/paste "
                "and search degrade",
            )
        )

    return findings


__all__ = [
    "ParityFinding",
    "check_artifact_parity",
    "target_script_counts",
    "target_script_ratio",
]
