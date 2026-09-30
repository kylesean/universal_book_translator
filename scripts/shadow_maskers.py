#!/usr/bin/env python
"""Phase-2 acceptance: the unified restore matches the four legacy maskers.

The math/code/citation maskers each carried their own copy of the restore +
integrity logic; they now delegate to one ``restore_masked``. This harness loads
the *pre-change* classes straight from git (``HEAD``), masks real corpus text
with the current maskers, and compares the legacy ``unmask_checked`` against the
unified one on a battery of deliberate mutations (faithful, checksum-less,
swapped, dropped, renumbered, duplicated, residual).

Equality must be exact across every :class:`UnmaskReport` field. A mismatch is
the one failure mode that would turn "collapse the copies" into "change the
verdict". Exit 0 iff every compared case is identical.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.cleaners.soup_math import SoupMathMasker

_TOKEN_RE = re.compile(r"⟦[^⟧]*⟧")


@dataclass
class Counts:
    compared: int = 0
    agreed: int = 0
    skipped: int = 0
    mismatches: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mismatches is None:
            self.mismatches = []

    @property
    def passed(self) -> bool:
        return not self.mismatches


def _load_legacy(name: str, ref: str, tmp: Path) -> Any:
    source = subprocess.run(
        ["git", "show", f"{ref}:ubt/core/cleaners/{name}.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    path = tmp / f"legacy_{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"legacy_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _variants(masked: str, mapping: dict[str, str]) -> dict[str, str]:
    tokens = [t for t in _TOKEN_RE.findall(masked) if t in mapping]
    out: dict[str, str] = {"faithful": masked}
    out["checksumless"] = re.sub(r"-[0-9a-z]{3}(⟧)", r"\1", masked)
    if tokens:
        out["dropped"] = masked.replace(tokens[0], "", 1)
        out["duplicated"] = masked + " " + mapping[tokens[0]]
        out["renumbered"] = masked.replace(
            tokens[0], re.sub(r"(\d{4})", lambda m: f"{int(m.group(1)) + 1:04d}", tokens[0]), 1
        )
        out["residual"] = masked + " ⟦MATH_MASK_9999-zzz⟧"
    if len(tokens) >= 2:
        swapped = masked.replace(tokens[0], "\0", 1).replace(tokens[1], tokens[0], 1)
        out["swapped"] = swapped.replace("\0", tokens[1], 1)
    return out


def _fields(report: Any) -> dict[str, Any]:
    return {
        "text": report.text,
        "missing": list(report.missing),
        "mismatched": list(report.mismatched),
        "mutated": list(report.mutated),
        "unverified": list(report.unverified),
        "reordered": list(report.reordered),
        "duplicated": list(report.duplicated),
    }


def _has_substring_original(mapping: dict[str, str]) -> bool:
    """True when one protected original contains another (a legacy-bug input)."""
    originals = sorted(
        {original for original in mapping.values() if original}, key=len, reverse=True
    )
    return any(
        shorter in longer for index, shorter in enumerate(originals) for longer in originals[:index]
    )


def _compare_pair(label: str, legacy: Any, modern: Any, text: str, counts: Counts) -> None:
    if not text.strip():
        counts.skipped += 1
        return
    masked, mapping = modern.mask(text)
    if not mapping:
        counts.skipped += 1
        return
    if _has_substring_original(mapping):
        # The legacy duplicate/missing check counts substring occurrences, so an
        # original inside a longer sibling (Γc inside Γcoeffect) false-positives
        # as a leak. That is a known legacy bug the unified restore fixes, so
        # such inputs are excluded from the equivalence check rather than
        # asserted to reproduce the bug.
        counts.skipped += 1
        return
    for variant_name, echo in _variants(masked, mapping).items():
        old = _fields(legacy.unmask_checked(echo, mapping))
        new = _fields(modern.unmask_checked(echo, mapping))
        counts.compared += 1
        if old == new:
            counts.agreed += 1
        elif len(counts.mismatches) < 30:
            diff = {k: (old[k], new[k]) for k in old if old[k] != new[k]}
            counts.mismatches.append(f"{label}/{variant_name}: {diff}")


def _parse_texts(document: Path, engine: str) -> list[str]:
    from ubt.adapters.factory import get_adapter_for_path

    adapter = get_adapter_for_path(document, pdf_engine=engine)

    async def collect() -> list[str]:
        texts: list[str] = []
        async for chapter in adapter.parse_stream(document):
            texts.extend(block.source_text for block in chapter.blocks if block.source_text)
        return texts

    return asyncio.run(collect())


def _load_cases(corpus_dir: Path) -> list[tuple[str, Path]]:
    from ubt.core.content.verify import load_corpus

    cases: list[tuple[str, Path]] = []
    for case in load_corpus(corpus_dir):
        if not case.document:
            continue
        document = Path(case.document)
        if not document.is_absolute():
            document = corpus_dir / document
        cases.append((case.id, document))
    return cases


def _extra_texts(globs: list[str], root: Path) -> list[str]:
    """Real local text (repo docs/sources) to widen coverage — e.g. fenced code."""
    texts: list[str] = []
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                try:
                    texts.append(path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--ref", default="HEAD", help="git ref holding the legacy maskers")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra repo glob(s) of real text to widen coverage (e.g. 'docs/**/*.md')",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="ubt-legacy-") as tmp_str:
        tmp = Path(tmp_str)
        legacy_math = _load_legacy("math_masker", args.ref, tmp)
        legacy_code = _load_legacy("code_masker", args.ref, tmp)
        legacy_citation = _load_legacy("citation_masker", args.ref, tmp)

        # math, code, citation: legacy class vs the (now delegating) new class.
        # soup: legacy == MathMasker(soup_prefix) (soup inherited math's restore),
        # so compare the legacy math class configured with the soup prefix.
        pairs: list[tuple[str, Any, Any]] = [
            ("math", legacy_math.MathMasker(), MathMasker()),
            ("code", legacy_code.CodeMasker(), CodeMasker()),
            ("citation", legacy_citation.CitationMasker(), CitationMasker()),
            ("soup", legacy_math.MathMasker("⟦SOUP_MASK_"), SoupMathMasker()),
        ]

        counts: dict[str, Counts] = {name: Counts() for name, _, _ in pairs}
        corpus_dir = Path(args.corpus)
        per_case_blocks = 0
        for case_id, document in _load_cases(corpus_dir):
            if not document.exists():
                continue
            try:
                texts = _parse_texts(document, args.engine)
            except Exception as exc:
                print(f"  {case_id}: parse error {exc}", file=sys.stderr)
                continue
            per_case_blocks += len(texts)
            for name, legacy, modern in pairs:
                for text in texts:
                    _compare_pair(f"{case_id}:{name}", legacy, modern, text, counts[name])

        extras = _extra_texts(args.extra, Path.cwd())
        for name, legacy, modern in pairs:
            for text in extras:
                _compare_pair(f"extra:{name}", legacy, modern, text, counts[name])

    total = sum(c.compared for c in counts.values())
    agreed = sum(c.agreed for c in counts.values())
    skipped = sum(c.skipped for c in counts.values())
    all_mismatches = [m for c in counts.values() for m in c.mismatches]
    passed = total > 0 and agreed == total and not all_mismatches

    print(f"\nPhase-2 masker differential — {corpus_dir} ({per_case_blocks} text blocks)")
    for name, c in counts.items():
        status = "pass" if c.passed else "FAIL"
        print(
            f"  {name:<10} {status:<5} compared={c.compared} agreed={c.agreed} skipped={c.skipped}"
        )
    print(f"\n  TOTAL: {agreed}/{total} identical ({skipped} skipped)")
    for line in all_mismatches[:20]:
        print(f"    MISMATCH {line}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
