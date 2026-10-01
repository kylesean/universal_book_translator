#!/usr/bin/env python
"""Phase-2 acceptance: the refactor does not degrade the translation (chrF).

The mask/restore collapse (four maskers -> ``PlaceholderEngine`` + the
``TranslationEngine``) must be output-preserving. This harness drives the *whole*
new translate path (mask -> translate -> resolve) and the pre-refactor path
(the four legacy masker classes loaded from git ``HEAD``, composed in the old
order) over real corpus text plus repo text, using a deterministic stand-in for
a translator, and scores the two resolved targets with chrF.

- the new target must match the legacy target (chrF >= ``--min-chrf``,
  default 0.999): the refactor changed *where* the order lives, not what it
  produces;
- every protected span must survive into the new target (the Axiom-B check);
- the mean/max chrF against the source is reported as the translation distance
  (informational — a real translator would move this, the stand-in barely does).

Exit 0 iff every compared sample clears the floor and no placeholder is lost.
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

from ubt.core.qe.chrf import chrf
from ubt.core.qe.fast_pass import REHEARSAL_MARKER
from ubt.segment.placeholders import default_placeholder_engine
from ubt.translate.engine import TranslationEngine

_TOKEN_RE = re.compile(r"⟦[^⟧]*⟧")


@dataclass
class _Counts:
    compared: int = 0
    passed: int = 0
    missing_spans: int = 0
    chrf_sum: float = 0.0
    source_chrf_sum: float = 0.0
    worst: tuple[float, str] | None = None
    failures: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []

    @property
    def mean_chrf(self) -> float:
        return self.chrf_sum / self.compared if self.compared else 0.0

    @property
    def mean_source_chrf(self) -> float:
        return self.source_chrf_sum / self.compared if self.compared else 0.0


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


class _LegacyPath:
    """The pre-refactor pipeline: four maskers composed in the old order."""

    def __init__(self, legacy_math: Any, legacy_code: Any, legacy_citation: Any) -> None:
        self._code = legacy_code.CodeMasker()
        self._math = legacy_math.MathMasker()
        self._soup = legacy_math.MathMasker("⟦SOUP_MASK_")
        self._citation = legacy_citation.CitationMasker()

    def resolve(self, text: str, masks: tuple[Any, ...]) -> str:
        code_map, math_map, soup_map, cite_map = masks
        citation = self._citation.unmask_checked(text, cite_map)
        soup = self._soup.unmask_checked(citation.text, soup_map)
        math = self._math.unmask_checked(soup.text, math_map)
        code = self._code.unmask_checked(math.text, code_map)
        return code.text

    def mask(self, text: str) -> tuple[str, tuple[Any, ...]]:
        masked, code_map = self._code.mask(text)
        masked, math_map = self._math.mask(masked)
        masked, soup_map = self._soup.mask(masked)
        masked, cite_map = self._citation.mask(masked)
        return masked, (code_map, math_map, soup_map, cite_map)


def _compare(
    label: str, legacy: _LegacyPath, engine: TranslationEngine, text: str, c: _Counts
) -> None:
    if not text.strip():
        return
    legacy_masked, legacy_masks = legacy.mask(text)
    old_target = legacy.resolve(f"{REHEARSAL_MARKER}{legacy_masked}", legacy_masks)

    masked = engine.mask(text)
    new_target = engine.resolve(f"{REHEARSAL_MARKER}{masked.text}", masked).text

    c.compared += 1
    score = chrf(new_target, old_target)
    c.chrf_sum += score
    c.source_chrf_sum += chrf(new_target, text)
    if c.worst is None or score < c.worst[0]:
        c.worst = (score, label)

    # A source that literally contains the placeholder grammar (the pipeline's
    # own .py files passed via --extra) nests one token inside another's
    # original; the reverse pass expands the inner token, so the outer original
    # is not reproduced verbatim. That is the same self-referential artifact
    # ``shadow_maskers`` documents and excludes, not a lost span.
    missing = [
        ph.token
        for ph in masked.placeholders
        if ph.original and not _TOKEN_RE.search(ph.original) and ph.original not in new_target
    ]
    if missing:
        c.missing_spans += len(missing)
        if len(c.failures) < 20:
            c.failures.append(f"{label}: lost {missing[:3]}")


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


def _extra_texts(globs: list[str], root: Path) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                try:
                    texts.append((str(path), path.read_text(encoding="utf-8", errors="replace")))
                except OSError:
                    continue
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--ref", default="HEAD", help="git ref holding the legacy maskers")
    parser.add_argument("--min-chrf", type=float, default=0.999)
    parser.add_argument(
        "--extra", action="append", default=[], help="Extra repo glob(s) of real text"
    )
    args = parser.parse_args()

    counts = _Counts()
    corpus_dir = Path(args.corpus)
    with tempfile.TemporaryDirectory(prefix="ubt-legacy-") as tmp_str:
        tmp = Path(tmp_str)
        legacy = _LegacyPath(
            _load_legacy("math_masker", args.ref, tmp),
            _load_legacy("code_masker", args.ref, tmp),
            _load_legacy("citation_masker", args.ref, tmp),
        )
        engine = TranslationEngine(placeholders=default_placeholder_engine())

        for case_id, document in _load_cases(corpus_dir):
            if not document.exists():
                continue
            try:
                texts = _parse_texts(document, args.engine)
            except Exception as exc:
                print(f"  {case_id}: parse error {exc}", file=sys.stderr)
                continue
            for text in texts:
                _compare(f"{case_id}", legacy, engine, text, counts)

        for label, text in _extra_texts(args.extra, Path.cwd()):
            _compare(f"extra:{label}", legacy, engine, text, counts)

    below_floor = counts.mean_chrf < args.min_chrf or (
        counts.worst is not None and counts.worst[0] < args.min_chrf
    )
    passed = counts.compared > 0 and not below_floor and counts.missing_spans == 0

    print(f"\nPhase-2 chrF lower-bound — {corpus_dir}")
    print(
        f"  compared={counts.compared}  mean_chrf={counts.mean_chrf:.6f}  "
        f"min_chrf={counts.worst[0] if counts.worst else 0.0:.6f}"
        + (f" ({counts.worst[1]})" if counts.worst else "")
    )
    print(f"  lost protected spans={counts.missing_spans}  floor={args.min_chrf}")
    print(f"  (informational) mean chrF newest-target vs source = {counts.mean_source_chrf:.6f}")
    print(f"\n  -> {'PASS' if passed else 'FAIL'}")
    for line in counts.failures[:20]:
        print(f"    {line}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
