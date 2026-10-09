#!/usr/bin/env python3
"""Read-only check of `path:line` references in the (unversioned) docs/ tree.

``docs/`` is deliberately gitignored and changes too often to track, so a
reference into the code drifts silently: the file moves, a symbol is renamed,
or a line number slides. This script reports the two failure modes worth acting
on, and stays quiet about the rest:

1. **A referenced file that does not exist.** References to the comparison repo
   (``TranslateBooksWithLLMs``) and to explicitly proposed files are skipped --
   the fiction/geometry docs are design drafts and legitimately name files that
   do not exist yet. Anything else is reported.
2. **A line number past the end of its file.** The reference cannot be right.

Off-by-a-few line numbers are *not* reported: the docs are prose with citations,
not a compiler, and demanding byte-exact line anchors would make the check noise.
Run it by hand after a refactor that moves code the docs cite.

    python scripts/check_doc_refs.py [--docs docs]

Exit code is 1 when anything is reported, so it can gate a docs refresh.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: A reference like ``ubt/core/config.py:321`` or ``engine_selector.py:130-252``.
_REF_RE = re.compile(r"`?([A-Za-z0-9_][A-Za-z0-9_./-]*\.py):(\d+)(?:-(\d+))?`?")

#: Directories a bare filename in the docs is resolved against (the docs cite
#: ``engine_selector.py:130`` without the full path).
_SEARCH_BASES = (
    "",
    "ubt",
    "ubt/adapters",
    "ubt/adapters/pdf",
    "ubt/adapters/docx",
    "ubt/adapters/epub",
    "ubt/adapters/html",
    "ubt/adapters/markdown",
    "ubt/core",
    "ubt/core/cleaners",
    "ubt/core/engine",
    "ubt/core/engine/stages",
    "ubt/core/ir",
    "ubt/core/memory",
    "ubt/core/content",
    "ubt/core/qe",
    "ubt/core/router",
    "ubt/core/validators",
    "ubt/core/metrics",
    "ubt/cli/commands",
    "ubt/model",
    "ubt/render",
    "ubt/pipeline",
    "ubt/analyze",
    "scripts",
)

#: Files the docs name on purpose although they do not exist in this repo:
#: design drafts ("新增 X") and paths in the comparison repo. A reference is
#: only skipped when its line also carries one of these markers.
_DRAFT_MARKERS = ("新增", "建议", "草案", "待评审", "提案")
_FOREIGN_REPO_MARKERS = ("TBL", "TranslateBooksWithLLMs", "VI-Translate", "pdf2zh")

#: Path prefixes that belong to a *comparison* repo, not UBT. A missing file
#: under one of these is expected, not drift.
_FOREIGN_PREFIXES = ("src/", "lib/", "app/", "benchmark/", "pdf2zh/", "uv/")


#: Bare filenames the docs cite from the *comparison* repo's tree (the fiction
#: draft is a two-repo review). Their absence in UBT is expected, and the
#: citations carry no ``TBL`` marker on the same line to key off.
_KNOWN_FOREIGN_BARE_NAMES = frozenset(
    {
        "assembler.py",
        "injector.py",
        "scripts/rejudge_all_via_poe.py",
    }
)


def _resolve(ref_file: str) -> Path | None:
    if Path(ref_file).is_file():
        return Path(ref_file)
    for base in _SEARCH_BASES:
        if not base:
            continue
        candidate = Path(base) / ref_file
        if candidate.is_file():
            return candidate
    return None


def _is_in_scope(ref_file: str) -> bool:
    """Whether a reference is expected to name a file in *this* repo.

    A bare filename (``engine_selector.py``) or a UBT-owned prefix
    (``ubt/...``, ``tests/...``, ``scripts/...``) is in scope. A comparison-repo
    path (``src/core/...``, ``pdf2zh/...``) is not -- its absence is expected.
    """
    if ref_file.startswith(_FOREIGN_PREFIXES):
        return False
    if ref_file in _KNOWN_FOREIGN_BARE_NAMES:
        return False
    return "/" not in ref_file or ref_file.startswith(("ubt/", "tests/", "scripts/"))


def check(docs_dir: Path) -> list[str]:
    problems: list[str] = []
    if not docs_dir.is_dir():
        return [f"docs directory not found: {docs_dir}"]

    for doc in sorted(docs_dir.rglob("*.md")):
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if any(marker in line for marker in _DRAFT_MARKERS):
                continue
            if any(marker in line for marker in _FOREIGN_REPO_MARKERS):
                continue
            for match in _REF_RE.finditer(line):
                ref_file, start, end = match.group(1), int(match.group(2)), match.group(3)
                if not _is_in_scope(ref_file):
                    continue
                resolved = _resolve(ref_file)
                if resolved is None:
                    problems.append(f"{doc}:{lineno}: referenced file not found: {ref_file}")
                    continue
                total = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
                highest = int(end) if end else start
                if highest > total:
                    problems.append(
                        f"{doc}:{lineno}: {ref_file}:{highest} is past end of file ({total} lines)"
                    )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()

    problems = check(args.docs)
    if problems:
        print(f"{len(problems)} doc reference problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"all doc references in {args.docs} resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
