"""Comments, docstrings, and doc links must point at things that exist.

A docstring saying "see ``ubt/adapters/pdf/visual_gate.py``" or naming a symbol
that has since moved is drift no type checker sees: lint, types, and tests all
pass while the prose points at a deleted file. This gate resolves backticked
repo-path and dotted ``ubt.*`` references in source comments and docstrings, so
a rename cannot silently leave a comment pointing at a file or symbol that no
longer exists.

It also checks markdown link targets in ``docs/`` and ``README.md``. ``docs/``
is unversioned, so its own references have nothing else watching them.

These three checks lived in ``tests/unit/test_reference_resolvability.py`` before
the suite was rebuilt and were not carried over; the 2026-10 audit found the
drift they would have caught (comments naming deleted modules, a README claim
contradicting the render code). Restored here.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCAN_DIRS = (_REPO_ROOT / "ubt", _REPO_ROOT / "tests")
#: Only unambiguous repo-relative paths are checked; a bare ``foo/bar.py`` may
#: be relative to anywhere and would produce false positives.
_FILE_REF_ROOTS = ("ubt/", "tests/", "scripts/", "docs/")
_FILE_REF = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|md|toml|json|sql))`")
_DOT_REF = re.compile(r"`(ubt(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Dotted spans that are not module references: ``ubt.toml`` (a config file, not a
#: module), and illustrative placeholders in docs/tests (``ubt.pkg.sub``).
_DOT_REF_SKIP = frozenset({"ubt.toml", "ubt.pkg.sub"})


def _resolve_module(parts: list[str]) -> tuple[Path | None, list[str]]:
    """Longest ``ubt.*`` prefix that is a module, plus the trailing attributes."""
    for i in range(len(parts), 0, -1):
        candidate = _REPO_ROOT.joinpath(*parts[:i])
        if candidate.with_suffix(".py").is_file():
            return candidate.with_suffix(".py"), parts[i:]
        if (candidate / "__init__.py").is_file():
            return candidate / "__init__.py", parts[i:]
    return None, []


def test_backticked_references_resolve() -> None:
    problems: list[str] = []
    for path in itertools.chain.from_iterable(root.rglob("*.py") for root in _SCAN_DIRS):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(_REPO_ROOT)
        for match in _FILE_REF.finditer(text):
            ref = match.group(1)
            if ref.startswith(_FILE_REF_ROOTS) and not (_REPO_ROOT / ref).exists():
                problems.append(f"{rel}: file reference `{ref}` does not exist")
        for match in _DOT_REF.finditer(text):
            ref = match.group(1)
            if ref in _DOT_REF_SKIP:
                continue
            module, attrs = _resolve_module(ref.split("."))
            if module is None:
                problems.append(f"{rel}: module reference `{ref}` resolves to no module")
                continue
            identifiers = set(_IDENT.findall(module.read_text(encoding="utf-8", errors="replace")))
            for attr in attrs:
                if attr not in identifiers:
                    problems.append(
                        f"{rel}: symbol reference `{ref}` — {attr!r} not found in "
                        f"{module.relative_to(_REPO_ROOT)}"
                    )
                    break
    assert not problems, "stale code references:\n" + "\n".join(problems)


#: Markdown links whose target is a repo path must resolve. Only *links* are
#: checked, not backticked spans: the docs quote external paths and deliberately
#: retired files in prose, and those are not live references.
_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
#: Code spans/fences are stripped before link detection: the docs quote Typst
#: such as ``#box[...](...)`` in prose, which is not a markdown link. Inline code
#: is matched within a single line (``[^`\n]*``) so a stray backtick cannot
#: swallow a whole region.
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_DOC_LINK_DIRS = (_REPO_ROOT / "docs",)
_DOC_LINK_FILES = (_REPO_ROOT / "README.md",)


def test_doc_markdown_links_resolve() -> None:
    """A link in the docs must point at a file that exists.

    Fragment (``#anchor``) validity is not asserted — only that the file half
    resolves; an anchor-only link (``#section``) is a same-page jump and is
    skipped.
    """
    problems: list[str] = []
    files = (
        [p for root in _DOC_LINK_DIRS for p in root.rglob("*.md")]
        if _DOC_LINK_DIRS[0].is_dir()
        else []
    )
    files.extend(p for p in _DOC_LINK_FILES if p.is_file())
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        prose = _INLINE_CODE.sub("", _FENCED_CODE.sub("", raw))
        rel = path.relative_to(_REPO_ROOT)
        for match in _MD_LINK.finditer(prose):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#", "file://")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            if not (path.parent / target).resolve().exists():
                problems.append(f"{rel}: link target `{target}` does not exist")
    assert not problems, "broken doc links:\n" + "\n".join(problems)


#: Shell harnesses pass their pytest targets as bare argv, not backticked spans,
#: so the backtick scan above cannot see them. A harness whose target was deleted
#: still passes ``bash -n``, is never collected by pytest (``testpaths = ["tests"]``)
#: and is run by hand, so nothing else in the tree would notice. Two harnesses sat
#: broken this way after the unit suite was rebuilt.
_SHELL_SCRIPT_DIRS = (_REPO_ROOT / "scripts",)
_SHELL_TEST_REF = re.compile(r"(?<![\w./-])(tests/[A-Za-z0-9_./-]+\.py)")


def test_shell_harness_test_targets_resolve() -> None:
    """A shell harness must not invoke a pytest file that no longer exists."""
    problems: list[str] = []
    for root in _SHELL_SCRIPT_DIRS:
        for path in root.rglob("*.sh"):
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = path.relative_to(_REPO_ROOT)
            for match in _SHELL_TEST_REF.finditer(text):
                ref = match.group(1)
                if not (_REPO_ROOT / ref).is_file():
                    problems.append(f"{rel}: pytest target `{ref}` does not exist")
    assert not problems, "shell harnesses point at missing tests:\n" + "\n".join(problems)


#: A test file is named for the module it pins, never for the review activity that
#: produced it ("One Behavior, One Home"). Review/round/date names scatter one
#: module's coverage across files and hide it from the next reviewer. Only the
#: *activity suffix* is matched (``..._fixes``, ``..._regressions``, ``roundN``,
#: ``phaseN``, a date): a module genuinely named ``review_endpoints`` or
#: ``page_preview_api`` is fine, so the bare words ``review``/``audit`` are not
#: markers -- they name modules too often.
_ACTIVITY_NAMED = re.compile(
    r"(_fixes?|_regressions?|_round\d+|_phase\d+)(_|$)|(_\d{4}_\d{2}_\d{2})$"
)

#: Pre-existing activity-named files, grandfathered so the gate can land without
#: a risky mass move. Each pins real behavior across several modules; the cases
#: belong in those modules' canonical files, but relocating 24 tests is a
#: separate, reviewable change. The gate blocks *new* offenders meanwhile, and
#: shrinking this set is the intended direction.
_GRANDFATHERED_ACTIVITY_FILES = frozenset(
    {
        "tests/unit/test_adversarial_redteam_fixes.py",
        "tests/unit/test_round3_p2_fixes.py",
        "tests/unit/test_round3_review_fixes.py",
        "tests/unit/test_round4_review_fixes.py",
    }
)


def test_test_files_are_named_for_their_module() -> None:
    offenders = sorted(
        str(path.relative_to(_REPO_ROOT))
        for path in (_REPO_ROOT / "tests").rglob("test_*.py")
        if _ACTIVITY_NAMED.search(path.stem)
        and str(path.relative_to(_REPO_ROOT)) not in _GRANDFATHERED_ACTIVITY_FILES
    )
    assert not offenders, (
        "test files named for the review that produced them instead of the module "
        "they pin; move each case into its module's canonical file:" + "\n" + "\n".join(offenders)
    )
