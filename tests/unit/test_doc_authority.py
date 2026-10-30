"""Guard the documentation-authority policy: what a clone can read, and what a
comment may promise the reader can open.

``docs/`` (the RFCs, the guides, the adaptive-delivery PRD and its execution
documents) is **not** in the repository: each file describes one HEAD and is
rewritten by the next audit round, so versioning them made a stale design note
indistinguishable from a current specification. A clone carries code and tests
only. That is safe only while two things hold, and both had already broken
before anything checked them:

1. **One policy, stated consistently.** The ignore file claimed the guide tree
   *was* versioned ("referenced as the authority ... those links must resolve in
   a fresh clone") while the working tree deleted those guides, and ``README.md``
   said the opposite in the same breath. Two sentences, one repo -- whichever the
   reader trusted, they were wrong half the time.

2. **No comment may point a reader at a file the clone does not have.** A
   ``docs/...`` citation is legal and expected: it names where a decision was
   recorded, and the reason that governs the code belongs in the comment itself.
   A pointer to any other path is an instruction. Two such instructions shipped
   green -- a marker rule in ``pyproject.toml`` sent readers to an
   agent-instructions file that had never existed here, and a corpus harness
   docstring sent them to a corpus note that had just been deleted. The tool the
   standards doc names as the guard, ``scripts/check_doc_refs.py``, cannot see
   this class at all: it walks only the Markdown files under ``docs/``, and its
   pattern matches only ``<file>.py:<line>``, so a ``.md`` target cited from
   ``pyproject.toml`` or a code comment is outside its grammar. Hence a *test*
   rather than a script: a script has to be remembered, and both dead pointers
   were written by people who had just run it green.

Why the detector is narrow. ``book.md``, ``in.md``, ``sample.md`` and
``output.md`` appear thousands of times in this suite as *fixture filenames*
(``tmp_path / "book.md"``), and they are not documentation. A reference is only
judged a documentation pointer when it either names a doc-shaped file (the
basename set below) or sits in pointer prose ("see", "design:", "documented in").
The two rules are asserted both ways by ``test_detector_*``: an injected dead
pointer must be found, a fixture filename must not be.

All checks are pure file reads -- no subprocess (the ``fast`` tier forbids it),
no imports of the tree under test -- so they cost milliseconds and cannot be
fooled by an import side effect.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where a documentation pointer is scanned for. These are the surfaces a fresh
#: clone actually receives, and the ones a reader follows to learn a rule.
_SCANNED_GLOBS: tuple[str, ...] = (
    "ubt/**/*.py",
    "tests/**/*.py",
    "scripts/**/*.py",
    "*.md",
    "*.toml",
    ".pre-commit-config.yaml",
    "web/*.md",
    "deploy/**",
    ".github/workflows/*.yml",
    "examples/**/*.md",
)

#: Sources exempt from the scan. ``CHANGELOG.md`` is a dated record: an entry
#: cites the tree it describes, and rewriting released history to satisfy a
#: present-day check falsifies it -- the same reasoning keeps
#: ``check_doc_refs.py`` out of ``docs/archive`` and ``docs/reviews``.
_EXCLUDED_SOURCE_PARTS: frozenset[str] = frozenset(
    {"output", "node_modules", ".venv", "__pycache__", "baseline", "documents"}
)
_EXCLUDED_SOURCE_NAMES: frozenset[str] = frozenset({"CHANGELOG.md"})

#: A markdown path reference. The leading character must be a name character, so
#: a glob written in prose (``docs/``, ``*.md``, ``docs/**/*.md``) is not read as
#: a path at all.
_MD_REF_RE = re.compile(r"(?P<path>[A-Za-z0-9_][A-Za-z0-9_.\-/]*\.md)")

#: Basenames that are documentation by shape, wherever they appear. ``README.md``
#: and ``DESIGN.md`` are the ones this repo actually uses; the rest are the names
#: a dead pointer historically took (``AGENTS.md``, ``SECURITY.md``,
#: ``CONTRIBUTING.md``, ``*_STRATEGY.md``, ``*_RFC.md``).
_DOC_BASENAME_RE = re.compile(
    r"(?:README|DESIGN|ARCHITECTURE|GLOSSARY|CHANGELOG|SECURITY|CONTRIBUTING|AGENTS"
    r"|SPEC|PLAN|NOTES|GUIDE|STRATEGY|RFC|PRD|RECORD|LOG|ANALYSIS|REVIEW)\.md$"
)

#: Prose that tells a reader to go look somewhere. A match in such a line is an
#: instruction, whatever the basename looks like.
_POINTER_PROSE_RE = re.compile(
    r"\b(?:see|refer to|documented in|described in|as described|for the plan|per|design|follows)\b"
    r"|依据|见[^a-zA-Z]",
    re.IGNORECASE,
)

#: Foreign trees the design reviews cite. Their absence here is expected.
_FOREIGN_MARKERS = ("TranslateBooksWithLLMs", "pdf2zh", "VI-Translate", "TBL")


def _is_docs_citation(target: str) -> bool:
    """Whether *target* lives in the un-versioned ``docs/`` note tree."""
    return target.startswith("docs/") or "/docs/" in target


def _dead_pointers(relative: str, text: str, *, base: Path | None = None) -> list[str]:
    """Documentation pointers in *text* that do not resolve in this tree.

    ``relative`` labels findings only; ``base`` is the directory the references
    are written from -- a link in ``web/README.md`` to ``DESIGN.md`` means
    ``web/DESIGN.md``, so the referencing file's own directory is tried first and
    the repo root second (the two spellings both appear in this tree).
    """
    root = _REPO_ROOT
    dirs = (base or root, root)
    problems: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if any(marker in line for marker in _FOREIGN_MARKERS):
            continue
        for match in _MD_REF_RE.finditer(line):
            raw = match.group("path")
            target = raw.lstrip("./")
            if _is_docs_citation(target):
                continue
            basename = target.rsplit("/", 1)[-1]
            if not (_DOC_BASENAME_RE.search(basename) or _POINTER_PROSE_RE.search(line)):
                continue
            if not any((directory / target).is_file() for directory in dirs):
                problems.append(f"{relative}:{lineno}: pointer to {raw!r} does not resolve")
    return problems


def _scanned_files() -> list[Path]:
    out: list[Path] = []
    for pattern in _SCANNED_GLOBS:
        for path in _REPO_ROOT.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(_REPO_ROOT)
            if any(part in _EXCLUDED_SOURCE_PARTS for part in rel.parts):
                continue
            if path.name in _EXCLUDED_SOURCE_NAMES:
                continue
            if path.suffix not in {".py", ".md", ".toml", ".yaml", ".yml"}:
                continue
            # Self-exemption, stated rather than hidden: this file *names* the
            # basenames it looks for and quotes the dead pointers it demonstrates
            # (``AGENTS.md``, the corpus note). Scanning it would flag its own
            # vocabulary, and the fix for that -- paraphrasing the strings away
            # from what they detect -- would make the guard worse than the gap.
            if path == Path(__file__).resolve():
                continue
            out.append(path)
    return sorted(set(out))


_SCANNED = _scanned_files()


@pytest.mark.parametrize("source", _SCANNED, ids=[str(p.relative_to(_REPO_ROOT)) for p in _SCANNED])
def test_no_pointer_to_a_file_the_clone_does_not_have(source: Path) -> None:
    """Every documentation pointer in the shipped tree must resolve.

    ``docs/`` targets are exempt by policy: they are citations into the
    un-versioned notes, and the reason they name is required to be inlined in the
    citing comment.
    """
    rel = str(source.relative_to(_REPO_ROOT))
    problems = _dead_pointers(
        rel, source.read_text(encoding="utf-8", errors="replace"), base=source.parent
    )
    assert not problems, "documentation pointer(s) a fresh clone cannot follow:\n" + "\n".join(
        problems
    )


def test_detector_flags_a_dead_pointer_and_ignores_a_fixture_name() -> None:
    """The detector must fire on the real failure and stay silent on noise.

    Both halves matter. The false negative is what happened: ``see AGENTS.md``
    sat in a file no tool read. The false positive is what would make the check
    get deleted: this suite builds fixtures named ``book.md``, ``in.md`` and
    ``sample.md`` by the thousand, and a guard that calls them dead pointers is a
    guard nobody keeps switched on.
    """
    dead = "    # A legacy check moves here instead of being deleted; see AGENTS.md §2.\n"
    assert _dead_pointers("pyproject.toml", dead), "an unresolvable AGENTS.md pointer must be found"

    corpus = '"""Harnesses read the real corpus PDFs (see ``corpus/README.md``)."""\n'
    assert _dead_pointers("tests/x.py", corpus), "an unresolvable corpus pointer must be found"

    fixture = '    src = tmp_path / "book.md"\n    src.write_text("# Title\\n")\n'
    assert not _dead_pointers("tests/y.py", fixture), "a Markdown *fixture* is not a doc pointer"

    prose = '    out = tmp_path / "report.md"  # writes report.md next to the artifact\n'
    assert not _dead_pointers("scripts/z.py", prose), "a generated output name is not a doc pointer"

    citation = "Design: ``docs/design/adaptive_delivery_prd.md`` §7.3.\n"
    assert not _dead_pointers("ubt/core/layout/cels.py", citation), (
        "a docs/ citation must be legal: the notes are un-versioned on purpose"
    )

    live = "See `deploy/docker/README.md` for container options.\n"
    assert not _dead_pointers("README.md", live), "a pointer that resolves must pass"

    sibling = "See DESIGN.md for the token table.\n"
    assert not _dead_pointers("web/README.md", sibling, base=_REPO_ROOT / "web"), (
        "a bare name resolves against the referencing file's own directory"
    )
    assert _dead_pointers("web/README.md", sibling), "...and is not a repo-root file"

    glob_prose = "It walks only the Markdown files under docs/ and matches `<file>.py:<line>`.\n"
    assert not _dead_pointers("x.md", glob_prose), "a glob written in prose is not a path"


def test_docs_tree_is_unversioned_with_no_exceptions() -> None:
    """``docs/`` is ignored whole.

    A single ``!/docs/guides/`` negation re-opens the directory and silently puts
    the notes back under version control, which is how the contradiction in the
    first place started. So the rule must be present and no negation may name
    anything inside it.
    """
    ignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    rules = [
        line.strip() for line in ignore.splitlines() if line.strip() and not line.startswith("#")
    ]
    assert "/docs/" in rules, ".gitignore must ignore the whole docs/ tree"
    exceptions = [rule for rule in rules if rule.startswith("!") and "docs/" in rule]
    assert not exceptions, (
        f"docs/ is un-versioned; re-including it contradicts README: {exceptions}"
    )


def test_the_unversioned_docs_policy_is_stated_in_the_versioned_docs() -> None:
    """A reader of the *shipped* docs learns the notes are not shipped.

    Without this sentence a ``docs/...§7.3`` citation reads like a broken link,
    and the reader concludes either that the repo is sloppy or that the decision
    was never written down. Both files must say it: ``README.md`` is where a user
    looks, ``ARCHITECTURE.md`` is where a contributor looks before editing.
    """
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    arch = (_REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "unversioned" in readme, "README must say docs/ is not versioned"
    assert "Docs are not in the repo" in arch, "ARCHITECTURE must own the policy"
    assert "citation" in arch, "ARCHITECTURE must define what a docs/...§N citation means"
