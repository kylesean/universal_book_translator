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
#: be relative to anywhere and would produce false positives. ``docs/`` is
#: intentionally unversioned (gitignored), so it counts as a checked root only
#: while it is present in the working tree — a fresh clone without docs must not
#: fail on the many ``docs/...`` references in code comments.
_FILE_REF_ROOTS = tuple(
    root for root in ("ubt/", "tests/", "scripts/", "docs/") if (_REPO_ROOT / root).exists()
)
_FILE_REF = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|md|toml|json|sql))`")
_DOT_REF = re.compile(r"`(ubt(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Dotted spans that are not module references: ``ubt.toml`` (a config file, not a
#: module), illustrative placeholders in docs/tests (``ubt.pkg.sub``), and
#: entry-point groups (``ubt.spi``).
_DOT_REF_SKIP = frozenset({"ubt.toml", "ubt.pkg.sub", "ubt.spi"})


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

#: A pre-existing activity-named file, grandfathered so the gate can land without
#: a risky mass move. It pins real behavior across several modules; the cases
#: belong in those modules' canonical files, but relocating them is a separate,
#: reviewable change. The gate blocks *new* offenders meanwhile, and shrinking
#: this set is the intended direction.
_GRANDFATHERED_ACTIVITY_FILES = frozenset(
    {
        "tests/unit/test_adversarial_redteam_fixes.py",
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


# --------------------------------------------------------------------------- #
# User-facing prose must not name a CLI flag, config field or mode value that
# the code does not have. A README describing a deleted engine is invisible to
# every other gate: the links resolve, the types check, the tests pass.
# --------------------------------------------------------------------------- #

#: A ``--flag`` in prose. The lookbehind keeps a negative number or a range
#: (``4pt-12pt``) from matching, and the trailing boundary keeps ``--render-mode``
#: from being read as ``--render``.
_DOC_FLAG = re.compile(r"(?<![\w-])--([a-z][a-z0-9-]+)")

#: Flags the docs legitimately mention that are not on the ``ubt`` CLI: shell
#: builtins and third-party tools (``pip install --extra``, ``git --no-verify``).
_DOC_FLAG_ALLOWLIST = frozenset(
    {
        "no-verify",  # git
        "extra",  # pip
        "force-exclude",  # ruff/pre-commit
        "all-files",  # pre-commit
        "no-cache",  # pip/uv
    }
)


def _cli_flag_names() -> set[str]:
    """Every option name the ``ubt`` CLI actually accepts, long form, no dashes."""
    import typer.main

    from ubt.cli.main import app

    click_app = typer.main.get_command(app)
    names: set[str] = set()

    def walk(command: object) -> None:
        for param in getattr(command, "params", []):
            for opt in getattr(param, "opts", []) + getattr(param, "secondary_opts", []):
                if opt.startswith("--"):
                    names.add(opt[2:].split("=", 1)[0])
        for sub in getattr(command, "commands", {}).values():
            walk(sub)

    walk(click_app)
    return names


def test_readme_flags_exist_on_the_cli() -> None:
    """A ``--flag`` the README tells a user to type must be a real option.

    The README described ``--render-mode`` (and its ``UBT_RENDER_MODE`` env var)
    long after the overlay engine was deleted, so a user following it got
    ``No such option``. The links resolved and every other gate passed; only a
    check that reads the prose against the live CLI catches this class.
    """
    readme = _REPO_ROOT / "README.md"
    if not readme.is_file():
        pytest.skip("README is not present in this checkout")
    known = _cli_flag_names()
    mentioned = set(_DOC_FLAG.findall(readme.read_text(encoding="utf-8")))
    unknown = sorted(mentioned - known - _DOC_FLAG_ALLOWLIST)
    assert not unknown, (
        "README names CLI flag(s) the `ubt` command does not accept (a deleted "
        f"feature leaves prose behind): {unknown}"
    )


def test_readme_env_vars_exist_on_the_config() -> None:
    """A ``UBT_*`` variable the README names must be real.

    Same failure mode as the flags: ``UBT_RENDER_MODE`` outlived the field it set.

    There are two legitimate kinds of ``UBT_*`` variable, and the check has to
    accept both: config-model fields (``ubt config`` lists these), and
    process-level switches read directly from the environment at their point of
    use (``UBT_LOG_LEVEL`` in ``core/log_config.py``, ``UBT_ALLOW_NO_AUTH`` in
    ``api/security.py``, …) which deliberately never become model fields because
    they configure the process, not a translation. A variable in neither group
    is stale prose.
    """
    from ubt.core.config import UBTConfig, env_var_names

    readme = _REPO_ROOT / "README.md"
    if not readme.is_file():
        pytest.skip("README is not present in this checkout")
    declared: set[str] = set()
    for field in UBTConfig.model_fields:
        declared |= set(env_var_names(field))
    mentioned = set(re.findall(r"\bUBT_[A-Z0-9_]+\b", readme.read_text(encoding="utf-8")))
    unknown = sorted(mentioned - declared - _process_level_env_vars())
    assert not unknown, f"README names UBT_* variable(s) that nothing reads: {unknown}"


def _process_level_env_vars() -> set[str]:
    """``UBT_*`` names the package reads straight from the environment.

    Derived by scanning the source rather than hand-listed, so a deleted switch
    cannot leave an allowance behind for the README to keep naming it.
    """
    found: set[str] = set()
    pattern = re.compile(r"""["'](UBT_[A-Z0-9_]+)["']""")
    for path in (_REPO_ROOT / "ubt").rglob("*.py"):
        found |= set(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
    return found


def test_readme_mode_values_exist_in_the_code() -> None:
    """A quoted mode value in the README must be one the engine defines.

    ``alternating``/``facing`` were described as bilingual modes after the
    ``DualMode`` literal was cut to ``mono``/``dual``/``auto``.
    """
    from typing import get_args

    from ubt.core.config import DualMode

    readme = _REPO_ROOT / "README.md"
    if not readme.is_file():
        pytest.skip("README is not present in this checkout")
    text = readme.read_text(encoding="utf-8")
    known = set(get_args(DualMode))
    # Any value named as a dual_mode choice in prose must be real. Match the
    # backticked/quoted spellings the README uses for mode names.
    for value in ("mono", "dual", "auto", "alternating", "facing", "inline"):
        if re.search(rf"[`\"']{value}[`\"']", text) and value not in known:
            raise AssertionError(
                f"README names bilingual mode {value!r}, which DualMode does not "
                f"define (real values: {sorted(known)})"
            )
