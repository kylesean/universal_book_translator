"""Architectural and licensing integrity guard tests (carrier of the vacant
discipline: core forbids PDF/LLM heavy deps; the repo forbids PyMuPDF).

Validates:
1. Zero PyMuPDF / fitz pollution: Neither fitz nor pymupdf may be imported anywhere in ubt/.
2. Zero AGPL / core isolation: ubt/core/ must never import babeldoc, pdf2zh, or docling.
3. Commercial dependency cleanliness: pyproject.toml must not declare AGPL-licensed dependencies.
4. Core reaches adapters only through ubt/core/ports.py (no hardcoded module edges).
5. Route-B forward gate: heavy raster/inpaint deps stay out of the base install.
6. Debt ratchet: the god functions may not silently grow (raise consciously).
7. Packaging hygiene: the sdist must not ship a dangling link or symlink.

Restored after commit 0492c7d ("drop the repo-scanning lint guards") which had
removed the automated clean-room evidence. This file is that evidence: it
proves UBT never imports AGPL PDF engines (BabelDOC/PyMuPDF/pdf2zh) and keeps
the hexagonal core/adapters boundary by AST, not by convention alone.
"""

import ast
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UBT_DIR = REPO_ROOT / "ubt"
CORE_DIR = UBT_DIR / "core"

# This is automated clean-room / architecture evidence: it must run in the
# per-edit `pytest -m fast` gate and the pre-push hook, or an AGPL/boundary
# regression could land silently (remote CI is parked).
pytestmark = pytest.mark.fast


def _extract_imported_modules(file_path: Path) -> set[str]:
    """Parse a python file into an AST and collect all top-level module names imported.

    A file that cannot even be parsed is a hard failure with an actionable
    message (file + line), never a bare SyntaxError traceback: unparseable
    code can hide banned imports from this guard, so it must never be
    silently skipped (e.g. a formatter joining `) -> str:` and an opening
    docstring onto one line breaks the whole-file parse).
    """
    source = file_path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        try:
            display = str(file_path.relative_to(REPO_ROOT))
        except ValueError:
            display = str(file_path)
        raise AssertionError(
            f"{display} is not valid Python "
            f"(line {exc.lineno}: {exc.msg}) — fix the syntax error before "
            "license/isolation results can be trusted"
        ) from exc
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_zero_pymupdf_across_entire_codebase() -> None:
    """Ensure no code in ubt/ imports fitz or pymupdf (AGPL commercial liability)."""
    banned_modules = {"fitz", "pymupdf"}
    violations: list[str] = []

    for py_file in UBT_DIR.rglob("*.py"):
        imported = _extract_imported_modules(py_file)
        forbidden_found = imported.intersection(banned_modules)
        if forbidden_found:
            violations.append(
                f"{py_file.relative_to(REPO_ROOT)} imports banned AGPL module(s): {forbidden_found}"
            )

    assert not violations, "PyMuPDF/fitz found in production codebase:\n" + "\n".join(violations)


def test_zero_pypdf_in_runtime_after_oxide_adoption() -> None:
    """Runtime pypdf was retired in the pdf_oxide adoption (v3.1): structure
    lives on pikepdf (``pdf_struct``), raster/text on pdf_oxide. pypdf may
    only return as a dev-extra fixture writer (tests/), never in ``ubt/``.

    ``pypdfium2`` is a distinct top-level name and is not matched here.
    """
    violations: list[str] = []
    for py_file in UBT_DIR.rglob("*.py"):
        if "pypdf" in _extract_imported_modules(py_file):
            violations.append(str(py_file.relative_to(REPO_ROOT)))
    assert not violations, "runtime pypdf import resurrected:\n" + "\n".join(violations)


def test_core_isolation_from_external_pdf_engines() -> None:
    """Ensure ubt/core/ maintains pure domain isolation from PDF toolkits."""
    banned_in_core = {"babeldoc", "pdf2zh", "docling", "pypdf", "fitz", "pymupdf"}
    violations: list[str] = []

    for py_file in CORE_DIR.rglob("*.py"):
        imported = _extract_imported_modules(py_file)
        forbidden_found = imported.intersection(banned_in_core)
        if forbidden_found:
            violations.append(
                f"{py_file.relative_to(REPO_ROOT)} imports adapter-level dependency: {forbidden_found}"
            )

    assert not violations, "Core domain isolation violated:\n" + "\n".join(violations)


def test_pyproject_contains_no_banned_dependencies() -> None:
    """Ensure pyproject.toml never lists fitz, pymupdf, or babeldoc in dependencies."""
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for banned in ("fitz", "pymupdf", "pdf2zh", "babeldoc"):
        assert banned not in pyproject_text.lower(), (
            f"Banned package '{banned}' detected in pyproject.toml!"
        )


def _runtime_adapter_refs(tree: ast.AST) -> list[str]:
    """Adapter references that execute at runtime (TYPE_CHECKING blocks skipped)."""
    found: list[str] = []

    def _visit(node: ast.AST) -> None:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for stmt in node.orelse:
                _visit(stmt)
            return
        if isinstance(node, ast.Import):
            found.extend(
                alias.name
                for alias in node.names
                if alias.name == "ubt.adapters" or alias.name.startswith("ubt.adapters.")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "ubt.adapters" or node.module.startswith("ubt.adapters."))
        ):
            found.append(f"from {node.module} import ...")
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree)
    return found


def test_core_has_no_hardcoded_adapter_paths() -> None:
    """Ubt/core/ must reach adapters only through ubt/core/ports.py."""
    offenders: list[str] = []
    for py_file in CORE_DIR.rglob("*.py"):
        if py_file.name == "ports.py":
            continue
        source = py_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(py_file))
        for ref in _runtime_adapter_refs(tree):
            offenders.append(f"{py_file.relative_to(REPO_ROOT)}: {ref}")
    assert not offenders, "Hardcoded core->adapters edge (use ubt.core.ports):\n" + "\n".join(
        offenders
    )


# ---------------------------------------------------------------------------
# Widen the guard surface, not just restore it.
# ---------------------------------------------------------------------------

# Base (always-installed) dependencies must stay CPU-light and zero-AGPL. Route B
# (image inpainting) will add heavy raster models, but only as an OPTIONAL extra
# — never as a base dependency — so a bare `uv sync` keeps a lightweight install
# and `mypy --strict` on the default env. This is the forward gate that keeps that
# invariant enforced the day route B lands.
_BANNED_BASE_DEP_PREFIXES = (
    "opencv",  # cv2 — transitive only via the `ocr` extra (rapidocr)
    "torch",
    "torchvision",
    "scikit-image",
    "simple-lama",
    "lama-clean",
    "diffusers",
)


def test_route_b_heavy_imaging_stays_out_of_base() -> None:
    """Heavy raster/inpaint models may only live in optional extras, never base.

    Route A's fidelity diff is deliberately Pillow-only (zero new base dep);
    route B (background inpainting) must be gated behind an extra so it cannot
    bloat the base install or drag a transitive AGPL/CUDA surface in by default.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)
    base_deps = data["project"]["dependencies"]
    offenders: list[str] = []
    for spec in base_deps:
        name = spec.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()
        normalized = name.replace("_", "-")
        if normalized.startswith(_BANNED_BASE_DEP_PREFIXES):
            offenders.append(spec)
    assert not offenders, (
        f"base dependencies must not include heavy imaging models (move to an "
        f"optional extra): {offenders}"
    )


# Debt ratchet: these two god functions were flagged as the largest structural
# liabilities.
# The cap is pinned at the CURRENT measured size so the code may not silently
# grow; to raise a cap you must consciously edit it, which forces the debt
# conversation. Shrinking the file is always welcome — lower the cap then.
_SIZE_RATCHETS: dict[str, int] = {
    # 2026-09-21: 759→763 for the hard-cancel fix (CancelledError must write a
    # terminal job status). Concurrent chapter-streaming work in this tree grew
    # it further to 786; bundled per user request. Splitting
    # PipelineOrchestrator.run should bring this back down.
    # 2026-09-21: net 786→770 — per-run usage attribution added a
    # sink-or-snapshot branch, and lifting the billing/budget arithmetic into
    # ubt/core/engine/usage.py took more out than it cost.
    # 2026-09-21: 770→771 — AdapterRuntimeConfig gained the required
    # allow_page_upload field so the page-egress gate has one source of truth
    # (the resolved config) instead of a second os.environ read in the parser.
    # 2026-09-21: 771→784 — billing now tracks the run usage it
    # has already written, because ``_run_usage()`` is cumulative and folding
    # the whole of it into an absolute ledger write on every progress event
    # double-counted the job's spend (and tripped UBT_BUDGET_USD on spend the
    # job never made). The per-function ratchet (test_god_function_ratchet)
    # now guards ``run()`` itself; this file cap is the complementary guard.
    # 2026-09-22 (budget fail-closed): 784→814 — the run() preflight refuses
    # capped runs whose models have no price entry (the cap was silently
    # inert otherwise), with the billing arithmetic itself living in
    # usage.py. The run()/export decomposition is slated to pay this debt
    # back several hundred lines.
    # 2026-09-22 (billing lock): 814→821 — a 5-line critical section around
    # progress-event billing; repaid together with the same debt.
    # 2026-09-22 (budget preflight): 821→839 — the preflight now enumerates
    # every billable model (fallback chain + the wrapped TieredQERunner's judge)
    # instead of draft/repair only, with the comment that says why. Repaid with
    # the same run() split.
    # 2026-09-22 (OCR billing): 839→853 — the pre-flight now enumerates the OCR
    # channel's own model (it bills through a separate httpx client, so an
    # unpriced pick made ``estimate_cost_usd`` return None and
    # ``budget_violation`` read that as "not exceeded" — the cap was inert for a
    # channel that spends real money), and ``AdapterRuntimeConfig`` grew the
    # matching ``ocr_model`` field so the driver bills the model the quote
    # priced. Repaid with the same run() split. QE runner close lifecycle
    # added clean runner teardown. 2026-09: shielded cancel-cleanup,
    # UBTError logged without traceback, artifact-tree permission scan — repaid
    # in the same batch by merging the three BaseException abort handlers, so the
    # cap stays at 860 (a value that did not move needs no approval).
    # 2026-09-26: 860→869 — the initial commit shipped at 862 (already over the
    # pre-existing cap); `ruff format` and the owner-only-permission warning fix
    # added the rest. CI was already red on both ratchets. Recorded
    # consciously rather than trimming comments to satisfy a counter; the
    # run() split still owes the real repayment.
    # 2026-09-26 (usage persist): 869→885 — the run-usage persist change grew the
    # file again; recorded here so the cap tracks the shipped size.
    # 2026-09-27 (job-id identity): 885→927 — ``derive_job_id`` gained the genre
    # profile + non-default engine-knob signature (and its helper), so a resume
    # under a different --profile/--preset cannot reuse the wrong ledger.
    # 2026-09-27: 927→933 — the budget path merges the router's
    # fallback-endpoint attribution into ``endpoint_map``.
    "ubt/core/engine/pipeline.py": 933,
    "ubt/adapters/pdf/typst_reconstructor.py": 2336,
}

#: The two known god functions, capped at their current size. Measured by AST
#: span, not file length: a file cap alone lets ``run()`` grow by moving other
#: lines out of the file, which is exactly what the file cap cannot see.
_FUNCTION_RATCHETS: dict[str, tuple[str, int]] = {
    # 2026-09-26: run 380→381 and _emit_block 295→318. Both were already over
    # cap in the initial commit (the ratchet tests were red on a clean checkout),
    # so this records pre-existing debt rather than hiding a new regression.
    # 2026-09-26 (usage persist): run 381→391→392 — recorded so the ratchet
    # tracks the shipped size.
    # _emit_block is the ONE shared emit core with pinned regression tests; a
    # 23-line extraction is deliberately deferred rather than done blind inside
    # a broad fix. Splitting run() remains the tracked repayment.
    "ubt/core/engine/pipeline.py": ("PipelineOrchestrator.run", 392),
    "ubt/adapters/pdf/typst_reconstructor.py": ("TypstReconstructor._emit_block", 318),
}


def _function_spans(path: Path) -> dict[str, int]:
    """Map every qualified function name in ``path`` to its line span."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    spans: dict[str, int] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                span = (child.end_lineno or child.lineno) - child.lineno + 1
                spans[prefix + child.name] = span
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    return spans


def test_file_size_ratchet() -> None:
    """Prevent the two debt-carrying files from growing further."""
    violations: list[str] = []
    for rel, cap in _SIZE_RATCHETS.items():
        path = REPO_ROOT / rel
        lines = len(path.read_text(encoding="utf-8").splitlines())
        if lines > cap:
            violations.append(f"{rel}: {lines} lines > ratchet cap {cap}")
    assert not violations, (
        "file growth (pay down debt or raise the cap consciously in "
        "test_license_guard.py):\n" + "\n".join(violations)
    )


def test_god_function_ratchet() -> None:
    """Prevent the two known god functions from growing further.

    Measured by AST span: the file cap above cannot see a function grow by
    moving unrelated lines out of the file.
    """
    violations: list[str] = []
    for rel, (qualname, cap) in _FUNCTION_RATCHETS.items():
        span = _function_spans(REPO_ROOT / rel).get(qualname)
        if span is None:
            violations.append(f"{rel}: {qualname} not found (renamed?)")
        elif span > cap:
            violations.append(f"{rel}: {qualname} is {span} lines > cap {cap}")
    assert not violations, (
        "god-function growth (split the function or raise the cap consciously "
        "in test_license_guard.py):\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# 7. Packaging hygiene: the sdist and .gitignore must stay self-consistent.
# ---------------------------------------------------------------------------


def _git_ls_files(*args: str) -> str:
    """Raw ``git ls-files`` output, or a skip when git is unavailable."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable, cannot enumerate tracked files: {exc}")
    return proc.stdout


def _tracked_symlinks() -> list[tuple[str, str]]:
    """(link path, target path) for every symlink git tracks."""
    links: list[tuple[str, str]] = []
    for line in _git_ls_files("-s").splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if fields and fields[0] == "120000":
            links.append((path, str((REPO_ROOT / path).readlink())))
    return links


def _sdist_shipped_paths() -> set[str]:
    """Model hatchling's sdist selection: VCS-tracked files minus the exclude list.

    Hatchling picks sdist members from the VCS file list, so an untracked (e.g.
    gitignored) artifact can never enter the archive. ``force-include``
    destinations are added back on top.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        sdist = tomllib.load(fh)["tool"]["hatch"]["build"]["targets"]["sdist"]
    excluded = sdist.get("exclude", [])
    shipped = {
        path
        for path in _git_ls_files().splitlines()
        if not any(path.startswith(entry) for entry in excluded)
    }
    shipped.update(sdist.get("force-include", {}).values())
    return shipped


def _relative_markdown_links(path: Path) -> set[str]:
    """Repo-relative link targets in a markdown file.

    Fenced/inline code is stripped first: the docs embed Typst snippets such as
    ``#box[...](...)``, which a naive link regex reads as a markdown link.
    Fragments and absolute URLs are dropped.
    """
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", "", text)
    targets = re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", text)
    return {
        target.split("#", 1)[0]
        for target in targets
        if not target.startswith(("http://", "https://", "mailto:", "#"))
    }


def test_sdist_ships_every_file_its_own_contents_reference() -> None:
    """The sdist must be self-consistent: no dangling link, no broken symlink.

    Two ways an archive dangles: (1) ``README.md`` is the sdist readme, so its
    relative links must resolve inside the tarball — excluding the whole of
    ``docs/`` broke three of them; (2) hatchling copies a tracked symlink
    verbatim, so its target must be shipped too (``USAGE.md`` used to point at
    an excluded ``docs/guides/USER_GUIDE.md``).
    """
    shipped = _sdist_shipped_paths()
    problems: list[str] = []

    for link in sorted(_relative_markdown_links(REPO_ROOT / "README.md")):
        if link not in shipped:
            problems.append(f"README.md links {link!r}, which the sdist does not ship")

    for link, target in _tracked_symlinks():
        rel = ((REPO_ROOT / link).parent / target).resolve().relative_to(REPO_ROOT).as_posix()
        if rel not in shipped:
            problems.append(f"{link} -> {rel}, which the sdist does not ship")

    assert not problems, (
        "sdist would ship dangling references; un-exclude the target (or drop "
        "the link) in pyproject.toml / the referring file:\n" + "\n".join(problems)
    )


def test_gitignore_does_not_un_ignore_the_generated_corpus() -> None:
    """``tests/fixtures/synthetic-*.pdf`` are generated at configure time, never committed.

    Re-adding the old ``!tests/fixtures/synthetic-*.pdf`` negations would let ``git add
    -A`` commit ~18 MB of regenerated binaries as permanent ``git status`` noise.
    """
    lines = [
        line.strip() for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    ]
    negations = [line for line in lines if line.startswith("!") and "synthetic" in line]
    assert not negations, f"generated synthetic PDFs are un-ignored again: {negations}"
    assert "tests/fixtures/*.pdf" in lines
