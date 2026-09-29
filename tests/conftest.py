"""Keep the suite independent of the machine it runs on.

``UBTConfig`` deliberately honors ambient ``UBT_*`` process environment
variables -- what a real run should do, and what makes "assert the default" a
statement about the developer's laptop. An ambient ``UBT_API_MODE=responses``
turned nine credential-isolation tests red with nothing wrong in the code, which
is exactly the kind of failure that teaches people to ignore red.

Every test therefore runs with the ambient credential/config variables removed;
a test that wants a value sets it explicitly, as most already do through
``monkeypatch.setenv``.

The console is pinned plain for the same reason. Typer forces a colour terminal
whenever ``GITHUB_ACTIONS`` is set, and rich's option-name highlighter then
splits ``--fresh`` across an escape sequence — so ``assert "--flag" in
result.stdout`` holds on a laptop and fails in CI.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests.stage_ctx_factory import build_stage_ctx
from ubt.core.engine.stage_context import StageContext

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_synthetic_corpus() -> str | None:
    """Regenerate tests/fixtures/synthetic-*.pdf when the suite deleted-or-missed them.

    The corpus is generated, not committed: ``scripts/
    make_sample_corpus.py`` is the source of truth and the acceptance runbook
    says so. Commit 62fcd75 removed the last committed copies, which silently
    turned eleven real-PDF tests red on every fresh checkout. Rebuilding them
    here at configure time (before collection) keeps the repo's existing
    ``skipif(not exists)`` guards honest: they now only fire when generation
    is genuinely impossible (no typst binary), never because nobody ran the
    generator. ``generate_all()`` writes atomically, so racing xdist workers
    each see either a complete file or their own complete build.
    """
    corpus = _REPO_ROOT / "tests" / "fixtures"
    script = _REPO_ROOT / "scripts" / "make_sample_corpus.py"
    spec = importlib.util.spec_from_file_location("ubt_make_sample_corpus", script)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        return f"cannot load {script}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    needed: list[str] = [n for n in module.CORPUS_NAMES if not (corpus / n).exists()]
    if not needed:
        return None
    if shutil.which("typst") is None:
        return f"typst not on PATH; missing {', '.join(needed)}"
    try:
        module.generate_all()
    except (Exception, SystemExit) as exc:
        return f"corpus generation failed: {exc}"
    return None


def pytest_configure(config: pytest.Config) -> None:
    """Render CLI help without colour, in CI and locally alike.

    Runs before collection, which is the requirement: typer resolves
    ``_TYPER_FORCE_DISABLE_TERMINAL`` once, in its module body, the first time
    ``ubt.cli.main`` is imported.
    """
    os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
    reason = _ensure_synthetic_corpus()
    if reason:
        print(
            f"WARNING: synthetic PDF corpus unavailable ({reason}); real-PDF tests will skip.",
            file=sys.stderr,
        )


# Prefixes UBTConfig reads: its own plus the provider names the credential
# fallback in ``api_key`` / ``base_url`` accepts.
_AMBIENT_PREFIXES = ("UBT_", "OPENCODE_", "OPENAI_", "ANTHROPIC_", "GEMINI_", "DEEPSEEK_")

# Test-harness control knobs that share the UBT_ prefix but are not config:
# stripping them silently broke the documented golden-regeneration flow
# (UBT_UPDATE_GOLDENS never reached assert_no_kpi_regression).
_HARNESS_ALLOWLIST = frozenset({"UBT_UPDATE_GOLDENS"})


#: What a machine with fonts-noto-cjk installed reports to the renderer. Tests
#: that assert *which family lands in the Typst preamble* are asserting the
#: resolver's decision; without this pin they also assert the runner's font
#: inventory, which is why five of them went red the first time the Windows job
#: reported anything. The inventory-free behaviour is covered by feeding
#: resolve_font_stack a fictional font list directly (test_font_probe.py).
NOTO_CJK_INVENTORY = frozenset(
    {
        "Noto Serif CJK SC",
        "Noto Sans CJK SC",
        "Noto Serif CJK TC",
        "Noto Sans CJK TC",
        "Noto Serif CJK JP",
        "Noto Sans CJK JP",
        "Noto Serif CJK KR",
        "Noto Sans CJK KR",
        "Source Han Serif SC",
        "Source Han Serif JP",
        "Source Han Serif KR",
        "Liberation Serif",
        "DejaVu Sans",
    }
)


@pytest.fixture
def noto_cjk_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the renderer's font view so preamble assertions are portable."""
    from ubt.adapters.pdf import font_probe

    monkeypatch.setattr(
        font_probe, "available_font_families", lambda typst_binary="typst": NOTO_CJK_INVENTORY
    )


@pytest.fixture
def system_probe_path() -> str:
    """An absolute path inside an OS system directory, for deny-list tests.

    Three tests asserted the refusal of ``/etc/passwd``. On Windows that string
    resolves under the current drive's ``\\etc\\passwd`` — not a system directory —
    so the guard answered correctly with the *containment* refusal and the
    assertion failed by comparing the suite's platform to the product's. The
    behaviour under test ("a system path is refused by name, fail closed") is
    platform-neutral; only the example was not.
    """
    if sys.platform == "win32":
        root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR") or r"C:\Windows"
        return str(Path(root) / "System32" / "config" / "SAM")
    return "/etc/passwd"


@pytest.fixture
def make_stage_ctx(tmp_path: Path) -> Callable[..., StageContext]:
    """See :func:`build_stage_ctx` in ``tests/stage_ctx_factory.py``."""
    return lambda **over: build_stage_ctx(tmp_path, **over)


@pytest.fixture(autouse=True)
def hermetic_config() -> Iterator[None]:
    """Neutralize ambient config for every test."""
    _PROXY_VARS = (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "NO_PROXY",
        "no_proxy",
    )
    saved_proxy = {name: os.environ.pop(name) for name in _PROXY_VARS if name in os.environ}
    saved = {
        name: value
        for name, value in os.environ.items()
        if name.upper().startswith(_AMBIENT_PREFIXES) and name not in _HARNESS_ALLOWLIST
    }
    for name in saved:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        os.environ.update(saved)
        os.environ.update(saved_proxy)


@pytest.fixture
def live_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the operator's live-provider env for one test body.

    ``hermetic_config`` (autouse, above) strips every ``UBT_*`` var before each
    test, but ``requires_live_llm`` is evaluated at collection against the real
    environment. Without this, a configured live run was un-skipped and then ran
    with ``mock-key`` against ``api.openai.com``. The snapshot is captured in
    ``_live_helpers`` at import time, before any fixture runs.
    """
    from tests.integration._live_helpers import LIVE_ENV_SNAPSHOT

    for key, value in LIVE_ENV_SNAPSHOT.items():
        monkeypatch.setenv(key, value)
