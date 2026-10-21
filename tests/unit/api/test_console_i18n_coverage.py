"""The console must not leave English chrome inside a Chinese screen.

The dictionaries are typed (``TranslationDictionary``), so `tsc -b` already
fails when a key exists in only one locale. That check cannot see the other
half of the problem: a view that renders a *literal*, or a raw engine
identifier, never consults the dictionary at all — so both locales show the
same English word and the type system is satisfied.

These guards cover the literal class, which is invisible to the compiler. They
run in the Python suite because the frontend has no test runner of its own.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VIEWS = _REPO_ROOT / "web" / "src" / "views"


def _source(path: Path) -> str:
    if not path.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    return path.read_text(encoding="utf-8")


def test_no_view_upper_cases_a_status_into_the_ui() -> None:
    """``status.toUpperCase()`` bypasses the dictionary entirely.

    That is how the Chinese console ended up with "COMPLETED" in the header
    badge, the queue rows and the execution-state panel — and "NEEDS_HUMAN" on
    every review card — while the surrounding chrome was Chinese.
    """
    offenders = []
    for view in sorted(_VIEWS.rglob("*.tsx")):
        for number, line in enumerate(_source(view).splitlines(), 1):
            if line.lstrip().startswith(("//", "*", "/*")):
                continue
            # A fallback inside a label helper is fine — it only fires for a
            # status the dictionary does not know. What must not happen is a
            # status rendered directly, i.e. `{status.toUpperCase()}` in JSX.
            if re.search(r"\{\s*\w*[Ss]tatus\s*\.toUpperCase\(\)\s*\}", line):
                offenders.append(f"{view.name}:{number}: {line.strip()}")
    assert not offenders, "a status is upper-cased instead of localized: " + "; ".join(offenders)


def test_the_dictionaries_name_every_engine_state_a_view_shows() -> None:
    """A status the dictionary does not name falls back to a raw identifier.

    The fallback exists so a new engine state is still visible, but the states
    the console already renders must be translated in both locales.
    """
    zh = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "zh.ts")
    en = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "en.ts")

    # Job statuses (ubt.core.engine.job_queue.JobStatus).
    for status in ("submitted", "queued", "running", "completed", "failed", "cancelled"):
        assert status in en and status in zh, f"job status {status!r} is not localized"
    # Block statuses (ubt.core.ir.models.BlockStatus) shown on review cards.
    for status in ("mtqe_passed", "needs_human", "blocked_human", "repaired"):
        assert status in en and status in zh, f"block status {status!r} is not localized"
    # Doctor verdicts.
    for status in ("OK", "WARN", "FAIL", "SKIP"):
        assert status in en and status in zh, f"doctor status {status!r} is not localized"


def test_every_deliverable_key_has_a_label() -> None:
    """The label switch fell through to the backend's English ``label``.

    ``DELIVERABLE_LABELS`` names the files the API will serve; the console
    translates them by key. ``html`` and ``xliff`` were missing from the switch,
    so those two buttons were the only deliverables still in English on a
    Chinese screen — and adding a key to the backend would silently repeat it.
    """
    from ubt.api.routes.shared import DELIVERABLE_LABELS

    view = _source(_VIEWS / "quality-gate" / "QualityGateView.tsx")
    switch = view[view.index("function deliverableLabel") :]
    switch = switch[: switch.index("\n}")]

    missing = [key for key in DELIVERABLE_LABELS if f"case '{key}':" not in switch]
    assert not missing, (
        f"these deliverable keys fall back to the backend's English label: {missing}"
    )
