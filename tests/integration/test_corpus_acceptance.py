"""The corpus acceptance harnesses, wired into the ``slow`` tier.

``scripts/shadow_*.py`` are CLI acceptance harnesses the ``fast`` tier cannot
carry: they read the real corpus PDFs (gitignored, see ``corpus/README.md``)
and run parser/render subprocesses. The strategy doc parks them in the
``slow``/``legacy_drift`` tiers; this module is that parking spot -- each
harness runs in a subprocess (principle 5: a harness that wedges must not
wedge the session) and must exit 0. The final test runs the corpus gate
itself, ``ubt verify --run``: the real pipeline over every document, each
delivery contract reconciled against its ``cases.json`` thresholds.

Everything cheap enough to run on synthetic fixtures was already converted
into ``tests/unit`` tests; what remains here needs the real documents.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_REPO = Path(__file__).resolve().parents[2]

#: (harness, extra args). ``shadow_reader``'s synthetic-free gate defaults to a
#: 0.98 character-coverage floor; the scanned-heavy book chapter tops out at
#: 0.973 while still round-tripping losslessly, so the floor is the documented
#: knob (a fraction), not a skipped contract.
_CASES: tuple[tuple[str, tuple[str, ...]], ...] = (("shadow_reader", ("--min-coverage", "0.97")),)


def _skip_without_corpus() -> None:
    documents = sorted((_REPO / "corpus" / "documents").glob("*.pdf"))
    if not documents:
        pytest.skip("corpus documents are not present (gitignored); see corpus/README.md")


@pytest.mark.parametrize(("script", "extra"), _CASES, ids=[c[0] for c in _CASES])
def test_the_corpus_acceptance_harness_passes(script: str, extra: tuple[str, ...]) -> None:
    _skip_without_corpus()
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(_REPO / "scripts" / f"{script}.py"), "--corpus", "corpus", *extra],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=540,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{script} exited {proc.returncode}\n"
            f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-1000:]}"
        )


@pytest.mark.timeout(900)
def test_the_ubt_verify_corpus_gate_passes() -> None:
    """``ubt verify --run``: the corpus README's full gate, in the suite.

    The seven harnesses above check components; this one runs the *real
    pipeline* over every corpus document (dry-run provider, no API key) and
    reconciles each delivery contract against its ``cases.json`` thresholds —
    the ground truth the corpus was created for.
    """
    _skip_without_corpus()
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "ubt",
            "verify",
            "--corpus",
            "corpus",
            "--run",
            "--require-all",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"ubt verify --run exited {proc.returncode}\n"
            f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-1000:]}"
        )
