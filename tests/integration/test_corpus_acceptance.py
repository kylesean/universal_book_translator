from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_REPO = Path(__file__).resolve().parents[2]

_CASES: tuple[tuple[str, tuple[str, ...]], ...] = (("shadow_reader", ("--min-coverage", "0.97")),)


def _skip_without_corpus() -> None:
    documents = sorted((_REPO / "corpus" / "documents").glob("*.pdf"))
    if not documents:
        pytest.skip(
            "corpus/documents is not present (gitignored by design), so the "
            "acceptance harness measured nothing. Place the benchmark PDFs named "
            "by corpus/cases.json there and run locally; CI cannot cover this gate.",
            allow_module_level=False,
        )


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
