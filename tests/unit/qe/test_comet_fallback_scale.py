"""The COMET subprocess fallback must not drift the quality gate's scale.

When the scorer's own environment lacks torch/comet it answers with a
length-ratio heuristic (0.1-0.85) -- a different scale than the pipeline's
12-band ``HeuristicQERunner``. The runner re-scores such a batch in-process so
the threshold means the same thing whichever path scored it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ubt.core.qe.comet_runner import SubprocessQERunner

pytestmark = pytest.mark.fast


def _runner() -> SubprocessQERunner:
    return SubprocessQERunner(
        python_bin=Path("/usr/bin/false"), script_path=Path("/nonexistent/scorer.py")
    )


def test_fallback_scores_are_rescored_on_the_heuristic_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()

    async def fake_backend(pairs: list[dict[str, str]]) -> list[float]:
        runner._last_engine = "heuristic_fallback"
        return [0.85 for _ in pairs]  # the subprocess's own length-ratio number

    monkeypatch.setattr(runner, "_score_via_backend", fake_backend)

    # An untranslated echo: the parent heuristic must flag it, not echo 0.85.
    scores = asyncio.run(runner.score_pairs([{"src": "Hello world", "mt": "Hello world"}]))
    assert scores[0] != 0.85
    assert scores[0] < 0.5
    assert runner.is_calibrated() is False


def test_neural_scores_pass_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()

    async def fake_backend(pairs: list[dict[str, str]]) -> list[float]:
        runner._last_engine = "neural"
        return [0.9 for _ in pairs]

    monkeypatch.setattr(runner, "_score_via_backend", fake_backend)

    scores = asyncio.run(runner.score_pairs([{"src": "a", "mt": "b"}]))
    assert scores == [0.9]
    assert runner.is_calibrated() is True


def test_fallback_short_circuits_subsequent_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    backend_calls = 0

    async def fake_backend(pairs: list[dict[str, str]]) -> list[float]:
        nonlocal backend_calls
        backend_calls += 1
        runner._last_engine = "heuristic_fallback"
        return [0.85 for _ in pairs]

    monkeypatch.setattr(runner, "_score_via_backend", fake_backend)

    # First batch detects non-neural and falls back
    scores1 = asyncio.run(runner.score_pairs([{"src": "Hello world", "mt": "Hello world"}]))
    assert backend_calls == 1
    assert scores1[0] < 0.5
    assert runner._neural_unavailable is True

    # Second batch short-circuits directly to heuristic without invoking backend
    scores2 = asyncio.run(runner.score_pairs([{"src": "Hello world", "mt": "Hello world"}]))
    assert backend_calls == 1
    assert scores2[0] < 0.5

    # Resetting residency resets neural unavailability
    runner.reset_residency()
    assert runner._neural_unavailable is False
    scores3 = asyncio.run(runner.score_pairs([{"src": "Hello world", "mt": "Hello world"}]))
    assert backend_calls == 2
    assert scores3[0] < 0.5
