"""Shared QE runners must not leak one job's language pair into another.

Two halves (2026-10-14 review, W2-2):

1. ``SubprocessQERunner.with_languages`` used to mutate ``self`` in place —
   the API holds ONE runner for concurrent jobs, so job B's rebind leaked
   into job A's in-flight scoring. It now returns a rebound copy.
2. ``PipelineOrchestrator._runtime_qe_runner`` passed an INJECTED runner
   through unchanged, so an injected subprocess runner scored every pair
   with its construction-time en→zh heuristic thresholds. It now runs
   every runner through ``with_languages`` (same-instance when the pair
   already matches, so the shared path stays cheap).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.qe.comet_runner import SubprocessQERunner

pytestmark = pytest.mark.fast


def _runner() -> SubprocessQERunner:
    return SubprocessQERunner(
        python_bin=Path("/usr/bin/false"), script_path=Path("/nonexistent/scorer.py")
    )


def test_subprocess_runner_with_languages_returns_a_copy_not_self() -> None:
    shared = _runner()  # construction-time en -> zh
    rebound = shared.with_languages("en", "ja")
    assert rebound is not shared
    assert (rebound._source_lang, rebound._target_lang) == ("en", "ja")
    # The shared instance keeps its own binding — no leak.
    assert (shared._source_lang, shared._target_lang) == ("en", "zh")


def test_subprocess_runner_same_pair_returns_self() -> None:
    shared = _runner()  # en -> zh
    assert shared.with_languages("en", "zh") is shared


def test_pipeline_binds_an_injected_runner_to_the_run_pair() -> None:
    """An injected runner reaches with_languages instead of passing through.

    The old explicit-runner branch returned the runner untouched, so its
    construction-time en→zh thresholds scored every other pair.
    """
    config = UBTConfig(draft_model="mock-draft")
    runner = _runner()
    orchestrator = PipelineOrchestrator(config=config, qe_runner=runner)
    bound = orchestrator._runtime_qe_runner("en", "ja")
    # The shared instance is untouched; the run gets a bound copy.
    assert (runner._source_lang, runner._target_lang) == ("en", "zh")
    assert (bound._source_lang, bound._target_lang) == ("en", "ja")
    # Same pair as construction: the shared instance itself (cheap path).
    same = orchestrator._runtime_qe_runner("en", "zh")
    assert same is runner
