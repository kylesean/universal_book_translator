"""Shared QE runners must not leak one job's language pair into another.

Two halves (2026-10-14 review, W2-2):

1. ``with_languages`` used to mutate ``self`` in place — the API holds ONE
   runner for concurrent jobs, so job B's rebind leaked into job A's in-flight
   scoring. It now returns a rebound copy.
2. ``PipelineOrchestrator._runtime_qe_runner`` passed an INJECTED runner
   through unchanged, so an injected runner scored every pair with its
   construction-time en→zh thresholds. It now runs every runner through
   ``with_languages`` (same-instance when the pair already matches, so the
   shared path stays cheap).

The concrete runner under test is the heuristic one; the neural subprocess
runner these tests originally used was retired for license compliance.
"""

from __future__ import annotations

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.qe.heuristic import HeuristicQERunner

pytestmark = pytest.mark.fast


def _runner() -> HeuristicQERunner:
    return HeuristicQERunner(source_lang="en", target_lang="zh")


def test_runner_with_languages_returns_a_copy_not_self() -> None:
    shared = _runner()  # construction-time en -> zh
    rebound = shared.with_languages("en", "ja")
    assert rebound is not shared
    assert (rebound.source_lang, rebound.target_lang) == ("en", "ja")
    # The shared instance keeps its own binding — no leak.
    assert (shared.source_lang, shared.target_lang) == ("en", "zh")


def test_runner_same_pair_returns_self() -> None:
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
    # The copy keeps the concrete class (a rebind must not degrade it to the
    # base runner) and carries the requested pair.
    assert isinstance(bound, HeuristicQERunner)
    # The shared instance is untouched; the run gets a bound copy.
    assert (runner.source_lang, runner.target_lang) == ("en", "zh")
    assert (bound.source_lang, bound.target_lang) == ("en", "ja")
    # Same pair as construction: the shared instance itself (cheap path).
    same = orchestrator._runtime_qe_runner("en", "zh")
    assert same is runner


def test_pipeline_binds_the_runs_single_filter_into_the_runner() -> None:
    """The draft stage's filter and the scorer's filter are one object.

    The runner used to build a private FastPassFilter that had to be taught the
    glossary and the document context by hand; three measured
    gate-versus-score contradictions came from that split. The pipeline hands
    the run's filter to the runner, so a binding the draft stage makes on
    ``services.fast_pass`` is, by identity, what ``score_pairs`` scores with.
    """
    from ubt.core.qe.fast_pass import FastPassFilter

    config = UBTConfig(draft_model="mock-draft")
    runner = _runner()  # shared: the API holds one runner per process
    orchestrator = PipelineOrchestrator(config=config, qe_runner=runner)
    run_filter = FastPassFilter(source_lang="en", target_lang="zh")
    bound = orchestrator._runtime_qe_runner("en", "zh", fast_pass=run_filter)
    assert isinstance(bound, HeuristicQERunner), "a rebind must not degrade the class"
    assert bound is not runner, "a shared runner must not be mutated in place"
    assert bound.fast_pass is run_filter
    assert runner.fast_pass is not run_filter, "the shared runner keeps its own"
    # Rebinding the same filter is idempotent.
    assert bound.with_fast_pass(run_filter) is bound


def test_glossary_rebind_keeps_the_runs_filter_identity() -> None:
    """The quality gate rebinds the glossary after the filter was handed over.

    The rebind must not silently swap in a fresh filter (losing the document
    context the draft stage bound on the run's object) — the clone carries the
    handed-over filter by identity and teaches it the glossary in place.
    """
    from ubt.core.qe.fast_pass import FastPassFilter

    run_filter = FastPassFilter(source_lang="en", target_lang="zh")
    bound = _runner().with_fast_pass(run_filter)
    rebound = bound.with_glossary([{"source": "MASK", "target": "掩码"}])
    assert rebound.fast_pass is run_filter
    assert rebound.is_glossary_aware()
