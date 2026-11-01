from __future__ import annotations

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.qe.heuristic import HeuristicQERunner

pytestmark = pytest.mark.fast


def _runner() -> HeuristicQERunner:
    return HeuristicQERunner(source_lang="en", target_lang="zh")


def test_runner_with_languages_returns_a_copy_not_self() -> None:
    shared = _runner()
    rebound = shared.with_languages("en", "ja")
    assert rebound is not shared
    assert (rebound.source_lang, rebound.target_lang) == ("en", "ja")
    assert (shared.source_lang, shared.target_lang) == ("en", "zh")


def test_runner_same_pair_returns_self() -> None:
    shared = _runner()
    assert shared.with_languages("en", "zh") is shared


def test_pipeline_binds_an_injected_runner_to_the_run_pair() -> None:
    config = UBTConfig(draft_model="mock-draft")
    runner = _runner()
    orchestrator = PipelineOrchestrator(config=config, qe_runner=runner)
    bound = orchestrator._runtime_qe_runner("en", "ja")
    assert isinstance(bound, HeuristicQERunner)
    assert (runner.source_lang, runner.target_lang) == ("en", "zh")
    assert (bound.source_lang, bound.target_lang) == ("en", "ja")
    same = orchestrator._runtime_qe_runner("en", "zh")
    assert same is runner


def test_pipeline_binds_the_runs_single_filter_into_the_runner() -> None:
    from ubt.core.qe.fast_pass import FastPassFilter

    config = UBTConfig(draft_model="mock-draft")
    runner = _runner()
    orchestrator = PipelineOrchestrator(config=config, qe_runner=runner)
    run_filter = FastPassFilter(source_lang="en", target_lang="zh")
    bound = orchestrator._runtime_qe_runner("en", "zh", fast_pass=run_filter)
    assert isinstance(bound, HeuristicQERunner), "a rebind must not degrade the class"
    assert bound is not runner, "a shared runner must not be mutated in place"
    assert bound.fast_pass is run_filter
    assert runner.fast_pass is not run_filter, "the shared runner keeps its own"
    assert bound.with_fast_pass(run_filter) is bound


def test_glossary_rebind_keeps_the_runs_filter_identity() -> None:
    from ubt.core.qe.fast_pass import FastPassFilter

    run_filter = FastPassFilter(source_lang="en", target_lang="zh")
    bound = _runner().with_fast_pass(run_filter)
    rebound = bound.with_glossary([{"source": "MASK", "target": "掩码"}])
    assert rebound.fast_pass is run_filter
    assert rebound.is_glossary_aware()
