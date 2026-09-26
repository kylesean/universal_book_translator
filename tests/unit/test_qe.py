"""Unit tests for FastPassFilter and QE scoring runners."""

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from ubt.core.config import packaged_comet_script
from ubt.core.ir.models import BlockType
from ubt.core.qe.comet_runner import MockQERunner, SubprocessQERunner
from ubt.core.qe.defect_taxonomy import (
    CRITICAL_DEFECT_MARKERS,
    ECHO_MARKER,
    NEAR_ECHO_MARKER,
    STRUCTURAL_DEFECT_MARKERS,
)
from ubt.core.qe.fast_pass import FastPassFilter

pytestmark = pytest.mark.fast


def test_fast_pass_filter_approves_clean_paragraph() -> None:
    """Validate that high quality, structurally sound translation is directly approved."""
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    src = "Psychological research shows that sleep deprivation significantly impairs cognitive ability."
    tgt = "心理学研究表明，睡眠不足会显著损害认知能力。"

    decision = fp.evaluate(src, tgt)
    assert decision.passed is True
    assert decision.target_ratio > 0.8
    assert "Flawless" in decision.reason


def test_fast_pass_filter_rejects_hallucinations_and_leaks() -> None:
    """Validate that repetitions and template XML residues are rejected."""
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    src = "The experiment proceeded without incident."

    # Repetition hallucination
    tgt_repeat = "实验顺利进行顺利进行顺利进行顺利进行顺利进行顺利进行。"
    dec_repeat = fp.evaluate(src, tgt_repeat)
    assert dec_repeat.passed is False
    assert "Repetitive loop" in dec_repeat.reason

    # Template artifact leakage
    tgt_leak = "<issues>Some notes</issues><translation>实验正常进行。</translation>"
    dec_leak = fp.evaluate(src, tgt_leak)
    assert dec_leak.passed is False
    assert "artifacts leaked" in dec_leak.reason


def test_fast_pass_filter_rejects_newline_separated_repetition_loop() -> None:
    """UBT the flat regex misses loops separated by newlines.

    ``(.{4,20}?)\\1{3,}`` can never match "sentence.\\n" x4 (the last repeat
    lacks the separator inside the group) and caps the unit at 20 chars, so
    the most common hallucination shape — the same sentence repeated line
    after line — must be caught by the line-level counter instead.
    """
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    src = "The experiment proceeded without incident. It was recorded in the log book."

    tgt_loop = "实验顺利进行。\n实验顺利进行。\n实验顺利进行。\n实验顺利进行。"
    dec_loop = fp.evaluate(src, tgt_loop)
    assert dec_loop.passed is False
    assert "Repetitive loop" in dec_loop.reason

    # Long English sentence repeated per line (unit beyond the flat 20-char cap)
    tgt_long = (
        "The committee reviewed the proposal carefully.\n"
        "The committee reviewed the proposal carefully.\n"
        "The committee reviewed the proposal carefully.\n"
        "The committee reviewed the proposal carefully."
    )
    dec_long = fp.evaluate(src, tgt_long)
    assert dec_long.passed is False
    assert "Repetitive loop" in dec_long.reason


def test_fast_pass_line_loop_allows_legitimate_repeats() -> None:
    """Non-contiguous repeats and numeric tables stay exempt (structural gate)."""
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    src = "Device parameters are listed below."

    # Sparse numeric table: identical zero rows carry no letters -> exempt
    numeric_table = "| 0 | 0 | 0 |\n| 0 | 0 | 0 |\n| 0 | 0 | 0 |\n| 0 | 0 | 0 |\n| 0 | 0 | 0 |"
    assert fp.validate_structural_invariants(src, numeric_table).passed is True

    # Repeated phrase inside a single legitimate line (no repeated lines) passes
    poetic = "远，远方的海面上，帆影点点，帆影点点，帆影点点。"
    assert fp.validate_structural_invariants(src, poetic).passed is True

    # Blank-line separated stanzas: the run resets between verses -> exempt
    verse = "轻轻的我走了。\n正如我轻轻的来。\n\n轻轻的我走了。\n正如我轻轻的来。"
    assert fp.validate_structural_invariants(src, verse).passed is True


def test_fast_pass_filter_rejects_broken_html_and_missing_numbers() -> None:
    """Validate that structural HTML and numeric errors fail fast-pass."""
    fp = FastPassFilter(source_lang="en", target_lang="zh")

    # Broken HTML
    dec_html = fp.evaluate(
        '<img src="cat.png" alt="A cat"/>',
        '<img src="cat.png" alt="一只"小猫""/>',
    )
    assert dec_html.passed is False
    assert "HTML delta failure" in dec_html.reason

    # Lost numbers
    dec_num = fp.evaluate(
        "Founded in 1998 with 500 members.",
        "该组织成立较早，拥有许多成员。",
    )
    assert dec_num.passed is False
    assert "Numeric fidelity failure" in dec_num.reason


def test_fast_pass_filter_ignores_urls_in_script_density() -> None:
    """URLs survive translation verbatim and must not dilute zh density."""
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    decision = fp.evaluate(
        "This e-book was created for MobileRead ( www.mobileread.com ).",
        "本书由 MobileRead (www.mobileread.com) 制作。",
    )
    assert decision.passed is True


@pytest.mark.asyncio
async def test_mock_qe_runner_scoring() -> None:
    """Validate MockQERunner batch scoring behavior."""
    runner = MockQERunner(default_score=0.88)
    pairs = [
        {"src": "Hello world", "mt": "你好世界"},
        {"src": "Bad chunk", "mt": "<issues>bad"},
        {"src": "Empty", "mt": ""},
    ]
    scores = await runner.score_pairs(pairs)
    assert len(scores) == 3
    assert scores[0] > 0.7
    assert scores[1] == 0.35
    assert scores[2] == 0.0


@pytest.mark.asyncio
async def test_subprocess_qe_runner_with_ipc_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate SubprocessQERunner executing the packaged comet scorer via JSON IPC."""
    import sys
    from pathlib import Path

    from ubt.core.qe.comet_runner import SubprocessQERunner

    monkeypatch.setenv("UBT_COMET_MOCK", "1")
    script_path = packaged_comet_script()
    assert script_path.exists(), "comet_score_ipc.py script must exist"

    runner = SubprocessQERunner(
        python_bin=Path(sys.executable),
        script_path=script_path,
        model_name="mock-model",
    )

    pairs = [
        {"src": "Hello world", "mt": "你好世界"},
        {"src": "Bad chunk", "mt": "<issues>leak</translation>"},
        {"src": "Empty chunk", "mt": ""},
    ]
    try:
        scores = await runner.score_pairs(pairs)
    finally:
        # The default path is the resident scorer; release its session (and
        # the armed idle reaper) so the loop can close cleanly.
        if runner._reaper is not None and not runner._reaper.done():
            runner._reaper.cancel()
        await runner._teardown_resident()
    assert len(scores) == 3
    assert scores[0] > 0.6
    assert scores[1] == 0.20
    assert scores[2] == 0.0
    # A mock/fallback batch must announce itself: discrete heuristic bands may
    # never be consumed as calibrated CometKiwi scores (rerank gate).
    assert runner.last_engine == "heuristic_fallback"
    assert runner.is_calibrated() is False


def test_subprocess_qe_parses_labelled_ipc_protocol(tmp_path: Path) -> None:
    """The {scores, engine} object protocol, with stdout chatter tolerated."""
    runner = SubprocessQERunner(
        python_bin=Path(sys.executable),
        script_path=Path("unused"),
        model_name="mock-model",
    )
    scores, engine = runner._parse_ipc_output('{"scores": [0.5, 0.75], "engine": "neural"}')
    assert scores == [0.5, 0.75]
    assert engine == "neural"
    # torch writes progress to stdout; parsing retries on the outermost object.
    scores, engine = runner._parse_ipc_output(
        'download: 100%\n{"scores": [0.42], "engine": "neural"}'
    )
    assert scores == [0.42]
    assert engine == "neural"


def test_subprocess_qe_unlabelled_output_is_not_calibrated() -> None:
    """A legacy bare-array reply cannot prove neural origin: fail closed.

    An external ``UBT_COMET_SCRIPT_PATH`` running the pre-labelling script falls
    back to the heuristic invisibly; treating that batch as calibrated is the
    exact masquerade the engine label exists to prevent.
    """
    from ubt.core.exceptions import MTQEEvaluationError

    runner = SubprocessQERunner(
        python_bin=Path(sys.executable),
        script_path=Path("unused"),
        model_name="mock-model",
    )
    scores, engine = runner._parse_ipc_output("[0.5, 0.9]")
    assert scores == [0.5, 0.9]
    assert engine == "heuristic_fallback"
    runner._last_engine = engine
    assert runner.is_calibrated() is False
    with pytest.raises(MTQEEvaluationError):
        runner._parse_ipc_output("no json anywhere")


@pytest.mark.asyncio
async def test_subprocess_qe_reports_a_missing_script_instead_of_unbound_proc() -> None:
    """A scorer that never starts must raise MTQEEvaluationError (review-2 X3).

    ``proc`` used to be bound inside the ``try``, so when
    ``create_subprocess_exec`` itself failed the handlers reaped an unbound name
    and replaced the real cause with ``UnboundLocalError`` — which the QE stage
    does not catch, so the failure was misrouted.
    """
    from ubt.core.exceptions import MTQEEvaluationError

    runner = SubprocessQERunner(
        python_bin=Path(sys.executable),
        script_path=Path("/nonexistent/ubt_comet_score_ipc.py"),
        model_name="mock-model",
    )
    with pytest.raises(MTQEEvaluationError):
        await runner.score_pairs([{"src": "hello", "mt": "你好"}])


def test_pipeline_orchestrator_qe_engine_selection() -> None:
    """Validate PipelineOrchestrator chooses appropriate QE runner based on config."""
    from ubt.core.config import UBTConfig
    from ubt.core.engine.pipeline import PipelineOrchestrator
    from ubt.core.qe.comet_runner import HeuristicQERunner, SubprocessQERunner

    # Default heuristic
    cfg_heuristic = UBTConfig(qe_engine="heuristic")
    orch_heuristic = PipelineOrchestrator(config=cfg_heuristic)
    assert isinstance(orch_heuristic.qe_runner, HeuristicQERunner)

    # Comet engine config
    cfg_comet = UBTConfig(qe_engine="comet")
    orch_comet = PipelineOrchestrator(config=cfg_comet)
    assert isinstance(orch_comet.qe_runner, SubprocessQERunner)


# ---------------------------------------------------------------------------
# The heuristic "score" is a discrete defect-class proxy — the
# published legend must cover every value the runner can actually emit.
# ---------------------------------------------------------------------------


def test_qe_defect_class_legend_covers_all_emittable_values() -> None:
    import asyncio

    from ubt.core.qe.comet_runner import (
        QE_DEFECT_CLASS_LEGEND,
        QE_SCORE_GLOSSARY_VIOLATION,
        HeuristicQERunner,
    )

    legend_values = {v for v, _ in QE_DEFECT_CLASS_LEGEND}
    # Pass value + empty + empty-source fabrication + untranslated residue.
    runner = HeuristicQERunner()
    emittable = {
        0.92,  # fast pass
        0.0,  # empty target
        0.15,  # empty source / untranslated residue
    }
    emittable.add(
        asyncio.run(runner.score_pairs([{"src": "x", "mt": "<issues>leak</translation>"}]))[0]
    )
    # Every rejection-reason mapping must land on a documented band.
    for reason in (
        "Empty target text",
        "Prompt template XML artifacts leaked into target text",
        "Repetitive loop hallucination detected",
        "HTML delta failure: mismatch",
        "Math span mismatch: source carries 1",
        "Numeric fidelity failure: missing 42",
        "Target text suspiciously truncated (ratio=0.10)",
        "Target text suspiciously inflated (ratio=9.99)",
        "Insufficient zh script density (ratio=0.10)",
    ):
        emittable.add(HeuristicQERunner.score_from_decision_reason(reason))
    # F3: the terminology band is emittable too.
    emittable.add(QE_SCORE_GLOSSARY_VIOLATION)

    undocumented = emittable - legend_values
    assert not undocumented, (
        f"heuristic emits values absent from QE_DEFECT_CLASS_LEGEND: {undocumented}"
    )


def test_format_only_matches_production_reason_strings() -> None:
    """The cheap-repair path must key off the reasons FastPass actually emits."""
    from ubt.core.qe.defect_taxonomy import is_format_only

    assert is_format_only(["HTML delta failure: mismatch"])
    assert is_format_only(["Target text suspiciously truncated (ratio=0.10)"])
    assert is_format_only(["Target text suspiciously inflated (ratio=2.50)"])
    assert not is_format_only(["HTML delta failure: mismatch", "Added reference(s): ['3.5']"])
    assert not is_format_only([])
    assert not is_format_only(["math_token_corrupt missing=['x'] mismatched=[] mutated=[]"])


# ---------------------------------------------------------------------------
# F3 (review): no terminology signal existed anywhere in the QE path. A fluent
# translation using the WRONG enforced term passed every gate and scored 0.92.
# ---------------------------------------------------------------------------

_GLOSSARY_SRC = "The subthreshold swing degrades as the channel length shrinks to 20 nm."
_GLOSSARY_GOOD = "当沟道长度缩短至 20 nm 时，亚阈值摆幅会退化。"
_GLOSSARY_BAD = "当沟道长度缩短至 20 nm 时，短沟道效应会加剧。"


def test_heuristic_scores_enforced_term_violation_below_pass() -> None:
    import asyncio

    from ubt.core.qe.comet_runner import (
        GLOSSARY_VIOLATION_MARKER,
        QE_SCORE_GLOSSARY_VIOLATION,
        QE_SCORE_PASS,
        HeuristicQERunner,
    )

    glossary = [{"source": "subthreshold swing", "translation": "亚阈值摆幅"}]
    pairs = [
        {"src": _GLOSSARY_SRC, "mt": _GLOSSARY_GOOD},
        {"src": _GLOSSARY_SRC, "mt": _GLOSSARY_BAD},
    ]

    # Without a glossary the altered term is structurally clean — exactly the
    # Hole the review found (0.92, never judged).
    blind = asyncio.run(HeuristicQERunner().score_pairs(pairs))
    assert blind == [QE_SCORE_PASS, QE_SCORE_PASS]

    scores = asyncio.run(HeuristicQERunner(glossary=glossary).score_pairs(pairs))
    assert scores[0] == QE_SCORE_PASS  # correct rendering unaffected
    assert scores[1] == QE_SCORE_GLOSSARY_VIOLATION
    assert scores[1] < 0.75  # default auto-pass band starts at qe_threshold

    runner = HeuristicQERunner(glossary=glossary)
    flag = runner.glossary_violation(_GLOSSARY_SRC, _GLOSSARY_BAD)
    assert flag is not None and GLOSSARY_VIOLATION_MARKER in flag
    assert runner.glossary_violation(_GLOSSARY_SRC, _GLOSSARY_GOOD) is None


def test_with_glossary_binds_the_terminology_signal() -> None:
    """The orchestrator binds the run glossary after the bible stage."""
    import asyncio

    from ubt.core.qe.comet_runner import (
        QE_SCORE_GLOSSARY_VIOLATION,
        HeuristicQERunner,
    )

    blind = HeuristicQERunner(target_lang="zh", source_lang="en")
    assert blind.with_glossary(None) is blind  # nothing to bind -> no new object

    bound = blind.with_glossary([{"source": "subthreshold swing", "translation": "亚阈值摆幅"}])
    assert bound is not blind
    assert (bound.target_lang, bound.source_lang) == ("zh", "en")
    scores = asyncio.run(bound.score_pairs([{"src": _GLOSSARY_SRC, "mt": _GLOSSARY_BAD}]))
    assert scores[0] == QE_SCORE_GLOSSARY_VIOLATION
    # The language rebind must not drop the glossary binding.
    rebound = bound.with_languages("ja", "zh")
    assert rebound is not bound
    assert rebound.glossary_violation(_GLOSSARY_SRC, _GLOSSARY_BAD) is not None


def test_glossary_violation_band_is_structural_and_caps_the_score() -> None:
    from ubt.core.qe.comet_runner import (
        GLOSSARY_VIOLATION_MARKER,
        QE_DEFECT_CLASS_LEGEND,
        QE_SCORE_FABRICATED,
        QE_SCORE_GLOSSARY_VIOLATION,
        QE_SCORE_HTML_DELTA,
        QE_SCORE_REPETITION,
        HeuristicQERunner,
    )
    from ubt.core.qe.defect_taxonomy import has_structural_defect

    assert QE_SCORE_GLOSSARY_VIOLATION in {value for value, _ in QE_DEFECT_CLASS_LEGEND}
    flag = f"{GLOSSARY_VIOLATION_MARKER}: Glossary terms drifted: ['x -> y']"
    # Reuses the existing quality-gate mechanism: a registered structural
    # marker keeps the block out of auto-pass whatever the runner returned.
    assert has_structural_defect([flag])
    assert HeuristicQERunner.score_from_flags([flag]) == QE_SCORE_GLOSSARY_VIOLATION
    # Every other condition keeps its documented value.
    assert (
        HeuristicQERunner.score_from_flags(["HTML delta failure: mismatch"]) == QE_SCORE_HTML_DELTA
    )
    assert (
        HeuristicQERunner.score_from_flags(["Repetitive loop hallucination detected", flag])
        == QE_SCORE_REPETITION
    )
    assert (
        HeuristicQERunner.score_from_flags(["Added reference(s): ['3.5']", flag])
        == QE_SCORE_FABRICATED
    )


def test_fast_pass_markdown_table_immunity() -> None:
    """Verify that FastPassFilter does not falsely detect repetition loops on Markdown tables."""
    fp = FastPassFilter()

    # Standard Markdown table separator with 4+ repeated |---| cells
    src_table = (
        "| Metric | Precision | Recall | F1 |\n|---|---|---|---|\n| Score | 0.95 | 0.93 | 0.94 |"
    )
    tgt_table = (
        "| 指标 | 精确率 | 召回率 | F1值 |\n|---|---|---|---|\n| 得分 | 0.95 | 0.93 | 0.94 |"
    )

    decision = fp.validate_structural_invariants(src_table, tgt_table)
    assert decision.passed is True, f"Erroneously failed: {decision.reason}"

    # Real textual loop hallucination should still be caught
    tgt_loop = "这个模型非常好 这个模型非常好 这个模型非常好 这个模型非常好 这个模型非常好"
    decision_loop = fp.validate_structural_invariants(src_table, tgt_loop)
    assert decision_loop.passed is False
    assert "Repetitive loop hallucination" in decision_loop.reason


@pytest.mark.asyncio
async def test_cancelled_score_pairs_reaps_the_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled QE call must not leave the scorer subprocess running.

    ``CancelledError`` derives from BaseException, so the ``except Exception``
    handler never saw it: the child outlived the cancelled task and kept its
    model weights resident.
    """
    script = tmp_path / "sleeper.py"
    script.write_text("import sys, time\nsys.stdin.buffer.read()\ntime.sleep(120)\n")

    spawned: list[asyncio.subprocess.Process] = []
    real_create = asyncio.create_subprocess_exec

    async def _capture_create(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_create(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _capture_create)

    runner = SubprocessQERunner(
        python_bin=Path(sys.executable), script_path=script, timeout_seconds=300
    )
    task = asyncio.create_task(runner.score_pairs([{"src": "hi", "mt": "你好"}]))
    await asyncio.sleep(1.0)
    assert spawned, "the scorer subprocess should have been started"

    proc = spawned[0]
    assert proc.returncode is None, "subprocess should still be running"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.5)
    assert proc.returncode is not None, "cancelled QE call leaked a live subprocess"


# ---------------------------------------------------------------------------
# Resident (--serve) scorer: one model load across batches, with per-call
# fallback as the rollback path.
# ---------------------------------------------------------------------------


def _resident_runner(tmp_path: Path, script: Path | None = None) -> SubprocessQERunner:
    ipc = script or packaged_comet_script()
    return SubprocessQERunner(
        python_bin=Path(sys.executable),
        script_path=ipc,
        model_name="mock-model",
        timeout_seconds=30,
        idle_reap_seconds=600,
    )


async def _cleanup(runner: SubprocessQERunner) -> None:
    if runner._reaper is not None and not runner._reaper.done():
        runner._reaper.cancel()
    await runner._teardown_resident()


@pytest.mark.asyncio
async def test_resident_scorer_is_reused_across_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_COMET_MOCK", "1")
    runner = _resident_runner(Path())
    try:
        first = await runner.score_pairs([{"src": "Hello world", "mt": "你好世界"}])
        proc = runner._resident
        assert proc is not None, "resident scorer must stay alive between batches"
        second = await runner.score_pairs([{"src": "Bad", "mt": "<issues>x</translation>"}])
        assert runner._resident is proc, "second batch must reuse the same process"
        assert first[0] > 0.6
        assert second[0] == 0.20
        assert runner.last_engine == "heuristic_fallback"
    finally:
        await _cleanup(runner)


@pytest.mark.asyncio
async def test_resident_scorer_recovers_transparently_after_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UBT_COMET_MOCK", "1")
    runner = _resident_runner(Path())
    try:
        await runner.score_pairs([{"src": "Hello world", "mt": "你好世界"}])
        victim = runner._resident
        assert victim is not None
        # Simulate a crashed scorer the way the runner sees one: kill it and
        # let its own teardown reap the transport, then the next request
        # finds no session and spawns a fresh one.
        victim.kill()
        await runner._teardown_resident(victim)
        scores = await runner.score_pairs([{"src": "Hello again", "mt": "你好again世界"}])
        assert scores and runner._resident is not None and runner._resident.pid != victim.pid
        assert runner._resident_broken is False
    finally:
        await _cleanup(runner)


@pytest.mark.asyncio
async def test_resident_path_disabled_by_rollback_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UBT_COMET_RESIDENT=0 must reproduce the historical per-call behaviour."""
    monkeypatch.setenv("UBT_COMET_MOCK", "1")
    monkeypatch.setenv("UBT_COMET_RESIDENT", "0")
    runner = _resident_runner(Path())
    scores = await runner.score_pairs([{"src": "Hello world", "mt": "你好世界"}])
    assert scores[0] > 0.6
    assert runner._resident is None


def test_subprocess_runner_is_not_calibrated_before_it_scores() -> None:
    """'Not scored yet' must not read as calibrated.

    ``RepairLoop._rerank_enabled`` is consulted before the first ``score_pairs``
    of a run; treating the pre-score state as calibrated let that first rerank
    rank candidates on scores the gate had not measured.
    """
    runner = _resident_runner(Path())
    assert runner.last_engine is None
    assert runner.is_calibrated() is False


def test_reset_residency_re_enables_the_resident_path() -> None:
    """The broken latch is per-run, but the API shares one runner across jobs.

    Without a reset, one job's protocol desync forced every later job in the
    process onto the per-call path — a full model load per QE batch.
    """
    runner = _resident_runner(Path())
    runner._resident_broken = True
    assert runner._resident_enabled() is False
    runner.reset_residency()
    assert runner._resident_broken is False
    assert runner._resident_enabled() is True


@pytest.mark.asyncio
async def test_resident_scorer_discards_stderr_so_it_cannot_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing drains the resident scorer's stderr.

    A piped child that fills the ~64 KiB buffer (model download progress, torch
    warnings) blocks in ``write()`` and stops answering, stalling the parent
    until the request timeout.
    """
    monkeypatch.setenv("UBT_COMET_MOCK", "1")
    runner = _resident_runner(Path())
    try:
        await runner.score_pairs([{"src": "Hello world", "mt": "你好世界"}])
        proc = runner._resident
        assert proc is not None
        assert proc.stderr is None, "resident scorer stderr must not be an undrained pipe"
    finally:
        await _cleanup(runner)


@pytest.mark.asyncio
async def test_request_error_envelope_does_not_tear_down_the_session(tmp_path: Path) -> None:
    """A per-request error means the session is healthy; only this request failed.

    Tearing the process down discarded the multi-GB loaded checkpoint, so the
    next batch paid a full model reload.
    """
    from ubt.core.exceptions import MTQEEvaluationError

    stub = tmp_path / "error_scorer.py"
    stub.write_text(
        """
import json, sys
if "--serve" in sys.argv:
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        request = json.loads(line)
        if request.get("cmd") == "quit":
            break
        reply = {"id": request["id"], "error": "model.predict failed"}
        sys.stdout.write(json.dumps(reply) + "\\n")
        sys.stdout.flush()
else:
    pairs = json.loads(sys.stdin.read())
    sys.stdout.write(json.dumps({"scores": [0.5] * len(pairs), "engine": "neural"}))
"""
    )
    runner = _resident_runner(tmp_path, script=stub)
    try:
        with pytest.raises(MTQEEvaluationError):
            await runner.score_pairs([{"src": "a", "mt": "b"}])
        assert runner._resident is not None, "a per-request error must not kill the session"
        assert runner._resident_broken is False
    finally:
        await _cleanup(runner)


@pytest.mark.asyncio
async def test_protocol_desync_falls_back_to_per_call(tmp_path: Path) -> None:
    """A scorer that answers with the wrong request id is replaced, then given up on.

    The per-call fallback stays functional, so a half-speaking external
    ``UBT_COMET_SCRIPT_PATH`` degrades throughput, never correctness.
    """
    stub = tmp_path / "desync_scorer.py"
    stub.write_text(
        """
import json, sys

if "--serve" in sys.argv:
    request = json.loads(sys.stdin.readline())
    reply = {"id": -999, "scores": [0.5], "engine": "neural"}
    sys.stdout.write(json.dumps(reply) + "\\n")
    sys.stdout.flush()
    sys.stdin.readline()
else:
    pairs = json.loads(sys.stdin.read())
    sys.stdout.write(json.dumps({"scores": [0.5] * len(pairs), "engine": "neural"}))
""",
        encoding="utf-8",
    )
    runner = _resident_runner(tmp_path, script=stub)
    scores = await runner.score_pairs([{"src": "a", "mt": "b"}, {"src": "c", "mt": "d"}])
    assert scores == [0.5, 0.5]
    assert runner._resident_broken is True
    await _cleanup(runner)


def test_comet_runner_teardown_has_newline() -> None:
    from unittest.mock import AsyncMock, MagicMock

    runner = SubprocessQERunner(python_bin=Path("python3"), script_path=packaged_comet_script())
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()

    asyncio.run(runner._teardown_resident(mock_proc))
    mock_proc.stdin.write.assert_called_once()
    written_bytes: bytes = mock_proc.stdin.write.call_args[0][0]
    assert written_bytes.endswith(b"\n"), f"Expected trailing newline, got {written_bytes!r}"


def test_term_shape_dialogue_sentence_count() -> None:
    from ubt.core.qe.term_shape import count_sentences

    dialogue = 'He said, "It is done." The next day, he left.'
    cnt = count_sentences(dialogue)
    assert cnt == 2, f"Expected 2 sentences for dialogue, got {cnt}"


@pytest.mark.fast
def test_gate_3b_accepts_redelimited_and_text_translated_math() -> None:
    from ubt.core.qe.fast_pass import FastPassFilter

    fp = FastPassFilter(source_lang="en", target_lang="zh")
    # 1. Delimiter redelimited \(...\) to $...$
    d1 = fp.evaluate("Given \\(x^2\\) is positive.", "已知 $x^2$ 为正。")
    assert d1.passed, f"Redelimited math rejected: {d1.reason}"

    # 2. Formula with \\text{} translated (skeleton_holds)
    d2 = fp.evaluate("Formula $\\text{cost} = x + 1$ holds.", "公式 $\\text{成本} = x + 1$ 成立。")
    assert d2.passed, f"Text-translated formula rejected: {d2.reason}"


@pytest.mark.fast
def test_fast_pass_short_heading_length_ratio_exemption() -> None:
    """Short headings (e.g. Introduction -> 引言, ratio 0.17) must not fail length ratio."""
    fp = FastPassFilter(source_lang="en", target_lang="zh")

    # 'Introduction' (12 chars) -> '引言' (2 chars): ratio 2/12 = 0.167 < 0.20
    d1 = fp.evaluate("Introduction", "引言")
    assert d1.passed, f"Short heading rejected: {d1.reason}"

    # 'Acknowledgements' (16 chars) -> '致谢' (2 chars): ratio 2/16 = 0.125 < 0.20
    d2 = fp.evaluate("Acknowledgements", "致谢")
    assert d2.passed, f"Short heading rejected: {d2.reason}"


@pytest.mark.fast
def test_fast_pass_proper_nouns_and_tech_terms_script_density() -> None:
    """Title-cased technical terms (Linux, Kubernetes, Prometheus) must not cause script density rejection."""
    fp = FastPassFilter(source_lang="en", target_lang="zh")

    # In "在 Linux 上使用 Prometheus 和 Grafana 配置 Kubernetes。", Latin proper nouns dominate char count
    src = "Configure Kubernetes with Prometheus and Grafana on Linux."
    tgt = "在 Linux 上使用 Prometheus 和 Grafana 配置 Kubernetes。"
    d = fp.evaluate(src, tgt)
    assert d.passed, f"Tech terms translation rejected: {d.reason}"


@pytest.mark.fast
def test_repeated_source_line_is_not_a_hallucination_loop() -> None:
    from ubt.core.qe.fast_pass import FastPassFilter

    fp = FastPassFilter()
    src = "The road goes ever on and on.\n" * 4
    tgt = "路一直向前延伸。\n" * 4
    assert fp.evaluate(src, tgt).passed

    # A target that loops while the source does not is still caught.
    decision = fp.evaluate(
        "Alpha line one.\nBeta line two.\nGamma line three.\nDelta line four.",
        "Loop line here.\n" * 4,
    )
    assert not decision.passed
    assert "Repetitive" in decision.reason


_a0920_SOURCE_PARAGRAPH = (
    "The device operates in inversion when the gate exceeds the threshold "
    "voltage across the oxide layer here."
)


def test_both_echo_phrasings_are_registered_in_every_defect_table() -> None:
    """The reason strings and the tables must not be able to drift apart."""
    from ubt.core.qe.fast_pass import FastPassFilter as _FP

    for marker in (ECHO_MARKER, NEAR_ECHO_MARKER):
        assert marker in STRUCTURAL_DEFECT_MARKERS, marker
        assert marker in CRITICAL_DEFECT_MARKERS, marker

    exact = _FP(target_lang="zh").evaluate(
        _a0920_SOURCE_PARAGRAPH, _a0920_SOURCE_PARAGRAPH, block_type=BlockType.NARRATIVE
    )
    assert exact.reason.startswith(ECHO_MARKER)
