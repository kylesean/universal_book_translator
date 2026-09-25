"""Final-round review fixes (2026-09-25, second pass).

Twelve confirmed defects found by the parallel-subagent review and re-verified
by hand before any production change. Each test below is written first (RED)
and pins the *observable* failure, not the implementation:

1.  ``--json`` stdout corrupted by the rigid-engine advisory warning.
2.  ``_assert_paintable`` fail-open: an all-rotated (geometry-less) PDF ships
    the source as the "translation".
3.  ``_is_bib_entry`` venue+pages tier fires on body prose / captions.
4.  Latin ``split_clauses`` consumes separator spaces, so the painted text
    loses them for en->fr/de/es.
5.  Docling-spaced three-level references (``3 . 4 . 1``) tokenize as ``3.4``,
    flagging a correct translation as fabricated.
6.  Superscript powers fold into the base number (``10^2`` -> ``10²`` reads as
    the phantom token ``102``) and report a lost number.
7.  Ordered-list numbers after an intro colon are pruned as page numbers.
8.  The line-repetition loop detector ignores the source, so a faithfully
    repeated source line is quarantined as a hallucination.
9.  A failed checkpoint batch is retried *after* newer updates for the same
    block, so the stale text/status wins in the ledger.
10. ``TieredQERunner`` cannot bind the glossary to its heuristic leg, so a
    repaired term never clears the violation marker.
11. ``_SUPERSCRIPT_DIGITS`` omits ``⁷`` (U+2077), missing bylines that use it.
12. ``metrics show/compare --json`` print a non-JSON error to stdout.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import threading

import pytest

pytestmark = pytest.mark.fast


# --- 1. --json stdout purity with the rigid engine --------------------------


def test_rigid_engine_warning_keeps_json_stdout_pure(tmp_path) -> None:
    """The advisory warning must not precede the JSON object on stdout."""
    book = tmp_path / "probe.md"
    book.write_text("# Chapter 1\n\nA short technical note.\n", encoding="utf-8")
    env = {
        **os.environ,
        "UBT_RENDER_ENGINE": "rigid",
        "UBT_OUTPUT_DIR": str(tmp_path / "out"),
    }
    # Under Profile-Aware adaptive defaults, an unset --dual-mode adapts cleanly
    # to 'monolingual' for rigid engines without a spurious warning. Pass
    # '--dual-mode inline' explicitly to trigger the downgrade advisory and verify
    # that it routes to stderr without corrupting the JSON payload on stdout.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ubt",
            "translate",
            str(book),
            "--dual-mode",
            "inline",
            "--dry-run",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tmp_path,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)  # exactly one JSON object, no preamble
    assert payload["status"] == "completed"
    # The warning is still delivered -- just not on the machine-readable stream.
    assert "monolingual-only" in proc.stderr


# --- 2. rigid render fails closed on a geometry-less PDF --------------------


def test_assert_paintable_rejects_geometry_skips() -> None:
    """No paintable zone because no page geometry was decoded must abort."""
    from ubt.adapters.pdf.rigid.typesetter import RigidReport, _assert_paintable
    from ubt.core.exceptions import DocumentParseError

    for reason in ("no_page_height", "no_bbox", "no_zone"):
        report = RigidReport()
        report.skipped.append(("b1", reason))
        with pytest.raises(DocumentParseError):
            _assert_paintable({}, report)


def test_assert_paintable_allows_documents_with_nothing_to_translate() -> None:
    """A document whose only blocks are non-prose is legitimately empty."""
    from ubt.adapters.pdf.rigid.typesetter import RigidReport, _assert_paintable

    report = RigidReport()
    report.skipped.append(("img1", "non_prose"))
    report.skipped.append(("fig1", "empty_target"))
    _assert_paintable({}, report)  # must not raise


# --- 3. body prose mentioning a venue and page numbers translates -----------


@pytest.mark.parametrize(
    "text",
    [
        "The 2021 IEEE Access paper spans pages 100-110.",
        "Figure 4. Results reported by Springer in 2020, pages 12-18.",
        "This approach was published by Springer in 2019, spanning pages 45-60.",
    ],
)
def test_body_prose_with_venue_and_page_numbers_translates(text: str) -> None:
    from ubt.core.cleaners.skip_rules import classify_skip

    assert classify_skip(text) is None, text


# --- 4. latin clause/sentence splits preserve the source exactly ------------


def test_latin_splits_reconstruct_the_source() -> None:
    from ubt.adapters.pdf.rigid.rows import split_clauses, split_sentences

    src = "Bonjour le monde. Ceci est un test, avec des virgules; et des deux-points: oui!"
    assert "".join(split_clauses(src)) == src
    assert "".join(split_sentences(src)) == src

    cjk = "第一句。第二句，第三句；第四句：第五句。"
    assert "".join(split_clauses(cjk)) == cjk


# --- 5. docling-spaced three-level references -------------------------------


def test_spaced_three_level_reference_is_seen() -> None:
    from ubt.core.qe.added_content import AddedContentGate, reference_tokens

    assert reference_tokens("Section 3 . 4 . 1") == frozenset({"3.4.1"})
    decision = AddedContentGate().evaluate(
        "As described in Section 3 . 4 . 1 , the method generalises.",
        "如第 3.4.1 节所述，该方法得到了推广。",
    )
    assert decision.passed, decision.reason


# --- 6. superscript power is not a lost number ------------------------------


def test_superscript_power_is_not_a_lost_number() -> None:
    from ubt.core.validators.consistency import NumericConsistencyValidator

    result = NumericConsistencyValidator().validate("The area is 10^2 m.", "面积为 10² 米。")
    assert result.is_valid, result.message


# --- 7. ordered-list numbers after an intro colon survive -------------------


def test_ordered_list_numbers_after_an_intro_colon_survive() -> None:
    from ubt.core.cleaners.lnds_pruner import collect_dropped_line_indices

    lines = [
        "Steps:",
        "1",
        "Install the package.",
        "2",
        "Run the tests.",
        "3",
        "Deploy.",
        "4",
        "Verify.",
    ]
    assert collect_dropped_line_indices(lines) == set()


# --- 8. a faithfully repeated source line is not a hallucination ------------


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


# --- 9. failed batch retries before newer updates ---------------------------


class _FlakyBlockingLedger:
    """First save blocks, then fails; later saves commit in call order."""

    def __init__(self) -> None:
        self.first_entered = threading.Event()
        self.release_first = threading.Event()
        self.calls = 0
        self.committed: list[dict[str, object]] = []

    def save_checkpoints_batch(self, updates, **_kwargs):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            self.first_entered.set()
            self.release_first.wait(timeout=5)
            raise sqlite3.OperationalError("database is locked")
        self.committed.extend(updates)
        return len(updates)


def test_failed_batch_is_retried_before_newer_updates() -> None:
    from ubt.core.engine.ledger_flusher import CheckpointBatchFlusher

    async def scenario() -> list[str]:
        ledger = _FlakyBlockingLedger()
        flusher = CheckpointBatchFlusher(ledger, flush_interval=0.01, max_batch_size=50)
        await flusher.enqueue({"block_id": "X", "status": "v1"})
        # Let the first save start, enqueue the newer update while it is in
        # flight, then let the save fail.
        await asyncio.to_thread(ledger.first_entered.wait, 5)
        await flusher.enqueue({"block_id": "X", "status": "v2"})
        ledger.release_first.set()
        await flusher.close()
        return [str(u["status"]) for u in ledger.committed]

    assert asyncio.run(scenario()) == ["v1", "v2"]


# --- 10. tiered runner binds the glossary to its heuristic leg --------------


def test_tiered_runner_binds_glossary_to_heuristic() -> None:
    from ubt.core.qe.comet_runner import HeuristicQERunner
    from ubt.core.qe.llm_judge import TieredQERunner

    tiered = TieredQERunner(heuristic=HeuristicQERunner())
    assert tiered.is_glossary_aware() is False

    bound = tiered.with_glossary([{"source": "term", "target": "术语"}])
    assert bound.is_glossary_aware() is True


# --- 11. superscript 7 in an author byline ----------------------------------


def test_superscript_seven_is_recognised_in_a_byline() -> None:
    from ubt.core.cleaners.skip_rules import _is_author_byline

    assert _is_author_byline("John Doe\u2077, Jane Roe\u2078") is True


# --- 12. metrics --json error path stays off stdout -------------------------


def test_metrics_json_errors_go_to_stderr(tmp_path) -> None:
    missing = tmp_path / "missing_metrics.json"
    commands = [
        [sys.executable, "-m", "ubt", "metrics", "show", str(missing), "--json"],
        [
            sys.executable,
            "-m",
            "ubt",
            "metrics",
            "compare",
            str(missing),
            str(missing),
            "--json",
        ],
    ]
    for command in commands:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )
        assert proc.returncode == 1
        assert proc.stdout.strip() == ""
        assert "not found" in proc.stderr
