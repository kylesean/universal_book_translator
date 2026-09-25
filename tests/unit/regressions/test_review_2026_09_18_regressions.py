"""Regression guards for the 2026-09-18 code review's confirmed pure bugs.

Every case here was executed against the pre-fix code and produced the wrong
result; the assertions below are the corrected behaviour.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ubt.core.cleaners.html_sanitizer import sanitize_html_fragment
from ubt.core.memory.tm import (
    PROVENANCE_HUMAN_PE,
    PROVENANCE_MACHINE,
    TMPendingEntry,
    TranslationMemory,
)
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.extractor import TranslationOutputExtractor

# --- 1. extractor: a prefix pattern without a required colon ate real translation

LEAD_INS = [
    ("Translation: 从前有一座山。", "从前有一座山。"),
    ("这是中文翻译：从前有一座山。", "从前有一座山。"),
    ("以下是最终精修中文翻译：从前有一座山。", "从前有一座山。"),
    ("Here is the final translation: Once upon a time.", "Once upon a time."),
]

MUST_SURVIVE = [
    "翻译过程中的注意事项：见下文。",
    "Translations of the term appear below.",
    "翻译如下所示。这是一段完整的译文。",
]


@pytest.mark.parametrize(("raw", "expected"), LEAD_INS)
def test_conversational_lead_in_is_still_stripped(raw: str, expected: str) -> None:
    assert TranslationOutputExtractor.extract(raw) == expected


@pytest.mark.parametrize("text", MUST_SURVIVE)
def test_prose_that_only_resembles_a_lead_in_is_untouched(text: str) -> None:
    # Pre-fix, the optional colon let these match and lose their first words.
    assert TranslationOutputExtractor.extract(text) == text


def test_leading_blockquote_marker_is_preserved() -> None:
    assert TranslationOutputExtractor.extract("> 引用的原文块。") == "> 引用的原文块。"


# --- 2. html_sanitizer: control characters hid a blocked scheme from the matcher

CONTROL_CHAR_URLS = [
    '<a href="java\tscript:alert(1)">x</a>',
    '<a href="java\nscript:alert(1)">x</a>',
    '<a href="HTM\x00L:alert(1)">x</a>',
    '<a href="vbscript:msgbox(1)">x</a>',
]


@pytest.mark.parametrize("fragment", CONTROL_CHAR_URLS)
def test_scheme_split_by_control_characters_is_dropped(fragment: str) -> None:
    # Pre-fix the scheme never matched the anchored regex, so the URL was
    # treated as a relative link and emitted verbatim.
    assert "href" not in sanitize_html_fragment(fragment)


def test_data_url_is_refused_for_navigation_but_allowed_for_images() -> None:
    assert "href" not in sanitize_html_fragment('<a href="data:text/html,<b>x</b>">x</a>')
    img = '<img src="data:image/png;base64,iVBORw0KGgo=" alt="i">'
    assert "data:image/png" in sanitize_html_fragment(img)


def test_legitimate_relative_and_absolute_links_survive() -> None:
    for href in ("chapter1.xhtml", "#anchor", "../img/a.png", "https://ok.example/x"):
        assert f'href="{href}"' in sanitize_html_fragment(f'<a href="{href}">x</a>')


# --- 3. fast_pass: the table branch stripped every verbatim term, so an
#          untouched table read as empty residue and passed

TABLE_SRC = "| Cell | Alpha | Beta Value |\n| --- | --- | --- |\n| 12 | 34 | 56 |"
TABLE_OK = "| 单元格 | 阿尔法 | Beta 值 |\n| --- | --- | --- |\n| 12 | 34 | 56 |"


def test_untranslated_table_no_longer_passes_the_script_density_gate() -> None:
    assert (
        not FastPassFilter(source_lang="en", target_lang="zh").evaluate(TABLE_SRC, TABLE_SRC).passed
    )


def test_translated_table_with_identifier_carryovers_still_passes() -> None:
    assert FastPassFilter(source_lang="en", target_lang="zh").evaluate(TABLE_SRC, TABLE_OK).passed


# --- 4. TM: a stored src==tgt passthrough came back as a finished translation

TM_SRC = "The kernel caches key-value tensors across decoding steps."
TM_NEAR = TM_SRC.replace("decoding", "decode")
TM_ZH = "内核在解码步之间缓存键值张量。"


def test_tm_refuses_passthrough_rows_on_both_read_paths(tmp_path: Path) -> None:
    tm = TranslationMemory(tmp_path / "tm.sqlite")
    try:
        tm.writeback([TMPendingEntry("en", "zh", TM_SRC, TM_SRC, provenance=PROVENANCE_MACHINE)])
        assert tm.entry_count() == 1  # write behaviour is unchanged by design
        assert tm.lookup_exact("en", "zh", TM_SRC) is None
        assert tm.lookup_fuzzy("en", "zh", TM_NEAR, threshold=0.7) is None

        # Control: the same two lookups succeed once a real translation lands.
        tm.writeback([TMPendingEntry("en", "zh", TM_SRC, TM_ZH, provenance=PROVENANCE_HUMAN_PE)])
        assert tm.lookup_exact("en", "zh", TM_SRC) is not None
        assert tm.lookup_fuzzy("en", "zh", TM_NEAR, threshold=0.7) is not None
    finally:
        tm.close()


# --- 5. typst_math: repeated trailing equation numbers


def test_repeated_trailing_equation_numbers_are_stripped() -> None:
    from ubt.adapters.pdf.typst_math import _clean_ocr_formula

    assert _clean_ocr_formula("f(x) = 1 (1)  (2)  (3)  ") == "f(x)=1"
    assert _clean_ocr_formula("v = a + b \\\\ (2.14) \\\\") == "v = a + b \\\\"
    assert _clean_ocr_formula("E = mc^2") == "E = mc^2"


def test_trailing_equation_number_run_is_matched_in_linear_time() -> None:
    """The nested quantifiers used to backtrack exponentially on a near-miss."""
    from ubt.adapters.pdf.typst_math import _clean_ocr_formula

    adversarial = "x = y " + "(1)  " * 16 + "X"
    start = time.perf_counter()
    _clean_ocr_formula(adversarial)
    elapsed = time.perf_counter() - start
    # Measured 12.1 s pre-fix; the linear peel is sub-millisecond.
    assert elapsed < 2.0, f"equation-number strip took {elapsed:.3f}s"


# --- 6. typst_healer: nullified lines left no trace outside the log stream


def test_persistent_comment_heal_reports_every_nullified_line() -> None:
    from ubt.adapters.pdf.typst_healer import _heal_persistent_comment_error

    lines = ["// already commented $x$", "", 'a = "unclosed', "", "", "// error line $y$"]
    audit: list[str] = []
    assert _heal_persistent_comment_error(lines, 5, audit)
    # The culprit (line 3) and the error line (line 6) both lose their content.
    assert audit == ['line 3: a = "unclosed', "line 6: // error line $y$"]


# --- 7. numeric gate: a quantity re-expressed with a unit word read as a
#         dropped number, so correct prose went to BLOCKED_HUMAN


def test_scale_rewriting_is_accepted_and_real_omissions_are_not() -> None:
    from ubt.core.validators.consistency import NumericConsistencyValidator

    validator = NumericConsistencyValidator()
    # (source, target, must_pass). Measured against the pre-fix code: both
    # scale cases reported a missing number and were caught below.
    cases = [
        (
            "该产品配备250万像素摄像头，售价1,200元。",
            "The device features a 2.5-megapixel camera priced at 1,200 yuan.",
            True,
        ),
        ("The device has a 2.5 million pixel camera.", "该产品配备250万像素摄像头。", True),
        ("a 150 million person country", "一个1.5亿人口的国家", True),
        ("a 30 millisecond delay", "30ms 延迟", True),
        # A genuinely dropped number stays a defect.
        ("a 2.5-megapixel camera and 7 sensors", "a camera with 7 sensors", False),
        ("配备250万像素摄像头", "配备摄像头", False),
        ("in 1984 the ratio was 15.6", "八十年代的比率", False),
    ]
    for source, target, must_pass in cases:
        result = validator.validate(source, target)
        assert result.is_valid is must_pass, f"{source!r} -> {target!r}: {result.message}"


# --- 8. cjk spacing: the space-removal pass ran on every target and deleted
#         Korean word boundaries in the delivered file


def test_cjk_space_removal_only_fires_for_spaceless_targets() -> None:
    from ubt.core.cleaners.cjk_spacing import normalize_publishing_cjk

    korean = "이것은 테스트 입니다."
    assert normalize_publishing_cjk(korean, target_lang="ko") == korean
    assert normalize_publishing_cjk(korean, target_lang="en") == korean
    # Chinese still loses the stray space between Han characters.
    assert normalize_publishing_cjk("中 文 书", target_lang="zh") == "中文书"
    # Trailing whitespace before a newline is noise in any language.
    assert normalize_publishing_cjk("il y a  \nrien", target_lang="fr") == "il y a\nrien"


def test_overlay_renderer_passes_the_target_language_through() -> None:
    """The anchored/overlay path hardcoded zh, so a Korean book lost its spaces twice."""
    from ubt.adapters.pdf.overlay_text import prepare_overlay_text

    korean = "이것은 테스트 입니다."
    assert prepare_overlay_text(korean, target_lang="ko") == korean
    assert prepare_overlay_text("中 文 书", target_lang="zh") == "中文书"


# --- 9. ledger: a metadata write for a missing job row left BEGIN IMMEDIATE
#         open, poisoning every later write on that connection


def test_metadata_write_for_unknown_job_leaves_no_transaction_open(tmp_path: Path) -> None:
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.exceptions import LedgerError

    ledger = SQLiteJobLedger(tmp_path / "ghost.sqlite")
    # Pre-fix: the second call raised "cannot start a transaction within a
    # transaction" and a second connection saw "database is locked". The
    # rollback-on-missing-row remains; the write now *also* surfaces (P1-12: a
    # silently-swallowed source_fingerprint write let the next resume clear the
    # whole book), so we expect LedgerError while still asserting no dangling
    # transaction/lock is left behind.
    with pytest.raises(LedgerError):
        ledger.set_job_metadata_value("ghost", "output_file", "/tmp/a.pdf")
    with pytest.raises(LedgerError):
        ledger.set_job_metadata_value("ghost", "report_file", "/tmp/a.md")
    # A second connection can still use the DB: proves no write lock leaked.
    ledger.close()
    reopened = SQLiteJobLedger(tmp_path / "ghost.sqlite")
    try:
        assert reopened.get_job_status("ghost") is None
    finally:
        reopened.close()


# --- 10. dry-run: each surface parsed the draft prompt its own way, and the
#          rehearsal run has to read the block back out of it


def test_draft_source_is_recovered_from_every_builder() -> None:
    import inspect

    from ubt.core.router.prompts import (
        build_hybrid_draft_prompt,
        build_minimal_draft_prompt,
        build_rich_draft_prompt,
        draft_source_from_prompt,
    )

    source = "This block has 250万 pixels.\n\nSecond paragraph."
    for builder in (
        build_minimal_draft_prompt,
        build_hybrid_draft_prompt,
        build_rich_draft_prompt,
    ):
        params = inspect.signature(builder).parameters
        kwargs = {
            "source_text": source,
            "target_lang": "zh",
            "source_lang": "en",
            "glossary_table": "node | 节点",
            "global_glossary": "FET | 场效应管",
            "few_shot_reference": "EN: x\nZH: y",
            "genre_profile": "textbook",
        }
        _, user_prompt = builder(**{k: v for k, v in kwargs.items() if k in params})
        recovered = draft_source_from_prompt(user_prompt)
        assert recovered == source.strip(), f"{builder.__name__} leaked: {recovered[:60]!r}"


# --- 11. api: request fields were typed `str` and described values the enums
#          do not have, so a bad preset stranded the job at "running"


def test_api_rejects_values_its_enums_do_not_define() -> None:
    from pydantic import ValidationError

    from ubt.api.app import JobSubmitRequest

    accepted = JobSubmitRequest.model_validate(
        {"input_path": "/tmp/a.pdf", "preset": "publication"}
    )
    assert accepted.preset == "publication"
    for bad in ({"preset": "draft"}, {"target_lang": 'zh") #import "x'}, {"formula_mode": "raw"}):
        payload = {"input_path": "/tmp/a.pdf", **bad}
        with pytest.raises(ValidationError):
            JobSubmitRequest.model_validate(payload)


@pytest.mark.asyncio
async def test_a_field_that_fails_to_map_fails_the_job_not_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import importlib
    from typing import Any

    from pydantic import SecretStr

    api_module = importlib.import_module("ubt.api.app")
    from ubt.api.app import JobManager, JobSubmitRequest
    from ubt.core.config import MOCK_API_KEY, UBTConfig
    from ubt.core.engine.events import TranslationProgressEvent

    def _unmapable(_request: Any) -> dict[str, Any]:
        raise ValueError("a request field no config accepts")

    monkeypatch.setattr(api_module, "overrides_from_request", _unmapable)
    manager = JobManager()
    record = manager.create_job(
        JobSubmitRequest(input_path="/tmp/nowhere.pdf"), job_id="stranded-guard"
    )
    subscriber: asyncio.Queue[TranslationProgressEvent | None] = asyncio.Queue()
    record.subscribers.append(subscriber)

    await manager.execute_job(record, UBTConfig(api_key=SecretStr(MOCK_API_KEY)))

    assert record.status == "failed"
    assert record.error is not None and "stranded-guard" in record.error
    # The termination sentinel must reach subscribers, or the SSE client hangs.
    assert subscriber.get_nowait() is None


# --- 12/13. one alias table, one job-id pattern, one default artifact path

# Repo root for the "does anyone re-spell it" scans below.
ubt_root = Path(__file__).resolve().parents[3]


def test_render_engine_is_canonically_rigid_and_old_spellings_are_gone() -> None:
    from ubt.core.config import canonical_render_engine

    # `rigid` is the single canonical name for the source-geometry route (the
    # old anchored/overlay duality was collapsed with zero backward compat).
    assert canonical_render_engine("rigid") == "rigid"
    assert canonical_render_engine("reflow") == "publication"
    owner = "ubt/core/config.py"
    offenders = [
        path.relative_to(ubt_root).as_posix()
        for path in sorted((ubt_root / "ubt").rglob("*.py"))
        if path.relative_to(ubt_root).as_posix() != owner
        and (
            '"anchored"' in path.read_text(encoding="utf-8")
            or '"overlay"' in path.read_text(encoding="utf-8")
        )
    ]
    assert offenders == [], f"retired render-engine spellings re-spelled in {offenders}"


def test_job_id_pattern_and_default_output_path_have_one_owner() -> None:
    import importlib

    api_app = importlib.import_module("ubt.api.app")
    mcp_server = importlib.import_module("ubt.mcp.server")
    tui_state = importlib.import_module("ubt.tui.state")
    from ubt.core.job_options import JOB_ID_RE, default_output_dir, default_output_path

    assert api_app.JOB_ID_RE is JOB_ID_RE
    assert mcp_server.JOB_ID_RE is JOB_ID_RE
    assert tui_state.JOB_ID_RE is JOB_ID_RE
    assert default_output_path("docs/synthetic-duo.pdf") == (
        default_output_dir() / "synthetic-duo_bilingual.pdf"
    )
    assert default_output_path("/x/book.epub") == (default_output_dir() / "book_bilingual.epub")
    # Every surface derives the same name; nothing may hardcode the directory.
    owner = "ubt/core/job_options.py"
    spellers = [
        path.relative_to(ubt_root).as_posix()
        for path in sorted((ubt_root / "ubt").rglob("*.py"))
        if path.relative_to(ubt_root).as_posix() != owner
        and 'Path("tmp/output")' in path.read_text(encoding="utf-8")
    ]
    assert spellers == [], f"tmp/output re-derived in {spellers}"


# --- 14. the configured RPM was per orchestrator, so concurrent jobs
#          multiplied it against the same credential


def test_orchestrator_honours_an_injected_shared_rate_limiter() -> None:
    from pydantic import SecretStr

    from ubt.core.config import MOCK_API_KEY, UBTConfig
    from ubt.core.engine.pipeline import PipelineOrchestrator
    from ubt.core.router.rate_limiter import AdaptiveTokenBucket

    shared = AdaptiveTokenBucket(initial_rpm=7, max_rpm=7)
    orchestrator = PipelineOrchestrator(
        config=UBTConfig(api_key=SecretStr(MOCK_API_KEY)), rate_limiter=shared
    )
    assert orchestrator.router.rate_limiter is shared


# --- 15. the visual repair crop rasterized a PDF page on the event loop,
#          freezing SSE, the TUI and the queue heartbeat


@pytest.mark.asyncio
async def test_visual_crop_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    from typing import Any

    import ubt.core.ports as ports
    from ubt.core.engine.repair_loop import RepairLoop
    from ubt.core.ir.models import BlockStatus, BoundingBox, IRBlock
    from ubt.core.qe.comet_runner import MockQERunner
    from ubt.core.router.provider import MockModelProvider
    from ubt.core.router.router import ModelRouter

    loop_thread = threading.current_thread()
    seen: dict[str, threading.Thread] = {}

    def _fake_crop(pdf_path: Any, block: Any, dpi: int = 150) -> None:
        seen["thread"] = threading.current_thread()
        return None

    monkeypatch.setattr(ports, "crop_block_image", _fake_crop)
    monkeypatch.setattr(ports, "is_visual_scalpel_applicable", lambda *a, **k: True)

    repair_loop = RepairLoop(
        router=ModelRouter(provider=MockModelProvider()), qe_runner=MockQERunner(default_score=0.9)
    )
    block = IRBlock(
        id="b_crop",
        spine_index=1,
        source_text="E = mc^2 broken",
        draft_text="质能方程",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.2,
        error_flags=["formula_corrupted"],
        bbox=BoundingBox(page=1, x0=0, y0=0, x1=100, y1=20),
    )
    await repair_loop.repair_single_block(block, source_pdf_path=Path("docs/synthetic-duo.pdf"))

    assert "thread" in seen, "visual crop never ran"
    assert seen["thread"] is not loop_thread


# --- 16. the documented `ubt-api --host ...` did nothing: the console script
#          calls run_server() with no arguments


def test_server_bind_flags_are_actually_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from ubt.api.app import _resolve_bind

    monkeypatch.setattr(sys, "argv", ["ubt-api", "--host", "0.0.0.0", "--port", "9000"])
    assert _resolve_bind(None, None) == ("0.0.0.0", 9000)
    monkeypatch.setattr(sys, "argv", ["ubt-api"])
    assert _resolve_bind(None, None) == ("127.0.0.1", 8000)
    # Explicit arguments still win (the bind-guard regression calls this way).
    assert _resolve_bind("0.0.0.0", 1) == ("0.0.0.0", 1)


# --- 17. terminology: the whole book-level sheet rode on every draft request,
#          so the dictionary cost more than the text being translated


def test_global_terminology_sheet_is_capped_by_rank() -> None:
    from ubt.core.config import UBTConfig
    from ubt.core.memory.glossary_table import build_global_glossary_table

    terms = [
        {"source": "node", "translation": "节点", "kind": "term", "frequency": 5, "aliases": []},
        {
            "source": "Elizabeth",
            "translation": "伊丽莎白",
            "kind": "person",
            "frequency": 3,
            "aliases": ["Lizzy"],
        },
        # What load_external_glossary seeds for a user-supplied term.
        {
            "source": "gate",
            "translation": "栅极",
            "kind": "term",
            "frequency": 10_000,
            "aliases": [],
        },
    ]
    assert build_global_glossary_table(terms, 0) == ""
    two = build_global_glossary_table(terms, 2)
    # Names outrank frequency; a user's own decision outranks a mined common term.
    assert "| Elizabeth |" in two and "| gate |" in two
    assert "| node |" not in two
    # The documented default is the cap itself; the guide's table cites it.
    assert UBTConfig().glossary_max_global_entries == 100


# --- 18. tiered QE: the gray band [0.70, 0.80) contains exactly one of the
#          heuristic's twelve discrete values, so 'tiered' bought almost nothing


@pytest.mark.asyncio
async def test_paid_judge_follows_defect_class_and_sampled_passes() -> None:
    from ubt.core.qe.comet_runner import (
        QE_SCORE_EMPTY,
        QE_SCORE_LEAK,
        QE_SCORE_PASS,
        QE_SCORE_STRUCTURAL_OTHER,
        HeuristicQERunner,
    )
    from ubt.core.qe.llm_judge import LLMJudgeQERunner, TieredQERunner

    class _ClassedHeuristic(HeuristicQERunner):
        """Emit a chosen defect class per pair instead of deriving it from content."""

        def __init__(self, scores: list[float]) -> None:
            super().__init__()
            self._scores = scores

        async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
            return self._scores[: len(pairs)]

    scores = [QE_SCORE_PASS, QE_SCORE_STRUCTURAL_OTHER, QE_SCORE_LEAK, QE_SCORE_EMPTY]
    pairs = [{"src": f"seg-{i}", "mt": "译文"} for i in range(len(scores))]
    judged: list[str] = []

    async def fake_judge(**kwargs: object) -> str:
        judged.append(str(kwargs.get("user_prompt", "")))
        return "score: 30"

    def _tiered(pass_sample: float) -> TieredQERunner:
        return TieredQERunner(
            heuristic=_ClassedHeuristic(scores),
            judge=LLMJudgeQERunner(judge_fn=fake_judge),
            pass_sample=pass_sample,
        )

    runner = _tiered(0.0)
    out = await runner.score_pairs(pairs)
    # Only the unclassified structural class is ambiguous enough to be worth asking.
    assert len(judged) == 1 and runner.judge_calls == 1
    assert out[0] == QE_SCORE_PASS, "a clean pass must not move without being sampled"

    judged.clear()
    runner = _tiered(1.0)
    out = await runner.score_pairs(pairs)
    # Sampling passes lets the judge lower a pass that merely broke no invariant;
    # hard defects still never reach it (a judge cannot un-drop a number).
    assert len(judged) == 2 and runner.judge_calls == 2
    assert out[0] == 0.30 and out[2] == QE_SCORE_LEAK and out[3] == QE_SCORE_EMPTY
