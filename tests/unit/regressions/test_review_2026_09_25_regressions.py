"""Regression guards for the 2026-09-25 review round (round 6).

One test per defect found by that round, each reproducing the user-visible
outcome rather than the internals, so reverting the fix fails the test.
"""

from __future__ import annotations

import pytest

from ubt.core.language_profile import (
    ZH,
    get_pair_policy,
    get_profile,
    is_supported_lang,
    normalize_lang_code,
)

# ---------------------------------------------------------------------------
# LANG-1: region tags (zh-CN / zh-TW / en-US) passed entry validation but
# crashed after a full ingest with "Unknown language profile: 'zh-cn'".
# ---------------------------------------------------------------------------


def test_region_tag_resolves_to_base_profile() -> None:
    """Region/script subtags must resolve to their base profile instead of crashing."""
    assert get_profile("zh-CN") == ZH
    assert get_profile("zh-TW") == ZH
    assert get_profile("zh-Hans") == ZH
    assert get_profile("en-US") == get_profile("en")
    assert get_profile("en_GB") == get_profile("en")
    assert normalize_lang_code("zh-Hans") == "zh"
    assert normalize_lang_code("zh-TW") == "zh"


def test_unsupported_base_language_still_raises() -> None:
    """Normalization must not turn an unsupported language into a silent fallback."""
    with pytest.raises(ValueError, match="Unknown language profile"):
        get_profile("pt-BR")
    with pytest.raises(ValueError, match="Unknown language profile"):
        get_profile("klingon")


def test_is_supported_lang_reports_entry_point_truth() -> None:
    assert is_supported_lang("zh-CN") is True
    assert is_supported_lang("zh-TW") is True
    assert is_supported_lang("en-US") is True
    assert is_supported_lang("zh") is True
    assert is_supported_lang("pt-BR") is False
    assert is_supported_lang("it") is False
    assert is_supported_lang("klingon") is False


def test_pair_policy_uses_calibrated_band_for_region_tags() -> None:
    """en-US -> zh-CN must hit the calibrated (0.2, 1.5) band, not the generic fallback."""
    policy = get_pair_policy("en-US", "zh-CN")
    assert policy.target_code == "zh"
    assert policy.source_code == "en"
    assert (policy.min_length_ratio, policy.max_length_ratio) == (0.2, 1.5)


def test_fast_pass_filter_accepts_region_tag_target() -> None:
    from ubt.core.qe.fast_pass import FastPassFilter

    fp = FastPassFilter(source_lang="en", target_lang="zh-CN")
    decision = fp.evaluate(
        "Psychological research shows that sleep deprivation impairs cognition.",
        "心理学研究表明，睡眠不足会损害认知能力。",
    )
    assert decision.passed is True


def test_api_rejects_unsupported_target_before_ingest() -> None:
    from pydantic import ValidationError

    from ubt.api.models import JobSubmitRequest

    with pytest.raises(ValidationError):
        JobSubmitRequest(input_path="book.md", target_lang="pt-BR")
    # A supported region tag is accepted and preserved verbatim for font selection.
    req = JobSubmitRequest(input_path="book.md", target_lang="zh-CN")
    assert req.target_lang == "zh-CN"


def test_mcp_check_lang_rejects_unsupported_target() -> None:
    from ubt.core.exceptions import UBTError
    from ubt.mcp.server import _check_lang

    with pytest.raises(UBTError, match="[Uu]nsupported"):
        _check_lang("pt-BR", field="target_lang")
    # A supported region tag passes through untouched.
    assert _check_lang("zh-CN", field="target_lang") == "zh-CN"
    # Shape violations are still rejected by the regex guard.
    with pytest.raises(UBTError):
        _check_lang("not a lang!", field="target_lang")


# ---------------------------------------------------------------------------
# PRICE-1: the pre-flight estimate priced a self-hosted endpoint by the model
# NAME (cloud rate), while the runtime report priced it $0 — so a free local run
# could be refused by UBT_BUDGET_USD. See pricing.py's own docstring.
# ---------------------------------------------------------------------------


def test_preflight_estimate_is_free_for_local_endpoint() -> None:
    from ubt.core.engine.cost_estimate import estimate_draft_cost_from_totals

    est = estimate_draft_cost_from_totals(
        billable_blocks=100,
        source_chars=200_000,
        draft_model="qwen3:8b",
        prefix_tokens=2_000,
        base_url="http://localhost:11434/v1",
    )
    assert est.cost_usd_uncached == 0.0
    assert est.cost_usd_cached == 0.0
    assert est.money_is_unknown is False


def test_preflight_and_runtime_agree_on_local_endpoint() -> None:
    from ubt.core.engine.cost_estimate import estimate_draft_cost_from_totals
    from ubt.core.router.pricing import estimate_cost_usd

    base = "http://localhost:11434/v1"
    est = estimate_draft_cost_from_totals(
        billable_blocks=10,
        source_chars=20_000,
        draft_model="qwen3:8b",
        prefix_tokens=2_000,
        base_url=base,
    )
    runtime = estimate_cost_usd(
        {"qwen3:8b": {"prompt_tokens": 100_000, "completion_tokens": 50_000}},
        base_url=base,
    )
    assert runtime == 0.0
    assert est.cost_usd_uncached == 0.0


def test_remote_endpoint_still_prices_by_name() -> None:
    """The local-free rule must not zero a real cloud bill."""
    from ubt.core.engine.cost_estimate import estimate_draft_cost_from_totals

    est = estimate_draft_cost_from_totals(
        billable_blocks=10,
        source_chars=20_000,
        draft_model="qwen3:8b",
        prefix_tokens=2_000,
        base_url="https://api.example.com/v1",
    )
    assert est.cost_usd_uncached is not None
    assert est.cost_usd_uncached > 0


def test_cost_preflight_does_not_refuse_a_free_local_run(tmp_path) -> None:
    """End-to-end through the stage: a local run under a tight budget must pass."""
    import asyncio

    from ubt.core.config import UBTConfig
    from ubt.core.engine.stages.preflight import run_cost_preflight_stage
    from ubt.core.ir.models import BlockType, IRBlock
    from ubt.core.router.provider import MockModelProvider
    from ubt.core.router.router import ModelRouter

    class _Ctx:
        job_id = "job_local"
        source_lang = "en"
        target_lang = "zh"

        def __init__(self, config, router, blocks):
            self.config = config
            self.router = router
            self._blocks = blocks

        async def current_blocks(self):
            return self._blocks

    blocks = [
        IRBlock(
            id=f"ch01#b{i:03d}",
            spine_index=i,
            block_type=BlockType.NARRATIVE,
            source_text="A sentence of source prose. " * 20,
        )
        for i in range(1, 6)
    ]
    router = ModelRouter(
        provider=MockModelProvider(default_response="x"),
        draft_model="qwen3:8b",
        repair_model="qwen3:8b",
    )
    config = UBTConfig(
        db_dir=tmp_path,
        base_url="http://localhost:11434/v1",
        draft_model="qwen3:8b",
        budget_usd=0.000001,
    )
    # Must not raise: a self-hosted run is $0 by construction.
    asyncio.run(run_cost_preflight_stage(_Ctx(config, router, blocks)))


def test_cost_preflight_still_refuses_a_paid_run_over_budget(tmp_path) -> None:
    """Positive control: the refusal path must stay live for a paid endpoint."""
    import asyncio

    from ubt.core.config import UBTConfig
    from ubt.core.engine.stages.preflight import run_cost_preflight_stage
    from ubt.core.exceptions import UBTError
    from ubt.core.ir.models import BlockType, IRBlock
    from ubt.core.router.provider import MockModelProvider
    from ubt.core.router.router import ModelRouter

    class _Ctx:
        job_id = "job_paid"
        source_lang = "en"
        target_lang = "zh"

        def __init__(self, config, router, blocks):
            self.config = config
            self.router = router
            self._blocks = blocks

        async def current_blocks(self):
            return self._blocks

    blocks = [
        IRBlock(
            id=f"ch01#b{i:03d}",
            spine_index=i,
            block_type=BlockType.NARRATIVE,
            source_text="A sentence of source prose. " * 200,
        )
        for i in range(1, 20)
    ]
    router = ModelRouter(
        provider=MockModelProvider(default_response="x"),
        draft_model="gpt-4o",
        repair_model="gpt-4o",
    )
    config = UBTConfig(
        db_dir=tmp_path,
        base_url="https://api.openai.com/v1",
        draft_model="gpt-4o",
        budget_usd=0.000001,
    )
    with pytest.raises(UBTError, match="exceeds --budget-usd"):
        asyncio.run(run_cost_preflight_stage(_Ctx(config, router, blocks)))


def test_gpt_41_does_not_inherit_legacy_gpt4_rates() -> None:
    """A newer family must not silently resolve to the shorter legacy prefix."""
    from ubt.core.router.pricing import resolve_model_prices

    assert resolve_model_prices("gpt-4.1") == (2.00, 8.00)
    assert resolve_model_prices("gpt-4.1-mini") == (0.40, 1.60)
    assert resolve_model_prices("gpt-4.1") != resolve_model_prices("gpt-4")


# ---------------------------------------------------------------------------
# OCR-1: the "no OCR driver" hint told users to run `--ocr vlm` for scanned
# pages, but a vision-LLM driver returns no measured boxes and recognition
# fails closed by design — so the advertised remediation could never work.
# ---------------------------------------------------------------------------


def test_ocr_unavailable_hint_does_not_offer_vision_llm_for_scans() -> None:
    from ubt.adapters.pdf.docling_parser import _ocr_unavailable_hint

    hint = _ocr_unavailable_hint("auto")
    # The numbered remediation options are the "offers"; none may be the
    # vision-LLM mode, which yields text but no geometry and so cannot
    # transcribe a scan.
    offered = "\n".join(ln for ln in hint.splitlines() if ln.strip()[:1].isdigit())
    assert "--ocr vlm" not in offered
    # A measured-box engine must be named instead.
    assert "rapidocr" in offered
    assert "sidecar" in offered
    assert "--ocr cloud" in offered
    # And the hint must say why the vision route is not the answer.
    assert "cannot transcribe a scanned page" in hint


def test_unmeasured_vlm_driver_is_not_scan_capable() -> None:
    from ubt.adapters.pdf.docling_parser import _driver_can_transcribe_scans
    from ubt.adapters.pdf.vlm.drivers.cloud_driver import CloudOcrDriver

    assert _driver_can_transcribe_scans(CloudOcrDriver(provider="vlm")) is False
    assert _driver_can_transcribe_scans(CloudOcrDriver(provider="cloud")) is True


# ---------------------------------------------------------------------------
# CLEAN-1: the kerning/subscript rule rewrote ordinary English prose during
# ingest ("I am," -> "I_am,"), corrupting the delivered source text.
# ---------------------------------------------------------------------------


def test_kerning_subscript_rule_does_not_corrupt_english_prose() -> None:
    from ubt.core.cleaners.lnds_pruner import strip_textbook_ocr_artifacts

    assert strip_textbook_ocr_artifacts("I am, therefore I think.") == "I am, therefore I think."
    assert (
        strip_textbook_ocr_artifacts("He is a lot, more than before.")
        == "He is a lot, more than before."
    )
    # A genuine flattened subscript before a math operator is still joined.
    assert "x_i" in strip_textbook_ocr_artifacts("The term x i = 3 appears.")


# ---------------------------------------------------------------------------
# CLEAN-2: the roman-numeral page-marker pattern (IGNORECASE `[ivxlcdm]+`)
# deleted the first word of ordinary prose blocks ("Mild\n..." -> dropped).
# ---------------------------------------------------------------------------


def test_leading_page_marker_does_not_eat_english_words() -> None:
    from ubt.core.cleaners.dynamic_boilerplate import BoilerplateFingerprint

    fp = BoilerplateFingerprint()
    for word in ("Mild", "Civil", "Dim", "Mix"):
        text = f"{word}\nBody sentence follows here."
        assert fp.clean_head(text)[0] == text, word
    # Real page markers are still stripped.
    assert fp.clean_head("xiv\nChapter body")[0] == "Chapter body"
    assert fp.clean_head("Page 42\nBody")[0] == "Body"


# ---------------------------------------------------------------------------
# CLEAN-3: benign paired tags were escaped into visible text
# ("<div>hi</div>" -> "&lt;div&gt;hi&lt;/div&gt;"), contradicting the module
# docstring that says they are dropped while inner text is kept.
# ---------------------------------------------------------------------------


def test_benign_paired_tags_are_dropped_not_escaped() -> None:
    import html

    from ubt.core.cleaners.html_sanitizer import sanitize_html_fragment

    assert sanitize_html_fragment("<div>hi</div>") == "hi"
    assert sanitize_html_fragment('<font color="red">hi</font>') == "hi"
    # Unpaired pseudo-tags must still survive as literal text (escaped in the
    # raw string, which is the sanitizer's contract; they render as-is).
    assert "List<T>" in html.unescape(sanitize_html_fragment("List<T>"))
    assert "<stdio.h>" in html.unescape(sanitize_html_fragment("<stdio.h>"))
    # A dangerous container is still dropped together with its content.
    dropped = sanitize_html_fragment("<div>ok<script>alert(1)</script></div>")
    assert "ok" in dropped and "alert" not in dropped


# ---------------------------------------------------------------------------
# GATE-1: the numeric-idiom exemption was a block-level flat set, so one idiom
# ("top 10") exempted every occurrence of "10" — a dropped "Chapter 10" passed.
# RED evidence (round 1): validate(...).is_valid was True before the fix.
# ---------------------------------------------------------------------------


def test_numeric_idiom_exemption_is_occurrence_scoped() -> None:
    from ubt.core.validators.consistency import NumericConsistencyValidator

    v = NumericConsistencyValidator()
    src = "The model ranks in the top 10 globally. Chapter 10 explains the method."
    tgt = "该模型在全球排名前十。"
    assert v.validate(src, tgt).is_valid is False
    # The idiom on its own still passes (idiomatic rendering without the digit).
    assert v.validate("It is in the top 10.", "它位列前十。").is_valid is True


# ---------------------------------------------------------------------------
# GATE-2: the numeric boundary `(?<!\d)…(?!\d)` ignored a decimal point, so a
# changed value ("3" -> "3.5") passed. RED evidence: is_valid was True.
# ---------------------------------------------------------------------------


def test_numeric_gate_rejects_changed_decimal() -> None:
    from ubt.core.validators.consistency import NumericConsistencyValidator

    v = NumericConsistencyValidator()
    assert v.validate("The value is 3 units.", "The value is 3.5 units.").is_valid is False
    assert v.validate("There are 5 items.", "There are 3.5 items.").is_valid is False
    # A sentence-final period is not a decimal: "3." is still the number 3.
    assert v.validate("The value is 3.", "The value is 3.").is_valid is True


# ---------------------------------------------------------------------------
# GATE-3: the added-content reference extractor only recognised two-part
# numbers ("3.11"), so appendix ("A.10") and three-level ("3.4.1") references
# produced no token at all — invisible to the gate in both source and target.
# ---------------------------------------------------------------------------


def test_appendix_and_three_level_references_are_seen() -> None:
    from ubt.core.qe.added_content import reference_tokens

    assert reference_tokens("See Appendix A.10 for the derivation.") == {"A.10"}
    assert reference_tokens("见附录 A.10 中的推导。") == {"A.10"}
    assert reference_tokens("Section 3.4.1 generalises Eq. (3.4).") == {"3.4.1", "3.4"}


# ---------------------------------------------------------------------------
# GATE-4: the "source carries that bare number" exemption ignored *how* the
# target used the number, so turning a plain source quantity into a figure
# callout ("3.5 times" -> "见图 3.5") was exempted — a fabricated reference
# the source never had. RED evidence (round 1): passed was True.
# ---------------------------------------------------------------------------


def test_fabricated_figure_reference_is_not_exempted_by_a_plain_source_number() -> None:
    from ubt.core.qe.added_content import AddedContentGate

    decision = AddedContentGate().evaluate(
        "The speedup was 3.5 times.", "见图 3.5，实现了 3.5 倍的加速。"
    )
    assert not decision.passed
    assert "3.5" in decision.fabricated_refs


def test_correct_figure_reference_still_passes_when_source_has_it() -> None:
    from ubt.core.qe.added_content import AddedContentGate

    decision = AddedContentGate().evaluate(
        "Fig. 3.5 shows the surface potential.", "图 3.5 展示了表面电势。"
    )
    assert decision.passed, decision.reason


# ---------------------------------------------------------------------------
# LOG-1: the CLI callback installs a RichHandler onto a *stdout* Console; a
# later `--json` run asks setup_logging for stderr routing, but because root
# already had a handler and no console was passed, the stdout handler stayed
# attached and log records polluted the machine-readable JSON on stdout.
# ---------------------------------------------------------------------------


def test_json_mode_repoints_logs_from_stdout_to_stderr() -> None:
    import io
    import logging

    from rich.console import Console

    from ubt.core.log_config import setup_logging

    root = logging.getLogger()
    saved = list(root.handlers)
    root.handlers.clear()
    try:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        # Human-mode CLI callback: records go through a stdout Rich console.
        setup_logging(console=Console(file=stdout_buf))
        # `--json` then requests stderr routing at WARNING.
        setup_logging(level="WARNING", stream=stderr_buf)

        logging.getLogger("ubt.some.module").warning("json-mode warning")

        assert "json-mode warning" in stderr_buf.getvalue()
        assert "json-mode warning" not in stdout_buf.getvalue()
    finally:
        for handler in list(root.handlers):
            if handler not in saved:
                root.removeHandler(handler)
        root.handlers[:] = saved


# ---------------------------------------------------------------------------
# LOG-2: the TUI's console-detach scan read only ``handler.stream``, which a
# RichHandler does not have — its destination is ``handler.console.file`` — so
# the CLI's RichHandler survived into the alternate screen and log records were
# drawn into the live TUI.
# ---------------------------------------------------------------------------


def test_tui_detaches_a_rich_console_handler(tmp_path) -> None:
    import logging
    import sys

    from rich.console import Console
    from rich.logging import RichHandler

    from ubt.tui.logsetup import route_logs_to_file

    root = logging.getLogger()
    saved = list(root.handlers)
    root.handlers.clear()
    try:
        rich_handler = RichHandler(console=Console(file=sys.stderr))
        root.addHandler(rich_handler)

        route_logs_to_file(tmp_path)

        assert rich_handler not in root.handlers
        assert not any(isinstance(h, RichHandler) for h in root.handlers)
    finally:
        for handler in list(root.handlers):
            if handler not in saved:
                root.removeHandler(handler)
        root.handlers[:] = saved


# ---------------------------------------------------------------------------
# API-1: with no UBT_ALLOWED_DIRS (the default deployment) a submit that omits
# output_path 403'd, because the derived default deliverable lives in
# ~/Documents/UBT while resolve_secure_path confines to cwd + db_dir.
# ---------------------------------------------------------------------------


def test_submit_without_output_path_works_without_an_allowlist(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from ubt.api.app import create_app
    from ubt.core.config import UBTConfig

    monkeypatch.delenv("UBT_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("XDG_DOCUMENTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "book.md"
    input_file.write_text("# Title\n\nSome source prose.\n", encoding="utf-8")

    config = UBTConfig(
        db_dir=tmp_path / "ledgers",
        rate_limit_rpm=600,
        draft_model="mock-draft",
        repair_model="mock-repair",
    )
    assert config.allowed_base_dirs() == []  # the bug's precondition

    client = TestClient(create_app(config=config))
    resp = client.post("/jobs/submit", json={"input_path": str(input_file), "target_lang": "zh"})
    assert resp.status_code == 202, resp.text


# ---------------------------------------------------------------------------
# LOG-3: the CLI callback installed the log handler on a stdout console for
# every subcommand, so a `--json` command that never re-configures logging
# (assess, status, config, metrics) had log records prepended to its JSON.
# In-process runs hide this because the offending third-party INFO fires at
# import time, before any test configures logging — only a fresh process
# reproduces what a user sees.
# ---------------------------------------------------------------------------


def test_json_stdout_is_pure_json_in_a_fresh_process(tmp_path) -> None:
    import json
    import subprocess
    import sys

    book = tmp_path / "probe.md"
    book.write_text("# Chapter 1\n\nA short technical note.\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "ubt", "assess", str(book), "--json"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)  # exactly one object, no log preamble
    assert payload["status"] == "ok"
