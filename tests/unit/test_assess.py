"""Unit tests for the assess quote engine (ubt.core.assess).

The PDF primitives are monkeypatched in most tests so fan-out numbers and
warning codes are asserted deterministically, without touching pdfium.
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from ubt.core.assess import AssessmentError, CostQuote, assess_document, assess_document_async
from ubt.core.config import UBTConfig
from ubt.core.router.pricing import MODEL_PRICES_USD_PER_MTOK

UBT_PKG = Path(__file__).resolve().parents[2]
PRICED_MODEL = "deepseek-chat"


def _config(**updates: object) -> UBTConfig:
    cfg = UBTConfig.from_env()
    return cfg.model_copy(
        update={"draft_model": PRICED_MODEL, "repair_model": PRICED_MODEL, **updates}
    )


@pytest.fixture
def md_book(tmp_path: Path) -> Path:
    f = tmp_path / "novel.md"
    f.write_text("# Start\n\n" + ("Some flowing prose for the quote. " * 60), encoding="utf-8")
    return f


def test_missing_file_raises_unreadable() -> None:
    with pytest.raises(AssessmentError) as exc:
        assess_document("/nonexistent/definitely-missing.pdf", _config())
    assert exc.value.code == "UNREADABLE"


def test_unsupported_suffix(md_book: Path) -> None:
    toml_like = md_book.with_suffix(".qzx")
    toml_like.write_text("x", encoding="utf-8")
    with pytest.raises(AssessmentError) as exc:
        assess_document(toml_like, _config())
    assert exc.value.code == "UNSUPPORTED"


def test_markdown_quote_is_serializable_and_priced(md_book: Path) -> None:
    report = assess_document(md_book, _config())
    payload = json.dumps(report.to_dict())  # JSON-safe end to end
    d = json.loads(payload)
    assert d["status"] == "ok"
    assert d["cost"]["billable_blocks"] >= 1
    assert d["cost"]["billable_blocks_is_exact"] is False
    assert isinstance(d["cost"]["prefix_tokens_per_call"], int)
    assert d["cost"]["total_cost_usd"] is not None
    assert d["cost"]["money_is_unknown"] is False
    assert [w["code"] for w in d["warnings"]].count("MODEL_UNPRICED") == 0


def test_unpriced_model_surfaces_unknown_money(md_book: Path) -> None:
    report = assess_document(
        md_book, _config(draft_model="zzz-nosuch-model", repair_model="zzz-nosuch-model")
    )
    assert report.cost.money_is_unknown is True
    assert report.cost.total_cost_usd is None
    assert any(w.code == "MODEL_UNPRICED" for w in report.warnings)


def test_self_hosted_endpoint_quotes_free_not_unknown(md_book: Path) -> None:
    """A quote against 127.0.0.1 is $0.00, not "unknown".

    "unknown" made the pre-flight money line useless for the self-hosted user
    the project is built for, and — with ``--budget-usd`` — refused the run.
    """
    report = assess_document(
        md_book,
        _config(
            base_url="http://127.0.0.1:9090/v1",
            draft_model="translategemma:4b",
            repair_model="translategemma:4b",
        ),
    )
    assert [w.code for w in report.warnings].count("MODEL_UNPRICED") == 0
    assert report.cost.money_is_unknown is False
    assert report.cost.total_cost_usd == 0.0


def test_fan_out_gates_follow_config(md_book: Path) -> None:
    plain = assess_document(md_book, _config())
    assert plain.cost.qe_calls == 0  # heuristic QE is zero-token
    assert plain.cost.vlm_page_calls == 0  # visual judge off by default
    tiered_off = assess_document(md_book, _config(qe_engine="tiered"))
    assert tiered_off.cost.qe_calls == 0
    tiered_on = assess_document(md_book, _config(qe_engine="tiered", qe_judge_enabled=True))
    assert tiered_on.cost.qe_calls > 0
    judged = assess_document(md_book, _config(visual_judge_enabled=True, allow_page_upload=True))
    assert judged.cost.vlm_page_calls >= 1
    # The shipped default forbids page egress: every judge call then fails
    # before the request, so pricing those pages billed spend that cannot occur.
    closed = assess_document(md_book, _config(visual_judge_enabled=True))
    assert closed.cost.vlm_page_calls == 0

    # The LLM judge only runs when the pipeline can wrap it, and it wraps the
    # heuristic runner alone -- on the COMET/subprocess engine scoring costs no
    # tokens at all, whatever qe_judge_enabled says.
    comet = assess_document(md_book, _config(qe_judge_enabled=True, qe_engine="subprocess"))
    assert comet.cost.qe_calls == 0


def test_repair_fanout_follows_the_engine_knobs(md_book: Path) -> None:
    """UBT_BOTTOM_PERCENTILE moves the engine's repair cut, so it must move the quote.

    ``RepairLoop`` selects ``ceil(bottom_percentile * scored)`` blocks per round;
    the quote kept using the historical 0.15, so a run configured to repair
    *everything* was quoted as if it repaired 15%.
    """
    narrow = assess_document(md_book, _config(bottom_percentile=0.15))
    wide = assess_document(md_book, _config(bottom_percentile=1.0))
    assert narrow.cost.repair_blocks >= 1
    assert wide.cost.repair_blocks > narrow.cost.repair_blocks

    # The short chain caps repair at one round regardless of the configured
    # ceiling, so quoting ``max_repair_rounds`` there priced passes that can
    # never run.
    fan_out = {"max_repair_rounds": 5, "bottom_percentile": 0.5}
    long_route = assess_document(md_book, _config(exec_mode="long", **fan_out))
    short_route = assess_document(md_book, _config(exec_mode="short", **fan_out))
    assert long_route.cost.repair_blocks > short_route.cost.repair_blocks


def test_prefix_measurement_needs_no_credentials(
    md_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("UBT_LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    report = assess_document(md_book, UBTConfig.from_env())
    assert isinstance(report.cost.prefix_tokens_per_call, int)


def _fake_pdf_env(monkeypatch: pytest.MonkeyPatch, *, scanned_share: float) -> None:
    """Patch the four PDF primitives in _pdf_facts (imported at call time)."""
    from ubt.adapters.pdf.extraction_witness import PageVerdict
    from ubt.adapters.pdf.page_profiler import PageKind

    n = 10
    n_scan = round(scanned_share * n)
    profiles = [
        SimpleNamespace(
            kind=PageKind.SCAN_IMAGE if i < n_scan else PageKind.EDITABLE_TEXT,
            facts=SimpleNamespace(n_chars=0 if i < n_scan else 1200, n_rect_rows=50),
        )
        for i in range(n)
    ]
    plan = SimpleNamespace(
        primary_engine="docling", has_vector_diagrams=False, has_formulas=True, has_multicolumn=True
    )
    monkeypatch.setattr(
        "ubt.adapters.pdf.engine_selector.inspect_pdf_route_plan", lambda *a, **k: plan
    )
    monkeypatch.setattr("ubt.adapters.pdf.page_profiler.profile_pdf", lambda *a, **k: profiles)
    monkeypatch.setattr(
        "ubt.adapters.pdf.extraction_witness.inspect_pdf",
        lambda *a, **k: [
            PageVerdict(page=i + 1, confirm_hits=3 if i == 0 else 0, risk_fonts=1, non_ascii=5)
            for i in range(n)
        ],
    )
    monkeypatch.setattr("ubt.adapters.pdf.short_doc.probe_pdf_pages", lambda *a, **k: (n, n * 1000))


def test_pdf_warnings_from_canned_primitives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = tmp_path / "paper.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")  # never opened: all primitives are patched
    _fake_pdf_env(monkeypatch, scanned_share=0.8)
    report = assess_document(doc, _config())
    codes = {w.code for w in report.warnings}
    assert "SCANNED_PAGES_DOMINANT" in codes
    assert "FONT_RESIDUE_RISK" in codes
    assert report.document.text_layer_coverage == pytest.approx(0.2)
    assert report.document.scan_page_share == pytest.approx(0.8)
    # Dominant scans with cloud OCR put per-page vision into the quote.
    cloud = assess_document(doc, _config(ocr_mode="cloud"))
    assert cloud.cost.ocr_page_calls > 0


def test_ocr_and_judge_channels_are_priced_on_their_own_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One rate for both vision channels mis-quoted the one that actually pays.

    The visual judge bills ``visual_judge_model``; the OCR driver bills
    ``ocr_model`` through its own httpx client. Pricing OCR at the judge's rate
    made the quote disagree with the spend in both directions.
    """
    doc = tmp_path / "paper.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")  # never opened: primitives are patched
    _fake_pdf_env(monkeypatch, scanned_share=0.8)

    def ocr_quote(model: str) -> CostQuote:
        return assess_document(
            doc, _config(ocr_mode="vlm", ocr_model=model, visual_judge_enabled=False)
        ).cost

    cheap = ocr_quote("gpt-4o-mini")
    dear = ocr_quote("gpt-4o")
    assert cheap.ocr_page_calls > 0
    assert cheap.vision_cost_usd is not None
    assert dear.vision_cost_usd is not None
    assert cheap.vision_cost_usd < dear.vision_cost_usd

    # An unpriced OCR model makes the vision figure unknown, not zero.
    unpriced = ocr_quote("unheard-of-vision-v9")
    assert unpriced.vision_cost_usd is None
    assert unpriced.money_is_unknown


def test_pathological_pages_warn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = tmp_path / "giant.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    from ubt.adapters.pdf.page_profiler import PageKind

    profiles = [
        SimpleNamespace(
            kind=PageKind.EDITABLE_TEXT,
            facts=SimpleNamespace(n_chars=900, n_rect_rows=99999 if i == 0 else 40),
        )
        for i in range(5)
    ]
    monkeypatch.setattr(
        "ubt.adapters.pdf.engine_selector.inspect_pdf_route_plan",
        lambda *a, **k: SimpleNamespace(
            primary_engine="pdfium",
            has_vector_diagrams=False,
            has_formulas=False,
            has_multicolumn=False,
        ),
    )
    monkeypatch.setattr("ubt.adapters.pdf.page_profiler.profile_pdf", lambda *a, **k: profiles)
    monkeypatch.setattr("ubt.adapters.pdf.extraction_witness.inspect_pdf", lambda *a, **k: [])
    monkeypatch.setattr("ubt.adapters.pdf.short_doc.probe_pdf_pages", lambda *a, **k: (5, 4500))
    report = assess_document(doc, _config())
    assert any(w.code == "PATHOLOGICAL_PAGE_RISK" for w in report.warnings)


def test_deep_mode_uses_exact_block_counts(md_book: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prose = [
        SimpleNamespace(source_text="A perfectly translatable paragraph about testing. " * 3),
        SimpleNamespace(source_text="Another solid paragraph worth drafting exactly once."),
        SimpleNamespace(source_text="SKIPME"),
    ]

    class _FakeAdapter:
        async def extract_manifest(self, path: Path) -> None:
            return None

        async def parse_stream(
            self, path: Path, pages: object = None
        ) -> AsyncIterator[SimpleNamespace]:
            yield SimpleNamespace(blocks=[prose[0]])
            yield SimpleNamespace(blocks=prose[1:])

    monkeypatch.setattr(
        "ubt.adapters.factory.get_adapter_for_path",
        lambda path, pdf_engine="docling": _FakeAdapter(),
    )
    monkeypatch.setattr(
        "ubt.core.cleaners.skip_rules.classify_skip",
        lambda text: "page_number" if text == "SKIPME" else None,
    )
    report = assess_document(md_book, _config(), deep=True)
    assert report.cost.billable_blocks_is_exact is True
    assert report.cost.billable_blocks == 2


@pytest.mark.parametrize("model", sorted(MODEL_PRICES_USD_PER_MTOK)[:1])
def test_priced_models_have_two_tier_money(model: str, md_book: Path) -> None:
    report = assess_document(md_book, _config(draft_model=model, repair_model=model))
    cost = report.cost
    assert cost.draft_cost_usd_cached is not None
    assert cost.draft_cost_usd_uncached is not None
    assert cost.draft_cost_usd_cached <= cost.draft_cost_usd_uncached


def test_real_pdf_smoke() -> None:
    sample = UBT_PKG / "tests" / "fixtures" / "synthetic-duo.pdf"
    if not sample.exists():
        pytest.skip("synthetic sample PDF unavailable")
    report = assess_document(sample, _config())
    assert report.status == "ok"
    assert report.document.pages >= 1
    assert report.document.text_layer_coverage is not None
    assert report.route.recommended_render_engine == "rigid"
    assert report.route.recommended_dual_mode == "monolingual"
    json.dumps(report.to_dict())


def test_empty_file_raises_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(AssessmentError) as exc:
        assess_document(empty, _config())
    assert exc.value.code == "EMPTY_FILE"


def test_macro_chunking_and_batch_api_discount(md_book: Path) -> None:
    single = assess_document(md_book, _config(macro_chunk_size=1, api_mode="chat"))
    batched = assess_document(md_book, _config(macro_chunk_size=5, api_mode="chat"))
    assert batched.cost.prompt_tokens < single.cost.prompt_tokens

    offline_batch = assess_document(
        md_book, _config(offline_batch_enabled=True, api_mode="chat", macro_chunk_size=1)
    )
    assert any(w.code == "BATCH_DISCOUNT_APPLIED" for w in offline_batch.warnings)
    assert offline_batch.cost.draft_cost_usd_uncached is not None
    assert single.cost.draft_cost_usd_uncached is not None
    assert offline_batch.cost.draft_cost_usd_uncached < single.cost.draft_cost_usd_uncached


def test_batch_quote_follows_the_route_that_can_actually_batch(md_book: Path) -> None:
    """A discount for a batch that cannot run under-quotes the job by 2x.

    The engine gates batch submission on ``supports_batch_api`` (chat wire only)
    and falls back to interactive full price, and one request per block cannot
    amortise the prompt prefix the way macro chunking does.
    """
    interactive = assess_document(md_book, _config(macro_chunk_size=1, api_mode="chat"))
    responses = assess_document(
        md_book, _config(offline_batch_enabled=True, api_mode="responses", macro_chunk_size=1)
    )
    assert any(w.code == "BATCH_DISCOUNT_UNAVAILABLE" for w in responses.warnings)
    assert responses.cost.draft_cost_usd_uncached == interactive.cost.draft_cost_usd_uncached

    batched = assess_document(
        md_book, _config(offline_batch_enabled=True, api_mode="chat", macro_chunk_size=5)
    )
    assert batched.cost.prompt_tokens >= interactive.cost.prompt_tokens


def test_deep_mode_filters_formulas_and_images(
    md_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.ir.models import BlockType

    blocks = [
        SimpleNamespace(source_text="Normal paragraph text.", block_type=BlockType.NARRATIVE),
        SimpleNamespace(source_text="E = mc^2", block_type=BlockType.FORMULA),
        SimpleNamespace(source_text="image_bytes", block_type=BlockType.IMAGE),
        SimpleNamespace(source_text="SKIPME", block_type=BlockType.NARRATIVE),
    ]

    class _FakeAdapter:
        async def extract_manifest(self, path: Path) -> None:
            return None

        async def parse_stream(
            self, path: Path, pages: object = None
        ) -> AsyncIterator[SimpleNamespace]:
            yield SimpleNamespace(blocks=blocks)

    monkeypatch.setattr(
        "ubt.adapters.factory.get_adapter_for_path",
        lambda path, pdf_engine="docling": _FakeAdapter(),
    )
    monkeypatch.setattr(
        "ubt.core.cleaners.skip_rules.classify_skip",
        lambda text: "page_number" if text == "SKIPME" else None,
    )
    report = assess_document(md_book, _config(), deep=True)
    assert report.cost.billable_blocks_is_exact is True
    assert report.cost.billable_blocks == 1


def test_cost_quote_rollup_cost_matches_total(md_book: Path, tmp_path: Path) -> None:
    book = tmp_path / "many_chapters.md"
    prose = "Flowing body prose for the chapter. " * 40
    book.write_text(
        "".join(f"# Chapter {i}\n\n{prose}\n\n" for i in range(4)),
        encoding="utf-8",
    )
    report = assess_document(book, _config(exec_mode="long", enable_rolling_summary=True))
    assert report.cost.rollup_calls >= 1
    assert report.cost.rollup_cost_usd is not None
    expected_sum = round(
        (report.cost.draft_cost_usd_uncached or 0.0)
        + (report.cost.repair_cost_usd or 0.0)
        + (report.cost.qe_cost_usd or 0.0)
        + (report.cost.vision_cost_usd or 0.0)
        + (report.cost.rollup_cost_usd or 0.0),
        5,
    )
    assert report.cost.total_cost_usd == pytest.approx(expected_sum)

    # A single-chapter document never rolls up (resolve_draft_policy disables it
    # for <= 1 chapter), so pricing a rollup call there billed nothing.
    one = assess_document(md_book, _config(exec_mode="long", enable_rolling_summary=True))
    assert one.cost.rollup_calls == 0


def test_render_engine_recommendation_and_overlay_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = tmp_path / "formula_dense.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    _fake_pdf_env(monkeypatch, scanned_share=0.0)
    report = assess_document(doc, _config(dual_mode="inline"))
    assert report.route.recommended_render_engine == "rigid"
    assert report.route.recommended_dual_mode == "monolingual"
    assert any(w.code == "OVERLAY_CONFLICT" for w in report.warnings)


def test_visual_and_qe_judge_model_pricing(md_book: Path) -> None:
    report = assess_document(
        md_book,
        _config(
            visual_judge_enabled=True,
            visual_judge_model="gpt-4o",
            qe_judge_enabled=True,
            qe_judge_model="deepseek-reasoner",
            allow_page_upload=True,
        ),
    )
    assert report.cost.vlm_page_calls >= 1
    assert report.cost.vision_cost_usd is not None
    assert report.cost.qe_calls >= 1
    assert report.cost.qe_cost_usd is not None


def test_scanned_pdf_ocr_cost_estimated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = tmp_path / "scanned.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    _fake_pdf_env(monkeypatch, scanned_share=1.0)
    # Mock probe_pdf_pages to return (10 pages, 0 characters)
    monkeypatch.setattr("ubt.adapters.pdf.short_doc.probe_pdf_pages", lambda *a, **k: (10, 0))
    report = assess_document(doc, _config(ocr_mode="rapidocr"))
    assert any(w.code == "SCANNED_PAGE_OCR_ESTIMATED" for w in report.warnings)
    assert report.cost.prompt_tokens > 1000
    assert report.cost.total_cost_usd is not None
    assert report.cost.total_cost_usd > 0.0


def test_encrypted_pdf_guard_raises_instead_of_being_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must not be caught by its own broad ``except Exception``.

    The old nested form raised ``AssessmentError("ENCRYPTED")`` inside an outer
    ``except (ImportError, Exception)``, which swallowed it: a locked PDF
    produced a degraded "route plan unavailable" report instead of a refusal.
    """
    import sys

    from ubt.core import assess as assess_mod

    class _PasswordError(Exception):
        pass

    class _FakePikepdf:
        """Minimal stand-in: ``open`` always reports a locked file."""

        PasswordError = _PasswordError

        @staticmethod
        def open(_path: str) -> object:
            raise _PasswordError("locked")

    monkeypatch.setitem(sys.modules, "pikepdf", _FakePikepdf)

    doc = tmp_path / "secret.pdf"
    doc.write_bytes(b"%PDF-1.4")

    with pytest.raises(AssessmentError) as exc:
        assess_mod._pdf_facts(doc, [])
    assert exc.value.code == "ENCRYPTED"


def test_repair_quote_scales_rerank_only_when_ranking_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rerank_k`` multiplies the repair quote only for a calibrated engine.

    The heuristic runner cannot rank candidates (``is_calibrated() is False``),
    so RepairLoop never fans out for it; a quote that multiplied its repair
    calls by ``rerank_k`` over-stated the line by that factor. For an engine
    that *does* rank, each candidate carries its own source span, so the source
    side scales with k too.
    """
    doc = tmp_path / "book.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    _fake_pdf_env(monkeypatch, scanned_share=0.0)
    monkeypatch.setattr("ubt.adapters.pdf.short_doc.probe_pdf_pages", lambda *a, **k: (10, 12000))

    heuristic_k1 = assess_document(
        doc, _config(qe_engine="heuristic", rerank_k=1, max_repair_rounds=2)
    )
    heuristic_k5 = assess_document(
        doc, _config(qe_engine="heuristic", rerank_k=5, max_repair_rounds=2)
    )
    comet_k5 = assess_document(doc, _config(qe_engine="comet", rerank_k=5, max_repair_rounds=2))

    assert heuristic_k1.cost.repair_cost_usd is not None
    assert comet_k5.cost.repair_cost_usd is not None
    # Heuristic ignores rerank_k; a calibrated engine's repair line grows with it.
    assert heuristic_k5.cost.repair_cost_usd == heuristic_k1.cost.repair_cost_usd
    assert comet_k5.cost.repair_cost_usd > heuristic_k5.cost.repair_cost_usd


def test_page_slice_quote_is_labelled_as_the_whole_document(tmp_path: Path) -> None:
    """``--pages 1-1`` drafts one page; the quote still covers the whole document.

    Scale the figure or say so — a bare number next to a sliced job reads as the
    cost of that slice and is wrong by the page ratio.
    """
    doc = tmp_path / "slice.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    whole = assess_document(doc, _config())
    sliced = assess_document(doc, _config(pages="1-1"))
    assert any(w.code == "PAGES_SLICE_QUOTED_WHOLE" for w in sliced.warnings)
    assert not any(w.code == "PAGES_SLICE_QUOTED_WHOLE" for w in whole.warnings)


def test_rollup_pricing_honours_the_recommended_profile() -> None:
    """An academic profile writes no rolling summaries, so pricing them bills air.

    The guard used to read ``config.profile`` -- a field UBTConfig does not have
    -- so it evaluated to "general" forever and the clause could never fire.
    """
    from ubt.core import assess
    from ubt.core.assess import _build_cost

    config = _config(exec_mode="long", enable_rolling_summary=True)

    def quote(profile_name: str) -> assess.CostQuote:
        return _build_cost(
            config,
            billable_blocks=50,
            source_chars=40_000,
            prefix=1000,
            is_exact=False,
            pages=20,
            scan_pages=0,
            chapters=6,
            route_mode="long",
            profile_name=profile_name,
            warnings=[],
        )

    assert quote("general").rollup_calls == 6
    academic = quote("textbook")
    assert academic.rollup_calls == 0
    assert academic.rollup_cost_usd is None


@pytest.mark.asyncio
async def test_assess_document_respects_language_pair_for_token_estimation(tmp_path: Path) -> None:
    doc = tmp_path / "book.md"
    doc.write_text("# Title\n\n" + "This is a sentence for translation. " * 50, encoding="utf-8")
    cfg = UBTConfig.from_env()

    report_en_zh = await assess_document_async(doc, cfg, source_lang="en", target_lang="zh")
    report_zh_en = await assess_document_async(doc, cfg, source_lang="zh", target_lang="en")

    # en -> zh ratio is 1.25, zh -> en ratio is 0.85
    # Completion tokens must be distinct and reflect the language pair
    assert report_en_zh.cost.completion_tokens > report_zh_en.cost.completion_tokens
