from __future__ import annotations

from typing import Any

import pytest

from ubt.core.engine.pipeline import engine_signature

pytestmark = pytest.mark.fast


def _config(**overrides: Any) -> Any:
    from ubt.core.config import UBTConfig

    return UBTConfig(**overrides)


def test_the_default_config_adds_no_suffix() -> None:
    assert engine_signature(_config()) == ""


def test_a_degraded_ingest_names_the_engine_that_ran() -> None:
    assert engine_signature(_config(), pdf_engine_fallback="oxide_fallback") == ("fboxidefallback")


def test_a_healthy_run_adds_no_fallback_token() -> None:
    assert "fb" not in engine_signature(_config(), pdf_engine_fallback=None)


def test_a_normal_pdfium_run_is_unchanged() -> None:
    assert engine_signature(_config(pdf_engine="pdfium"), pdf_engine_fallback=None) == ""


def test_the_fallback_token_composes_with_other_engine_knobs() -> None:
    sig = engine_signature(_config(draft_model="some-model"), pdf_engine_fallback="oxide_fallback")
    assert "dmsomemodel" in sig.replace("-", "")
    assert sig.endswith("fboxidefallback"), (
        "the fallback token must be present; a different engine knob must not crowd it out"
    )


def test_derive_job_id_namespaces_the_fallback() -> None:
    from ubt.core.engine.pipeline import derive_job_id

    plain = derive_job_id(
        doc_id="abc123", target_lang="zh", pages=None, start_chapter=1, max_chapters=None
    )
    degraded = derive_job_id(
        doc_id="abc123",
        target_lang="zh",
        pages=None,
        start_chapter=1,
        max_chapters=None,
        engine_signature=engine_signature(_config(), pdf_engine_fallback="oxide_fallback"),
    )
    assert plain != degraded
    assert "fb" in degraded


def test_the_ingest_identity_uses_the_same_fallback_rule() -> None:
    from ubt.core.engine.pipeline import degraded_pdf_engine

    metadata = {"pdf_parser_engine": "oxide_fallback"}
    assert degraded_pdf_engine(_config(), metadata) == "oxide_fallback"
    sig = engine_signature(_config(), pdf_engine_fallback=degraded_pdf_engine(_config(), metadata))
    assert sig == "fboxidefallback"


def test_a_deliberate_pdfium_choice_is_not_a_degradation() -> None:
    from ubt.core.engine.pipeline import degraded_pdf_engine

    metadata = {"pdf_parser_engine": "pdfium"}
    assert degraded_pdf_engine(_config(pdf_engine="pdfium"), metadata) is None


def test_a_full_docling_run_is_not_a_degradation() -> None:
    from ubt.core.engine.pipeline import degraded_pdf_engine

    metadata = {"pdf_parser_engine": "docling"}
    assert degraded_pdf_engine(_config(pdf_engine="auto"), metadata) is None


def test_an_empty_manifest_names_no_engine() -> None:
    from ubt.core.engine.pipeline import degraded_pdf_engine

    assert degraded_pdf_engine(_config(pdf_engine="auto"), {}) is None
    assert degraded_pdf_engine(_config(pdf_engine="auto"), None) is None
