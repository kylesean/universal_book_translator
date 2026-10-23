"""A degraded PDF ingest gets its own ledger namespace.

``auto`` selects docling; when the package is missing the adapter falls back to
the geometric pdfium reader, which cannot represent tables, figures or formulas.
The blocks that fallback produced were filed under the *same* job id as a full
docling ingest would use, so installing the extra later resumed the degraded
ledger and skipped parsing — the loss was permanent and silent.

The degraded engine now contributes an ``fb<engine>`` token to the job id. A
normal run (including an explicit ``--pdf-engine pdfium``) adds nothing, so
existing ledgers stay resumable.
"""

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
    """The token is normalized like every other signature part: alnum only."""
    assert engine_signature(_config(), pdf_engine_fallback="oxide_fallback") == ("fboxidefallback")


def test_a_healthy_run_adds_no_fallback_token() -> None:
    """Only an actual degradation namespaces; a full docling run must not."""
    assert "fb" not in engine_signature(_config(), pdf_engine_fallback=None)


def test_a_normal_pdfium_run_is_unchanged() -> None:
    """An explicit fast-path choice is not a degradation."""
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
    """The resume guard compares the same signature the job id was built from.

    Reading only the config made the guard disagree with the derived id and
    refuse a legitimate resume of a degraded job. Both call sites now share
    ``degraded_pdf_engine``, so this pins the rule rather than a copy of it.
    """
    from ubt.core.engine.pipeline import degraded_pdf_engine

    metadata = {"pdf_parser_engine": "oxide_fallback"}
    assert degraded_pdf_engine(_config(), metadata) == "oxide_fallback"
    sig = engine_signature(_config(), pdf_engine_fallback=degraded_pdf_engine(_config(), metadata))
    assert sig == "fboxidefallback"


def test_a_deliberate_pdfium_choice_is_not_a_degradation() -> None:
    """``--pdf-engine pdfium`` is a choice; the manifest still says pdfium."""
    from ubt.core.engine.pipeline import degraded_pdf_engine

    metadata = {"pdf_parser_engine": "pdfium"}
    assert degraded_pdf_engine(_config(pdf_engine="pdfium"), metadata) is None


def test_a_full_docling_run_is_not_a_degradation() -> None:
    from ubt.core.engine.pipeline import degraded_pdf_engine

    metadata = {"pdf_parser_engine": "docling"}
    assert degraded_pdf_engine(_config(pdf_engine="auto"), metadata) is None


def test_an_empty_manifest_names_no_engine() -> None:
    """A non-PDF run records nothing; the rule must not invent a fallback."""
    from ubt.core.engine.pipeline import degraded_pdf_engine

    assert degraded_pdf_engine(_config(pdf_engine="auto"), {}) is None
    assert degraded_pdf_engine(_config(pdf_engine="auto"), None) is None
