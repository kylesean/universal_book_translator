"""Unit tests for formula enrichment auto-strategy and font family configuration."""

from unittest.mock import patch

import pytest

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter
from ubt.core.config import UBTConfig

pytestmark = pytest.mark.fast


def test_formula_enrichment_mode_off() -> None:
    adapter = DoclingPDFAdapter(formula_enrichment="off", render_engine="publication")
    assert adapter._resolve_formula_enrichment() is False


def test_formula_enrichment_mode_on() -> None:
    adapter = DoclingPDFAdapter(formula_enrichment="on", render_engine="rigid")
    assert adapter._resolve_formula_enrichment() is True


def test_formula_enrichment_auto_anchored_and_overlay() -> None:
    with patch("ubt.adapters.pdf.docling_adapter._has_accelerator", return_value=True):
        adapter_anchored = DoclingPDFAdapter(formula_enrichment="auto", render_engine="rigid")
        assert adapter_anchored._resolve_formula_enrichment() is False

        adapter_overlay = DoclingPDFAdapter(formula_enrichment="auto", render_engine="rigid")
        assert adapter_overlay._resolve_formula_enrichment() is False


def test_formula_enrichment_auto_image_render() -> None:
    with patch("ubt.adapters.pdf.docling_adapter._has_accelerator", return_value=True):
        adapter = DoclingPDFAdapter(
            formula_enrichment="auto",
            render_engine="publication",
            formula_render="image",
        )
        assert adapter._resolve_formula_enrichment() is False


def test_formula_enrichment_auto_publication_gpu_vs_cpu() -> None:
    adapter = DoclingPDFAdapter(
        formula_enrichment="auto",
        render_engine="publication",
        formula_render="witness",
    )
    with patch("ubt.adapters.pdf.docling_adapter._has_accelerator", return_value=True):
        assert adapter._resolve_formula_enrichment() is True

    with patch("ubt.adapters.pdf.docling_adapter._has_accelerator", return_value=False):
        assert adapter._resolve_formula_enrichment() is False


def test_anchored_typesetter_font_family_default_serif() -> None:
    typesetter = RigidTypesetter(target_lang="zh")
    assert typesetter.font_family == "Noto Serif CJK SC"

    custom_typesetter = RigidTypesetter(font_family="Noto Sans CJK SC", target_lang="zh")
    assert custom_typesetter.font_family == "Noto Sans CJK SC"


def test_ubt_config_overlay_and_reflow_engines() -> None:
    """The `rigid` name must survive config untouched (no silent re-mapping).

    The retired-engine migration in `UBTConfig` rewrites old stored values; this
    pins that it leaves the live names alone. The accepted-values themselves are
    owned by test_render_preflight.py, which is where the engine Literal lives.
    """
    cfg_overlay = UBTConfig(render_engine="rigid", font_family="Noto Serif CJK SC")
    assert cfg_overlay.render_engine == "rigid"

    cfg_reflow = UBTConfig(render_engine="reflow")
    assert cfg_reflow.render_engine == "reflow"
