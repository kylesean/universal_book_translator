"""Tests for the formula VLM crop-margin knob (``UBT_FORMULA_CROP_MARGIN``).

Docling crops each formula with an ``expansion_factor`` margin before the
CodeFormulaV2 VLM reads it; its default of 0.18 reaches into the equation
number and the neighbouring prose line on a tight academic layout, so the model
transcribes that text into the LaTeX (docling#4548). The parser lowers the
margin and enables Docling's repetition guard through Docling's own preset API.

These tests use a stub ``symbols`` hook so they assert *our* wiring — which
preset, which kwargs, and that nothing is set when enrichment is off — without
depending on the installed Docling exposing the newer preset kwargs.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import pytest

import ubt.adapters.pdf.docling_parser as dp

pytestmark = pytest.mark.fast


class _StubCodeFormulaOptions:
    """Stand-in for ``docling``'s ``CodeFormulaVlmOptions`` preset factory."""

    calls: list[tuple[str, dict[str, Any]]] = []
    preset_id: str
    overrides: dict[str, Any]

    @classmethod
    def from_preset(cls, preset_id: str, **overrides: Any) -> _StubCodeFormulaOptions:
        cls.calls.append((preset_id, overrides))
        self = cls()
        self.preset_id = preset_id
        self.overrides = overrides
        return self


class _StubPipelineOptions:
    """Minimal ``PdfPipelineOptions`` shape the parser assigns onto."""

    def __init__(self) -> None:
        self.accelerator_options = type("Accel", (), {"device": "auto"})()
        self.code_formula_options: Any = None


def _stub_symbols() -> tuple[Any, Any, Any, Any, Any]:
    class _Format:
        PDF = "pdf"

    class _FormatOption:
        def __init__(self, pipeline_options: Any = None) -> None:
            self.pipeline_options = pipeline_options

    class _Converter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    _StubCodeFormulaOptions.calls = []
    return (
        type("InputFormat", (), {"PDF": _Format.PDF}),
        _StubPipelineOptions,
        _Converter,
        _FormatOption,
        _StubCodeFormulaOptions,
    )


def _run_extract(tmp_path: Path, *, enrich: bool, margin: float = 0.08) -> _StubPipelineOptions:
    """Drive ``extract_with_docling`` far enough to inspect the options it built."""
    pdf = tmp_path / "page.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")  # content never parsed by the stub

    captured: dict[str, _StubPipelineOptions] = {}
    symbols = _stub_symbols()
    real_pipeline_cls = symbols[1]
    real_converter_cls = symbols[2]

    class _CapturingConverter(real_converter_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            for value in (kwargs.get("format_options") or {}).values():
                captured["options"] = value.pipeline_options

    symbols = (symbols[0], real_pipeline_cls, _CapturingConverter, symbols[3], symbols[4])
    # The stub conversion fails once it is past the options we assert on; the
    # build itself (which happens before conversion) is what matters here.
    with contextlib.suppress(Exception):
        dp.extract_with_docling(
            pdf, None, symbols=lambda: symbols, enrich=enrich, formula_crop_margin=margin
        )
    stub = captured.get("options")
    assert stub is not None, "the parser must build pipeline options before converting"
    return stub


def test_enrich_sets_lowered_crop_margin_and_repetition_guard(tmp_path: Path) -> None:
    options = _run_extract(tmp_path, enrich=True, margin=0.08)
    assert options.code_formula_options is not None
    preset_id, overrides = _StubCodeFormulaOptions.calls[-1]
    assert preset_id == "codeformulav2"
    # The margin must reach Docling, not silently keep its 0.18 default.
    assert overrides["expansion_factor"] == 0.08
    # Repetition guard rides along: a tighter crop can otherwise loop.
    assert overrides["stop_on_repetition"] is True


def test_margin_is_tunable_not_hardcoded(tmp_path: Path) -> None:
    _run_extract(tmp_path, enrich=True, margin=0.05)
    _preset_id, overrides = _StubCodeFormulaOptions.calls[-1]
    assert overrides["expansion_factor"] == 0.05


def test_no_crop_options_when_enrichment_is_off(tmp_path: Path) -> None:
    options = _run_extract(tmp_path, enrich=False)
    assert options.code_formula_options is None
    assert _StubCodeFormulaOptions.calls == []


def test_crop_margin_reaches_the_adapter() -> None:
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
    from ubt.core.ports import AdapterRuntimeConfig

    adapter = DoclingPDFAdapter()
    adapter.apply_config(
        AdapterRuntimeConfig(
            ocr_mode="off",
            ocr_endpoint=None,
            ocr_api_key="",
            ocr_model="",
            formula_enrichment="on",
            docling_device="cpu",
            formula_crop_margin=0.07,
            toc_mode="auto",
            font_family=None,
            allow_page_upload=False,
        )
    )
    assert adapter.formula_crop_margin == 0.07


def test_default_crop_margin_config_value() -> None:
    from ubt.core.config import UBTConfig

    os.environ.pop("UBT_FORMULA_CROP_MARGIN", None)
    assert UBTConfig.from_env().formula_crop_margin == 0.08


def test_crop_margin_reads_env() -> None:
    from unittest.mock import patch

    from ubt.core.config import UBTConfig

    with patch.dict(os.environ, {"UBT_FORMULA_CROP_MARGIN": "0.12"}):
        assert UBTConfig.from_env().formula_crop_margin == 0.12
