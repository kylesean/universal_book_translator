"""Tests for the Docling accelerator-device knob (``UBT_DOCLING_DEVICE``).

Whether Docling *runs* and whether it touches the *GPU* are separate decisions.
A local LLM sharing one card fails to load its weights when Docling holds a
~1.6 GB CUDA context, so ``UBT_DOCLING_DEVICE=cpu`` must keep Docling off the
card without changing which engine was chosen.

See docs/guides/PDF_ENGINE_ROUTING_REDESIGN.md §L4.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.core.config import UBTConfig
from ubt.core.ports import AdapterRuntimeConfig

pytestmark = pytest.mark.fast


def test_default_docling_device_is_auto() -> None:
    # ``auto`` keeps Docling's own probe, i.e. existing behaviour.
    os.environ.pop("UBT_DOCLING_DEVICE", None)
    assert UBTConfig.from_env().docling_device == "auto"


def test_docling_device_reads_env() -> None:
    with patch.dict(os.environ, {"UBT_DOCLING_DEVICE": "cpu"}):
        assert UBTConfig.from_env().docling_device == "cpu"
    with patch.dict(os.environ, {"UBT_DOCLING_DEVICE": "cuda"}):
        assert UBTConfig.from_env().docling_device == "cuda"


def test_docling_device_rejects_unknown_value() -> None:
    from pydantic import ValidationError

    with patch.dict(os.environ, {"UBT_DOCLING_DEVICE": "bogus"}), pytest.raises(ValidationError):
        UBTConfig.from_env()


def test_runtime_config_carries_device_to_adapter() -> None:
    adapter = DoclingPDFAdapter()
    runtime = AdapterRuntimeConfig(
        ocr_mode="auto",
        ocr_endpoint=None,
        ocr_api_key="",
        ocr_model="",
        formula_enrichment="auto",
        docling_device="cpu",
        formula_crop_margin=0.08,
        font_family=None,
        allow_page_upload=False,
    )
    adapter.apply_config(runtime)
    assert adapter.docling_device == "cpu"


def test_device_reaches_extract_with_docling() -> None:
    # The adapter must forward the resolved device, not a hardcoded default.
    import ubt.adapters.pdf.docling_adapter as da

    captured: dict[str, object] = {}

    def spy(*args: object, **kwargs: object) -> None:
        captured.update(kwargs)
        raise RuntimeError("stop before the real Docling conversion")

    adapter = DoclingPDFAdapter()
    adapter.docling_device = "cpu"
    with (
        patch.object(da, "extract_with_docling", spy),
        pytest.raises(RuntimeError, match="stop before"),
    ):
        adapter._extract_with_docling(Path("/tmp/does-not-matter.pdf"), None)
    assert captured.get("device") == "cpu"


def test_cpu_device_pins_docling_pipeline_options(tmp_path: Path) -> None:
    # End-to-end through the parser: the device must land on the Docling
    # pipeline options the converter is built from. ``auto`` leaves Docling's
    # own probe in charge (no assignment).
    from pdf_builders import write_text_pdf

    import ubt.adapters.pdf.docling_adapter as da
    import ubt.adapters.pdf.docling_parser as dp

    pdf = write_text_pdf(
        tmp_path / "page.pdf", [["Hello world test page content here"] * 3 for _ in range(3)]
    )
    symbols = da._docling_symbols()
    _input_format, _options_cls, converter_cls, _format_cls, _code_formula_cls = symbols
    seen: dict[str, object] = {}
    original_init = converter_cls.__init__

    def spy_init(self: object, *args: object, **kwargs: object) -> None:
        # ``format_options`` is Docling's own mapping; ``isinstance`` narrows the
        # ``object`` a ``**kwargs`` lookup yields so this stays typed without an
        # ignore (the symbols hook is deliberately ``Any``-typed).
        format_options = kwargs.get("format_options") or (args[0] if args else None)
        if isinstance(format_options, dict):
            for value in format_options.values():
                options = getattr(value, "pipeline_options", None)
                if options is not None:
                    seen["device"] = options.accelerator_options.device
        original_init(self, *args, **kwargs)

    converter_cls.__init__ = spy_init
    try:
        dp.extract_with_docling(pdf, None, symbols=da._docling_symbols, enrich=False, device="cpu")
    except Exception:  # noqa: BLE001 - conversion may fail; the device is the assertion
        pass
    finally:
        converter_cls.__init__ = original_init

    assert seen.get("device") == "cpu"
