from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.core.config import UBTConfig
from ubt.core.ports import AdapterRuntimeConfig

pytestmark = pytest.mark.fast


class _ConverterBuilt(RuntimeError):
    pass


def test_default_docling_device_is_auto() -> None:
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
        toc_mode="auto",
        font_family=None,
        allow_page_upload=False,
    )
    adapter.apply_config(runtime)
    assert adapter.docling_device == "cpu"


def test_device_reaches_extract_with_docling() -> None:
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
    pytest.importorskip("docling", reason="needs the `pdf` or `academic` extra")
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
        format_options = kwargs.get("format_options") or (args[0] if args else None)
        if isinstance(format_options, dict):
            for value in format_options.values():
                options = getattr(value, "pipeline_options", None)
                if options is not None:
                    seen["device"] = options.accelerator_options.device
        raise _ConverterBuilt

    converter_cls.__init__ = spy_init
    try:
        with pytest.raises(_ConverterBuilt):
            dp.extract_with_docling(
                pdf, None, symbols=da._docling_symbols, enrich=False, device="cpu"
            )
    finally:
        converter_cls.__init__ = original_init

    assert seen.get("device") == "cpu"


def test_a_crowded_card_is_detected_before_the_converter_is_built() -> None:
    from ubt.adapters.pdf.docling_parser import _docling_vram_verdict

    with patch("ubt.adapters.pdf.docling_parser._free_vram_gib", return_value=0.06):
        device, note = _docling_vram_verdict("auto")
    assert device == "cpu"
    assert note is not None
    assert "0.06 GiB" in note


def test_a_roomy_card_keeps_docling_on_its_own_probe() -> None:
    from ubt.adapters.pdf.docling_parser import _docling_vram_verdict

    with patch("ubt.adapters.pdf.docling_parser._free_vram_gib", return_value=6.0):
        assert _docling_vram_verdict("auto") == ("auto", None)


@pytest.mark.parametrize("pinned", ["cpu", "cuda"])
def test_an_explicit_device_is_never_overridden(pinned: str) -> None:
    from ubt.adapters.pdf.docling_parser import _docling_vram_verdict

    with patch("ubt.adapters.pdf.docling_parser._free_vram_gib", return_value=0.01):
        assert _docling_vram_verdict(pinned) == (pinned, None)


def test_no_cuda_device_leaves_the_request_alone() -> None:
    from ubt.adapters.pdf.docling_parser import _docling_vram_verdict

    with patch("ubt.adapters.pdf.docling_parser._free_vram_gib", return_value=None):
        assert _docling_vram_verdict("auto") == ("auto", None)


def test_the_note_names_the_processes_holding_the_card() -> None:
    from ubt.adapters.pdf.docling_parser import _vram_hint

    fake = SimpleNamespace(
        stdout="431065, /usr/local/bin/llama-server, 6616 MiB\n430649, python3, 748 MiB\n",
        returncode=0,
    )
    with patch("ubt.adapters.pdf.docling_parser.subprocess.run", return_value=fake):
        hint = _vram_hint()
    assert "llama-server" in hint
    assert "6616 MiB" in hint


def test_an_unavailable_nvidia_smi_is_not_fatal() -> None:
    from ubt.adapters.pdf.docling_parser import _vram_hint

    with patch("ubt.adapters.pdf.docling_parser.subprocess.run", side_effect=OSError):
        assert _vram_hint() == ""
