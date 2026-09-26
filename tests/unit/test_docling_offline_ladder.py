"""L4: the offline retry must not repoint Hugging Face at a mirror unasked.

``extract_with_docling``'s "model not cached, go online" branch used to
``os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`` silently —
an egress decision (the model downloads of an unpublished manuscript go to a
third-party host) that the operator never made. It now follows the
``allow_page_upload`` posture: nothing switches without
``UBT_ALLOW_HF_MIRROR=1``, the warning says how to opt in, and the ``finally``
restore still keeps whatever did change from outliving the ladder.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytest

from ubt.adapters.pdf import docling_parser


def _run_offline_ladder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str | None]:
    """Run the conversion retry ladder with an always-offline converter.

    Returns the ``HF_ENDPOINT`` each ``convert()`` call observed, so the test
    can assert what the *retry* saw — the final restore would hide it
    otherwise.
    """
    monkeypatch.setattr(docling_parser, "configure_hf_environment", lambda **_k: None)
    monkeypatch.setattr(docling_parser, "read_docling_document", lambda *_a, **_k: None)

    seen_endpoints: list[str | None] = []

    class OfflineModeError(Exception):
        pass

    class _Options:
        do_formula_enrichment = False

        def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
            return {}

    class _Converter:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def convert(self, _path: Any, **_kwargs: Any) -> Any:
            seen_endpoints.append(os.environ.get("HF_ENDPOINT"))
            raise OfflineModeError("offline mode: model is not cached")

    class _InputFormat:
        PDF = "pdf"

    def symbols() -> tuple[Any, Any, Any, Any]:
        return (_InputFormat, _Options, _Converter, lambda **_k: object())

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(OfflineModeError):
        docling_parser.extract_with_docling(pdf, None, symbols=symbols, enrich=False)
    return seen_endpoints


def _clear_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HF_ENDPOINT",
        "UBT_ALLOW_HF_MIRROR",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_offline_retry_keeps_the_official_endpoint_without_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with caplog.at_level(logging.WARNING, logger="ubt.adapters.pdf.docling_parser"):
        seen = _run_offline_ladder(monkeypatch, tmp_path)

    # The retry ran (both convert() calls happened)...
    assert len(seen) == 2
    # ...but the second one still pointed at the official endpoint.
    assert seen[-1] is None, "HF_ENDPOINT switched without UBT_ALLOW_HF_MIRROR=1"
    # ...and the operator was told how to opt in instead.
    assert "UBT_ALLOW_HF_MIRROR=1" in caplog.text
    # The ladder's finally restored everything it saw.
    assert "HF_ENDPOINT" not in os.environ
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_offline_retry_uses_the_mirror_only_with_explicit_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("UBT_ALLOW_HF_MIRROR", "1")

    seen = _run_offline_ladder(monkeypatch, tmp_path)

    assert seen[-1] == "https://hf-mirror.com", "opt-in must actually switch the mirror"
    # The mirror must not outlive the ladder: later HF downloads in this
    # process (the DeepSeek worker's snapshot_download included) would
    # silently inherit it.
    assert "HF_ENDPOINT" not in os.environ
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
