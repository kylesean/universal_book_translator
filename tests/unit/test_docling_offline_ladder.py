"""``extract_with_docling``'s offline→online retry ladder.

Two behaviours live here:

1. ``HF_ENDPOINT`` must not be repointed at the third-party ``hf-mirror.com``
   unasked — an egress decision (an unpublished manuscript's model downloads go
   to a host the operator never chose) that follows the ``allow_page_upload``
   posture: nothing switches without ``UBT_ALLOW_HF_MIRROR=1``.
2. A *successful* retry must keep the document it produced. The stale offline
   exception used to fall through to the ``do_formula_enrichment`` fallback,
   discarding the enriched result and re-converting with the VLM disabled.
"""

from __future__ import annotations

import contextlib
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


def test_configure_hf_environment_needs_opt_in_before_mirroring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The setup step must obey the same opt-in as the retry ladder.

    ``configure_hf_environment`` used to ``setdefault`` the mirror for uncached
    models with no opt-in, outside the ladder's try/finally — so it persisted
    process-wide for every later download, contradicting the policy the retry
    branch documents and tests.
    """
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))

    docling_parser.configure_hf_environment()

    assert "HF_ENDPOINT" not in os.environ


def test_configure_hf_environment_mirrors_only_with_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    monkeypatch.setenv("UBT_ALLOW_HF_MIRROR", "1")

    docling_parser.configure_hf_environment()

    assert os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com"


def _run_retry_with_enrichment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[bool]:
    """Retry ladder where the first (offline) convert fails and the second succeeds.

    Returns the ``do_formula_enrichment`` value each ``convert()`` observed.
    """
    monkeypatch.setattr(docling_parser, "configure_hf_environment", lambda **_k: None)
    monkeypatch.setattr(docling_parser, "read_docling_document", lambda *_a, **_k: None)
    monkeypatch.setattr(docling_parser, "write_docling_document", lambda *_a, **_k: None)

    calls: list[bool] = []
    holder: dict[str, Any] = {}
    state = {"n": 0}

    class OfflineModeError(Exception):
        pass

    class _Options:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.do_formula_enrichment = True
            holder["opt"] = self

        def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
            return {}

    class _Converter:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def convert(self, _path: Any, **_kwargs: Any) -> Any:
            state["n"] += 1
            calls.append(bool(holder["opt"].do_formula_enrichment))
            if state["n"] == 1:
                raise OfflineModeError("offline mode: model is not cached")
            return type("_Result", (), {"document": object()})()

    class _InputFormat:
        PDF = "pdf"

    def symbols() -> tuple[Any, Any, Any, Any]:
        return (_InputFormat, _Options, _Converter, lambda **_k: object())

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    # Extraction past the ladder cannot consume the fake document; the ladder
    # log is the observable under test, so any downstream failure is ignored.
    with contextlib.suppress(Exception):
        docling_parser.extract_with_docling(pdf, None, symbols=symbols, enrich=True)
    return calls


def test_successful_offline_retry_is_not_downgraded_to_no_vlm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    calls = _run_retry_with_enrichment(monkeypatch, tmp_path)

    assert len(calls) == 2, f"successful retry was discarded and re-converted: {calls}"
    assert all(calls), f"enrichment silently disabled after a successful retry: {calls}"
