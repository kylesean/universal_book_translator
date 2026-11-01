from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ubt.adapters.pdf.visual_gate import VisualGateResult
from ubt.core.engine.visual_verify_loop import VisualVerifyLoop, _typst_source_path
from ubt.core.ir.models import BookManifest

pytestmark = pytest.mark.fast

_CANVAS_TYP = (
    "// Universal Book Translator - 1:1 Facsimile Canvas Document\n"
    "#page(width: 540.0pt, height: 666.0pt, margin: 0pt, "
    'background: image("bg_stripped.pdf", page: 1, width: 100%, height: 100%))[\n'
    "]\n"
)


@dataclass
class _QuietLedger:
    visual_reports: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def save_checkpoints_batch(self, checkpoints: Sequence[dict[str, Any]]) -> None:
        pass

    def record_visual_report(self, job_id: str, payload: dict[str, Any]) -> None:
        self.visual_reports.append((job_id, payload))

    def get_job_metadata_value(self, job_id: str, key: str) -> Any:
        return None


def _loop(tmp_path: Path, manifest: BookManifest) -> VisualVerifyLoop:
    return VisualVerifyLoop(
        adapter=Any,  # type: ignore[arg-type]
        manifest=manifest,
        ledger=_QuietLedger(),  # type: ignore[arg-type]
        job_id="job-canvas",
        target_lang="zh",
    )


def _manifest(tmp_path: Path, *, background: Path | None) -> BookManifest:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    manifest = BookManifest(
        doc_id="doc",
        title="Book",
        source_path=str(source),
        chapters=[],
    )
    if background is not None:
        manifest.metadata["background_pdf"] = str(background)
    return manifest


def test_the_recorded_background_reaches_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    background = tmp_path / "bg_stripped.pdf"
    background.write_bytes(b"%PDF-1.4\n%%EOF\n")
    manifest = _manifest(tmp_path, background=background)
    loop = _loop(tmp_path, manifest)

    assert loop._canvas_background_path() == background

    captured: dict[str, Any] = {}

    async def _runner(*_args: Any, **kwargs: Any) -> VisualGateResult:
        captured.update(kwargs)
        return VisualGateResult(passed=True)

    import ubt.core.engine.visual_verify_loop as loop_module

    monkeypatch.setattr(loop_module, "get_visual_gate_runner", lambda: _runner)
    asyncio.run(loop._evaluate_gate(tmp_path / "artifact.pdf", []))

    assert captured["canvas_background_pdf"] == background


def test_no_recorded_background_passes_none(tmp_path: Path) -> None:
    loop = _loop(tmp_path, _manifest(tmp_path, background=None))
    assert loop._canvas_background_path() is None


def test_a_recorded_background_that_no_longer_exists_passes_none(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, background=tmp_path / "gone_bg_stripped.pdf")
    assert _loop(tmp_path, manifest)._canvas_background_path() is None


def test_a_manifest_without_a_metadata_mapping_is_tolerated(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, background=None)
    manifest.metadata = None  # type: ignore[assignment]
    assert _loop(tmp_path, manifest)._canvas_background_path() is None


def test_a_directory_is_not_a_background(tmp_path: Path) -> None:
    directory = tmp_path / "bg_dir.pdf"
    directory.mkdir()
    manifest = _manifest(tmp_path, background=directory)
    assert _loop(tmp_path, manifest)._canvas_background_path() is None


def test_the_facsimile_source_is_found(tmp_path: Path) -> None:
    artifact = tmp_path / "book.pdf"
    artifact.write_bytes(b"%PDF-1.4\n%%EOF\n")
    build = tmp_path / ".build_book_facsimile"
    build.mkdir()
    typ = build / "facsimile.typ"
    typ.write_text(_CANVAS_TYP, encoding="utf-8")

    assert _typst_source_path(artifact) == typ


def test_the_publication_source_is_still_found(tmp_path: Path) -> None:
    artifact = tmp_path / "book.pdf"
    artifact.write_bytes(b"%PDF-1.4\n%%EOF\n")
    build = tmp_path / ".book_typst"
    build.mkdir()
    typ = build / "main.typ"
    typ.write_text("#set text(size: 10pt)\n", encoding="utf-8")

    assert _typst_source_path(artifact) == typ


def test_a_typ_sibling_is_still_found(tmp_path: Path) -> None:
    artifact = tmp_path / "book.pdf"
    artifact.write_bytes(b"%PDF-1.4\n%%EOF\n")
    typ = tmp_path / "book.typ"
    typ.write_text("#set text(size: 10pt)\n", encoding="utf-8")

    assert _typst_source_path(artifact) == typ


def test_the_newest_source_wins_when_a_stale_one_is_left_behind(tmp_path: Path) -> None:
    import os
    import time

    artifact = tmp_path / "book.pdf"
    artifact.write_bytes(b"%PDF-1.4\n%%EOF\n")
    stale = tmp_path / ".book_typst"
    stale.mkdir()
    (stale / "main.typ").write_text("#set text(size: 9pt)\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(stale / "main.typ", (old, old))

    build = tmp_path / ".build_book_facsimile"
    build.mkdir()
    typ = build / "facsimile.typ"
    typ.write_text(_CANVAS_TYP, encoding="utf-8")

    assert _typst_source_path(artifact) == typ


def test_no_source_at_all_is_none_rather_than_an_error(tmp_path: Path) -> None:
    artifact = tmp_path / "book.pdf"
    artifact.write_bytes(b"%PDF-1.4\n%%EOF\n")

    assert _typst_source_path(artifact) is None
