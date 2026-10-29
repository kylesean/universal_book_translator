"""The canvas background reaches the visual gate through the manifest.

``VisualVerifyLoop`` runs a whole stage after the render, so it holds the
manifest and nothing from the render call. The facsimile renderer records where
it wrote the stripped background; this loop reads it back and hands it to
``run_visual_gate``. That thread is the only thing connecting the L1 channel to
the one tier that can see the L1 channel's own failure mode.

These tests pin the thread itself, because every failure of it is silent:
``run_visual_gate`` treats a missing background as "the tier did not run", which
is a note in the report and a green verdict. A typo in the metadata key, a
renamed build directory, or a gate runner that drops the kwarg would all read
exactly like a book with no ink collisions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ubt.adapters.pdf.visual_gate import VisualGateResult
from ubt.core.engine.visual_verify_loop import VisualVerifyLoop
from ubt.core.ir.models import BookManifest

pytestmark = pytest.mark.fast


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


def test_the_recorded_background_reaches_the_gate(tmp_path: Path) -> None:
    background = tmp_path / "bg_stripped.pdf"
    background.write_bytes(b"%PDF-1.4\n%%EOF\n")
    manifest = _manifest(tmp_path, background=background)
    loop = _loop(tmp_path, manifest)

    # The reader is a method on the loop, not on the runner: assert on it first
    # so a failure names the actual break.
    assert loop._canvas_background_path() == background

    captured: dict[str, Any] = {}

    async def _runner(*_args: Any, **kwargs: Any) -> VisualGateResult:
        captured.update(kwargs)
        return VisualGateResult(passed=True)

    import ubt.core.engine.visual_verify_loop as loop_module

    original = loop_module.get_visual_gate_runner
    loop_module.get_visual_gate_runner = lambda: _runner  # type: ignore[assignment]
    try:
        asyncio.run(loop._evaluate_gate(tmp_path / "artifact.pdf", []))
    finally:
        loop_module.get_visual_gate_runner = original  # type: ignore[assignment]

    assert captured["canvas_background_pdf"] == background


def test_no_recorded_background_passes_none(tmp_path: Path) -> None:
    """A reflow delivery has no background; the gate must be told so explicitly.

    Not by omitting the kwarg: ``run_visual_gate``'s default is also ``None``, so
    a dropped kwarg and a reflow delivery are indistinguishable from the report.
    The argument is always passed, which is why the assertion is on the value.
    """
    loop = _loop(tmp_path, _manifest(tmp_path, background=None))
    assert loop._canvas_background_path() is None


def test_a_recorded_background_that_no_longer_exists_passes_none(tmp_path: Path) -> None:
    """Stale metadata from a cleaned build directory must not crash the gate.

    The path lives under the hidden ``.build_*`` directory. A resumed job whose
    build directory was removed still has the metadata; handing the dead path to
    the gate would make ``render_pages_to_png`` return nothing and report the
    tier as failed, which is a scarier report than the truth.
    """
    manifest = _manifest(tmp_path, background=tmp_path / "gone_bg_stripped.pdf")
    assert _loop(tmp_path, manifest)._canvas_background_path() is None


def test_a_manifest_without_a_metadata_mapping_is_tolerated(tmp_path: Path) -> None:
    """``metadata`` is optional on ``BookManifest``; the reader must not assume it."""
    manifest = _manifest(tmp_path, background=None)
    manifest.metadata = None  # type: ignore[assignment]
    assert _loop(tmp_path, manifest)._canvas_background_path() is None


def test_a_directory_is_not_a_background(tmp_path: Path) -> None:
    """A recorded path that happens to be a directory is not a readable page."""
    directory = tmp_path / "bg_dir.pdf"
    directory.mkdir()
    manifest = _manifest(tmp_path, background=directory)
    assert _loop(tmp_path, manifest)._canvas_background_path() is None
