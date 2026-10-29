"""The facsimile Typst source is what the canvas tier reads the page map from.

``VisualVerifyLoop`` runs a whole stage after the render, so it holds the
manifest and nothing from the render call. It supplies ``run_visual_gate`` with
two things:

* the stripped background, recorded by the facsimile renderer on the manifest;
* the artifact's **Typst source**, which is the only place the artifact-page ->
  background-page map is written down.

The second is easy to lose and expensive to lose silently. The source lookup
originally knew two spellings — ``.<stem>_typst/main.typ`` (publication) and a
``.typ`` sibling — and the facsimile renderer writes a third,
``.build_<stem>_facsimile/facsimile.typ``. The gate therefore never saw a canvas
page map on any facsimile delivery, reported "no canvas pages recorded", and
green-lit every page: a check that cannot fail.

Every failure of this thread reads exactly like a book with no ink collisions,
which is why each spelling is pinned separately.
"""

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

    # The reader is a method on the loop, not on the runner: assert on it first
    # so a failure names the actual break.
    assert loop._canvas_background_path() == background

    captured: dict[str, Any] = {}

    async def _runner(*_args: Any, **kwargs: Any) -> VisualGateResult:
        captured.update(kwargs)
        return VisualGateResult(passed=True)

    import ubt.core.engine.visual_verify_loop as loop_module

    # ``monkeypatch.setattr`` rather than a hand-rolled save/restore: the module
    # imports ``get_visual_gate_runner`` from ``ubt.core.ports`` for its own
    # use and does not re-export it, so assigning the attribute needs a
    # ``type: ignore`` that then has to be kept in step with the import. The
    # fixture does the same thing without weakening the type of the module.
    monkeypatch.setattr(loop_module, "get_visual_gate_runner", lambda: _runner)
    asyncio.run(loop._evaluate_gate(tmp_path / "artifact.pdf", []))

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


# --------------------------------------------------------------------------- #
# The Typst source lookup — the map the canvas tier reads
# --------------------------------------------------------------------------- #


def test_the_facsimile_source_is_found(tmp_path: Path) -> None:
    """**The spelling that was missing, and the reason the tier never ran.**

    ``FacsimileRenderer.render_document`` writes
    ``build_dir / "facsimile.typ"`` where ``build_dir`` is
    ``output.parent / ".build_<stem>_facsimile"``. Nothing else writes that path,
    and until it was added here the gate saw no source, no map, and reported
    every facsimile delivery green.
    """
    artifact = tmp_path / "book.pdf"
    artifact.write_bytes(b"%PDF-1.4\n%%EOF\n")
    build = tmp_path / ".build_book_facsimile"
    build.mkdir()
    typ = build / "facsimile.typ"
    typ.write_text(_CANVAS_TYP, encoding="utf-8")

    assert _typst_source_path(artifact) == typ


def test_the_publication_source_is_still_found(tmp_path: Path) -> None:
    """The reflow renderer's spelling must keep working."""
    artifact = tmp_path / "book.pdf"
    artifact.write_bytes(b"%PDF-1.4\n%%EOF\n")
    build = tmp_path / ".book_typst"
    build.mkdir()
    typ = build / "main.typ"
    typ.write_text("#set text(size: 10pt)\n", encoding="utf-8")

    assert _typst_source_path(artifact) == typ


def test_a_typ_sibling_is_still_found(tmp_path: Path) -> None:
    """The last-resort spelling, for a PDF moved out of its build directory."""
    artifact = tmp_path / "book.pdf"
    artifact.write_bytes(b"%PDF-1.4\n%%EOF\n")
    typ = tmp_path / "book.typ"
    typ.write_text("#set text(size: 10pt)\n", encoding="utf-8")

    assert _typst_source_path(artifact) == typ


def test_the_newest_source_wins_when_a_stale_one_is_left_behind(tmp_path: Path) -> None:
    """A resumed job that changed delivery mode leaves the old build directory.

    Both spellings can therefore exist at once, and the older source describes a
    document that was never delivered — reading it would make the gate look for
    a canvas page map in markup for a different artifact. The rule is newest
    wins; the tuple order is only a tie-break.
    """
    import os
    import time

    artifact = tmp_path / "book.pdf"
    artifact.write_bytes(b"%PDF-1.4\n%%EOF\n")
    stale = tmp_path / ".book_typst"
    stale.mkdir()
    (stale / "main.typ").write_text("#set text(size: 9pt)\n", encoding="utf-8")
    # Make the stale file unambiguously older than the facsimile one.
    old = time.time() - 3600
    os.utime(stale / "main.typ", (old, old))

    build = tmp_path / ".build_book_facsimile"
    build.mkdir()
    typ = build / "facsimile.typ"
    typ.write_text(_CANVAS_TYP, encoding="utf-8")

    assert _typst_source_path(artifact) == typ


def test_no_source_at_all_is_none_rather_than_an_error(tmp_path: Path) -> None:
    """A delivery with no reachable source must degrade, never raise."""
    artifact = tmp_path / "book.pdf"
    artifact.write_bytes(b"%PDF-1.4\n%%EOF\n")

    assert _typst_source_path(artifact) is None
