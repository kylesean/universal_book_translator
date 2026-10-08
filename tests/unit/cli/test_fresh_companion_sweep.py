"""``--fresh`` redoes a delivery; it must not leave half of the old one behind.

``ubt translate -o X_mono.md --fresh`` regenerates X_mono.md, so the sibling
deliverables of the same run (X_bilingual.md, X_dual.md) are stale by
definition. Two things the sweep used to get wrong:

* it deleted the document but not the document's reports, leaving
  ``x_bilingual_md_quality_report.json`` beside nothing -- a report the next
  reader picks up as the current deliverable's;
* it also swept ``_rigid``/``_reflow`` names that nothing in the tree produces,
  so a file that merely looked like that was deleted for no reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.cli.commands.translate import _clean_stale_companions
from ubt.core.job_options import RUN_REPORT_KINDS, sidecar_path

pytestmark = pytest.mark.fast


def _deliverable(directory: Path, name: str, *, reports: bool = True) -> Path:
    path = directory / name
    path.write_text("delivered", encoding="utf-8")
    if reports:
        for kind in RUN_REPORT_KINDS:
            sidecar_path(path, kind).write_text("{}", encoding="utf-8")
    return path


def test_a_fresh_mono_run_drops_the_stale_companions(tmp_path: Path) -> None:
    output = _deliverable(tmp_path, "book_mono.md")
    bilingual = _deliverable(tmp_path, "book_bilingual.md")
    dual = _deliverable(tmp_path, "book_dual.md")

    removed = _clean_stale_companions(output)

    assert sorted(p.name for p in removed) == ["book_bilingual.md", "book_dual.md"]
    assert not bilingual.exists()
    assert not dual.exists()


def test_a_deleted_companion_takes_its_reports_with_it(tmp_path: Path) -> None:
    output = _deliverable(tmp_path, "book_mono.md")
    bilingual = _deliverable(tmp_path, "book_bilingual.md")

    _clean_stale_companions(output)

    orphans = [sidecar_path(bilingual, kind) for kind in RUN_REPORT_KINDS]
    assert [p.name for p in orphans if p.exists()] == []


def test_a_deliverable_that_is_not_a_companion_is_left_alone(tmp_path: Path) -> None:
    output = _deliverable(tmp_path, "book_mono.md")
    # Not one of the family names: another format's deliverable, a hand-made
    # file, or the legacy render names that no longer have a producer.
    others = [
        _deliverable(tmp_path, "book_mono_rigid.pdf", reports=False),
        _deliverable(tmp_path, "book.epub"),
        _deliverable(tmp_path, "other_bilingual.md"),
    ]

    removed = _clean_stale_companions(output)

    assert removed == []
    assert all(p.exists() for p in others)


def test_the_input_document_is_never_deleted(tmp_path: Path) -> None:
    # -o book_mono.md beside an input literally named book_bilingual.md: the
    # name pattern matches the source, and the source must survive --fresh.
    source = _deliverable(tmp_path, "book_bilingual.md")
    output = _deliverable(tmp_path, "book_mono.md")

    removed = _clean_stale_companions(output, input_path=source)

    assert removed == []
    assert source.exists()
    assert source.read_text(encoding="utf-8") == "delivered"


def test_a_dual_primary_drops_the_mono_render(tmp_path: Path) -> None:
    output = _deliverable(tmp_path, "book_dual.md")
    mono = _deliverable(tmp_path, "book_mono.md")

    removed = _clean_stale_companions(output)

    assert [p.name for p in removed] == ["book_mono.md"]
    assert not mono.exists()


def test_an_output_with_no_family_suffix_sweeps_nothing(tmp_path: Path) -> None:
    # -o delivered.md: the mono/dual/bilingual names are unrelated to it, and
    # guessing which of them "belongs" to this run is how a sweep deletes
    # somebody else's delivery.
    output = _deliverable(tmp_path, "delivered.md")
    bystander = _deliverable(tmp_path, "book_bilingual.md")

    assert _clean_stale_companions(output) == []
    assert bystander.exists()


def test_no_output_means_nothing_to_sweep() -> None:
    assert _clean_stale_companions(None) == []
