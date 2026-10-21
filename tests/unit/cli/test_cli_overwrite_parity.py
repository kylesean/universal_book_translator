"""One overwrite rule on every surface: a re-run of *this* job, and nothing else.

Overwrite is authorization, not convenience. The rule the project states is "is
this the file this very job already produced?" — REST and MCP have always
answered it through :func:`ubt.core.job_options.own_prior_deliverable`, and the
CLI answered a different question ("does any file exist?") by refusing
everything. The observable effect: a job that finished, leaving its deliverable
on disk, could be re-run through the API and refused through `ubt translate`.

The tests below pin both halves — the exemption fires for the job's own file,
and every neighbouring case still refuses. A guard narrowed to "allow if the
path exists" would pass the first and fail the rest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from ubt.cli.commands.translate import _prior_output_from_ledger, _refuse_existing_output
from ubt.core.job_options import PriorOutput

pytestmark = pytest.mark.fast


def _input(tmp_path: Path) -> Path:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return source


def _existing(tmp_path: Path, name: str = "out.pdf") -> Path:
    target = tmp_path / name
    target.write_bytes(b"previous delivery")
    return target


def test_the_jobs_own_prior_file_may_be_overwritten(tmp_path: Path) -> None:
    """A finished job's deliverable is not a foreign file."""
    source = _input(tmp_path)
    target = _existing(tmp_path)
    prior = PriorOutput(output_path=str(target), input_path=str(source))

    assert _refuse_existing_output(target, fresh=False, input_path=source, prior=prior) == target


def test_without_a_prior_record_every_existing_file_is_refused(tmp_path: Path) -> None:
    """No job id, no ledger, no exemption."""
    source = _input(tmp_path)
    target = _existing(tmp_path)
    with pytest.raises(typer.BadParameter, match="already exists"):
        _refuse_existing_output(target, fresh=False, input_path=source, prior=None)


def test_another_jobs_file_is_refused(tmp_path: Path) -> None:
    """The prior record must name *this* file, not just some file."""
    source = _input(tmp_path)
    own = _existing(tmp_path, "own.pdf")
    foreign = _existing(tmp_path, "foreign.pdf")
    prior = PriorOutput(output_path=str(own), input_path=str(source))

    with pytest.raises(typer.BadParameter, match="already exists"):
        _refuse_existing_output(foreign, fresh=False, input_path=source, prior=prior)


def test_a_prior_record_for_a_different_input_is_refused(tmp_path: Path) -> None:
    """Same output name, different source document: a different deliverable."""
    source = _input(tmp_path)
    other_source = tmp_path / "other.pdf"
    other_source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    target = _existing(tmp_path)
    prior = PriorOutput(output_path=str(target), input_path=str(other_source))

    with pytest.raises(typer.BadParameter, match="already exists"):
        _refuse_existing_output(target, fresh=False, input_path=source, prior=prior)


def test_fresh_still_overwrites_without_any_prior_record(tmp_path: Path) -> None:
    """``--fresh`` remains the explicit "I mean it" escape hatch."""
    source = _input(tmp_path)
    target = _existing(tmp_path)
    assert _refuse_existing_output(target, fresh=True, input_path=source, prior=None) == target


def test_the_input_document_is_never_overwritable(tmp_path: Path) -> None:
    """Even a matching prior record cannot authorize clobbering the source."""
    source = _input(tmp_path)
    prior = PriorOutput(output_path=str(source), input_path=str(source))
    with pytest.raises(typer.BadParameter, match="equals the input"):
        _refuse_existing_output(source, fresh=False, input_path=source, prior=prior)


def test_a_missing_job_id_records_no_prior_output(tmp_path: Path) -> None:
    assert _prior_output_from_ledger(None, tmp_path) is None


def test_an_unknown_job_id_records_no_prior_output(tmp_path: Path) -> None:
    """A mistyped id must not grant an exemption by accident."""
    assert _prior_output_from_ledger("no-such-job", tmp_path) is None


def test_a_recorded_job_is_read_back_from_its_ledger(tmp_path: Path) -> None:
    """The pointer the export stage persists is what the guard reads.

    Round-trips through a real ledger file so the key name and the metadata
    shape are exercised, not just the helper's parsing.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest, ChapterMeta

    db_path = tmp_path / "job_x.sqlite"
    ledger = SQLiteJobLedger(db_path)
    try:
        manifest = BookManifest(
            doc_id="doc",
            title="book",
            source_path=str(_input(tmp_path)),
            source_lang="en",
            target_lang="zh",
            chapters=[ChapterMeta(chapter_id="c1", title="book", spine_index=1)],
        )
        ledger.init_job_from_manifest("job_x", manifest)
        ledger.set_job_metadata_value("job_x", "output_file", str(tmp_path / "out.pdf"))
    finally:
        ledger.close()

    prior = _prior_output_from_ledger("job_x", tmp_path)
    assert prior is not None
    assert prior.output_path == str(tmp_path / "out.pdf")


# --- the shared authority itself ----------------------------------------


def test_the_shared_helper_refuses_a_record_for_another_document(tmp_path: Path) -> None:
    """The record must name this document, not merely this output path.

    ``prior.input_path`` was only ever used to *derive* the candidate names, so
    a record whose source was a different document still authorized overwriting
    the path. That is a file belonging to another document; the re-run rule is
    about one job's own deliverable.
    """
    from ubt.core.job_options import own_prior_deliverable

    source = _input(tmp_path)
    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF-1.4\n%%EOF\n")
    target = _existing(tmp_path)
    prior = PriorOutput(output_path=str(target), input_path=str(other))

    assert not own_prior_deliverable(target, prior, input_path=source)
    # The same record for the right document is authorized.
    assert own_prior_deliverable(target, prior, input_path=other)


def test_the_shared_helper_without_a_current_input_keeps_working(tmp_path: Path) -> None:
    """A caller that cannot supply the source keeps the previous contract.

    Deliberate: the check is opt-in so a surface that has no current document at
    hand is not silently refused. All three in-tree callers pass it.
    """
    from ubt.core.job_options import own_prior_deliverable

    target = _existing(tmp_path)
    prior = PriorOutput(output_path=str(target), input_path=str(_input(tmp_path)))
    assert own_prior_deliverable(target, prior)
