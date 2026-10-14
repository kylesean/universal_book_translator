from pathlib import Path

import pytest

from ubt.core.job_options import (
    clean_source_stem,
    default_output_path,
    delivery_verdict_from_blocks,
    delivery_verdict_from_summary,
    resolve_target_output,
)

pytestmark = pytest.mark.fast


def test_clean_source_stem_strips_upload_prefix() -> None:
    assert clean_source_stem("20261007-055316-84614c7e-chapter-3.pdf") == "chapter-3"
    assert clean_source_stem("/tmp/uploads/20261007-042355-491c135b-book.pdf") == "book"
    assert (
        clean_source_stem(Path("/tmp/uploads/20261007-042355-491c135b-book_mono.pdf"))
        == "book_mono"
    )


def test_clean_source_stem_preserves_regular_stems() -> None:
    assert clean_source_stem("chapter-3.pdf") == "chapter-3"
    assert clean_source_stem("book.pdf") == "book"
    assert clean_source_stem("/path/to/my-document.epub") == "my-document"


def test_default_output_path_with_upload_prefix() -> None:
    raw = "/path/to/uploads/20261007-055316-84614c7e-chapter-3.pdf"
    out_mono = default_output_path(raw, monolingual=True)
    assert out_mono.name == "chapter-3_mono.pdf"

    out_dual = default_output_path(raw, monolingual=False)
    assert out_dual.name == "chapter-3_bilingual.pdf"


def test_resolve_target_output_with_directory(tmp_path: Path) -> None:
    upload_input = "/uploads/20261007-055316-84614c7e-chapter-3.pdf"

    # Directory output with monolingual=True
    target_mono = resolve_target_output(tmp_path, upload_input, monolingual=True)
    assert target_mono == tmp_path / "chapter-3_mono.pdf"

    # Directory output with monolingual=False
    target_dual = resolve_target_output(tmp_path, upload_input, monolingual=False)
    assert target_dual == tmp_path / "chapter-3_bilingual.pdf"


def test_resolve_target_output_respects_explicit_file(tmp_path: Path) -> None:
    upload_input = "/uploads/20261007-055316-84614c7e-chapter-3.pdf"
    explicit_file = tmp_path / "custom_output.pdf"

    target = resolve_target_output(explicit_file, upload_input, monolingual=True)
    assert target == explicit_file


# --- The one overwrite-authorization authority (REST / CLI / MCP share it) ---


def test_deliverable_candidates_cover_every_name_a_run_can_write(tmp_path: Path) -> None:
    """A directory (or default) output has TWO names; a file output has one.

    The mono/bilingual label is decided inside the run, so an entry-point guard
    cannot know which one a re-run will write — it must guard both. The CLI
    checked only the bilingual name and let a monolingual re-run replace the
    previous delivery.
    """
    from ubt.core.job_options import deliverable_candidates

    inp = tmp_path / "book.md"
    inp.write_text("x")
    # Explicit file: one candidate, its own name.
    assert [c.name for c in deliverable_candidates(tmp_path / "out.md", inp)] == ["out.md"]
    # Suffix-less path resolves to the input's suffix (the CLI used to miss this).
    assert [c.name for c in deliverable_candidates(tmp_path / "delivery", inp)] == ["delivery.md"]
    # A directory (existing, or trailing-slash) yields both labelled names.
    assert {c.name for c in deliverable_candidates(tmp_path, inp)} == {
        "book_bilingual.md",
        "book_mono.md",
    }
    assert {c.name for c in deliverable_candidates(f"{tmp_path}/", inp)} == {
        "book_bilingual.md",
        "book_mono.md",
    }


def test_own_prior_deliverable_licenses_only_the_exact_file(tmp_path: Path) -> None:
    """A prior run licenses its own deliverable — never a sibling, never a dir tree.

    This is the finding: REST's ``_is_own_prior_output`` compared a directory's
    *parent*, so a job id that once named ``/books/`` could then overwrite
    ``/books/someone_elses_manuscript.md``.
    """
    from ubt.core.job_options import PriorOutput, own_prior_deliverable

    inp = tmp_path / "book.md"
    inp.write_text("x")
    books = tmp_path / "books"
    books.mkdir()

    dir_prior = PriorOutput(output_path=str(books), input_path=str(inp))
    # The derived deliverable names inside the directory ARE its own output.
    assert own_prior_deliverable(books / "book_bilingual.md", dir_prior)
    assert own_prior_deliverable(books / "book_mono.md", dir_prior)
    # Any other file in that directory is NOT.
    assert not own_prior_deliverable(books / "someone_elses_manuscript.md", dir_prior)

    file_prior = PriorOutput(output_path=str(tmp_path / "out.md"), input_path=str(inp))
    assert own_prior_deliverable(tmp_path / "out.md", file_prior)
    assert not own_prior_deliverable(tmp_path / "other.md", file_prior)
    # No prior, or an unusable one, licenses nothing.
    assert not own_prior_deliverable(tmp_path / "out.md", None)
    assert not own_prior_deliverable(
        tmp_path / "out.md", PriorOutput(output_path="", input_path="")
    )


def test_prior_output_of_reads_both_record_shapes(tmp_path: Path) -> None:
    """The manager record (``.request``) and the queue payload (dict) both work."""
    from ubt.core.job_options import prior_output_of

    inp = tmp_path / "book.md"
    inp.write_text("x")
    payload = {"output_path": str(tmp_path / "o.md"), "input_path": str(inp)}

    class _ManagerRec:
        def __init__(self, data: dict[str, str]) -> None:
            self.request = type("R", (), {"model_dump": lambda self: dict(data)})()

    class _QueueRec:
        def __init__(self, data: dict[str, str]) -> None:
            self.payload = dict(data)

    for rec in (_ManagerRec(payload), _QueueRec(payload), payload):
        parsed = prior_output_of(rec)
        assert parsed is not None and parsed.output_path == str(tmp_path / "o.md")
    assert prior_output_of(None) is None
    assert prior_output_of(object()) is None


def test_delivery_verdict_is_the_single_three_state_rule() -> None:
    """The rule four facades share: failed/blocked = BLOCKED, needs_human =
    WARNING, else PASSED; zero completed is BLOCKED.

    Before this lived in one place the CLI banner and the web quality gate
    disagreed: the gate read ``failed == 0`` and announced PRODUCTION READY
    over a delivery the CLI called BLOCKED (quarantined blocks shipping
    source) — 2026-10-14 review, H3.
    """
    # Quarantined blocks ship source: BLOCKED even with zero failed.
    assert (
        delivery_verdict_from_blocks(
            failed_blocks=0, blocked_human_blocks=20, needs_human_blocks=0, completed_blocks=310
        )
        == "blocked"
    )
    # Failed blocks are BLOCKED.
    assert (
        delivery_verdict_from_blocks(
            failed_blocks=3, blocked_human_blocks=0, needs_human_blocks=0, completed_blocks=300
        )
        == "blocked"
    )
    # Needs-human still ships its translation: WARNING only.
    assert (
        delivery_verdict_from_blocks(
            failed_blocks=0, blocked_human_blocks=0, needs_human_blocks=5, completed_blocks=310
        )
        == "warning"
    )
    # Nothing completed through QA: BLOCKED (even with needs_human pending).
    assert (
        delivery_verdict_from_blocks(
            failed_blocks=0, blocked_human_blocks=0, needs_human_blocks=12, completed_blocks=0
        )
        == "blocked"
    )
    # Clean run: PASSED.
    assert (
        delivery_verdict_from_blocks(
            failed_blocks=0, blocked_human_blocks=0, needs_human_blocks=0, completed_blocks=310
        )
        == "passed"
    )


def test_delivery_verdict_from_summary_maps_report_keys() -> None:
    """The quality-report summary shape (``*_blocks`` keys) maps identically."""
    assert (
        delivery_verdict_from_summary(
            {"completed_blocks": 310, "blocked_human_blocks": 20, "needs_human_blocks": 20}
        )
        == "blocked"
    )
    assert delivery_verdict_from_summary({"completed_blocks": 310}) == "passed"
    # Missing keys read as zero, not as an error.
    assert delivery_verdict_from_summary({}) == "blocked"
