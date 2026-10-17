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


def test_resolve_profile_name_handles_auto_everywhere() -> None:
    """``auto`` used to be CLI-only; the other surfaces silently got an empty glossary.

    ``ubt translate --profile auto`` probed the document, while REST/MCP/Web
    passing the same string fell through to ``seed_entries_for_profile("auto")``
    — an unknown directory, so the run started with no seed glossary and no
    error. The shared resolver now runs at ``run_kwargs_from_request``, the one
    point every surface passes through (2026-10-14 review, W4-1).
    """
    from ubt.core.job_options import resolve_profile_name, run_kwargs_from_request

    # Explicit names pass through unchanged.
    assert resolve_profile_name("paper") == "paper"
    assert resolve_profile_name("semiconductor") == "semiconductor"
    # No path to probe (or an unreadable one): the CLI's old fallback.
    assert resolve_profile_name("auto") == "general"
    assert resolve_profile_name("auto", "/nonexistent/x.pdf") == "general"
    # Missing/empty falls back to the engine default.
    assert resolve_profile_name(None) == "general"
    assert resolve_profile_name("") == "general"
    # The shared extraction point resolves it for every facade.
    kwargs = run_kwargs_from_request({"profile": "auto", "input_path": "/nonexistent/x.pdf"})
    assert kwargs["profile_name"] == "general"
    assert run_kwargs_from_request({"profile": "paper"})["profile_name"] == "paper"


# --------------------------------------------------------------------------- #
# A --pages run must not share a filename with the full document
#
# Measured during a gate audit: a `--pages 1-6` smoke test wrote its 2-page
# result over the 14-page delivery in the default directory. The next baseline
# check read a two-page document as a 91% content loss, and the run that
# produced it was correct — the name lied about what it was.
# --------------------------------------------------------------------------- #


def test_a_full_run_carries_no_page_scope() -> None:
    from ubt.core.job_options import page_scope_label

    assert page_scope_label(None) == ""
    assert page_scope_label(set()) == ""
    assert default_output_path(Path("book.pdf"), monolingual=True).name == "book_mono.pdf"


def test_a_partial_run_names_the_pages_it_covers() -> None:
    from ubt.core.job_options import page_scope_label

    assert page_scope_label({1, 2, 3, 4, 5, 6}) == "p1-6"
    assert page_scope_label({1, 3, 5}) == "p1,3,5"
    assert page_scope_label({1, 2, 7, 8, 9}) == "p1-2,7-9"
    name = default_output_path(Path("book.pdf"), monolingual=True, pages={1, 2, 3, 4, 5, 6}).name
    assert name == "book_mono_p1-6.pdf"


def test_a_partial_run_never_shares_the_full_document_name() -> None:
    """The defect: both runs resolved to the same path."""
    inp = Path("/tmp/2606.23344v1.pdf")
    full = default_output_path(inp, monolingual=True)
    partial = default_output_path(inp, monolingual=True, pages=set(range(1, 7)))
    assert full.name != partial.name
    assert partial.stem.endswith("_p1-6")


def test_a_huge_scattered_selection_stays_bounded_and_distinct() -> None:
    """Two different page sets must never collide on a shortened label.

    A truncated list would give ``p1,3,5,...`` the same name for two different
    selections, which is exactly the overwrite this exists to prevent.
    """
    from ubt.core.job_options import page_scope_label

    wide_a = page_scope_label(set(range(1, 400, 2)))
    wide_b = page_scope_label(set(range(2, 400, 2)))
    assert wide_a != wide_b
    assert len(wide_a) <= 48 and len(wide_b) <= 48
    assert wide_a.startswith("p")


def test_the_guard_looks_for_the_scoped_name(tmp_path: Path) -> None:
    """An existing partial deliverable is guarded; the full one is not its target."""
    from ubt.core.job_options import deliverable_candidates

    inp = tmp_path / "book.pdf"
    inp.write_text("x")
    scoped = {c.name for c in deliverable_candidates(tmp_path, inp, pages={1, 2, 3})}
    assert scoped == {"book_bilingual_p1-3.pdf", "book_mono_p1-3.pdf"}
    unscoped = {c.name for c in deliverable_candidates(tmp_path, inp)}
    assert scoped.isdisjoint(unscoped)


def test_an_explicit_output_is_never_decorated(tmp_path: Path) -> None:
    """``-o`` names the file; adding a scope marker would deliver elsewhere."""
    from ubt.core.job_options import resolve_target_output

    inp = tmp_path / "book.pdf"
    inp.write_text("x")
    target = resolve_target_output(tmp_path / "chosen.pdf", inp, monolingual=True, pages={1, 2, 3})
    assert target.name == "chosen.pdf"
