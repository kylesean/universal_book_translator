from pathlib import Path

from ubt.core.job_options import clean_source_stem, default_output_path, resolve_target_output


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
