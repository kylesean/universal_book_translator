"""Default deliverable path: absolute, platform-appropriate, CWD-independent.

``default_output_path`` decides where a run with no ``-o`` writes its book. It
used to be ``tmp/output/`` *relative to the process CWD*, which meant the same
command produced a different location depending on where it was invoked from,
and ``tmp/`` is a name a build tree already wants.
"""

import sys
from pathlib import Path

import pytest

from ubt.core.job_options import default_output_path


def test_default_output_dir_is_absolute_and_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forgotten ``-o`` must not scatter books across whatever directory the
    user happened to be standing in."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("UBT_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("XDG_DOCUMENTS_DIR", raising=False)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")

    out = default_output_path("book.pdf")
    assert out.is_absolute(), out
    assert out.name == "book_bilingual.pdf"
    assert out.parent.name == "UBT"


def test_output_dir_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_OUTPUT_DIR", str(tmp_path / "custom"))
    assert default_output_path("book.epub") == tmp_path / "custom" / "book_bilingual.epub"


def test_xdg_documents_dir_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Linux desktop that relocated Documents must be followed."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "Pappers"))
    monkeypatch.delenv("UBT_OUTPUT_DIR", raising=False)
    assert default_output_path("x.pdf").parent == tmp_path / "Pappers" / "UBT"


def test_default_output_path_resolves_on_every_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same call must work on Windows, macOS and Linux; the platform only
    changes which home directory name is conventional."""
    monkeypatch.delenv("UBT_OUTPUT_DIR", raising=False)
    for plat in ("win32", "darwin", "linux"):
        monkeypatch.setattr(sys, "platform", plat)
        monkeypatch.setenv("HOME", str(tmp_path / plat))
        monkeypatch.delenv("XDG_DOCUMENTS_DIR", raising=False)
        out = default_output_path("book.pdf")
        assert out.is_absolute(), plat
        assert out.parent.name == "UBT", plat
        assert out.name == "book_bilingual.pdf", plat


def test_explicit_output_path_is_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the no-flag default moves; an explicit -o is still honoured verbatim."""
    from ubt.core.job_options import resolve_target_output

    monkeypatch.setenv("UBT_OUTPUT_DIR", str(tmp_path / "ignored"))
    assert resolve_target_output(tmp_path / "mine.pdf", "book.pdf") == tmp_path / "mine.pdf"
    assert resolve_target_output(str(tmp_path / "dir") + "/", "book.pdf") == (
        tmp_path / "dir" / "book_bilingual.pdf"
    )
