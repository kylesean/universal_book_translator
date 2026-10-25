"""Unit tests for typst_compile adapter."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ubt.adapters.pdf import typst_compile

pytestmark = pytest.mark.fast


def test_typst_available_when_cli_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: "/bin/typst")
    assert typst_compile.typst_available("typst") is True


def test_typst_available_when_cli_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: None)
    assert typst_compile.typst_available("typst") is False


def test_typst_compile_subprocess_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: "/bin/typst")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        ok, err = typst_compile.typst_compile("input.typ", "output.pdf")
        assert ok is True
        assert err == ""
        # The timestamp is pinned so two compiles of one source are identical
        # bytes (Typst otherwise stamps the wall clock into /CreationDate).
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/bin/typst"
        assert cmd[1] == "compile"
        assert "--creation-timestamp" in cmd
        assert str(typst_compile.REPRODUCIBLE_TIMESTAMP) in cmd
        assert cmd[-2:] == ["input.typ", "output.pdf"]


def test_typst_compile_forwards_font_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: "/bin/typst")
    monkeypatch.setattr(
        typst_compile,
        "_font_args",
        lambda lang: ["--font-path", "/fake/fonts", "--ignore-system-fonts"],
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        ok, err = typst_compile.typst_compile("input.typ", "output.pdf", target_lang="zh")
        assert ok is True
        cmd = mock_run.call_args[0][0]
        assert "--font-path" in cmd
        assert "/fake/fonts" in cmd
        assert "--ignore-system-fonts" in cmd


def test_typst_compile_binary_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: None)
    ok, err = typst_compile.typst_compile("input.typ", "output.pdf")
    assert ok is False
    assert "not found on PATH" in err


def test_typst_compile_failure_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: "/bin/typst")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="error: unknown variable on line 12")
        ok, err = typst_compile.typst_compile("input.typ", "output.pdf")
        assert ok is False
        assert "error: unknown variable on line 12" in err


def test_typst_compile_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: "/bin/typst")
    with patch("subprocess.run", side_effect=OSError("permission denied")):
        ok, err = typst_compile.typst_compile("input.typ", "output.pdf")
        assert ok is False
        assert "could not be executed: permission denied" in err


def test_typst_compile_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: "/bin/typst")
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="typst", timeout=120)):
        ok, err = typst_compile.typst_compile("input.typ", "output.pdf", timeout=120)
        assert ok is False
        assert "timed out after 120s" in err


def test_typst_version_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: "/bin/typst")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="typst 0.13.1\n")
        assert typst_compile.typst_version() == "typst 0.13.1"


def test_typst_version_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: None)
    assert typst_compile.typst_version() is None


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_two_compiles_of_one_source_are_byte_identical(tmp_path: Path) -> None:
    """The artifact is a function of the source, not of when it was built.

    Typst stamps ``/CreationDate`` and ``/ModDate`` with the wall clock unless
    told otherwise, so the same document compiled twice produced different
    bytes — which contradicted ``ARCHITECTURE.md``'s "same blocks in, same bytes
    out" and made every artifact diff a false positive.
    """
    import hashlib

    source = tmp_path / "a.typ"
    source.write_text("#set page(width: 100pt, height: 100pt)\nHello\n", encoding="utf-8")

    first, second = tmp_path / "one.pdf", tmp_path / "two.pdf"
    ok_one, err_one = typst_compile.typst_compile(str(source), str(first))
    assert ok_one, err_one
    ok_two, err_two = typst_compile.typst_compile(str(source), str(second))
    assert ok_two, err_two

    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )
