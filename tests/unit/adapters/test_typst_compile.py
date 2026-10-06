"""Unit tests for typst_compile adapter."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from ubt.adapters.pdf import typst_compile


def test_typst_available_when_py_binding_present(monkeypatch):
    monkeypatch.setattr(typst_compile, "_HAS_TYPST_PY", True)
    assert typst_compile.typst_available("nonexistent-typst-binary") is True


def test_typst_available_when_cli_found(monkeypatch):
    monkeypatch.setattr(typst_compile, "_HAS_TYPST_PY", False)
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: "/bin/typst")
    assert typst_compile.typst_available("typst") is True


def test_typst_compile_via_py_binding_success(monkeypatch, tmp_path):
    monkeypatch.setattr(typst_compile, "_HAS_TYPST_PY", True)
    mock_typst = MagicMock()
    monkeypatch.setattr("ubt.adapters.pdf.typst_compile.typst", mock_typst, raising=False)
    with patch.dict("sys.modules", {"typst": mock_typst}):
        ok, err = typst_compile.typst_compile("input.typ", "output.pdf")
        assert ok is True
        assert err == ""
        mock_typst.compile.assert_called_once_with("input.typ", output="output.pdf")


def test_typst_compile_py_binding_fallback_to_cli(monkeypatch):
    monkeypatch.setattr(typst_compile, "_HAS_TYPST_PY", True)
    mock_typst = MagicMock()
    mock_typst.compile.side_effect = RuntimeError("binding crash")
    with patch.dict("sys.modules", {"typst": mock_typst}):
        monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: "/bin/typst")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            ok, err = typst_compile.typst_compile("input.typ", "output.pdf")
            assert ok is True
            assert mock_run.called


def test_typst_compile_binary_not_found(monkeypatch):
    monkeypatch.setattr(typst_compile, "_HAS_TYPST_PY", False)
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: None)
    ok, err = typst_compile.typst_compile("input.typ", "output.pdf")
    assert ok is False
    assert "not found on PATH" in err


def test_typst_compile_timeout(monkeypatch):
    monkeypatch.setattr(typst_compile, "_HAS_TYPST_PY", False)
    monkeypatch.setattr(typst_compile, "resolve_typst_binary", lambda b: "/bin/typst")
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="typst", timeout=120)):
        ok, err = typst_compile.typst_compile("input.typ", "output.pdf", timeout=120)
        assert ok is False
        assert "timed out after 120s" in err


def test_typst_version_py_binding(monkeypatch):
    monkeypatch.setattr(typst_compile, "_HAS_TYPST_PY", True)
    mock_typst = MagicMock()
    mock_typst.__version__ = "0.11.0"
    with patch.dict("sys.modules", {"typst": mock_typst}):
        ver = typst_compile.typst_version()
        assert ver == "typst-py 0.11.0"
