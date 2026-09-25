"""The MathJax SVG cache lives in the per-user cache, not a shared /tmp path."""

import importlib
import tempfile
from pathlib import Path

import pytest

import ubt.adapters.pdf.math_renderer as module


def _reload(monkeypatch: pytest.MonkeyPatch, extra_env: dict[str, str]) -> Path:
    for var in ("UBT_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)
    for var, value in extra_env.items():
        monkeypatch.setenv(var, value)
    return Path(importlib.reload(module).MATH_CACHE_DIR)


def test_math_cache_dir_respects_ubt_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    got = _reload(monkeypatch, {"UBT_CACHE_DIR": str(tmp_path)})
    assert got == tmp_path / "math_svg"


def test_math_cache_dir_honors_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    got = _reload(monkeypatch, {"XDG_CACHE_HOME": str(tmp_path)})
    assert got == tmp_path / "ubt" / "math_svg"


def test_default_math_cache_dir_is_never_world_tmp(monkeypatch: pytest.MonkeyPatch) -> None:
    got = _reload(monkeypatch, {})
    assert not str(got).startswith(str(Path(tempfile.gettempdir()))), (
        "math SVG cache must not sit in a predictable world-writable directory"
    )
