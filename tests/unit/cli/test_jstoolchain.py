from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.cli.jstoolchain import JsToolchainError, detect_js_toolchain

pytestmark = pytest.mark.fast


def _fake_web(tmp_path: Path, tools: tuple[str, ...] = ()) -> Path:
    web = tmp_path / "web"
    bin_dir = web / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    for tool in tools:
        path = bin_dir / tool
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
    return web


def test_local_binaries_are_preferred_over_any_package_manager(tmp_path: Path) -> None:
    web = _fake_web(tmp_path, ("vite", "tsc"))
    with patch.object(shutil, "which", return_value="/usr/bin/npm"):
        toolchain = detect_js_toolchain(web)

    assert toolchain.local("vite") == web / "node_modules" / ".bin" / "vite"
    assert toolchain.dev_argv() == [str(web / "node_modules" / ".bin" / "vite")]
    assert toolchain.build_steps() == [
        [str(web / "node_modules" / ".bin" / "tsc"), "-b"],
        [str(web / "node_modules" / ".bin" / "vite"), "build"],
    ]


def test_a_package_manager_is_the_fallback_when_node_modules_is_absent(tmp_path: Path) -> None:
    web = tmp_path / "web"
    web.mkdir()
    with patch.object(
        shutil, "which", side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None
    ):
        toolchain = detect_js_toolchain(web)

    assert toolchain.package_manager == "npm"
    assert toolchain.dev_argv() == ["npm", "run", "dev"]
    assert toolchain.build_steps() == [["npm", "run", "build"]]


def test_pnpm_is_preferred_over_npm_when_both_exist(tmp_path: Path) -> None:
    web = tmp_path / "web"
    web.mkdir()
    with patch.object(shutil, "which", return_value="/usr/bin/whatever"):
        toolchain = detect_js_toolchain(web)
    assert toolchain.package_manager == "pnpm"


def test_no_toolchain_fails_with_an_actionable_message(tmp_path: Path) -> None:
    web = tmp_path / "web"
    web.mkdir()
    with patch.object(shutil, "which", return_value=None):
        toolchain = detect_js_toolchain(web)

    assert toolchain.local("vite") is None
    assert toolchain.package_manager is None
    with pytest.raises(JsToolchainError) as excinfo:
        toolchain.dev_argv()
    message = str(excinfo.value)
    assert "no JavaScript toolchain found" in message
    assert "npm install" in message
    assert "node_modules" in message
    assert "without --dev" in message

    with pytest.raises(JsToolchainError):
        toolchain.build_steps()


def test_vite_alone_is_enough_for_dev_but_not_for_build(tmp_path: Path) -> None:
    web = _fake_web(tmp_path, ("vite",))
    with patch.object(shutil, "which", return_value=None):
        toolchain = detect_js_toolchain(web)

    assert toolchain.local("vite") is not None
    assert toolchain.local("tsc") is None
    assert toolchain.dev_argv() == [str(web / "node_modules" / ".bin" / "vite")]
    with pytest.raises(JsToolchainError):
        toolchain.build_steps()


def test_the_repository_itself_resolves_without_pnpm() -> None:
    repo = Path(__file__).resolve().parents[3]
    web = repo / "web"
    if not (web / "node_modules").is_dir():
        pytest.skip("web dependencies are not installed in this checkout")

    toolchain = detect_js_toolchain(web)
    argv = toolchain.build_steps()
    assert argv
    assert all("pnpm" not in part for step in argv for part in step) or (
        toolchain.local("vite") is None
    )


def test_windows_prefers_the_cmd_stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os, "name", "nt")
    web = _fake_web(tmp_path, ("vite", "tsc"))
    bin_dir = web / "node_modules" / ".bin"
    (bin_dir / "vite.cmd").write_text("@echo off\r\n", encoding="utf-8")
    (bin_dir / "tsc.cmd").write_text("@echo off\r\n", encoding="utf-8")

    with patch.object(shutil, "which", return_value=None):
        toolchain = detect_js_toolchain(web)

    assert toolchain.local("vite") == bin_dir / "vite.cmd"
    assert toolchain.dev_argv() == [str(bin_dir / "vite.cmd")]
    assert toolchain.local("tsc") == bin_dir / "tsc.cmd"


def test_windows_falls_back_to_the_plain_stub_without_a_batch_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(os, "name", "nt")
    web = _fake_web(tmp_path, ("vite", "tsc"))
    bin_dir = web / "node_modules" / ".bin"
    (bin_dir / "vite.exe").write_bytes(b"MZ")
    (bin_dir / "tsc.exe").write_bytes(b"MZ")

    with patch.object(shutil, "which", return_value=None):
        toolchain = detect_js_toolchain(web)

    assert toolchain.local("vite") == bin_dir / "vite.exe"
