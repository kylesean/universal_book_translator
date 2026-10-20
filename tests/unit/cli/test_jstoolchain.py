"""The console must not require a package manager it does not need.

`ubt console --dev` and `scripts/build_console_assets.py` both ran a bare
`pnpm ...`. On a machine with npm but no pnpm that is a raw `FileNotFoundError`
traceback, and the build path is worse: `ubt console` calls the script
automatically when the bundle is missing, so a first-time run failed too and
reported only "Failed to build static assets".

These tests pin the resolution order (local `node_modules/.bin` first, then
pnpm → npm → yarn) and the readable failure when nothing is available.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.cli.jstoolchain import JsToolchainError, detect_js_toolchain

pytestmark = pytest.mark.fast


def _fake_web(tmp_path: Path, tools: tuple[str, ...] = ()) -> Path:
    """A ``web/`` directory with the given executables in ``node_modules/.bin``."""
    web = tmp_path / "web"
    bin_dir = web / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    for tool in tools:
        path = bin_dir / tool
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
    return web


def test_local_binaries_are_preferred_over_any_package_manager(tmp_path: Path) -> None:
    """Dependencies installed means no package manager is needed at all.

    This is the repository's own state: ``web/node_modules`` is populated, so
    requiring pnpm to *use* it was requiring a tool the task does not need.
    """
    web = _fake_web(tmp_path, ("vite", "tsc"))
    with patch.object(shutil, "which", return_value="/usr/bin/npm"):
        toolchain = detect_js_toolchain(web)

    assert toolchain.local("vite") == web / "node_modules" / ".bin" / "vite"
    assert toolchain.dev_argv() == [str(web / "node_modules" / ".bin" / "vite")]
    # Two steps, not one `&&`-joined string: `&&` is a shell operator, so a
    # non-shell subprocess would pass it to tsc as an argument.
    assert toolchain.build_steps() == [
        [str(web / "node_modules" / ".bin" / "tsc"), "-b"],
        [str(web / "node_modules" / ".bin" / "vite"), "build"],
    ]


def test_a_package_manager_is_the_fallback_when_node_modules_is_absent(tmp_path: Path) -> None:
    """A fresh clone has no dependencies yet; the manager runs the npm script."""
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
    """The committed lockfile is pnpm's, so pnpm wins when it is available."""
    web = tmp_path / "web"
    web.mkdir()
    with patch.object(shutil, "which", return_value="/usr/bin/whatever"):
        toolchain = detect_js_toolchain(web)
    assert toolchain.package_manager == "pnpm"


def test_no_toolchain_fails_with_an_actionable_message(tmp_path: Path) -> None:
    """The old failure was `FileNotFoundError: 'pnpm'` — not actionable.

    A user reading it cannot tell whether to install pnpm, install dependencies,
    or drop the flag. The message must name all three.
    """
    web = tmp_path / "web"
    web.mkdir()
    with patch.object(shutil, "which", return_value=None):
        toolchain = detect_js_toolchain(web)

    assert not toolchain.can_dev
    assert not toolchain.can_build
    with pytest.raises(JsToolchainError) as excinfo:
        toolchain.dev_argv()
    message = str(excinfo.value)
    assert "no JavaScript toolchain found" in message
    # All three remedies, because any one of them unblocks the user.
    assert "npm install" in message
    assert "node_modules" in message
    assert "without --dev" in message

    with pytest.raises(JsToolchainError):
        toolchain.build_steps()


def test_vite_alone_is_enough_for_dev_but_not_for_build(tmp_path: Path) -> None:
    """`dev` needs only vite; `build` also needs tsc.

    Distinguishing them means a partially installed tree still gets the dev server
    instead of a blanket refusal.
    """
    web = _fake_web(tmp_path, ("vite",))
    with patch.object(shutil, "which", return_value=None):
        toolchain = detect_js_toolchain(web)

    assert toolchain.can_dev is True
    assert toolchain.can_build is False
    assert toolchain.dev_argv() == [str(web / "node_modules" / ".bin" / "vite")]
    with pytest.raises(JsToolchainError):
        toolchain.build_steps()


def test_the_repository_itself_resolves_without_pnpm() -> None:
    """The real checkout must build here, where pnpm is not installed.

    This is the regression that started it: the machine has npm and a populated
    ``node_modules`` but no pnpm, so a pnpm-only path cannot work.
    """
    repo = Path(__file__).resolve().parents[3]
    web = repo / "web"
    if not (web / "node_modules").is_dir():
        pytest.skip("web dependencies are not installed in this checkout")

    toolchain = detect_js_toolchain(web)
    assert toolchain.can_build, "the repository cannot build its own console"
    argv = toolchain.build_steps()
    assert all("pnpm" not in part for step in argv for part in step) or (
        toolchain.local("vite") is None
    )
