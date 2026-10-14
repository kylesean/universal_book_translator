"""Wheel packaging smoke: build, install into a clean venv, check the archive.

Every other test runs against the SOURCE tree, where ``ubt/api/static`` and
every module exist by definition. The wheel is what an operator actually
installs, and nothing checked it: ``packages = ["ubt"]`` could drop a
subpackage, the SPA bundle could fall out of the archive, or a module could
import something only present in the dev extra — and every gate would stay
green while ``pip install universal-book-translator`` produced a broken
package.

Slow tier: it shells out to ``uv build`` and ``uv pip install`` into a fresh
virtual environment. Skipped when ``uv`` is not on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed; cannot build/install the wheel")
    return uv


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One wheel build for the whole module (it is the slow part)."""
    uv = _uv()
    out_dir = tmp_path_factory.mktemp("wheel")
    proc = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out_dir)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"uv build failed:\n{proc.stdout}\n{proc.stderr}"
    wheels = sorted(out_dir.glob("*.whl"))
    assert wheels, f"no wheel produced in {out_dir}"
    return wheels[-1]


def test_the_wheel_carries_the_package_and_the_spa_bundle(built_wheel: Path) -> None:
    """The archive itself: package modules and the prebuilt console."""
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()

    assert any(name.startswith("ubt/") for name in names), "ubt package missing from the wheel"
    # The console ships inside the package so a pip-installed wheel serves it
    # with no Node.js on the end-user machine (pyproject's own comment says so).
    assert "ubt/api/static/index.html" in names, (
        "the SPA bundle is missing: run scripts/build_console_assets.py before packaging"
    )
    assert any(name.startswith("ubt/api/static/assets/") for name in names)
    # A few load-bearing subpackages; a `packages` regression drops these.
    for required in (
        "ubt/core/engine/pipeline.py",
        "ubt/adapters/pdf/docling_adapter.py",
        "ubt/render/typst/document_renderer.py",
        "ubt/cli/main.py",
        "ubt/mcp/server.py",
    ):
        assert required in names, f"{required} missing from the wheel"


_PROBE = """
import json, pathlib
import ubt.api
static = pathlib.Path(ubt.api.__file__).parent / "static"
index = static / "index.html"
html = index.read_text(encoding="utf-8", errors="ignore") if index.exists() else ""
assets_dir = static / "assets"
print(json.dumps({
    "exists": index.exists(),
    "is_html": "<!DOCTYPE HTML>" in html.upper(),
    "asset_count": len(list(assets_dir.glob("*"))) if assets_dir.exists() else 0,
}))
"""


def test_the_installed_wheel_ships_a_usable_console_bundle(
    built_wheel: Path, tmp_path: Path
) -> None:
    """Install into a clean venv and read the bundle from the INSTALLED package.

    ``--no-deps`` on purpose: this smoke is about the archive's own contents,
    not about resolving the dependency tree (the dev env already proves that,
    and a network-free CI must not fetch). The probe therefore imports only
    ``ubt.api``'s package location, not FastAPI.
    """
    uv = _uv()
    venv = tmp_path / "venv"
    subprocess.run(
        [uv, "venv", str(venv)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    install = subprocess.run(
        [uv, "pip", "install", "--python", str(python), "--no-deps", str(built_wheel)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert install.returncode == 0, f"install failed:\n{install.stdout}\n{install.stderr}"

    probe = subprocess.run(
        [str(python), "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert probe.returncode == 0, f"probe failed:\n{probe.stdout}\n{probe.stderr}"
    result = json.loads(probe.stdout.strip().splitlines()[-1])
    assert result["exists"], "ubt/api/static/index.html is missing from the installed wheel"
    assert result["is_html"], "the installed console index is not HTML"
    assert result["asset_count"] > 0, "no hashed assets shipped in the installed wheel"


def test_the_wheel_declares_its_entry_points(built_wheel: Path) -> None:
    """The console scripts are declared in the wheel's own metadata.

    Installing with ``--no-deps`` keeps this smoke network-free, so the CLI
    cannot actually run here (typer is a runtime dependency). What CAN be
    checked from the archive alone is that the entry points an operator calls
    are declared and point at modules the wheel actually carries — a
    ``[project.scripts]`` typo ships a command that does not exist.
    """
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        entry_points = archive.read(metadata_name).decode("utf-8")
        metadata = archive.read(
            next(name for name in names if name.endswith(".dist-info/METADATA"))
        ).decode("utf-8")

    assert "[console_scripts]" in entry_points
    for script in ("ubt", "ubt-api", "ubt-mcp"):
        assert f"{script} =" in entry_points, f"{script} entry point missing"
    # The modules those entry points name must be inside the wheel.
    for module in ("ubt/cli/main.py", "ubt/api/app.py", "ubt/mcp/_entrypoint.py"):
        assert module in names, f"{module} (an entry point target) missing from the wheel"
    # The package metadata is real, not a stub.
    assert "Name: universal-book-translator" in metadata
    assert "Requires-Python:" in metadata
