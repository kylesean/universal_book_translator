"""Unit tests for 'ubt doctor --deep' and automated environment remediation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from ubt.cli.commands.doctor_deep import (
    detect_package_manager,
    generate_fix_script,
    probe_cjk_coverage,
    probe_ghostscript,
    probe_gpu_acceleration,
    probe_offline_assets,
    probe_system_memory,
    probe_typst_version,
)
from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_detect_package_manager(monkeypatch: pytest.MonkeyPatch):
    """Verify package manager detection across platforms."""
    # Darwin -> brew
    monkeypatch.setattr("sys.platform", "darwin")
    assert detect_package_manager() == "brew"

    # Linux -> apt
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(
        "shutil.which", lambda cmd: "/usr/bin/apt-get" if cmd == "apt-get" else None
    )
    assert detect_package_manager() == "apt"

    # Linux -> pacman
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/pacman" if cmd == "pacman" else None)
    assert detect_package_manager() == "pacman"


def test_probe_gpu_acceleration_nvidia(monkeypatch: pytest.MonkeyPatch):
    """Verify GPU probe with mock nvidia-smi output."""
    monkeypatch.setattr(
        "shutil.which", lambda cmd: "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else None
    )

    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = "NVIDIA RTX 4090, 24576, 18432, 535.104.05\n"
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: mock_res)

    records = []
    probe_gpu_acceleration(lambda *args, **kwargs: records.append((args, kwargs)))

    assert len(records) == 1
    args, kwargs = records[0]
    assert args[0] == "GPU acceleration"
    assert args[1] == "OK"
    assert kwargs["code"] == "gpu_acceleration"
    assert "RTX 4090" in args[2]


def test_probe_gpu_acceleration_cpu_fallback(monkeypatch: pytest.MonkeyPatch):
    """Verify fallback to CPU execution when no GPU is detected."""
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    monkeypatch.setattr("importlib.util.find_spec", lambda mod: None)

    records = []
    probe_gpu_acceleration(lambda *args, **kwargs: records.append((args, kwargs)))

    assert len(records) == 1
    args, kwargs = records[0]
    assert args[1] == "WARN"
    assert kwargs["detail_code"] == "gpu_acceleration.none"


def test_probe_system_memory(monkeypatch: pytest.MonkeyPatch):
    """Verify system RAM headroom classifications."""
    # 1. Healthy (16GB total, 8GB available)
    mock_vm_ok = MagicMock()
    mock_vm_ok.total = 16 * (1024**3)
    mock_vm_ok.available = 8 * (1024**3)
    monkeypatch.setattr("psutil.virtual_memory", lambda: mock_vm_ok)

    rec_ok = []
    probe_system_memory(lambda *args, **kwargs: rec_ok.append((args, kwargs)))
    assert rec_ok[0][0][1] == "OK"

    # 2. Critical (< 2.0GB available)
    mock_vm_crit = MagicMock()
    mock_vm_crit.total = 16 * (1024**3)
    mock_vm_crit.available = 1.2 * (1024**3)
    monkeypatch.setattr("psutil.virtual_memory", lambda: mock_vm_crit)

    rec_crit = []
    probe_system_memory(lambda *args, **kwargs: rec_crit.append((args, kwargs)))
    assert rec_crit[0][0][1] == "FAIL"
    assert rec_crit[0][1]["detail_code"] == "system_memory.critical"


def test_probe_ghostscript(monkeypatch: pytest.MonkeyPatch):
    """Verify Ghostscript binary version inspection."""
    # Present
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/gs" if cmd == "gs" else None)
    mock_res = MagicMock(returncode=0, stdout="10.03.1\n")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: mock_res)

    rec_ok = []
    probe_ghostscript(lambda *args, **kwargs: rec_ok.append((args, kwargs)))
    assert rec_ok[0][0][1] == "OK"
    assert "10.03.1" in rec_ok[0][0][2]

    # Missing
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    rec_missing = []
    probe_ghostscript(lambda *args, **kwargs: rec_missing.append((args, kwargs)))
    assert rec_missing[0][0][1] == "WARN"
    assert rec_missing[0][1]["detail_code"] == "ghostscript.missing"


def test_probe_typst_version(monkeypatch: pytest.MonkeyPatch):
    """Verify Typst version parsing and compatibility threshold."""
    # 1. Modern Typst 0.15.1 -> OK
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/typst" if cmd == "typst" else None)
    mock_modern = MagicMock(returncode=0, stdout="typst 0.15.1 (a6d077d)\n")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: mock_modern)

    rec_ok = []
    probe_typst_version(lambda *args, **kwargs: rec_ok.append((args, kwargs)))
    assert rec_ok[0][0][1] == "OK"
    assert "0.15.1" in rec_ok[0][0][2]

    # 2. Outdated Typst 0.8.0 -> WARN
    mock_old = MagicMock(returncode=0, stdout="typst 0.8.0\n")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: mock_old)

    rec_old = []
    probe_typst_version(lambda *args, **kwargs: rec_old.append((args, kwargs)))
    assert rec_old[0][0][1] == "WARN"
    assert rec_old[0][1]["detail_code"] == "typst_version.outdated"

    # 3. Missing -> WARN
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    rec_miss = []
    probe_typst_version(lambda *args, **kwargs: rec_miss.append((args, kwargs)))
    assert rec_miss[0][0][1] == "WARN"
    assert rec_miss[0][1]["detail_code"] == "typst_version.missing"


def test_probe_cjk_coverage(monkeypatch: pytest.MonkeyPatch):
    """Verify CJK Serif and Sans font coverage classifications."""
    # Both Serif and Sans
    monkeypatch.setattr(
        "ubt.adapters.pdf.font_probe.available_font_families",
        lambda: ["Noto Serif CJK SC", "Noto Sans CJK SC", "DejaVu Sans"],
    )
    rec_full = []
    probe_cjk_coverage(lambda *args, **kwargs: rec_full.append((args, kwargs)))
    assert rec_full[0][0][1] == "OK"
    assert rec_full[0][1]["detail_code"] == "cjk_coverage.complete"

    # Only Sans
    monkeypatch.setattr(
        "ubt.adapters.pdf.font_probe.available_font_families",
        lambda: ["Noto Sans CJK SC", "Arial"],
    )
    rec_part = []
    probe_cjk_coverage(lambda *args, **kwargs: rec_part.append((args, kwargs)))
    assert rec_part[0][0][1] == "WARN"
    assert rec_part[0][1]["detail_code"] == "cjk_coverage.partial"


def test_probe_offline_assets():
    """Verify core offline assets probe."""
    rec = []
    probe_offline_assets(lambda *args, **kwargs: rec.append((args, kwargs)))
    assert len(rec) == 1
    assert rec[0][0][1] == "OK"
    assert rec[0][1]["detail_code"] == "offline_assets.ok"


def test_generate_fix_script():
    """Verify remediation script generation for missing dependencies."""
    from ubt.cli.commands.doctor import _Check

    mock_checks = [
        _Check(
            group="Rendering & toolchain",
            name="Typst",
            status="WARN",
            detail="missing",
            code="typst",
            fix="install typst",
        ),
        _Check(
            group="Rendering & toolchain",
            name="Ghostscript",
            status="WARN",
            detail="missing",
            code="ghostscript",
            fix="install ghostscript",
        ),
        _Check(
            group="Rendering & toolchain",
            name="Poppler",
            status="WARN",
            detail="missing",
            code="poppler",
            fix="install poppler-utils",
        ),
        _Check(
            group="Deep offline & hardware",
            name="CJK font coverage",
            status="FAIL",
            detail="missing",
            code="cjk_coverage",
            fix="install CJK fonts",
        ),
        _Check(
            group="Deep offline & hardware",
            name="Local inference engine",
            status="WARN",
            detail="missing",
            code="local_inference",
            fix="ollama pull qwen2.5:7b",
        ),
    ]

    # Debian / Ubuntu (apt)
    apt_script = generate_fix_script(mock_checks, target_mgr="apt")
    assert "apt-get install -y" in apt_script
    assert "typst" in apt_script
    assert "fonts-noto-cjk" in apt_script
    assert "ghostscript" in apt_script
    assert "poppler-utils" in apt_script
    assert "ollama pull qwen2.5:7b" in apt_script

    # Arch Linux (pacman)
    pacman_script = generate_fix_script(mock_checks, target_mgr="pacman")
    assert "pacman -S --noconfirm" in pacman_script
    assert "noto-fonts-cjk" in pacman_script

    # macOS (brew)
    brew_script = generate_fix_script(mock_checks, target_mgr="brew")
    assert "brew install" in brew_script


def test_doctor_deep_cli_execution(tmp_path: Path):
    """Verify CLI integration for `ubt doctor --deep` and `--fix-script`."""
    script_path = tmp_path / "fix_env.sh"

    # 1. Human / terminal invocation
    res = runner.invoke(app, ["doctor", "--deep", "--fix-script", str(script_path)])
    assert res.exit_code in (0, 1)  # Depending on local warnings/fails
    assert "Deep offline & hardware" in res.stdout
    assert "GPU acceleration" in res.stdout
    assert "Remediation script generated:" in res.stdout
    assert script_path.exists()
    assert "#!/usr/bin/env bash" in script_path.read_text(encoding="utf-8")

    # 2. JSON output invocation
    res_json = runner.invoke(app, ["doctor", "--deep", "--json"])
    assert res_json.exit_code in (0, 1)
    data = json.loads(res_json.stdout)
    assert "summary" in data
    assert "checks" in data
    names = [c["name"] for c in data["checks"]]
    assert "GPU acceleration" in names
    assert "System memory" in names
    assert "Offline assets" in names
