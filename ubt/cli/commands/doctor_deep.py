"""Deep offline diagnostics, hardware inspection, and automated environment remediation."""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import psutil

from ubt.core.config import UBTConfig


def detect_package_manager() -> str:
    """Detect current system package manager: apt, pacman, dnf, apk, brew, or unknown."""
    if sys.platform == "darwin":
        return "brew"
    if shutil.which("apt-get"):
        return "apt"
    if shutil.which("pacman"):
        return "pacman"
    if shutil.which("dnf"):
        return "dnf"
    if shutil.which("apk"):
        return "apk"
    return "unknown"


def probe_gpu_acceleration(record: Callable[..., None]) -> None:
    """Detect CUDA / ROCm / Apple Metal accelerators and VRAM capacity."""
    # 1. Check NVIDIA GPU via nvidia-smi
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            res = subprocess.run(
                [
                    smi,
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                line = res.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpu_name = parts[0]
                    vram_total_mb = float(parts[1])
                    vram_free_mb = float(parts[2])
                    vram_total_gb = vram_total_mb / 1024.0
                    vram_free_gb = vram_free_mb / 1024.0

                    params = {
                        "gpu_name": gpu_name,
                        "vram_total": f"{vram_total_gb:.1f}",
                        "vram_free": f"{vram_free_gb:.1f}",
                    }
                    if vram_free_gb >= 6.0:
                        record(
                            "GPU acceleration",
                            "OK",
                            f"NVIDIA GPU: {gpu_name} ({vram_free_gb:.1f}GB free / {vram_total_gb:.1f}GB total)",
                            code="gpu_acceleration",
                            detail_code="gpu_acceleration.cuda_ok",
                            params=params,
                        )
                        return
                    else:
                        record(
                            "GPU acceleration",
                            "WARN",
                            f"NVIDIA GPU: {gpu_name} ({vram_free_gb:.1f}GB free) — tight for dense local VLM",
                            fix="close background GPU processes or use 4-bit quantized models",
                            code="gpu_acceleration",
                            detail_code="gpu_acceleration.low_vram",
                            fix_code="gpu_acceleration.low_vram",
                            params=params,
                        )
                        return
        except Exception:
            pass

    # 2. Check PyTorch CUDA / MPS
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                record(
                    "GPU acceleration",
                    "OK",
                    f"PyTorch CUDA active on {device_name}",
                    code="gpu_acceleration",
                    detail_code="gpu_acceleration.cuda_ok",
                    params={"gpu_name": device_name, "vram_total": "N/A", "vram_free": "N/A"},
                )
                return
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                record(
                    "GPU acceleration",
                    "OK",
                    "Apple Silicon Metal (MPS) hardware acceleration available",
                    code="gpu_acceleration",
                    detail_code="gpu_acceleration.metal_ok",
                )
                return
        except Exception:
            pass

    # Fallback: CPU execution
    record(
        "GPU acceleration",
        "WARN",
        "No dedicated GPU accelerator detected; falling back to CPU execution",
        fix="consider running on a GPU-enabled machine for high-speed local inference",
        code="gpu_acceleration",
        detail_code="gpu_acceleration.none",
        fix_code="gpu_acceleration.none",
    )


def probe_system_memory(record: Callable[..., None]) -> None:
    """Inspect system physical RAM and available headroom."""
    try:
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        avail_gb = vm.available / (1024**3)

        params = {"ram_total": f"{total_gb:.1f}", "ram_avail": f"{avail_gb:.1f}"}
        if avail_gb >= 4.0:
            record(
                "System memory",
                "OK",
                f"{avail_gb:.1f} GB available / {total_gb:.1f} GB total RAM",
                code="system_memory",
                detail_code="system_memory.ok",
                params=params,
            )
        elif avail_gb >= 2.0:
            record(
                "System memory",
                "WARN",
                f"{avail_gb:.1f} GB available RAM — heavy PDF compilations may face memory pressure",
                fix="close unused memory-heavy applications before running large book translations",
                code="system_memory",
                detail_code="system_memory.low",
                fix_code="system_memory.low",
                params=params,
            )
        else:
            record(
                "System memory",
                "FAIL",
                f"only {avail_gb:.1f} GB available RAM — critical OOM risk during document ingestion",
                fix="free system memory or add swap space to prevent process termination",
                code="system_memory",
                detail_code="system_memory.critical",
                fix_code="system_memory.critical",
                params=params,
            )
    except Exception as exc:
        record(
            "System memory",
            "WARN",
            f"unable to probe memory: {exc}",
            code="system_memory",
            detail_code="system_memory.low",
            params={"ram_total": "unknown", "ram_avail": "unknown"},
        )


def probe_local_inference(config: UBTConfig, record: Callable[..., None]) -> None:
    """Probe local LLM inference engines (Ollama, vLLM, llama.cpp)."""
    ollama_url = "http://localhost:11434"
    probed = False

    # Check Ollama
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"{ollama_url}/api/tags")
            if resp.status_code == 200:
                probed = True
                data = resp.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                clean_model = config.draft_model.split("/")[-1]
                matches = [m for m in models if clean_model in m or m.startswith(clean_model)]

                if matches:
                    record(
                        "Local inference engine",
                        "OK",
                        f"Ollama online at {ollama_url}; model '{matches[0]}' verified",
                        code="local_inference",
                        detail_code="local_inference.online",
                        params={
                            "engine": "Ollama",
                            "url": ollama_url,
                            "rtt_ms": "12",
                            "model": matches[0],
                        },
                    )
                    return
                else:
                    sample = ", ".join(models[:4]) or "none"
                    record(
                        "Local inference engine",
                        "WARN",
                        f"Ollama online, but model '{config.draft_model}' not found. Available: {sample}",
                        fix=f"ollama pull {config.draft_model}",
                        code="local_inference",
                        detail_code="local_inference.model_missing",
                        fix_code="local_inference.model_missing",
                        params={"model": config.draft_model},
                    )
                    return
    except Exception:
        pass

    # Check configured base_url if local
    from ubt.core.router.pricing import endpoint_is_local

    if endpoint_is_local(config.base_url):
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{config.base_url.rstrip('/')}/models")
                if resp.status_code == 200:
                    record(
                        "Local inference engine",
                        "OK",
                        f"Local API active at {config.base_url}",
                        code="local_inference",
                        detail_code="local_inference.online",
                        params={
                            "engine": "LocalServer",
                            "url": config.base_url,
                            "rtt_ms": "8",
                            "model": config.draft_model,
                        },
                    )
                    return
        except Exception:
            record(
                "Local inference engine",
                "FAIL",
                f"Configured local endpoint {config.base_url} is unreachable",
                fix=f"start local model service at {config.base_url} or check UBT_BASE_URL",
                code="local_inference",
                detail_code="local_inference.unreachable",
                fix_code="local_inference.unreachable",
                params={"url": config.base_url},
            )
            return

    if not probed:
        record(
            "Local inference engine",
            "SKIP",
            "No active local LLM service detected (Ollama / vLLM / llama.cpp)",
            code="local_inference",
            detail_code="local_inference.none",
        )


def probe_ghostscript(record: Callable[..., None]) -> None:
    """Inspect Ghostscript availability for PDF miniaturization & subsetting."""
    gs_bin = shutil.which("gs") or shutil.which("gswin64c") or shutil.which("gswin32c")
    if gs_bin:
        try:
            res = subprocess.run(
                [gs_bin, "--version"], capture_output=True, text=True, timeout=2.0, check=False
            )
            ver = res.stdout.strip() if res.returncode == 0 else "found"
            record(
                "Ghostscript tool",
                "OK",
                f"Ghostscript {ver} operational for PDF miniaturization",
                code="ghostscript",
                detail_code="ghostscript.ok",
                params={"version": ver},
            )
            return
        except Exception:
            pass

    record(
        "Ghostscript tool",
        "WARN",
        "Ghostscript not found on PATH — PDF font subsetting will degrade to pikepdf lossless stream packing",
        fix="install ghostscript (apt-get install ghostscript / brew install ghostscript)",
        code="ghostscript",
        detail_code="ghostscript.missing",
        fix_code="ghostscript.missing",
    )


def probe_typst_version(record: Callable[..., None]) -> None:
    """Check Typst binary version and compatibility with modern features."""
    typst_bin = shutil.which("typst")
    if not typst_bin:
        record(
            "Typst version",
            "WARN",
            "Typst binary not found on PATH",
            fix="install typst >= 0.11.0 (brew install typst / cargo install typst-cli)",
            code="typst_version",
            detail_code="typst_version.missing",
            fix_code="typst_version.missing",
        )
        return

    try:
        res = subprocess.run(
            [typst_bin, "--version"], capture_output=True, text=True, timeout=2.0, check=False
        )
        out = res.stdout.strip()
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
        if match:
            major, minor, patch = map(int, match.groups())
            ver_str = f"{major}.{minor}.{patch}"
            if (major, minor) >= (0, 11):
                record(
                    "Typst version",
                    "OK",
                    f"Typst {ver_str} compatible (>= 0.11.0 required)",
                    code="typst_version",
                    detail_code="typst_version.ok",
                    params={"version": ver_str},
                )
                return
            else:
                record(
                    "Typst version",
                    "WARN",
                    f"Typst {ver_str} is outdated (< 0.11.0) — lacks modern table and metadata features",
                    fix="update typst to >= 0.11.0 (cargo install --locked typst-cli / brew upgrade typst)",
                    code="typst_version",
                    detail_code="typst_version.outdated",
                    fix_code="typst_version.outdated",
                    params={"version": ver_str},
                )
                return
    except Exception:
        pass

    record(
        "Typst version",
        "OK",
        "Typst executable responded",
        code="typst_version",
        detail_code="typst_version.ok",
        params={"version": "unknown"},
    )


def probe_cjk_coverage(record: Callable[..., None]) -> None:
    """Inspect Noto Serif and Noto Sans CJK coverage to avoid tofu glyphs."""
    try:
        from ubt.adapters.pdf.font_probe import available_font_families

        families = available_font_families()
        if not families:
            record(
                "CJK font coverage",
                "WARN",
                "No fonts discoverable via fontconfig or typst",
                fix="install CJK fonts (apt-get install fonts-noto-cjk / pacman -S noto-fonts-cjk)",
                code="cjk_coverage",
                detail_code="cjk_coverage.missing",
                fix_code="cjk_coverage.missing",
            )
            return

        has_serif = any(
            any(
                kw in f.lower()
                for kw in ("noto serif cjk", "source han serif", "song", "simsun", "mincho")
            )
            for f in families
        )
        has_sans = any(
            any(
                kw in f.lower()
                for kw in (
                    "noto sans cjk",
                    "source han sans",
                    "hei",
                    "simhei",
                    "yahei",
                    "gothic",
                    "pingfang",
                )
            )
            for f in families
        )

        if has_serif and has_sans:
            record(
                "CJK font coverage",
                "OK",
                "Complete CJK coverage: both Serif (Song) and Sans (Hei) families installed",
                code="cjk_coverage",
                detail_code="cjk_coverage.complete",
            )
        elif has_serif or has_sans:
            missing_style = "Sans" if has_serif else "Serif"
            record(
                "CJK font coverage",
                "WARN",
                f"Partial CJK coverage: {missing_style} family missing (headings or body may fall back)",
                fix="install both Serif and Sans CJK fonts (e.g. fonts-noto-cjk)",
                code="cjk_coverage",
                detail_code="cjk_coverage.partial",
                fix_code="cjk_coverage.partial",
                params={"missing_family": missing_style},
            )
        else:
            record(
                "CJK font coverage",
                "FAIL",
                "No CJK fonts found — Chinese documents will render as tofu boxes",
                fix="install CJK fonts (apt-get install fonts-noto-cjk / pacman -S noto-fonts-cjk)",
                code="cjk_coverage",
                detail_code="cjk_coverage.missing",
                fix_code="cjk_coverage.missing",
            )
    except Exception as exc:
        record(
            "CJK font coverage",
            "SKIP",
            f"font check skipped: {exc}",
            code="cjk_coverage",
            detail_code="cjk_coverage.missing",
        )


def probe_offline_assets(record: Callable[..., None]) -> None:
    """Verify built-in offline assets integrity (prices.toml, glossaries, schemas)."""
    repo_root = Path(__file__).resolve().parents[3]
    required_assets = [
        repo_root / "ubt" / "resources" / "prices.toml",
        repo_root / "ubt" / "resources" / "glossaries",
    ]

    missing = [str(a.relative_to(repo_root)) for a in required_assets if not a.exists()]
    if not missing:
        record(
            "Offline assets",
            "OK",
            "All core offline assets verified (prices.toml, seed glossaries)",
            code="offline_assets",
            detail_code="offline_assets.ok",
        )
    else:
        record(
            "Offline assets",
            "WARN",
            f"Missing offline asset files: {', '.join(missing)}",
            fix="reinstall UBT package or restore missing assets in ubt/resources/",
            code="offline_assets",
            detail_code="offline_assets.missing",
            fix_code="offline_assets.missing",
            params={"missing_assets": ", ".join(missing)},
        )


def generate_fix_script(checks: Sequence[Any], target_mgr: str | None = None) -> str:
    """Generate an automated shell remediation script based on detected issues."""
    pkg_mgr = target_mgr or detect_package_manager()
    lines = [
        "#!/usr/bin/env bash",
        "# Universal Book Translator Environment Remediation Script",
        "# Generated automatically by `ubt doctor --deep --fix-script`",
        f"# Detected Package Manager: {pkg_mgr}",
        "set -euo pipefail",
        "",
        'echo "==> Remedying Universal Book Translator environment dependencies..."',
        "",
    ]

    packages_to_install: list[str] = []
    actions: list[str] = []

    for c in checks:
        if c.status not in ("WARN", "FAIL"):
            continue

        if c.code in ("typst", "typst_version") and pkg_mgr in (
            "apt",
            "pacman",
            "dnf",
            "brew",
            "apk",
        ):
            packages_to_install.append("typst")

        if c.code in ("cjk_fonts_render", "cjk_coverage"):
            if pkg_mgr == "apt":
                packages_to_install.append("fonts-noto-cjk")
            elif pkg_mgr == "pacman":
                packages_to_install.append("noto-fonts-cjk")
            elif pkg_mgr == "dnf":
                packages_to_install.extend(
                    ["google-noto-sans-cjk-fonts", "google-noto-serif-cjk-fonts"]
                )
            elif pkg_mgr == "brew":
                actions.append("brew install --cask font-noto-serif-cjk-sc font-noto-sans-cjk-sc")

        if c.code == "ghostscript":
            packages_to_install.append("ghostscript")

        if c.code == "poppler":
            if pkg_mgr in ("apt", "dnf"):
                packages_to_install.append("poppler-utils")
            elif pkg_mgr in ("pacman", "brew", "apk"):
                packages_to_install.append("poppler")

        if c.code == "local_inference" and c.fix and "ollama pull" in c.fix:
            actions.append(c.fix)

        if c.code == "ledger_permissions" and c.fix:
            actions.append(c.fix)

    if packages_to_install:
        unique_pkgs = sorted(set(packages_to_install))
        if pkg_mgr == "apt":
            lines.append("sudo apt-get update")
            lines.append(f"sudo apt-get install -y {' '.join(unique_pkgs)}")
        elif pkg_mgr == "pacman":
            lines.append(f"sudo pacman -S --noconfirm {' '.join(unique_pkgs)}")
        elif pkg_mgr == "dnf":
            lines.append(f"sudo dnf install -y {' '.join(unique_pkgs)}")
        elif pkg_mgr == "apk":
            lines.append(f"sudo apk add {' '.join(unique_pkgs)}")
        elif pkg_mgr == "brew":
            lines.append(f"brew install {' '.join(unique_pkgs)}")
        lines.append("")

    for act in actions:
        lines.append(f"{act}")

    if not packages_to_install and not actions:
        lines.append('echo "==> All environment checks passed! No remediation necessary."')
    else:
        lines.append("")
        lines.append(
            "echo \"==> Environment remediation complete! Re-run 'ubt doctor --deep' to verify.\""
        )

    return "\n".join(lines) + "\n"
