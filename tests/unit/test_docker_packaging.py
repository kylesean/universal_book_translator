"""Tests for Docker container packaging and All-in-One deployment contracts (P2-3).

Validates:
1. Multi-stage Dockerfile configuration, Typst 0.15.1 pinned checksums (amd64 and arm64),
   and fonts-noto-cjk font setup.
2. Non-root user execution, storage volume layout, healthchecks, and exposed ports.
3. docker-compose.yml service topology and environment configuration.
4. deploy/docker/entrypoint.sh dispatch contracts and script syntax.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO_ROOT / "Dockerfile"
_COMPOSE_FILE = _REPO_ROOT / "docker-compose.yml"
_DOCKERIGNORE = _REPO_ROOT / ".dockerignore"
_ENTRYPOINT = _REPO_ROOT / "deploy" / "docker" / "entrypoint.sh"


def test_dockerfile_structure_and_pinned_dependencies() -> None:
    """Dockerfile must be a valid multi-stage build pinning Typst and Noto CJK fonts."""
    assert _DOCKERFILE.is_file(), "Root Dockerfile must exist"
    content = _DOCKERFILE.read_text(encoding="utf-8")

    # Multi-stage build
    assert "FROM python:3.12-slim-bookworm AS builder" in content
    assert "FROM python:3.12-slim-bookworm AS runtime" in content

    # Pinned Typst version and SHA256 checksums
    assert 'TYPST_VERSION="0.15.1"' in content
    assert (
        'TYPST_AMD64_SHA256="a6d077d0a95eed5a2eba715b2dae06be954f624ccbf85758a03f389ded33118c"'
        in content
    )
    assert (
        'TYPST_ARM64_SHA256="5aa8d74a3d906e60ea12a66ac2f37f8eef1b14cbad7182a745e393a10c23dcee"'
        in content
    )

    # CJK fonts and fontconfig for zero-tofu rendering
    assert "fonts-noto-cjk" in content
    assert "fontconfig" in content
    assert "fc-cache" in content

    # Security hardening: Non-root user (least privilege)
    assert "useradd" in content and "10001" in content
    assert "USER ubt" in content

    # Storage volume & exposed port
    assert 'VOLUME ["/data"]' in content
    assert "EXPOSE 8000" in content

    # Healthcheck checking /health endpoint
    assert "HEALTHCHECK" in content
    assert "http://localhost:8000/health" in content


def test_dockerignore_excludes_unnecessary_files() -> None:
    """Ensure .dockerignore strips source control, caches, test assets, and temporary files."""
    assert _DOCKERIGNORE.is_file(), ".dockerignore must exist"
    lines = {
        line.strip()
        for line in _DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert ".git" in lines
    assert ".venv" in lines
    assert "__pycache__" in lines
    assert "tests" in lines
    assert "tmp" in lines


def test_docker_compose_specification() -> None:
    """docker-compose.yml must define the ubt service with volume and healthcheck."""
    assert _COMPOSE_FILE.is_file(), "docker-compose.yml must exist"
    content = _COMPOSE_FILE.read_text(encoding="utf-8")

    # Service definitions
    assert "ubt:" in content
    assert "image: universal-book-translator:latest" in content
    assert "ports:" in content
    assert ':8000"' in content
    assert "./data:/data" in content

    # Container environment variables
    assert "UBT_HOST=0.0.0.0" in content
    assert "UBT_PORT=8000" in content
    assert "UBT_ALLOWED_DIRS=/data" in content
    assert "UBT_DB_DIR=/data/dbs" in content
    assert "UBT_OUTPUT_DIR=/data/output" in content


def test_container_entrypoint_script_syntax_and_executability() -> None:
    """deploy/docker/entrypoint.sh must exist, be executable, and define proper command routing."""
    assert _ENTRYPOINT.is_file(), "Entrypoint script must exist"
    assert os.access(_ENTRYPOINT, os.X_OK), "Entrypoint script must be executable (+x)"

    content = _ENTRYPOINT.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash")
    assert "set -eo pipefail" in content
    assert "ubt-api" in content
    assert "ubt doctor" in content
    assert "ubt translate" in content
    assert "ubt worker" in content
    assert "ubt-mcp" in content
