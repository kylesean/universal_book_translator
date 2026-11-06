"""The container entrypoint must refuse to start the console without a key.

The script is the last layer between an operator's ``docker run`` and an
unauthenticated console, so the fail-closed branch is exercised as a real
process rather than by reading the script text.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_ENTRYPOINT = Path(__file__).resolve().parents[2] / "deploy" / "docker" / "entrypoint.sh"


def _minimal_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """Only what the script needs to reach its key check.

    The storage directories point into ``tmp_path`` so the script's ``mkdir``
    does not touch ``/data``, and no ``UBT_*`` variable is inherited.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "UBT_DB_DIR": str(tmp_path / "dbs"),
        "UBT_OUTPUT_DIR": str(tmp_path / "output"),
        "UBT_ALLOWED_DIRS": str(tmp_path),
    }
    env.update(extra)
    return env


def test_the_entrypoint_exits_one_without_an_api_key(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise the container entrypoint")

    result = subprocess.run(
        [bash, str(_ENTRYPOINT), "api"],
        env=_minimal_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1, result.stderr
    assert "UBT_API_KEY is not set" in result.stderr


def test_an_explicit_keyless_opt_in_starts_the_console(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise the container entrypoint")

    # The script execs ``ubt-api``, which would start a real server and hang.
    # A stub on PATH records the invocation and exits immediately instead.
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "ubt-api"
    stub.write_text('#!/bin/sh\necho "started $*"\n', encoding="utf-8")
    stub.chmod(0o755)
    env = _minimal_env(tmp_path, UBT_ALLOW_NO_AUTH="1")
    env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        [bash, str(_ENTRYPOINT), "api"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "started --host 0.0.0.0 --port 8000" in result.stdout
    assert "UBT_API_KEY is not set" not in result.stderr
