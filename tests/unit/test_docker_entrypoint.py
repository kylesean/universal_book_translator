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


def _minimal_env(tmp_path: Path) -> dict[str, str]:
    """Only what the script needs to reach its key check.

    The storage directories point into ``tmp_path`` so the script's ``mkdir``
    does not touch ``/data``, and no ``UBT_*`` variable is inherited.
    """
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "UBT_DB_DIR": str(tmp_path / "dbs"),
        "UBT_OUTPUT_DIR": str(tmp_path / "output"),
        "UBT_ALLOWED_DIRS": str(tmp_path),
    }


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
