"""``UBTConfig()`` construction must be safe in every environment.

``UBTConfig`` is built in the most surprising places -- notably inside
``ModelCapabilityRegistry._load_environment_profiles`` while
``get_default_registry()`` holds its bootstrap lock. A capability override
in the environment makes ``UBTConfig._check_invariants`` call
``get_default_registry()`` again before the singleton exists, so the
registry bootstrap must never build a ``UBTConfig`` (it reads its two
profile env vars directly for exactly this reason).

A hang cannot be observed in-process without poisoning the test session
(the wedged thread holds the lock forever), so each case runs in a
throwaway subprocess with a hard timeout: a positive control without
overrides, plus one case per capability override that reaches the
registry re-entry.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONSTRUCT = "from ubt.core.config import UBTConfig; UBTConfig()"
# A healthy construction is ~0.1s (measured), so 5s is a wide margin that still
# keeps a regression from dominating the fast tier.
_SUBPROCESS_TIMEOUT = 5

# Each override gates the capability-registration branch of
# ``UBTConfig._check_invariants`` (ubt/core/config.py, "Dynamic model
# capability profile registration"), which calls ``get_default_registry()``.
_CAPABILITY_OVERRIDES: list[dict[str, str]] = [
    {"UBT_SUPPORTS_TEMPERATURE": "true"},
    {"UBT_SUPPORTS_REASONING_EFFORT": "true"},
    {"UBT_CAPABILITY_PROFILE": "reasoning"},
]


def _clean_env() -> dict[str, str]:
    """The ambient environment minus every ``UBT_*`` override."""
    return {key: value for key, value in os.environ.items() if not key.startswith("UBT_")}


def _construct(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _CONSTRUCT],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        check=False,
    )


@pytest.mark.fast
def test_config_constructs_without_capability_overrides() -> None:
    # Positive control: proves the subprocess harness itself is healthy, so a
    # failure in the override cases below is the re-entry and not a broken import.
    proc = _construct(_clean_env())
    assert proc.returncode == 0, proc.stderr


@pytest.mark.fast
@pytest.mark.parametrize(
    "override", _CAPABILITY_OVERRIDES, ids=lambda override: next(iter(override))
)
def test_config_with_capability_override_does_not_deadlock(override: dict[str, str]) -> None:
    try:
        proc = _construct({**_clean_env(), **override})
    except subprocess.TimeoutExpired:
        pytest.fail(f"UBTConfig() hung under {override!r} (registry bootstrap re-entry)")
    assert proc.returncode == 0, (
        f"UBTConfig() exited {proc.returncode} for {override!r}:\n{proc.stderr}"
    )
