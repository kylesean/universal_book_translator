"""``UBTConfig()`` construction must be safe in every environment.

``UBTConfig`` is built in the most surprising places -- notably inside
``ModelCapabilityRegistry._load_environment_profiles`` while
``get_default_registry()`` already holds its bootstrap lock. A capability
override in the environment makes ``UBTConfig._check_invariants`` call
``get_default_registry()`` again, re-entering that non-reentrant lock and
deadlocking the process.

The hang cannot be observed in-process without poisoning the test session
(the wedged thread holds the lock forever), so each case runs in a throwaway
subprocess with a hard timeout. Two cases are pinned:

- the default construction (a positive control -- this must keep passing);
- a capability override that triggers the re-entry (a known, confirmed defect,
  marked ``xfail(strict=True)`` so fixing it turns the suite red and forces the
  marker's removal).

See ``docs/reviews/2026-10-02-architecture-review.md`` (H1).
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
# keeps the deadlock case from dominating the fast tier.
_SUBPROCESS_TIMEOUT = 5


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
    # red xfail case below really is the deadlock and not a broken import.
    proc = _construct(_clean_env())
    assert proc.returncode == 0, proc.stderr


@pytest.mark.fast
@pytest.mark.xfail(
    strict=True,
    reason=(
        "H1 (confirmed): a capability override makes UBTConfig._check_invariants "
        "call get_default_registry(), which is already holding the non-reentrant "
        "_bootstrap_lock during ModelCapabilityRegistry construction -> deadlock. "
        "Remove this marker once ubt/core/router/registry.py no longer builds "
        "UBTConfig under the lock."
    ),
)
def test_config_with_capability_override_does_not_deadlock() -> None:
    # One representative trigger: _check_invariants gates on
    # ``capability_profile or supports_temperature is not None or
    # supports_reasoning_effort is not None`` (ubt/core/config.py:1009), so
    # UBT_CAPABILITY_PROFILE and UBT_SUPPORTS_REASONING_EFFORT reach the same
    # get_default_registry() re-entry (both verified to hang identically).
    override = {"UBT_SUPPORTS_TEMPERATURE": "true"}
    try:
        proc = _construct({**_clean_env(), **override})
    except subprocess.TimeoutExpired:
        pytest.xfail("H1 deadlock confirmed: UBTConfig() hung under UBT_SUPPORTS_TEMPERATURE=true")
    assert proc.returncode == 0, (
        f"UBTConfig() exited {proc.returncode} for {override!r} "
        f"instead of hanging or succeeding:\n{proc.stderr}"
    )
