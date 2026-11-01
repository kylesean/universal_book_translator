from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ubt.core.config import UBTConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONSTRUCT = "from ubt.core.config import UBTConfig; UBTConfig()"
_SUBPROCESS_TIMEOUT = 5

_CAPABILITY_OVERRIDES: list[dict[str, str]] = [
    {"UBT_SUPPORTS_TEMPERATURE": "true"},
    {"UBT_SUPPORTS_REASONING_EFFORT": "true"},
    {"UBT_CAPABILITY_PROFILE": "reasoning"},
]


def _clean_env() -> dict[str, str]:
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


@pytest.mark.fast
def test_ubt_config_from_env_does_not_bootstrap_registries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ubt.core.router import pricing

    sentinel = "http://sentinel.invalid/v1"
    before = len(pricing._CUSTOM_FREE_ENDPOINTS)

    monkeypatch.setenv("UBT_IS_FREE", "true")
    monkeypatch.setenv("UBT_BASE_URL", sentinel)

    cfg = UBTConfig.from_env()

    assert cfg.is_free is True, "the environment value must reach the config"
    assert len(pricing._CUSTOM_FREE_ENDPOINTS) == before, (
        "from_env() bootstrapped process globals; it must only build a config"
    )

    cfg.bootstrap_runtime_environment()
    assert len(pricing._CUSTOM_FREE_ENDPOINTS) == before + 1
