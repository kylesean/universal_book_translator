from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.fast

_SUBPROCESS_TIMEOUT = 20

_IMPORT_ONLY = ("ubt.core.engine", "ubt.core.engine.reporter")

_PROBE_IMPORT = (
    "import sys, importlib; "
    "importlib.import_module({module!r}); "
    "print(any(m == 'ubt.adapters' or m.startswith('ubt.adapters.') for m in sys.modules))"
)

_PROBE_MASK = (
    "from ubt.core.engine.reporter import _placeholder_engine; "
    "assert _placeholder_engine() is _placeholder_engine(), "
    "'the engine is resolved once and cached'; "
    "print(_placeholder_engine().mask('hello'))"
)


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("UBT_")}
    env.pop("FORCE_COLOR", None)
    return env


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        env=_clean_env(),
    )


@pytest.mark.parametrize("module", _IMPORT_ONLY)
def test_core_engine_import_does_not_pull_adapters(module: str) -> None:
    proc = _run(_PROBE_IMPORT.format(module=module))
    assert proc.returncode == 0, f"{module}: probe failed\n{proc.stderr[-2000:]}"
    assert proc.stdout.strip() == "False", (
        f"importing {module} loaded ubt.adapters -- the SPI provider is being "
        "resolved at import time again; keep it behind the lazy port lookup."
    )


def test_the_placeholder_engine_accessor_resolves_lazily_and_works() -> None:
    proc = _run(_PROBE_MASK)
    assert proc.returncode == 0, f"mask probe failed\n{proc.stderr[-2000:]}"
    assert proc.stdout.strip() != ""
