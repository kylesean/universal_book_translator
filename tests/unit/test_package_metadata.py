"""The package's own version surface: packaging metadata, with a fallback.

``ubt.__version__`` has exactly one source (the installed distribution's
metadata) and exactly one fallback (a source checkout that was never
installed reads ``0+unknown``). Both halves are pinned here: the happy path
through ``importlib.metadata``, and the fallback by re-importing the root
package with the metadata lookup poisoned — the branch is otherwise
unreachable on any machine that ran ``uv sync``, which is every machine the
suite runs on.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Iterator

import pytest

import ubt

pytestmark = pytest.mark.fast


@pytest.fixture()
def _restore_root_package() -> Iterator[None]:
    yield
    importlib.reload(ubt)


def test_the_version_comes_from_the_installed_distribution() -> None:
    assert ubt.__version__ == importlib.metadata.version("universal-book-translator")


def test_an_uninstalled_checkout_reports_0_unknown(
    monkeypatch: pytest.MonkeyPatch, _restore_root_package: None
) -> None:
    """The except branch of ubt.__init__'s version lookup, reached for real."""

    def _no_metadata(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _no_metadata)
    reloaded = importlib.reload(ubt)
    assert reloaded.__version__ == "0+unknown"
