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

    def _no_metadata(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _no_metadata)
    reloaded = importlib.reload(ubt)
    assert reloaded.__version__ == "0+unknown"
