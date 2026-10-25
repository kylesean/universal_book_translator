"""The doctor must not measure what the platform cannot measure.

The ledger-permissions probe reads POSIX mode bits (``st_mode & 0o077``). On
Windows those bits are synthesized — a writable file always reports 0666 — so
the scan would flag every file as group/other-readable and the printed remedy
(``chmod -R go-rwx``) would be noise. The check therefore SKIPs there, the
runtime warning stays silent, and the console's forbidden-path panel renders
the deny list the server actually enforces instead of hardcoded Unix paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

# collect_checks lazily imports pikepdf (and friends) inside the PDF checks;
# those imports branch on sys.platform, so they must be primed while the test
# process still claims the real platform, before any test fakes win32.
import pikepdf  # noqa: F401
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.cli.commands.doctor import collect_checks
from ubt.core.config import UBTConfig
from ubt.core.fs_perms import warn_world_readable

pytestmark = pytest.mark.fast


def _config(tmp_path: Path) -> UBTConfig:
    return UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr("platform-test-key"),
    )


def test_ledger_permissions_skips_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Called through the extracted helper, not collect_checks: the other probes
    # shell out (shutil.which), and CPython's own win32 branches crash under a
    # faked platform — this test is about OUR platform gate, not theirs.
    monkeypatch.setattr(sys, "platform", "win32")
    from ubt.cli.commands.doctor import _record_ledger_permissions

    recorded: list[tuple[str, str | None]] = []

    def _record(name: str, status: str, detail: str, fix: str | None = None, **kw: object) -> None:
        recorded.append((status, kw.get("detail_code") or None))

    _record_ledger_permissions(_config(tmp_path), _record)
    assert recorded == [("SKIP", "ledger_permissions.windows_acl")]


def test_ledger_permissions_still_scans_on_posix(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX bit semantics only exist off Windows")
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)
    world_readable = db_dir / "ledger.sqlite"
    world_readable.write_bytes(b"x")
    world_readable.chmod(0o644)
    checks = collect_checks(_config(tmp_path), probe=False)
    perm = next(c for c in checks if c.code == "ledger_permissions")
    assert perm.status == "WARN"
    assert perm.detail_code == "ledger_permissions.exposed"


def test_runtime_warning_never_scans_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the POSIX-bit scan must not run on Windows")

    monkeypatch.setattr("ubt.core.fs_perms.world_readable_files", _boom)
    warn_world_readable(tmp_path)


def test_boundary_endpoint_reports_the_real_deny_list(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))
    body = client.get("/system/boundary", headers={"X-API-Key": "platform-test-key"}).json()
    prefixes = body["forbidden_prefixes"]
    assert prefixes, "the server always refuses something"
    if sys.platform == "win32":
        assert any("Windows" in p or "Program" in p for p in prefixes)
    else:
        assert "/etc" in prefixes
