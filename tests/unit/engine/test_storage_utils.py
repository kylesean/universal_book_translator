from __future__ import annotations

import logging
import unittest.mock
from pathlib import Path

import pytest

from ubt.core.engine.storage_utils import (
    NETWORK_FILESYSTEMS,
    detect_filesystem_type,
    warn_if_network_filesystem,
)

pytestmark = pytest.mark.fast


def test_detect_filesystem_type_on_local_disk(tmp_path: Path) -> None:
    fs_type, is_net = detect_filesystem_type(tmp_path)
    assert isinstance(fs_type, str)
    assert isinstance(is_net, bool)
    assert is_net is False


def test_detect_filesystem_type_identifies_nfs() -> None:
    mock_mounts = (
        "sysfs /sys sysfs rw 0 0\n"
        "/dev/sda1 / ext4 rw 0 0\n"
        "192.168.1.100:/export/data /mnt/nfs_share nfs4 rw,relatime 0 0\n"
    )
    with (
        unittest.mock.patch("pathlib.Path.open", unittest.mock.mock_open(read_data=mock_mounts)),
        unittest.mock.patch("sys.platform", "linux"),
    ):
        fs_type, is_net = detect_filesystem_type(Path("/mnt/nfs_share/ubt_db"))
        assert fs_type == "nfs4"
        assert is_net is True
        assert "nfs4" in NETWORK_FILESYSTEMS


def test_warn_if_network_filesystem_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with (
        unittest.mock.patch(
            "ubt.core.engine.storage_utils.detect_filesystem_type",
            return_value=("nfs4", True),
        ),
        caplog.at_level(logging.WARNING),
    ):
        res = warn_if_network_filesystem(Path("/mnt/nfs/ledger.db"), "Test ledger")
        assert res == "nfs4"
        assert "network filesystem 'nfs4'" in caplog.text
