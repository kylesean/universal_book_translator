from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator, rehearsal_db_dir

pytestmark = pytest.mark.fast


def test_the_rehearsal_directory_is_a_sibling_not_a_child(tmp_path: Path) -> None:
    real = tmp_path / ".ubt" / "ledgers"
    mock = rehearsal_db_dir(real)

    assert mock == tmp_path / ".ubt" / "ledgers-mock"
    assert mock.parent == real.parent
    assert not mock.is_relative_to(real), f"{mock} must not live under {real}"


def test_a_rehearsal_orchestrator_does_not_point_at_the_real_ledger(tmp_path: Path) -> None:
    real = tmp_path / ".ubt" / "ledgers"
    config = UBTConfig(db_dir=real)

    assembler = create_dry_run_orchestrator(config)

    assert assembler.config.db_dir == rehearsal_db_dir(real)
    assert config.db_dir == real


def test_a_relative_db_dir_stays_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert rehearsal_db_dir(Path(".ubt/ledgers")) == Path(".ubt/ledgers-mock")


def test_the_rehearsal_still_closes_every_egress_path(tmp_path: Path) -> None:
    assembler = create_dry_run_orchestrator(UBTConfig(db_dir=tmp_path / "ledgers"))
    config = assembler.config

    assert config.ocr_mode == "off"
    assert config.allow_page_upload is False
    assert config.visual_judge_enabled is False
