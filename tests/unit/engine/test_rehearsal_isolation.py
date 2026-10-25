"""A rehearsal must not put anything in the directory an operator reads.

Before this, a ``--dry-run`` wrote its ledger, uploads and deliverables into the
same ``.ubt/ledgers/`` as real work, distinguished only by a ``_mock`` segment in
the job id. Nothing prunes a ledger, so every rehearsal left a permanent job in
the console — seven of them after one afternoon of validating the render path —
and telling them from real work meant reading the job id.

The guard here is structural: the rehearsal's config points at a *different*
directory, and the real one is never touched. Asserted on what the orchestrator
would write, so it fails if the isolation is quietly undone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator, rehearsal_db_dir

pytestmark = pytest.mark.fast


def test_the_rehearsal_directory_is_a_sibling_not_a_child(tmp_path: Path) -> None:
    """A child of ``ledgers`` would still be walked by anything scanning it.

    Most of the code this protects enumerates ``<db_dir>/*.sqlite``; a
    ``ledgers/mock`` subtree would sit inside that glob's parent and keep
    answering the wrong question.
    """
    real = tmp_path / ".ubt" / "ledgers"
    mock = rehearsal_db_dir(real)

    assert mock == tmp_path / ".ubt" / "ledgers-mock"
    assert mock.parent == real.parent
    assert not mock.is_relative_to(real), f"{mock} must not live under {real}"


def test_a_rehearsal_orchestrator_does_not_point_at_the_real_ledger(tmp_path: Path) -> None:
    """The isolation that matters: the ledger directory the run will write to."""
    real = tmp_path / ".ubt" / "ledgers"
    config = UBTConfig(db_dir=real)

    assembler = create_dry_run_orchestrator(config)

    assert assembler.config.db_dir == rehearsal_db_dir(real)
    # And the caller's config is untouched — a rehearsal must not mutate the
    # operator's config as a side effect of running.
    assert config.db_dir == real


def test_a_relative_db_dir_stays_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default is ``.ubt/ledgers``; the sibling must not become absolute.

    ``UBTConfig`` anchors a *untouched* default to the config file's directory,
    which is why an absolute result here would be a silent relocation of the
    rehearsal store rather than a bug in the sibling rule.
    """
    monkeypatch.chdir(tmp_path)
    assert rehearsal_db_dir(Path(".ubt/ledgers")) == Path(".ubt/ledgers-mock")


def test_the_rehearsal_still_closes_every_egress_path(tmp_path: Path) -> None:
    """Restating the existing guarantees, because the db_dir change edits the same block.

    A rehearsal's whole promise is zero spend and zero page upload; the config
    copy that moves the ledger must not have dropped the other three overrides.
    """
    assembler = create_dry_run_orchestrator(UBTConfig(db_dir=tmp_path / "ledgers"))
    config = assembler.config

    assert config.ocr_mode == "off"
    assert config.allow_page_upload is False
    assert config.visual_judge_enabled is False
