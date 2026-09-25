"""The CometKiwi scorer must resolve from the *installed package*, not the cwd.

A wheel install carries ``packages=["ubt"]`` only: there is no ``scripts/``
directory. The historical cwd-relative probe (``Path("scripts/comet_score_ipc.py")``)
therefore returned ``None`` for every wheel user, and ``SubprocessQERunner``
silently fell back to the twelve-band heuristic scorer while the job still
reported a "neural" QE configuration. These tests pin the packaged location so
that regression cannot come back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.config import UBTConfig

pytestmark = pytest.mark.fast


def test_comet_script_resolves_independently_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``comet_script_path`` must be a real file even from an unrelated cwd."""
    monkeypatch.delenv("UBT_COMET_SCRIPT_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = UBTConfig.from_env()
    assert cfg.comet_script_path is not None, (
        "packaged comet scorer was not resolved; a cwd-relative probe returns "
        "None for every wheel install"
    )
    assert Path(cfg.comet_script_path).is_file()
    assert Path(cfg.comet_script_path).name == "comet_score_ipc.py"


def test_comet_scorer_ships_inside_ubt_package() -> None:
    """The scorer lives under ``ubt/`` so ``packages=["ubt"]`` includes it."""
    import ubt

    pkg_root = Path(ubt.__file__).resolve().parent
    packaged = pkg_root / "core" / "qe" / "comet_score_ipc.py"
    assert packaged.is_file(), f"expected packaged scorer at {packaged}"
