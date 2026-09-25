"""Router decide(): unified-entry short vs long chain routing."""

from pathlib import Path

import pytest

from tests.pdf_builders import blank_pdf, text_pdf
from ubt.core.router_mode import decide


def test_short_born_digital_chapter(tmp_path: Path) -> None:
    pdf = text_pdf(tmp_path / "ch26.pdf", 26)
    d = decide(pdf)
    assert d.mode == "short"
    assert d.pages == 26
    assert d.to_dict()["mode"] == "short"


def test_long_book_routes_long(tmp_path: Path) -> None:
    pdf = text_pdf(tmp_path / "long.pdf", 31, chars_per_page=100)
    d = decide(pdf, short_max_pages=30)
    assert d.mode == "long"


def test_blank_pdf_routes_long(tmp_path: Path) -> None:
    assert decide(blank_pdf(tmp_path / "blank.pdf")).mode == "long"


def test_forced_modes_win(tmp_path: Path) -> None:
    pdf = text_pdf(tmp_path / "ch.pdf", 26)
    assert decide(pdf, exec_mode="long").mode == "long"
    assert decide(pdf, exec_mode="short").mode == "short"


def test_missing_file_routes_long(tmp_path: Path) -> None:
    assert decide(tmp_path / "nope.pdf").mode == "long"


def test_short_max_pages_config_override_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """decide() reads the validated config at call time, not the import-time constant.

    A value set after import must move the decision — the import-time
    ``SHORT_CHAIN_MAX_PAGES`` would have ignored it.
    """
    pdf = text_pdf(tmp_path / "ch10.pdf", 10)
    monkeypatch.setenv("UBT_SHORT_MAX_PAGES", "5")
    assert decide(pdf).mode == "long"  # 10 pages > 5
    monkeypatch.setenv("UBT_SHORT_MAX_PAGES", "20")
    assert decide(pdf).mode == "short"  # 10 pages <= 20
