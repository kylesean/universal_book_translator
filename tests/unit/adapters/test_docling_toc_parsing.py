from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.docling_blocks import parse_toc_entry_line
from ubt.core.config import UBTConfig
from ubt.core.job_options import adaptive_dual_mode
from ubt.render.outputs import TypstFragmentTypesetter

pytestmark = pytest.mark.fast


def test_parse_toc_entry_line_leaded() -> None:
    line = "1.1 Programming languages . . . . . . . . . . . . . . . . . . . . . . . . . 3"
    assert parse_toc_entry_line(line) == ("1.1 Programming languages", "3", True)

    dot_leader = "1.2 Input and output ··········· 4"
    assert parse_toc_entry_line(dot_leader) == ("1.2 Input and output", "4", True)


def test_parse_toc_entry_line_unleaded_chapters_and_parts() -> None:
    assert parse_toc_entry_line("1 Introduction 3") == ("1 Introduction", "3", False)
    assert parse_toc_entry_line("Preface ix") == ("Preface", "ix", False)
    assert parse_toc_entry_line("I Basic techniques 1") == ("I Basic techniques", "1", False)
    assert parse_toc_entry_line("2 Time complexity 17") == ("2 Time complexity", "17", False)
    assert parse_toc_entry_line("Bibliography 281") == ("Bibliography", "281", False)


def test_parse_toc_entry_line_non_toc_rejected() -> None:
    assert parse_toc_entry_line("Contents") is None
    assert parse_toc_entry_line("iii") is None
    assert parse_toc_entry_line("") is None
    assert parse_toc_entry_line("   ") is None


def test_adaptive_dual_mode_textbook_defaults_to_monolingual() -> None:
    assert adaptive_dual_mode(None, "textbook") == "monolingual"
    assert adaptive_dual_mode(None, "paper") == "monolingual"
    assert adaptive_dual_mode(None, "fiction") == "monolingual"
    assert adaptive_dual_mode(None, "novel") == "monolingual"
    assert adaptive_dual_mode(None, "general") is None

    # Explicit mode always wins
    assert adaptive_dual_mode("inline", "textbook") == "inline"
    assert adaptive_dual_mode("facing", "paper") == "facing"


def test_ubt_config_dual_mode_defaults_to_auto() -> None:
    config = UBTConfig()
    assert config.dual_mode == "auto"


def test_toc_source_markup_leaders_toggle(tmp_path: Path) -> None:
    typesetter = TypstFragmentTypesetter(cache_dir=tmp_path)
    source_with_leaders = typesetter._toc_source(
        "1.1 Introduction", "3", 400.0, 15.0, 10.0, has_leaders=True
    )
    assert "repeat(gap: 3.5pt)[.]" in source_with_leaders

    source_unleaded = typesetter._toc_source(
        "1 Introduction", "3", 400.0, 15.0, 10.0, has_leaders=False
    )
    assert "repeat(gap: 3.5pt)[.]" not in source_unleaded
    assert "#h(1fr)" in source_unleaded
