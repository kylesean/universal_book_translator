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


def test_markdown_counts_all_heading_levels(tmp_path: Path) -> None:
    """An H2-only multi-chapter document must not read as one short chapter."""
    md = tmp_path / "book.md"
    md.write_text(
        "\n\n".join(f"## Chapter {i}\n" + "word " * 400 for i in range(1, 6)),
        encoding="utf-8",
    )
    assert decide(md).chapters == 5


def test_html_strip_markup_ignores_script_and_style(tmp_path: Path) -> None:
    """<script>/<style> bodies are not prose; counting them inflated the estimate."""
    html = tmp_path / "page.html"
    html.write_text(
        "<html><head><style>"
        + "body{color:red;}" * 500
        + "</style><script>"
        + "var x=1;" * 500
        + "</script></head><body><h2>Real</h2><p>"
        + "word " * 50
        + "</p></body></html>",
        encoding="utf-8",
    )
    # Only the visible prose counts; the ~6 KB of script/style noise must be gone.
    assert decide(html).chars < 2000


def test_non_pdf_token_estimate_is_script_aware(tmp_path) -> None:
    """A CJK source must not be priced as if it were English (N12).

    ``_probe_non_pdf`` used ``chars // 4`` (the ASCII rule), under-counting a
    Chinese markdown book by ~3x, so ``ubt assess`` quoted a fraction of the
    real cost.
    """
    from ubt.core.router_mode import decide

    md = tmp_path / "book.md"
    text = "# 标题\n\n" + ("这是一段中文文本，用来测试分词估算。" * 200)
    md.write_text(text, encoding="utf-8")

    decision = decide(doc_path=md)
    assert decision.estimated_tokens > len(text) // 4
