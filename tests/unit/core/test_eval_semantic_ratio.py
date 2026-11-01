from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.eval_semantic import _ledger_stats

pytestmark = pytest.mark.fast


def _ledger(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "job_test.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE blocks ("
        " status TEXT, error_flags_json TEXT, target_text TEXT, draft_text TEXT,"
        " source_text TEXT, skip_translate INTEGER, policy_translate INTEGER,"
        " block_type TEXT)"
    )
    for row in rows:
        conn.execute(
            "INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.get("status", "delivered"),
                row.get("error_flags_json", "[]"),
                row.get("target_text"),
                row.get("draft_text"),
                row.get("source_text", ""),
                row.get("skip_translate", 0),
                row.get("policy_translate"),
                row.get("block_type"),
            ),
        )
    conn.commit()
    conn.close()
    return path


def test_kept_blocks_are_excluded_from_the_translatable_ratio(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path,
        [
            {"target_text": "这是一段翻译得很好的中文段落内容", "skip_translate": 0},
            {"target_text": "这是一段翻译得很好的中文段落内容", "skip_translate": 0},
            {
                "target_text": "Vaswani, A. Attention is all you need. NeurIPS 2017.",
                "skip_translate": 1,
            },
        ],
    )
    stats = _ledger_stats(ledger)

    assert stats["translatable_blocks"] == 2
    assert stats["source_kept_blocks"] == 1
    assert stats["cjk_ratio_translatable"] == pytest.approx(1.0)


def test_a_policy_verdict_outranks_the_extractor_default(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path,
        [
            {"target_text": "中文译文", "skip_translate": 0, "policy_translate": 0},
            {"target_text": "另一段中文译文", "skip_translate": 0, "policy_translate": 1},
        ],
    )
    stats = _ledger_stats(ledger)

    assert stats["source_kept_blocks"] == 1
    assert stats["translatable_blocks"] == 1


def test_a_policy_verdict_of_true_does_not_clear_a_skip_flag(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path,
        [
            {
                "target_text": "这是译文内容",
                "skip_translate": 1,
                "policy_translate": 1,
            },
        ],
    )
    stats = _ledger_stats(ledger)

    from ubt.core.ir.models import BlockType, IRBlock, make_element
    from ubt.render.typst.document_renderer import _keeps_source

    element = make_element(
        id="b1",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="x",
        level=1,
        skip_translate=True,
    )
    renderer_says_kept = _keeps_source(IRBlock(element=element, policy_translate=True))
    assert renderer_says_kept is True

    assert stats["source_kept_blocks"] == 1
    assert stats["translatable_blocks"] == 0


def test_a_draft_substitutes_for_a_missing_target(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path,
        [{"target_text": None, "draft_text": "草稿中的中文内容", "skip_translate": 0}],
    )
    stats = _ledger_stats(ledger)

    assert stats["cjk_ratio_translatable"] == pytest.approx(1.0)


def test_no_translatable_block_yields_none_not_zero(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path,
        [{"target_text": "Vaswani, A. NeurIPS 2017.", "skip_translate": 1}],
    )
    stats = _ledger_stats(ledger)

    assert stats["cjk_ratio_translatable"] is None


def test_both_ratios_are_reported_so_the_gap_is_visible(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path,
        [
            {"target_text": "中文", "skip_translate": 0},
            {"target_text": "averyverylongenglishbibliographyentry", "skip_translate": 1},
        ],
    )
    stats = _ledger_stats(ledger)

    assert stats["cjk_ratio_translatable"] == pytest.approx(1.0)
    assert stats["total_blocks"] == 2
    assert stats["source_kept_blocks"] == 1


def _typst_table(header_cells: str) -> str:
    return (
        "#table(\n"
        "  columns: 2,\n"
        "  inset: 5pt,\n"
        "  align: (col, row) => if row == 0 { center + horizon } else { left + horizon },\n"
        "  stroke: none,\n"
        "  table.hline(stroke: 1.1pt),\n"
        f"  {header_cells},\n"
        "  table.hline(stroke: 0.5pt),\n"
        "  [甲], [1.0],\n"
        "  table.hline(stroke: 1.1pt),\n"
        ")\n"
    )


def test_a_fully_translated_table_scores_one(tmp_path: Path) -> None:
    from scripts.eval_semantic import _table_header_cjk

    typ = tmp_path / "main.typ"
    typ.write_text(_typst_table("[*模型*], [*准确率*]"), encoding="utf-8")
    assert _table_header_cjk(typ) == pytest.approx(1.0)


def test_an_untranslated_table_scores_zero(tmp_path: Path) -> None:
    from scripts.eval_semantic import _table_header_cjk

    typ = tmp_path / "main.typ"
    typ.write_text(_typst_table("[*Model*], [*Accuracy*]"), encoding="utf-8")
    assert _table_header_cjk(typ) == pytest.approx(0.0)


def test_a_verbatim_identifier_lowers_the_ratio_and_is_listed(tmp_path: Path) -> None:
    from scripts.eval_semantic import _table_header_cells, _table_header_cjk

    typ = tmp_path / "main.typ"
    typ.write_text(_typst_table("[*模型*], [*MNIST*]"), encoding="utf-8")

    assert _table_header_cjk(typ) == pytest.approx(0.5)
    assert _table_header_cells(typ) == ["MNIST"]


def test_a_document_with_no_tables_reports_zero(tmp_path: Path) -> None:
    from scripts.eval_semantic import _table_header_cjk

    typ = tmp_path / "main.typ"
    typ.write_text("#set text(size: 10.5pt)\n\nprose only\n", encoding="utf-8")
    assert _table_header_cjk(typ) == pytest.approx(0.0)
