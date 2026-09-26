"""Unit tests for human PE revision re-import: ledger update + TM human_pe."""
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pe_import import PEImportError, import_pe_revisions
from ubt.core.engine.pe_queue import export_pe_queue_csv, export_pe_queue_xliff
from ubt.core.ir.models import BlockStatus, FlowID, IRBlock
from ubt.core.memory.tm import TranslationMemory

_XLIFF_NS = "urn:oasis:names:tc:xliff:document:2.1"

_SOURCE_A = "The reactor outputs 42 megawatts daily."
_SOURCE_B = "Mr. Darcy entered the room."


def _blocks() -> list[IRBlock]:
    return [
        IRBlock(
            id="ch01#b001",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            source_text=_SOURCE_A,
            draft_text="反应堆每天输出84兆瓦。",
            target_text="反应堆每天输出84兆瓦。",
            status=BlockStatus.BLOCKED_HUMAN,
            mtqe_score=0.42,
            mqm_severity="critical",
        ),
        IRBlock(
            id="ch01#b002",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            source_text=_SOURCE_B,
            draft_text="德西先生走进了房间。",
            target_text="德西先生走进了房间。",
            status=BlockStatus.NEEDS_HUMAN,
            mtqe_score=0.62,
            mqm_severity="major",
        ),
    ]


def _ledger(tmp_path: Path) -> SQLiteJobLedger:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger,
        "job_pe",
        SeedDoc(
            doc_id="test_doc_sha256",
            source_path="/tmp/test_book.epub",
            format_type="epub",
            metadata={"title": "Test Book", "source_lang": "en"},
            blocks=_blocks(),
        ),
        target_lang="zh",
    )
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch01#b001",
                "status": BlockStatus.BLOCKED_HUMAN,
                "target_text": "反应堆每天输出84兆瓦。",
                "mtqe_score": 0.42,
            },
            {
                "block_id": "ch01#b002",
                "status": BlockStatus.NEEDS_HUMAN,
                "target_text": "德西先生走进了房间。",
                "mtqe_score": 0.62,
            },
        ]
    )
    return ledger


def test_csv_reimport_updates_ledger_and_tm(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    tm = TranslationMemory(tmp_path / "tm.sqlite")

    queue_blocks = {b.id: b for b in ledger.get_all_blocks("job_pe")}
    csv_path = tmp_path / "queue.csv"
    export_pe_queue_csv(list(queue_blocks.values()), csv_path, job_id="job_pe")

    # Human fills in the revised_translation column for both segments.
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if row["block_id"] == "ch01#b001":
            row["revised_translation"] = "反应堆每天输出42兆瓦。"
        elif row["block_id"] == "ch01#b002":
            row["revised_translation"] = "达西先生走进了房间。"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    # No explicit langs: the pair must be derived from the ledger (en/zh),
    # never from a caller-side default.
    result = import_pe_revisions(
        job_id="job_pe",
        file_path=csv_path,
        ledger=ledger,
        tm=tm,
    )

    assert result.total_rows == 2
    assert result.imported == 2
    assert result.tm_written == 2
    assert set(result.revised_block_ids) == {"ch01#b001", "ch01#b002"}

    updated = {b.id: b for b in ledger.get_all_blocks("job_pe")}
    assert updated["ch01#b001"].status == BlockStatus.REPAIRED
    assert updated["ch01#b001"].target_text == "反应堆每天输出42兆瓦。"
    assert "human_pe_imported" in updated["ch01#b001"].error_flags
    assert updated["ch01#b002"].target_text == "达西先生走进了房间。"

    # Revisions flow into the shared TM as human_pe provenance.
    hit_a = tm.lookup_exact("en", "zh", _SOURCE_A)
    assert hit_a is not None
    assert hit_a.provenance == "human_pe"
    assert hit_a.target_text == "反应堆每天输出42兆瓦。"
    hit_b = tm.lookup_exact("en", "zh", _SOURCE_B)
    assert hit_b is not None and hit_b.provenance == "human_pe"

    ledger.close()
    tm.close()


def test_xliff_reimport_updates_ledger_and_tm(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    tm = TranslationMemory(tmp_path / "tm.sqlite")

    queue_blocks = list(ledger.get_all_blocks("job_pe"))
    xliff_path = tmp_path / "queue.xliff"
    export_pe_queue_xliff(
        queue_blocks, xliff_path, source_lang="en", target_lang="zh", job_id="job_pe"
    )

    # Human edits the target of the first unit only.
    tree = ET.parse(xliff_path)
    root = tree.getroot()
    units = root.findall(f".//{{{_XLIFF_NS}}}unit")
    assert units[0].get("id") == "ch01#b001"
    target_el = units[0].find(f".//{{{_XLIFF_NS}}}target")
    assert target_el is not None
    target_el.text = "反应堆每天输出42兆瓦。"
    tree.write(xliff_path, encoding="utf-8", xml_declaration=True)

    result = import_pe_revisions(
        job_id="job_pe",
        file_path=xliff_path,
        ledger=ledger,
        tm=tm,
        source_lang="en",
        target_lang="zh",
    )

    assert result.total_rows == 2  # both units carry non-empty targets
    assert result.imported == 1  # only unit 1 differs from the ledger target
    assert result.skipped == 1  # unit 2 is unchanged -> skipped
    assert result.tm_written == 1

    updated = {b.id: b for b in ledger.get_all_blocks("job_pe")}
    assert updated["ch01#b001"].target_text == "反应堆每天输出42兆瓦。"
    assert updated["ch01#b001"].status == BlockStatus.REPAIRED
    # The second block remains untouched in NEEDS_HUMAN.
    assert updated["ch01#b002"].status == BlockStatus.NEEDS_HUMAN

    hit = tm.lookup_exact("en", "zh", _SOURCE_A)
    assert hit is not None and hit.provenance == "human_pe"

    ledger.close()
    tm.close()


def test_reimport_skips_empty_unknown_and_unchanged(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    csv_path = tmp_path / "queue.csv"
    export_pe_queue_csv(list(ledger.get_all_blocks("job_pe")), csv_path, job_id="job_pe")

    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["block_id"] = "ch01#does_not_exist"
    rows[0]["revised_translation"] = "某个修订。"
    # rows[1] keeps revised_translation empty -> skipped
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    result = import_pe_revisions(job_id="job_pe", file_path=csv_path, ledger=ledger, tm=None)
    assert result.total_rows == 1
    assert result.imported == 0
    assert result.skipped == 1

    updated = {b.id: b for b in ledger.get_all_blocks("job_pe")}
    assert updated["ch01#b001"].status == BlockStatus.BLOCKED_HUMAN
    ledger.close()


def test_reimport_rejects_unbound_file(tmp_path: Path) -> None:
    """A CSV with no job_id column cannot be proven to belong to this ledger."""
    ledger = _ledger(tmp_path)
    csv_path = tmp_path / "unbound.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["block_id", "revised_translation"])
        writer.writeheader()
        writer.writerow({"block_id": "ch01#b001", "revised_translation": "反应堆每天输出42兆瓦。"})
    with pytest.raises(PEImportError, match="no job binding"):
        import_pe_revisions(job_id="job_pe", file_path=csv_path, ledger=ledger, tm=None)
    # Nothing was applied.
    updated = {b.id: b for b in ledger.get_all_blocks("job_pe")}
    assert updated["ch01#b001"].status == BlockStatus.BLOCKED_HUMAN
    ledger.close()


def test_reimport_rejects_foreign_job(tmp_path: Path) -> None:
    """A file bound to a different job is refused (the same-book cross-lang case)."""
    ledger = _ledger(tmp_path)
    csv_path = tmp_path / "foreign.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["block_id", "job_id", "revised_translation"])
        writer.writeheader()
        writer.writerow(
            {
                "block_id": "ch01#b001",
                "job_id": "job_pe__other_lang",
                "revised_translation": "反应堆每天输出42兆瓦。",
            }
        )
    with pytest.raises(PEImportError, match="cross-job import"):
        import_pe_revisions(job_id="job_pe", file_path=csv_path, ledger=ledger, tm=None)
    updated = {b.id: b for b in ledger.get_all_blocks("job_pe")}
    assert updated["ch01#b001"].status == BlockStatus.BLOCKED_HUMAN
    ledger.close()


def test_reimport_rejects_unknown_job(tmp_path: Path) -> None:
    """Importing into a job id the ledger does not contain is refused before parsing."""
    ledger = _ledger(tmp_path)
    csv_path = tmp_path / "queue.csv"
    export_pe_queue_csv(list(ledger.get_all_blocks("job_pe")), csv_path, job_id="job_pe")
    with pytest.raises(PEImportError, match="no job"):
        import_pe_revisions(job_id="nope_missing", file_path=csv_path, ledger=ledger, tm=None)
    ledger.close()


def test_reimport_rejects_malformed_files(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("id,text\n1,2\n", encoding="utf-8")
    with pytest.raises(PEImportError):
        import_pe_revisions(job_id="job_pe", file_path=bad_csv, ledger=ledger)

    bad_xml = tmp_path / "bad.xliff"
    bad_xml.write_text("<html><body>not xliff</body></html>", encoding="utf-8")
    with pytest.raises(PEImportError):
        import_pe_revisions(job_id="job_pe", file_path=bad_xml, ledger=ledger)

    unsupported = tmp_path / "queue.xlsx"
    unsupported.write_text("whatever", encoding="utf-8")
    with pytest.raises(PEImportError):
        import_pe_revisions(job_id="job_pe", file_path=unsupported, ledger=ledger)

    ledger.close()


def test_reimport_rejects_non_queue_blocks(tmp_path: Path) -> None:
    """16: a stray row must not overwrite a block outside the PE queue."""
    ledger = _ledger(tmp_path)
    ledger.save_checkpoint(
        block_id="ch01#b002",
        status=BlockStatus.MTQE_PASSED,
        target_text="德西先生走进了房间。",
    )
    csv_path = tmp_path / "queue.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["block_id", "job_id", "revised_translation"])
        writer.writeheader()
        writer.writerow(
            {"block_id": "ch01#b002", "job_id": "job_pe", "revised_translation": "某个未审改法。"}
        )
    result = import_pe_revisions(job_id="job_pe", file_path=csv_path, ledger=ledger, tm=None)
    assert result.imported == 0
    assert result.skipped == 1
    updated = {b.id: b for b in ledger.get_all_blocks("job_pe")}
    assert updated["ch01#b002"].target_text == "德西先生走进了房间。"
    assert updated["ch01#b002"].status == BlockStatus.MTQE_PASSED
    ledger.close()


def test_reimport_clears_stale_machine_verdict(tmp_path: Path) -> None:
    """16: a human-fixed block drops stale flags/scores measured on the old draft."""
    ledger = _ledger(tmp_path)
    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.BLOCKED_HUMAN,
        error_flags=["mqm_critical_blocked", "Repair error: boom"],
    )
    csv_path = tmp_path / "queue.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["block_id", "job_id", "revised_translation"])
        writer.writeheader()
        writer.writerow(
            {
                "block_id": "ch01#b001",
                "job_id": "job_pe",
                "revised_translation": "反应堆每天输出42兆瓦。",
            }
        )
    result = import_pe_revisions(job_id="job_pe", file_path=csv_path, ledger=ledger, tm=None)
    assert result.imported == 1
    updated = {b.id: b for b in ledger.get_all_blocks("job_pe")}
    blk = updated["ch01#b001"]
    assert blk.status == BlockStatus.REPAIRED
    assert blk.error_flags == ["human_pe_imported"]
    assert blk.mtqe_score is None
    assert blk.mqm_severity is None
    ledger.close()
