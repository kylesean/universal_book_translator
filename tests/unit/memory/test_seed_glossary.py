"""Contract tests for the curated seed glossaries and external glossary loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ubt.core.memory.seed_glossary import (
    _SEED_FREQUENCY,
    _SEED_PROFILES,
    _SEMICONDUCTOR_SEEDS,
    load_external_glossary,
    seed_entries_for_profile,
)

pytestmark = pytest.mark.fast


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Built-in seed registry
# --------------------------------------------------------------------------- #


def test_seed_frequency_outranks_mined_entries() -> None:
    assert _SEED_FREQUENCY == 10_000


def test_semiconductor_profiles_share_the_same_seed_tuple() -> None:
    assert set(_SEED_PROFILES) == {
        "semiconductor",
        "semiconductor_paper",
        "semiconductor_textbook",
    }
    assert _SEED_PROFILES["semiconductor"] is _SEMICONDUCTOR_SEEDS
    assert _SEED_PROFILES["semiconductor_paper"] is _SEMICONDUCTOR_SEEDS
    assert _SEED_PROFILES["semiconductor_textbook"] is _SEMICONDUCTOR_SEEDS


def test_seed_tuple_contains_keep_latin_blockers_and_canonical_terms() -> None:
    seeds = dict(_SEMICONDUCTOR_SEEDS)
    assert seeds["MOSFET"] == "MOSFET"  # transliteration blocker
    assert seeds["subthreshold swing"] == "亚阈值摆幅"
    assert seeds["gate dielectric"] == "栅介质"


# --------------------------------------------------------------------------- #
# seed_entries_for_profile
# --------------------------------------------------------------------------- #


def test_packaged_glossary_is_loaded_for_semiconductor() -> None:
    entries = seed_entries_for_profile("semiconductor")
    assert entries
    assert all(e.frequency == _SEED_FREQUENCY for e in entries)
    assert all(e.kind == "term" for e in entries)
    assert {"MOSFET", "CMOS"} <= {e.source for e in entries}


def test_external_glossary_takes_precedence_over_builtin_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ubt.core.memory.bible import BibleEntry

    sentinel = [BibleEntry(source="SENTINEL", translation="哨兵", kind="term")]
    monkeypatch.setattr(
        "ubt.core.memory.seed_glossary.load_external_glossary", lambda _path: sentinel
    )
    assert seed_entries_for_profile("semiconductor") == sentinel


def test_profile_prefix_resolves_the_packaged_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ubt.core.memory.bible import BibleEntry

    sentinel = [BibleEntry(source="SENTINEL", translation="哨兵", kind="term")]
    seen: list[str] = []

    def fake(path: Path) -> list[BibleEntry]:
        seen.append(str(path))
        return sentinel

    monkeypatch.setattr("ubt.core.memory.seed_glossary.load_external_glossary", fake)
    # semiconductor_paper resolves to the "semiconductor" directory.
    assert seed_entries_for_profile("semiconductor_paper") == sentinel
    assert seen and "semiconductor/en-zh.json" in seen[0]


def test_unknown_profile_yields_no_seeds() -> None:
    assert seed_entries_for_profile("general") == []
    assert seed_entries_for_profile("") == []


@pytest.mark.parametrize("bad", ["../evil", "semiconductor/evil", "a\\b"])
def test_unsafe_profile_name_is_rejected(bad: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="ubt.core.memory.seed_glossary"):
        assert seed_entries_for_profile(bad) == []
    assert any("Rejecting unsafe glossary profile name" in r.message for r in caplog.records)


def test_builtin_seeds_fall_back_when_external_yields_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("ubt.core.memory.seed_glossary.load_external_glossary", lambda _path: [])
    with caplog.at_level("ERROR", logger="ubt.core.memory.seed_glossary"):
        entries = seed_entries_for_profile("semiconductor")
    assert len(entries) == len(_SEMICONDUCTOR_SEEDS)
    assert all(e.frequency == _SEED_FREQUENCY for e in entries)
    assert any("yielded no entries" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# load_external_glossary — missing / malformed
# --------------------------------------------------------------------------- #


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_external_glossary(tmp_path / "nope.json") == []


def test_directory_path_returns_empty(tmp_path: Path) -> None:
    assert load_external_glossary(tmp_path) == []


def test_malformed_json_returns_empty(tmp_path: Path) -> None:
    path = _write(tmp_path, "bad.json", "{not json")
    assert load_external_glossary(path) == []


# --------------------------------------------------------------------------- #
# load_external_glossary — JSON
# --------------------------------------------------------------------------- #


def test_json_mapping_form(tmp_path: Path) -> None:
    path = _write(tmp_path, "g.json", json.dumps({"MOSFET": "MOSFET", "gate": "栅"}))
    entries = load_external_glossary(path)
    by_source = {e.source: e for e in entries}
    assert by_source["MOSFET"].translation == "MOSFET"
    assert by_source["gate"].frequency == _SEED_FREQUENCY


def test_json_list_form_accepts_alternate_keys(tmp_path: Path) -> None:
    payload = [
        {"source": "a", "translation": "甲"},
        {"term": "b", "target": "乙"},
        {"src": "c", "tgt": "丙"},
    ]
    entries = load_external_glossary(_write(tmp_path, "g.json", json.dumps(payload)))
    assert {e.source for e in entries} == {"a", "b", "c"}


def test_json_list_skips_items_missing_a_side(tmp_path: Path) -> None:
    payload = [{"source": "a"}, {"translation": "乙"}, "not-a-dict"]
    assert load_external_glossary(_write(tmp_path, "g.json", json.dumps(payload))) == []


# --------------------------------------------------------------------------- #
# load_external_glossary — CSV / TSV
# --------------------------------------------------------------------------- #


def test_csv_without_header_uses_first_two_columns(tmp_path: Path) -> None:
    path = _write(tmp_path, "g.csv", "MOSFET,MOSFET\ngate,栅\n")
    entries = load_external_glossary(path)
    assert {e.source for e in entries} == {"MOSFET", "gate"}


def test_csv_header_and_reordered_columns(tmp_path: Path) -> None:
    path = _write(tmp_path, "g.csv", "translation,source\n栅,gate\nMOSFET,MOSFET\n")
    entries = load_external_glossary(path)
    by_source = {e.source: e.translation for e in entries}
    assert by_source == {"gate": "栅", "MOSFET": "MOSFET"}


def test_csv_skips_short_rows_and_blanks(tmp_path: Path) -> None:
    path = _write(tmp_path, "g.csv", "source,translation\ngate,栅\nonly_one\n,\n")
    entries = load_external_glossary(path)
    assert [e.source for e in entries] == ["gate"]


def test_tsv_uses_tab_delimiter(tmp_path: Path) -> None:
    path = _write(tmp_path, "g.tsv", "MOSFET\tMOSFET\ngate\t栅\n")
    entries = load_external_glossary(path)
    assert {e.source for e in entries} == {"MOSFET", "gate"}


def test_empty_csv_returns_empty(tmp_path: Path) -> None:
    assert load_external_glossary(_write(tmp_path, "g.csv", "")) == []


def test_txt_is_parsed_as_comma_csv(tmp_path: Path) -> None:
    path = _write(tmp_path, "g.txt", "MOSFET,MOSFET\n")
    assert [e.source for e in load_external_glossary(path)] == ["MOSFET"]


def test_loaded_entries_pass_clean_guardrails(tmp_path: Path) -> None:
    # A slogan-length source is dropped by clean_bible_entry.
    path = _write(tmp_path, "g.csv", "one two three four five,太长口号\n")
    assert load_external_glossary(path) == []
