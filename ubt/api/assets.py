"""Asset-center helpers: glossary file CRUD and translation-memory access.

The operator console's Language Assets screen edits two things the engine
already owns:

* the **glossary** — an operator-configured external file (``config.glossary_path``)
  that the engine loads through :func:`ubt.core.memory.seed_glossary.load_external_glossary`.
  The console must not invent a second terminology store, so CRUD here rewrites
  that same file in the format it was written in (JSON dict / JSON list /
  CSV / TSV) and the engine keeps reading it unchanged.
* the **translation memory** — the shared ``{db_dir}/tm.sqlite`` store, exposed
  through the existing :class:`ubt.core.memory.tm.TranslationMemory` scan/evict
  surface.

All helpers are pure with respect to process state (they take an explicit path),
so they can be unit-tested without a running server.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ubt.core.memory.seed_glossary import load_external_glossary
from ubt.core.memory.tm import (
    PROVENANCE_HUMAN_PE,
    PROVENANCE_MACHINE,
    TMPendingEntry,
    TranslationMemory,
    normalize_for_tm,
)

#: Column names that identify the source and target columns in a CSV/TSV
#: glossary header. Mirrors ``load_external_glossary`` so a file round-trips.
_HEADER_SOURCE = ("source", "term", "src", "en")
_HEADER_TARGET = ("translation", "target", "tgt", "zh", "cn")


class GlossaryFormatError(ValueError):
    """The glossary file is not in a format the engine can read back."""


def _detect_delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".tsv" else ","


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle, delimiter=_detect_delimiter(path)))


def _csv_columns(rows: list[list[str]]) -> tuple[int, int, int]:
    """Return ``(src_col, tgt_col, start_row)`` for a CSV/TSV glossary."""
    if not rows:
        return 0, 1, 0
    header = [cell.strip().lower() for cell in rows[0]]
    src_col, tgt_col, start_row = 0, 1, 0
    if any(cell in _HEADER_SOURCE for cell in header):
        start_row = 1
        for idx, cell in enumerate(header):
            if cell in _HEADER_SOURCE:
                src_col = idx
            elif cell in _HEADER_TARGET:
                tgt_col = idx
    return src_col, tgt_col, start_row


def read_glossary_terms(path: Path) -> list[dict[str, str]]:
    """Every (source, target) pair in the glossary file, in file order.

    Delegates parsing to the engine's own loader so the console shows exactly
    the terms a run would enforce.
    """
    return [
        {"source": entry.source, "target": entry.translation}
        for entry in load_external_glossary(path)
        if entry.source and entry.translation
    ]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_glossary_term(path: Path, source: str, target: str) -> None:
    """Insert or update one term, preserving the file's existing format."""
    source, target = source.strip(), target.strip()
    if not source or not target:
        raise GlossaryFormatError("source and target must both be non-empty")
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data[source] = target
        elif isinstance(data, list):
            replaced = False
            for item in data:
                if (
                    isinstance(item, dict)
                    and (item.get("source") or item.get("term") or item.get("src")) == source
                ):
                    item["source"] = source
                    item["translation"] = target
                    replaced = True
                    break
            if not replaced:
                data.append({"source": source, "translation": target})
        else:
            raise GlossaryFormatError("unsupported JSON glossary shape")
        _write_json(path, data)
        return

    rows = _read_csv_rows(path)
    src_col, tgt_col, start_row = _csv_columns(rows)
    width = max((len(r) for r in rows), default=2)
    width = max(width, src_col + 1, tgt_col + 1)
    for row in rows[start_row:]:
        padded = row + [""] * (width - len(row))
        if padded[src_col].strip() == source:
            padded[tgt_col] = target
            row[:] = padded
            break
    else:
        new_row = [""] * width
        new_row[src_col] = source
        new_row[tgt_col] = target
        rows.append(new_row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter=_detect_delimiter(path)).writerows(rows)


def remove_glossary_term(path: Path, source: str) -> bool:
    """Delete one term by source text; ``True`` when a row was removed."""
    source = source.strip()
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if source not in data:
                return False
            del data[source]
        elif isinstance(data, list):
            kept = [
                item
                for item in data
                if not (
                    isinstance(item, dict)
                    and (item.get("source") or item.get("term") or item.get("src")) == source
                )
            ]
            if len(kept) == len(data):
                return False
            data = kept
        else:
            raise GlossaryFormatError("unsupported JSON glossary shape")
        _write_json(path, data)
        return True

    rows = _read_csv_rows(path)
    src_col, _tgt_col, start_row = _csv_columns(rows)
    kept_rows = list(rows[:start_row])
    removed = False
    for row in rows[start_row:]:
        if not removed and row and row[src_col].strip() == source:
            removed = True
            continue
        kept_rows.append(row)
    if not removed:
        return False
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter=_detect_delimiter(path)).writerows(kept_rows)
    return True


def list_tm_entries(
    tm_path: Path,
    *,
    limit: int = 200,
    offset: int = 0,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
) -> dict[str, Any]:
    """A page of the shared translation memory, newest ids last.

    ``scan`` reads the whole store (the CLI ``ubt tm scan`` uses it the same
    way); the console is a local tool, so slicing in memory is acceptable and
    keeps the store's single read API.
    """
    if not tm_path.exists():
        return {"total": 0, "entries": []}
    tm = TranslationMemory(tm_path)
    try:
        entries = tm.scan()
    finally:
        tm.close()
    if src_lang:
        entries = [e for e in entries if e.src_lang == src_lang]
    if tgt_lang:
        entries = [e for e in entries if e.tgt_lang == tgt_lang]
    total = len(entries)
    window = entries[offset : offset + limit]
    return {
        "total": total,
        "entries": [
            {
                "id": e.id,
                "src_lang": e.src_lang,
                "tgt_lang": e.tgt_lang,
                "source_text": e.source_text,
                "target_text": e.target_text,
                "provenance": e.provenance,
                "domain": e.domain,
                "use_count": e.use_count,
            }
            for e in window
        ],
    }


def evict_tm_entries(tm_path: Path, ids: list[int]) -> int:
    """Delete TM rows by id; returns the number removed."""
    if not tm_path.exists() or not ids:
        return 0
    tm = TranslationMemory(tm_path)
    try:
        return tm.evict_ids(ids)
    finally:
        tm.close()


# --------------------------------------------------------------------------- #
# Glossary conflict detection (PRD §4.4.2)
# --------------------------------------------------------------------------- #


def detect_glossary_conflicts(terms: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Sources configured with more than one distinct target rendering.

    The loader preserves duplicate rows, so a CSV/list glossary can carry the
    same source twice with different translations — the exact ambiguity the
    engine would resolve by file order. The console surfaces it instead of
    silently picking one.
    """
    by_source: dict[str, list[str]] = {}
    for term in terms:
        source = term.get("source", "")
        target = term.get("target", "")
        if not source:
            continue
        targets = by_source.setdefault(source, [])
        if target not in targets:
            targets.append(target)
    return [
        {"source": source, "targets": targets}
        for source, targets in by_source.items()
        if len(targets) > 1
    ]


# --------------------------------------------------------------------------- #
# Translation-memory import (TMX / JSON)
# --------------------------------------------------------------------------- #


class TMImportError(ValueError):
    """The imported payload could not be parsed into TM pairs."""


def _local_name(tag: str) -> str:
    """Strip the XML namespace from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1]


#: The XML namespace carrying ``xml:lang``.
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def parse_tmx(content: str) -> list[dict[str, str]]:
    """Parse TMX into ``{src_lang, tgt_lang, source_text, target_text}`` rows.

    For each ``<tu>`` the two ``<tuv>`` children are paired by language when the
    caller's ``xml:lang`` matches, otherwise by document order (the TMX spec's
    first/second convention). A ``<tu>`` without exactly two usable segments is
    skipped rather than guessed at.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise TMImportError(f"invalid TMX: {exc}") from exc

    rows: list[dict[str, str]] = []
    for node in root.iter():
        if _local_name(node.tag) != "tu":
            continue
        segments: list[tuple[str, str]] = []
        for child in node:
            if _local_name(child.tag) != "tuv":
                continue
            lang = child.get(_XML_LANG) or child.get("lang") or ""
            text = ""
            for seg in child:
                if _local_name(seg.tag) == "seg":
                    text = "".join(seg.itertext()).strip()
                    break
            if text:
                segments.append((lang, text))
        if len(segments) != 2:
            continue
        (src_lang, source_text), (tgt_lang, target_text) = segments
        rows.append(
            {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "source_text": source_text,
                "target_text": target_text,
            }
        )
    return rows


def parse_tm_json(content: str) -> list[dict[str, str]]:
    """Parse a JSON TM payload: a list of source/target pairs.

    Accepts ``source_text``/``target_text`` or ``source``/``target`` keys, with
    optional per-row ``src_lang``/``tgt_lang``.
    """
    try:
        data = json.loads(content)
    except ValueError as exc:
        raise TMImportError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise TMImportError("JSON TM payload must be a list of pairs")
    rows: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        source = item.get("source_text") or item.get("source")
        target = item.get("target_text") or item.get("target")
        if not source or not target:
            continue
        rows.append(
            {
                "src_lang": str(item.get("src_lang") or ""),
                "tgt_lang": str(item.get("tgt_lang") or ""),
                "source_text": str(source).strip(),
                "target_text": str(target).strip(),
            }
        )
    return rows


def parse_tm_payload(content: str, fmt: str) -> list[dict[str, str]]:
    """Dispatch to the TMX or JSON parser (``fmt`` is ``tmx`` or ``json``)."""
    if fmt == "tmx":
        return parse_tmx(content)
    if fmt == "json":
        return parse_tm_json(content)
    raise TMImportError(f"unsupported TM format: {fmt}")


def import_tm_entries(
    tm_path: Path,
    rows: list[dict[str, str]],
    *,
    default_src_lang: str,
    default_tgt_lang: str,
    provenance: str,
) -> int:
    """Write imported pairs into the shared TM; returns rows written.

    A row's own languages win; otherwise the caller's defaults fill in. Rows
    with an empty side are dropped. ``provenance`` is validated against the
    store's two known values so an import cannot invent a third.

    A ``machine`` import additionally *skips* any pair that already exists under
    ``human_pe``: the store's writeback keeps the human label but still lets the
    latest text win, so a bulk import would otherwise rewrite human-approved
    text while the row kept claiming human provenance.
    """
    if provenance not in (PROVENANCE_MACHINE, PROVENANCE_HUMAN_PE):
        raise TMImportError(f"unknown provenance: {provenance}")
    entries = [
        TMPendingEntry(
            src_lang=row["src_lang"] or default_src_lang,
            tgt_lang=row["tgt_lang"] or default_tgt_lang,
            source_text=row["source_text"],
            target_text=row["target_text"],
            provenance=provenance,
        )
        for row in rows
        if row.get("source_text") and row.get("target_text")
    ]
    if not entries:
        return 0
    tm = TranslationMemory(tm_path)
    try:
        if provenance == PROVENANCE_MACHINE:
            protected = {
                (entry.src_lang, entry.tgt_lang, normalize_for_tm(entry.source_text))
                for entry in tm.scan()
                if entry.provenance == PROVENANCE_HUMAN_PE
            }
            entries = [
                entry
                for entry in entries
                if (entry.src_lang, entry.tgt_lang, normalize_for_tm(entry.source_text))
                not in protected
            ]
        if not entries:
            return 0
        return tm.writeback(entries)
    finally:
        tm.close()
