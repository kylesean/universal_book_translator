"""Curated seed glossaries by domain profile (P11 term-consistency lever).

Miners + LLM backfill start from zero every book (chapter-1 bible held ~2
abbreviation pairs → glossary_hits 0/98 → terminology drift like
摆动幅度/倾斜/硅栅体). Seeds are human-approved (source, translation)
pairs merged FIRST in the bible stage, so first-translation-wins keeps them
over anything mined or backfilled later.

Precision rules (do NOT relax):
- Only terms whose translation is domain-unambiguous. Bare common words
  (source/drain/gate/channel/scaling) are EXCLUDED — the enforcer would
  over-normalize prose.
- Bare ``FET`` is EXCLUDED: first-use expansion (场效应晶体管) vs later
  keep-acronym is correct practice; forcing one way is wrong.
- Keep-Latin entries (MOSFET→MOSFET) exist to BLOCK transliteration
  (``Z. 刘``-class accidents), not to translate.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from ubt.core.job_options import profile_name_is_valid
from ubt.core.memory.bible import BibleEntry, clean_bible_entry

logger = logging.getLogger(__name__)

# (source, translation): semiconductor device / compact-model domain.
_SEMICONDUCTOR_SEEDS: tuple[tuple[str, str], ...] = (
    # Keep-Latin: transliteration blockers.
    ("MOSFET", "MOSFET"),
    ("FinFET", "FinFET"),
    ("GAA", "GAA"),
    ("BSIM", "BSIM"),
    ("SPICE", "SPICE"),
    ("CMOS", "CMOS"),
    ("SoC", "SoC"),
    ("IEDM", "IEDM"),
    ("VLSI", "VLSI"),
    ("TCAD", "TCAD"),
    # Canonical Chinese renderings (textbook-standard).
    ("subthreshold swing", "亚阈值摆幅"),
    ("short-channel effects", "短沟道效应"),
    ("thin-body", "薄体"),
    ("gate dielectric", "栅介质"),
    ("gate oxide", "栅氧化层"),
    ("compact model", "紧凑模型"),
    ("field-effect transistor", "场效应晶体管"),
    ("threshold voltage", "阈值电压"),
    ("leakage current", "漏电流"),
    ("random dopant fluctuation", "随机掺杂涨落"),
    ("gate-all-around", "全环绕栅极"),
    ("undoped body", "非掺杂体"),
    ("work function", "功函数"),
    ("drain-induced barrier lowering", "漏致势垒降低"),
    ("gradual channel approximation", "渐变沟道近似"),
    ("double-gate", "双栅"),
    ("channel potential", "沟道电势"),
    ("depletion charge", "耗尽电荷"),
    ("depletion region", "耗尽区"),
    ("depletion layer", "耗尽层"),
    ("drift-diffusion", "漂移-扩散"),
    ("inversion charge", "反型电荷"),
    ("inversion layer", "反型层"),
    ("quantum confinement", "量子限域"),
    ("flat-band voltage", "平带电压"),
    ("effective mobility", "有效迁移率"),
    ("ballistic transport", "弹道输运"),
    ("quasi-Fermi level", "准费米能级"),
    ("permittivity", "介电常数"),
    ("surface potential", "表面电势"),
    ("gate voltage", "栅电压"),
    ("drain voltage", "漏电压"),
    ("drain current", "漏极电流"),
)

_SEED_PROFILES: dict[str, tuple[tuple[str, str], ...]] = {
    "semiconductor": _SEMICONDUCTOR_SEEDS,
    "semiconductor_paper": _SEMICONDUCTOR_SEEDS,
    "semiconductor_textbook": _SEMICONDUCTOR_SEEDS,
}

# Curated-approved entries outrank mined ones in top-N selection.
_SEED_FREQUENCY = 10_000


def load_external_glossary(file_path: Path | str) -> list[BibleEntry]:
    """Load user or domain glossary from an external file (.csv, .tsv, or .json).

    Supported formats:
    - JSON:
      {"term": "translation", ...}
      or [{"source": "term", "translation": "translation"}, ...]
    - CSV / TSV:
      Rows of (source, translation) or with headers (source, translation) / (term, target)
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        logger.warning("External glossary file not found: %s", file_path)
        return []
    ext = p.suffix.lower()
    entries: list[BibleEntry] = []

    try:
        if ext == ".json":
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    e = clean_bible_entry(source=str(k), translation=str(v), kind="term")
                    if e:
                        e.frequency = _SEED_FREQUENCY
                        entries.append(e)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        src = item.get("source") or item.get("term") or item.get("src")
                        tgt = item.get("translation") or item.get("target") or item.get("tgt")
                        if src and tgt:
                            e = clean_bible_entry(
                                source=str(src), translation=str(tgt), kind="term"
                            )
                            if e:
                                e.frequency = _SEED_FREQUENCY
                                entries.append(e)
        elif ext in (".csv", ".tsv", ".txt"):
            delim = "\t" if ext == ".tsv" else ","
            with p.open(encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=delim)
                rows = list(reader)
                if not rows:
                    return []
                header = [h.strip().lower() for h in rows[0]]
                src_col = 0
                tgt_col = 1
                start_row = 0
                if any(h in ("source", "term", "src", "en") for h in header):
                    start_row = 1
                    for idx, h in enumerate(header):
                        if h in ("source", "term", "src", "en"):
                            src_col = idx
                        elif h in ("translation", "target", "tgt", "zh", "cn"):
                            tgt_col = idx
                for row in rows[start_row:]:
                    if len(row) > max(src_col, tgt_col):
                        src, tgt = row[src_col].strip(), row[tgt_col].strip()
                        if src and tgt:
                            e = clean_bible_entry(source=src, translation=tgt, kind="term")
                            if e:
                                e.frequency = _SEED_FREQUENCY
                                entries.append(e)
    except Exception as exc:
        logger.warning("Failed to parse external glossary %s: %s", file_path, exc)
        return []

    return entries


def seed_entries_for_profile(
    profile_name: str, source_lang: str = "en", target_lang: str = "zh"
) -> list[BibleEntry]:
    """Curated seed entries for ``profile_name`` (empty for unknown domains).

    Checks external resource directory first:
      ubt/resources/glossaries/{profile_name}/{source_lang}-{target_lang}.json
    Falls back to built-in seeds if present.
    """
    clean_prof = (profile_name or "").lower().split("_")[0]
    if profile_name and not profile_name_is_valid(profile_name):
        # A profile names one packaged directory; a separator or ``..`` would
        # escape ``ubt/resources/glossaries/`` (an absolute component resets the
        # joined path). Never read outside the packaged tree.
        logger.warning("Rejecting unsafe glossary profile name %r", profile_name)
        return []
    resource_path = (
        Path(__file__).parent.parent.parent
        / "resources"
        / "glossaries"
        / clean_prof
        / f"{source_lang.lower()}-{target_lang.lower()}.json"
    )
    if resource_path.exists():
        loaded = load_external_glossary(resource_path)
        if loaded:
            return loaded
        # Present but empty or unparseable is NOT the "this profile has no
        # external glossary" case the built-in seeds exist to cover. Staying
        # silent here ships a book with no terminology control and no signal —
        # and for profiles outside _SEED_PROFILES the fallback is empty too.
        logger.error(
            "External glossary %s exists but yielded no entries; falling back to "
            "built-in seeds. Terminology may be unenforced for profile %r.",
            resource_path,
            profile_name,
        )

    seeds = _SEED_PROFILES.get((profile_name or "").lower(), ())
    entries: list[BibleEntry] = []
    for source, translation in seeds:
        entry = clean_bible_entry(source=source, translation=translation, kind="term")
        if entry is not None:
            entry.frequency = _SEED_FREQUENCY
            entries.append(entry)
    return entries
