"""Translation backfill channel for mined bible entries.

Mined entries enter the bible with ``translation=""`` — abbreviation pairs
(``abbreviation_miner``) and person names (``character_miner``). This module
fills those translations with ONE bulk draft-model call per batch instead of
paying a per-block prompt cost, then hands the decided renderings back to the
bible so the term glossary enforces them for the rest of the book.

Failures degrade gracefully: an unparseable response leaves the entry's
translation empty and the prompt-side abbreviation table keeps working.
"""

import re
from collections.abc import Awaitable, Callable
from typing import Any

# "1. WM — Working Memory" style item line in the curated prompt
_MAX_TRANSLATION_CHARS = 80
_MAX_BATCH = 60

_RESPONSE_LINE = re.compile(
    r"^\s*(?:\d+[.)](?!\d)\s*|[-*]\s*)?"
    r"(?P<acro>[^\s:=]+)"
    r"\s*[:=]\s*(?P<tr>.+?)\s*$",
    re.MULTILINE,
)

_STRIP_QUOTES = "《》〈〉「」『』\"'“”"

#: Shortest all-caps token treated as a technical acronym that must survive into
#: its rendering. Two-letter initialisms (WM, CU) are conventionally expanded in
#: Chinese, so they are deliberately below this bar.
_ACRONYM_KEEP_MIN_LEN = 3


def build_backfill_prompt(entries: list[dict[str, Any]], target_lang: str) -> tuple[str, str]:
    """Build the single bulk terminology-curating prompt for a batch of entries."""
    system_prompt = (
        f"You are a deterministic bilingual terminology curator for translating a book into {target_lang}.\n"
        "Follow the output format EXACTLY. Never invent extra entries."
    )
    lines = [
        "### Task",
        (
            "For each numbered item below, provide the standard "
            f"{target_lang} rendering of its source term."
        ),
        "",
        "Rules:",
        (
            "1. Items marked (person name): transliterate/translate the name core "
            "WITHOUT the courtesy title (Mr./Mrs./Miss/Sir/Lady etc.)."
        ),
        "2. Acronym items: keep the acronym itself unchanged (Latin letters) inside the translation.",
        "3. Answer with EXACTLY one line per item, in the same order, formatted as:",
        "   KEY = <rendering>",
        "4. No extra commentary, no markdown fences.",
        "",
        "### Items",
    ]
    for idx, e in enumerate(entries, start=1):
        key = str((e.get("aliases") or [""])[0])
        source = str(e.get("source", "")).replace("\n", " ")
        marker = " (person name)" if e.get("kind") == "person" else ""
        lines.append(f"{idx}. {key} — {source}{marker}")
    return system_prompt, "\n".join(lines)


def _clean_translation(raw: str, acronym: str) -> str | None:
    tr = raw.strip().strip(_STRIP_QUOTES).strip()
    if not tr or len(tr) > _MAX_TRANSLATION_CHARS:
        return None
    if tr.lower() == acronym.lower():
        return None  # model echoed the acronym instead of translating
    if "\n" in tr or "=" in tr:
        return None
    # Rule 2 of the prompt: "Acronym items: keep the acronym itself unchanged
    # (Latin letters) inside the translation." The model breaks it — the ATC'25
    # run rendered IOMMU as 输入/输出内存管理单元, dropping the acronym — and every
    # downstream reader then saw an expanded phrase where the paper writes the
    # token: 19 occurrences in prose, and a table column that no longer fit
    # ("Aster. no IOMMU(s)" became "Aster.（禁用输入/输出内存管理单元，秒）").
    #
    # The rule is acceptance criteria, not advice, so it is enforced here.
    #
    # Scope: only multi-letter all-caps tokens. A two-letter initialism like
    # ``WM`` (Working Memory → 工作记忆) is *conventionally* expanded in Chinese
    # and the model's rendering is right, so rejecting it would be the mirror
    # error. The three-plus-letter all-caps form is the technical-acronym shape
    # (IOMMU, TCP, MMIO, DMA) that readers expect to see verbatim.
    if len(acronym) >= _ACRONYM_KEEP_MIN_LEN and acronym.isupper() and acronym not in tr:
        return None
    return tr


def _apply_response(response: str, batch: list[dict[str, Any]]) -> int:
    """Parse `ACRONYM = translation` lines and fill matching empty entries."""
    by_acronym: dict[str, str] = {}
    for m in _RESPONSE_LINE.finditer(response or ""):
        acronym = m.group("acro")
        translation = _clean_translation(m.group("tr"), acronym)
        if translation and acronym.lower() not in by_acronym:
            by_acronym[acronym.lower()] = translation

    filled = 0
    for e in batch:
        if str(e.get("translation", "")).strip():
            continue  # never overwrite an existing decision
        aliases = [str(a).lower() for a in e.get("aliases", []) if a]
        for alias in aliases:
            if alias in by_acronym:
                e["translation"] = by_acronym[alias]
                filled += 1
                break
    return filled


async def backfill_abbreviation_translations(
    entries: list[dict[str, Any]],
    complete: Callable[[str, str], Awaitable[str]],
    target_lang: str = "zh",
    cancel_check: Callable[[], None] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Fill empty translations of mined abbreviation pairs in bulk.

    ``complete`` is an async ``(system_prompt, user_prompt) -> str`` callable
    (typically ``ModelRouter.complete_raw``). Batches of ``_MAX_BATCH`` pairs
    share one call. Returns ``(entries, filled_count)``; pairs whose rendering
    could not be parsed keep ``translation=""``.
    """
    pending = [e for e in entries if not str(e.get("translation", "")).strip()]
    # One prompt item per alias key: surface variants of the same name core
    # (Mr. Bingley / Miss Bingley) share the decided rendering. Entries that
    # share a key are applied together so a deduped variant is still filled.
    by_key: dict[str, list[dict[str, Any]]] = {}
    for e in pending:
        key = str((e.get("aliases") or [""])[0]).lower()
        by_key.setdefault(key, []).append(e)
    filled_total = 0
    groups = list(by_key.values())
    for start in range(0, len(groups), _MAX_BATCH):
        if cancel_check is not None:
            cancel_check()
        batch_groups = groups[start : start + _MAX_BATCH]
        representatives = [group[0] for group in batch_groups]
        system_prompt, user_prompt = build_backfill_prompt(representatives, target_lang)
        response = await complete(system_prompt, user_prompt)
        # Apply each response only to this batch's entries; the former call
        # passed every pending entry, so one batch's acronyms could fill
        # unrelated entries from another batch.
        filled_total += _apply_response(response, [e for group in batch_groups for e in group])
    return entries, filled_total
