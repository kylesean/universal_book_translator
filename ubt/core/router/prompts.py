"""Draft and repair prompt assembly.

Pure functions of their arguments -- no router state, no provider, no I/O -- so
they do not have to live in :mod:`ubt.core.router.router` for that file to mean
"routing". Keeping them here also puts the marker literals the draft builders
write and :func:`draft_source_from_prompt` reads back into one file: they are one
contract, and a builder whose heading the parser does not know about leaks its
instruction tail into what ``--dry-run`` shows the user.
"""

from __future__ import annotations

import html
import re

from ubt.core.ir.bifurcation import SEMANTIC_BREAK_TOKEN
from ubt.core.language_profile import PROFILES

#: Wrapper tags the extractor treats as the answer envelope. A source span that
#: merely *mentions* one ("use <translation> tags") must not be mistaken for the
#: envelope, so the literal ``<`` is escaped in the injected source. The extractor
#: is hardened independently (it greedily takes the last close tag).
_RESERVED_WRAPPER_RE = re.compile(
    r"<\s*/?\s*(?:final_translation|translation|issues|think|thought|thinking|reasoning)\b",
    re.IGNORECASE,
)


def _neutralize_reserved_tags(text: str) -> str:
    """Escape the ``<`` of reserved wrapper-tag mentions in a source span."""
    return _RESERVED_WRAPPER_RE.sub(lambda m: m.group(0).replace("<", "&lt;"), text)


# The literal the three draft builders put before the source span, and the tails
# they append after it. ``build_minimal_draft_prompt`` has no heading: it ends
# with the source after this instruction line.
_DRAFT_SOURCE_MARKER = "### Source Paragraph to Translate\n"
_DRAFT_SOURCE_TAILS = (
    "\n\nTranslate only the text under this heading",
    "\n\n### Translation:",
    "\n\nProvide the direct",
)
_DRAFT_MINIMAL_ANCHOR = "Output ONLY the translation without any title, prefix, or commentary:\n\n"


def draft_source_from_prompt(prompt: str) -> str:
    """Recover the source span a draft prompt carried.

    ``--dry-run`` previews a translation without a provider, so it has to read
    the block back out of the prompt it was handed. This is the one place that
    knows the markers: parsing them per surface let a builder's instruction tail
    leak into the "translation" the user saw.
    """
    if "<blocks>" in prompt:
        remainder = prompt.split("<blocks>", 1)[1]
        if "</blocks>" in remainder:
            return remainder.split("</blocks>", 1)[0].strip()
        return remainder.strip()
    if _DRAFT_SOURCE_MARKER in prompt:
        remainder = prompt.split(_DRAFT_SOURCE_MARKER, 1)[1]
        for tail in _DRAFT_SOURCE_TAILS:
            if tail in remainder:
                remainder = remainder.split(tail, 1)[0]
                break
        return remainder.strip()
    if _DRAFT_MINIMAL_ANCHOR in prompt:
        return prompt.rsplit(_DRAFT_MINIMAL_ANCHOR, 1)[1].strip()
    return prompt.strip()


def build_minimal_draft_prompt(
    source_text: str,
    glossary_table: str = "",
    target_lang: str = "zh",
    source_lang: str = "en",
    global_glossary: str = "",
    few_shot_reference: str = "",
    genre_profile: str = "general",
    domain: str | None = None,
) -> tuple[str, str]:
    """Direct, minimalist draft prompt without metaprompts.

    Glossary sections merge global (book-static) and per-chunk entries.
    ``domain`` is the operator's explicit subject descriptor (``--domain``); it
    outranks the profile-derived hint when supplied.
    """
    src_profile = PROFILES.get(source_lang.strip().lower()) if source_lang else None
    tgt_profile = PROFILES.get(target_lang.strip().lower()) if target_lang else None
    src_name = (
        src_profile.name if src_profile else (source_lang.upper() if source_lang else "SOURCE")
    )
    tgt_name = (
        tgt_profile.name if tgt_profile else (target_lang.upper() if target_lang else "TARGET")
    )

    glossary_sections = [
        section.strip() for section in (global_glossary, glossary_table) if section.strip()
    ]
    parts = []
    if glossary_sections:
        parts.append("Glossary:\n" + "\n\n".join(glossary_sections))
    if few_shot_reference.strip():
        parts.append(few_shot_reference.strip())
    domain_hint = ""
    if domain:
        domain_hint = f" Use standard {domain} terminology."
    elif genre_profile and genre_profile.lower() not in ("general", "unknown", "auto"):
        domain_hint = f" Use standard {genre_profile} domain terminology."
    parts.append(
        f"Translate the following {src_name} text into fluent, natural {tgt_name}.{domain_hint} Output ONLY the translation without any title, prefix, or commentary:\n\n{_neutralize_reserved_tags(source_text).strip()}"
    )
    return "", "\n\n".join(parts)


def build_hybrid_draft_prompt(
    source_text: str,
    glossary_table: str = "",
    neighbor_context: str = "",
    target_lang: str = "zh",
    source_lang: str = "en",
    genre_profile: str = "general",
    rolling_summary: str = "",
    global_glossary: str = "",
    few_shot_reference: str = "",
    epoch_summary: str = "",
    domain: str | None = None,
) -> tuple[str, str]:
    """Hybrid draft prompt for instruction-tuned models requiring concise instructions.

    Prefix-cache topology: sections are ordered by mutability
    so the cumulative prompt prefix stays stable across blocks —
    book-static global glossary → L3 epoch summaries (change only every
    N steps) → chapter-level rolling summary → per-block chunk
    glossary → neighbors → few-shot reference → source. Global and chunk
    glossaries are merged (the former if/elif silently dropped chunk
    terms whenever a global glossary existed).
    """
    src_profile = PROFILES.get(source_lang.strip().lower()) if source_lang else None
    tgt_profile = PROFILES.get(target_lang.strip().lower()) if target_lang else None
    src_name = (
        src_profile.name if src_profile else (source_lang.upper() if source_lang else "SOURCE")
    )
    tgt_name = (
        tgt_profile.name if tgt_profile else (target_lang.upper() if target_lang else "TARGET")
    )

    if genre_profile == "fiction":
        style_desc = (
            f"You are an acclaimed literary translator translating from {src_name} into {tgt_name}.\n"
            "Emphasize natural prose rhythm, character voice, subtext, and atmospheric immersion."
        )
    elif genre_profile in ("academic", "paper"):
        style_desc = (
            f"You are a scholarly scientific translator translating from {src_name} into {tgt_name}.\n"
            "Maintain strict scientific rigor, objective academic register, and precise domain terms."
        )
    elif genre_profile == "textbook":
        style_desc = (
            f"You are an educational textbook translator translating from {src_name} into {tgt_name}.\n"
            "Prioritize pedagogical clarity, conceptual fidelity, and retain all visual callouts."
        )
    else:
        style_desc = (
            f"You are a professional book translator translating from {src_name} into {tgt_name}.\n"
            "Translate accurately while maintaining natural prose and literary flow."
        )
    if domain:
        style_desc += f"\nUse standard {domain} terminology."

    if "zh" in tgt_name.lower() or (tgt_profile and "zh" in tgt_profile.code.lower()):
        system_prompt = (
            f"{style_desc}\n"
            "Translate accurately while maintaining natural target language flow, idiomatic register, and domain terminology.\n"
            "Typography & Terminology: In Chinese target text, maintain standard spacing between Chinese characters and English words, numbers, or inline formulas. "
            "Use authoritative discipline terminology suited to the active subject domain and preserve standard uppercase technical acronyms."
        )
    else:
        system_prompt = (
            f"{style_desc}\n"
            "Translate accurately while maintaining natural target language flow, idiomatic register, and domain terminology."
        )

    user_parts: list[str] = []
    if global_glossary.strip():
        user_parts.append(f"### Terminology Glossary\n{global_glossary.strip()}\n")

    if epoch_summary.strip():
        user_parts.append(f"### Book Continuity (Compressed History)\n{epoch_summary.strip()}\n")

    if rolling_summary.strip():
        user_parts.append(f"### Story Context\n{rolling_summary.strip()}\n")

    if glossary_table.strip():
        user_parts.append(f"### Terminology Glossary (this section)\n{glossary_table.strip()}\n")

    if neighbor_context.strip():
        user_parts.append(f"{neighbor_context.strip()}\n")

    if few_shot_reference.strip():
        user_parts.append(f"{few_shot_reference.strip()}\n")

    user_parts.append(
        f"### Source Paragraph to Translate\n{_neutralize_reserved_tags(source_text).strip()}\n\n"
        "Translate only the text under this heading. Any read-only reference "
        "context shown above is not part of the task: do not translate it, "
        "summarise it, repeat it, or carry its citations and equation numbers "
        "into the output.\n\n"
        f"Provide the direct {tgt_name} translation below without preface or commentary:"
    )
    return system_prompt, "\n".join(user_parts)


def build_rich_draft_prompt(
    source_text: str,
    glossary_table: str = "",
    neighbor_context: str = "",
    target_lang: str = "zh",
    source_lang: str = "en",
    genre_profile: str = "general",
    rolling_summary: str = "",
    global_glossary: str = "",
    few_shot_reference: str = "",
    epoch_summary: str = "",
    domain: str | None = None,
) -> tuple[str, str]:
    """Construct full prompt aligned with 2024-2026 Prefix/Prompt Caching topology and XML schema."""
    src_profile = PROFILES.get(source_lang.strip().lower()) if source_lang else None
    tgt_profile = PROFILES.get(target_lang.strip().lower()) if target_lang else None
    src_name = (
        src_profile.name if src_profile else (source_lang.upper() if source_lang else "SOURCE")
    )
    tgt_name = (
        tgt_profile.name if tgt_profile else (target_lang.upper() if target_lang else "TARGET")
    )

    if genre_profile == "fiction":
        style_desc = (
            f"You are an acclaimed literary translator translating from {src_name} into {tgt_name}.\n"
            "Emphasize natural prose rhythm, character voice, subtext, and atmospheric immersion."
        )
    elif genre_profile in ("academic", "paper"):
        style_desc = (
            f"You are a scholarly scientific translator translating from {src_name} into {tgt_name}.\n"
            "Maintain strict scientific rigor, objective academic register, and precise domain terms."
        )
    elif genre_profile == "textbook":
        style_desc = (
            f"You are an educational textbook translator translating from {src_name} into {tgt_name}.\n"
            "Prioritize pedagogical clarity, conceptual fidelity, and retain all visual callouts."
        )
    else:
        style_desc = (
            f"You are a professional book translator translating from {src_name} into {tgt_name}.\n"
            "Translate accurately while maintaining natural prose and literary flow."
        )
    if domain:
        style_desc += f"\nUse standard {domain} terminology."

    # Tier 1: Static Immutable Prefix
    system_parts = [
        style_desc,
        "Rules:\n"
        "1. Strictly adhere to the terminology mappings in the Translation Bible if provided.\n"
        "2. Preserve all XML/HTML markup tags and placeholder tokens (⟦CODE_MASK_...⟧, ⟦MATH_MASK_...⟧, ⟦CITE_MASK_...⟧) verbatim in their original relative order without omitting, modifying, or swapping them.\n"
        f"3. [Natural Register & Translationese Elimination]:\n"
        f"   - Eliminate rigid, literal translationese (avoid mechanical passive constructions such as '被...', '由...所...'; avoid redundant boilerplate such as '当...的时候', '作为一个...').\n"
        f"   - In CJK targets (Chinese/Japanese/Korean), naturalize passive voice into active, subjectless, or topic-prominent forms whenever natural (e.g. 'Measurements were taken' -> '进行了测量', NOT '测量被执行').\n"
        f"   - Syntactic re-chunking: deconstruct long, multi-clause hypotactic Western sentences into clean, logically balanced target prose (意合结构), maintaining natural narrative rhythm without omitting meaning.\n"
        f"4. [Restorative Denoising & Column Disentangling]: If the source text originates from scanned/OCR pages and contains OCR artifacts:\n"
        f"   - Intelligently reconstruct and translate only the genuine authorial/conceptual content into clean, fluent {tgt_name}.\n"
        f"   - Autonomously correct minor OCR typos, broken ligatures, and hyphenation in context.\n"
        f"   - If sidebar/callout text is horizontally interleaved across columns with the main narrative, disentangle them: translate the main narrative coherently, separating sidebars/notes clearly.\n"
        f"   - Format distinct conceptual thoughts into natural, readable paragraphs separated by double newlines (\\n\\n).\n"
        f"   - [Semantic Break & Layout Collision Detection]: If the input paragraph contains an accidental layout collision (e.g. text from two different columns glued together, mid-sentence column jumps, or stray page numbers/headers inserted into prose), insert the delimiter {SEMANTIC_BREAK_TOKEN} at the boundary between the disjoint passages in your translation so the system can bifurcate them into distinct blocks.\n"
        f"   - DO NOT transcribe or echo nonsensical OCR symbol strings, stray page numbers, or broken publisher boilerplate fragments into the target output.\n"
        f"5. [Scientific Math & Formula Fidelity]:\n"
        f"   - Reconstruct all inline physical variables, mathematical symbols, Greek letters, and sub/superscript notations (such as Vtm, kBT/q, Vch(0)=Vs, psi(x,y), ni, eps_si) into standard LaTeX inline math syntax enclosed by single dollar signs ($...$, e.g. $V_{{tm}}$, $k_B T / q$, $V_{{ch}}(0) = V_s$, $\\psi(x, y)$). Never leave them as flattened plain ASCII strings.\n"
        f"   - Only wrap notation that is ALREADY mathematical in the source. NEVER invent LaTeX control sequences (\\mathrm, \\text, \\mathbf, ...) or $...$ delimiters for plain alphanumeric labels that appear as plain text in the source (dimension tags like 2D/3D, figure/table callouts like Fig. 1.1, units like 10nm) — reproduce those verbatim.\n"
        f"   - Never omit or truncate explanatory clauses that follow formulas (such as clauses starting with 'where...', 'in which...', '式中...', '其中...'). Translate them fully and faithfully.\n"
        f"6. [Publishing Typography & CJK Spacing]:\n"
        f"   - In Chinese/CJK target text, always maintain standard typographic spacing: leave a single half-width space between CJK characters and Latin words, numbers, and inline formulas (e.g. '在第 3 章'、'公式 $x$ 中').\n"
        f"   - Full-width punctuation marks (，。！？；：“”‘’（）《》) must remain snug against adjacent text without extra spaces.\n"
        f"7. [Domain Terminology & Acronyms]:\n"
        f"   - Consistently use standard national and discipline terminology appropriate for the document's subject field.\n"
        f"   - Standard domain acronyms must be preserved in uppercase Latin letters when introduced, accompanied by their established target language translation in parentheses where appropriate. NEVER mechanically duplicate or invent awkward pseudo-translations.\n"
        "8. Output your final translation strictly enclosed inside <translation>...</translation> tags. Any thinking, context analysis, or commentary must remain outside these tags. Do not include markdown code fences.",
    ]

    if global_glossary.strip():
        system_parts.append(
            f"### Translation Bible (Strict Global Terminology Mappings)\n{global_glossary.strip()}"
        )

    system_prompt = "\n\n".join(system_parts)

    # Tier 2: Semi-static L3 epoch (changes only every N steps)
    # then Macro Snapshot Prefix & Chunk Glossary
    user_parts: list[str] = []
    if epoch_summary.strip():
        user_parts.append(f"### Book Continuity (Compressed History)\n{epoch_summary.strip()}\n")
    if rolling_summary.strip():
        user_parts.append(
            f"### Document Continuation Context (Macro Snapshot)\n{rolling_summary.strip()}\n"
        )
    if glossary_table.strip():
        user_parts.append(f"### Translation Bible (Strict Mappings)\n{glossary_table.strip()}\n")

    # Tier 3: Dynamic Tail (L1 Neighbor Excerpts + TM few-shot + Target IRBlock)
    if neighbor_context.strip():
        user_parts.append(f"{neighbor_context.strip()}\n")

    if few_shot_reference.strip():
        user_parts.append(f"{few_shot_reference.strip()}\n")

    user_parts.append(
        f"### Source Paragraph to Translate\n{_neutralize_reserved_tags(source_text).strip()}\n\n"
        "Translate only the text under this heading. Any read-only reference "
        "context shown above is not part of the task: do not translate it, "
        "summarise it, repeat it, or carry its citations and equation numbers "
        "into the output.\n\n"
        "### Translation:"
    )
    return system_prompt, "\n".join(user_parts)


def build_macro_chunk_draft_prompt(
    blocks: list[tuple[str, str]],
    glossary_table: str = "",
    neighbor_context: str = "",
    target_lang: str = "zh",
    source_lang: str = "en",
    genre_profile: str = "general",
    rolling_summary: str = "",
    global_glossary: str = "",
    few_shot_reference: str = "",
    epoch_summary: str = "",
    domain: str | None = None,
) -> tuple[str, str]:
    """Construct structured multi-block prompt for high-throughput macro-chunk drafting."""
    src_profile = PROFILES.get(source_lang.strip().lower()) if source_lang else None
    tgt_profile = PROFILES.get(target_lang.strip().lower()) if target_lang else None
    src_name = (
        src_profile.name if src_profile else (source_lang.upper() if source_lang else "SOURCE")
    )
    tgt_name = (
        tgt_profile.name if tgt_profile else (target_lang.upper() if target_lang else "TARGET")
    )

    if genre_profile == "fiction":
        style_desc = (
            f"You are an acclaimed literary translator translating from {src_name} into {tgt_name}.\n"
            "Emphasize natural prose rhythm, character voice, subtext, and atmospheric immersion."
        )
    elif genre_profile in ("academic", "paper"):
        style_desc = (
            f"You are a scholarly scientific translator translating from {src_name} into {tgt_name}.\n"
            "Maintain strict scientific rigor, objective academic register, and precise domain terms."
        )
    elif genre_profile == "textbook":
        style_desc = (
            f"You are an educational textbook translator translating from {src_name} into {tgt_name}.\n"
            "Prioritize pedagogical clarity, conceptual fidelity, and retain all visual callouts."
        )
    else:
        style_desc = (
            f"You are a professional book translator translating from {src_name} into {tgt_name}.\n"
            "Translate accurately while maintaining natural prose and literary flow."
        )
    if domain:
        style_desc += f"\nUse standard {domain} terminology."

    system_parts = [
        style_desc,
        "Rules:\n"
        "1. Strictly adhere to the terminology mappings in the Translation Bible if provided.\n"
        "2. Preserve all XML/HTML markup tags and placeholder tokens (⟦CODE_MASK_...⟧, ⟦MATH_MASK_...⟧, ⟦CITE_MASK_...⟧) verbatim in their original relative order without omitting, modifying, or swapping them.\n"
        f"3. [Natural Register & Translationese Elimination]:\n"
        f"   - Eliminate rigid, literal translationese (avoid mechanical passive constructions such as '被...', '由...所...'; avoid redundant boilerplate such as '当...的时候', '作为一个...').\n"
        f"   - In CJK targets (Chinese/Japanese/Korean), naturalize passive voice into active, subjectless, or topic-prominent forms whenever natural (e.g. 'Measurements were taken' -> '进行了测量', NOT '测量被执行').\n"
        f"   - Syntactic re-chunking: deconstruct long, multi-clause hypotactic Western sentences into clean, logically balanced target prose (意合结构), maintaining natural narrative rhythm without omitting meaning.\n"
        f"4. [Restorative Denoising & Column Disentangling]: If the source text originates from scanned/OCR pages and contains OCR artifacts:\n"
        f"   - Intelligently reconstruct and translate only the genuine authorial/conceptual content into clean, fluent {tgt_name}.\n"
        f"   - Autonomously correct minor OCR typos, broken ligatures, and hyphenation in context.\n"
        f"   - If sidebar/callout text is horizontally interleaved across columns with the main narrative, disentangle them: translate the main narrative coherently, separating sidebars/notes clearly.\n"
        f"   - Format distinct conceptual thoughts into natural, readable paragraphs separated by double newlines (\\n\\n).\n"
        f"   - DO NOT transcribe or echo nonsensical OCR symbol strings, stray page numbers, or broken publisher boilerplate fragments into the target output.\n"
        f"5. [Scientific Math & Formula Fidelity]:\n"
        f"   - Reconstruct all inline physical variables, mathematical symbols, Greek letters, and sub/superscript notations (such as Vtm, kBT/q, Vch(0)=Vs, psi(x,y), ni, eps_si) into standard LaTeX inline math syntax enclosed by single dollar signs ($...$, e.g. $V_{{tm}}$, $k_B T / q$, $V_{{ch}}(0) = V_s$, $\\psi(x, y)$). Never leave them as flattened plain ASCII strings.\n"
        f"   - Only wrap notation that is ALREADY mathematical in the source. NEVER invent LaTeX control sequences (\\mathrm, \\text, \\mathbf, ...) or $...$ delimiters for plain alphanumeric labels that appear as plain text in the source (dimension tags like 2D/3D, figure/table callouts like Fig. 1.1, units like 10nm) — reproduce those verbatim.\n"
        f"   - Never omit or truncate explanatory clauses that follow formulas (such as clauses starting with 'where...', 'in which...', '式中...', '其中...'). Translate them fully and faithfully.\n"
        f"6. [Publishing Typography & CJK Spacing]:\n"
        f"   - In Chinese/CJK target text, always maintain standard typographic spacing: leave a single half-width space between CJK characters and Latin words, numbers, and inline formulas (e.g. '在第 3 章'、'公式 $x$ 中').\n"
        f"   - Full-width punctuation marks (，。！？；：“”‘’（）《》) must remain snug against adjacent text without extra spaces.\n"
        f"7. [Domain Terminology & Acronyms]:\n"
        f"   - Consistently use standard national and discipline terminology appropriate for the document's subject field.\n"
        f"   - Standard domain acronyms must be preserved in uppercase Latin letters when introduced, accompanied by their established target language translation in parentheses where appropriate. NEVER mechanically duplicate or invent awkward pseudo-translations.\n"
        '8. [Structured Macro-Blocks Output]: You must output your translation inside <blocks>...</blocks> tags. Each block must be enclosed in a <block id="...">...</block> tag with the EXACT matching id from the source. Translate ONLY the text of each block. Do not include markdown code fences or conversational commentary outside the tags.',
    ]

    if global_glossary.strip():
        system_parts.append(
            f"### Translation Bible (Strict Global Terminology Mappings)\n{global_glossary.strip()}"
        )

    system_prompt = "\n\n".join(system_parts)

    user_parts: list[str] = []
    if epoch_summary.strip():
        user_parts.append(f"### Book Continuity (Compressed History)\n{epoch_summary.strip()}\n")
    if rolling_summary.strip():
        user_parts.append(
            f"### Document Continuation Context (Macro Snapshot)\n{rolling_summary.strip()}\n"
        )
    if glossary_table.strip():
        user_parts.append(f"### Translation Bible (Strict Mappings)\n{glossary_table.strip()}\n")

    if neighbor_context.strip():
        user_parts.append(f"{neighbor_context.strip()}\n")

    if few_shot_reference.strip():
        user_parts.append(f"{few_shot_reference.strip()}\n")

    # Escape source text and ids: a block containing </block> or </blocks>
    # (books about XML/HTML, code samples) would otherwise close the envelope
    # early and silently hide every later block from the model.
    blocks_xml = "\n".join(
        f'<block id="{html.escape(bid, quote=True)}">{html.escape(btext.strip(), quote=True)}</block>'
        for bid, btext in blocks
    )

    user_parts.append(
        "### Source Blocks to Translate\n"
        "<blocks>\n"
        f"{blocks_xml}\n"
        "</blocks>\n\n"
        "Translate each block faithfully. Maintain all block ids exactly. "
        'Output strictly inside <blocks>...</blocks> with corresponding <block id="..."> tags:\n'
        "<blocks>\n"
    )
    return system_prompt, "\n".join(user_parts)


def _detect_table_grid(text: str) -> tuple[int, int] | None:
    """Return (num_rows, num_cols) if text contains a markdown table, else None."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if ln.count("|") >= 2]
    if len(table_lines) >= 2:
        col_counts = [len(ln.strip("|").split("|")) for ln in table_lines]
        if col_counts:
            from collections import Counter

            common_cols = Counter(col_counts).most_common(1)[0][0]
            return len(table_lines), common_cols
    return None


def build_minimal_repair_prompt(
    source_text: str,
    draft_text: str = "",
    error_flags: list[str] | None = None,
    glossary_table: str = "",
    target_lang: str = "zh",
    source_lang: str = "en",
) -> tuple[str, str]:
    """Direct, minimalist repair prompt without metaprompts.

    Minimal means small, not blind: the draft and its defect flags travel with
    the request, or a "repair" degenerates into an unconditioned re-translation
    that bills a repair round for a fresh draft with no error context.
    """
    src_profile = PROFILES.get(source_lang.strip().lower()) if source_lang else None
    tgt_profile = PROFILES.get(target_lang.strip().lower()) if target_lang else None
    src_name = (
        src_profile.name if src_profile else (source_lang.upper() if source_lang else "SOURCE")
    )
    tgt_name = (
        tgt_profile.name if tgt_profile else (target_lang.upper() if target_lang else "TARGET")
    )

    parts = []
    if glossary_table.strip():
        parts.append(f"Glossary:\n{glossary_table.strip()}")
    grid_info = _detect_table_grid(source_text)
    if grid_info:
        rows, cols = grid_info
        parts.append(
            f"CRITICAL TABLE PRESERVATION GUARDRAIL:\n"
            f"The text is a markdown table with {rows} rows and {cols} columns. "
            f"You MUST strictly preserve the exact {rows}x{cols} table grid structure. "
            f"Do NOT delete, drop, or merge any rows or columns during repair."
        )
    flags = ", ".join(error_flags or []) or "none"
    draft_body = _neutralize_reserved_tags(draft_text).strip() if draft_text.strip() else "(empty)"
    parts.append(
        f"Fix the draft translation below. Source {src_name} text:\n\n"
        f"{_neutralize_reserved_tags(source_text).strip()}\n\n"
        f"Detected problems in the draft: {flags}.\n"
        f"Current draft ({tgt_name}):\n{draft_body}\n\n"
        f"Output ONLY the corrected {tgt_name} translation without any title, prefix, or commentary."
    )
    return "", "\n\n".join(parts)


def build_hybrid_repair_prompt(
    source_text: str,
    draft_text: str,
    error_flags: list[str],
    glossary_table: str = "",
    target_lang: str = "zh",
    source_lang: str = "en",
    annotated_draft: str = "",
    has_error_spans: bool = False,
) -> tuple[str, str]:
    """Hybrid repair prompt for instruction-tuned models."""
    src_profile = PROFILES.get(source_lang.strip().lower()) if source_lang else None
    tgt_profile = PROFILES.get(target_lang.strip().lower()) if target_lang else None
    src_name = (
        src_profile.name if src_profile else (source_lang.upper() if source_lang else "SOURCE")
    )
    tgt_name = (
        tgt_profile.name if tgt_profile else (target_lang.upper() if target_lang else "TARGET")
    )

    grid_info = _detect_table_grid(source_text)
    table_guardrail = ""
    if grid_info:
        rows, cols = grid_info
        table_guardrail = (
            f"\n\n[CRITICAL TABLE STRUCTURE GUARDRAIL]:\n"
            f"The text contains a markdown table with {rows} rows and {cols} columns.\n"
            f"You MUST STRICTLY PRESERVE the exact {rows} rows and {cols} columns ({rows}x{cols} grid).\n"
            f"NEVER delete, drop, or merge any rows or columns during repair, even if a critique reports repetition or loops.\n"
            f"Output the complete table with identical row and column count, translating only translatable cell text."
        )

    system_prompt = (
        f"You are an expert bilingual editor refining a translation from {src_name} into {tgt_name}.\n"
        "Fix all identified issues and output the corrected translation."
        f"{table_guardrail}"
    )

    issues_formatted = (
        "\n".join(f"- {flag}" for flag in error_flags)
        if error_flags
        else "- General fluency / quality score below threshold"
    )

    user_parts: list[str] = []
    if glossary_table.strip():
        user_parts.append(f"### Terminology Glossary\n{glossary_table.strip()}\n")

    if grid_info:
        rows, cols = grid_info
        user_parts.append(
            f"### Table Preservation Requirement\n"
            f"Strictly maintain the exact {rows} rows and {cols} columns of the table grid. "
            f"Do not delete or merge any rows or columns.\n"
        )

    if has_error_spans and annotated_draft.strip():
        user_parts.append(
            f"### Original Source\n{_neutralize_reserved_tags(source_text).strip()}\n\n"
            f"### Draft with Marked Errors\n{annotated_draft.strip()}\n\n"
            f"### Issues to Fix\n{issues_formatted}\n\n"
            "Please provide the corrected full translation directly:"
        )
    else:
        user_parts.append(
            f"### Original Source\n{_neutralize_reserved_tags(source_text).strip()}\n\n"
            f"### Previous Draft\n{draft_text.strip()}\n\n"
            f"### Issues to Fix\n{issues_formatted}\n\n"
            "Please provide the polished and corrected translation directly:"
        )

    return system_prompt, "\n".join(user_parts)


def build_rich_repair_prompt(
    source_text: str,
    draft_text: str,
    error_flags: list[str],
    glossary_table: str = "",
    target_lang: str = "zh",
    source_lang: str = "en",
    annotated_draft: str = "",
    has_error_spans: bool = False,
) -> tuple[str, str]:
    """Targeted repair prompt following 2026 Task-Structured Reasoning and MQM Infilling standards."""
    src_profile = PROFILES.get(source_lang.strip().lower()) if source_lang else None
    tgt_profile = PROFILES.get(target_lang.strip().lower()) if target_lang else None
    src_name = (
        src_profile.name if src_profile else (source_lang.upper() if source_lang else "SOURCE")
    )
    tgt_name = (
        tgt_profile.name if tgt_profile else (target_lang.upper() if target_lang else "TARGET")
    )

    issues_formatted = (
        "\n".join(f"- {flag}" for flag in error_flags)
        if error_flags
        else "- General fluency / quality score below threshold"
    )

    grid_info = _detect_table_grid(source_text)

    user_parts: list[str] = []
    if glossary_table.strip():
        user_parts.append(f"### Translation Bible\n{glossary_table.strip()}\n")

    if grid_info:
        rows, cols = grid_info
        user_parts.append(
            f"### Table Structure Guardrail\n"
            f"Strictly preserve the exact {rows} rows and {cols} columns of the source table grid. "
            f"Do NOT delete, drop, or merge any rows or columns.\n"
        )

    if has_error_spans and annotated_draft.strip():
        table_infilling_note = ""
        if grid_info:
            rows, cols = grid_info
            table_infilling_note = (
                f"\n4. Table Structure: Preserve all {rows} rows and {cols} columns of the table grid. "
                "Never delete or drop table rows."
            )
        system_prompt = (
            f"You are an automated precision translation repair agent refining a translation from {src_name} into {tgt_name}.\n"
            "Your task is MINIMAL IN-PLACE CORRECTION to eliminate over-editing.\n\n"
            "Infilling Protocol:\n"
            "1. DO NOT rewrite or alter any text outside the <error_span> tags. Retain surrounding fluent prose.\n"
            "2. Correct each identified error span accurately.\n"
            "3. Output each correction using format:\n"
            '<correction id="1">replacement_text</correction>\n'
            'If multiple spans exist, output <correction id="X"> for each. '
            "Or output the complete text with only the marked errors repaired in <final_translation>...</final_translation>."
            f"{table_infilling_note}"
        )

        user_parts.append(
            f"### Original Source\n{_neutralize_reserved_tags(source_text).strip()}\n\n"
            f"### Translation Draft with Error Spans\n{annotated_draft.strip()}\n\n"
            f"### Quality Critique Issues\n{issues_formatted}\n\n"
            "### Instruction:\n"
            'Correct the marked error spans with minimal in-place edits using <correction id="...">replacement</correction>:'
        )
    else:
        table_rule = ""
        if grid_info:
            rows, cols = grid_info
            table_rule = (
                f"6. [Table Structure Preservation ({rows}x{cols} Grid)]: The source is a table with {rows} rows and {cols} columns. "
                f"Strictly preserve the exact row count and column count. Never delete, drop, or merge rows or columns during repair, "
                f"even if a critique notes repetition or loops. Maintain the complete grid shape perfectly.\n\n"
            )

        system_prompt = (
            f"You are a master bilingual editor and precision repair agent refining a translation from {src_name} into {tgt_name}.\n"
            "A draft translation failed quality checks. Conduct a focused, structured review before outputting the final translation.\n\n"
            "Evaluation & Reflection Protocol:\n"
            "1. [Terminology]: Verify all terms against the Translation Bible strictly.\n"
            "2. [Critique Resolution]: Address and fix every flagged quality defect.\n"
            "3. [Fluency & Denoising]: Polish target prose to eliminate stiff translationese and mechanical passive constructions (naturalize passive voice to active/topic-prominent phrasing in CJK, 意合重组). Suppress any raw OCR glyph noise, stray symbols, or broken boilerplate fragments that leaked into the draft.\n"
            "4. [Scientific Math & Definition Integrity]: Reconstruct all inline physical variables, symbols, and subscripts/superscripts (e.g. Vtm, kBT/q, Vch) into LaTeX inline math ($...$). Only wrap notation already mathematical in the source — never invent LaTeX commands (\\mathrm, \\text, ...) or $...$ for plain labels verbatim in the source (2D, 3D, Fig. 1.1); reproduce those exactly. Never drop explanatory definition clauses that follow equations (e.g. 'where...', '其中...').\n"
            "5. [Publishing Typography & CJK Spacing]: In Chinese target text, leave a single half-width space between Chinese characters and English words, digits, or inline math ($...$), and eliminate stray spaces around full-width CJK punctuation.\n"
            f"{table_rule}"
            "Output Requirement:\n"
            "Wrap the final polished translation strictly inside <final_translation>...</final_translation> without commentary or markdown code blocks."
        )

        user_parts.append(
            f"### Original Source\n{_neutralize_reserved_tags(source_text).strip()}\n\n"
            f"### Problematic Draft\n{draft_text.strip()}\n\n"
            f"### Quality Critique Issues\n{issues_formatted}\n\n"
            "### Instruction:\n"
            "Reflect on the issues above and provide the corrected translation wrapped in <final_translation>...</final_translation>:"
        )

    return system_prompt, "\n".join(user_parts)


__all__ = [
    "build_hybrid_draft_prompt",
    "build_hybrid_repair_prompt",
    "build_macro_chunk_draft_prompt",
    "build_minimal_draft_prompt",
    "build_minimal_repair_prompt",
    "build_rich_draft_prompt",
    "build_rich_repair_prompt",
    "draft_source_from_prompt",
]
