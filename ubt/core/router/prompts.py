"""Draft and repair prompt assembly, and the prompt-layout identity.

Pure functions of their arguments -- no router state, no provider, no I/O -- so
they do not have to live in :mod:`ubt.core.router.router` for that file to mean
"routing". Keeping them here also puts the marker literals the draft builders
write and :func:`draft_source_from_prompt` reads back into one file: they are one
contract, and a builder whose heading the parser does not know about leaks its
instruction tail into what ``--dry-run`` shows the user.

``PROMPT_VERSION`` is the *identity* half of a translation-memory exact hit
(see :func:`ubt.core.memory.tm.compute_tm_context`): the TM key includes it, so
a stored translation is reused only when the layout that produced it is the
layout in use now. It used to be a hand-maintained ``"v1"`` in ``tm.py`` with
a comment asking the next person to bump it — and nothing checked the promise,
so the day a layout constant changed without a bump, stored translations would
have become silent exact hits for a different prompt.
"""

from __future__ import annotations

import hashlib
import html
import re

from ubt.core.ir.emphasis import (
    BOLD_CLOSE,
    BOLD_OPEN,
    ITALIC_CLOSE,
    ITALIC_OPEN,
    strip_emphasis_markers,
)
from ubt.core.language_profile import PROFILES
from ubt.core.prompts.assembler import (
    assemble_continuity_context,
    assemble_critique_issues,
    assemble_terminology_table,
)
from ubt.core.prompts.budget import truncate_text_to_budget

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

#: The repair builders' source heading and the sections that follow it. A
#: repair prompt carries no draft marker, so without these the rehearsal echo
#: fell through to ``prompt.strip()`` and replayed the whole scaffold. The tails
#: are every heading the three repair builders emit after the source, in the
#: order they can appear (``### Translation Draft with Error Spans`` is the
#: infilling shape, ``### Problematic Draft`` / ``### Previous Draft`` the
#: others).
_REPAIR_SOURCE_MARKER = "### Original Source\n"
_REPAIR_SOURCE_TAILS = (
    "\n\n### Translation Draft with Error Spans",
    "\n\n### Draft with Marked Errors",
    "\n\n### Problematic Draft",
    "\n\n### Previous Draft",
    "\n\n### Issues to Fix",
    "\n\n### Quality Critique Issues",
)
#: ``build_minimal_repair_prompt`` has no headings: it inlines the source after
#: ``Source <language> text:`` and follows it with the draft's own defects. The
#: language name is interpolated, so this one is a pattern rather than a literal.
_REPAIR_MINIMAL_SOURCE_RE = re.compile(
    r"Fix the draft translation below\. Source [^\n]*? text:\n\n"
)
_REPAIR_MINIMAL_TAIL = "\n\nDetected problems in the draft: "

#: A source this short, with no sentence-ending punctuation, is a heading,
#: label, or list item rather than prose: a model that expands it into a
#: sentence/paragraph trips the QE length gate and the block falls back to the
#: source. Nudge it to translate in kind. Appended after the existing
#: instruction so ``draft_source_from_prompt``'s anchors still parse.
_SHORT_SOURCE_MAX_CHARS = 60
#: A source the extractor *labelled* a heading is heading-like whatever its
#: length -- a paper title routinely runs past the short-source heuristic
#: (``DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure ...`` is 96
#: chars) and a small model then expands it into a title-plus-abstract-summary
#: draft that the QE length gate quarantines. The cap stays to keep a
#: mislabelled long paragraph from being told to "render equally short".
_HEADING_MAX_CHARS = 200
_HEADING_BLOCK_TYPES = frozenset({"heading", "title"})
_SHORT_SOURCE_HINT = (
    " The source is a short heading or label: render it as an equally short "
    "heading or label, never expanded into a sentence or paragraph, and add no "
    "explanation of your own."
)

#: The draft stage brackets a partial bold source span with these sentinels (see
#: :mod:`ubt.core.ir.emphasis`); the model must carry them onto the translated
#: text so the render can re-apply the weight the source-derived run cannot
#: locate. The rule is static so the prompt prefix stays cacheable.
_EMPHASIS_RULE = (
    f"9. [Emphasis Preservation]: Source text bracketed by {BOLD_OPEN} ... "
    f"{BOLD_CLOSE} is bold, and {ITALIC_OPEN} ... {ITALIC_CLOSE} is italic. Wrap the corresponding translated text with the SAME "
    f"markers, kept balanced and in place. Never "
    "translate, drop, reorder, duplicate, or add emphasis markers of your own."
)
_EMPHASIS_HINT = (
    f" Preserve any {BOLD_OPEN} ... {BOLD_CLOSE} or {ITALIC_OPEN} ... {ITALIC_CLOSE} emphasis markers around the "
    "corresponding translated text, balanced and in place."
)

#: A source that mentions its own wrapper tags must not be mistaken for the
#: answer envelope; the escaping above is part of the layout.
#:
#: The layout identity: built from every literal that shapes what the model
#: sees. Change one and ``PROMPT_VERSION`` changes with it — which is the
#: point. Hand-bumping a version string is what the old design asked for and
#: what nothing enforced.
_PROMPT_LAYOUT_PARTS: tuple[str, ...] = (
    _DRAFT_SOURCE_MARKER,
    *_DRAFT_SOURCE_TAILS,
    _DRAFT_MINIMAL_ANCHOR,
    _REPAIR_SOURCE_MARKER,
    *_REPAIR_SOURCE_TAILS,
    _SHORT_SOURCE_HINT,
    _EMPHASIS_RULE,
    _EMPHASIS_HINT,
    # The reserved-tag escaping pattern is behaviour, not decoration: relaxing
    # it would let a source span become the envelope.
    _RESERVED_WRAPPER_RE.pattern,
)
_PROMPT_LAYOUT_DIGEST = hashlib.sha256(
    "\x00".join(_PROMPT_LAYOUT_PARTS).encode("utf-8")
).hexdigest()[:12]

#: The translation-memory identity for this prompt layout. It changes whenever
#: a layout literal above changes, so a stored translation can never be reused
#: across a prompt-layout change (see ``compute_tm_context``).
PROMPT_VERSION = f"v1-{_PROMPT_LAYOUT_DIGEST}"


def _short_source_hint(source_text: str, block_type: str | None = None) -> str:
    """A "do not expand" clause for a heading-like source, else empty.

    The block's own type is the reliable signal: an explicit ``heading`` fires
    the clause up to :data:`_HEADING_MAX_CHARS` regardless of punctuation. A
    caller that passes no type (the cost probe, older call sites) falls back to
    the short-and-unpunctuated heuristic.
    """
    stripped = strip_emphasis_markers(source_text or "").strip()
    if not stripped:
        return ""
    is_heading = str(block_type or "").strip().lower() in _HEADING_BLOCK_TYPES
    limit = _HEADING_MAX_CHARS if is_heading else _SHORT_SOURCE_MAX_CHARS
    if len(stripped) > limit:
        return ""
    if not is_heading and stripped.endswith(
        (".", "?", "!", "。", "？", "！", ":", "：", ";", "；")
    ):
        return ""
    return _SHORT_SOURCE_HINT


def draft_source_from_prompt(prompt: str) -> str:
    """Recover the source span a prompt carried, draft *or* repair.

    ``--dry-run`` previews a translation without a provider, so it has to read
    the block back out of the prompt it was handed. This is the one place that
    knows the markers: parsing them per surface let a builder's instruction tail
    leak into the "translation" the user saw.

    The repair builders matter as much as the draft ones. The rehearsal provider
    answers *every* request through this function, repair rounds included, and a
    repair prompt carries no draft marker at all -- so the fallback returned the
    whole prompt, and the "repaired" text became the prompt scaffold: measured
    on the corpus as ``### [模拟翻译] Terminology Glossary … ### Original
    Source … ### Previous Draft …``, which the QE pass then (correctly) flagged
    as *prompt scaffold leaked into the target*.
    """
    if "<blocks>" in prompt:
        remainder = prompt.split("<blocks>", 1)[1]
        if "</blocks>" in remainder:
            return strip_emphasis_markers(remainder.split("</blocks>", 1)[0].strip())
        return strip_emphasis_markers(remainder.strip())
    if _DRAFT_SOURCE_MARKER in prompt:
        remainder = prompt.split(_DRAFT_SOURCE_MARKER, 1)[1]
        for tail in _DRAFT_SOURCE_TAILS:
            if tail in remainder:
                remainder = remainder.split(tail, 1)[0]
                break
        return strip_emphasis_markers(remainder.strip())
    if _DRAFT_MINIMAL_ANCHOR in prompt:
        return strip_emphasis_markers(prompt.rsplit(_DRAFT_MINIMAL_ANCHOR, 1)[1].strip())
    if _REPAIR_SOURCE_MARKER in prompt:
        remainder = prompt.split(_REPAIR_SOURCE_MARKER, 1)[1]
        for tail in _REPAIR_SOURCE_TAILS:
            if tail in remainder:
                remainder = remainder.split(tail, 1)[0]
                break
        return strip_emphasis_markers(remainder.strip())
    minimal_repair = _REPAIR_MINIMAL_SOURCE_RE.search(prompt)
    if minimal_repair is not None:
        remainder = prompt[minimal_repair.end() :]
        if _REPAIR_MINIMAL_TAIL in remainder:
            remainder = remainder.split(_REPAIR_MINIMAL_TAIL, 1)[0]
        return strip_emphasis_markers(remainder.strip())
    return prompt.strip()


def _build_style_description(
    genre_profile: str,
    src_name: str,
    tgt_name: str,
    domain: str | None = None,
) -> str:
    genre = (genre_profile or "").strip().lower()
    if genre == "fiction":
        desc = (
            f"You are an acclaimed literary translator translating from {src_name} into {tgt_name}.\n"
            "Emphasize natural prose rhythm, character voice, subtext, and atmospheric immersion."
        )
    elif genre in ("academic", "paper"):
        desc = (
            f"You are a scholarly scientific translator translating from {src_name} into {tgt_name}.\n"
            "Maintain strict scientific rigor, objective academic register, and precise domain terms."
        )
    elif genre == "textbook":
        desc = (
            f"You are an educational textbook translator translating from {src_name} into {tgt_name}.\n"
            "Prioritize pedagogical clarity, conceptual fidelity, and retain all visual callouts."
        )
    elif genre == "technical":
        desc = (
            f"You are an expert technical translator translating from {src_name} into {tgt_name}.\n"
            "Prioritize technical precision, unambiguous terminology, operational clarity, and structural consistency."
        )
    elif genre in ("nonfiction", "non-fiction"):
        desc = (
            f"You are an authoritative non-fiction book translator translating from {src_name} into {tgt_name}.\n"
            "Balance intellectual rigor, factual accuracy, engaging narrative cadence, and clear exposition."
        )
    elif genre == "humanities":
        desc = (
            f"You are a scholarly humanities translator translating from {src_name} into {tgt_name}.\n"
            "Preserve nuanced philosophical arguments, cultural context, intellectual register, and interpretive depth."
        )
    else:
        desc = (
            f"You are a professional book translator translating from {src_name} into {tgt_name}.\n"
            "Translate accurately while maintaining natural prose and literary flow."
        )
    if domain:
        desc += f"\nUse standard {domain} terminology."
    return desc


def build_minimal_draft_prompt(
    source_text: str,
    glossary_table: str = "",
    target_lang: str = "zh",
    source_lang: str = "en",
    global_glossary: str = "",
    few_shot_reference: str = "",
    genre_profile: str = "general",
    domain: str | None = None,
    block_type: str | None = None,
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

    b_global, b_chunk = assemble_terminology_table(global_glossary, glossary_table)
    glossary_sections = [section.strip() for section in (b_global, b_chunk) if section.strip()]
    parts = []
    if glossary_sections:
        parts.append("Glossary:\n" + "\n\n".join(glossary_sections))
    if few_shot_reference.strip():
        parts.append(truncate_text_to_budget(few_shot_reference.strip(), 1024))
    domain_hint = ""
    if domain:
        domain_hint = f" Use standard {domain} terminology."
    elif genre_profile and genre_profile.lower() not in ("general", "unknown", "auto"):
        domain_hint = f" Use standard {genre_profile} domain terminology."
    parts.append(
        f"Translate the following {src_name} text into fluent, natural {tgt_name}.{domain_hint}{_short_source_hint(source_text, block_type)}{_EMPHASIS_HINT} Output ONLY the translation without any title, prefix, or commentary:\n\n{_neutralize_reserved_tags(source_text).strip()}"
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
    block_type: str | None = None,
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

    style_desc = _build_style_description(genre_profile, src_name, tgt_name, domain)

    if "zh" in tgt_name.lower() or (tgt_profile and "zh" in tgt_profile.code.lower()):
        system_prompt = (
            f"{style_desc}\n"
            "Translate accurately while maintaining natural target language flow, idiomatic register, and domain terminology.\n"
            "Typography & Terminology: In Chinese target text, maintain standard spacing between Chinese characters and English words, numbers, or inline formulas. "
            "Use authoritative discipline terminology suited to the active subject domain and preserve standard uppercase technical acronyms.\n"
            + _EMPHASIS_RULE
        )
    else:
        system_prompt = (
            f"{style_desc}\n"
            "Translate accurately while maintaining natural target language flow, idiomatic register, and domain terminology.\n"
            + _EMPHASIS_RULE
        )

    b_global, b_chunk = assemble_terminology_table(global_glossary, glossary_table)
    b_epoch, b_rolling = assemble_continuity_context(epoch_summary, rolling_summary)
    b_neighbor = truncate_text_to_budget(neighbor_context.strip(), 1536)
    b_few_shot = truncate_text_to_budget(few_shot_reference.strip(), 1024)

    user_parts: list[str] = []
    if b_global.strip():
        user_parts.append(f"### Terminology Glossary\n{b_global.strip()}\n")

    if b_epoch.strip():
        user_parts.append(f"### Book Continuity (Compressed History)\n{b_epoch.strip()}\n")

    if b_rolling.strip():
        user_parts.append(f"### Story Context\n{b_rolling.strip()}\n")

    if b_chunk.strip():
        user_parts.append(f"### Terminology Glossary (this section)\n{b_chunk.strip()}\n")

    if b_neighbor.strip():
        user_parts.append(f"{b_neighbor.strip()}\n")

    if b_few_shot.strip():
        user_parts.append(f"{b_few_shot.strip()}\n")

    user_parts.append(
        f"### Source Paragraph to Translate\n{_neutralize_reserved_tags(source_text).strip()}\n\n"
        "Translate only the text under this heading. Any read-only reference "
        "context shown above is not part of the task: do not translate it, "
        "summarise it, repeat it, or carry its citations and equation numbers "
        f"into the output.{_short_source_hint(source_text, block_type)}\n\n"
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
    block_type: str | None = None,
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

    style_desc = _build_style_description(genre_profile, src_name, tgt_name, domain)

    # Tier 1: Static Immutable Prefix
    system_parts = [
        style_desc,
        "Rules:\n"
        "1. Strictly adhere to the terminology mappings in the Translation Bible if provided.\n"
        "2. Preserve all XML/HTML markup tags and placeholder tokens (⟦CODE_MASK_...⟧, ⟦MATH_MASK_...⟧, ⟦CITE_MASK_...⟧) without omitting, modifying, or duplicating them. Their order may follow the target language's natural syntax.\n"
        f"3. [Natural Register]:\n"
        f"   - Write fluent, idiomatic {tgt_name} in the register the source uses, without awkward literal translationese.\n"
        f"   - In CJK targets (Chinese/Japanese/Korean), prefer active, subjectless, or topic-prominent phrasing where it reads more naturally than a passive construction (e.g. 'Measurements were taken' -> '进行了测量').\n"
        f"   - Syntactic re-chunking: render long, multi-clause Western sentences as clean, logically balanced target prose (意合结构), maintaining natural narrative rhythm without omitting meaning.\n"
        f"4. [Faithful Reconstruction]: Render the source's genuine authorial/conceptual content as clean, fluent {tgt_name}. Where the source came from a scanned/OCR page, reconstruct the intended text rather than reproducing scanner noise, and keep any sidebar/callout text distinct from the main narrative.\n"
        f"5. [Scientific Math & Formula Fidelity]:\n"
        f"   - Reconstruct all inline physical variables, mathematical symbols, Greek letters, and sub/superscript notations (such as variables, constants, e.g. $x_i$, $k_B T$, $\\psi(x, y)$) into standard LaTeX inline math syntax enclosed by single dollar signs ($...$). Never leave them as flattened plain ASCII strings.\n"
        f"   - Only wrap notation that is ALREADY mathematical in the source. NEVER invent LaTeX control sequences (\\mathrm, \\text, \\mathbf, ...) or $...$ delimiters for plain alphanumeric labels that appear as plain text in the source (dimension tags like 2D/3D, figure/table callouts like Fig. 1.1, units like 10nm) — reproduce those verbatim.\n"
        f"   - Never omit or truncate explanatory clauses that follow formulas (such as clauses starting with 'where...', 'in which...', '式中...', '其中...'). Translate them fully and faithfully.\n"
        f"6. [Publishing Typography & CJK Spacing]:\n"
        f"   - In Chinese/CJK target text, always maintain standard typographic spacing: leave a single half-width space between CJK characters and Latin words, numbers, and inline formulas (e.g. '在第 3 章'、'公式 $x$ 中').\n"
        f"   - Full-width punctuation marks (，。！？；：“”‘’（）《》) must remain snug against adjacent text without extra spaces.\n"
        f"7. [Domain Terminology & Acronyms]:\n"
        f"   - Consistently use standard national and discipline terminology appropriate for the document's subject field.\n"
        f"   - Standard domain acronyms must be preserved in uppercase Latin letters when introduced, accompanied by their established target language translation in parentheses where appropriate. NEVER mechanically duplicate or invent awkward pseudo-translations.\n"
        "8. Output your final translation strictly enclosed inside <translation>...</translation> tags. Any thinking, context analysis, or commentary must remain outside these tags. Do not include markdown code fences.\n"
        + _EMPHASIS_RULE,
    ]

    b_global, b_chunk = assemble_terminology_table(global_glossary, glossary_table)
    b_epoch, b_rolling = assemble_continuity_context(epoch_summary, rolling_summary)
    b_neighbor = truncate_text_to_budget(neighbor_context.strip(), 1536)
    b_few_shot = truncate_text_to_budget(few_shot_reference.strip(), 1024)

    if b_global.strip():
        system_parts.append(
            f"### Translation Bible (Strict Global Terminology Mappings)\n{b_global.strip()}"
        )

    system_prompt = "\n\n".join(system_parts)

    # Tier 2: Semi-static L3 epoch (changes only every N steps)
    # then Macro Snapshot Prefix & Chunk Glossary
    user_parts: list[str] = []
    if b_epoch.strip():
        user_parts.append(f"### Book Continuity (Compressed History)\n{b_epoch.strip()}\n")
    if b_rolling.strip():
        user_parts.append(
            f"### Document Continuation Context (Macro Snapshot)\n{b_rolling.strip()}\n"
        )
    if b_chunk.strip():
        user_parts.append(f"### Translation Bible (Strict Mappings)\n{b_chunk.strip()}\n")

    # Tier 3: Dynamic Tail (L1 Neighbor Excerpts + TM few-shot + Target IRBlock)
    if b_neighbor.strip():
        user_parts.append(f"{b_neighbor.strip()}\n")

    if b_few_shot.strip():
        user_parts.append(f"{b_few_shot.strip()}\n")

    user_parts.append(
        f"### Source Paragraph to Translate\n{_neutralize_reserved_tags(source_text).strip()}\n\n"
        "Translate only the text under this heading. Any read-only reference "
        "context shown above is not part of the task: do not translate it, "
        "summarise it, repeat it, or carry its citations and equation numbers "
        f"into the output.{_short_source_hint(source_text, block_type)}\n\n"
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
    section_context: str = "",
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

    style_desc = _build_style_description(genre_profile, src_name, tgt_name, domain)

    system_parts = [
        style_desc,
        "Rules:\n"
        "1. Strictly adhere to the terminology mappings in the Translation Bible if provided.\n"
        "2. Preserve all XML/HTML markup tags and placeholder tokens (⟦CODE_MASK_...⟧, ⟦MATH_MASK_...⟧, ⟦CITE_MASK_...⟧) without omitting, modifying, or duplicating them. Their order may follow the target language's natural syntax.\n"
        f"3. [Natural Register]:\n"
        f"   - Write fluent, idiomatic {tgt_name} in the register the source uses, without awkward literal translationese.\n"
        f"   - In CJK targets (Chinese/Japanese/Korean), prefer active, subjectless, or topic-prominent phrasing where it reads more naturally than a passive construction (e.g. 'Measurements were taken' -> '进行了测量').\n"
        f"   - Syntactic re-chunking: render long, multi-clause Western sentences as clean, logically balanced target prose (意合结构), maintaining natural narrative rhythm without omitting meaning.\n"
        f"4. [Faithful Reconstruction]: Render the source's genuine authorial/conceptual content as clean, fluent {tgt_name}. Where the source came from a scanned/OCR page, reconstruct the intended text rather than reproducing scanner noise, and keep any sidebar/callout text distinct from the main narrative.\n"
        f"5. [Scientific Math & Formula Fidelity]:\n"
        f"   - Reconstruct all inline physical variables, mathematical symbols, Greek letters, and sub/superscript notations (such as variables, constants, e.g. $x_i$, $k_B T$, $\\psi(x, y)$) into standard LaTeX inline math syntax enclosed by single dollar signs ($...$). Never leave them as flattened plain ASCII strings.\n"
        f"   - Only wrap notation that is ALREADY mathematical in the source. NEVER invent LaTeX control sequences (\\mathrm, \\text, \\mathbf, ...) or $...$ delimiters for plain alphanumeric labels that appear as plain text in the source (dimension tags like 2D/3D, figure/table callouts like Fig. 1.1, units like 10nm) — reproduce those verbatim.\n"
        f"   - Never omit or truncate explanatory clauses that follow formulas (such as clauses starting with 'where...', 'in which...', '式中...', '其中...'). Translate them fully and faithfully.\n"
        f"6. [Publishing Typography & CJK Spacing]:\n"
        f"   - In Chinese/CJK target text, always maintain standard typographic spacing: leave a single half-width space between CJK characters and Latin words, numbers, and inline formulas (e.g. '在第 3 章'、'公式 $x$ 中').\n"
        f"   - Full-width punctuation marks (，。！？；：“”‘’（）《》) must remain snug against adjacent text without extra spaces.\n"
        f"7. [Domain Terminology & Acronyms]:\n"
        f"   - Consistently use standard national and discipline terminology appropriate for the document's subject field.\n"
        f"   - Standard domain acronyms must be preserved in uppercase Latin letters when introduced, accompanied by their established target language translation in parentheses where appropriate. NEVER mechanically duplicate or invent awkward pseudo-translations.\n"
        '8. [Structured Macro-Blocks Output]: You must output your translation inside <blocks>...</blocks> tags. Each block must be enclosed in a <block id="...">...</block> tag with the EXACT matching id from the source. Translate ONLY the text of each block. Do not include markdown code fences or conversational commentary outside the tags.\n'
        + _EMPHASIS_RULE,
    ]

    b_global, b_chunk = assemble_terminology_table(global_glossary, glossary_table)
    b_epoch, b_rolling = assemble_continuity_context(epoch_summary, rolling_summary)
    b_neighbor = truncate_text_to_budget(neighbor_context.strip(), 1536)
    b_few_shot = truncate_text_to_budget(few_shot_reference.strip(), 1024)
    b_section = truncate_text_to_budget(section_context.strip(), 1536)

    if b_global.strip():
        system_parts.append(
            f"### Translation Bible (Strict Global Terminology Mappings)\n{b_global.strip()}"
        )

    system_prompt = "\n\n".join(system_parts)

    user_parts: list[str] = []
    if b_epoch.strip():
        user_parts.append(f"### Book Continuity (Compressed History)\n{b_epoch.strip()}\n")
    if b_rolling.strip():
        user_parts.append(
            f"### Document Continuation Context (Macro Snapshot)\n{b_rolling.strip()}\n"
        )
    if b_section.strip():
        # Where these blocks sit in the document. Placed ahead of the per-chunk
        # dynamic glossary to preserve prefix caching across chunks in the same section.
        user_parts.append(f"### Section Context\n{b_section.strip()}\n")
    if b_chunk.strip():
        user_parts.append(f"### Translation Bible (Strict Mappings)\n{b_chunk.strip()}\n")

    if b_neighbor.strip():
        user_parts.append(f"{b_neighbor.strip()}\n")

    if b_few_shot.strip():
        user_parts.append(f"{b_few_shot.strip()}\n")

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
    """Return (num_rows, num_cols) if text contains a markdown table, else None.

    The acceptance gate's own row standard, not a private one: a repair prompt
    must assert "preserve the exact NxM grid" exactly when the gate will check
    for a grid, or the model is ordered to preserve a table the gate then
    ignores (and prose with inline pipes was exactly that case: any line with
    two ``|`` counted as a grid row here, while :func:`markdown_grid_shape`
    requires the row to open with a pipe).
    """
    from collections import Counter

    from ubt.core.qe.fast_pass import markdown_grid_shape

    shape = markdown_grid_shape(text)
    if shape:
        return len(shape), Counter(shape).most_common(1)[0][0]
    return None


def build_minimal_repair_prompt(
    source_text: str,
    draft_text: str = "",
    error_flags: list[str] | None = None,
    glossary_table: str = "",
    target_lang: str = "zh",
    source_lang: str = "en",
    global_glossary: str = "",
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

    b_global, b_chunk = assemble_terminology_table(global_glossary, glossary_table)
    parts = []
    glossary_sections = [s.strip() for s in (b_global, b_chunk) if s.strip()]
    if glossary_sections:
        parts.append("Glossary:\n" + "\n\n".join(glossary_sections))
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
    global_glossary: str = "",
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
        "Fix all identified issues and output ONLY the corrected translation directly. "
        "Do NOT include any commentary, analysis, explanations, notes, or conversational preambles."
        f"{table_guardrail}"
    )

    b_global, b_chunk = assemble_terminology_table(global_glossary, glossary_table)
    issues_formatted = assemble_critique_issues(error_flags)

    user_parts: list[str] = []
    if b_global.strip():
        user_parts.append(f"### Terminology Glossary\n{b_global.strip()}\n")

    if b_chunk.strip():
        heading = (
            "### Terminology Glossary (this section)\n"
            if b_global.strip()
            else "### Terminology Glossary\n"
        )
        user_parts.append(f"{heading}{b_chunk.strip()}\n")

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
            "Please provide the corrected full translation directly without any commentary or explanations:"
        )
    else:
        user_parts.append(
            f"### Original Source\n{_neutralize_reserved_tags(source_text).strip()}\n\n"
            f"### Previous Draft\n{draft_text.strip()}\n\n"
            f"### Issues to Fix\n{issues_formatted}\n\n"
            "Please provide the polished and corrected translation directly without any commentary or explanations:"
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
    global_glossary: str = "",
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

    b_global, b_chunk = assemble_terminology_table(global_glossary, glossary_table)
    issues_formatted = assemble_critique_issues(error_flags)

    grid_info = _detect_table_grid(source_text)

    user_parts: list[str] = []
    if b_global.strip():
        user_parts.append(f"### Translation Bible (Global Terminology)\n{b_global.strip()}\n")
    if b_chunk.strip():
        user_parts.append(f"### Translation Bible\n{b_chunk.strip()}\n")

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
            "Or output the complete text with only the marked errors repaired in <final_translation>...</final_translation>.\n\n"
            "Output Requirement:\n"
            "Output ONLY the correction tags or <final_translation>...</final_translation>. "
            "Do NOT output any reflections, reviews, commentary, analysis, markdown notes, or conversational text outside the tags."
            f"{table_infilling_note}"
        )

        user_parts.append(
            f"### Original Source\n{_neutralize_reserved_tags(source_text).strip()}\n\n"
            f"### Translation Draft with Error Spans\n{annotated_draft.strip()}\n\n"
            f"### Quality Critique Issues\n{issues_formatted}\n\n"
            "### Instruction:\n"
            'Correct the marked error spans directly without commentary using <correction id="...">replacement</correction> or <final_translation>...</final_translation>:'
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
            "A draft translation failed quality checks. Fix all identified defects and output the final polished translation.\n\n"
            "Quality Guidelines:\n"
            "1. [Terminology]: Verify all terms against the Translation Bible strictly.\n"
            "2. [Critique Resolution]: Address and fix every flagged quality defect.\n"
            "3. [Fluency & Denoising]: Polish target prose to eliminate stiff translationese and mechanical passive constructions (naturalize passive voice to active/topic-prominent phrasing in CJK, 意合重组). Suppress any raw OCR glyph noise, stray symbols, or broken boilerplate fragments that leaked into the draft.\n"
            "4. [Scientific Math & Definition Integrity]: Reconstruct all inline physical variables, symbols, and subscripts/superscripts (e.g. x_i, k_B T, \\psi) into LaTeX inline math ($...$). Only wrap notation already mathematical in the source — never invent LaTeX commands (\\mathrm, \\text, ...) or $...$ for plain labels verbatim in the source (2D, 3D, Fig. 1.1); reproduce those exactly. Never drop explanatory definition clauses that follow equations (e.g. 'where...', '其中...').\n"
            "5. [Publishing Typography & CJK Spacing]: In Chinese target text, leave a single half-width space between Chinese characters and English words, digits, or inline math ($...$), and eliminate stray spaces around full-width CJK punctuation.\n"
            f"{table_rule}"
            "Output Requirement:\n"
            "Output ONLY the final polished translation strictly wrapped inside <final_translation>...</final_translation>. "
            "Do NOT output any reflections, reviews, commentary, analysis, markdown notes, or conversational text outside the tags."
        )

        user_parts.append(
            f"### Original Source\n{_neutralize_reserved_tags(source_text).strip()}\n\n"
            f"### Problematic Draft\n{draft_text.strip()}\n\n"
            f"### Quality Critique Issues\n{issues_formatted}\n\n"
            "### Instruction:\n"
            "Fix the issues above and output ONLY the corrected translation wrapped in <final_translation>...</final_translation> directly (no reflections or commentary):"
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
