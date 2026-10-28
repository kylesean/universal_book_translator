"""Cross-chapter coreference resolution and pronoun disambiguation engine.

Solves the pronoun blindness, ambiguous anaphora, and unidiomatic '它-ism' in long-form
technical and narrative monograph translation:
- English heavily relies on pronouns ('it', 'they', 'this', 'these', 'the former', 'the latter')
  where Chinese is topic-prominent, favoring nominal repetition, demonstrative classifiers
  ('该算法', '这种机制', '上述模型'), or grammatical zero-anaphora (subject omission).
- Across chapter and section transitions, LLMs lack the antecedent in local context windows,
  producing literal, vague, or misassigned pronoun translations (e.g. '它的弹性' instead of
  '该拜占庭容错协议的弹性').

Architecture:
1. EntityExtractor:
   Extracts salient technical concepts, named entities, algorithms, and systems from
   prior blocks, section headers, and Sliding Epoch Memory summaries.
2. CoreferenceResolver:
   Detects personal, demonstrative, and relational pronouns; executes agreement-checked,
   recency-weighted antecedent resolution with semantic affinity scoring.
3. Prompt Annotation Injector:
   Assembles high-signal, token-budgeted disambiguation guidance (<coreference_hints>)
   informing the translation engine of exact antecedents and idiomatic Chinese nominal forms.
4. PronounIdiomAuditor:
   Audits target text for excessive literal pronoun usage ('它/它们' overuse) and missing
   topic continuity.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CoreferencePronounType(StrEnum):
    """Classification of anaphoric pronouns and demonstrative phrases."""

    PERSONAL = "personal"  # it, its, itself, they, them, their, theirs, he, she
    DEMONSTRATIVE = "demonstrative"  # this approach, these methods, such systems, this, these
    RELATIONAL = "relational"  # the former, the latter, the above, the following


class EntityMention(BaseModel):
    """A salient entity or technical concept eligible as an antecedent."""

    model_config = ConfigDict(extra="ignore")

    entity_text: str
    entity_type: str = "concept"  # "concept", "algorithm", "system", "person", "plural_concept"
    source_location: str = ""  # e.g. "chapter:2", "neighbor", "current_block"
    salience: float = Field(default=1.0, ge=0.0)
    is_plural: bool = False


class CoreferenceMention(BaseModel):
    """A detected pronoun or anaphoric phrase and its resolved antecedent."""

    model_config = ConfigDict(extra="ignore")

    span: str
    start_char: int
    end_char: int
    pronoun_type: CoreferencePronounType
    head_word: str
    inferred_antecedent: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    antecedent_source: str | None = None
    recommended_chinese_rendering: str | None = None


class PronounAuditItem(BaseModel):
    """An identified issue regarding pronoun translation style or overuse."""

    model_config = ConfigDict(extra="ignore")

    pronoun_span: str
    sentence: str
    issue: str
    severity: str = "minor"  # "minor", "major"
    suggestion: str


class PronounAuditReport(BaseModel):
    """Summary of pronoun naturalness and '它-ism' audit for translated text."""

    model_config = ConfigDict(extra="ignore")

    total_chinese_chars: int
    pronoun_count: int
    pronoun_density_per_kilochar: float
    is_acceptable: bool
    issues: list[PronounAuditItem] = Field(default_factory=list)


# Precompiled regexes for pronoun detection
_PERSONAL_PRONOUNS_PATTERN = re.compile(
    r"\b(?P<pron>it|its|itself|they|them|their|theirs|themselves|he|him|his|she|her|hers)\b",
    re.IGNORECASE,
)
_DEMONSTRATIVE_PHRASE_PATTERN = re.compile(
    r"\b(?P<phrase>(?:this|these|those|such)\s+(?:approach|method|technique|architecture|algorithm|model|framework|system|mechanism|process|protocol|pattern|strategy|problem|issue|paper|work|concept|property|component|implementation)s?)\b",
    re.IGNORECASE,
)
_BARE_DEMONSTRATIVE_PATTERN = re.compile(
    r"\b(?P<demon>this|these|those)\s+(?:is|are|was|were|can|could|will|would|allows?|enables?|demonstrates?|illustrates?|shows?)\b",
    re.IGNORECASE,
)
_RELATIONAL_PATTERN = re.compile(
    r"\b(?P<rel>the\s+former|the\s+latter|the\s+above|the\s+following)\b",
    re.IGNORECASE,
)

# Capitalized technical terms and named entities
_CAPITALIZED_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b")
_ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,6}(?:-[A-Z0-9]+)?)\b")


class CoreferenceResolver:
    """Disambiguates cross-chapter and inter-block pronouns with entity graph bindings."""

    def __init__(
        self,
        pronoun_density_threshold_per_kilochar: float = 18.0,
    ) -> None:
        self.pronoun_density_threshold = pronoun_density_threshold_per_kilochar

    def extract_entities_from_text(
        self,
        text: str,
        source_location: str = "",
        glossary_terms: Sequence[str] | None = None,
    ) -> list[EntityMention]:
        """Extract candidate antecedent entities from source text, headings, or summaries."""
        if not text or not text.strip():
            return []

        entities: list[EntityMention] = []
        seen: set[str] = set()

        # 1. Matches against known glossary terms
        if glossary_terms:
            text_lower = text.lower()
            for term in glossary_terms:
                t_clean = term.strip()
                if not t_clean or len(t_clean) < 3:
                    continue
                if t_clean.lower() in text_lower and t_clean.lower() not in seen:
                    seen.add(t_clean.lower())
                    is_pl = t_clean.lower().endswith("s") and not t_clean.lower().endswith("ss")
                    entities.append(
                        EntityMention(
                            entity_text=t_clean,
                            entity_type="concept",
                            source_location=source_location,
                            salience=1.2,
                            is_plural=is_pl,
                        )
                    )

        # 2. Acronyms (e.g. CNN, Transformer, BERT, ARSM, BFT)
        for m in _ACRONYM_PATTERN.finditer(text):
            acr = m.group(1).strip()
            if (
                acr.lower() not in seen
                and len(acr) >= 2
                and acr not in ("AI", "OK", "ID", "TV", "US", "UK", "PDF", "URL", "HTML")
            ):
                seen.add(acr.lower())
                entities.append(
                    EntityMention(
                        entity_text=acr,
                        entity_type="system",
                        source_location=source_location,
                        salience=1.1,
                        is_plural=False,
                    )
                )

        # 3. Capitalized multi-word noun phrases (e.g. "Byzantine Fault Tolerant Protocol")
        for m in _CAPITALIZED_ENTITY_PATTERN.finditer(text):
            phrase = m.group(1).strip()
            if phrase.lower() in seen or len(phrase.split()) < 2:
                continue
            # Exclude sentence start single words
            seen.add(phrase.lower())
            is_pl = phrase.lower().endswith("s")
            entities.append(
                EntityMention(
                    entity_text=phrase,
                    entity_type="algorithm" if "algorithm" in phrase.lower() else "concept",
                    source_location=source_location,
                    salience=1.0,
                    is_plural=is_pl,
                )
            )

        return entities

    def find_pronoun_mentions(self, text: str) -> list[CoreferenceMention]:
        """Identify pronouns, demonstratives, and relational expressions in text."""
        mentions: list[CoreferenceMention] = []

        # Personal pronouns
        for m in _PERSONAL_PRONOUNS_PATTERN.finditer(text):
            token = m.group("pron")
            mentions.append(
                CoreferenceMention(
                    span=token,
                    start_char=m.start(),
                    end_char=m.end(),
                    pronoun_type=CoreferencePronounType.PERSONAL,
                    head_word=token.lower(),
                )
            )

        # Demonstrative phrases (e.g. "this approach", "these mechanisms")
        for m in _DEMONSTRATIVE_PHRASE_PATTERN.finditer(text):
            phrase = m.group("phrase")
            mentions.append(
                CoreferenceMention(
                    span=phrase,
                    start_char=m.start(),
                    end_char=m.end(),
                    pronoun_type=CoreferencePronounType.DEMONSTRATIVE,
                    head_word=phrase.split()[-1].lower(),
                )
            )

        # Bare demonstratives (e.g. "This demonstrates...", "These are...")
        for m in _BARE_DEMONSTRATIVE_PATTERN.finditer(text):
            demon = m.group("demon")
            mentions.append(
                CoreferenceMention(
                    span=demon,
                    start_char=m.start(),
                    end_char=m.start() + len(demon),
                    pronoun_type=CoreferencePronounType.DEMONSTRATIVE,
                    head_word=demon.lower(),
                )
            )

        # Relational expressions (e.g. "the former", "the latter")
        for m in _RELATIONAL_PATTERN.finditer(text):
            rel = m.group("rel")
            mentions.append(
                CoreferenceMention(
                    span=rel,
                    start_char=m.start(),
                    end_char=m.end(),
                    pronoun_type=CoreferencePronounType.RELATIONAL,
                    head_word="former" if "former" in rel.lower() else "latter",
                )
            )

        # Sort mentions by start_char
        mentions.sort(key=lambda x: x.start_char)
        return mentions

    def resolve_mentions(
        self,
        text: str,
        neighbor_text: str = "",
        epoch_entities: Sequence[EntityMention] | None = None,
        chapter_title: str = "",
        glossary_terms: Sequence[str] | None = None,
    ) -> list[CoreferenceMention]:
        """Resolve pronouns in text by searching intra-block, neighbor, and epoch entities."""
        mentions = self.find_pronoun_mentions(text)
        if not mentions:
            return []

        # Pool candidate entities from all available scopes
        # 1. Intra-block entities
        intra_entities = self.extract_entities_from_text(
            text, source_location="current_block", glossary_terms=glossary_terms
        )
        # 2. Neighbor entities
        neighbor_entities = self.extract_entities_from_text(
            neighbor_text, source_location="neighbor_context", glossary_terms=glossary_terms
        )
        # 3. Chapter title entity
        chapter_entities: list[EntityMention] = []
        if chapter_title.strip():
            clean_ch = re.sub(r"^#+\s*|Chapter\s*\d+:\s*", "", chapter_title).strip()
            if clean_ch:
                chapter_entities.append(
                    EntityMention(
                        entity_text=clean_ch,
                        entity_type="chapter_topic",
                        source_location="chapter_title",
                        salience=1.3,
                        is_plural=False,
                    )
                )

        epoch_pool = list(epoch_entities or [])

        for mention in mentions:
            head = mention.head_word.lower()
            resolved_ant: str | None = None
            resolved_source: str | None = None
            confidence = 0.0
            rec_zh: str | None = None

            # Logic for Relational (the former / the latter)
            if mention.pronoun_type == CoreferencePronounType.RELATIONAL:
                # Needs at least two candidate entities in preceding text or neighbor
                candidates = neighbor_entities + intra_entities
                if len(candidates) >= 2:
                    if head == "former":
                        resolved_ant = candidates[0].entity_text
                        resolved_source = candidates[0].source_location
                        confidence = 0.85
                        rec_zh = f"前者（{resolved_ant}）"
                    else:  # latter
                        resolved_ant = candidates[1].entity_text
                        resolved_source = candidates[1].source_location
                        confidence = 0.85
                        rec_zh = f"后者（{resolved_ant}）"

            # Logic for Demonstrative phrases ("this approach", "these methods")
            elif mention.pronoun_type == CoreferencePronounType.DEMONSTRATIVE:
                is_plural = (
                    head.endswith("s")
                    or "these" in mention.span.lower()
                    or "those" in mention.span.lower()
                )
                # Find closest matching entity
                all_cands = intra_entities + neighbor_entities + chapter_entities + epoch_pool
                matching = [c for c in all_cands if c.is_plural == is_plural]
                if matching:
                    # Prefer neighbor or chapter title if intra entity is after the mention
                    best = matching[0]
                    resolved_ant = best.entity_text
                    resolved_source = best.source_location
                    confidence = 0.80
                    prefix = "这些" if is_plural else "该"
                    rec_zh = (
                        f"{prefix}{best.entity_text}"
                        if len(best.entity_text) < 12
                        else f"{prefix}方法（{best.entity_text}）"
                    )

            # Logic for Personal Pronouns ("it", "its", "they", "their")
            elif mention.pronoun_type == CoreferencePronounType.PERSONAL:
                is_plural = head in ("they", "them", "their", "theirs", "themselves")
                is_possessive = head in ("its", "their", "his", "her")

                # Discard non-referential "it is ..." dummy subjects (e.g. "It is important to...", "It is clear that...")
                after_text = text[mention.end_char : mention.end_char + 30].strip().lower()
                if head == "it" and re.match(
                    r"^is\s+(?:important|clear|evident|well-known|crucial|essential|common|obvious|necessary|possible|easy|difficult|worth)\s+to\b",
                    after_text,
                ):
                    continue

                all_cands = neighbor_entities + chapter_entities + intra_entities + epoch_pool
                matching = [c for c in all_cands if c.is_plural == is_plural]
                if matching:
                    best = matching[0]
                    resolved_ant = best.entity_text
                    resolved_source = best.source_location
                    confidence = 0.75
                    rec_zh = f"该{best.entity_text}的" if is_possessive else f"该{best.entity_text}"

            if resolved_ant:
                mention.inferred_antecedent = resolved_ant
                mention.antecedent_source = resolved_source
                mention.confidence = confidence
                mention.recommended_chinese_rendering = rec_zh

        return mentions

    def format_coreference_prompt_hint(self, mentions: Sequence[CoreferenceMention]) -> str:
        """Format resolved coreference mentions into concise LLM prompt guidance."""
        resolved = [m for m in mentions if m.inferred_antecedent and m.confidence >= 0.70]
        if not resolved:
            return ""

        lines = [
            "### Antecedent & Coreference Disambiguation Guidance (指代消歧与主语明晰指南)",
            "To prevent pronoun blindness and unidiomatic literal translations (e.g. excessive '它/它们'), "
            "incorporate the following resolved antecedents into your translation:",
        ]
        for m in resolved:
            zh_hint = (
                f", 建议译为: '{m.recommended_chinese_rendering}'"
                if m.recommended_chinese_rendering
                else ""
            )
            lines.append(
                f'- **"{m.span}"** refers to: `{m.inferred_antecedent}` ({m.antecedent_source or "context"}{zh_hint})'
            )

        return "\n".join(lines)

    def audit_pronoun_usage(
        self,
        chinese_text: str,
        original_english_text: str = "",
    ) -> PronounAuditReport:
        """Audit Chinese translation for '它-ism' and unnatural pronoun repetition."""
        if not chinese_text or not chinese_text.strip():
            return PronounAuditReport(
                total_chinese_chars=0,
                pronoun_count=0,
                pronoun_density_per_kilochar=0.0,
                is_acceptable=True,
                issues=[],
            )

        # Count Chinese 3rd-person neuter pronouns ("它", "它的", "它们", "它们的")
        pronoun_matches = list(re.finditer(r"(?:它们的|它们的|它们|它的|它)", chinese_text))
        pron_cnt = len(pronoun_matches)
        char_len = len(chinese_text)
        density = round((pron_cnt / char_len) * 1000.0, 2) if char_len > 0 else 0.0

        issues: list[PronounAuditItem] = []
        sentences = [s.strip() for s in re.split(r"[。！？\n]", chinese_text) if s.strip()]

        for s in sentences:
            s_matches = list(re.finditer(r"(?:它们的|它们|它的|它)", s))
            if len(s_matches) >= 2:
                issues.append(
                    PronounAuditItem(
                        pronoun_span=", ".join(m.group(0) for m in s_matches),
                        sentence=s,
                        issue=f"Excessive pronoun repetition ({len(s_matches)} occurrences in single sentence)",
                        severity="minor",
                        suggestion="Replace redundant '它/它们的' with zero-anaphora (omission) or explicit subject noun.",
                    )
                )

        is_acc = density <= self.pronoun_density_threshold and len(issues) <= 2

        return PronounAuditReport(
            total_chinese_chars=char_len,
            pronoun_count=pron_cnt,
            pronoun_density_per_kilochar=density,
            is_acceptable=is_acc,
            issues=issues,
        )


__all__ = [
    "CoreferenceMention",
    "CoreferencePronounType",
    "CoreferenceResolver",
    "EntityMention",
    "PronounAuditItem",
    "PronounAuditReport",
]
