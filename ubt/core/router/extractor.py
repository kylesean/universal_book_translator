"""Robust semantic output extractor for modern (2026) LLM translation pipelines."""

import html
import re

from ubt.core.router.capabilities import ExtractionStrategy


class TranslationOutputExtractor:
    """Robust extractor for LLM translation outputs.

    Adheres to the Semantic XML Boundary Tagging standard and supports
    configurable ExtractionStrategy (RAW, XML_TAG, CONVERSATIONAL_PREFIX, AUTO).
    """

    # Matches <final_translation>...</final_translation> (higher priority) or
    # <translation>...</translation>. GREEDY to the LAST close tag: a translation
    # that itself *mentions* the tag ("use <translation> tags") would otherwise
    # be truncated at the first literal close.
    _FINAL_TAG_PATTERN = re.compile(
        r"<\s*final_translation[^>]*>([\s\S]*)<\s*/\s*final_translation\s*>",
        flags=re.IGNORECASE,
    )
    _TAG_PATTERN = re.compile(
        r"<\s*translation[^>]*>([\s\S]*)<\s*/\s*translation\s*>",
        flags=re.IGNORECASE,
    )
    # Unclosed trailing tag (output truncated at max_tokens). Only accepted when
    # the tag OPENS the output — a literal tag mentioned mid-prose is content,
    # not a wrapper, and must never swallow the prefix.
    _FINAL_TAG_OPEN_PATTERN = re.compile(
        r"^\s*<\s*final_translation[^>]*>\s*([\s\S]*)$",
        flags=re.IGNORECASE,
    )
    _TAG_OPEN_PATTERN = re.compile(
        r"^\s*<\s*translation[^>]*>\s*([\s\S]*)$",
        flags=re.IGNORECASE,
    )

    # In-stream reasoning traces from local Ollama/vLLM or models outputting <think>, <thought>, etc.
    _REASONING_BLOCK_PATTERN = re.compile(
        r"<\s*(?:think|thought|thinking|reasoning)[^>]*>[\s\S]*?<\s*/\s*(?:think|thought|thinking|reasoning)\s*>",
        flags=re.IGNORECASE,
    )
    _UNCLOSED_REASONING_PATTERN = re.compile(
        # Anchored at the start: an unclosed reasoning tag is a leaked preamble
        # when it opens the output, but a mid-prose mention ('To enable
        # <reasoning> mode, set the flag. …') is real content. The previous
        # unanchored DOTALL form deleted everything from the mention to EOF.
        r"^\s*<\s*(?:think|thought|thinking|reasoning)[^>]*>[\s\S]*$",
        flags=re.IGNORECASE,
    )

    # Markdown code fences (when models wrap the answer in ```markdown ... ```)
    _CODE_FENCE_PATTERN = re.compile(
        r"^```(?:markdown|text|xml)?\s*([\s\S]*?)\s*```$",
        flags=re.IGNORECASE,
    )
    _MACRO_BLOCK_PATTERN = re.compile(
        r'<\s*block\s+id=["\'](.*?)["\']\s*>([\s\S]*?)<\s*/\s*block\s*>',
        flags=re.IGNORECASE,
    )

    # Clean prefix patterns strictly anchored to the start of string (^).
    # Only matches conversational lead-ins, NOT legitimate section headings.
    # Every pattern requires an explicit colon: the qualifiers are optional in
    # natural prose, so a pattern that could match a bare "翻译…" or
    # "Translation…" would eat the first words of a real translation.
    _CONVERSATIONAL_PREFIX_PATTERNS = [
        re.compile(
            r"^\s*Here\s+(?:is|are)\s+the\s+(?:final\s+|refined\s+|polished\s+)?translation:\s*",
            re.IGNORECASE,
        ),
        re.compile(r"^\s*Here\s+is\s+my\s+translation:\s*", re.IGNORECASE),
        re.compile(
            r"^\s*(?:这是|以下是)?(?:最终|精修|精练|润色)*(?:中文)?翻译(?:如下)?[：:]\s*",
            re.IGNORECASE,
        ),
        re.compile(r"^\s*(?:###|##)?\s*Translation:\s*", re.IGNORECASE),
        re.compile(r"^\s*(?:###|##)?\s*Polished\s+Translation:\s*", re.IGNORECASE),
        re.compile(r"^\s*要翻译的(?:原文|段落|内容)[：:]\s*", re.IGNORECASE),
    ]

    @classmethod
    def _clean_reasoning(cls, raw_text: str) -> str:
        """Strip complete or unclosed reasoning/thought blocks."""
        text = cls._REASONING_BLOCK_PATTERN.sub("", raw_text)
        if cls._UNCLOSED_REASONING_PATTERN.search(text):
            text = cls._UNCLOSED_REASONING_PATTERN.sub("", text)
        return text

    @classmethod
    def strip_reasoning(cls, raw_text: str) -> str:
        """De-noise at transport level only: reasoning traces out, nothing else.

        Used by the provider for models whose extraction strategy (XML_TAG) is
        owned by a strategy-aware caller: an AUTO unwrap here would consume the
        <translation> boundaries before the real extraction pass sees them.
        """
        return cls._clean_reasoning(raw_text).strip()

    @classmethod
    def _extract_xml_tag(cls, text: str) -> str | None:
        """Extract content inside <final_translation> or <translation> tag.

        Tries the closed wrapper first (greedy to the last close tag) and only
        then the start-anchored unclosed form, so a literal tag mention inside
        the translation cannot truncate it.
        """
        for closed, opened in (
            (cls._FINAL_TAG_PATTERN, cls._FINAL_TAG_OPEN_PATTERN),
            (cls._TAG_PATTERN, cls._TAG_OPEN_PATTERN),
        ):
            for pattern in (closed, opened):
                match = pattern.search(text)
                if not match:
                    continue
                extracted = match.group(1).strip()
                if not extracted or extracted in ("...", "…"):
                    continue
                fence_match = cls._CODE_FENCE_PATTERN.match(extracted)
                if fence_match:
                    extracted = fence_match.group(1).strip()
                return extracted
        return None

    @classmethod
    def _clean_conversational_prefix(cls, text: str) -> str:
        """Strip markdown fences and conversational lead-in phrases."""
        cleaned = text.strip()
        fence_match = cls._CODE_FENCE_PATTERN.match(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        for pat in cls._CONVERSATIONAL_PREFIX_PATTERNS:
            cleaned = pat.sub("", cleaned).strip()

        # Clean orphan translation tag remnants at the edges only. Mid-text
        # '<translation></translation>' is a mention in real prose (a book about
        # prompting), and removing it silently deletes the author's words.
        cleaned = re.sub(
            r"^(?:\s*</?(?:translation|final_translation)[^>]*>)+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"(?:</?(?:translation|final_translation)[^>]*>\s*)+$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        return cleaned

    @classmethod
    def extract(
        cls,
        raw_text: str,
        strategy: ExtractionStrategy | str | None = None,
    ) -> str:
        """Extract pristine target translation from raw LLM output using chosen strategy.

        Sanitization is deliberately NOT applied here. Repair-loop protocol tags
        (``<correction id="...">``) and HTML-preserving flows pass through
        extraction and would be destroyed by an allowlist at this layer.
        Untrusted LLM output is instead sanitized at the artifact boundaries
        where stored XSS could actually materialize: EPUB DOM injection,
        Markdown rendering, and Typst escaping.
        """
        if not raw_text or not isinstance(raw_text, str):
            return ""

        # Resolve extraction strategy from string or default to AUTO.
        if strategy is None:
            resolved_strategy = ExtractionStrategy.AUTO
        elif isinstance(strategy, str):
            try:
                resolved_strategy = ExtractionStrategy(strategy.lower())
            except ValueError:
                resolved_strategy = ExtractionStrategy.AUTO
        else:
            resolved_strategy = strategy

        if resolved_strategy == ExtractionStrategy.RAW:
            return cls._clean_reasoning(raw_text).strip()

        cleaned_text = cls._clean_reasoning(raw_text)

        if resolved_strategy == ExtractionStrategy.XML_TAG:
            tag_content = cls._extract_xml_tag(cleaned_text)
            return tag_content or ""

        if resolved_strategy == ExtractionStrategy.CONVERSATIONAL_PREFIX:
            return cls._clean_conversational_prefix(cleaned_text)

        # ExtractionStrategy.AUTO: try XML tag first; if not found, clean conversational prefix
        tag_content = cls._extract_xml_tag(cleaned_text)
        if tag_content:
            return tag_content
        return cls._clean_conversational_prefix(cleaned_text)

    @classmethod
    def extract_macro_blocks(cls, raw_text: str) -> dict[str, str]:
        """Extract mapping of block_id -> translated_text from macro-chunk outputs.

        Handles reasoning trace stripping, markdown code fences, and whitespace trimming.

        Undoes the envelope escaping ``build_macro_chunk_draft_prompt`` applies
        to block ids and text: the model is shown ``AT&amp;T`` and mirrors it
        back, so without the inverse a source ``&``/``<``/``>`` is stored as an
        entity and rendered literally downstream.
        """
        cleaned = cls._clean_reasoning(raw_text).strip()
        fence_match = cls._CODE_FENCE_PATTERN.match(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        results: dict[str, str] = {}
        for match in cls._MACRO_BLOCK_PATTERN.finditer(cleaned):
            bid = html.unescape(match.group(1).strip())
            content = html.unescape(match.group(2).strip())
            if bid and content:
                results[bid] = content
        return results
