"""Detection and classification engine for backmatter partitions in technical documents."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from ubt.core.backmatter.models import (
    BackmatterConfig,
    BackmatterKind,
    BackmatterManifest,
    BackmatterSection,
)
from ubt.core.ir.models import IRBlock

#: Appendix pattern: "附录 A: 证明", "Appendix B: Proofs", "附录 一", "Appendix"
_APPENDIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:附录\s*([A-Za-z0-9一二三四五六七八九十]*)|appendix\s*([A-Za-z0-9]*))[:：\s]*(.*)",
    re.IGNORECASE,
)

#: Bibliography pattern: "References", "Bibliography", "参考文献", "引用文献"
_BIBLIOGRAPHY_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:references|bibliography|works\s+cited)\b|(?:参考文献|引用文献|文献列表))[:：\s]*(.*)",
    re.IGNORECASE,
)

#: Glossary pattern: "Glossary", "Notation", "术语表", "词汇表", "符号表"
_GLOSSARY_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:glossary|notations?|terminology)\b|(?:术语表|词汇表|符号表))[:：\s]*(.*)",
    re.IGNORECASE,
)

#: Index pattern: "Index", "Subject Index", "索引", "主题索引"
_INDEX_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:index|subject\s+index|author\s+index)\b|(?:索引|主题索引))[:：\s]*(.*)",
    re.IGNORECASE,
)

#: Afterword pattern: "Afterword", "Epilogue", "后记", "跋", "结语", "尾声"
_AFTERWORD_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:afterword|epilogue|postscripts?)\b|(?:后记|跋|结语|尾声))[:：\s]*(.*)",
    re.IGNORECASE,
)

#: Colophon pattern: "Colophon", "版权页", "图书在版编目"
_COLOPHON_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:colophon|publication\s+data)\b|(?:版权页|图书在版编目))[:：\s]*(.*)",
    re.IGNORECASE,
)

_CHINESE_DIGIT_MAP: Final[dict[str, str]] = {
    "一": "A",
    "二": "B",
    "三": "C",
    "四": "D",
    "五": "E",
    "六": "F",
    "七": "G",
    "八": "H",
    "九": "I",
    "十": "J",
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "E",
    "6": "F",
    "7": "G",
    "8": "H",
    "9": "I",
}


def _block_text(block: IRBlock) -> str:
    # ``source_text`` is the IR's canonical projection of ``element``: it
    # already reads ``.text``/``.source``/``.markup``/``.asset_id`` by element
    # kind, so the old ``element.text`` fallback was dead code that only type-
    # checked by accident (``Element`` is a union and most arms have no
    # ``.text``).
    return (block.target_text or block.source_text or "").strip()


def _is_heading_1(block: IRBlock) -> bool:
    b_type = str(getattr(block.block_type, "value", block.block_type) or "")
    if b_type != "heading":
        return False
    el = getattr(block, "element", None)
    lvl = getattr(el, "level", None) or getattr(block, "level", None) or 1
    return lvl == 1


def classify_heading(text: str) -> tuple[BackmatterKind, str, str] | None:
    """Classify heading text into (kind, letter_indicator, clean_title), or None if not backmatter."""
    clean = text.strip()
    if not clean:
        return None

    # 1. Appendix
    m_app = _APPENDIX_RE.match(clean)
    if m_app:
        letter_raw = (m_app.group(1) or m_app.group(2) or "").strip().upper()
        title_rest = m_app.group(3).strip()
        letter = _CHINESE_DIGIT_MAP.get(letter_raw, letter_raw)
        return BackmatterKind.APPENDIX, letter, title_rest or clean

    # 2. Bibliography
    m_bib = _BIBLIOGRAPHY_RE.match(clean)
    if m_bib:
        return BackmatterKind.BIBLIOGRAPHY, "", clean

    # 3. Glossary
    m_glo = _GLOSSARY_RE.match(clean)
    if m_glo:
        return BackmatterKind.GLOSSARY, "", clean

    # 4. Index
    m_idx = _INDEX_RE.match(clean)
    if m_idx:
        return BackmatterKind.INDEX, "", clean

    # 5. Afterword
    m_aft = _AFTERWORD_RE.match(clean)
    if m_aft:
        return BackmatterKind.AFTERWORD, "", clean

    # 6. Colophon
    m_col = _COLOPHON_RE.match(clean)
    if m_col:
        return BackmatterKind.COLOPHON, "", clean

    return None


def detect_backmatter_manifest(
    blocks: Sequence[IRBlock], config: BackmatterConfig | None = None
) -> BackmatterManifest:
    """Scan document blocks to detect backmatter partitions, appendices, and sequential letters."""
    _ = config or BackmatterConfig()
    manifest = BackmatterManifest()

    current_section: BackmatterSection | None = None
    appendix_counter = 0

    for block in blocks:
        if _is_heading_1(block):
            text = _block_text(block)
            cls_result = classify_heading(text)

            if cls_result:
                kind, letter_cand, clean_title = cls_result

                letter_index = ""
                if kind == BackmatterKind.APPENDIX:
                    appendix_counter += 1
                    # If letter_cand is a valid single letter, use it; otherwise sequential A, B, C...
                    if letter_cand and len(letter_cand) == 1 and letter_cand.isalpha():
                        letter_index = letter_cand
                    else:
                        letter_index = chr(ord("A") + appendix_counter - 1)

                    sec_id = f"bm-app-{letter_index.lower()}"
                else:
                    sec_id = f"bm-{kind.value}"

                current_section = BackmatterSection(
                    section_id=sec_id,
                    kind=kind,
                    raw_title=text,
                    clean_title=clean_title,
                    letter_index=letter_index,
                    heading_block_id=block.id,
                    block_ids=[block.id],
                )
                manifest.sections.append(current_section)

                if manifest.first_backmatter_block_id is None:
                    manifest.first_backmatter_block_id = block.id

                if kind == BackmatterKind.APPENDIX and manifest.first_appendix_block_id is None:
                    manifest.first_appendix_block_id = block.id
                continue
            else:
                # Normal mainmatter chapter heading resets the current backmatter section
                current_section = None

        if current_section is not None:
            current_section.block_ids.append(block.id)

    return manifest
