"""User-visible labels for target-language companion artifacts.

This is a *resource table*, not a translation engine: appendix titles, count
labels and column headers are structural text the pipeline writes verbatim.
Keeping them here means a renderer never has to decide which language's label
to inline, and adding a target language is one mapping entry rather than a
search-and-replace across the export stage.

Keys are BCP-47 language prefixes. Unknown targets fall back to English, never
to Chinese: an English companion is honest for a language the project has not
calibrated yet, while Chinese text in an Arabic or Spanish deliverable is not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppendixLabels:
    """Structural labels for the table and spill appendices."""

    table_title: str
    table_count: str
    table_intro: str
    table_page: str
    spill_title: str
    spill_count: str
    spill_intro: str
    spill_headers: tuple[str, str, str, str, str]


_ZH = AppendixLabels(
    table_title="# 核心图表与数据矩阵对照附录 (Translated Tables Appendix)",
    table_count="收录图表数",
    table_intro=(
        "以下为本次交付中判定为复杂版式的图表完整译文（正文已内联重排；"
        "本附录供对照核校与复制取用）："
    ),
    table_page="## 原书第 {page} 页图表 (`{label}`)",
    spill_title="# 排版溢出优雅降级对照附录 (Layout Spill Appendix)",
    spill_count="溢出段落数",
    spill_intro="以下段落在最小字号收缩策略下仍超出原始版面物理边界，已安全降级并在此提供完整双语对照：",
    spill_headers=("段落 ID", "页码", "原因", "原文", "译文"),
)

_JA = AppendixLabels(
    table_title="# 主要図表とデータマトリックス対照付録 (Translated Tables Appendix)",
    table_count="収録図表数",
    table_intro=(
        "以下は今回の納品で複雑なレイアウトと判定された図表の完全な訳文です"
        "（本文にはインラインで再配置済み。本付録は照合・コピー用です）："
    ),
    table_page="## 原書 {page} ページの図表 (`{label}`)",
    spill_title="# レイアウト溢れの安全な縮退付録 (Layout Spill Appendix)",
    spill_count="溢れ段落数",
    spill_intro="以下の段落は最小フォント縮小後も元の版面の物理境界を超えたため、安全に縮退し、ここに完全な対訳を提供します：",
    spill_headers=("段落 ID", "ページ", "理由", "原文", "訳文"),
)

_KO = AppendixLabels(
    table_title="# 핵심 도표 및 데이터 매트릭스 대조 부록 (Translated Tables Appendix)",
    table_count="수록 도표 수",
    table_intro=(
        "다음은 이번 산출물에서 복잡한 레이아웃으로 판정된 도표의 완전한 번역입니다"
        " (본문에는 이미 인라인으로 재배치되었으며, 이 부록은 검토와 복사용입니다):"
    ),
    table_page="## 원서 {page} 페이지 도표 (`{label}`)",
    spill_title="# 레이아웃 오버플로 안전 저하 부록 (Layout Spill Appendix)",
    spill_count="오버플로 문단 수",
    spill_intro="다음 문단은 최소 글꼴 축소 후에도 원본 레이아웃의 물리적 경계를 초과하여 안전하게 저하되었으며, 아래에 전체 대역을 제공합니다:",
    spill_headers=("문단 ID", "페이지", "이유", "원문", "번역"),
)

_EN = AppendixLabels(
    table_title="# Translated Tables Appendix",
    table_count="Tables",
    table_intro=(
        "The following complex tables were translated in full. The body re-typesets "
        "them inline; this appendix is for review and copy/paste."
    ),
    table_page="## Source page {page} table (`{label}`)",
    spill_title="# Layout Spill Appendix",
    spill_count="Overflow paragraphs",
    spill_intro=(
        "These paragraphs exceeded their source page bounds even after the "
        "minimum-font fallback; they were safely degraded and are reproduced in full below."
    ),
    spill_headers=("Block ID", "Page", "Reason", "Source", "Target"),
)

_LABELS: dict[str, AppendixLabels] = {
    "zh": _ZH,
    "ja": _JA,
    "ko": _KO,
    "en": _EN,
}


def appendix_labels(target_lang: str) -> AppendixLabels:
    """Labels for ``target_lang``; unknown targets get English, not Chinese."""
    code = (target_lang or "en").strip().lower().replace("_", "-")
    return _LABELS.get(code.split("-", 1)[0], _EN)


__all__ = ["AppendixLabels", "appendix_labels"]
