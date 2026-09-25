# cognitive-psychology-ch03

Academic textbook chapter baseline for Universal Book Translator.

- File: `cognitive_psychology_ch03.md`
- Source: Based on Sternberg & Sternberg *Cognitive Psychology*, Chapter 3 structure.
- Why this baseline: Multi-column textbook typography with isolated sidebars (`flow_id="sidebar_aside"`), academic tables, mathematical equations ($$ d' $$), executable Python code blocks, and academic citations/footnotes.

## Testing Purpose
1. Verify 4-layer defense against flow pollution: Sidebars must never contaminate `main_story` neighbor context window.
2. Verify code block & math formula masking: Equations and Python functions must have `skip_translate=1` and be preserved byte-for-byte.
3. Verify scientific terminology consistency via Translation Bible (e.g. "Working Memory" -> "工作记忆", "Phonological Loop" -> "语音回路", "Visuospatial Sketchpad" -> "视空间画板", "Central Executive" -> "中央执行系统").
4. Verify table Markdown formatting and column preservation.
