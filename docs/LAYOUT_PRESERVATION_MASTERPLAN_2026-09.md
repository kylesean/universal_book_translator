# 版面保持 / 出版级翻译 — 总体方案 (LAYOUT PRESERVATION MASTERPLAN)

日期：2026-09-20 ｜ 状态：活文档 ｜ 目标：长短文档、主流格式，通用 / 工业 / 出版级翻译

## 0. 一句话现状

UBT 的路线 A（现名 **`rigid`**：保留源页几何、就地把译文重绘回原 bbox、非文字逐像素不变）已有完整通路，本次补齐了三块地基：**命名消歧、许可/隔离守卫恢复、像素级 fidelity 度量（advisory）**。下一步是把 `rigid` 的"兜住率"从 ~42% 提到出版级，再增量补路线 B（栅格 inpaint 兜底）。

## 1. 两条路线的定义（命名标准）

| 引擎值 | 含义 | 非文字内容 | 可选中文本 |
|---|---|---|---|
| **`rigid`** | 源页为画布，抹译文区文字、按原 bbox 自适应缩字号重绘（pikepdf `add_overlay` 叠加机制） | 逐像素不变（本方案要度量它） | ✅ |
| `publication`（别名 `reflow`） | 从 IR 用 Typst 从零重排 | 会重建 | ✅ |
| `auto` | 按密度路由：公式/结构密集→`rigid`，正文→`publication` | — | — |
| 未来 `raster`（路线 B） | 页面栅格化 + 文字框 inpaint 擦背景 + 贴回译文 | 复杂背景压字场景 | ❌（需例外） |

**命名标准（2026-09-20 定案，零向后兼容）**：路线 A 唯一名 `rigid`；旧 `anchored`/`overlay` 两个引擎名已根除（`RenderEngine` Literal 不再含之，传入即 ValidationError）。`overlay` 一词回退为"仅指 pikepdf 叠加机制"，故 `overlay_text.py` / `page.add_overlay` / `_page_overlay` **保留不改名**。历史别名 `inplace→rigid` 仍可被 pydantic 归一（既有退役机制，非新兼容）。

## 2. 信息论天花板（为何"1:1"有上限）

译文长度≠原文长度；原文本是按原文字数量身定做的几何盒子。因此"逐像素等于原页"在**被翻译文字上数学不可能**。可达目标 = **非文字区逐像素不变 + 译文落回同 bbox 同基线，仅自适应字号/换行微调**。这是 `rigid` 与 BabelDOC/PDFMathTranslate 共同的工业上限。

## 3. 本次已落地（三地基）

### 3.1 许可 / 架构隔离守卫（恢复 + 加强）
- 恢复 `tests/unit/test_license_guard.py`（4 项 AST：全仓禁 `fitz`/`pymupdf`；`ubt/core/` 禁 `babeldoc`/`pdf2zh`/`docling`/`pypdf`/`fitz`/`pymupdf`；`pyproject` 无 AGPL；core 只能经 `ports.py` 触达 adapters）与 `tests/unit/test_core_ports_isolation.py`（运行时 + 惰性解析）。这两者被 commit `0492c7d` 删除，正是"没抄 BabelDOC"的可执行证据。
- 新增两项前瞻守卫：route-B 重图像依赖禁止进 base dependencies（`opencv`/`torch`/`scikit-image`/`lama`/`diffusers` 只能待在 optional extra）；god-function 尺寸棘轮（`pipeline.py`、`typst_reconstructor.py`，只降不升；`pipeline.py` 棘轮因 09-21/22 的硬取消/渲染保真等修复被有意识地上调过数次，**当前帽值以 `tests/unit/test_license_guard.py::_SIZE_RATCHETS` 为准**，此文档不复述易漂移的具体数字）。
- 恢复守卫立刻抓出两处删守卫期间的漂移违规并**真修**：`core/assess.py`、`core/archetype.py` 曾直连 adapters、直 `import pypdf`。已把 `inspect_pdf_route_plan`/`profile_pdf_pages`/`page_kind_enum`/`probe_min_chars`/`supported_suffixes`/`sample_pdf_pages` 全改为经 `ports.py` 惰性桥接，pypdf 采样下沉到 `adapters/pdf/plain_text_extractor.py`。顺带修掉一处错误模块引用（`PROBE_MIN_CHARS` 实为 core 定义、`PROFILE_CACHE_DIR` 实为 `engine_selector` 定义）。

### 3.2 命名消歧 → `rigid`
`config.py`（`RenderEngine`/`canonical_render_engine`/`RIGID_ENGINES`/文档串）、`adaptive_policy.py`、`docling_render.py`、包 `anchored/`→`rigid/`（`git mv`）、`Anchored*`→`Rigid*`、`ANCHORED_*`→`RIGID_*`、CLI/API 值、测试文件与 `tests/baselines/anchored-overlay/`→`rigid/`、文档。回归新增 `test_render_engine_is_canonically_rigid_and_old_spellings_are_gone`（扫全仓 `"anchored"`/`"overlay"` 引擎值残留）。

### 3.3 像素级 fidelity 度量（advisory）
- `adapters/pdf/render_fidelity.py`：源页/译文页**同一 pdfium 引擎、300 DPI** 整页栅格化，按 `PROSE_BLOCK_TYPES` 的 bbox 建掩膜（复用 `visual_scalpel._compute_crop_coords`），**Pillow-only**（`ImageChops` + 直方图，零新依赖）比对掩膜外区域。产出 `non_text_diff_ratio`（非文字残差，应≈0）与 `masked_coverage_ratio`（涂写覆盖率，正面量化 42% 问题）。纯函数 `diff_outside_masks` 可单测。
- advisory 接线：`reflow_loop.run` 在 parity 合并后、仅当 `_output_keeps_source_geometry()` 时计算，把数值折进 `gate.stats`、以 `info` 级并入 findings，**绝不碰 `gate.passed`**，并落 `manifest.metadata["fidelity"]` → `visual_report.json`。
- 结构化 + KPI：`reporter.QualityReport.fidelity`（`ReportFidelity`）；`metrics` 新增 `fidelity_non_text_residual`（越低越好）、`rigid_painted_coverage`（越高越好）。`SCHEMA_VERSION` 暂不升（新增 key 在对比中被忽略，避免波及既有 golden 的再生成清扫）。

## 4. 尚未做（里程碑，按 ROI 排序）

- **M1 收紧 fidelity 基线**：跑 ForMaT 子集，锁定 `rigid` 逐文档 residual/coverage 基线数字（`docs` 记录的 `scripts/biou_score.py` 已有 BabelDOC 方法论级评测骨架）。
- **M2 `rigid` 闭环扩框**：解除 `reflow_loop` 对 `rigid` 的自我排除（缺陷 B2，`_typography_retune_possible`），溢出时受控"向空白 margin 借空间"而非直接 `NEEDS_HUMAN`；正面攻 42% 覆盖率。
- **M3 fit-truth 回读核对**：Python 证明容量后 Typst 仍自由断字（`clip:true` 兜底可能静默裁字）；渲染后回读实际字形 bbox 与预期比对，超出者转 M2。
- **M4 rotation 进 IR**：`BoundingBox` 无 rotation 字段、`rigid/extract.py` 硬拒旋转页 → 补字段 + CTM 重建，消灭一类硬失败。
- **M5 路线 B（raster inpaint 兜底）**：`visual_scalpel` 已有裁图（需提到 300 DPI）、`vlm` 已有 `measured_box`；缺**背景修复**（现仅 `diagram_localizer` 白矩形假填充）。接缝 = `docling_render.py` 的 `UBT_LOCALIZE_DIAGRAMS` 分支 + `ports` 对称的 `paste_block_image`/`InpaintPort`；许可安全选型（LaMa/Apache、Inpaint-NS/OpenCV，过 §3.1 前瞻守卫）；raster 区在 parity 里显式豁免。
- **M6 非 PDF 格式通用化**：`docx`/`epub`/`html`/`md` 现为纯文本 thin adapter（无 bbox/无保版），"universal"实为"PDF-first"。优先做 docx/epub **结构级原位往返**（其模型本自结构化，可近 1:1），并让它们复用同一套 `policy`/QE/witness/fidelity 工具链。
- **横切**：`pipeline.run()` / `typst_reconstructor.py` 拆分期（棘轮已围栏，逐步偿还）；route 决策写进 manifest + 空书硬失败；真实 e2e 测试补齐。

## 5. 洁净室边界（AGPL）

BabelDOC（arXiv 2605.10845）、PDFMathTranslate（arXiv 2507.03009）是 AGPL。本仓**只读论文/方法**（评测协议、IR/自适应重排/迭代缩字思路、BBox-IoU+LLM-as-judge），**绝不逐行移植其源码**；全仓对二者的引用均为 "self-implemented, Zero-AGPL" 方法论级。`test_license_guard.py` 是这条边界的自动化执行者，不得再删。竞品差异化保留项：视觉见证（`formula_witness`）、提取见证（`extraction_witness`）、账本级计费可靠性、Typst 编译反馈自愈——本次 fidelity 度量进一步把"可证伪的保真度"可见化。
