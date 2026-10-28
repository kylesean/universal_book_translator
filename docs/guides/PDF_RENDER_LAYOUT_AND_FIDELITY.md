# PDF 渲染排版与保真（语义重排）

> **状态**：🟢 活文档（随代码演进更新；代码符号与路径以 `ubt/` 源码为准）
>
> **2026-10-12 重写**：本文档原先描述 overlay（源页画布 + 译文片段合成）架构。
> overlay 语义已随语义重排重构删除（见
> [`MODE_SPACE_CONSOLIDATION_RFC.md`](MODE_SPACE_CONSOLIDATION_RFC.md) §7），
> 第 1–4 节据此重写；第 7、9–11 节讲的是**翻译质量**而非渲染，内容仍然有效。
>
> **2026-10-28 增补**：第二渲染通道“1:1 版式（facsimile）”的设计规格见
> [`../design/inplace_delivery.md`](../design/inplace_delivery.md)。它**不是**已删除的
> overlay——背景层只删 `BT…ET` 文本块，译文层由 Typst 在原坐标重排（同一
> `typst_compile`、同一 `math_transpiler`、同一 gate），不写内容流、不嵌字体、
> 不自研断行。本文档描述的重排路径仍是**默认且唯一已落地**的 PDF 路径。
>
> 本文档描述：翻译产物 PDF 如何由 IR **重建**并由 Typst 排版，以及若干
> 跨语言的格式保真问题（内联样式、引用、标题膨胀等）。
>
> 本地文档：`docs/` 在 `.gitignore` 中（`/docs/`），仅供本地查阅，不会提交。
>
> 代码为准，行号可能随迭代漂移；关键符号名稳定，可用 `codegraph explore` 重新定位。

---

## 0. TL;DR

| 关注点 | 归属层 | 状态 |
|---|---|---|
| 正文字号统一 | `document_renderer.py` 的 Typst preamble | 已实现（10.5pt） |
| 图（figure）保留 | `#figure(image(...))` | 已实现 |
| 表重建（表头随译） | `render_table` → `#table` | 已实现 |
| 公式 | `math_transpiler` → 真实数学；读不出的裁源图 | 已实现 |
| 双语对照 | `TypstDocumentRenderer(bilingual=True)` 逐块配对 | 已实现 |
| 行距 / 缩进 | Typst `#set par(...)` | 已实现 |
| 有序列表序号（`(1)`…） | 解析期 marker 恢复 + 渲染期还原 | 已实现 |
| 邮箱/URL 被翻译坏（`research@`） | 占位符引擎新增 `email` family | 已实现（§7） |
| 作者-年份引用被翻译 | `citation_masker` 保护括号引用 | 已实现（§7） |
| 引用链接点不动/丢失 | 合成以源 PDF 为基底，保留命名目标 | 已实现（§7） |
| 引用颜色丢失、`‡` 上标丢失 | 内联 run（pdfium 逐字符 + Typst） | 已实现（§7） |
| 首页书目标识区（单位/符号图例）被翻译一半 | `skip_rules` 新增 bibliographic identity | 已实现（§9） |
| 论文标题被模型续写成"标题+摘要段"导致 draft 膨胀 | 标题块 no-expand 提示 + 标题草稿首段兜底 | 已实现（§10） |
| 图内文字（坐标轴标题/图例）被当正文翻译 | `docling_blocks.is_inside_picture` 图内归属判定 | 已实现（§11） |
| **首句局部加粗（译文对齐）** | 内联 run 基建已就绪，跨语言对齐未做 | **部分，见 §5.1** |

---

## 1. 渲染模型（语义重排）

PDF 目标由 **IR 重建**，交给 Typst 排版：`ubt/render/typst/document_renderer.py`。

```
IRBlock 序列
   ├─ heading   → #heading(level: n)[…]
   ├─ table     → render_table() → #table(…)         （表头随译）
   ├─ formula   → math_transpiler → $ … $            （真实数学）
   │                读不出 LaTeX 的 → 解析期裁源图为 IMAGE 块
   ├─ image     → #figure(image(…), caption: […])
   └─ prose     → render_prose()  （转义 + 数学转译 + 内联样式）
        ↓
   一份 Typst 源 → typst compile → PDF
```

**为什么不是「源页画布 + 片段合成」**：那条路径（overlay）把每个译文钉死在源段落
框内，导致字号被迫缩到 4pt、表格留在源语言层、公式只保留源几何。重建则让正文字号
统一（实测 10.5pt）、表头随译、公式成为真实数学。对比实测（arXiv 2101.03961）：

| | overlay（已删） | publish |
| :--- | ---: | ---: |
| 字号离散度 | 44 种 | 3–5 种 |
| 表格 | 源语言层 | 重建，表头随译 |
| 公式残留率 | 0.857 | 0.000 |
| 页数 | 41 | 29 |

**双语**：`TypstDocumentRenderer(bilingual=True)` 在每个块上方渲染灰色斜体原文。
是**逐块配对**而非页面 zip —— 重排后页数已变（41→29），页面级对齐不成立。

**降级可见**：公式无法转换时降级为 Unicode 并计入 `stats.formulas_degraded`；
连 Unicode 都为空时计入 `formulas_dropped`。资产缺失渲染可见占位符并计入
`stats.assets_missing`。**没有静默丢弃路径。**

---

## 2. 内联样式（粗体 / 斜体 / 上标 / 颜色）

**来源两条**：
- `StyleMeta.inline_runs`：pdfium 逐字符探测（`textgeom.styled_runs_in_box`）；
- `StyleMeta.target_runs`：模型在译文里标的粗体标记（`ir/emphasis.parse_bold`）。

**定位**：run 的 `text` 是字面字符，在译文里按子串查找（`_locate_runs`）。译文改写
过的 span 找不到匹配，按契约降级为普通文本。目标侧 CJK 规范化（全角、pangu 空格、
破折号变体）会破坏精确匹配，因此有**折叠回退**（`_fold_for_match`）。

**渲染**：`_wrap_style` 输出 `#strong`/`#emph`/`#super`/`#text(fill: rgb(...))`，
在转义**之后**套用，所以注入的标记不会被转义。

**边界**：落在数学 span 内的 run 会被丢弃 —— 否则会把公式切成两半并转义成字面
文本（`$x_i$` → `\$#strong[x\_i]\$`）。数学归转译器所有。

---

## 3. 语言与方向

`TypstDocumentRenderer.preamble(target_lang)` 从目标语言派生 `lang` 与
`dir`。**不是硬编码 `zh`**：阿拉伯语/希伯来语目标若用 `lang: "zh"` 会得到中文断行
与从左到右排版，每一页都是错的。Typst 自己跑 UAX #9，所以 RTL 目标只需声明方向，
文本仍按逻辑序。

---

## 4. 视觉门与保真度量

视觉门（`visual_gate` + `reflow_loop`）对**所有** PDF 产物运行，与渲染方式无关。
保真度量（`render_fidelity` / `artifact_parity`）中依赖「源页是画布」的项在语义
重排下不再适用 —— 重排合法地重建页面，页数会变。

## 5. 已知限制与后续工作

### 5.1 首句局部加粗（内联粗体）— 独立专项，尚未实现

**现象**：源文里「列表项首句加粗」这类**段内局部**粗体，译文丢失（整段渲染为正
常字重）。例如：
`(1) **Rollout and evaluation jobs create sandboxes in a bursty manner.** A single job may …`

**现状（已实现）**：渲染端 `render_prose(text, runs)` 会把 run 映射到译文并输出
`#strong`/`#emph`/`#super`/`#text(fill:)`（§2）。**仍受限**的是**跨语言对齐**：源文
被翻译的那段 bold（如首句）在译文里没有对应边界，只能靠子串匹配；译文改写了这段
文字，匹配失败即降级为普通字重。

**剩余工作（第 4 步：译文格式对齐）**：
- 现有 `mask`/`masked_spans`（`ubt/segment`、`ubt/model/segment.py`、QE 的
  `placeholder.retention_rate`）是「保护需保留的 span」的通道。
- 方案 A（推荐）：把源 bold 区间作为**受保护的标记 span** 注入 prompt，要求模型保留；
  导出前校验数量/顺序，缺失则走修复轮（复用现有 repair 机制）。
- 方案 B（兜底）：无标记时按启发式对齐（如「首句」= 第一个句末标点前），把源 bold
  应用到译文对应位置。仅在方案 A 失败时启用，且必须可关闭。
- 对齐结果回填到 `Overlay.runs`（`StyledRun`），复用第 3 步渲染。

**验收**：`(1)`/`(2)` 段首句在译文中加粗；`(2)` 这类源文只加粗首句的段落不得把
整段加粗；QE 的 placeholder 保留率为 1.0；视觉门不新增 major/critical。

### 5.2 其他小限制
- **`auto` 路由把公式文档送给 pdfium**：`select_pdf_engine` 依赖
  `formula_debris_share`（孤立单字母占比）判定 `has_formulas`。公式为完整表达式
  （`f(x)`、`Ax − b`）而非碎片时全部低于阈值，文档被路由到 `PDFiumAdapter`，而它
  走 `read_pdf_blocks`，**完全不产出 formula/image 块** —— 公式从未被表示。
  实测 alphafold2 四页 share 0.035/0.094/0.092/0.088，全部低于 0.10 阈值。
  临时绕过：`--pdf-engine docling`（实测产物出现 11 个公式图）。
- **整块粗体**已支持（`provenance["is_bold"]`），但非 headless 的**整行**粗体若与
  正文同行混合，同样受 5.1 限制。

---

## 6. 源码地图

| 主题 | 位置 |
|---|---|
| 语义重排渲染器 | `ubt/render/typst/document_renderer.py` |
| 数学转译 | `ubt/render/typst/math_transpiler.py` |
| Typst 转义 | `ubt/render/typst/escape.py` |
| 表格重建 | `ubt/render/table_compiler.py` |
| 渲染策略 / 分发 | `ubt/adapters/pdf/docling_render.py` |
| 公式裁剪（读不出 LaTeX 时） | `ubt/adapters/pdf/docling_parser.py`（`_crop_formula_asset`） |
| 内联 run 探测 | `ubt/adapters/pdf/textgeom.py`（`styled_runs_in_box`） |
| 内联 run 渲染 | `ubt/render/typst/document_renderer.py`（`_locate_runs`/`_wrap_style`） |
| 首行缩进解析 | `ubt/adapters/pdf/docling_parser.py` |
| 有序序号恢复 | `ubt/adapters/pdf/docling_parser.py` |
| `StyleMeta` 字段 | `ubt/core/ir/models.py` |
| 布局诊断探针 | `ubt/adapters/pdf/pdf_profiler.py` |
| 渲染保真度量 | `ubt/adapters/pdf/render_fidelity.py` |
| 视觉门 | `ubt/adapters/pdf/visual_gate.py` |

> 行号刻意省略（迭代即漂移）；用 `codegraph explore` 或符号名定位。

---

## 7. 引用保真与内联样式（新增）

### 7.1 邮箱 / URL 保护（占位符新 family `email`）
- 新增 `ubt/core/cleaners/email_masker.py`（`EmailMasker`：邮箱 + `http(s)://`/`mailto:`）；
  token `⟦EMAIL_MASK_…⟧`，`checksumless_restores=False`（严格）。
- 引擎顺序变为 **email → code → math → soup → citation**（`ubt/segment/placeholders.py`）：
  URL 先于邮箱，避免 `mailto:user@host` 被拆成裸邮箱。
- 同步点：`MaskedSource`/`RestoreOutcome`/`PlaceholderEngine`、`draft.py` 的 `_DraftInputs`、
  `ports.masked_source`、`reporter.py` 的损坏计数与正则、`glossary_enforcer` 的保护集。
- 效果：`research@deepseek.com` 不再被翻译成 `研究@deepseek.com`。

### 7.2 作者-年份引用不翻译（`citation_masker`）
- `_PAREN_CITATION_PATTERN` 保护 `(Guo et al., 2025; Jimenez et al., 2024; OpenAI et al., 2024)`、
  `(DeepSeek-AI, 2026)` 等；门控是「作者词 + 紧跟 `, YYYY`」，所以 `(DSH)`/`(RL)`/
  `(Section 2, 2024)` 不被误掩码。
- 政策变更：原先 docstring 明确「口语式引用故意不掩码」，现改为保护（引用是标识符，
  须与参考文献表一致）。

### 7.3 引用链接可点击
- **根因**：`LayerCompositor.compose`/`compose` 原用 `pikepdf.new()` 再 append 页面，
  catalog 的 `/Names /Dests` 命名目标丢失 → GoTo 目标悬空。
- **修复**：两处合成改为**以源 PDF 为基底文档**（`pdf_struct.open_pdf`），catalog
  （命名目标/书签/元数据/AcroForm）随基底保留，页面对象不变、无需重映射。
- 另：`link_annotations.relocate_page_annotations` 重定位失败时**保留**注解而非 prune
  （宁可能点偏也不要死链）。
- 实测：第 2 页链接 **21 → 21**，`/Names /Dests` 存在，`cite.deepseek-r1` 解析到有效页。

### 7.4 内联 run：上标与颜色
- **模型**：`ubt/core/ir/models.py` 新增 `InlineRun(text,bold,italic,superscript,color_hex)`，
  `StyleMeta.inline_runs`（pydantic，经 `style_json` 自动往返）。
- **提取**：`textgeom.extract_char_styles`（逐字符 `GetCharBox/GetFontSize/GetFontWeight/
  GetFillColor`）+ `styled_runs_in_box`（按样式合并 run，上标 = 字号 < 0.8×块中位；
  空格随所在 run）。在 `docling_crosscheck` 里写入 `block.style`。
- **映射**：`outputs.overlays_from_blocks` 把源 run 包成 `StyledRun` 放进 `Overlay.runs`；
  渲染前 `_locate_runs` 在译文里**按顺序子串查找**——只有逐字保留的 span（被 §7.1/§7.2
  掩码的邮箱与引用、符号 `‡`）能命中；被翻译的文本找不到就跳过。
- **渲染**：`overlay_text.render_overlay_line` 在**转义之后**按 run 边界包裹
  `#text(fill: rgb("#RRGGBB"))[...]` / `#super(typographic:false,size:0.62em)[...]` /
  `#strong` / `#emph`（与既有 `_emit_typst_superscripts` 一致）。
- 实测：机构行 `‡` 恢复上标；引用 run 渲染为蓝色 `(0,0,255)`。

### 7.5 源码地图（新增）
| 主题 | 位置 |
|---|---|
| 邮箱/URL 掩码 | `ubt/core/cleaners/email_masker.py` |
| 引擎 family 编排 | `ubt/segment/placeholders.py` |
| 作者-年份引用掩码 | `ubt/core/cleaners/citation_masker.py` |
| 合成基底（命名目标） | `ubt/render/outputs.py` `compose`/`LayerCompositor.compose` |
| 链接保留不 prune | `ubt/adapters/pdf/link_annotations.py` |
| 逐字符样式提取 | `ubt/adapters/pdf/textgeom.py`（`extract_char_styles`/`styled_runs_in_box`） |
| 内联 run 接线 | `ubt/adapters/pdf/docling_crosscheck.py` |
| `InlineRun` 模型 | `ubt/core/ir/models.py` |
| `Overlay.runs` / `_locate_runs` | `ubt/render/outputs.py` |
| run 渲染 | `ubt/adapters/pdf/overlay_text.py`（`render_overlay_line`/`_wrap_style`） |

---

## 8. 续段统一字号 与 分析缓存失效

### 8.1 跨页续段字体忽小（"智能缩放"）
- **现象**：跨页/跨栏段落，第一页那一半字号远小于正文（如 7.5pt vs 11.5pt），极不和谐。
- **根因（两层）**：
  1. **流式拆分与绘制字号不一致**：`solve_flow` 用 `typesetter.measure`（字号 `_size_pt=9.96`）
     算断点，绘制却用 `cap=源字号×1.05≈11.46`；在 9.96 下装得下的前缀在 11.46 下溢出，
     于是逐框 fit 把它缩小。
  2. **融合续段丢多框链**：`fuse_continuation_blocks` 设了 `CompositeSpan`，但 provenance
     只写 `fused_block_ids`，没写 `physical_boxes`；ledger 往返时 `_row_to_block` 只从
     `physical_boxes` 重建 span → 融合段变回单框（首行 1 行）→ 整段挤进 1 行 → 缩到极小。
- **修复**：
  - `fuse_continuation_blocks`（`ubt/core/ir/continuation.py`）补写 `physical_boxes`。
  - `LayerCompositor._flow_plan`（`ubt/render/outputs.py`）：**整条续段用统一字号**——先按
    整段文本与整链容量算一个尺寸（装得下就用 cap，装不下就整体等比缩小），`solve_flow`
    在该尺寸下拆分，每个 part 用 `typeset_fixed` 按**同一尺寸**绘制（`exact_size`），
    不再逐框重新 fit/缩小。容量按**裸框高**（不含 slack）计，与 `solve_flow` 一致，避免溢出裁剪。
- **实测**：第 3 页"2. DSec 概述"后的正文恢复 11.5pt；全篇"中文正文明显缩小段"从 {6,19}
  降到 {9: 25 字}；中文字符总数无丢失（19474→19479）。

### 8.2 内联 run 没生效（`‡` 仍未上标）——分析缓存未失效
- **根因**：`analysis_cache.docling_identity()` 只含 `docling_parser`+`docling_blocks`；
  而内联 run 的代码在 `docling_crosscheck`/`textgeom`/`ubt.core.ir.models`。改这些模块
  不会改变缓存键 → 命中旧块（`inline_runs=[]`）。
- **修复**：`docling_identity()` 纳入 `docling_crosscheck`、`textgeom`、`ubt.core.ir.models`
  的 size+mtime。
- **实测**：机构行 `‡` 恢复上标（字号 7.69 vs 正文 ~11）。

### 8.3 源码地图（新增）
| 主题 | 位置 |
|---|---|
| 续段统一字号 | `ubt/render/outputs.py`（`LayerCompositor._flow_plan`、`_compile_form(exact_size=)`） |
| 融合续段 provenance | `ubt/core/ir/continuation.py`（`fuse_continuation_blocks`） |
| 分析缓存身份 | `ubt/adapters/pdf/analysis_cache.py`（`docling_identity`） |

---

## 9. 首页书目标识区不翻译（bibliographic identity）

### 9.1 现象：页脚"翻译一半"
第 1 页底部出现中英混排。实测（arXiv 2609.22978，job `649edc87e70b`）：

| 块 | 原文 | 修复前 | 修复后 |
|---|---|---|---|
| `b0004` 作者名单 | `Jialiang Huang †‡, Hongxuan Tang †, …` | 保留（byline 规则） | 保留 |
| `b0005` 单位/邮箱行 | `DeepSeek-AI ‡ Tsinghua University research@deepseek.com` | **翻译**（`Tsinghua University`→`清华大学`） | 保留原文 |
| `b0010` 符号图例 | `∗ Corresponding author. † DSec project developers. ‡ Tsinghua University.` | **翻译**（`通讯作者 / 项目开发者 / 清华大学`） | 保留原文 |
| `b0011` 作者小传 | `Jialiang Huang is a Ph.D. student …` | 保留（`PAGE_FOOTER`→FOOTER chrome） | 保留 |

### 9.2 根因：按 docling 标签各自为政
- docling 把图例标为 `FOOTNOTE` → `region=BODY` → verdict 规则 5 不拦 → 翻译；
- 把作者小传标为 `PAGE_FOOTER` → `region=FOOTER` → chrome 规则保留；
- 单位行被标成普通正文段落 → 翻译。

同一组 `∗/†/‡` 引用符号簇被拆成三种策略，于是底部中英混排。

### 9.3 出版/翻译规范判断
`∗ † ‡ § ¶` 是**引用符号**，必须与作者名单、单位、图例一一对应；机构名是专有名词、
邮箱不可改。按 GB/T 7714 与学术惯例，这属于**书目标识（bibliographic identity）**，
原则是**整簇保留原文或整簇本地化**，最忌半译。因此选定策略：**首页书目标识区整体
保留原文**（作者、单位行、符号图例、页脚小传），只翻译标题与正文。

### 9.4 实现
在 `ubt/core/cleaners/skip_rules.py` 新增 `_is_bibliographic_identity`，接入
`classify_skip`（在 author byline 之后、debris 之前），命中即 `skip:<reason>`、原文
直出、不打模型。判定（precision-first，三道闸）：

1. **符号闸**：含引用符号 `∗ † ‡ § ¶`（ASCII `*` 仅当独立成 token）**且**含机构词或邮箱；
   或符号数 ≥2（符号图例）。
2. **机构闸**：机构词（`University|Institute|Academy|Laboratory|College|School|Lab`）
   加邮箱。
3. **长度闸**：`_IDENTITY_MAX_WORDS = 40`——超过即视为散文（致谢段、投稿说明），
   仍走翻译。

### 9.5 边界（近似用例必须仍翻译）
- `The dataset was released by Tsinghua University.` → 无符号/邮箱 → 翻译。
- `Contact us at research@deepseek.com for the dataset.` → 有邮箱无机构 → 翻译。
- 含 `a*b` 乘法的散文 → ASCII `*` 非独立 token → 翻译。
- 标题、摘要 → 无符号/机构 → 翻译。

### 9.6 源码地图（新增）
| 主题 | 位置 |
|---|---|
| 标识区判定 | `ubt/core/cleaners/skip_rules.py`（`_is_bibliographic_identity`、`classify_skip`） |
| 接入点 | `ubt/core/engine/stages/ingest.py`（`classify_skip` 调用处，verdict 之后） |
| 单测 | `tests/unit/cleaners/test_skip_rules.py`（bibliographic identity 段） |

> 注：本修复在**解析/入库期**生效，需 `--fresh` 重新入库才会反映到产物。

## 10. 标题块 draft 膨胀（模型把摘要续写进标题）

### 10.1 现象
实测（arXiv 2609.22978，job `649edc87e70b`）`pdf_main#b0003`（`block_type=heading`，
源文 96 字符的论文标题）草稿为 322 字符：

```
DeepSeek弹性计算（DSec）：用于高效大规模智能体训练的沙盒基础设施
\n\n
DeepSeek弹性计算（DSec）是一种专为大规模智能体训练设计的沙盒基础设施。该系统通过动态资源分配…
```

即模型把标题译完之后**又续写了一段"摘要概述"**。QE 长度闸判
`Target text suspiciously inflated (ratio=3.35)`，2 轮 repair 未能去除（repair 模型
`gemma4-e4b` 同样输出了膨胀文本），最终 `needs_human` → 渲染回退保留英文原文。

### 10.2 根因
1. **提示词缺口**：`ubt/core/router/prompts.py` 的 `_short_source_hint()` 本会给"短标题"
   追加"不得扩写"指令，但它以 **60 字符**为阈值判断"短"，而该标题 96 字符，因此提示词里
   **没有**这条约束。`block_type=heading` 才是可靠信号。
2. **术语表助推**：`build_chunk_glossary_table` 会为块注入"本地命中 + 全文档 top-N"
   术语（`select_terms_for_chunk`，top_n=20）。该标题的术语表含 15 条，
   其中 8 条（`microVM / full-VM / 3FS / RL / 可抢占式GPU训练 / 多智能体训练框架 /
   有状态展开执行 / 沙盒基础设施`）与那段续写文本**逐词吻合**——小模型据此"编"出了一段摘要。
   （注：本地复现 6 次均未复现，属低概率采样事件；以上为事后取证。）

### 10.3 修复（两层）
1. **预防**：`_short_source_hint(source_text, block_type)` 对 `heading`/`title`
   放宽到 `_HEADING_MAX_CHARS = 200`，且不受"句末标点"限制——标题无论多长都拿到
   "render it as an equally short heading" 指令。`block_type` 经
   `ModelRouter.draft/build_draft_prompt`、`BatchDraftRequest`、`_DraftInputs` 一路传入，
   并进入翻译缓存键（避免 heading 与同名正文块缓存串味）。
2. **兜底**：`ubt/core/engine/stages/draft.py::trim_heading_expansion`——标题译文若出现
   源文没有的**段落分隔**（源文单行、译文含 `\n\n`），只保留首段。标题本就是一行，
   续写的解释/摘要必在首段之后。真实样本实测：`ratio=3.09` 被 FastPass 拦截的草稿，
   修剪后为正确标题且 FastPass 通过。

### 10.4 边界
- 源文本身含换行（真正的多行标题）→ 不修剪。
- 单段译文（正常情况）→ 原样返回。
- 非 `heading` 块 → 不触发（正文段落换行有意义）。
- 宏块路径（`macro_chunk_size > 1`）的 `build_macro_chunk_draft_prompt` 暂不带 no-expand 指令，
  但其草稿同样经 `finalize_draft` 的兜底修剪，症状仍被兜住。

### 10.5 源码地图（新增）
| 主题 | 位置 |
|---|---|
| 标题 no-expand 提示 | `ubt/core/router/prompts.py`（`_short_source_hint`、三个 draft builder） |
| block_type 贯穿 | `ubt/core/router/router.py`（`build_draft_prompt`/`draft`/`BatchDraftRequest`/`draft_batch`）、`ubt/core/engine/stages/draft.py`（`_DraftInputs`、`prompt_context`） |
| 标题草稿首段兜底 | `ubt/core/engine/stages/draft.py`（`trim_heading_expansion`、`finalize_draft`） |
| 单测 | `tests/unit/test_draft_prompt_short_source.py`、`tests/unit/test_heading_draft_trim.py` |

---

## 11. 图内文字不翻译（figure text 归属判定）

### 11.1 现象
第 9 页 Figure 2 的 x 轴标题 `Sandbox count per task` 被当正文翻译成"每任务沙盒数量"，
以源字号的 15pt 画进它那约 100pt 宽的框里、折成两行并压到刻度标签上；而同一张图里的
`CDF`、`p50 = 352`、`Container`、`microVM` 仍留在被保留的源图里（英文）——**一张图被
翻成一半中文一半英文，且排版被破坏**。

### 11.2 根因
`is_inside_picture()`（`ubt/adapters/pdf/docling_blocks.py`）原先要求文本块**完全落在**
picture bbox 内才判为图内文字。而图自己的文字恰恰是**压在图上**的：x 轴标题的基线比
绘图区 bbox 下沿还低几个点（`Sandbox count per task`：块 y=207.6–218.2，picture
y=211.1–355.2 → 仅 67% 在框内），旋转的 y 轴标题则会跨在边缘上。于是它被判成普通正文块：
翻译 → 覆盖 → 破坏图。

### 11.3 规范（本项目的判定规则）
图 = L0 画布上的**整体资产**（`_overlayable` 明确排除 `IMAGE`/`TABLE`：合成器无法重绘
图形，所以图表整块保留）。因此：

| 图里的东西 | 处理 | 理由 |
|---|---|---|
| 坐标轴标题 / 图例 / 刻度 / 图内注释 | **不翻译**（随图保留原文） | 与图的其余文字同源；覆盖式排版无法在光栅图里对位；翻一半比不翻更差 |
| 图题 caption（`Figure N | …`，图**下方**） | **翻译** | 它是描述图的散文，且在 picture bbox 之外，不受本规则影响 |
| 脚注 | 翻译（即使被 bbox 包住） | 调用点显式排除 `FOOTNOTE` 与 caption 候选 |
| 表格单元格 | 翻译 | 表格不在 verbatim 集合里（见 `layout_policy`） |

判定规则（`is_inside_picture`）：文本项**满足任一**即属于该图 →
1. 完全落在 picture bbox 内（±2pt 容差）；或
2. 自身面积有 ≥50% 与 picture 相交（`_PICTURE_LABEL_SHARE`）；或
3. 自身中心点落在 picture 内。

图题在图**下方**、与 picture 零相交，因此永远不会命中这三条 → caption 始终可翻译。

### 11.4 边界
- 完全放在 picture 框**外**、且与框零相交的旋转轴标题（例如紧贴左边缘外侧的 y 轴标题）
  在本文档语料中并未作为文本项抽出，故未加入"邻接窄高块"启发式——几何上无法与紧邻图的
  正文区分，宁可漏判也不误吞正文。
- 若将来要真正本地化图（重绘图形），那是**图资产生成**功能，不是逐标签 overlay。

### 11.5 源码地图（新增）
| 主题 | 位置 |
|---|---|
| 图内文字判定（三条规则） | `ubt/adapters/pdf/docling_blocks.py`（`is_inside_picture`、`_overlap_share`、`_PICTURE_LABEL_SHARE`） |
| 判定调用点 + 丢弃日志 | `ubt/adapters/pdf/docling_parser.py`（`map_iterated_items`，`Dropping ... as picture text`） |
| 单测 | `tests/unit/adapters/test_docling_picture_text.py` |
