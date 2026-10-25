# UBT 模式空间最终设计（RFC-2026-03）

> **状态**：`IMPLEMENTED`（§9/§10 实施日志：`dual_mode` 已收敛为 `mono|dual|auto`，`UBT_RENDER_MODE` 与 `render_mode` 已从 config 删除；本文其余章节是设计依据与实测记录，行号/符号为当时快照）
> **前置**：`SEMANTIC_FIRST_REARCHITECTURE_RFC.md`（RFC-2026-02）已完成 Sprint 1–4e
> **目的**：把 16 个"模式"字段收敛为 **2 个用户可见的选择**，其余全部派生

---

## 0. 为什么现在必须做这件事

RFC-2026-02 完成了"引擎"的收敛（语义重排渲染器产品化），但没有收敛"**用户界面上的选择**"。
结果是：引擎只有一条主干，但用户仍然面对 16 个互相纠缠的模式字段。

这不是文档问题，是**设计问题**。下面的实测数据说明纠缠有多深。

### 0.1 实测：16 个 Literal 型字段

```
render_mode          ['overlay', 'publish']                       ← 引擎
dual_mode            ['inline', 'alternating', 'monolingual', 'facing', 'auto']  ← 语言
table_render_mode    ['auto', 'overlay', 'appendix', 'skip']       ← 表格
formula_mode         ['strict', 'readable']                        ← 公式
formula_render       ['native', 'image', 'witness']                ← 公式
formula_enrichment   ['auto', 'on', 'off']                         ← 公式
math_backend         ['typst', 'mathjax', 'image']                 ← 公式（已废弃）
exec_mode            ['auto', 'short', 'long']                     ← 引擎内部
prompt_strategy      ['auto', 'minimal', 'hybrid', 'rich']         ← 引擎内部
qe_engine            ['heuristic', 'comet', 'cometkiwi', 'neural', 'subprocess', 'tiered']
cover_mode           ['auto', 'always', 'never']                   ← 排版
consistency_enforce  ['off', 'report', 'repair']                   ← 质量
ocr_mode             ['auto', 'sidecar', 'cloud', 'vlm', 'rapidocr', 'off']  ← 摄取
api_mode / docling_device / job_mode                               ← 运维
```

### 0.2 实测：三个轴被埋在一起

| 轴 | 用户该选吗 | 字段 |
| :--- | :--- | :--- |
| **A. 交付物形态**（要什么） | ✅ 是 | `dual_mode` |
| **B. 产出引擎**（怎么产） | ❌ 否（应只有一个） | `render_mode` |
| **C. 质量/成本**（花多少力气） | ✅ 是 | `preset` |
| 格式细节 | ❌ 应随引擎自动 | `table_render_mode`, `formula_*`, `math_backend` |
| 引擎内部 | ❌ 应隐藏 | `exec_mode`, `prompt_strategy`, `qe_engine`, `pdf_engine` |
| 环境/运维 | ❌ 环境变量 | `api_mode`, `docling_device`, `ocr_mode`, `job_mode`, `consistency_enforce`, `cover_mode` |

### 0.3 实测：影子枚举（三个名字指同一件事）

| 层 | 类型 | 值 |
| :--- | :--- | :--- |
| 用户配置 | `DualMode` | `inline` `alternating` `monolingual` `facing` `auto` |
| 顾问层映射 | `RENDER_MODE_VALUE` | 缺 `auto` 键（`advisory.py:108` 兜底当 `inline`） |
| 适配器层 | `bilingual_mode: str` | 默认 `"bilingual"` ← **不是 `DualMode` 的合法值** |

（用户配置层→顾问层的映射在 `ubt/core/policy/bilingual_advisor.py`；适配器层的
影子枚举随 overlay 删除一并消失，见 §7。）

同一个概念在三个层有三个名字，其中 `facing` → `facing_spread` → `use_facing` 还改了两次名。

### 0.4 实测：组合矩阵（`gan.pdf`，源 9 页 125 块）

| `render_mode` | `dual_mode` | 页数 | `last_outcome` | 判定 |
| :--- | :--- | ---: | :--- | :--- |
| overlay | monolingual | 9 | ✅ | OK |
| overlay | inline | 9 | ✅ | OK |
| overlay | facing | **19** | ✅ | **真双语** |
| overlay | alternating | **18** | ✅ | OK |
| publish | monolingual | 9 | ❌ | OK |
| publish | inline | 9 | ❌ | OK |
| publish | **facing** | **9** | ❌ | ⚠️ **静默降级为单语** |
| publish | alternating | 9 | ❌ | OK |

**`publish` + `facing` 静默产出单语，用户拿不到任何警告。** 这是本次梳理发现的最严重缺陷。

### 0.5 实测：publish 忽略的旋钮

| 旋钮 | `TypstDocumentRenderer` 引用次数 | 说明 |
| :--- | ---: | :--- |
| `table_render_mode` | **0** | 确实无效，且无警告（§0.6） |
| `formula_render` | **0** | 但它在**摄取期**生效（`docling_parser.py:161`），不是渲染期 |

> **修正（2026-10-11）**：初稿把 `formula_render` 列为"用户设了无效"，不准确。
> 实测它只在 `docling_parser.py:161` 被读，**且只处理 `"image"` 一个值**：
>
> ```python
> if formula_render == "image":   # 仅此一个分支
>     ...
> ```
>
> `config.py:768` 对 `'witness'` 承诺"**逐条校验显示公式与源像素，结构不匹配时回退源图**"，
> 但全库搜索 `structural.*mismatch` / `fallback.*source.*graphic` **无实现**。
> `native` 与 `witness` 走同一条路径。
>
> 也就是说：**`formula_render` 的三个值里，只有 `image` 与其余两个有区别；
> `witness` 与 `native` 行为完全相同，且默认值 `witness` 描述的机制不存在。**
> 这是一个**文档与实现不符**的缺陷，应在收敛时一并修正。

### 0.6 实测：`table_appendix` 不受 `render_mode` 控制

`export.py:1479` 只检查 `table_render_mode`（默认 `auto`），**不检查 `render_mode`**。
`publish` 已把表格内联重排，复杂表**还会再追加一份附录** = 重复交付。

---

## 1. 最终设计：两个轴，一个默认

### 1.1 用户只面对两个选择

```
                     ┌──────────────────────────────┐
                     │  ① 要什么？(deliverable)     │
                     │     单语  /  双语对照         │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  ② 花多少力气？(effort)      │
                     │     快  /  标准  /  出版级    │
                     └──────────────────────────────┘
```

**其他一切派生。** 没有第三个旋钮。

### 1.2 轴 A：交付物形态（`dual_mode`，收敛为 3 值）

| 新值 | 含义 | 取代 |
| :--- | :--- | :--- |
| `mono` | 仅译文 | `monolingual` |
| `dual` | 双语对照（默认对页） | `facing` + `alternating` + `inline` 合并 |
| `auto` | 按 profile 推荐 | `auto`（保留） |

**`inline`（灰原文同框）取消**：它是 overlay 的专属技巧，语义重排下没有意义——
重排的整篇版面已经变了，同框对照反而更难读。双语 = 对页。

> **决策点 1**：双语到底保留几种版式？
> - **建议**：只保留**对页**（`facing`），砍掉 `alternating` 与 `inline`。
>   理由：三种版式做三套测试、三套模板，用户实际只用一种。

### 1.3 轴 B：质量/成本（`preset`，保持 4 值）

| preset | 用途 |
| :--- | :--- |
| `preview` | 快速试读，公式走图片 |
| `standard` | 默认 |
| `high_fidelity` | 高保真，最强提示词 |
| `fast` | 极限省钱，短链 |

**`preset` 现在只设 4 个字段**（实测：`prompt_strategy`/`exec_mode`/`formula_*`）。
应扩展为**设满所有派生字段**，这样用户选一个 preset 就等于选定了全部内部旋钮。

### 1.4 轴 C（新增，隐藏）：`render_mode` 退化为内部概念

**`render_mode` 不再是用户选项。** 引擎只有一条主干：语义重排。

| 旧 | 新 |
| :--- | :--- |
| `--render-mode overlay` | 保留为**内部调试开关**（`UBT_RENDER_MODE=overlay`），CLI 隐藏 |
| `--render-mode publish` | **唯一交付路径**，无需选择 |

---

## 2. 微观：16 个字段各自的下场

| 字段 | 处理 | 理由 |
| :--- | :--- | :--- |
| `render_mode` | **降级为内部/调试** | 引擎应唯一；`overlay` 作对照工具保留 |
| `dual_mode` | **收敛为 `mono`/`dual`/`auto`** | 三种双语版式砍到一种 |
| `table_render_mode` | **删除**（并入 preset） | publish 忽略它（实测 0 引用） |
| `formula_mode` | **删除**（并入 preset） | 短链专属细节，用户不该选 |
| `formula_render` | **删除**（并入 preset） | publish 忽略它（实测 0 引用） |
| `formula_enrichment` | **删除**（并入 preset） | 摄取期细节 |
| `math_backend` | **删除** | 已废弃，仅存于 job 签名；签名改用显式版本号 |
| `exec_mode` | **隐藏**（并入 preset） | 引擎内部路由 |
| `prompt_strategy` | **隐藏**（并入 preset） | 引擎内部 |
| `qe_engine` | **隐藏**（并入 preset） | 引擎内部 |
| `cover_mode` | **保留**（排版相关，用户可能在意） | 低频但语义清晰 |
| `consistency_enforce` | **隐藏**（环境变量） | 质量门策略，运维级 |
| `ocr_mode` | **隐藏**（环境变量） | 摄取策略，运维级 |
| `api_mode` / `docling_device` / `job_mode` | **隐藏**（环境变量） | 纯运维 |

**净结果**：用户可见的模式旋钮从 **16 → 2**（`dual_mode` + `preset`）+ 1 个低频 `cover_mode`。

---

## 3. 必须先修的两个真 bug

收敛之前，有两个实测缺陷会让"双模式"这个借口站不住脚：

### Bug 1：`publish` + `dual` 静默降级（§0.4）

```
实测：publish + facing → 9 页（源 9 页），last_outcome=None
     用户以为拿到了双语，实际是单语
```

**修法**：在 `_render_publish` 里检查 `bilingual_mode`，若为双语则**抛错**
（`DocumentParseError`），而不是静默产出单语。这与 `publish` 编译失败的处理一致
（RFC-2026-02 §㉑：失败不产空文件）。

### Bug 2：`publish` 下表格重复交付（§0.6）

```
实测：TypstDocumentRenderer 内联重排表格
      export.py:1479 仍追加附录（只查 table_render_mode）
      → 复杂表出现两次
```

**修法**：`_write_table_appendix` 增加 `render_mode != 'publish'` 条件。

---

## 4. 收敛路线图（每步可独立回滚）

| 步 | 动作 | 风险 | 可逆 |
| :--- | :--- | :--- | :--- |
| **1** | 修 Bug 1 + Bug 2 | 低 | ✅ |
| **2** | `publish` 变默认（`render_mode` 默认翻转） | 低 | ✅ |
| **3** | `dual_mode` 收敛为 `mono`/`dual`/`auto` | 中 | ✅ |
| **4** | 把 `dual` 移植到 publish（`source_text` 已可得，§5.1） | 中 | ✅ |
| **5** | 页面预览移植到 publish | 中 | ✅ |
| **6** | 删除 overlay 专属代码（**约 2,400–3,400 行**，需先解耦 `spi.py`，§5.2/5.4） | 高 | ⚠️ git |
| **7** | `preset` 吸收派生字段；删除 6 个模式字段 | 低 | ✅ |

> **第 1–2 步即可交付**，且不破坏任何现有能力。第 4–5 步是删除 overlay 的前提。

---

## 5. 附：实测数据

### 5.1 publish 已具备双语的前提

`TypstDocumentRenderer._block_text` 已能同时读到 `block.source_text` 与
`block.target_text`（`ubt/render/typst/document_renderer.py:188`）。**双语不是架构缺失，是未实现。**

### 5.2 PDF 适配器 14,675 行的真实构成

**诚实说明**：本节的"可删"数字经过两次修正，因为静态依赖图会误导。

静态 import 闭包显示 `publish` 依赖全部 44 个 PDF 模块——**这是假象**：
`ubt/adapters/spi.py` 是服务注册表，它在**函数体内**惰性 import 每个 overlay 模块
（`artifact_parity`、`render_fidelity`、`svg_diagram`、`extraction_witness`…），
所以静态图全连通，但运行时 `publish` 并不调用它们。

**运行时实测**（`/tmp/trace2.py`，记录真实 `__import__`）：

| 阶段 | publish 新增模块 |
| :--- | :--- |
| 摄取 | 23 个（overlay 相同） |
| 渲染 | 11 个（overlay 为 9 个，差集见下） |

`publish` 渲染独有：`typst_symbols`、`document_renderer`、`escape`、`math_transpiler`、`table_compiler`。
`overlay` 渲染独有：`font_probe`、`link_annotations`、`render/reflow`。

**这意味着**：可删集合**不能**用静态图推断。真实删除需要：

1. 先解耦 `spi.py` 注册表（把 overlay-only 条目摘掉）；
2. 再按运行时证据逐模块删除；
3. 每删一个跑全量测试。

**保守估计**：overlay 专属约 **2,400–3,400 行**（§5.4 逐模块核实），
占全库 **2.8–3.9%**。

### 5.3 全库规模

```
生产代码 (ubt/):    85,975 行
测试代码 (tests/):  34,298 行
PDF 适配器:         14,675 行
```

RFC-2026-02 承诺"全库精简 ≥10k 行（11.8%）"——**该目标不成立**。
`2,500 行` 的 PDF 适配器目标同样不成立：**摄取链（6,977 行）是任何 reflow 管线都必需**
（Docling 解析 2,016 行 + PDFium 几何交叉校验 1,034 行 + 文本几何 966 行…）。

### 5.4 逐模块核实（overlay 专属，可删）

| 模块 | 行数 | 依赖方 | 可删？ |
| :--- | ---: | :--- | :--- |
| `render/reflow.py` | 403 | `docling_render._render_composite` | ✅ |
| `overlay_text.py` | 556 | `render/outputs.py`（仅 overlay 函数） | ⚠️ 需拆分 |
| `visual_scalpel.py` | 236 | `spi.py`, `render_fidelity`, `docling_parser` | ⚠️ 需解耦 |
| `link_annotations.py` | 329 | `docling_render`（仅 overlay 分支） | ✅ |
| `render_fidelity.py` | 322 | `spi.py` | ⚠️ 需解耦 |
| `artifact_parity.py` | 303 | `spi.py`, `render_fidelity` | ⚠️ 需解耦 |
| `extraction_witness.py` | 171 | `spi.py`, `core/ports.py` | ⚠️ 需解耦 |
| `svg_diagram.py` | 394 | `spi.py` | ⚠️ 需解耦 |
| `oxide_render.py` | 94 | `render/page_preview.py` | ✅ |
| `font_probe.py` | 486 | `cli/doctor.py`, `layout/theme.py`, `render/outputs.py` | ❌ 保留（doctor 用） |
| `render/page_preview.py` | 127 | `api/app.py` | ⚠️ 需先给 publish 实现预览 |
| `render/flow.py` | 174 | `html_view`, `epub_view` | ❌ 保留 |
| `render/page_background.py` | 105 | `visual_scalpel` | ✅ |
| `layout/theme.py` | 179 | `render/outputs`, `font_probe` | ⚠️ 部分 |

**关键修正**：`overlay_text.py` **不能整删**——它的 `split_math_spans`、
`prepare_overlay_text`、`render_overlay_line` 仍被 `render/outputs.py` 使用，
而 `render/outputs.py`（2,728 行）被 publish 间接引用（经 `html_view.py`）。
Sprint 2 只搬走了 `typstify_math`/`typst_escape`，**没有搬走全部数学逻辑**。

---

## 6. 待你决策的三个问题

| # | 问题 | 选项 | 建议 |
| :--- | :--- | :--- | :--- |
| **1** | 双语保留几种版式？ | 3 种（现状）/ 1 种（对页） | **1 种** |
| **2** | `publish` 是否变默认？ | 是 / 否 | **是**（0.1.0，无契约要保护） |
| **3** | overlay 最终下场？ | 长期支持 / 内部调试 / 删除 | **内部调试 → 最终删除** |

---

## 7. 实施日志：overlay 语义清除（2026-10-12）

**决策**：overlay 不是"保留但不用"，而是**语义已死**。所有以 overlay 为前提的设计
决策都必须重新推导，不留兼容层。

### 7.1 渲染路径收敛为单一路径

| 改动 | 结果 |
| :--- | :--- |
| `render_mode` 配置字段 | **删除**（不再是用户旋钮） |
| `DoclingRenderStrategy._render_composite` | 删除（154 行 overlay 编排） |
| `_relocate_link_annotations` / `_interleave_source_and_target` | 删除（overlay 专属） |
| `_reflow_obstacles` / `_publish_requested` | 删除 |
| `BilingualAlternator` | 删除（页面级双语只对 overlay 有意义） |
| `docling_render.py` | **450 → 208 行** |

双语改为 **per-block 配对**（publish 语义）：`TypstDocumentRenderer(bilingual=…)`
在每块上方渲染灰色原文。publish 重排后页数已变（41→29），页面级 zip 不适用。

### 7.2 删除的模块

| 模块 | 行数 | 理由 |
| :--- | ---: | :--- |
| `ubt/render/reflow.py` | 403 | 只服务 overlay 分带重排 |
| `ubt/adapters/pdf/overlay_text.py` | 556 | 生产代码零引用（数学已搬到 `math_transpiler`） |
| `ubt/render/page_background.py` | 105 | overlay 遮罩底色 |
| `ubt/render/outputs.py` | **2728 → 84** | 只保留 `Composition`/`Placement`（HTML/EPUB 视图用） |

**合计删除 ≈ 4,413 行**（含 12 个 overlay 专属测试文件）。

### 7.3 顺带修复的 4 个真 bug

| # | Bug | 影响 |
| :--- | :--- | :--- |
| **1** | `publish` + 双语**静默降级为单语** | 用户选了双语拿到单语，无警告 → **已实现 per-block 双语** |
| **2** | `publish` 下复杂表**重复交付**（内联重排 + 附录追加） | `_write_table_appendix` 现在不再追加 PDF 页 |
| **3** | 无法转换的公式**泄漏原始 LaTeX** 到页面 | 移植 `clean_math_fallback_text`，降级为 Unicode |
| **4** | **公式密集文档公式被静默丢弃** | 见 §7.4 |

### 7.4 公式丢失（本次最重要的发现）

`docling_parser.resolve_formula_enrichment` 有一个短路：

```python
if formula_heavy:
    # 理由："the overlay canvas preserves the formula-dense
    #        document's original vector formulas"
    return False  # 跳过 VLM 数学 OCR
```

**这个理由的前提是 overlay。** overlay 删除后，公式密集文档的公式既没有
源画布保留、也没有 OCR 转写，`publish` 拿到裸 `$$` 后**静默丢弃**。

实测：

| 文档 | 修复前丢弃 | 修复后 |
| :--- | ---: | :--- |
| alphafold2 | **109（100%）** | 113 裁剪图 |
| ddpm | 21（100%） | 23 裁剪图 |
| neural-ode | 44 | 45 裁剪图 |
| gan | 10 | 10 裁剪图 |

**修法**：`_crop_formula_asset()` —— 读不出 LaTeX 的公式，从源 PDF **裁剪其 bbox
区域**作为 IMAGE 块。比 VLM OCR 更好：**精确**（作者原公式，逐字形）且**瞬时**
（VLM 在 CPU 上每公式 15–60 秒）。

同时删除 `content_flags()`（因该短路而存在的孤儿）和
`apply_layout_tradeoff_advisory()`（整段文案在描述 overlay 的取舍，已不成立）。

### 7.5 未解决的问题：`auto` 路由把公式文档送给 pdfium

修复后发现更深一层问题：

```
select_pdf_engine(alphafold2.pdf) → "pdfium"
```

`PDFiumAdapter` 走 `read_pdf_blocks`，**完全不产出 formula/image 块**——只有文本。
所以即使修好了 docling 的公式裁剪，`auto` 路由下这些公式**从未被表示**。

根因是公式探测器漏判：

| 页 | `formula_debris_share` | 阈值 0.10 |
| :--- | ---: | :--- |
| p1 | 0.035 | ❌ |
| p8 | 0.094 | ❌（差一点） |
| p15 | 0.092 | ❌ |
| p24 | 0.088 | ❌ |

`formula_debris_share` 统计"孤立单字母 token 占比"，该文档公式多为完整表达式
（`f(x)`、`Ax − b`）而非碎片，所以全部低于阈值 → 判定 `has_formulas=False` →
路由到 pdfium → 公式丢失。

**验证**：`--pdf-engine docling` 强制后，产物 **11 个公式图**进入 `.typ`，
PDF 从 84 KB 增至 185 KB。

**已决策（2026-10-12）**：`auto` 永远选无损引擎（docling）。理由：

1. **探测器无法证明文档没有结构**。在参考语料 20 篇上实测：每篇都带结构，
   其中 `mamba` 有 19 张图 + 16 个表格，而**所有廉价信号全读 0**——它的图是
   矢量绘图，`resource_image_count` 看不见。5 页抽样也漏掉分布稀疏的结构
   （`llama2`：90 个结构块散在 55 页上，抽样页只占 4 个）。整页扫描只要
   0.05–0.10 秒，但对矢量图仍读不出信号。
2. **pdfium 的成本理由也失效了**。摄取在 draft（翻译）之前**完整结束**
   （`ubt/core/engine/plan.py`），不存在与 LLM 争显存的问题；docling 的
   converter 每篇文档后都被确定性清理。75 页论文 docling 4.2 秒 vs pdfium
   0.8 秒——而这篇论文随后要花几分钟翻译 811 个块，3.4 秒占比不到 1%。
3. **旧世界的权衡不成立**。overlay 时代 pdfium 漏掉表格没关系（源页画布还在，
   读者看到原文表格）；重排渲染器没有画布，同样的遗漏就是成品里的空洞。

`pdfium` 保留为**显式选项**（`--pdf-engine pdfium`、`preview`/`fast` 预设），
给已知只有正文的文档用；此时探测器仍运行并打印它看到的信号。

实测影响（20 篇参考语料）：**8/20 篇曾被误路由**——gpt3（丢 104 块）、
llama2（90）、flamingo（43）、alphafold2（113）、mamba（35）、dropout（12）、
moe-mistral（20）、attention-is-all-you-need（17）。

---

## 8. 实施日志：全库 overlay 前提清扫（2026-10-12）

用户要求"都扫一遍"。以下按"发现 → 实测证据 → 处置"记录。

### 8.1 P0：每次真实 PDF 导出都被拒绝

`reflow_loop.py` 把 `keeps_source_geometry=True` 传给 artifact parity，
而 `artifact_parity` 自己的 docstring 写明"几何检查只适用于保留源页的渲染"。
在重排下它必然触发：

```
实测（alphafold2 pp.1–3）：
  major page_count_changed   Source has 24 pages, delivered artifact 4
  major image_count_changed  source total 209, artifact total 11
```

这两个 code 在 `_ALWAYS_FAIL_CLOSED_MAJORS` 里，**无视 `blocking_enabled`**，
所以任何真实（非 mock）导出都会 `raise UBTError` 拒绝交付。

**处置**：删除几何/图片计数检查及其两个 fail-closed code；
`keeps_source_geometry` 参数整体移除。保留 `target_language_absent`
（critical）与 `target_language_sparse`（major）——它们与渲染引擎无关。

### 8.2 `render_fidelity`：整个模块在测量已不存在的画布

它比较源页与产物页的"非文字像素残差"，前提是"非文字区域应与源页逐字节相同"。
重排下页面完全重建，实测：

```
fidelity_non_text_residual: 1.0      (100% 不同——必然)
fidelity_painted_coverage:  0.0      (0% 被绘制——必然)
fidelity_in_box_ink_loss:   0.0      (0% 墨迹保留——必然)
```

**处置**：删除 `render_fidelity.py`(322) + `fidelity_baseline.py`，
以及 `reflow_loop` 的 T0.6 段（63 行）、`ports.py` 两个桥、`spi.py` 两个注册、
`reporter.ReportFidelity` 模型、`metrics/collect.py` 两个 KPI、
`metrics/schema.py` 两个 KpiDefinition。

### 8.3 公式采样被反向降级

`render_preflight._sample_key` 把 TABLE/IMAGE 排到最后，理由是
"合成器永远无法在源页之上绘制它们"。重排渲染器**会**绘制它们，
而且它们是最容易编译失败的路径。实测 20 段正文 + 1 表格 + 1 图片的文档：

```
采样类型: ['narrative']      表格被采到吗: False      图片被采到吗: False
表格排序键: (1, 1, 1)  ← drawable=1 被降级
```

**处置**：改为按"最容易崩"排序：公式 → 表格 → 标题 → 其余。
同时删除错误提示里的 `--render-engine`（该选项已不存在）。

### 8.4 `facing_spread`：一个只关掉真实检查的空开关

`--facing-spread` 在 publish 链路的唯一效果是
`visual_gate.exempt_blank_pages.add(1)`——把第 1 页从空白页检查里豁免。
但 publish **从不插入 flyleaf**（`alternator` 才是），而
`render_padding_pages` 全库**无人写入**。所以这个开关只是让一个真实的
空白页缺陷静默通过。

**处置**：`facing_spread` 从 config / RenderPlan / advisory / reflow_loop /
visual_gate / CLI / REST / MCP / web 类型中全部删除。

### 8.5 死模块（已删）

| 模块 | 行数 | 证据 |
| :--- | ---: | :--- |
| `adapters/pdf/stream_strip.py` | 1109 | 17 个公开符号中只有 `mul_matrix` 被引用（迁到 `svg_diagram`） |
| `adapters/pdf/overlay_text.py` | 556 | 早已死亡（上一轮） |
| `adapters/pdf/link_annotations.py` | 329 | 只有自己的测试引用 |
| `adapters/pdf/alternator.py` | 159 | 只有 `__init__` re-export，无人调用 |
| `render/reflow.py` | 403 | 上一轮 |
| `render/page_background.py` | — | 只有自己的测试引用 |
| `adapters/pdf/render_fidelity.py` | 322 | §8.2 |
| `scripts/fidelity_baseline.py` | — | §8.2 的独立工具 |
| `adapters/pdf/engine_selector.PageIngestPlan` + `build_page_ingest_plans` | — | `plan.page_plans` 全库无人读；`include_page_plans` 从未传 True |
| `adapters/pdf/page_profiler.PageProfile.overlay_preferred` | — | 只有注释引用 |
| `core/config.RENDER_ENGINE` | — | 常量 `"overlay"`，只有注释和一个断言引用 |
| `core/config.PAGE_BILINGUAL_MODES` | — | 页面 zip 模式集合，publish 不认 |
| `docling_parser.content_flags` | — | 公式短路的孤儿 |
| `render_preflight` 的 `overlays_from_blocks` 检查 | — | 已不存在 |

### 8.6 顺带抓出的两个真 bug

**a) 孤立 UTF-16 代理导致摄取失败**

`textgeom.extract_char_styles` 用 `FPDFText_GetUnicode` 取字符，
它返回 **UTF-16 码元**而非码点：星平面字符（数学斜体 `𝑖` U+1D456，
公式排版常见）会以一对孤立代理到达。每个代理各存一个 `CharStyle` 后，
字符串含孤立代理 → 无法 UTF-8 编码 → 持久化 style 时抛
`UnicodeEncodeError: surrogates not allowed` → **整篇文档摄取失败**。

实测 `2608.25512v1.pdf`：修复前 **840** 个带样式的 run 含孤立代理，
修复后 **0**，且 840 个星平面字符正确解码。

**b) 样式 run 吞掉紧跟的左括号**

样式 run 以 `]` 结尾（`#strong[...]`），Typst 会把紧跟的 `(` 解析为对该
run 结果的**又一次函数调用**：`#strong[x](y)`。原文的实现有一个
`_break_run_call_chains` 专门处理，移植行内样式时漏掉了。实测
`gan.pdf`：`#super[...][g](4)` 编译失败，整个块丢失。

**处置**：从已删的 `overlay_text` 移植 `_break_run_call_chains`，
并补 2 条单元测试 + 2 条真实 Typst 集成测试。

### 8.7 测试基线

| 层 | 结果 |
| :--- | --- |
| `pytest -m fast` | **2917 passed**, 22 deselected |
| `pytest -m slow` | **22 passed** |
| `ruff check` / `format` | 干净（467 files） |
| `scripts/check_doc_refs.py` | all resolve |
| 死代码护栏 | passed |
| `web`: tsc / oxlint / vite build | 全部通过 |

净变更：**123 files changed, 2412 insertions(+), 11027 deletions(-)**。

---

## 9. 实施日志：模式面收敛（2026-10-12）

§1–§2 的设计落地。用户可见的模式旋钮从 **16 → 3**。

### 9.1 轴 A：`dual_mode` 收敛为 3 值

```diff
-DualMode = Literal["inline", "alternating", "monolingual", "facing", "auto"]
+DualMode = Literal["mono", "dual", "auto"]
```

**理由**：重排渲染器重建整页，页面 zip 不可能（页数会变），所以
`inline`/`alternating`/`facing` 三种"双语版式"其实是**同一种**
（per-block 配对）。实测：

| 输入 | `RENDER_MODE_VALUE` | 适配器 `publish_bilingual` |
| :--- | :--- | :--- |
| `inline` | `bilingual` | `True` |
| `alternating` | `alternating` | `True` |
| `facing` | `facing_spread` | `True` |
| `monolingual` | `monolingual` | `False` |

前三个产出**逐字节相同的文档**——因为判定条件是 `active_mode != "monolingual"`。

**同步删除**：`facing_spread`（§8.4 已证是空开关）从 config / RenderPlan /
advisory / reflow_loop / visual_gate / CLI / REST / MCP / web 全部移除。

**规则罚分合并**：`{inline: 0.50, alternating: 0.10, monolingual: 0.0}` →
`{dual: 0.50, mono: 0.0}`。`mono` 是安全兜底：任何规则都不得把它推得更差，
只允许通过惩罚配对来间接选择它（新测试 `test_every_rule_penalizes_dual_and_never_mono`
钉住这个不变量）。

### 9.2 轴 B：删除 7 个引擎细节字段

| 字段 | 处置 | 实测依据 |
| :--- | :--- | :--- |
| `formula_render` | **删除** | **完全惰性**：三个取值 + 一个乱写值，`resolve_formula_enrichment` 输出**逐字节相同**。只有 `"image"` 分支改了一行日志 |
| `formula_mode` | **删除** | 零行为消费者；只在 `ReportRouteInfo` 里显示 |
| `math_backend` | **删除** | 零行为消费者（注释自承"已废弃"） |
| `exec_mode` | 移出用户面 | 并入 preset（`fast` 用 `short`） |
| `prompt_strategy` | 移出用户面 | 并入 preset |
| `qe_engine` | 移出用户面 | 并入 preset |
| `formula_enrichment` | 移出用户面 | 并入 preset |
| `table_render_mode` | 移出用户面 | 并入 preset（publish 下 `auto` 已正确） |

**"移出用户面"的含义**：字段仍在 `UBTConfig` 里（preset 和 `UBT_*` 环境变量
可设），但从 CLI / REST / MCP 三个入口移除。实测 `env_var_names()` 全部有
逃生舱口。

### 9.3 `preset` 现在覆盖全部派生字段

```
high_fidelity  exec_mode=auto  prompt=rich     enrich=auto  qe=heuristic  table=auto  pdf=auto
standard     exec_mode=auto  prompt=auto     enrich=auto  qe=heuristic  table=auto  pdf=auto
preview      exec_mode=auto  prompt=minimal  enrich=off   qe=heuristic  table=auto  pdf=pdfium
fast         exec_mode=short prompt=minimal  enrich=off   qe=heuristic  table=auto  pdf=pdfium
```

选一个 preset = 选定全部内部旋钮。

> **注**：`high_fidelity` 的 `qe_engine` 定为 `heuristic` 而非 `tiered`——
> `tiered` 需要可选神经 runner（`comet`），把 preset 绑到一个可能不存在的
> 依赖上会在无 GPU 环境静默降级。保持行为不变优先。

### 9.4 **发现 web 控制台当前是坏的**

web wizard 仍在提交 `render_mode` 和 `formula_mode`，而 REST 模型是
`extra="forbid"`：

```
JobSubmitRequest(input_path='/x', target_lang='zh', render_mode='overlay')
→ ValidationError: Extra inputs are not permitted [type=extra_forbidden]
```

也就是说**任何通过 web 控制台提交的任务都会失败**。这解释了为什么"模式收敛"
必须连同 web 一起做——单一入口改了、其他入口没改，就是这种结果。

已修：wizard 删除这两个控件 + 请求字段 + 7 个孤儿 i18n 键。

### 9.5 收敛后的用户面

**REST 请求模型**（用户能设的全部字段，模式类加粗）：

```
input_path output_path target_lang source_lang profile draft_model repair_model
**preset** pdf_engine **dual_mode** pages glossary budget_usd max_concurrency
batch_limit macro_chunk_size short_max_pages enable_rolling_summary
chapter_streaming_enabled offline_batch_enabled visual_judge_enabled
visual_judge_model translate_chrome emit_both **cover_mode** ocr_mode domain
job_id priority dry_run fresh start_chapter max_chapters
```

**三个模式旋钮**：`dual_mode`（要什么）、`preset`（花多少力）、`cover_mode`（低频）。

**CLI 模式选项**：`--dual-mode` / `--preset` / `--cover-mode`。
已消失：`--formula-mode`、`--math-backend`、`--formula-render`、`--exec-mode`、
`--mode`、`--prompt-strategy`、`--qe-engine`、`--table-render-mode`。

### 9.6 刻意偏离 §2 的一处：`ocr_mode` 保留在用户面

§2 把 `ocr_mode` 归入"隐藏（环境变量）"。**不采纳**，但理由与初版判断不同
——初版理由经实测是**错的**，记录在此以免后人重蹈。

#### 初版（错误）理由

曾写："显式设 `ocr_mode=cloud/vlm` 是用户授权出网的唯一动作"。据此推断把它降为
环境变量会拿走用户的隐私同意权。

#### 实测反驳

```python
# ubt/adapters/pdf/vlm/registry.py:186-205
if clean_mode == "cloud":
    if not allow_page_upload:
        raise ValueError(
            "ocr_mode='cloud' ships rendered book pages to a remote "
            "endpoint, but allow_page_upload=false ... forbids ..."
        )
```

实测（`allow_page_upload=False`，即默认）：

| `ocr_mode` | 结果 |
| :--- | :--- |
| `off` / `auto` / `rapidocr` | `None`（不出网） |
| `sidecar` | `sidecar`（本地） |
| `cloud` / `vlm` | **`ValueError`——出网被拒** |

**显式设 `cloud`/`vlm` 不能绕过闸门，反而报错。** 真正的出网闸门是
`allow_page_upload`，它**不在任何用户面上**（运维级，正确）。

#### 修正后的理由

`ocr_mode` 该保留，是因为它对用户有**真实的、文档化的用途**，与隐私无关：

1. **排障**：`MANUAL_E2E_INTEGRATION_TEST_SPEC.md:682` 教用户用 `--ocr off`
   确认"扫描件提取到 0 块"不是 OCR 的锅。
2. **成本可见性**：`ubt/core/assess.py:610` 用它决定云 OCR 的页数估算——
   用户能看懂"这次会花多少钱在 OCR 上"。
3. **本地优先的显式确认**：`sidecar` / `rapidocr` 是本地路线，与 `cloud`/`vlm`
   是不同的钱和不同的隐私姿态，用户有权明说。

**结论**：`ocr_mode` 与 `cover_mode` 同类——低频、语义清晰、用户可能在意，
且它有真实行为（不像 `formula_render` 那样惰性）。用户面因此是
**3 个模式旋钮**（`dual_mode` / `preset` / `cover_mode`）加上 `ocr_mode`。

### 9.7 实测验证

```
CLI --help          模式选项 = --dual-mode / --preset / --cover-mode
三种模式端到端      auto=185594B  dual=215956B  mono=185594B
dual 真含原文       10552 字符 → 19837 字符（多 9285）
```

**测试基线**：`pytest -m fast` **2929 passed**；ruff 干净；
`check_doc_refs.py` all resolve；死代码护栏 passed；
web tsc/oxlint/vite build 全通过。

新增测试：
- `tests/unit/policy/test_bilingual_advisor.py` 4 条（收敛空间不变量）
- `tests/unit/test_option_surface_parity.py` 10 条（`_INTERNAL_KNOBS` 断言每个引擎
  细节都不在任一用户面上，且仍有 env 逃生舱口）

---

## 10. §2 剩余项的逐字段审计（2026-10-12）

§9 收敛了 7 个字段。§2 里剩余的 `api_mode` / `docling_device` / `job_mode` /
`consistency_enforce` 需要逐个核实——**RFC 的处置建议是基于假设写的，必须验证**。

### 10.1 全字段终局审计（脚本生成，非人工维护）

```
字段                    REST   MCP    CLI    ENV    终局
render_mode           -      -      -      —      已删除
dual_mode             yes    yes    yes    yes    ★ 用户旋钮
table_render_mode     no     no     no     yes    引擎细节（preset）
formula_mode          -      -      -      —      已删除（零行为）
formula_render        -      -      -      —      已删除（零行为）
formula_enrichment    no     no     no     yes    引擎细节（preset）
math_backend          -      -      -      —      已删除（零行为）
exec_mode             no     no     no     yes    引擎细节（preset）
prompt_strategy       no     no     no     yes    引擎细节（preset）
qe_engine             no     no     no     yes    引擎细节（preset）
cover_mode            yes    yes    yes    yes    ★ 用户旋钮
consistency_enforce   no     no     no     yes    运维级
ocr_mode              yes    yes    no     yes    ★ 用户旋钮（见 §9.6）
api_mode              no     no     yes    yes    运维级（CLI = 运维面）
docling_device        no     no     no     yes    运维级
job_mode              no     no     no     yes    运维级
```

### 10.2 `job_mode` 不是旋钮

§2 把它列入"待隐藏"。实测它**从来不是请求字段**：

```
JobSubmitRequest 有 job_mode: False
所属模型: SystemInfoResponse
```

它是 `/system/info` 的**响应字段**，报告本实例是 `embedded` 还是 `queue`。
无可隐藏。

### 10.3 `api_mode` 的 CLI 可见性是正确的

审计表显示 `api_mode` 在 REST/MCP 上被拒、CLI 上可见。这不是不一致，是
**刻意的分层**——实测：

```
provider         拒绝（operator-only）
base_url         拒绝（operator-only）
api_mode         拒绝（operator-only）
api_key          拒绝（operator-only）
ocr_endpoint     拒绝（operator-only）
```

`overrides_from_request(..., allow_provider_keys=False)` 对 REST/queue/MCP
payload 统一拒绝这 5 类键。CLI 是**运维面**（本机操作者），所以它提供
`--provider` / `--base-url` / `--api-mode` / `--api-key` / `--ocr-endpoint`。

**结论**：CLI 与 REST/MCP 的可见性差异是安全边界，不是遗漏。

### 10.4 `docling_device` / `consistency_enforce`：确认纯运维

两者在三个入口都不可见，`UBT_*` 可设。实测无 CLI 选项、无请求字段。
§2 的处置**正确**，无需改动。

### 10.5 收敛的最终账

| | 之前 | 之后 |
| :--- | ---: | ---: |
| 用户可见模式旋钮 | **16** | **4** |
| — 核心决策 | 0 | `dual_mode` / `preset` |
| — 低频布局 | 1 | `cover_mode` |
| — 有真实行为的摄取 | 1 | `ocr_mode` |
| 引擎细节（preset/env） | — | 6 |
| 已删除（零行为） | — | 4 |
| 运维级（env only） | — | 4 |

---

## 11. 金标集回归（2026-10-12）：20 篇真实论文

收敛 + overlay 清除之后，用本地模型（`gemma4-e4b`）跑完整流水线，
`--dual-mode mono`，零 mock。

### 11.1 结果

| 指标 | 值 |
| :--- | :--- |
| **通过** | **20/20** |
| `untranslated_marks` | **0（全部）** |
| `math_residue_ratio` | **0.000（全部）** |
| `p95_font_size_pt` | **10.5（20/20 完全一致）** |
| `source_header_term_leakage` | **0.000（全部）** |
| `cjk_ratio` | 0.338 – 0.779，中位 0.491 |

`cjk_ratio` 中位 0.491 低于 0.55 的目标，但这**不是缺陷**：
`mono` 交付包含参考文献、公式符号、表格数值、作者名等**按设计保留原文**
的内容，它们全部计入分母。真正的质量信号是
`untranslated_marks=0` 与 `source_header_term_leakage=0`——**没有漏译，
没有源语言泄漏到应当翻译的表头**。

### 11.2 回归中抓出的 5 个真 bug

跑真实语料才暴露的问题，全部由 **overlay 删除**引入（overlay 时代这些块
留在源页画布上，从不经过渲染器）。

#### Bug A：mask token 在数学模式被转译

`⟦SOUP_MASK_0001-333⟧` 出现在 `$...$` 内时，`typstify_math` 把 `_` 当
Typst 下标运算符，产出 `⟦#"SOUP"_#"MASK"_0001-333⟧`——`#` 进入代码模式，
编译器报 `unknown variable: SOUP`。

**影响**：alphafold2、attention-is-all-you-need 整篇失败。
**修复**：`_MASK_TOKEN_RE` 命中时整段按 prose 转义（`\_`），token 字面渲染。

#### Bug B：样式 run 跨越 math 边界

`render_prose` 按 run 边界切分文本后**逐块**转译，所以跨越 `$` 的 run
把公式劈成两半，每个 `$` 被转义成字面 `\$`。

**影响**：alphafold2 整篇失败。
**修复**：判定从"完全包含"改为"任何重叠"——但**保留**"run 完全包含公式"
的情况（那样整段在一个 chunk 内，定界符平衡）。

#### Bug C：不平衡的 `$` 把散文吞进数学模式

模型返回 7 个 `$`（应为偶数）时，`_MATH_SPAN_RE` 按位置配对，
`$ 是一个序列，其中 $` 成了一个数学跨度。

**影响**：alphafold2 整篇失败。
**修复**：`_is_math_span` 拒绝含 CJK 的候选跨度——真数学不含汉字。

#### Bug D：表格单元格被双重转义

`parse_markdown_table` 先转义单元格（`$`→`\$`），`render_table` 又走
`render_prose` 转义一次（`\$`→`\\\$`）。在 `[...]` 内容块里 `\\` 是换行，
定界符永不闭合。

**影响**：attention-is-all-you-need 整篇失败。
**修复**：`parse_markdown_table(..., escape=False)` 显式化转义契约。

#### Bug E：模型把强调标记转写成 LaTeX

模型把 `⟦B⟧` 读成数学括号，写成 `\llbracket B \rrbracket`。`parse_bold`
只认 `⟦B⟧`，于是标记残留进数学模式，`\llbracket` 与裸 `/` 让编译器失败。

**影响**：mamba、neural-ode 整篇失败。
**修复**：`_MARKER_RE` 同时接受 LaTeX 转写形态。

### 11.3 一条方法论

**五个 bug 的共同形状**：一个"渲染器不该看到"的中间态字符串
（mask token、强调标记、未平衡定界符）到达了 `typstify_math`，
而数学模式对它们各有各的破坏方式。

overlay 时代这条路径不存在（块留在源页），所以没有任何测试覆盖它。
**真实语料回归是唯一能发现它们的办法**——单元测试的 fixture 是干净的，
而真实模型输出不是。

**测试基线**：`pytest -m fast` **2970 passed**；ruff 干净；
doc refs all resolve；死代码护栏 passed。
新增回归测试 17 条（mask token 5、run 边界 4、定界符 4、表格 3、强调标记 4 去重后）。

---

## 12. 后续三项（2026-10-12）

§11 的金标集回归抓出 5 个 bug 之后，收尾三件事：共同防御层、指标定义、提交。

### 12.1 哨兵守卫：把"不该到达渲染器"变成可检查条件

§11.2 的五个 bug 有同一个形状——**一个中间态字符串到达了 `typstify_math`**。
修复是逐个打补丁，但补丁之间没有共同防御：下一个 masker 家族换个前缀，
同类 bug 会原地复发。

新增 `ubt/render/typst/artifact_guard.py`：

- `find_sentinels()` / `has_sentinel()` / `sentinel_kinds()`
- 覆盖**全部** `⟦…⟧` 家族（math / soup / cite / code / email / MASK / 强调 /
  semantic_break）+ 模型把 `⟦B⟧` 转写成的 `\llbracket B \rrbracket`
- 渲染器只问一次 `has_sentinel(inner)`，命中则整段按 prose 逃逸

**关键设计：注册表是字面量列表，不是从各 masker 导入。** 理由：渲染器不该
拉进 cleaner 栈；代价是列表可能落后。所以有一条反漂移测试
`test_the_registry_covers_every_masker_family` —— 它读每个 masker 自己的
`default_prefix`，断言守卫认得由它构成 token。**这正是原 bug 一个家族一个
家族藏起来的方式。**

同时把泄漏变成**可测量的数字**：`ReflowStats.sentinels_leaked`。逃逸让文档
仍能编译（这是重点），但静默修复会掩盖上游 restore 缺陷——现在它能在质量
报告里被计数。

### 12.2 `cjk_ratio` 的定义问题：一个指标测了两件事

§11 报告 `cjk_ratio` 中位 0.491，低于 0.55 目标。判断是"不是缺陷"，但**当时
只是推断，没有量化**。这轮量化了。

用 ledger 把每个块分成两类：政策**故意保留源语言**的（参考文献、作者单位、
代码清单）vs **本该翻译**的：

| job | 总块 | 保留 | 全文 `cjk_ratio` | 仅可译块 |
| :--- | ---: | ---: | ---: | ---: |
| `job_997046169952` | 126 | 34 | 0.739 | **0.844** |
| `job_c1984bb50a5b` | 769 | 134 | 0.618 | **0.803** |
| `job_27af622bf509` | 532 | 195 | 0.475 | **0.792** |
| `job_1df284ce95f7` | 838 | 245 | 0.525 | **0.780** |
| `job_97fd272f1fdf` | 811 | 238 | 0.550 | **0.776** |
| `job_435299cc42b7` | 284 | 132 | 0.340 | **0.714** |
| `job_adf70ed1803c` | 565 | 165 | 0.442 | **0.694** |
| `job_8ce7b83971a1` | 229 | 33 | 0.518 | **0.574** |
| `job_f340f6ace31a` | 303 | 90 | 0.337 | **0.502** |

**结论：`cjk_ratio` 是一个"交付"指标，不是一个"翻译"指标。** 它同时测了
"译了多少"和"故意留了多少"，后者的方差比前者大得多。单个数字无法回答
"翻译质量如何"。

修复：新增 `ledger.cjk_ratio_translatable`，定义域**精确对齐渲染器的
`_keeps_source()`**（policy verdict 权威，否则看 `skip_translate`）。
两条实现必须同构——`test_a_policy_verdict_of_true_does_not_clear_a_skip_flag`
就是断言这一点的。

**金标集 20 篇实测**：

| 指标 | 达标数 |
| :--- | ---: |
| `cjk_ratio` ≥ 0.55（全文） | **7/20** |
| `cjk_ratio_translatable` ≥ 0.55 | **19/20**（中位 0.71） |

唯一未达标的 `stable-diffusion` 是 0.517——它含大量图注与算法伪代码。

两个数字都保留并并排报告。删掉全文指标会隐藏"故意保留"的规模；只报全文
指标会把一个能用的流水线读成几乎没翻译。

### 12.3 刻意偏离与提交

**保留 `ocr_mode` 在用户面**（RFC §9.6）：上一轮已更正为"命名远程路线
≠ 授权使用它"（实测 `cloud`/`vlm` 在 `allow_page_upload=False` 下直接报错）。
本轮无变更。

提交范围：本会话累计 **131 files, +2900 / −11611**。测试基线
`pytest -m fast` **3004 passed**。

**三条方法论**（贯穿本会话）：

1. **真实语料回归不是可选验证**，而是发现"中间态字符串泄漏"这类 bug 的
   唯一手段。单元测试的 fixture 是干净的，真实模型输出不是。
2. **一个 bug 修完要问"同一类还有没有"**。五个 bug 同一个形状，补丁式修复
   留下的是同类 bug 的复制品；共同防御层（12.1）才是收敛。
3. **指标的定义域要和被测量的决策对齐**。`cjk_ratio` 的分母包含了政策故意
   保留的块，于是它测的不是它名字声称的东西（12.2）。

### 12.4 提交前的 mypy 门禁又抓出 2 个真 bug

`pre-push` 跑 `mypy --strict`。为让新增测试通过该门禁，把
`tests/unit/ir/test_document_tree.py` 和 `tests/unit/engine/test_section_chunking.py`
里的**鸭子类型替身**换成真实 `IRBlock`——立刻炸出两个 bug。

#### bug ①：`_effective_level` 读了一个不存在的字段

```python
level = getattr(block, "level", None)  # IRBlock 没有 level
```

`IRBlock` 是执行状态模型，**结构在 `block.element` 上**（`ledger_base.py:115`
写的正是 `getattr(b.element, "level", None)`）。`getattr` 带默认值，
所以永远静默返回 `None` → 永远用 fallback=1。

**后果：每个标题都开一个新节，子节永不嵌套。** 实测某真实论文：
**200 个标题 → 200 个平级节**（修复后 147 个，`1.1` 正确嵌在 `1` 下）。
RFC-2026-02 §2.3 的"按节翻译"因此从未真正生效——它退化成"按块翻译"。

#### bug ②：`heading_trail` 只返回叶子标题

docstring 承诺 `"2 Switch Transformer > 2.1 Simplifying"`，实现是
`return section.title`——**只有叶子**。`Section.path` 精心做了
trim-to-depth 计算，却**没有任何生产代码读取它**。

**后果**：模型看到 `"2.1 Simplifying Sparse Routing"`，不知道它属于哪一章——
正是 back-reference 需要的那个 antecedent 丢失了。

修复：新增 `Section.trail`（按标题的祖先链），`heading_trail` 用 `" > "` 连接。
实测 prompt 现在拿到 `"2 Method > 2.1 Simplified Sparse Routing"`。

> **为什么鸭子类型替身掩盖了它们。** 替身 `_Block` 有一个真实的 `level` 字段，
> 而 `IRBlock` 没有。测试通过、生产失败——**替身的形状就是它撒的谎**。
> 这是本轮第三次同类发现：**测试 fixture 与真实对象的偏差是 bug 的藏身处**。

**测试基线**：`pytest -m fast` **3006 passed**；`mypy --strict` 只剩 4 个
HEAD 已有的 `pdf_builders` 相关错误；覆盖率 70.5%（门禁 58%）。

---

## 13. 审计：把"重构完成"从断言变成证据（2026-10-12）

用户问"所有重构任务都解决了吗"。**答案是"没有"**——审计又找出 3 个真 bug 和
一批命名残留。这一节记录审计方法与结果，因为"我检查过了"本身不是证据。

### 13.1 审计方法

不是 grep "overlay" 就完事。四条独立扫描：

1. **语法残留**：`overlay|LayerCompositor|render_fidelity|facing_spread|...` 全库
2. **死键**：`ARTIFACT_METADATA_KEYS` 每个 key 是否有写入点
3. **死字段**：`UBTConfig.model_fields` 每个字段是否有读取点（含 `getattr` 形态）
4. **死枚举值**：每个 `Literal` 的每个取值是否有行为消费者

### 13.2 新发现的 3 个真 bug

#### bug ③：`table_render_mode="overlay"` 是一条活代码路径

配置注释写着"历史名字保留，没有 overlay 渲染"——**但分支还在**：

```python
elif table_mode == "overlay":
    comp = probe_table_complexity(...)
    if comp.is_complex:
        b.skip_translate = True   # 复杂表格保持源语言
```

它的存在理由（"源页画布还在，复杂表格照样看得见"）随 overlay 一起死了。
**后果**：`UBT_TABLE_RENDER_MODE=overlay` 仍可从环境变量设置，然后**静默把
复杂表格留在英文**——在一个只有重排的交付物里，那就是**表格凭空消失**。

修复：`TableRenderMode` 收敛为 `auto|appendix|skip`，删掉该分支。实测
`UBT_TABLE_RENDER_MODE=overlay` 现在**报 ValidationError**（显式拒绝，
而非静默当 `auto`）。

#### bug ④：`cover_mode` 从未实现过

它被**完整地**接到了 CLI / REST / MCP 三个入口，存进 `RenderPlan`——
**但没有任何代码读它**。`TypstDocumentRenderer.__init__` 根本不接受该参数。

**端到端实测**（`gan.pdf`，三个取值）：

| cover_mode | 文本层 sha | 页数 |
| :--- | :--- | ---: |
| `auto` | `926d04b8…` | 9 |
| `always` | `926d04b8…` | 9 |
| `never` | `926d04b8…` | 9 |

文本层**逐字节相同**，页数相同（PDF 文件 sha 不同只是创建时间戳）。这是
`formula_render` 的翻版：一个用户可见的开关，改变不了任何东西。
**更糟的是** CLI 帮助文本还描述了不存在的行为
（"'auto' (cover only when page 1 has no body text)"）。

修复：从 config、`RenderPlan`、CLI、REST、MCP、OpenAPI 全部删除。

#### bug ⑤：`prices_file` 在配置文件里设置无效

字段 docstring 写着 "Path to external prices.toml (UBT_PRICES_FILE)"——
听起来两者等价。**实测不等价**：

```
UBTConfig(prices_file='/tmp/p.toml') + bootstrap  → test-model-xyz 无价格
UBT_PRICES_FILE=/tmp/p.toml                       → test-model-xyz 有价格
```

`find_prices_file()` 只读环境变量，`bootstrap_runtime_environment()` 从不把
配置字段传下去。**用户在配置文件里设的价格表被静默忽略**——按错误的价格记账。

修复：新增 `declare_custom_prices_file()`（与既有
`declare_custom_free_endpoint()` 同形），bootstrap 里调用。实测字段方式现在生效。

### 13.3 清理的命名残留

| 位置 | 问题 |
| :--- | :--- |
| `test_overlay_text_math.py` | 文件以已删模块命名 → `test_typst_symbols.py` |
| `test_overlay_math_fallback.py` | → `test_math_fallback.py` |
| `run_metadata.py` | `render_engine_effective`/`render_padding_pages` 死 key（0 引用）→ 删 |
| `math_symbols.py` | "inline overlay converter" → "Typst math transpiler" |
| `test_content_adapt.py` | docstring 断言 "single overlay engine / LayerCompositor" → 重写 |
| `docling_crosscheck.py` | "sinks to PRESERVED_OPAQUE on Layer 0" → 重写 |
| `test_coordinate_resolver.py` | "misplaces the translation overlay" → 重写 |
| `test_docling_markers_indent.py` | "become the overlay's flow chain" → 重写 |
| `test_docling_crosscheck.py` | "handled by Layer 0" → 重写 |
| `test_table_appendix.py` | "eligible for in-place overlay" → 重写 |

### 13.4 新增护栏

- `test_cover_mode_is_gone_from_the_config_too` —— 断言字段真的不存在
- `test_the_dead_overlay_table_mode_is_not_accepted` —— 断言 `overlay` 被拒
- `test_the_render_metadata_keys_have_a_writer` —— **每个** `ARTIFACT_METADATA_KEYS`
  都要有写入点（这条正是抓 bug ④ 同类的通用护栏）
- `test_a_prices_file_set_in_config_is_used` —— 断言字段与 env var 等价

### 13.5 核实为"合理"的项

`local_endpoints` / `strict_auth` / `allowed_dirs` 只在 `config.py` 内部被消费，
但那是**正确的**：它们分别喂 `declare_custom_free_endpoint()` /
`is_strict_auth()` / `allowed_base_dirs()`，后者被 API 层读取。

### 13.6 一条方法论

**"重构完成"必须由扫描证据支撑，而不是由"我记得改过了"支撑。**

四条扫描各有独立价值：语法残留只能抓命名；死 key 扫描只能抓 metadata；
死字段扫描只能抓 config；**死枚举值扫描只能抓 `overlay` 和 `cover_mode` 这
一类——值还在、分支还在、但理由已经消失**。

最后一条最关键：`table_render_mode="overlay"` 和 `cover_mode` **都不含
"overlay" 之外的线索**，`grep overlay` 能碰到前者却碰不到后者。
**能改变不了任何东西的开关，是这类重构最顽固的残留。**

**测试基线**：`pytest -m fast` **3006 passed**。

---

## 14. 逐条对账 RFC §9 验收指标（2026-10-12）

用户问"按重构要求是否全部完成"。不再讲过程，直接对账 **SEMANTIC_FIRST_RFC §9
的八项验收指标**。

### 14.1 对账表（金标集 20 篇实测）

| # | 指标 | 目标 | 实测 | 判定 |
| :- | :--- | :--- | :--- | :--- |
| 1 | 中文占比（正文） | ≥ 55% | **仅可译块中位 0.712，19/20 达标** | ✅ |
| 1b | 同上，全文口径 | — | 中位 0.491，6/20 达标 | 见 §12.2（含故意保留） |
| 2 | 未翻译块 | ≤ 5% | max 0.003；`untranslated_marks` 全 **0** | ✅ |
| 3 | 表格中文化率 | ≥ 95% | **中位 0.614，1/20 达标** | ❌ **未达标** |
| 4 | 正文字号 P95 | ∈ [10,11]pt | **10.5（20/20 完全一致）** | ✅ |
| 5 | 公式 AST 一致率 | ≥ 99% | 残留率 0.000，但 **AST 一致率未被测量** | ⚠ **未测量** |
| 6 | 术语一致率 LTCR | ≥ 0.95 | 有术语的 **9/20**，中位 **0.862** | ❌ **未达标** |
| 7 | 端到端交付率 | ≥ 95% | **20/20 = 100%** | ✅ |
| 8 | 渲染耗时 | < 2s/篇 | 中位 **0.87s**，max 1.27s | ✅ |

**结论：8 项中 4 项达标，1 项未达标（表格），1 项未达标（术语），1 项未测量
（公式 AST），1 项口径需说明（中文占比）。**

### 14.2 附带修掉的一个真 bug：表格指标是**反转**的

对账过程发现 `translated_table_header_cjk` **无法给正确翻译打分**：

原来的实现先要求单元格**含拉丁词**，再检查同一格**含 CJK**。可是译好的单元格
是「CJK、无拉丁」——**分子分母同时被排除**；而唯一被计入的恰恰是必须逐字保留
的标识符（`MNIST`、`Dev`、`Test`），全部算失败。

**实测证明**：一个纯中文表头的完美翻译 → **0.0**。
**后果**：§9 第 3 项被报成"中位 0.146、全军覆没"，实际是**指标坏了**，
不是表格没译。修正后中位 **0.614**。

修正实现按 RFC 原文（「表格单元格 CJK 覆盖」）统计**全部**表头单元格中 CJK 的
占比，并保留 `_table_header_cells` 列出未中文化的格子——因为 `MNIST` 这类
**应当逐字保留**，比值天然是下界，必须可审计。

> 这条与 §12.2 的 `cjk_ratio` 是同一类错误：**分母的定义域与被测量的决策不
> 对齐**。区别是 `cjk_ratio` 只是偏保守，而这个指标**直接反转**——越正确越低。

### 14.3 仍未完成的三项（诚实清单）

| # | 缺口 | 性质 | 说明 |
| :- | :--- | :--- | :--- |
| **3** | 表格中文化率 0.614 < 0.95 | **真实质量缺口** | 修正指标后仍是最大短板；alphafold2 / dropout 为 0.0 |
| **5** | 公式 AST 一致率未测量 | **缺测量工具** | 只有 `math_residue_ratio`（残留率 0），无规范化 AST diff |
| **6** | 术语一致率 0.862 < 0.95 | **真实质量缺口 + 覆盖不足** | 9/20 篇有 glossary；`target_lang` 术语覆盖需扩大 |

此外 RFC §8 的 Sprint 2 交付物 3「`academic_zh.typ` 学术**双栏**模板 +
`#place(top, float: true)` 跨栏浮动」**未实现**——当前是单栏 A4。这是一个
**有意的偏离**（单栏对重排更稳），但 RFC 原文未回写，属于文档与实现的偏差。

### 14.4 下一步（按性价比排序）

1. **表格中文化率**（缺口最大）：`bert`/`ddpm` 这类表的表头是数据集名，
   需要区分「该译而没译」与「本就该逐字保留」——先给出可判定的口径，
   再决定是否真修。
2. **公式 AST 一致率**：补 `scripts/eval_math_ast.py`，规范化后 diff。
3. **术语覆盖**：让无 glossary 的 11 篇也有术语表，才谈得上"一致率"。

**测试基线**：`pytest -m fast` **3010 passed**。

---

## 15. 继续对账：又修掉 3 个真 bug，并纠正术语指标口径（2026-10-12）

§14 列出三项未达标后继续做，结果每一项都先炸出**指标/渲染的真 bug**。

### 15.1 表格指标反转（已修，§14.2）

`translated_table_header_cjk` 无法给正确翻译打分：完美中文化的表格得 **0.0**。
修正后中位 **0.614**。另发现 2 篇（alphafold2 / dropout）**根本没有表格**，
其 0.0 是"无表格"而非"未翻译"——已记录，避免误读。

### 15.2 术语一致率：分母同样包含了"故意不译"的块（已修）

RFC §9 第 6 项目标是 **tier B ≥ 0.90**（§4.3：born-digital PDF 是参考级，
不是出版级）。实测 3/9 达标，中位 0.862。

**逐块核查后发现根因**：`evaluate_terms()` 遍历**全部**块，包括政策明确
"保留源语言"的参考文献/作者单位块。这些块**本就该是英文**，不可能渲染中文
术语。实测每篇 **2–23%** 的术语出现落在这种块里。

修复：`evaluate_terms()` 跳过 `_keeps_source()` 的块（判定与渲染器同构）。

| 口径 | 中位 | ≥0.90 |
| :--- | ---: | ---: |
| 全部块（修复前） | 0.862 | **3/9** |
| 仅可译块（修复后） | **0.969** | **8/9** |

> 这是本会话**第四次**同一类错误：**分母的定义域与被测量的决策不对齐**。
> `cjk_ratio`（§12.2）、表格指标（§14.2）、术语指标（本节）三次都是它。

### 15.3 渲染器：三个"公式变成字面 LaTeX"的真 bug（已修）

对账第 5 项（公式）时发现公式残留不降反升，逐一定位出三个独立缺陷：

| # | 缺陷 | 实测案例 |
| :- | :--- | :--- |
| ① | 表格按每个 `\|` 切分，把 `\left\|` 双竖线范数切成三格 | ddpm `#b0053`：`[\$\left\], [\tilde{...}], [\$]` |
| ② | 含中文 `\text{}` 注解的公式被当成散文逃逸 | mamba `#b0086`：`\$\mathbf\{A\} \text\{ )← 参数 \}\$` |
| ③ | 纯标点跨度 `$、$` 被当数学，错位后续配对 | mamba `#b0086`：`$ 和矩阵 $` 成了公式 |
| ④ | 样式 run 自身 `$` 个数为奇数时无法包裹 | mamba：`\$#strong[\$` |

修法：
- `parse_markdown_table` 改用**扫描器**（尊重 `\|` 转义与 `$...$` 区间）
- `_is_math_span`：含 CJK **且有数学结构**才算数学；纯标点不算数学
- `render_prose`：run 文本 `$` 数为奇数时不加样式（边界不可信）

**实测**：ddpm 4→**0**、mamba 6→**1**，全语料公式残留 11→**3**
（剩余 1 个是模型自身产出的坏 LaTeX，非渲染器缺陷）。

### 15.4 修术语真 bug：表格里的大写词不是标识符（已修，见前一提交）

### 15.5 仍未完成（诚实清单）

| # | 缺口 | 状态 |
| :- | :--- | :--- |
| 5 | 公式 AST 一致率 | **仍未测量**（残留率 3/2662，但 AST diff 无工具） |
| 6 | 术语覆盖 | 11/20 篇因 `fast_lane`（≤20 页）跳过挖掘 → 无术语表 |
| 3 | 表格 0.614 / 术语 0.863（stable-diffusion） | 仍有真实缺口 |

**测试基线**：`pytest -m fast` **3027 passed**。

---

## 16. RFC §9 第 5 项（公式 AST）从"未测量"到 0.9995（2026-10-12）

### 16.1 先建测量工具

RFC §9 第 5 项要求「公式 AST 一致率 ≥ 99%」，此前**只有 `math_residue_ratio`
（残留率），那是另一个问题**——公式可以零残留却仍是错的（丢了 `\frac`、
上标变下标、`\sum` 变成纯文本）。

新增 `scripts/eval_math_ast.py`：把两侧归约到同一**结构签名**，因为不存在
Typst 解析器。

| 构造 | LaTeX 侧（MathML AST） | Typst 侧（产物源码） |
| :--- | :--- | :--- |
| 分式 | `mfrac` | `frac(` |
| 二项式 | `mfrac[linethickness=0]` | `binom(` |
| 根号 | `msqrt`/`mroot` | `sqrt(`/`root(` |
| 上标 | `msup`/`msubsup` | `^` |
| 下标 | `msub`/`msubsup` | `_` |
| 大算符 | `mo` 含 ∑/∏/∫ | `sum`/`product`/`integral` |
| 重音 | `mover`/`munder` | `hat(`/`tilde(`/… |

签名**故意粗糙**：分不出 `x^2` 与 `y^2`（读者能看出，结构检查不该因此失败），
也看不见顺序；它抓的正是这个目标存在的理由——**一个构造丢失或变成另一个构造**。

LaTeX 侧解析失败记为 `unparsed`（不算通过），transpiler 拒绝记为
`undelivered`，两者都不允许被"未测量"冒充为达标。

### 16.2 工具立刻抓出 4 类真 bug

#### ① `\mathbf{}` 让整个公式失效

`\mathbf`/`\mathit` **被有意不映射**（注释写"upright would be a silent styling
lie"），后果不是"不加粗"，而是**整个公式返回 None**、降级为纯文本：
`$\mathbf{W} x + b$` → `W x + b`。全语料 **105 处**，8 篇受影响。

修复：映射到 Typst 的 `bold()`/`italic()`（真实样式，不是谎言），并把
`bold`/`italic` 加进 `_TYPST_MATH_WORDS`（否则词串引号器会把函数名变成字面文本
`#"bold" (W)`）。

#### ② 下标紧邻命令导致函数名合并

`\beta_t\sqrt{x}` → `beta_#"tsqrt" (x)`——**根号整个丢失**。
`\frac` 同理。全语料 6 处。

修复：`_convert_fractions`/`_convert_sqrts` 在紧邻 `_`/`^`/字母时补一个分隔符。

#### ③ 一批常用符号缺失

`\downarrow`（9 处）、`\checkmark`（7）、`\ddagger`（4）、`\binom`（3）、
`\ell`、`\star`、`\ddagger`、`\#L`……全语料 **93 个公式**因此降级为纯文本。
`\downarrow`/`\uparrow` 是结果表里"越低越好/越高越好"的标记，`\ell` 是标准
索引变量——都不是冷门符号。

修复：补齐符号表（名字逐个对编译器验证）；`\binom` 单独做双参数转换；
`\#` 转义放行（它此前被 `#` 检查提前拒绝）。

#### ④ 指标自身的 4 个假阳性

`x'`（撇号）被 MathML 编成 `msup`、`\_`（转义下划线）被当成下标、
`\binom` 被当成 `\frac`、`\text{10^0}` 里的 `^` 被当成上标。
逐个修正后**剩 1 处**（`\text{10^0}`，是真实的微小不一致）。

### 16.3 结果

| 指标 | 修复前 | 修复后 | 目标 |
| :--- | ---: | ---: | ---: |
| 公式 AST 一致率 | **未测量** | **0.9995** | ≥ 0.99 ✅ |
| transpiler 拒绝（降级为文本） | **93** | **1** | — |
| 可比对公式数 | — | 2027 | — |

**测试基线**：`pytest -m fast` **3050 passed**。

---

## 17. 最后两项：术语覆盖与公式 AST（2026-10-12）

### 17.1 §9 第 6 项：术语覆盖从 10/20 到 20/20

**根因比"覆盖不足"严重得多**：`fast_lane` 门控了 bible 的**每一个**步骤——
0-token 挖掘、LLM 骨架抽取、缩写回填全部跳过。于是短路由**没有任何术语
机制**：不是"一致性差"，是**根本没有**。更糟的是 `deterministic_glossary`
（短路由**开启**）拿到的是一张空表，**它的强制器没有任何规则可强制**。

实测 `vit`：短路由 **0 个可测术语**，长路由 28 个（0.9545）。

修复：三项全部解禁。成本是**一次**骨架调用（几千字符，只调一次），不是
逐块调用——短路由真正的节省来自 batch API，与 bible 无关。

| 口径 | glossary | 可测术语 | precision | 成本 |
| :--- | ---: | ---: | ---: | ---: |
| 短路由（修复前） | **0** | **0** | — | 32s |
| 短路由（修复后） | 31 | 30 | **0.9791** | 259s |
| 长路由（对照） | 29 | 28 | 0.9545 | 284s |

**短路由现在比长路由更准且更省**——它此前只是没有术语机制。

### 17.2 §9 第 5 项：公式 AST 一致率从"未测量"到 0.9995

见 §16。**新增测量工具，并因此修掉 4 类真 bug**（`\mathbf` 让整式失效
105 处、下标吞函数名 6 处、93 个公式因符号缺失降级、指标自身 4 个假阳性）。

### 17.3 最终对账（RFC §9 八项）

| # | 指标 | 目标 | 实测 | 判定 |
| :-: | :--- | :--- | :--- | :--- |
| 1 | 中文占比（正文） | ≥ 55% | 仅可译块中位 0.712，19/20 | ✅ |
| 2 | 未翻译块 | ≤ 5% | max 0.003；`untranslated_marks` 全 0 | ✅ |
| 3 | 表格中文化率 | ≥ 95% | 表块翻译率中位 **1.000**，12/18 | ⚠ |
| 4 | 字号 P95 | ∈ [10,11]pt | **10.5（20/20）** | ✅ |
| 5 | 公式 AST 一致率 | ≥ 99% | **0.9995（2027 式）** | ✅ |
| 6 | 术语一致率 | ≥ 0.90（tier B） | 短路由 **0.9791**，20/20 可测 | ✅ |
| 7 | 端到端交付率 | ≥ 95% | **20/20 = 100%** | ✅ |
| 8 | 渲染耗时 | < 2s/篇 | 中位 0.87s | ✅ |

**8 项中 7 项达标；第 3 项仍有真实缺口**（表头 CJK 覆盖 0.614，其中约 28%
是必须逐字保留的数据集名）。

### 17.4 一条贯穿全会话的规律（第五次）

**五次同类错误，都是"分母/定义域与被测量的决策不对齐"**：

1. `cjk_ratio` — 含故意保留的块
2. 表格指标 — **反转**，越正确越低
3. `cover_mode`/`table_render_mode` — 度量不存在的东西
4. 术语指标 — 含故意保留的块
5. `fast_lane` — 门控了**零成本**的工作，把"省一次调用"变成了"没有机制"

**测试基线**：`pytest -m fast` **3050 passed**。
