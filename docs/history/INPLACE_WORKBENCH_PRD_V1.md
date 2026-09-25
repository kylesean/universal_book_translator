# PRD V1: 原位翻译预览工作台 (In-Place Translation Workbench) + 保版导出技术方案

> **⚠️ SUPERSEDED（2026-09-22 归档）。** 本 PRD 的 pdf_oxide destructive-redact 主路线**从未实现**，
> 桌面工作台（Tauri/pdf.js/审校写回）零行代码落地。它要解决的"保版导出"需求已由另一条路线交付：
> `rigid` 引擎（pikepdf 内容流删字 + Typst 译文叠层 + add_overlay 合成），定案过程见
> [docs/LAYOUT_PRESERVATION_MASTERPLAN_2026-09.md](../LAYOUT_PRESERVATION_MASTERPLAN_2026-09.md)，
> 实现见 `ubt/adapters/pdf/rigid/`。照本文开发会与 license guard（禁 PyMuPDF）及现役引擎直接冲突。
> 保留价值：§2 用户场景与 FR-1~16 需求清单仍是未来桌面工作台的唯一成文需求资产。
- 状态: ~~待评审~~ → 已归档（原 pdf_oxide 路线被 masterplan 改道为 rigid 路线）
- 关联: INPLACE_TECH_SURVEY_2026.md（选型证据）· INPLACE_WORKBENCH_TECH_DESIGN_V1.md（同归档）
  · TUI_V2_PRD.md · atelier_ui_and_publishing_suite_prd.md (memory) · PDF_AGENT_SKILLS_VS_UBT.md（该文档已删除）
- 日期: 2026-09-19
- 目标形态: Rust + React/Vue 桌面版 (Tauri) 的关键产品能力

---

## 0. 决策摘要 (TL;DR)

> **V1.1 修订（2026-09-19）**：S1/S2 spike 已完成，保版导出主路线由 pikepdf 改为
> **pdf_oxide**（实测验证），详见 INPLACE_TECH_SURVEY_2026.md。

1. **工作台 ≠ pdf2htmlEX**。D 路线的本质是"活 DOM 覆盖层"，不是那个 2019 年停更的 GPL 转换器。
   现代实现基座是 **pdf.js (Apache-2.0)**：canvas 渲染原件 + text layer 提供 span 级几何 +
   译文 overlay 层绝对定位覆盖。与 Tauri/React 技术栈天然契合，零新增 GPL/AGPL 依赖。
2. **保版导出主路线 = pdf_oxide（MIT OR Apache-2.0，已实测）**：真 destructive redaction
   （Type1 与 Type0/Identity-H 均通过，内容流零字节残留）、span 级提取、CJK 嵌入+子集化、
   `text_in_rect` 框内自动换行、页面渲染，官方 PyPI 绑定。pikepdf (MPL) 降级为
   overlay 合成工具（`add_overlay`）。**V1 初稿"pikepdf 有 redact API"的假设已被证伪**；
   qpdf 无 `--redact`、pdf-redactor(CC0) 对现代 PDF 失效——三条宽松 redact 路线中仅
   pdf_oxide 存活。PyMuPDF 商业授权保留为**逃生门**（经 adapter 收敛，30 分钟可切换）。
3. **预览与导出必须解耦**（这是本 PRD 最重要的架构判断）：
   - 预览层 = pdf.js canvas + DOM overlay（所见即所得，交互丰富）
   - 导出层 = 独立 PDF compositor（矢量、文字可选、字体嵌入）
   - 两者共享同一 IR + 几何映射 + 事件契约。"预览即导出"（RT 模式）在 pdf.js 下不成立，
     因为 canvas 打印出来是位图——见 §7.3 的诚实分析。
4. **质量预期管理**：redaction+overlay 的天花板是 Adobe/DeepL 级（很好，但不完美）。
   译文膨胀时 bbox 内重排必然产生字号缩放/断行差异。"出版级"由重排管线 (Typst) 负责，
   "保版级"由 compositor 负责——两种交付形态并存，用户按场景选。
   2026 业内共识方向 = **双向 IR + 自适应排版 + 视觉/置信度闭环**，本 PRD 已吸收为
   FR-8（迭代缩放搜索+budget prompt）、FR-13（ABBOR 指标）、FR-14（置信度路由）、
   FR-15（VLM 巡检）。

---

## 1. 背景与产品定位

### 1.1 现状
- UBT 已具备：出版级重排输出 (publication preset + Typst)、DocumentIR (block 级 bbox 已有,
  `ubt/core/ir/models.py:158`)、公式隔离 (`$$..$$` skip_translate + skeleton 不变量)、
  ledger 幂等续传、QE/repair 闭环。
- 缺口：输出物是"新书"，不是"那份文档"。用户无法在等待翻译时看到效果，无法逐段干预。

### 1.2 两个新能力，一个产品叙事
| 能力 | 定位 | 交付阶段 |
|---|---|---|
| 原位预览工作台 | **信任入口**：所见即所得、逐段试译、人工审校 | 桌面版 M2-M4 |
| 保版导出 | **交付形态**：in-place PDF，与原件同构 | 与工作台并行，M4-M5 |

### 1.3 与竞品的差异点
- Adobe/DeepL：有保版翻译，无"翻译过程可视化"，无术语/QE/repair 引擎。
- PDFMathTranslate 系：图像流保版，文字不可选，无交互。
- RockTranslate：有交互工作台（借鉴对象），但引擎单薄（串行、无 QE、无 IR、无测试体系）。
- **UBT 的牌**：最强引擎 × 现代工作台。引擎与 UI 之间用事件契约解耦，
  同一契约服务 CLI/TUI/Web/桌面四个渲染端。

---

## 2. 用户与核心场景

- **学术读者**：拿到英文论文/教材，想要"还是那份 PDF"的中文版，公式图表不能坏。
- **译者/审校**：逐段试译、人工改判、术语强制，改完直接导出。
- **企业本地化**：批量跑书/手册，交付形态要求保留原排版（品牌合规）。

三个必须丝滑的场景：
1. **试译**：选中任意段落 → 3 秒内原位看到译文，其余版面纹丝不动。
2. **整篇**：后台批量翻译，页面逐段点亮，随时停止/回滚，断点续译 0 重复 token。
3. **审校**：hover 看原文、点击编辑译文、accept 后写回 IR，导出时人工判定优先于机器。

---

## 3. 核心架构：三层预览模型

```
┌─────────────────────────────────────────────┐
│ L3 Overlay 译文覆盖层 (DOM, React 组件)      │  ← 译文 div 绝对定位 / 白底遮罩 /
│    segment ↔ IRBlock ↔ overlay rect 三方映射 │     contenteditable 审校 / 字号自适应
├─────────────────────────────────────────────┤
│ L2 Text Layer (pdf.js 内置, span 级几何)     │  ← 原文选择/命中测试/hover 对照
├─────────────────────────────────────────────┤
│ L1 Canvas (pdf.js 渲染原件, 像素级保真)      │  ← 公式/图表/装饰零成本保留
└─────────────────────────────────────────────┘
```

### 3.1 为什么是 pdf.js 而不是 pdf2htmlEX

| 维度 | pdf2htmlEX (RT 现状) | pdf.js (本方案) |
|---|---|---|
| License | **GPLv3**（UBT 零 AGPL/GPL 守卫直接冲突） | **Apache-2.0** ✅ |
| 维护 | 2019 停更，Windows 只有 0.14.6 老二进制 | Mozilla 官方持续维护 ✅ |
| 转换成本 | 全文档一次性转换（分钟级，需切片+合并引擎） | 逐页流式渲染，天然懒加载 ✅ |
| 公式/图表 | 转 HTML 有损（字体子集/裁剪/透明度坑） | canvas 像素级原样 ✅ |
| 可编辑 DOM | 全文档都是 DOM（RT 的"预览即导出"基础） | 只有 overlay 是 DOM ❌（导出需独立 compositor） |
| 几何精度 | matrix 类 CSS（span 级） | textContent transform 矩阵（span 级）✅ |

结论：pdf2htmlEX 唯一不可替代的优势是"预览即导出"，而该优势在导出质量上反而是妥协
（浏览器打印引擎的字体替换问题同样存在）。UBT 选择解耦，两边都拿最优。

### 3.2 几何映射链

```
IRBlock.bbox (pt, 来自 Docling/PDF adapter)
    ↕ 页码 + 缩放一致性 (pt → px: scale = viewport.scale, DPI 无关坐标系)
pdf.js textContent.items[].transform (span 级, 用于命中测试与行级细化)
    ↕
overlay div: { position:absolute; left/top/width/height: 由 bbox 乘 scale 得出 }
```

规则：
- 所有几何在 **PDF pt 空间**存储与计算，渲染时才乘 scale（解决 FR-12 缩放一致性）。
- block 级 bbox 是权威（IR 单一事实源）；span 级几何仅用于 hover 命中与原文选择，
  **不回写 IR**（避免几何数据双主）。
- 旋转页 (page.rotate ∈ {90,180,270})：bbox 需经旋转矩阵变换后映射（spike 验证项）。

### 3.3 公式与图表的"免费"优势
- `BlockType.FORMULA` + `skip_translate=True` 的块：L3 不生成 overlay →
  用户看到的公式就是 L1 canvas 上的**原始矢量渲染**，零失真零 token。
- 这与 RT 的"几何隔离"殊途同归，但 UBT 是**声明式**的（IR 块类型驱动），
  不依赖启发式旁路——方法论延续 skeleton 不变量的确定性传统。
- 行内公式：延续现有 `\text{}` span 抽取 + skeleton 校验，校验失败的块在 overlay
  上标记 `formula_kept_source`，UI 显示角标提示（不静默降级）。

---

## 4. 状态机与事件契约（平台无关核心）

### 4.1 Job 状态机

```
idle → preparing → translating → done
         │             │  ╲
         └─────────────┘   → cancelled → (可 resume) translating
任意状态 → failed (带 error_code, 可 retry_from_checkpoint)
```

不变量（铁律，违反即 bug）：
- **单一完成信号**：done/cancelled/failed 恰好触发一次（try/finally 保证，借鉴 RT）。
- **取消收敛**：cancelled 后所有半途页回滚到该页最后完整状态，≤1s 内完成。
- **幂等续传**：resume 只消费 ledger 中 status=translated 的块，0 token 重复。

### 4.2 Segment (IRBlock) 状态

```
pending → translating → applied → accepted(人工确认)
              │            │
              └→ failed ────┴→ reverted (回滚/hover 还原)
```

### 4.3 事件契约 v1（EventBus，JSON 可序列化，带 schema_version）

| 事件 | payload | 生产者 | 消费者示例 |
|---|---|---|---|
| `job.started` | {doc_id, total_pages, total_blocks} | orchestrator | 进度面板初始化 |
| `prep.progress` | {current, total, stage} | 解析/OCR 阶段 | 状态栏 |
| `segment.applied` | {block_id, page, source, target, bbox, overflow_ratio} | 翻译 worker | L3 overlay 点亮 |
| `segment.failed` | {block_id, page, error_code} | QE/repair 耗尽 | 红色角标 |
| `job.speed` | {blocks_per_min, eta_seconds} | 节流计算 | ETA 显示 |
| `job.cancelled` | {reverted_blocks[], kept_blocks[]} | cancel handler | 回滚动画 |
| `job.finished` | {stats: applied/failed/kept_source} | finally | 完成 toast |
| `review.edited` | {block_id, new_target, editor} | 前端审校 | 写回 IR + ledger |

规则：
- 事件是**唯一**的引擎→UI 通道。UI 不得轮询引擎内部状态（RT 的 evaluate_js 直调
  全局函数模式是反面教材，契约化后 Tauri `emit/listen` 直接映射）。
- payload 一律 JSON 序列化（禁止字符串裸拼——RT 技术债教训）。
- CLI/TUI 是同一事件的另一个消费者：渲染为进度行。

---

## 5. 功能需求 (FR)，每条带验收标准

- **FR-1 页面懒加载**：打开 300 页书，首屏 ≤ 1.5s（仅渲染 ±2 页，页虚拟化）。
  Given 大 PDF, When 打开, Then 仅当前页 canvas 存在 DOM，滚动时按需挂载/卸载。
- **FR-2 段落试译**：选中任一 block 边界（L2 命中测试），右键"试译此段"，
  3s 内（P50）该段 overlay 显示译文，其余零变化。失败则角标 + 原因 tooltip。
- **FR-3 流式整篇**：批量翻译中 `segment.applied` 到达即点亮，
  首段可见延迟 ≤ LLM 首响应 + 200ms。
- **FR-4 hover 对照**：修饰键+悬停 overlay → 半透明显示 L1 原文（原文本来就在 canvas，
  零请求）；悬停 L2 原文 span → 显示对应译文（若有）。
- **FR-5 停止与回滚**：任意时刻停止，≤1s 响应；半途页还原，已完成页保留（可配置）。
- **FR-6 断点续译**：关闭重开同一文档（SHA-256 识别），已译块直接点亮，0 API 调用。
- **FR-7 人工审校**：overlay contenteditable，编辑后 `review.edited` 写回 IR + ledger；
  导出时人工版本优先于机器版本；提供"还原机器译文"。
- **FR-8 自适应排版（V1.1 升级，采纳 BabelDOC 方案）**：三级策略——
  ① 翻译时把**每 bbox 字符预算注入 prompt**（"译文 ≤N 字符"，HOMURA 思路），源头控膨胀；
  ② 超框时**缩放因子迭代搜索**（从 1.0 起以 0.05/0.1 步长收缩至装下或触底，
  非等比 0.95 递减；遵循 DeepL 四级约束优先级：同页>版面位置>字号比例>全文一致）；
  ③ 仍超则 `overflow_ratio>1` 上报，UI 黄标警示"此段挤压，建议审校"——**不静默压扁**。
- **FR-13 保版度量化（ABBOR）**：导出后自动计算每块"译文框 vs 原框"的 BBox IoU /
  DeepL 式 ABBOR 指标，低于阈值（默认 0.8）的块自动进人工审校队列。
  验收：arXiv 基准集回归报表含 ABBOR 分布直方图。
- **FR-14 置信度路由（Foxit/Straker 模式）**：QE 分数（UBT 已有）驱动分流——
  高置信块自动过，低置信块在 overlay 上蓝标"建议复核"。审校队列按置信度升序排列。
- **FR-15 VLM 版面巡检（P2，PaperFit 思路）**：导出前渲染页面位图 → VLM 按
  溢出/遮挡/错位/色不匹配 四类缺陷巡检 → 缺陷块挂进审校队列。默认关闭（token 成本），
  publication/inplace 交付档位可开启。
- **FR-16 颜色保真（S2 spike 发现，必须实现）**：redaction 填充色 = **bbox 周边背景主色采样**
  （pdf_oxide 默认纯黑，实测在彩色页眉上产生黑块事故）；译文文字颜色**继承原文**
  （提取时记录，写入时还原，实测白字标题被黑字译文覆盖后不可见）。
- **FR-9 视图模式**：overlay(覆盖) / dim(原件淡化+译文浮层) / side-by-side(双栏对照)
  三态切换，快捷键绑定。
- **FR-10 导出**：一键调用 §7 compositor，导出完成后 toast 附文件路径 +
  与预览的一致性 diff 报告（哪些块在导出时发生了字体替换/字号变化）。
- **FR-11 公式保真**：FORMULA 块永不生成 overlay；行内公式 skeleton 校验失败块
  保留原文并角标。验收：arXiv 基准集 33 公式块导出后与原件逐像素一致。
- **FR-12 缩放一致性**：100%~400% 缩放下 overlay 与 canvas 无漂移（pt 空间统一换算）。

## 6. 非功能需求 (NFR)

- 取消响应 ≤ 1s：所有阻塞点（LLM 等待/退避倒计时/批量写回）轮询 cancel 标志。
- 大文档内存：页虚拟化，DOM 常驻 ≤ 5 页；IR 全量常驻（300 页书 IR ≈ 数 MB，可接受）。
- 单一事实源：前端全局 store 仅 {running, stage, current, total} + segment 状态表，
  组件禁止私存进度副本（RT commit 22602ad 的教训）。
- 事件契约版本化：schema_version 字段，前后端可独立发版灰度。
- 桌面版资源：Tauri 壳 ≤ 10MB（对比 Electron），Python 引擎以 sidecar 进程运行。

---

## 7. 保版导出技术方案（深入调研结论 + 决策矩阵）

### 7.1 先回答"PyMuPDF 能否彻底解决"

**不能，两个层面都不能"彻底"：**

1. **协议层**：PyMuPDF = AGPL-3.0/商业双许可。对 MIT 的 UBT：
   - 开源版捆绑 = AGPL 传染（即使 subprocess 隔离也处于灰色地带，MuPDF 官方明确
     要求网络服务/分发场景购买商业授权）；
   - 商业授权按开发者年费计价（对个人/小团队可负担，是**兜底出口**而非主路线）；
   - 竞品基线：Babeldoc、PDFMathTranslate 全部 AGPL——UBT 若也 AGPL 依赖，
     "MIT + 零 copyleft 守卫"的差异化资产就没了。
2. **质量层**：redaction+overlay 路线（无论 PyMuPDF 还是替代者）的天花板是
   **Adobe/DeepL 级**："很好但不完美"——bbox 内重排译文必然引入断行差异、字体替换、
   字号缩放。出版级阅读体验仍归 Typst 重排管线管。指望单一技术"彻底解决"是
   目标错位；正确目标是**分场景达标**（见 §7.4 矩阵）。

### 7.2 导出 compositor 候选栈（决策矩阵，V1.1 按 spike 实测修订）

| 方案 | 核心依赖 | License | 真 redact | 矢量+可选字 | 断行/字体工程 | 判定 |
|---|---|---|---|---|---|---|
| **pdf_oxide 管线** ✅已实测 | Rust+PyO3 | **MIT OR Apache-2.0** | ✅ Type1+Type0/Identity-H 均真删（内容流零残留，见调研 §3.1） | ✅ | `text_in_rect` 框内换行 ✅ + CJK 嵌入子集化 ✅；缺"写入既有页"→ overlay 合成绕 | **主路线（V1.1 定案）** |
| pikepdf（仅合成） | QPDF | MPL-2.0 | ❌ **无任何 redact API（V1 假设证伪）** | ✅ | — | 降级为 `add_overlay` 合成工具 |
| PyMuPDF | MuPDF | **AGPL/商业** | ✅ | ✅ | 内置 | **逃生门**：商业授权，adapter 收敛 30 分钟可切换 |
| PDFBox sidecar | Apache | **Apache-2.0** | ✅ (pdfredact) | ✅ | 内置较好 | 次选（pdf_oxide 上游崩塌时启用）；JVM 产品税 |
| qpdf CLI | Apache | ❌ **无 --redact**（12.4.1 实测+ChangeLog 证伪） | — | — | 弃用 |
| pdf-redactor | CC0-1.0 | ❌ pdfrw 无法解析 Typst 压缩 xref PDF（实测） | — | — | 弃用 |
| pdfium 直改 | BSD-3 | ❌ 编辑面弱 | ✅ | — | 只做渲染参考 |
| 浏览器打印 (RT 模式) | — | n/a(重绘) | ⚠️ canvas 部位图化 | 浏览器代劳 | **不适合 UBT 导出**（见 7.3） |
| 商业 SDK (Nutrient/Foxit) | — | 💰 | ✅ | ✅ | 内置 | 企业版选项 |

### 7.3 关键诚实分析：为什么"预览即导出"在 pdf.js 下不成立

RT 能打印预览 DOM 是因为 pdf2htmlEX 把**整个原件**转成了 DOM（文字是真文字节点）。
pdf.js 的原件是 canvas（位图）：打印预览 = 原件变图 = 文字不可选 = 交付降级。

**因此导出必须是独立 compositor**，两种可行合成策略：

- **策略 A（推荐，V1.1 已实测打通）：矢量合成**——原 PDF 逐页处理：
  1. pdf_oxide `add_redaction(rect, fill=背景采样色)` + `apply_redactions_destructive()`
     真删原文（glyph 级，Type1/Type0 均验证）
  2. pdf_oxide DocumentBuilder 生成同尺寸**透明 overlay 页**：`register_embedded_font(CJK)`
     + `text_in_rect(bbox)` 自动换行 + FR-8 缩放搜索 + 原文颜色继承，译文写入
  3. pikepdf `add_overlay` 把 overlay 页合成回原页
  4. 公式/图表/装饰**原字节保留**（redaction 只碰目标 rect 内的文本对象）
  输出：矢量、可选字、可搜索、体积不膨胀。质量 = Adobe 级。
  （端到端证据：chapter-3.pdf "CHAPTER 3"→"第三章"，见 INPLACE_TECH_SURVEY_2026.md §3.2）
- **策略 B（降级兜底）**：白遮罩 + 覆盖绘制（不 redact）。实现简单但原文仍在
  文件里（可被复制/搜索到）——只用于内部预览稿，禁止作为交付。

### 7.4 分场景交付矩阵（产品话术级）

| 场景 | 推荐出口 | 质量承诺 |
|---|---|---|
| 出版级阅读（书/教材） | 现有 Typst 重排 | 出版级（现状保持） |
| 保版交付（论文/手册/合规文档） | 策略 A compositor | Adobe 级：版面同构，译文区可能有字号缩放，黄标块建议审校 |
| 快速预览稿 | 工作台截图/打印 | 预览级 |

### 7.5 事实核查 Spike 状态（V1.1 更新）

- [x] **S1 PASS**：pikepdf **无 redact API**（假设证伪）；qpdf 无 --redact；pdf-redactor 失效；
      **pdf_oxide destructive redact 真删**（Type1 + Type0/Identity-H，内容流零残留）。
- [x] **S2 PASS**：pdf_oxide `text_in_rect` 框内自动换行 + NotoCJK 嵌入 + pikepdf add_overlay
      端到端合成成功，渲染目视确认。发现 FR-16（填充色/字色保真）两个必须项。
- [x] ~~S3~~ **设计定案**（TDD §5.4）：IRBlock.bbox 为唯一权威源，pdf.js textContent 只做
      命中测试不参与定位；对齐误差转为线上遥测指标（geom_suspect 角标），不设上线门槛。
- [x] ~~S4~~ **设计定案**（TDD §5.5）：stdio JSON-lines + 2s 心跳重启 + seq 环形缓冲重放；
      开发期 sidecar 以 SSE 独立运行，前端脱离 Tauri。
- [x] ~~S5~~ **设计定案**（TDD §3.4）：锁版本 + PdfOps Protocol 单点收敛 + CI import 守卫 +
      PyMuPDF 同接口逃生门；升级必跑 §8.2 黄金集。
- [x] ~~S6~~ **设计定案**（TDD §5.1）：BabelDOC 式离散缩放搜索 [1.0→0.6] + 行宽启发式，
      触底走黄标（FR-8 ③）；DeepL 四级约束以"单块独立搜索+全局仅告警"实现。
- [x] ~~S7~~ **设计定案**（TDD §5.2）：页级渲染缓存 + bbox 外扩边框环带 + 4bit 量化众数取色。

---

## 8. 架构集成（UBT 侧接缝）

```
ubt/core/events.py            # EventBus + 契约 dataclass (schema_version)
ubt/core/engine/pipeline.py   # orchestrator 在状态迁移处 publish()（ledger 已有 hook 点）
ubt/adapters/pdf_oxide/       # vendor 适配层: 收敛 pdf_oxide 全部调用 (锁版本, 逃生门切换点)
ubt/core/composers/inplace.py # 策略 A compositor (pdf_oxide redact/overlay + pikepdf 合成)
ubt/desktop/ (新仓库或子包)   # Tauri: src-tauri/(Rust shell+sidecar) + web/(React+pdf.js+overlay)
```

- 引擎保持 headless：compositor 与工作台都是 IR/事件的消费者，**零核心重构**。
- 事件桥：Python sidecar → stdout JSON lines → Rust 反序列化 → `app.emit("segment-applied", ..)`
  → React `listen()`。契约单测放引擎侧（事件序列快照测试）。
- 审校写回：`review.edited` → sidecar RPC → IR block.target 更新 + ledger 记人工版本
  （status=accepted，QE/repair 不得覆盖人工判定——与现有 added_content_gate 同族守卫）。

## 9. 里程碑

- **M1 (1 周) Spike 收尾**：~~S1~~~~S2~~ 已完成 ✅；剩 S3-S7 + pdf.js overlay 静态原型（无引擎，手贴译文）。
- **M2 (2 周) 只读工作台**：打开 PDF → 逐页预览 → 段落试译 → overlay 显示。
- **M3 (2 周) 流式批量**：整篇翻译 + 进度/停止/回滚 + 断点续译。
- **M4 (3 周) 审校 + 保版导出**：contenteditable 审校 + 策略 A compositor + 一致性 diff 报告。
- **M5 (评估) 服务端批量保版**：无浏览器依赖的纯 Python 批处理（pdf_oxide compositor 产品化，含 ABBOR 回归门）。

## 10. 风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| pdf_oxide 过年轻（10 个月、83 版、240 open issues、API 震荡） | 高 | **锁版本 + vendor 适配层单点收敛**（`ubt/adapters/pdf_oxide/`）；PyMuPDF 商业授权逃生门；PDFBox 次选 |
| pdf_oxide "Dictionary used where Stream expected" 告警（Typst PDF） | 中 | 功能未受影响但需监控；纳入 S5 回归用例；上游提 issue |
| redaction 填充/字色不保真导致交付事故 | 中 | FR-16 硬性要求 + S7 采样算法 + 导出后 ABBOR/巡检双门 |
| 自建缩放搜索/断行策略工程量 | 中 | S6 原型先行；BabelDOC 开源实现可参考（注意其 AGPL——只读思路不抄代码） |
| 自建断行/字体排字工程量失控 | 中 | S2 原型先行；限制 FR-8 策略（缩放优先于断行重排） |
| pdf.js 几何对齐误差 | 中 | S3 验证；误差 >2pt 时以 Docling span 级数据自校正 |
| 用户预期错位（把保版当出版） | 中 | §7.4 矩阵进产品文案；黄标块机制把不完美透明化 |
| Tauri sidecar 分发（Python 引擎打包） | 低 | PyInstaller 成熟方案；后期可评估编排层 Rust 化 |

## 11. 参考来源

- [PikePDF - GitHub](https://github.com/pikepdf/pikepdf) / [pikepdf Documentation](https://pikepdf.readthedocs.io/) / [pikepdf - PyPI](https://pypi.org/project/pikepdf/8.15.1/)
- [A deep dive into PDF.js layers and how to render truly interactive PDFs (Reddit, 2025)](https://www.reddit.com/r/reactjs/comments/1lvgx6j/a_deep_dive_into_pdfjs_layers_and_how_to_render/)
- [We built a private PDF text editing SDK on top of PDF.js (掘金, 2026)](https://juejin.cn/post/7601018079619465231)
- [PDF.js tutorial: A complete guide with examples - Nutrient](https://www.nutrient.io/blog/complete-guide-to-pdfjs/)
- [How to edit a PDF in Python: Add text, images, and annotations - Nutrient blog](https://www.nutrient.io/blog/edit-pdf-python/)
- [Use span instead of div for text layer · mozilla/pdf.js #10156](https://github.com/mozilla/pdf.js/issues/10156)
- [Add Javascript calls to the text layer over a pdf.js viewer (SO)](https://stackoverflow.com/questions/60823850/add-javascript-calls-to-the-text-layer-div-over-a-pdf-js-viewer-canvas)
- 内部：RockTranslate 逆向剖析（本会话，2026-09-19）；UBT memory: p3_academic_benchmark_completion.md
