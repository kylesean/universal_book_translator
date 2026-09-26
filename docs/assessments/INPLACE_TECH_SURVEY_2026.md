# 保版翻译技术调研 2026：论文综述、库评估与 Spike 实验报告

- 状态: Final (S1/S2 实验闭环；S3/S4 待 M1) — **2026-09-22 勘误见下**
- 日期: 2026-09-19
- 关联: [docs/history/INPLACE_WORKBENCH_PRD_V1.md](../history/INPLACE_WORKBENCH_PRD_V1.md)（已归档）· PDF_AGENT_SKILLS_VS_UBT.md（已删除）
- 方法: 并行 web 检索代理 × 2 + 本地 spike 实验（venv `/tmp/spike_venv`，证据文件 `/tmp/e2e*.pdf|png`）

> **2026-09-22 跟进勘误。** ① 本调研判定的主路线（pdf_oxide destructive redact + text_in_rect）
> **最终未采纳**：保版导出实际走 pikepdf stream-strip + Typst overlay 的 `rigid` 路线，
> pdf_oxide 在仓内只承担渲染/提取/探测（`ubt/adapters/pdf/oxide_render.py`），定案过程见
> [LAYOUT_PRESERVATION_MASTERPLAN_2026-09.md](../design/LAYOUT_PRESERVATION_MASTERPLAN_2026-09.md)。
> ② 版本号至今未漂移：`pdf_oxide 0.3.78`、`pikepdf >=10.13` 与 `uv.lock`/`pyproject.toml` 一致，
> "pikepdf 无 redact API"的更正仍成立；`<0.4` 封顶与"升版前必须 dir() 重探"的告诫仍然有效。
> ③ 第一/二部分的论文综述（BabelDOC 方法论、ABBOR/BBox IoU、HOMURA 预算思路）仍是
> `scripts/biou_score.py` 与 `ubt/adapters/pdf/render_fidelity.py` 的思想出处，继续有效。
> ④ spike 证据全在 `/tmp`，样本 `docs/chapter-3.pdf` 已不在仓库，第二部分实验不可复现。

---

## 第一部分：业内最新论文与产品实践（2024–2026）

### 1.1 没有出现"第三范式"——共识在收敛，不在颠覆

对 2024-2026 学术与工业界的系统检索显示：**redaction+overlay（B 路线）与 HTML/IR 中间态（D 路线）
仍是仅有的两条工程可行路线**，但两者的内部方法论发生了质变，共识方向 =
**双向 IR + 自适应排版 + 视觉/置信度闭环**。

### 1.2 关键论文

| 论文 | 核心贡献 | 对 UBT 的意义 |
|---|---|---|
| **BabelDOC: Better Layout-Preserving PDF Translation via Intermediate Representation** (arXiv:2605.10845, 2026-05) | 双向 IR（视觉布局元数据 ⊥ 语义内容）；**自适应排版引擎**：每段从缩放因子 1.0 起以 0.05/0.1 步长迭代收缩直到装进原 bbox；嵌套 CTM 矩阵重建保旋转/缩放一致；评测用 **BBox IoU + 多模态 LLM-as-judge**（200 页基准） | ①"迭代收缩搜索"应替换我们 PRD 里 0.95 递减的朴素策略；②BBox IoU 可直接做成自动回归指标 |
| **PDFMathTranslate** (EMNLP 2025 Demo, arXiv:2507.03009) | DocLayout-YOLO 版面检测 + 拆分 + 重渲染全管线，附与 Doc2X/Google/DeepL 保版对比表 | 37k stars 项目路线背书：仍在 redaction/overlay 框架内演进 |
| **PaperFit** (arXiv:2605.10341) | 正式提出 *Visual Typesetting Optimization*：agent 循环"渲染 → VLM 诊断 13 类版面缺陷 → 受约束修复"，附 PaperFit-Bench | **"平时自动修、出错人工审"的学术对应物**——VLM 巡检可作为导出前质量门 |
| **VFLM** (arXiv:2603.22187) | 视觉接地奖励 + OCR 准确性做 RL，布局模型反思渲染结果迭代自纠 | 远期参考（需训练预算） |
| **HOMURA** (arXiv:2601.10187) / COLING 2025 跨语言句子压缩 | token/时间预算约束 LLM 输出长度 | **把"每 bbox 字符预算"注入翻译 prompt**——从源头减少膨胀，缩字只作兜底 |
| Marathi 政府文档管线 (arXiv:2606.28796) | 坐标约束提取 → LLM 翻译 → HTML 重建 | "HTML 中间态"路线的学术印证 |

### 1.3 商业方案现状（2026）

- **DeepL**（官方博客 2025-09）：PDF→**DOCX 中间态**→回渲染。自研质量指标
  **ABBOR**（Average Bounding Box Overlap Ratio，SSIM 被其否决）；定下**四级约束优先级**：
  同页 > 版面位置 > 字号比例 > 全文字号一致；承认纯数学缩放因子因字号离散化失效。
  第三方 2026 评测：多栏/表格保版仍弱（Lara 对比给排版 2/5）。
- **Google Cloud 文档翻译**：保版垫底（多栏被压平，1/5）。Adobe Acrobat 同为重排式。
- **Foxit + Straker**（2026-06）：按 element ID 抽取-翻译-回渲染，**每段返回置信度分数**
  驱动"高置信自动过、低置信转人工"路由——与我们审校 UI 的触发机制完全同构。
- **BabelDOC**（funstory-ai，9.6k stars，沉浸式翻译团队）：上述 IR 论文的开源实现，
  已被 pdf2zh-next 集成为 `--mode precise` 后端（2026-03），2026-09 又加入 OCR +
  段落重组 + 自适应排版。**PDFMathTranslate 未换路线但全面向 BabelDOC 靠拢**。

### 1.4 调研结论：五条最值得落地借鉴的技术点

1. **段落级最小缩放因子迭代搜索**（BabelDOC §3.4）+ DeepL 式四级约束优先级，替换朴素等比缩字。
2. **ABBOR / BBox IoU 做成自动回归指标**：每次导出计算"译文框与原框重叠率"，
   低于阈值（如 0.8）的块自动进人工审校队列——把"出版级/保版级"从话术变成可测数字。
3. **每 bbox 字符预算注入翻译 prompt**（HOMURA 思路）：给 LLM 目标长度约束
   （"译文须 ≤N 字符"），源头控膨胀；缩字/断行只作兜底。
4. **渲染后 VLM 版面缺陷巡检**（PaperFit 思路的翻译特化版）：导出前自动跑
   溢出/遮挡/错位/色不匹配 四类检测，产出缺陷报告挂进审校 UI。
5. **段级置信度路由**（Foxit/Straker 模式）：QE 分数（UBT 已有！）直接驱动
   "自动过/人工审"分流——UBT 的 QE 体系天然适配，改造成本最低、收益最直接。

---

## 第二部分：库评估

### 2.1 pdf_oxide（yfedoseev/pdf_oxide）—— 本次调研最大发现

**判定：可用（保版导出主路线），部分能力待自建。当前 Rust 生态最接近 PyMuPDF 的 MIT 替代者。**

| 维度 | 事实 | 来源 |
|---|---|---|
| License | **MIT OR Apache-2.0 双许可**（另有品牌商标条款，不影响代码使用）| crates.io API / raw README |
| 活跃度 | 2025-11 创建，最新提交 2026-09-18（当天）；1036 stars；83 版/10 个月；v0.3.78；crates.io 累计下载 ~97 万 | GitHub/PyPI API |
| Python 绑定 | **官方一等公民**：`pip install pdf_oxide`（PyO3，Python 3.8–3.14，三平台 wheels）——UBT 无需 sidecar | PyPI |
| ① 真 redaction | ✅ v0.3.50 "True destructive redaction"：删文本对象本身（glyph 级），含 text/path/image/font_scrub 管线；**fail-closed 设计**（拒绝无法安全改写的字体而非欠删）| CHANGELOG + 本地实验 |
| ② bbox 排字 | ⚠️ 部分：`text_in_rect`（框内自动换行+对齐）✅、`at()` 绝对定位 ✅、`EmbeddedFont` TTF 嵌入+真子集化 ✅、harfrust 整形 ✅；缺"对既有页面直接写入"的 Python API（Rust 侧 DocumentEditor 有）→ 用 overlay 合成绕（见 §3.3）| 本地实验 |
| ③ span 级提取 | ✅ `extract_words/chars/lines`，TextWord 带 bbox(x,y,w,h)/font_name/font_size/rotation | 本地实验 |
| ④ 渲染 | ✅ `render_page`（tiny-skia，RGBA pixmap，CJK fallback）| 本地实验 |
| 生态对比 | lopdf(MIT) 仅低层对象；printpdf 偏创建无 redact；pdf-writer 只写；qpdf(Apache) **无 redact**（见 §2.3）| README 基准表 + 本地验证 |

**风险登记**：项目仅 10 个月历史、240 open issues、发版极快（API 可能震荡）——
必须**锁版本 + vendor 适配层**（把 pdf_oxide 调用收敛到 `ubt/adapters/pdf/oxide_render.py` 单点；初稿误写作 `ubt/adapters/pdf_oxide/`，该目录不存在 —— 见文首勘误），
且 PyMuPDF 商业授权作为 30 分钟可切换的逃生门。

### 2.2 pikepdf —— 从"主路线候选"降级为"合成工具"

**更正先前 PRD 的错误论断**：pikepdf 10.13 **不存在任何 redact API**
（`Page`/`PageCollection` 均无 redact 属性，实验 §3.1 证伪）。其价值重定位为：
overlay 合成（`Page.add_overlay`，MPL-2.0）与内容流级检查工具。

### 2.3 其他候选

| 库 | License | 结论 |
|---|---|---|
| qpdf 12.4.1 CLI | Apache-2.0 | **无 `--redact` 选项**（ChangeLog 无 redact 记录，实验证伪；社区 redaction 提案未进主线）|
| pdf-redactor (JoshData) | **CC0-1.0** | 概念好（内容流真删文本）但底层 pdfrw 太旧：对 Typst 生成的压缩 xref PDF 直接失效（实验 §3.1），且无 bbox 定位（仅正则全局匹配）。**不采用** |
| PyMuPDF | AGPL/商业 | 能力天花板仍在；定位 = **商业逃生门**（授权按开发者年费，小团队可负担），不是主路线 |
| PDFBox sidecar | Apache-2.0 | redaction 生态成熟（pdfredact），但 JVM 依赖是产品税。pdf_oxide 失败时的次选 |

---

## 第三部分：Spike 实验记录（可复现）

### 3.0 环境

```bash
python3 -m venv /tmp/spike_venv
/tmp/spike_venv/bin/pip install pdf_oxide pikepdf pdfminer.six \
  -i https://mirrors.aliyun.com/pypi/simple/   # 需清除代理环境变量
测试样本: docs/chapter-3.pdf (26页, Type1 字体, 蓝底页眉白字)
```

### 3.1 S1: redaction 能力矩阵 —— 结果 PASS（pdf_oxide）

| 实验 | 方法 | 结果 |
|---|---|---|
| pikepdf redact API | `dir(Page)/dir(PageCollection)` 探查 | ❌ **不存在**（PRD V1 的 §7.5-S1 假设被证伪）|
| qpdf --redact | `qpdf --help` + ChangeLog grep | ❌ 不存在（12.4.1）|
| pdf-redactor | `content_filters` 正则删 "CHAPTER" | ❌ pdfrw 解析 Typst PDF 失败（"No /Root object"）|
| **pdf_oxide destructive** | `add_redaction(page,rect)` + `apply_redactions_destructive()` | ✅ **`{'glyphs_removed': 8, 'bytes_removed': 8}`**；pdfminer 回读 "CHAPTER" 消失；**pikepdf 解剖内容流确认零字节残留（真删非遮罩）**；周围文本零副作用 |
| **Type0 兼容性**（关键） | 对含 Identity-H CIDFontType2 (NotoCJK) 的 PDF 重复实验 | ✅ `glyphs_removed: 3`——**docstring 的 "refuses Type0" 是过时文档**，实际支持（与上游 #748 一致）|

### 3.2 S2: CJK 写入与合成 —— 结果 PASS（含两个产品级发现）

管线（全部宽松许可）：
```
pdf_oxide.PdfDocument ──add_redaction(fill=背景色)──→ redacted.pdf
pdf_oxide.DocumentBuilder ──register_embedded_font(NotoCJK ttc)──→
    .page(w,h).font().text_in_rect(x,y,w,h,text,align) ──→ overlay.pdf (同尺寸透明页)
pikepdf ──pages[i].add_overlay(overlay.pages[i])──→ final.pdf
```

验证：pdfminer 回读"第三章"存在 ✅ / 周围英文完好 ✅ / `render_page(dpi=150)` 目视确认
中文精确落在原 "CHAPTER 3" 位置 ✅。

**产品级发现（必须进 PRD）**：
1. **redaction 默认填充纯黑**——蓝底色带上出现黑块。`add_redaction(fill=rgb)` 可配，
   生产方案 = **采样 bbox 周边背景主色**作为填充色。
2. **译文颜色必须继承原文**——原 "CHAPTER 3" 是白字，overlay 默认黑字在黑块上不可见。
   需要：提取时记录原文颜色（pdf_oxide chars 级/或 pdfminer 色彩通道），写入时还原。
3. pdf_oxide 对 Typst PDF 有 "Dictionary used where Stream expected" 告警（8 次/页），
   功能未受影响，但**健壮性监控项**。
4. API 细节：`TextWord.bbox` 是 **(x, y, w, h)** 非 (x0,y0,x1,y1)；`text_in_rect` 签名为
   `(x, y, w, h, text, align)`；`DocumentBuilder.save()` 无 `to_bytes`。

### 3.3 遗留 Spike 收口（2026-09-19 决定：不再前置验证，转设计定案）

S3-S7 全部以设计定案收口于 **INPLACE_WORKBENCH_TECH_DESIGN_V1.md**
（§5.1 缩放搜索 / §5.2 背景采样 / §5.4 几何对齐 / §5.5 Tauri 桥 / §3.4 版本锁），
其中 S3 对齐误差由"上线门槛"改为"线上遥测指标"。本报告保留实验记录作为选型证据。

---

## 第四部分：修订后的技术选型结论

```
保版导出主路线 (Phase 1):
  pdf_oxide (MIT/Apache, 锁定 v0.3.7x)
    ├─ extract_words/chars     → 几何与原文颜色
    ├─ add_redaction + destructive apply → 真删原文 (Type1/Type0 均验证)
    ├─ DocumentBuilder + text_in_rect    → 译文 overlay 页 (CJK 嵌入+子集化)
    └─ render_page             → 预览/巡检位图
  pikepdf (MPL) ─ add_overlay  → 矢量合成回原页
  UBT 自建 ─ 缩放搜索/背景采样/ABBOR 指标 → 质量闭环

逃生门: PyMuPDF 商业授权 (接口经 adapter 收敛, 30 分钟可切换)
次选:   PDFBox (Apache) sidecar
工作台预览: pdf.js (Apache) 三层模型不变 (见 PRD §3)
```

## 附：来源清单

论文：[BabelDOC 2605.10845](https://arxiv.org/abs/2605.10845) · [PDFMathTranslate 2507.03009](https://arxiv.org/abs/2507.03009) · [PaperFit 2605.10341](https://arxiv.org/html/2605.10341v1) · [VFLM 2603.22187](https://arxiv.org/abs/2603.22187) · [HOMURA 2601.10187](https://arxiv.org/html/2601.10187v1) · [COLING 2025 句子压缩](https://aclanthology.org/2025.coling-main.429/) · [Marathi 管线 2606.28796](https://arxiv.org/html/2606.28796v1)
项目：[pdf_oxide](https://github.com/yfedoseev/pdf_oxide) · [pikepdf](https://github.com/pikepdf/pikepdf) · [pdf-redactor](https://github.com/JoshData/pdf-redactor) · [BabelDOC](https://github.com/funstory-ai/BabelDOC) · [PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate) · [pdf2zh-next docs](https://pdf2zh-next.com/advanced/advanced.html)
商业：[DeepL 技术博客](https://www.deepl.com/en/blog/tech/improving-document-translation) · [Foxit-Straker](https://developer-api.foxit.com/developer-blogs/api-guides-tutorials/pdf-translation-api-confidence-scoring/) · [Lara 2026 评测](https://blog.laratranslate.com/translate-pdf-without-losing-formatting-2026/) · [Google Cloud](https://docs.cloud.google.com/translate/docs/advanced/translate-documents) · [Adobe](https://www.adobe.com/acrobat/resources/translate-pdf.html)
