# RFC-2026-02：UBT 语义优先重构方案（Semantic-First Reflow Architecture）

> **文档标识**：`UBT-RFC-2026-02`
> **状态**：`IMPLEMENTED`（方向经两套基线实证（§0、§13）；§5 的数学转译器已建成并有属性测试与 AST 一致性度量，公式残留率 0.857→0）
> **说明**：方向已经两套基线实证（§0、§13），Sprint 1–4e 核心项全部落地。
> 本文其余章节是设计依据与实测记录，行号/符号为当时快照——如 `ReflowControlLoop`
> 已演进为 `VisualVerifyLoop`，overlay 引擎已整体删除（见 RFC-2026-03 §7–§8）。
> **取代**：`RFC-2026-01` / `RFC-2026-01-Rev2`（"双轨制 + Typst 新引擎"方案）
> **作者**：UBT Core Team
> **生效范围**：`ubt/adapters/`、`ubt/core/ir/`、`ubt/render/`、`ubt/core/engine/`、`cli/`
> **验证产物**：`/tmp/ubt_reflow/重排验证产物.pdf`（29 页）、`/tmp/ubt_reflow/main.typ`

---

## 0. 摘要：一次实证推翻了旧方案的根基

> **实施进度（2026-10-11）：Sprint 1–4d 已完成。**
> ① 基线评测脚本 `scripts/eval_semantic.py` + 金标集 `corpus/semantic_gold/`（20 篇）；
> ② 数值误报修复（实测 16 例中 7 例为纯 bug，已修 + 10 条回归测试）；
> ③ quarantine 改 fail-open（实测 18/18 均为质量判定，17 例为误杀）；
> ④ TOC 改读 PDF 原生书签（实测补回 15 条一级标题）；⑤ `DocumentTree`/章节分组；
> ⑥ 数学转译器搬家 + 三条 P0 修复（公式残留率 **0.857 → 0.000**）；
> ⑦ Section 级切分 + 章节上下文注入 + 标题层级修复（实测产物 40 个 `<h1>` → 真实层级）；
> ⑧ 死代码清理（25 个定义 / 619 行）+ 死代码护栏 + `RUN_ONLY_KEYS` 契约断言；
> ⑨ 语义重排渲染器产品化（`ubt/render/typst/document_renderer.py`，实测抓到 3 个真 bug）；
> ⑩ `--render-mode overlay|publish` 接入 adapter + README 更新；
> ⑪ 全语料扫描：**20/20 篇** `publish` 渲染编译通过，0 丢失资产，字号统一 10.5pt；
> ⑫ 补齐 web 端：`render_mode` 接入 REST/MCP/控制台 + 类型收紧 + 入口一致性测试。
> 全套测试 **fast 3,047 + slow 51 通过**。详见 §13 实施日志。

RFC-2026-01 与 Rev2 都建立在同一个假设上：

> "问题出在**渲染引擎**——把 overlay 换成 Typst 全文档重排就能解决。"

**这个假设是错的。** 我们在同一份文档（`2101.03961v3.pdf`，40 页 arXiv 论文）上做了对照实验，用项目**已有的** `scripts/export_pdf_to_markdown.py` 走语义路径，结果：

| 指标 | 现行 PDF 产物（overlay） | **语义重排产物** |
| :--- | :--- | :--- |
| 中文占比 | 31.1% | **58.0%** |
| 未翻译标记 | 17 处 | **0 处** |
| 表格表头 | 英文 | **中文** |
| 表格渲染 | 保留源语言图层（0% 中文化） | **原生三线表** |
| 正文字号 | 4pt ~ 12pt 抖动 | **统一 10.5pt** |
| 版面空白 | 多栏断裂、大面积空白 | **流式，无异常空白** |
| quarantine 块 | **32 个（10.6%）** | **0 个** |
| 图片 | 盖印在原页 | **13 张独立嵌入** |
| 公式 | 行内剥离错位 | **Typst 数学正确渲染** |
| 页数 | 41（含表格附录） | 29 |
| 渲染代码量 | PDF 适配器 14,927 LOC | Markdown 适配器 324 LOC |

**结论：问题不在渲染引擎，在输入格式的选择。**

PDF 是**打印指令流**（"在坐标 (x,y) 画字形"），不是**文档结构**（"这是一个表格"）。
在绘制指令上做 overlay，等价于在**丢失了语义的字节码**上做语义编辑——这是信息论上不完整的逆问题。overlay 引擎、`Band Reflow`、`repair_*`、`is_inside_picture`、正则 guard、`table_appendix`——**全部是在为这个逆问题打补丁**。

---

## 1. 根本诊断：三个范畴错误

### 1.1 错误一：把 PDF 当作主输入格式

| 格式 | 本质 | 语义 | 翻译后重排 |
| :--- | :--- | :--- | :--- |
| DOCX / HTML / Markdown / EPUB / LaTeX | **文档结构** | 自带 | 天然可行 |
| Born-digital PDF | **打印指令流** | 需逆向推断 | 受限 |
| Scanned PDF | **图像** | 需 OCR | 受 OCR 质量限制 |

项目 87% 的适配器代码（14,927 / 17,181 LOC，33 个专属模块）投入在**唯一一个没有语义结构的格式**上。

### 1.2 错误二：把"版面保真"当作质量目标

"像素级保真"与"内容完整"**不可兼得**：

| 选择 | 得到 | 失去 |
| :--- | :--- | :--- |
| 保真优先（overlay） | 版面一致 | 表格留英文、字号缩到 4pt、图内文字乱翻 |
| **质量优先（重排）** | 内容完整、表格中文、字号统一 | 版面改变（页数/图位置变动） |

对翻译产品而言，**没有任何用户想要"版面完美但表格还是英文"的译本**。

### 1.3 错误三：把"出版级"当作可测量的工程指标

真正的出版流程是 `人工翻译 → 人工审校 → 人工排版 → 人工校对`。**每一环都是人。** 2026 年的行业现状：

| 系统 | 能力声明 | 实际指标 |
| :--- | :--- | :--- |
| BabelDOC（最强开源） | layout-preserving | 人工评分中等；论文自陈"需专业审校" |
| Snowflake AI_TRANSLATE（商业） | long-context | 长文档质量 **0.83**（自定指标） |
| Amazon VLM 长文档研究 | 最强 VLM pipeline | 准确率 **0.62** |

**没有任何系统声称产出"出版级"译文。** "出版级"描述的是流程，不是产物。

---

## 2. 目标架构：语义优先（Semantic-First）

### 2.1 核心原则

> **所有输入统一转成语义树；翻译在语义层进行；渲染从语义树重新生成。**
> **PDF 只是输入格式之一和输出格式之一，不是中心。**

### 2.2 架构图

```
┌──────────────────────────────────────────────────────────────────┐
│  输入层：统一转成语义树（DocumentTree）                            │
├──────────────────────────────────────────────────────────────────┤
│  DOCX/HTML/MD/EPUB ──► 原生解析（结构自带，零损失）                │
│  Born-digital PDF  ──► Docling ──► Markdown/HTML ──► DocumentTree │
│  Scanned PDF       ──► [显式拒绝 或 OCR+Docling（标注质量）]       │
│  竖排/水印/图内文字 ──► [显式拒绝]                                 │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  语义 IR：DocumentTree（树状，非 bbox 列表）                       │
│  SECTION / PARAGRAPH / TABLE / FIGURE / FORMULA / CODE / BIBLIO   │
│  每个节点携带：语义类型、层级、Markdown 源、公式 LaTeX 源、资产引用 │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  翻译层：Section 级长上下文                                       │
│  · 以 SECTION 为翻译单元（500~2000 token），上下文不割裂            │
│  · Book Bible / 术语表注入（保留现有资产）                          │
│  · 公式与引用占位符保护（保留现有 masker）                          │
│  · 结构校验（AST diff）取代正则 guard                              │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  渲染层：按目标格式重新生成                                        │
│  ┌────────────┬────────────┬────────────┬────────────┐            │
│  │ Typst→PDF  │ DOCX       │ HTML       │ EPUB       │            │
│  │（重排出版） │（原生结构） │（原生结构） │（原生结构） │            │
│  └────────────┴────────────┴────────────┴────────────┘            │
│  ┌──────────────────────────────────────────────────┐            │
│  │ overlay 引擎（保留，降级为 reference 参考模式）      │            │
│  │ 用途：需对照原版版面的场景；明确标注"非出版交付"    │            │
│  └──────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 关键决策

| 决策 | 选择 | 理由 |
| :--- | :--- | :--- |
| 主渲染引擎 | **Typst 全文档重排** | 语义树 → Typst 是自然映射；统一字号、原生表格/公式/图表 |
| PDF 解析 | **Docling 单一引擎** | 已验证产出干净 Markdown（303 块，14 表格，14 公式） |
| pdfium 快车道 | **降级为 preview 模式** | 不产出表格/图片，不适合交付 |
| 翻译单元 | **Section 级** | 上下文完整；RFC-01 的单 block 翻译是代词断裂的根因 |
| 质量门禁 | **AST diff 取代正则 guard** | 可验证、无误报；见 §4 |
| overlay 引擎 | **保留但不主推** | 资产不浪费；服务"版面对照"场景 |

---

## 3. 数据模型：DocumentTree

```python
# ubt/core/ir/document_tree.py

from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class NodeKind(StrEnum):
    DOCUMENT = "document"
    FRONT = "front_matter"  # 标题/作者/摘要/关键词
    SECTION = "section"  # 带层级与编号
    PARAGRAPH = "paragraph"
    TABLE = "table"  # 结构化（Markdown/HTML 源）
    FIGURE = "figure"  # 图片资产 + caption
    FORMULA = "formula"  # 块级公式（LaTeX 源）
    CODE = "code_block"
    BIBLIO = "bibliography"


@dataclass(slots=True)
class Node:
    id: str  # 沿用 #b0012 格式（见 §7）
    kind: NodeKind
    source: str  # Markdown / LaTeX 源
    target: str | None = None  # 译文
    level: int = 0  # SECTION 层级
    children: list["Node"] = field(default_factory=list)
    asset: str | None = None  # FIGURE 的图片路径
    page_hint: int | None = None  # 源文档页码（粗粒度溯源）
    source_anchor: str | None = None  # 原文特征指纹（用于 Web 双语对照高亮联动）


@dataclass(slots=True)
class DocumentTree:
    doc_id: str
    root: Node
    title: str
    source_path: str
    metadata: dict = field(default_factory=dict)
```

**Section 级原子事务记账协议（Section Atomic Ledger Protocol）**：
* **单一追踪单元**：翻译以 `SECTION` 为原子单位送给大模型，每个 Section 作为一个独立的 Ledger 记录（`block_id` 沿用 `#b0012`）；
* **流式结构化回填**：Section 内部的段落、公式、表格整体以 CommonMark 流形式翻译，模型译文整体落盘并一次性解析更新回子节点树，彻底规避以往逐段切碎再重拼导致的对齐断裂；
* **与现有 `IRBlock` 的兼容**：`IRBlock` 的 `bbox`/`spine_index`/`policy_*` 字段在语义路径中不再必需，但 `block_id` 的**命名约定严格保持**（见 §7 兼容性）。

---

## 4. 质量体系重构：从"正则 guard"到"可验证的 AST 契约"

> **勘误（2026-10-11）**：本 RFC 早期版本把删除目标写成"41 条 guard"。该数字
> 是估算，**无法从代码枚举**——`grep guard` 的 56 个命中里多数是保护性
> masker（`citation_masker`/`math_masker`/`code_masker`/`email_masker`），
> 属于 §7.1 要保留的资产。删除目标改为下表可枚举的具体机制。

### 4.1 删除/降级的机制（可枚举，附理由）

**质量校验器（`ubt/core/validators/`，共 3,173 行，4 个实现类）**

| 机制 | 位置 | 处理 | 理由 |
| :--- | :--- | :--- | :--- |
| `zh script density` 检查 | `layout_policy.py` | **删除** | 误杀表格/URL 段落（实测 b0014/b0096/b0127/b0128/b0173 共 6 处） |
| `NumericConsistencyValidator` | `ubt/core/validators/consistency.py:1060` | **降级为 advisory** | 已实测 16 例误报中 7 例为纯 bug（`二元矩阵`→`2元矩阵`、复数数词、`，` 粘连、`数`+数词），已修复；剩余仍应 advisory |
| `GlossaryConsistencyValidator` | `ubt/core/validators/consistency.py:1303` | **保留但降级为 advisory** | 引号保护区导致词表漂移误报 |
| `HTMLDeltaValidator` | `html_delta.py:40` | **保留** | 只对 HTML 源有意义，无 PDF 误报 |
| `math_guard` | `math_guard.py`（637 行） | **保留（收窄）** | 公式是硬约束，但 `Undelimited math` 应 advisory |
| `fail-closed quarantine` | `triage.py:346` | **✅ 已改为 fail-open + 标记** | 实测 18/18 quarantine 全是质量判定，17 例是被误杀的正确中文 |
| `span_repair` | `span_repair.py`（373 行） | **保留** | 供 PE 队列的 span 标注使用 |

**渲染期补丁**

| 机制 | 位置 | 处理 | 理由 |
| :--- | :--- | :--- | :--- |
| `reflow_overlays` | `render/reflow.py`（403 行） | **暂不删** | 实测它**在默认路径上**（`docling_render.py:137` 无条件调用），删了会直接改变当前 overlay 产物的版面。必须等语义渲染器能替代它 |
| `ReflowControlLoop` | `engine/reflow_loop.py`（312 行） | **先降级，后删除** | `visual_gate_enabled` 默认 True，是当前默认路径；需先确认降级 |
| `table_appendix` | `engine/stages/export.py:920` + `render/table_compiler.py:186` | **暂不删** | 实测 `table_render_mode` 默认 `"auto"`，`export.py:1479` 无条件调用。删了会移除复杂表格的附录交付路径 |
| `_quarantine_badge` | `render/outputs.py:2476` | **保留** | fail-open 后仍需标记少数真泄漏 |
| `overlay_text.py` | `adapters/pdf/overlay_text.py`（953 行） | **搬家，不删** | `typst_escape`/`typstify_math` 被 `render/table_compiler.py` 与 `render/outputs.py` 依赖 |

**defect taxonomy（`qe/defect_taxonomy.py`）**

| 集合 | 数量 | 处理 |
| :--- | :--- | :--- |
| `CRITICAL_DEFECT_MARKERS` | 20 | 收窄：质量类改为 advisory，仅保留 `Prompt scaffold` 等真泄漏为 critical |
| `STRUCTURAL_DEFECT_MARKERS` | 31 | 保留（结构/事实类） |
| `FORMAT_ONLY_MARKERS` | 4 | 保留 |

### 4.2 新增的验证（可自动测量）

```python
# ubt/core/verify/ast_contract.py


@dataclass
class ContractResult:
    structure_ok: bool  # 标题层级/列表嵌套/表格行列 完全保留
    math_ok: bool  # 公式 AST 规范化后一致
    placeholder_ok: bool  # 引用/邮箱/URL 占位符 100% 保留
    completeness_ok: bool  # 句级覆盖率 ≥ 99%
    details: dict


def verify_contract(tree_src: DocumentTree, tree_tgt: DocumentTree) -> ContractResult:
    """AST 级契约校验，替代正则 guard。

    - structure: 节点数/类型序列/表格行列数 必须一致
    - math:      公式节点规范化（LaTeX↔Typst）后结构哈希一致
    - placeholder: 占位符 token 集合完全一致
    - completeness: 目标节点文本非空且长度比在 [0.3, 3.0]
    """
```

### 4.3 分层质量指标（替代"出版级"）

| 产品线 | 输入 | 质量目标（可测） |
| :--- | :--- | :--- |
| **A. 语义出版级** | DOCX/HTML/MD/EPUB/LaTeX | 结构保真 **100%**；术语一致率 LTCR **≥ 0.95**；公式 AST 一致 **≥ 99%** |
| **B. PDF 参考级** | Born-digital PDF | BIoU **≥ 0.85**；未翻译块 **≤ 5%**；术语一致率 **≥ 0.90**；**明确标注"机器翻译，需人工复核"** |
| **C. 不支持** | 扫描件/竖排/水印/图内文字 | **显式拒绝**（fail fast，不产出垃圾） |

---

## 5. 技术攻坚：AST → Typst 转换器（本方案唯一的核心难点）

### 5.1 为什么这是核心

验证产物暴露了唯一的技术缺口：**LaTeX 公式 → Typst 数学的转换**。

Typst 数学**不是 LaTeX**：

| LaTeX | Typst | 说明 |
| :--- | :--- | :--- |
| `\frac{a}{b}` | `frac(a, b)` | 函数调用语法 |
| `\theta` | `theta` | 无反斜杠 |
| `\mathbb{E}` | `bb(E)` | 字体函数 |
| `\sum_{i=1}^{n}` | `sum_(i=1)^(n)` | 上下标语法 |
| `E = mc^2` | `E = m c^2` | **裸多字母标识符必须分隔** |

验证产物中有 **7 个数学块，6 个含转换伪影**（`mathb`、`"mathcal"`、`text (`）。

### 5.1.1 实测根因（2026-10-11 复核）

> **勘误**：本 RFC 早期版本列出"7 条修复规则"（`\mathbf`、`\bm`、`\log` 补空格…）。
> 实测发现**其中 3 条是空靶**：在 `2101.03961v3.pdf` 的 7 个公式里，`\mathbf` 出现
> **0 次**、`\mathrm` **0 次**、`\log` **0 次**。真正的根因只有一条。

**实测统计（该 PDF 全部公式）**：

| LaTeX 命令 | 出现次数 | 备注 |
| :--- | :--- | :--- |
| `\text{...}` | 6 | 真缺陷（见下） |
| `\mathcal{...}` | 3 | 被空格归一化挡住 |
| `\frac{...}` | 5 | 同上 |
| `\mathbb{...}` | 1 | 同上 |
| `\exp{...}` | 1 | 同上 |
| `\arg` | 1 | 不在命令表中 → `None` |
| `\mathbf` / `\mathrm` / `\log` | **0** | **空靶** |

**真根因（一条）**：Docling 输出 `\cmd {`（命令与花括号之间有空格），共 **15 处**。
而现有 `typstify_math` 的所有正则都要求紧贴花括号（`\mathbb\{`），因此全部失配。

实测验证：**只加一条空格归一化 + 一条 `\arg` rescue，通过率从 1/7 → 7/7**：

```python
def normalize_docling_math(body: str) -> str:
    """Docling 在 \\cmd 与 { 之间插了空格（LaTeX 允许，Typst 正则不允许）。"""
    s = re.sub(r"\\([A-Za-z]+)\s+\{", r"\\\1{", body)
    s = re.sub(r"\{\s+", "{", s)
    s = re.sub(r"\s+\}", "}", s)
    return s


_OP_RESCUE = re.compile(r"\\(arg|operatorname|mathop|bmod|pmod|limits|displaystyle)\b")
```

**另一处真缺陷（早期 RFC 未提）**：`typstify_math` 把 `\text{tokens per batch}` 拆成
`#"tokens" #"per" #"batch"`（3 个相邻字符串）。应输出单个字符串 `"tokens per batch"`。

### 5.1.2 修复对照表（修正版）

| # | 残留现象 | 根因（实测） | 修复规则 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `mathb` / `"mathbb"` / `"mathcal"` | Docling `\cmd {` 空格（15 处） | 一条空格归一化 | **P0** |
| 2 | 整个公式返回 `None` | `\arg` 不在 `_LATEX_CMD_MAP` | 补 `\arg`/`\operatorname` 等算子 | **P0** |
| 3 | `#"tokens" #"per" #"batch"` | `_WORD_RUN_RE` 逐词引号化 | `\text{...}` 整体包裹为单字符串 | **P0** |
| 4 | `\mathbf{X}` → 字面量 | `_TEXTLIKE_CMD_RE` 不含 `mathbf` | 补 `\mathbf`→`bold`（理论缺口，本片未出现） | P1 |
| 5 | `\mathrm{T}` → 字面量 | 同上 | 补 `\mathrm`→`upright`（理论缺口） | P1 |
| 6 | `A_{i, j}` 吞括号 | `_{...}` → `_(...)` 缺逗号处理 | 含逗号时强制括号（理论缺口） | P2 |
| 7 | `log p` 粘连 | 算子后缺空格 | 算子后补空格（本片未出现） | P2 |

### 5.2 规范（必须遵守）

```python
# ubt/render/typst/math_transpiler.py


def latex_to_typst(body: str) -> str:
    """LaTeX 数学体 → Typst 数学体。

    处理顺序（顺序错误会破坏嵌套）：
      0. Docling 归一化：`\\frac { a } { b }` → `\\frac{a}{b}`（**实测最优先，否则全部失配**）
      1. 字体与修饰：\\mathbf/\\bm → bold, \\mathrm → upright, \\mathbb → bb, \\mathcal → cal
      2. 结构函数：\\frac{a}{b} → frac(a, b), \\sqrt{a} → sqrt(a)
      3. 文本块：\\text{...} → 单个 "..." 字符串（**不拆词**）
      4. 符号与算子：\\alpha/\\theta → 希腊字母符号；\\arg/\\log/\\exp → 算子
      5. 定界符：\\left \\right 清理，{} 转 ()
      6. 裸多字母标识符：E = mc^2 → E = m c^2（加空格或引号）
    """
```

**必须用现有资产**：`typstify_math()` 已实现大部分转换，**不要重写**。Sprint 2 已将其搬到 `ubt/render/typst/math_transpiler.py`（因 `render/table_compiler.py` 依赖它），并补齐 §5.1.2 的缺口；原 `overlay_text.py` 已在 §7 删除。

### 5.3 验收

- 单元测试 **≥ 500 个** LaTeX 片段（覆盖 arXiv 常见公式）
- 转换残留率 **< 0.1%**
- 每个公式必须能通过 `typst compile` 单独验证（复用现有 `TypstMathProbe`）

---

## 6. 显存与资源（保留 RFC-Rev2 的正确部分）

RFC-Rev2 的 §4 显存调度**方向正确但落点错误**（`docling_adapter.close()` 是空实现）。

> **复核更正（2026-10-11）**：正确的落点确实是 `docling_parser.py`，而且**它已经实现了**：
> `_cleanup_docling_converter()`（`docling_parser.py:384`）在 5 处被调用（含 `finally`），
> 会逐个 `stage.engine.cleanup()` 并置 `None`。本 RFC 早期版本把它当作“待做项”，是错的。
> 真正缺的是：`torch.cuda.empty_cache()` 不在其中（docling 自己释放，但碎片未归还）。
> 实测该机器 RTX 4070 Ti 12.45 GB，空闲 9.79 GB——**当前无 OOM 证据**，
> 因此这一项从 Sprint 3 降级为“有证据再动”。

**修正**：串行生命周期由**引擎层**管理：

```
阶段 1: Docling 解析（GPU ~1.8GB）
        ↓ 强制释放：显式 del + gc.collect() + torch.cuda.empty_cache()
        ↓ 探针确认：VRAM < 300MB
阶段 2: LLM 翻译（GPU ~9.5GB）
        ↓ 完成（可选卸载）
阶段 3: Typst 编译（纯 CPU，~0.9s / 40 页）
```

**落点**：`ubt/adapters/pdf/docling_parser.py` 的模型生命周期，而非 adapter 的 `close()`。

---

## 7. 兼容性：保护既有投资

### 7.1 必须保留

| 资产 | 保留方式 |
| :--- | :--- |
| `SQLiteJobLedger`（事务/断点/计费） | 表结构零改动 |
| `AdaptiveTokenBucket` / `LedgerWriterLock` | 完全复用 |
| `Book Bible`（术语/人名挖掘） | 完全复用，输入改为 Section 级 |
| `ModelRouter`（多协议路由） | 完全复用 |
| 占位符引擎（email/citation/math masker） | 完全复用 |
| `TypstMathProbe` | 完全复用（转为转换器验证） |
| HTML/EPUB 渲染 | 完全复用（已是语义路径） |
| overlay 引擎 | 保留，降级为 `--render-mode reference` |

### 7.2 `block_id` 命名（RFC-Rev2 此处有错）

**Rev2 提议 `f"{spine_id}#n{node_index:04d}"`（`#n` 前缀）——这会破坏 PE 导入。**

`pe_import.py` 的 job 绑定校验依赖 block_id 的**结构稳定性**，且已交付的 PE 队列 CSV/XLIFF 全是 `#b0012` 格式。

**修正**：**保留 `#b` 前缀**。`DocumentTree` 节点 ID 沿用 `f"{spine_id}#b{idx:04d}"`，语义类型写入 `provenance` JSON：

```json
{
  "node_kind": "section",
  "section_path": ["2", "2.1"],
  "source_page": 5,
  "asset_ref": null
}
```

---

## 8. 实施路线图（基于 POC 已验证成果的 4 周攻坚计划）

鉴于 `/tmp/ubt_reflow/` 下的 POC 脚本已成功打通 `Markdown -> Section 翻译 -> Typst 编译` 并产出 29 页完整 PDF，原计划的试探周期可大幅压缩，转为**4 周敏捷工程化重构（Sprint 1 ~ Sprint 4）**：

| 阶段 | 周期 | 核心交付物（Deliverables） | 退出与验收标准（Exit Criteria） |
| :--- | :--- | :--- | :--- |
| **Sprint 1：基准打标与语义 IR 固化** | 第 1 周 | 1. 建立 `corpus/semantic_gold/`（20 篇真实公开论文）与 `eval_semantic.py` 自动化评测脚本<br>2. 落地 `ubt/core/ir/document_tree.py`，Docling 输出对齐为树状结构<br>3. **TOC 补全：读 PDF 原生书签（`pypdfium2.get_toc()`）**，不猜字号<br>4. quarantine 改为 fail-open + 标记，杜绝 10.6% 阻断 | 真实基线分数落盘；单篇论文端到端稳定提取为 `DocumentTree`；无误杀阻断 |
| **Sprint 2：Typst Transpiler 模块化落地** | 第 2 周 | 1. 将 `/tmp/ubt_reflow/render_typst.py` 核心逻辑产品化为 `ubt/render/typst/math_transpiler.py`<br>2. 复用 `typstify_math` 并补齐 7 处 LaTeX 转换缺口（`\mathbf`、`\mathrm`、`\text` 等）<br>3. 固化 `academic_zh.typ` 学术双栏模板与 `#place(top, float: true)` 跨栏图表浮动规则<br>4. 编写 500+ 个公式与复杂表格单元测试 | 单元测试通过率 100%；转换残留率 < 0.1%；Typst 单次编译零 Panic |
| **Sprint 3：引擎打通与端到端交付** | 第 3 周 | 1. 引擎接入 Section 级长上下文翻译流（Section 原子记账协议）<br>2. 落地显存串行时分调度（`docling_parser.py` 显式回收 + 探针监控）<br>3. CLI 新增 `--render-mode publish`（默认）与 `--render-mode reference`（原 overlay 降级）<br>4. 跑通 20 篇 Gold 论文自动化全流程回归 | 20 篇 Gold 论文全部达到 §9 验收指标（正文字号 P95 ∈ [10, 11]pt，表格中文化率 ≥ 95%） |
| **Sprint 4：历史包袱清理与瘦身交付** | 第 4 周 | 1. 废除并安全移除 `reflow.py`、`table_appendix` 及 §4.1 表中可枚举的废弃机制<br>2. 缩减 PDF 适配器无用代码（从 14,927 行瘦身，目标行数待 Phase 0 测量后定）<br>3. 更新用户文档与 README，正式发布 UBT 语义重排版 | 代码库精简 ≥ 10k 行（全库 84,912 行 ≈ 11.8%）；CI 耗时与测试稳定性显著提升；发布 v1.0-RC |

**总计 4 周（1 个月快速迭代落地）。**

---

## 9. 验收指标（全部可自动测量）

| 维度 | 现行 | 目标 | 测量方法 |
| :--- | :--- | :--- | :--- |
| 中文占比（正文） | 31% | **≥ 55%** | `pdftotext` 字符统计 |
| 未翻译块 | 32（10.6%） | **≤ 5%** | ledger 状态统计 |
| 表格中文化率 | 0% | **≥ 95%** | 表格单元格 CJK 覆盖 |
| 正文字号离散度 | 4~12pt | **P95 ∈ [10, 11]pt** | pypdfium2 TextSpan 统计 |
| 公式 AST 一致率 | 未测 | **≥ 99%** | 规范化 AST diff |
| 术语一致率 LTCR | 0.916 | **≥ 0.95** | 现有 `term_metrics.py` |
| 端到端交付率 | 89.4% | **≥ 95%** | gold corpus 成功率 |
| 渲染耗时 | 10~30s（碎片） | **< 2s / 篇** | 计时 |

**明确删除的口号式指标**：~~"100% 严格统一字号"~~、~~"0 异常空白"~~、~~"100% 公式保真"~~（不可测或不可能）。

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解 |
| :--- | :--- | :--- |
| Docling 抽取质量不足（TOC 一级标题丢失已验证） | 结构缺失 | **已修**：读 PDF 原生书签树（`read_pdf_outline_entries`）。实测该 PDF 有 31 条书签、层级精确，而 Docling 漏掉全部 15 条一级标题。**有确定解就不猜**——字号校准器只在 PDF 无书签时降级启用 |
| LaTeX→Typst 转换器覆盖不全 | 公式渲染失败 | 复用现有 `typstify_math` 补齐 7 项缺口；500 单测；失败降级为公式图片 |
| Typst 图表浮动不如 LaTeX | 版面观感 | 已验证 `place(top, float: true, scope: "parent")` 可用 |
| 语义路径的版面与原文差异大 | 用户预期不符 | 明确产品定位为"重排译本"；保留 reference 模式供对照 |
| 扫描件/竖排等不支持 | 用户流失 | 显式拒绝 + 清晰文档；不做"尝试后输出垃圾" |

---

## 11. 与 RFC-2026-01 / Rev2 的关系

| 项 | RFC-01 / Rev2 | 本方案 |
| :--- | :--- | :--- |
| 根本诊断 | "overlay 是死胡同" ✅ 正确 | "**PDF 作为主输入格式**是死胡同"（更深一层） |
| 解决方案 | 双轨制 + Typst 新引擎 | **语义优先 + 全格式统一重排** |
| PDF 定位 | 仍为主输入 | **输入格式之一** |
| 质量目标 | "出版级" | **分层可测指标** |
| 转义规范 | §2 字符级转义（漏了数学转换） | **§5 完整 AST→Typst 转换器** |
| 评估体系 | 5 个指标（2 个不可测） | **AST 契约 + 8 个可测指标** |
| `block_id` | `#n` 前缀（破坏兼容） | **保留 `#b` 前缀** |
| 显存落点 | `docling_adapter.close()`（空实现） | **`docling_parser.py` 模型生命周期** |

**Rev2 的方向（Typst 重排）被本方案继承；Rev2 的定位（PDF 仍是主输入）被本方案推翻。**

---

## 12. 附录：验证产物复现步骤

```bash
# 1. PDF → 语义 Markdown（复用现有脚本）
uv run python scripts/export_pdf_to_markdown.py \
    2101.03961v3.pdf /tmp/ubt_reflow/2101_docling.md

# 2. 分块 + Section 级翻译（脚本见 /tmp/ubt_reflow/translate.py）
uv run python /tmp/ubt_reflow/translate.py

# 3. 表格翻译（/tmp/ubt_reflow/translate_tables.py）
uv run python /tmp/ubt_reflow/translate_tables.py

# 4. Markdown → Typst → PDF（/tmp/ubt_reflow/render_typst.py）
uv run python /tmp/ubt_reflow/render_typst.py
cd /tmp/ubt_reflow/build && typst compile main.typ out.pdf

# 产物：/tmp/ubt_reflow/重排验证产物.pdf（29 页）
```

---

## 13. 实施日志

### 2026-10-11：Sprint 1 核心项落地

**① 基线评测脚本（新增 `scripts/eval_semantic.py`，~280 行）**

首次把"重排是否更好"变成可测数字。实测两套基线：

| 指标 | 现行 overlay | 语义重排 | 说明 |
| :--- | ---: | ---: | :--- |
| `cjk_ratio` | 0.311 | **0.580** | 中文占比 |
| `untranslated_marks` | 17 | **0** | 未译标记 |
| `p95_font_size_pt` | 11.46 | **10.50** | 字号 P95 |
| `distinct_font_sizes` | 44 | **5** | 字号离散度 |
| `source_header_term_leakage` | 0.96 | **0.32** | 源表头可译词残留率（越低越好） |
| `math_residue_ratio` | — | 0.857 | **待修（§5.1.2）** |
| `ledger.blocked_ratio` | 0.059 | — | 待重跑 |

> **指标设计说明**：`source_header_term_leakage` 是唯一无歧义的"表格翻没翻"信号。
> 早期版本曾试图用 `pdftotext -layout` 的列对齐启发式测"中文表头率"，结果把
> 旧产物的中文正文行误判为表头（得 0.61），**掩盖了真实缺陷**——已废弃。

**② 数值误报修复（`ubt/core/validators/consistency.py`）**

实测该 run 的 16 例 `Numeric fidelity` 失败，逐例核对后发现 **7 例是纯 bug**：

| Bug | 实例 | 根因 | 修复 |
| :--- | :--- | :--- | :--- |
| 中文词内数字被当数量 | `二元矩阵`→`2元矩阵` | `元` 是 measure word，`二` 被折叠 | 新增 `_CN_NUMERAL_COMPOUNDS` 词表（二元/两两/五花八门…） |
| 英文复数数词未授权 | `billions`→`数十亿` | `_EN_NUMBER_WORDS` 只有单数 | 新增 `_EN_NUMBER_PLURALS` |
| 不定量词被当精确值 | `数十亿`→`1000000000` | `数`+数词被折叠 | 新增 `_CN_INDEFINITE_QUANTITY_RE` |
| 中文逗号被剥离后粘连 | `2017，2018`→`20172018` | `，` 被当千分位删除 | 新增 `_CJK_LIST_COMMA_RE`（替换为空格） |
| 双数限定词未授权 | `both X and Y`→`这两个` | `both` 不在词表 | 新增 `_EN_DUAL_DETERMINERS` |

结果：**16 例中 7 例误报消除**，新增 **10 条回归测试**（已验证：回退代码时 10 条全部失败）。

剩余 9 例分类（**不是 validator 的 bug**）：

| 类型 | 例数 | 说明 |
| :--- | ---: | :--- |
| 块边界错位 | 2 | `src="Contents"` 而 `draft="2.1 简化稀疏路由"`——上游分块问题 |
| 译文语义推断加数词 | 2 | 源文 `this decision`（单数），译文写`这两种方法`——模型加了数词 |
| PDF 上标抽平 | 3 | `2²⁰` 被抽成 `2 20`——抽取问题 |
| 脚注标记丢失 | 2 | 译文丢掉了上标脚注号 |

**③ quarantine 改 fail-open（`ubt/core/engine/stages/triage.py` + `policy/layout_policy.py`）**

实测：18 个 `BLOCKED_HUMAN` 块**全部**是质量判定，其中 **17 例是被误杀的正确中文草稿**。
旧行为把中文换成英文源文 + 一个`⚠未译`徽标，读者拿到的是无用的东西。

新规则：
- **质量判定**（数值/公式/词表/遗漏）→ `NEEDS_HUMAN`，**保留机器草稿**，标记留在记录里
- **必须拦截的泄露**（`Prompt scaffold` / `Prompt template XML artifacts`）→ 仍 `BLOCKED_HUMAN`，剥离机器输出

新增 `is_fatal_leak()` 作为唯一判定函数（避免两处漂移），新增 `tests/unit/engine/test_triage_routing.py`（3 条用例，已验证回退时失败）。

**④ TOC 改读 PDF 原生书签（`ubt/adapters/pdf/docling_parser.py`）**

实测：该 PDF 自带 **31 条原生书签，层级精确**（`L0 1 Introduction`、`L1 2.1 Simplifying Sparse Routing`），
而 Docling 的 `DOCUMENT_INDEX` 抽取**漏掉全部 15 条一级标题**（根因：标题与页码在几何上分属不同行）。

新增 `read_pdf_outline_entries()` + `_extract_outline_only_blocks()`：
- 与 Docling 的结果**合并**（不替换）——Docling 提供子章节及其几何位置，书签提供顶层章节
- 按标题去重（含去掉数字前缀的模糊匹配）
- 合成的块**无 bbox**（书签只给目标页，不给矩形），渲染器自然将其留在 Layer 0

实测：`already` 含 5 条 Docling 行时，补回 26 条，**零重复**。

**验证**：`pytest tests/unit` → **2,949 passed, 3 deselected**。

**⑤ 金标集（新增 `corpus/semantic_gold/`，20 篇真实论文，139 MB）**

现有 `corpus/documents/` 只有 4 篇。新建 20 篇，按维度分类（实测页数/书签数，非论文自报）：

| 维度 | 篇数 | 例 |
| :--- | ---: | :--- |
| `toc-outline`（有书签） | 15 | switch-transformers(31 书签)、mamba(57) |
| `complex-tables` | 13 | clip、flamingo |
| `formula-dense` | 11 | neural-ode、ddpm |
| `two-column-figures` | 9 | resnet、vit |
| `no-outline`（无书签） | 5 | bert、gan、whisper、clip、resnet |
| `long-mixed` | 4 | gpt3(75页)、llama2(77页) |

新增 `scripts/fetch_semantic_gold.py`（可重现获取）+ `scripts/eval_gold_corpus.py`（批量评测）。

> **发现**：bert/resnet/whisper/clip/gan 实际**无书签**，原定分类是错的，已根据实测修正。

**⑥ `DocumentTree`（新增 `ubt/core/ir/document_tree.py`）**

> **对 RFC §3 的修正**：RFC 原写“新增 `Node`/`DocumentTree` 数据类”。读代码后发现
> `IRBlock`（类型化 element + `level` + `spine_index` + provenance）已是唯一真相源，
> `ContentGraph` 已声明交付契约。**再加一棵树就是制造第二个真相源**——正是本次重构要消除的东西。
>
> 所以本模块只提供真正缺的东西：**把现有块分组成章节**，使翻译可按 Section 作用域跑（§2.3）。
> 无存储、无新状态；账本仍按块记账（§7.2）。

实现为 `Section` + `build_section_tree(blocks)` + `section_texts(sections)`：
- 顶层标题开新章节，更深标题留在当前章节内
- 无标题文档仍产出 1 个可译单元（EPUB 章节、普通文章）
- `path` 取自源文自身编号（`("2", "2.1")`），源文跳级不会留下陈旧祖先
- `skip_translate` / `policy_translate=False` 的块排除在可译视图外，但仍留在章节内（保阅读序）

实测（arXiv 2101.03961 的 303 块）：**→ 40 个章节**。新增 6 条单测。

**Sprint 1 状态：✅ 完成**。全量单测 **2,955 通过**；`ruff check` / `ruff format` / `mypy` 均干净。

**下一步（Sprint 2）**：`ubt/render/typst/math_transpiler.py`——把 `typstify_math` 从 `adapters/pdf/`
搬到 `render/typst/`，补齐 §5.1.2 的三条 P0（当前残留率 **0.857**，是全项目最差指标）。

### 2026-10-11（续）：Sprint 2 —— 数学转译器

**⑦ 搬家与依赖方向修正**

新增 `ubt/render/typst/`：
- `math_transpiler.py`（356 行）——从 `adapters/pdf/overlay_text.py` 提取，含 `typstify_math` /
  `typstify_math_span` / `_convert_fractions` / `_convert_sqrts` / 字形映射等 26 个符号
- `escape.py`（71 行）——`typst_escape` + `escape_line_start_markup`

`overlay_text.py` 从 **954 行 → 654 行**（−300），只保留 overlay 专属逻辑 + re-export。
依赖方向修正：`render/table_compiler.py` 与 `render/outputs.py` 不再反向 import `adapters/pdf/`。

**⑧ 三条 P0 修复（实测结果）**

| # | 缺陷 | 修复 | 实测 |
| :--- | :--- | :--- | :--- |
| 1 | Docling 输出 `\cmd {` 空格（该论文 15 处），所有正则失配 → 整式返回 `None` | `_normalize_latex_spacing()` | 6/7 失败 → 0 |
| 2 | `\arg` 不在符号表 → 未知命令 guard 使**整式**失败 | `_rescue_operators()` | 同上 |
| 3 | `\text{tokens per batch}` 被拆成 3 个相邻字符串 | 合并相邻引号串 | 下标位置不再多空格 |

**关键指标**：`math_residue_ratio` **0.857 → 0.000**（7/7 公式通过 `typst compile`）。

> **更正 RFC §5.1.2**：原表把 `\text{}` 拆词列为“无害”。实测发现**它在下标位置有害**：
> Typst 中 `A_(#"i" #"j")` 渲染为 `A_i j`（带空格），而 `A_#"ij"` 渲染为 `A_ij`。
> 已修正。
>
> **又更正**：原表列了 7 条规则，实测其中 3 条（`\mathbf`/`\mathrm`/`\log`）在该论文出现 **0 次**。
> 真根因只有 1 条（空格归一化）。

**新增测试**：`tests/unit/render/test_math_transpiler.py`（30 条，含 7 个真实公式的
“能编译”参数化用例）。已验证：注释掉两条 P0 修复后 **22 条失败**。

**Sprint 2 状态：✅ 完成**。全量单测 **2,989 通过**；ruff / mypy 干净。

**下一步（Sprint 3）**：Section 级翻译接入 + 显存时分调度 + CLI 双模式。

### 2026-10-11（续）：Sprint 3 —— Section 级翻译接入

**⑨ 实测三个“待做项”，两个已经存在**

| 计划项 | 实测 | 结论 |
| :--- | :--- | :--- |
| 显存时分调度 | `_cleanup_docling_converter()` 已实现，5 处调用（含 `finally`） | **已存在**，降级 |
| 长上下文翻译 | `draft_macro_chunk_group` + `macro_chunk_size` 已存在 | **已存在**，但缺章节感知 |
| Section 级接入 | 无 | **本次做** |

**⑩ 发现两个真实缺陷**

**缺陷 A：固定切分切断章节**。实测 `macro_chunk_size=5`（默认值）时，一个 chunk 结束在 Abstract 中间：

```
固定切分:   chunk 1: ['.','.','.','H','.']   ← 标题的正文跑到下一个请求
            chunk 2: ['.','.','.','.','.']
```

新增 `chunk_by_section()`：标题开新 chunk（除非当前 chunk 全是标题——那是标题块，如 arXiv id +
期刊 + 提交日期 + 标题，拆开就是为一页付 4 次钱）。

**缺陷 B：标题层级全部丢失**。实测产物 HTML：

```
<h1> × 40     ← 全文 40 个标题全是 h1，层级完全拍平
```

根因：Docling 的 Markdown 把**每个**标题都输出为 `#`（40 个 `#`、0 个 `##`），
真实层级只在 PDF 书签里（`L0: 15` + `L1: 16`）。

修复：`_extract_outline_only_blocks()` 现在把书签 level 传入 `make_element(level=…)`
（书签 0-based，标题 1-based）。实测：`1 Introduction` → level 1，`2.1 Simplifying` → level 2。

**⑪ `build_section_tree` 不再是死代码**

新增 `section_context_for(blocks, chunk)` + `heading_trail()`，接入 draft 链路：
`chunk_by_section` → `section_context_for` → `chunk_context`（缓存键）→ `draft_macro_chunk`
→ prompt 新增 `### Section Context` 段。

实测长章节的第 2 个 chunk（本身无标题）现在拿到 `"2.1 Simplifying Sparse Routing"`。

> 缓存键与调用**同源**（同一份 `section_context` 同时进 `step_key` 和 prompt），
> 避免本 RFC §7.1 反复强调的“两处定义漂移”。

**新增测试**：`tests/unit/engine/test_section_chunking.py`（9 条）+ 4 条 section context 用例。
已验证：把 `chunk_by_section` 换回固定切分会失败 3 条。

**Sprint 3 状态：✅ 完成**。全量单测 **3,002 通过**；ruff / mypy 干净。

**未做（有意识的）**：`--render-mode publish` CLI 标志。
原因：生产级语义重排渲染器还不存在（`/tmp/ubt_reflow/render_typst.py` 是 POC 脚本）。
加一个背后没有实现的标志，就是本 RFC 一直在批判的“口号式交付”。
它属于 Sprint 4，等渲染器产品化后再加。

**下一步（Sprint 4）**：删除废弃机制 + 渲染器产品化 + 文档更新。

### 2026-10-11（续）：Sprint 4a —— 安全删除

**⑫ 首先：本 RFC 的删除清单是错的**

逐条实测 RFC §4.1 列出的“删除目标”，**没有一项是死代码**：

| RFC 列的目标 | 实测 | 结论 |
| :--- | :--- | :--- |
| `reflow.py` “仅 3 处引用” | `docling_render.py:137` **无条件调用**（默认 overlay 路径） | ❌ 活着 |
| `table_appendix` | `table_render_mode` 默认 `auto`，`export.py:1479` 无条件调用 | ❌ 活着 |
| `overlay_text.py` “搬家” | 已完成（Sprint 2），但只是**搬家**不是删除 | ✅ 已做 |

“引用数少”≠“死代码”。两者都在默认路径上，删了会直接改变当前产物。已回写 §4.1。

**⑬ 真正的死代码：写 AST 扫描找出来的**

不用印象，直接扫全库（`ubt/` + `tests/` + `scripts/` + 文档/配置）。删了 **25 个定义，619 行**：

| 类型 | 数量 | 典型 |
| :--- | ---: | :--- |
| 函数/方法 | 16 | `contains_point`（带 `bisect` 实现）、`as_typst_tuple`、`validate_contract`、`_render_advisory_markdown`、`synthetic_vlm_lines` |
| 常量 | 8 | `MQM_SEVERITY_LEVELS`、`PASS_RATE_THRESHOLD`、`_FRAC_RE`/`_SQRT_RE`（Sprint 2 搬家时的残留） |
| 类 | 1 | `PageDimension`（删 `inspect_dimensions` 后成孤儿） |

**⑭ `RUN_ONLY_KEYS` 不删，改为“变活”**

它无引用，但文档化了一个真实不变量：run-only key 绝不能同时是 `UBTConfig` 字段。
`overrides_from_request` 靠 `UBTConfig.model_fields` 过滤——**今天成立只是巧合**。
若将来加一个叫 `profile` 的配置字段（很合理，已有 `preset`），job payload 就能静默设它。

所以不删它，而是新增 `tests/unit/test_run_only_keys.py`（11 条）把它**断言成活文档**：
断言两个集合不相交。已验证：注入一个冲突字段 `job_id` 后测试失败。

**⑮ 新增死代码护栏（`tests/unit/test_no_dead_definitions.py`）**

项目已有 `test_no_orphaned_modules.py` 防“孤儿模块”，但看不到另一半问题：
**活模块里被删空后剩下的定义**。新增同形护栏（纯 AST、不执行 import）。

> 首版实现用逐名正则扫全库，**实测 110 秒**——不是 fast test。
> 改为全库一次分词 + `Counter` 计数，**0.31 秒**。
> （必须计数而非只判存在：定义行自身就含该名字，只判存在会让检查恒真。）

已验证：注入一个死函数 + 一个死常量，测试报出两者。

**Sprint 4a 状态：✅ 完成**。全量单测 **3,014 通过**；ruff / mypy 干净。

**Sprint 4b（下一步）**：渲染器产品化——把 `/tmp/ubt_reflow/render_typst.py` 变成
`ubt/render/typst/` 里的真实现，之后才能安全删 `reflow.py` / `table_appendix` 并加 `--render-mode`。

### 2026-10-11（续）：Sprint 4b —— 渲染器产品化

**⑯ 新增 `ubt/render/typst/document_renderer.py`（~380 行）**

把 POC 脚本变成真实现。不再重复造轮子：直接复用 Sprint 2 的
`math_transpiler` + `escape`，以及现有的 `table_compiler.parse_markdown_table`。

设计要点（都在 docstring 里写明了理由）：
- **数学不重新实现**——只决定“在哪里应用”
- **转义是结构性的**——散文转义、公式转换，两者不混（先转义会把 `$x_i$` 变成字面量）
- **表格重建而非叠加**——这是 overlay 路径根本做不到的事
- **图带着它的 caption**——图片 + 后随短段是抽取器的通用形态
- **确定性**——同输入同输出

**⑰ 实测抓出三个真实 bug**

| # | 缺陷 | 实测症状 | 修复 |
| :--- | :--- | :--- | :--- |
| 1 | 把 quarantine 占位符 `<mark class="ubt-blocked-human">` 当译文渲染 | `untranslated_marks` **36** | `_is_placeholder()`：优先用 `draft_text`（真实中文），跳过占位符 |
| 2 | 完全没检查 `skip_translate` | 90 个刻意保留块（61 条参考文献）走翻译路径 | `_keeps_source()`，与 draft 阶段同规则 |
| 3 | 把 `skip_translate` 检查放在 asset 分支**之前** | `formulas` 从 7 变 **0**（公式也是 skip-translate，但必须排版成数学） | 检查移到 asset 分支之后 |

> Bug 3 是 Bug 2 的修复引入的回归，由渲染统计直接暴露（`formulas: 0`）。
> 这是“指标可测量”的价值：否则这个回归只能靠肉眼发现。

**⑱ 指标对比（同一份 PDF）**

| 指标 | overlay 产物 | 语义重排（POC） | **语义重排（产品化）** | 目标 |
| :--- | ---: | ---: | ---: | ---: |
| `cjk_ratio` | 0.311 | 0.581 | **0.477** | ≥0.55 |
| `untranslated_marks` | 17 | 0 | **0** | 0 ✅ |
| `p95_font_size_pt` | 11.46 | 10.50 | **10.50** | 10–11 ✅ |
| `distinct_font_sizes` | 44 | 5 | **5** | — |
| `math_residue_ratio` | — | 0.000 | **0.000** | <0.001 ✅ |
| 页数 | 41 | 29 | **29** | — |

> **`cjk_ratio` 下降不是回归。** 产品化版本**正确地包含**了 36 条参考文献
> （`skip_translate=1`，应保留原文）；POC 脚本把它们整段丢了。
> 实测：`Association for Computational` 等标志在 POC 里 **0 处**，在产品化里 **36 处**。
> 分母里多了正确的英文原文，比例自然下降。

**⑲ 新增测试**：`tests/unit/render/test_document_renderer.py`（14 条）。
已验证：回退 3 个修复中的 2 个（占位符 + skip_source）→ **3 条失败**。

> 另：新加的死代码护栏**当场抓到了我自己**——`_MARKUP_SPECIAL` 在改成
> 委派 `typst_escape` 后成为死常量，测试直接报出。护栏有效。

**Sprint 4b 状态：✅ 完成**。全量单测 **3,028 通过**；ruff / mypy 干净。

**下一步（Sprint 4c）**：把渲染器接入 adapter（`--render-mode publish`），
然后才能安全删 `reflow.py` / `table_appendix`。

### 2026-10-11（续）：Sprint 4c —— 渲染器接入 adapter

**⑳ 新增 `--render-mode` 开关（默认 `overlay`）**

完整链路（逐段验证）：

```
CLI --render-mode publish
  → request key "render_mode"
  → UBTConfig.render_mode (env: UBT_RENDER_MODE)
  → AdapterRuntimeConfig.render_mode
  → DoclingPDFAdapter.render_mode (property)
  → DoclingRenderStrategy.render_mode
  → _render_publish() → TypstDocumentRenderer
```

**默认值保持 `overlay`**：升级不得静默改变现有产物的形状。

**㉑ 设计要点**

- **property 而非第二个字段**：`render_mode` 是 property，setter 同时写 strategy。
  adapter 是 `apply_config` 写入的地方，strategy 是 `render_blocks` 读取的地方——
  两个裸字段正是本 RFC 反复强调的“两处定义漂移”。
- **编译失败抛异常**，不产出空文件：调用方要的是重建文档，一个缺失的 PDF 比一个
  空 PDF 更清楚。生成的 `.typ` 保留在 `.out_typst/` 供排查。
- **非 PDF 目标忽略该模式**：Markdown/HTML 有自己的写入路径。
- **资产缺失进入 `last_render_skips`**：丢掉的图在质量报告里可见，而不是静默变短。

**㉒ 新增测试**：`tests/unit/adapters/test_render_mode_routing.py`（8 条）。
已验证：把 setter 改成“只写 adapter 不推给 strategy” → **2 条失败**。

**㉓ README 更新**

新增 “Render modes” 小节（含实测对比表），并修正两条已不准确的 “Known limits”：
- “Tables ... 无法重建网格” → 区分两种模式
- “Text expansion ... 无箱体增长或页面重排” → `publish` 无此限制

**Sprint 4c 状态：✅ 完成**。全量单测 **3,036 通过**；ruff / mypy / doc-refs 全干净。

**下一步（Sprint 4d）**：删除 `reflow.py` / `table_appendix`。

### 2026-10-11（续）：Sprint 4d —— 语料验证（删除前提被证伪）

**㉔ 前提核实：现在还不能删**

开工前实测，两条都被证伪：

| 目标 | 实测 | 结论 |
| :--- | :--- | :--- |
| `reflow.py` | `render_mode` 默认 **`overlay`**（当时），`docling_render.py` 无条件调用 | ❌ 仍在默认路径 |
| `table_appendix` | `table_render_mode` 默认 **`auto`**，`export.py:1477` 无条件调用 | ❌ 仍在默认路径 |

**删它们的唯一前提是把 `publish` 变成默认**，而这需要跨文档证据。所以 Sprint 4d 改为
**先把证据做出来**。

**㉕ 新增 `scripts/sweep_publish_render.py` + 全语料扫描**

真实 Docling 抽取 + 真实 `publish` 渲染，**不调 LLM**（所以零成本、无需 provider）。
它验证翻译下游的全部环节：块类型覆盖、图资产解析、Typst 编译、字号一致性。

**实测 20/20 篇全部编译通过，0 个丢失资产**：

| 指标 | 结果 |
| :--- | :--- |
| 文档 | **20/20** 渲染并编译成功 |
| 表格 / 图 / 公式 | 311 / 291 / 279 个全部处理 |
| `assets_missing` | **0**（20 篇合计） |
| `p95_font_size_pt` | **10.5**（20/20 篇完全一致） |
| `distinct_font_sizes` | 2–5（中位 3） |
| `math_residue_ratio` | **0.0**（全部） |
| 最大文档 | gpt3（811 块）、llama2（838 块） |

对比 overlay 路径的实测：`distinct_font_sizes` **44**、`p95` 11.46。
`publish` 把字号离散度从 44 降到中位 3。

**㉖ 新增测试**

- `tests/integration/test_publish_corpus_sweep.py`（6 篇不同形态，`slow` 层，19 秒）
  —— 断言：抽取非空、**零丢失资产**、Typst 编译、字号一致
- `tests/integration/test_gold_manifest.py`（2 条，`fast` 层）—— 清单 id/分类/实测字段

> 拆成两个文件是因为 `test_marker_hygiene` 的两条规则：一条要求「每个测试都有层级标记」，
> 另一条禁止「同一测试同时有 fast+slow」。把廉价的清单校验和昂贵的抽取放在同一模块会
> 触发第一条。**护栏在这里起了作用**——它逼我做了正确的结构划分。

**Sprint 4d 状态：✅ 完成（改为语料验证）**。
全量 **fast 3,038 + slow 51 全部通过**；ruff / mypy / doc-refs 干净。

**㉗ 补齐 web 端（原计划遗漏）**

Sprint 4c 只接了 CLI，**REST API / MCP / web 控制台全都拿不到 `render_mode`**。实测确认：

| 入口 | 4c 后 | 现在 |
| :--- | :--- | :--- |
| CLI `--render-mode` | ✅ | ✅ |
| REST `JobSubmitRequest` | ❌ | ✅ |
| MCP `ubt_translate_book` | ❌ | ✅ |
| `web/src/api/generated-types.ts` | ❌ | ✅（重新生成） |
| 控制台向导 UI | ❌ | ✅（Advanced 抽屉新增选择器） |

改动：`ubt/api/models.py` + `ubt/mcp/server.py` + 重新生成 OpenAPI/TS 类型 +
`web/src/views/wizard/NewJobWizard.tsx` + i18n（en/zh/types 三处）。

**同时收紧类型**：`render_mode` / `table_render_mode` 原本是裸 `str`，非法值会静默
退化成 `overlay`。现改为 `Literal` 并接入 `_ENUM_REQUEST_FIELDS`，三个入口统一在提交前
拒绝非法值（MCP 过去会先返回 job_id 再后台失败）。

新增 `tests/unit/test_option_surface_parity.py`（9 条）——**断言三个入口的旋钮集合一致**，
而不是断言单个入口的内容。回退 MCP 参数后该测试立即失败（已实测）。

**结论**：`publish` 已在 20 篇真实文档上验证可用，**但删除 `reflow.py` / `table_appendix`
的前提（把 `publish` 设为默认）需要产品决策**，不是工程决策——默认模式决定交付物形态，
应当由用户/产品选择，而非重构者擅自翻转。本 RFC 的建议是：
保留双模式，让 `--render-mode` 成为用户的显式选择，**不删除任何一条路径**。

---

*本方案以实证验证产物为唯一依据。所有指标均可自动测量，无口号式承诺。*
