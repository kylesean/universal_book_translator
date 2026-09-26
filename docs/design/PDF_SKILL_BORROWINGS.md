# UBT 可借鉴清单：来自 `pdf` skill 的硬化设计

> **状态**：🟢 活文档（D1–D11 的权威定义处，随实现状态更新；引用前以代码符号为准）

> **语料出处说明（2026-09-19）**：文中对 `chapter-1.pdf` / `chapter-3.pdf` 的引用基于当时的 Elsevier
> 版权样本；该样本已因合规撤换为 `tests/fixtures/synthetic-*.pdf` 合成语料，历史测量结论不受影响。


- 状态：**D6′/D1/D2′/D10/D9 已实现**（D9 于 2026-09-20 `f621705` 落盘，形态为 `job_meta.usage_totals` 元数据键而非提案中的独立表）；D2（收缩后）/D3/D4/D5/D7/D8/D11 仍是 Proposal
- 目标读者：UBT 维护者 + 实现这些改动的 agent
- 范围：从 `pdf` skill 的"翻译保版"路线抽取**可迁移的验证纪律与资产纪律**，映射到 UBT 现有模块
- 非目标：替换 UBT 的排版引擎、引入浏览器依赖、照搬 `pdf` skill 的模板结构
- 修订：**v2** — §4 由"明确不借鉴"重构为三类结论（已正确 / 不借鉴 / 兜底通道）；新增证据强度限定；更正原表一处自相矛盾的论据（TOC）与一处错误归因（agent 管线）
- 修订：**v3**（2026-09-18）— ① **补齐归属**：本文通篇的 "`pdf` skill" 指 **MiniMax 内置 `pdf` skill v3.0**（`/home/kyle/.minimax/.builtin-skills/pdf/`，`SKILL.md:11`），原稿从未写明；② 撤回三处错误归因（"README 自陈不适用 PDF 输入"、smask 与"四样验收"的出处、D4 中把本方推论当作对方立场）；③ **D2 判据失效，改出 D2′**；④ D6 由 P1 升 **P0**，因 §7 记录了它已实际踩响两次

> **v3 阅读前提。** 横向对比（哪些 agent 带 pdf skill、各自能不能做这道题、成本量级）原载
> `docs/PDF_AGENT_SKILLS_VS_UBT.md`，该文档已于 2026-09-22 删除，实测数据存于 git 历史。
> 本文档只保留"UBT 该借什么"的落地设计，且自 2026-09-22 起为 D1–D11 的权威定义处。

## 1. 背景与问题

本节的 "`pdf` skill" 现予明确署名：**MiniMax 内置 `pdf` skill v3.0**（`/home/kyle/.minimax/.builtin-skills/pdf/`）。其 `translate-preserve-layout` 是从一个 EML 邮件报告案例蒸馏出来的配方（`templates/translate-preserve-layout/README.md:3` "Distilled from deepforge bench m09"）。

> **路径约定**：下文出现的 `pitfalls-index.md`、`docs/email-translation-goldman-two-sessions-case.md`、`templates/...` 等引用均为**对方 skill 目录内的相对路径**（根为上面那个绝对目录），不是本仓库文件。行号基于 2026-09-18 的对方版本，属历史引述。

> **v3 更正（原稿三处错误归因，已撤回）**
> 1. 原稿称"该模板 README 自己写明**不适用于 PDF 输入**"—— **无此句**；相反 `README.md:20` 明写 `Parse the source (.eml / .html / .pdf text layer)`，PDF 是它宣称的受支持输入。原稿据此贬低对方适用性的论据**不成立**，删除。
> 2. 原稿称 smask 去重"被其 pitfalls 明确记录"—— smask 只出现在 `docs/email-translation-goldman-two-sessions-case.md:62,316-318`；`pitfalls-index.md:361` 的 P5 自己用的恰恰是朴素 `tail -n +3 | wc -l`。**即：对方并没有这条纪律，D1.2 是 UBT 要替它补的，不是从它借的。**
> 3. 原稿 D1.1 的"容差 ≤ 1pt"无出处；`pitfalls-index.md:359-360` 只说 "Must MATCH"。改为"页尺寸严格相等"。
> 另注：其配方内 `html_parse`、`translate.py`（`README.md:62,67`）**在树内不存在**，是对方自己的悬空引用。

本次在 `chapter-3.pdf` 上把它临时改造到 PDF 场景，跑通了"原页位图化 → 白匣遮盖 → 中文回填"的覆盖式链路，最终**失败**（见 §6 附录）。

本文档不复述那次失败，而是回答一个问题：**这次实践里，哪些纪律是 UBT 真正缺的？**

### 1.1 先校正：UBT 已经做过的，不在本文档范围

为避免重复建设，以下能力**已存在**，本文档不提议重建：

| 能力 | 现有实现 |
| --- | --- |
| 渲染后分层视觉门（T0 确定性 / T1 结构 / T2 采样 VLM） | `ubt/adapters/pdf/visual_gate.py` |
| 零 token 全管线彩排 | `ubt/core/engine/dry_run.py` |
| 黄金语料 + 方向感知 KPI 回归门 | `tests/baselines/` + `ubt/core/metrics/compare.py` |
| 公式渲染视觉见证（结构与源裁剪比对，fail-open） | `ubt/adapters/pdf/formula_witness.py` |
| 术语圣经 / TM / 分层记忆 | `ubt/core/memory/` |
| 0-token 快检、漏译、增译、术语度量 | `ubt/core/qe/` |
| 公式守卫、术语强制、一致性、跨度修补 | `ubt/core/validators/` |
| 事务账本与断点续跑 | `ubt/core/engine/ledger.py`（facade；实现在 `ledger_base.py` + `ledger_mixins.py`，2026-09-22 `a713b0d` 拆分） |
| 质量报告 | `ubt/core/engine/reporter.py` |

`pdf` skill 值得借鉴的，是它**在 `visual_gate` 之外**的那几类检查，以及两条流程契约。下面逐条说明。

## 2. 结论速览

| 编号 | 借鉴点 | 落点 | 优先级 | 成本 |
| --- | --- | --- | --- | --- |
| **D1** | 最终产物的"来源守恒"校验（物证门） | ✅ **已实现**：`ubt/adapters/pdf/artifact_parity.py` + reflow_loop 合并 + A7 式 fail-closed | P0 | 零 token、零新依赖 |
| D2 | 抽取通道交叉见证 | `ubt/adapters/pdf/extraction_witness.py`（新） | P1（**v3 收缩：仅覆盖空格/词边界退化**） | 低（零 token） |
| **D2′** | **字体级 + 产物级公式乱码判据**（本类文件上唯一有效的抽取门） | ✅ **已实现**：`ubt/adapters/pdf/extraction_witness.py` + pipeline Stage 1.4（block 打标 + metadata） | **P0** | 零 token |
| D3 | 资产保真账本 + 计数守恒 | `asset_extractor.py` + `overlay_text.py` | P1 | 中 |
| D4 | 公式保真度从"失败后回退"改为"主动分级" | `ubt/core/policy/formula_fidelity.py`（新） | P1 | 中 |
| D5 | 症状索引式故障图谱（文档） | `docs/PITFALLS_INDEX.md`（新） | P1 | 低 |
| **D6′** | "花钱前"真实样本编译 pre-flight（doctor 幂等修复仍 Proposal） | ✅ **已实现**：`ubt/core/engine/render_preflight.py` + pipeline Stage 2.9 | **P0** | 低 |
| D7 | 交付说明契约（强制声明例外） | `ubt/core/engine/reporter.py` | P2 | 低 |
| **D8** | `ToUnicode` 覆盖率进入 `page_profiler`，作公式密集页风险因子（v3 新增） | `ubt/adapters/pdf/page_profiler.py` | P1 | 低 |
| **D9** | **usage 快照落盘**：现有 `estimate_cost_usd()` 结果写入账本 + 报告 `usage{}`（v3 新增，与 D7 合并实现） | `ledger`（实际形态：`job_meta.usage_totals`）+ `reporter.py` | ✅ **已实现**（2026-09-20 `f621705`，"bill the job, not the process"；含 OCR 通道计价 `e55276c`，无记录时报 unknown 而非假 `$0.00`） | 低 |
| **D10** | **CJK 字体链路改为真实探测**（移植 `resolve_cjk_font()` 式阶梯 + `fc-list :lang=zh` 硬校验；v3.1 新增） | `typst_reconstructor.py` 字体阶梯 + `doctor` | ✅ **已实现** | 低 |
| **D11** | **视觉重排公式 + witness 复核**（agent 路线唯一压在 UBT 之上的能力；v3.1 新增） | `formula_witness.py` + 新视觉通道 | P2 | 中 |

> **v3 编号说明**：D8/D9/D10/D11 是对比工作的产出（原权威定义在已删除的
> `docs/PDF_AGENT_SKILLS_VS_UBT.md` §7–§8，见 git 历史）；自 2026-09-22 起，本文档表格即权威定义。

---

## 3. 逐项设计

### D1. 最终产物的"来源守恒"校验（物证门）

**问题。** `visual_gate` 的 T0 查的是：Typst 源里的禁用 unicode、PDF 页数、空白页候选（当时走 pypdf；现已改为 pdf_oxide 进程内提取，`23f9755`/`3f69833`）。这些是**自证**——它检验管线自身产出的内部一致性。缺的是**物证**：把交付出去的那个 PDF 二进制，与**源文件**做守恒比对。

`pdf` skill（MiniMax v3.0）的验收协议恰好是这四样，且全部只看产物。**出处更正：这四条不在 `pitfalls-index.md` 的 P5，而在 `docs/email-translation-goldman-two-sessions-case.md:286-301`；P5（`:359-367`）只要求页尺寸相等、图片数相等、`pdftotext -layout | head -60` 见目标语言、抽 5 数字 + 3 专名存在。**

```bash
pdfinfo  out.pdf | grep -E "Pages|Page size"   # 页数与页尺寸 vs 源/预期
pdfimages -list out.pdf                        # 图片数 vs 源（smask 感知 —— 此条为 UBT 自加，见上）
pdftotext -layout out.pdf - | head -40         # 最终 PDF 文本层可读、无 tofu
python -c '...'                                # /Annots 里的 URI 链接计数
```

**为什么值得。** 本次实测就是"内部全绿、物证全错"的典型：CSS 漏 `position: absolute` 导致中文层被 z-index:1 的白匣盖住，而页数、页尺寸、编译状态全部正常。**只有"从产物反向提取文本并断言目标语言字符出现"才能发现它。**

**落点。** 在 `visual_gate.py` 的 T0 与 T1 之间新增 T0.5 `artifact_parity`（或独立 `ubt/core/validate/artifact_parity.py`，只吃两个路径、不 import pipeline 模块）。检查项：

1. **页尺寸守恒**：`pdfinfo` 的 Page size 与源/预期**严格相等**（对方原文即 "Must MATCH"，无容差数值）。UBT 做 overlay 时页面几何应当等于源；做 reflow 时等于目标版式——两者都应有**显式期望值**可比。
2. **图片计数守恒**：`pdfimages -list` 源 vs 产物。注意 smask 陷阱——透明 logo 会产生 `image` + `smask` 两行，朴素 `wc -l` 会误报，需按 `(page, object)` 去重后再比。**此纪律为本方推论，非对方立场**（对方 P5 用的正是朴素计数）。`chapter-3.pdf` 实测 19 张，其中 2 张 CMYK JPEG，正是该陷阱的高发场景。
3. **最终 PDF 文本层可读性**：对每页 `pdftotext`，统计目标语言字符占比、`U+FFFD` 替换字符率、`(cid:` 字形串出现情况。**这一层只看产物，不看 Typst 源**——现有 T0 的 unicode 扫描是源级的，抓不到"源干净但渲染后文本层损坏"。
4. **导航完整性**：outline/bookmarks 与 `/Annots` URI 计数；若为多页交付则应 > 0（详见 D7 的契约）。
5. **术语落点抽查**：从 Bible 取 K 个高频术语，断言其在最终 PDF 文本中出现。

**验收标准。** 构造一个"中文被白匣遮盖"的破损 PDF 作为回归样本，T0.5 必须报 FAIL（当前 visual_gate 对它会全绿）。

**取舍。** 保持 `pdf` skill 的 warn-only 风格与 UBT 现状一致；但"产物不可读"应与 A7 一样 fail-closed。

---

### D2. 抽取通道交叉见证（extraction witness）— **v3：范围收缩，主判据移交给 D2′**

**问题仍然成立。** UBT 的 PDF 摄取走 PDFium / Docling 单通道。仓库里虽已用 poppler（`diagram_localizer.py`、`svg_diagram.py`、`docling_render.py`、`visual_gate.py`），但用途都在**渲染与版面侧**，未见用于**抽取正确性的交叉验证**。

**原始实测证据。** 同一页、同一个 PDF，两条通道结果不同：

```text
pdfplumber  (char 级定位)  → Intheimplementationofcircuitsimulators,compactmodelsarepreferredoverother
pdftotext   (默认模式)     → In the implementation of circuit simulators, compact models are preferred over other
```

**空格全部丢失。** 这类退化不会让 `fast_pass.py` 的长度比或数字集检查失败——它改变的是词边界，正是"物理意义在，但 token 已经不是原句"的情形。**静默传播到初译，QE 也判不出来。**

**落点（收缩后）。** 新增 `ubt/adapters/pdf/extraction_witness.py`，在 Ingest 之后逐页比对 UBT 主通道 vs poppler `pdftotext`（默认模式，**不要** `-layout`）：

- 非空白字符数（容忍 ±5%）
- 词数（词边界退化会让此指标剧烈变化，是最敏感的探针）
- 数字 token 集合（绝对一致）

任一超阈值 → 该页标脏，进 triage 而非继续翻译。

**⚠ v3 关键限定：D2 的能力边界远小于原稿假设。** 在 `chapter-3.pdf` 上重测五条抽取通道，公式乱码计数**完全一致**：

| 通道 | 全文字符 | 错解码字形 | 词数 |
| --- | --- | --- | --- |
| `pdftotext` 默认 | 47,373 | **279** | 8,126 |
| `pdftotext -layout` | 70,939 | **279** | 8,169 |
| `pdftotext -raw` | 46,194 | **280** | 7,294 |
| `pypdfium2`（UBT 快路径） | 47,670 | **279** | 7,743 |
| `pypdf` | 47,602 | **277** | 8,104 |

20 / 26 页受损、共 275 处（真实样例：`Nch ¼ 2 % 1018 cm-3` 实为 `Nch = 2 × 10¹⁸ cm⁻³`）。根因：12 支承载正文/公式的 Adobe `AdvP*` 字体**完全没有 `/ToUnicode`**，poppler / PDFium / pypdf **共用同一套 MacRoman 回退启发式**——所以它们错得一模一样。

**结论：对"公式被解成乱码"这一类（也是本文件最严重的一类）缺陷，D2 的交叉比对会判"干净"然后照常翻译。** D2 只能保留其原始动机（空格/词边界退化），不能作为公式乱码防线。

### D2′. 字体级 + 产物级公式乱码判据（**v3 新增，取代 D2 的主判据地位**）

零 token、纯本地、**已在两份文件上验证**：

```python
ARTIFACTS = "¼ðÞ½Œœ"                                   # MacRoman 回退残留区
A(page) = any(ch in ARTIFACTS for ch in text)            # 确认判据（高精度）
B(page) = any(font lacks /ToUnicode) and non_ascii(text) > 0   # 风险判据（根因，高召回）
# 采用：B 作先验 → A 作断言 → 二者不一致时降级为"该页必须看图"
```

| 判据 | chapter-3（受损） | chapter-1（对照） |
| --- | --- | --- |
| **D2 原设计**（通道差异） | **0 / 26 命中（失效）** | — |
| **A** 确认判据 | **20 / 26，275 处** | 1 / 13，2 处 |
| **B** 风险判据 | 25 / 26 | 11 / 13 |

实现要点：B 需排除合法非 ASCII（`— – ' ' " " © ® µ β ψ`），否则噪声过高；上表 A 的 275 是**未做白名单排除的上界**。落点建议与 D2 合并为同一模块（`extraction_witness.py`），但**导出两个独立结论**：`degraded_by_channel_disagreement`（D2）与 `degraded_by_font_encoding`（D2′），因为二者的修复路径不同——前者换通道，后者**必须**走视觉/OCR 或像素回退，换库无效。

**为什么这条对 UBT 特别重要。** UBT 的核心成本风险是"花钱之后才发现抽取错了"。这两个见证器零 token、纯本地，把该风险前移；而 D2′ 覆盖的正是本类文件上**唯一真实存在**的静默退化。

**验收标准。** (a) 注入一个"去除所有空格"的文本层变体，D2 必须命中该页；(b) 直接以 `chapter-3.pdf` 为样本，D2′ 必须命中 ≥ 20 页；(c) 以 `chapter-1.pdf` 为对照，D2′ 误报 ≤ 2 页。

---

### D3. 资产保真账本 + 计数守恒

**问题。** `asset_extractor.py` 会把图**重渲染为高分辨率 PNG**。这在 reflow 场景合理；但对 overlay 引擎（像素级覆盖回填），重渲染会**破坏原图保真**：CMYK JPEG、ICC profile、透明 smask 经 PIL 往返可能退化。

**借鉴的纪律。** `pdf` skill 的三条硬规则：不 PIL 后处理、`cid:`/`data-uri` 引用原样保留、计数守恒。其 pitfalls 还记录了两个具体坑：图片数不一致 = P0 失败；经 Pillow/base64 重编码会丢透明与 EXIF。

**落点。**

1. **资产账本**：`asset_id → {source_hash, output_hash, transform}`，`transform ∈ {passthrough, rasterized, vectorized}`。
2. **计数守恒断言**：并入 D1 的 T0.5，smask 感知。
3. **交付指标**：质量报告输出 `passthrough_rate`。

**验收标准。** overlay 单语任务的 `passthrough_rate` 应为 100%；任何 < 100% 必须能在报告里找到对应的显式降级声明。

---

### D4. 公式保真度：从"失败后回退"改为"主动分级"

**问题。** UBT 的机制已经完备：`formula_witness.py` 做渲染结果与源裁剪的结构比对，失败则换回源图（`TypstReconstructor._witness_math_lines`，无损）。**缺的是事前决策规则**——哪些公式压根不该尝试重排，而不是等见证器报错再回退。

**借鉴的框架。** `pdf` skill 这条路线对公式的处理是"**根本不动它**"：公式就是原页像素，零解析、零 OCR、零重建，因此**数学上 100% 等于原文**。代价是公式不可选、不可搜、不可编辑、不重新编号。这个取舍框架是有价值的。

> **v3 限定。** "不动它"是**本方对其 overlay 路线的推论**，对方文本中无对应表述：`translate-preserve-layout` 全目录（README + skeleton + terminology + case）对 formula/equation/math **零提及**，其 `STRICTLY PRESERVE` 清单（`pitfalls-index.md:340-344`）只有数字/日期/货币/百分比/专名；`README.md:40-41` 只承诺**图片字节级保留**，并未延伸到公式。因此 D4 是 **UBT 自建的档位体系**，不是移植品——这不影响它该做，只影响它的署名。

**落点。** 新增 `ubt/core/policy/formula_fidelity.py`（与 `layout_policy.py` 并列），三档：

| 档位 | 行为 | 适用 |
| --- | --- | --- |
| `editable` | 全部 MathJax 重排，不换图 | 需要可搜索/可编辑 PDF 的场景 |
| `witnessed` | 重排 + 见证失败即换原图（**当前默认行为**） | 通用 |
| `pixel` | display 公式一律原图切片 | 公式密集且正确性优先于可编辑性 |

决策依据可量化：公式是否被正文按编号引用、识别置信、图元复杂度、是否跨页。

**验收标准。** 同一本书跑三档，产出 KPI 对比表：可编辑率 / witness 命中数 / 产物体积 / 编译成功率。

**注意。** 这**不是**引入 `pdf` skill 的做法，而是把它作为一种**已实现的档位**正式暴露出来并给出选择依据。UBT 现状（`witnessed`）本来就是更优的默认值。

---

### D5. 症状索引式故障图谱（文档）

**问题。** UBT 有 `tests/baselines/`（语料）和 `ubt/core/qe/defect_taxonomy.py`（缺陷分类），但**没有一份"症状签名 → 根因 → 修复 trace"的可查索引**。`output/` 下十几份 `chapter-3*_quality_report.md` 是一次性产物，调试经验没有沉淀。

**借鉴的做法。** `pdf` skill 的 `docs/pitfalls-index.md`：10 个典型 case，每个带"匹配签名（样例查询）→ 过去的失败 → 推荐的完整 trace"，并要求**逐字复用**而非重新推导。

**素材现成。** `formula_witness.py` 的 docstring 里那段 2026-09-16 校准笔记就是范式——它记录了"在 chapter-3 的 68 个 display 公式上，存活的 flag 是哪两个、为什么不把 ink density 用作判据"。这种知识现在散落在代码注释里。

**落点。** `docs/PITFALLS_INDEX.md`，条目格式：

```markdown
### <症状签名>
- 首次观测：<日期> / <文档>
- 根因：<机制>
- 推荐 trace：<命令或调用序列>
- 回归用例：<tests/baselines 或 fixtures 路径>
- 状态：open / mitigated / closed
```

**首批可入库条目**（均可从现有证据提炼）：

1. 抽取层空格丢失（D2 的实测）
2. display 公式一行被重排成两行 —— `formula_witness` 已记录 `pdf_main#b0178`
3. 公式凭空多出两项 —— 已记录 `pdf_main#b0176`
4. 覆盖层被遮盖块压住（z-index/静态定位失效）—— 本文档 §6 实测
5. **公式被解成拉丁-1 乱码**（`¼`→`=`、`$`→`−`、`%`→`×`、`ð/Þ`→括号）—— 签名：`pdftotext` 输出含 `¼`/`ð`；根因：数学字体缺 `/ToUnicode`，各库共用 MacRoman 回退；trace：D2′ 判据 A/B，**换抽取库无效**，必须走视觉或像素回退（对比文档 §3）
6. **Stage 6 编译失败** —— 签名：`the character \`#\` is not valid in code`；根因：`#box[...](...)` 被解析成链式调用切入 code mode；状态：**closed（`a98a315`）**。**本条附一条规约：断言"待修"之前必须做一次最小复现验活**（本文 v3 就是没查 mtime/`git log` 而写错）

**另注（顺手修）。** `ubt/adapters/pdf/visual_gate.py` 的 docstring 引用了 `docs/pdf-layout-comparison-and-sota-architecture.md`，该文件在仓库中**不存在**（全仓仅该 docstring 引用）。要么补文档，要么修引用。

---

### D6. `doctor` 幂等修复 + "花钱前"硬闸门 — **v3：P1 → P0，风险已实测发生**

**问题（原稿）。** UBT README 自承：`typst` 缺失会导致"PDF 任务在 Stage 6 失败（**此时整本书的 token 已花完**）"。这是账单末尾的地雷。

**⚠ v3 实测：这颗雷已经响了，两次。**

```text
.ubt/logs/ubt-tui-20260918-111444.log:975   2026-09-18 11:38:56 [ERROR] Pipeline failed for job job_fc1d7bd7b799_zh: Typst compilation failed (exit code 1)
.ubt/logs/ubt-tui-20260918-114201.log:559   2026-09-18 11:43:17 [ERROR] 同上
```

同一账本（`job_fc1d7bd7b799_zh`）显示：209 块已在 **3 分 03 秒**内全部译完（201 `mtqe_passed` / 5 `repaired` / 2 `blocked_human` / 1 `needs_human`），随后死在 Stage 6 —— **token 全花完、零产物**，与原稿描述的情形逐字吻合。

**且成因不是"typst 缺失"**：本机 `typst 0.15.1` 在装（`~/.cargo/bin/typst`）。真实成因是**生成代码自身缺陷**：

```typst
error: the character `#` is not valid in code   ┌─ tmp/output/chapter-3_bilingual.typ:137:270
warning: no whitespace before raw text              #text(...)[为了求解 #box(...)[#image("…/inline-….svg", ...)]，… ````$E_"x"$```` = 0， …]
```

(a) 行内数学的 SVG 片段注入 content 参数时 `#` 未转义；(b) 数学被四反引号包成 **raw text** 而非 Typst 数学模式（即便编译通过也会渲染成等宽正文）。

> **✅ v3.1：该缺陷已修，D6 的立论不变、且更硬。** 同日 **11:50:31** 由 `a98a315 fix(typst): decouple inline box/image calls followed by parentheses to prevent code-mode syntax crash` 修复 —— 真实根因不是"漏转义"，而是 **`#box[...](...)` / `#image(...)(...)` 被 Typst 解析成链式调用、切入 code mode**；修复为 `_INLINE_CALL_COLLISION_RE` + `_decouple_inline_box_calls()`（把紧随的 `(` 转义成 `\(`，版式不变），并给 `_degrade_failing_line` 的 `$/…/$` 加 `(?<!`)…(?!`)` 守卫。**本轮复核**：`pytest tests/unit/test_typst_fragments.py tests/unit/test_typst_fallback.py -q` → 12 passed；那份 `.typ` 现在 `typst compile` → **exit 0**（10.6 MB PDF）。
>
> D6 的价值恰恰因此更硬：**一个语法层的局部缺陷就能烧掉整本书的 token**，而"typst 在不在"这种探测对它零信息量。**pre-flight 必须是真实样本编译。**
>
> **⚠ 方法论教训（登记为 D5 首批条目 5）**：本文 v3 曾把它写成"未修的 P0"，因为只查了日志与账本、**没查产物 mtime（11:50:19）与 `git log`**。规约：**断言"某缺陷待修"之前，必须做一次最小复现验活** —— 日志记录的是"曾经失败"，不是"现在仍失败"。

**这条实测把 D6 的范围从"探测二进制存在"扩成"探测渲染器能否真的编译本任务"** —— 只查 `which typst` 对本次两次失败**完全无效**。

**借鉴的做法。** `pdf` skill（MiniMax v3.0）的 `make.sh check` → `fix` → 再 `check` 直到全绿：幂等、可反复运行、`fix` 传播真实 pip 失败而不"假绿"。

> **v3 限定（不要高估对方）。** 其 `make.sh:140,142` 把 `pdfinfo` / `qpdf` 定为 **WARN-only**（缺失仍继续），所以"花钱前硬闸门"在对方实现里**并不硬**；且 **P5 没有 `make.sh check` 前置步**（P1/P2/P3 都有，`pitfalls-index.md:57-59,119,175`）—— 属对方自身不一致。可借的是**幂等 check/fix 结构与"fix 不假绿"的注释纪律**，不是它的实际强度。

**落点。**

1. `ubt doctor --fix`：幂等修复，而非仅报告。
2. 把**渲染引擎可用性**提升为 Stage 3（Draft）**之前**的硬闸门，非零退出 + 明确指引。
3. **把 `dry_run` 的渲染探针接入正常路径。** 已核对：`dry_run.py` 只 mock **provider**（`DryRunModelProvider`）与 **QE**（`MockQERunner`），**不 mock 渲染器**——其 docstring 明确写着"a rehearsal should prove the *plumbing* reaches a rendered artifact"。也就是说，一次 `--dry-run` **本来就会**在 0 token 的前提下暴露渲染问题。
4. **（v3 新增，关键）pre-flight 必须是"真实样本编译"而非"二进制探测"**：在 Stage 3 之前，用**前 K 块**（K≈5，含至少 1 个带行内数学的块）走一遍 Typst 生成 + 编译。这一步零 token、纯本地，能在花钱前复现本次的 `#` 转义与 raw-text 两类失败。

**验收标准。** (a) 临时移开 `typst`，`ubt translate` 必须在**任何 token 消耗之前**非零退出；(b) 人为在模板里注入一个未转义 `#`，pre-flight 同样必须在花钱前非零退出（当前实现两项都过不了）。

---

### D7. 交付说明契约（delivery note as contract）

**问题。** `reporter.py` 的字段已经丰富（含缓存命中率这类 TCO 指标）。但"**没做的事**"没有强制字段，容易静默降级。

**借鉴的做法。** `pdf` skill 要求最终交付说明固定包含：输出路径、页尺寸、页数、图片数、链接数、术语核对结果，并**显式声明例外**——例如"单页表单可省略 TOC，须说明保留了什么导航"。

**落点。** 在质量报告中增设两个强制段：

- `navigation{}`：outline 数、URI 链接数、TOC 目标是否可达。
- `exceptions[]`：被跳过的、被降级的（如资产 `transform != passthrough`、公式回退到 pixel、某页抽取标脏）。

关键是**强制**：写不出来就不能算交付完成。

---

## 4. 不借鉴 / 已正确 / 兜底通道

> **证据强度限定（必读）。** 本节结论基于**一次未优化的原型实验**（2026-09-18，见 §6）。该原型为"快速验证链路能否跑通"而写，未做任何体积或性能优化。因此本节论据为**方向性，非定量性**——§6 中的 10.6 MB / 8.4 MB 是**该实现**的代价，不是任何技术路线的固有成本。凡涉及定量断言处已逐条标注。

### 4.1 已正确，保持不变（不必借鉴，也不要退回）

**区域级像素回退（region-level pixel fallback）**

把"光栅"整体判为不借鉴是**过度概括**。必须区分两件事：

| 做法 | 判断 |
| --- | --- |
| **整页**光栅作**主基底** | 不借鉴（见 §4.2 ①） |
| 对**无法重排的区域**（图表/公式）嵌入**源像素** | **已正确，保持** |

后者正是 UBT 现有做法：`formula_witness.py` 见证失败即换回源图（其 docstring 归因于 `TypstReconstructor._witness_math_lines`，标注无损），`asset_extractor.py` 抽取图表资产。**这个内核与 `pdf` skill 的"不动它 = 零风险"是同一思路，UBT 已实现且做得更细（有见证器 + 回退决策）**，无需借鉴，只需保持。

同理，UBT 的 `overlay_text.py` 是**真实文本 + Typst 数学模式**渲染，并非像素覆盖，文字可选可搜。因此"文字被像素化导致不可选、不可搜"**对正文不成立**：被像素化的只有图表与公式。真实差异仅在**公式是否可搜索**。

### 4.2 不借鉴

**① 整页光栅打底 + 白匣覆盖作为主基底**

理由（已按证据强度重写）：

1. **架构性**：整页位图使最终 PDF 的**文本层失去页面结构**，TOC 因此只能程序化**外挂合成**（如 pypdf `add_outline_item`），无法从产物自动推导。这不构成绝对阻断，但要多维护一条合成链。
2. **工程性**：整页位图 base64 内联会再膨胀约 1/3；逐页 PNG 的冗余远高于"仅图表取像素"。
3. **定量数据仅作方向性参考**：原型 150 dpi 整页 PNG 内联 → HTML 10.6 MB / PDF 8.4 MB（26 页）。**此数字是该未优化实现的代价，不是技术路线固有成本**——改用外链图片、仅区域取像素、降低 DPI 或对文字区二值压缩，均可显著下降。

> **一致性提醒（原表此处的错误）。** 原表曾以"无法生成可点击 TOC"否定光栅路线，但 **UBT 当前两条路线都没有 PDF 大纲/书签**（全仓 `grep add_outline|bookmark` 零命中）。该能力对双方都是缺口，**不能**用作否定对方的差异点；它只应作为 **D1.4 / D7** 的待补项。

> **v3 补充：不要靠"体积/性能"否定 overlay 路线，要靠覆盖率。** 在密排书版上，覆盖回填路线**双方实测都不可交付**：
> - 对方侧：`MIMO/scripts/overlay_text.py:198` 是单行 `drawString`（无换行、无 autofit），`:149-165` 断言**同页任意两框不得重叠**；`PDFKIT/overlay_text.py:196-207` 字号 `max(6, rect_h*0.9)` 且无换行 → 整段变成一个超宽白块。
> - **UBT 侧同一结论**：当时的质量报告实测 Render Coverage ≈42%（88/209 on page）、116 块露出源文（chrome×8、non_prose×74、policy×34）。**2026-09-26 对齐**：该 `output/` 报告已不在仓库；可复跑的证据是 `scripts/rigid_coverage_sweep.py`（正文数字已按引用规约 §3.6 撤除）。
>
> 所以正确表述是"**overlay 路线在 26 页书版正文上已被双方实测共同排除**"，而不是"overlay 路线成本太高所以不用"。附带好处：这条否定不再依赖那个被 §4 限定过的 8.4 MB 数字。

**② Chromium 作为 PDF 渲染依赖**

**结论保留，理由修正。** 原表以"约 114 MB 浏览器二进制"为主要论据，这是**次要因素**（一次性安装、有缓存，对桌面工具不致命）。真正的理由是：

1. **确定性**：需等 JS settle、字体回退链不可控，且 `@page` 与 CLI 参数会互相覆盖——本次即被迫自写 `render.cjs`，因为 `pdf` skill 自带渲染器**只认 A4/Letter**，不支持任意页面尺寸。
2. **运维面**：headless 需拖 sandbox、字体、依赖一整套，失败模式多于 Typst。

**例外，不得一刀切**：UBT 已有 HTML 输入适配器（`ubt/adapters/html/adapter.py`）。若走 HTML **输入**路线，Chromium 是 CSS 语义的参考实现，届时值得引入。本条仅否定"为 **PDF** 渲染引入浏览器"。

**③ EML 中心的模板结构**

`translate-preserve-layout` 的 metadata/logo/team-card/免责声明骨架是邮件报告专用，PDF 场景不适用（该模板 README 自己写明）。其分段类型学（metadata/heading/body/list_item/caption/footnote/table_cell）本可迁移，但 UBT 的 `ubt/core/ir/models.py` 已有 `BlockType`，已覆盖，故不单列。

> 措辞修正：原表称"可迁移的**只有**验证纪律与资产纪律（D1–D3）"过于绝对。准确说：**验证纪律**见 D1/D2/D5/D7，**资产纪律**见 D1/D3，**抽取纪律**见 D2，均属本文档范围，不限于 D1–D3。

**④ 照抄式 canonical query 模板**

`pdf` skill 的 `docs/pitfalls-index.md` 要求"匹配成功后**逐字复用**该 canonical query"，那是**给 LLM 提示词工程用的模式**，依赖模型对模板的模仿。对确定性管线没有意义：UBT 的 `dry_run.py` / `metrics/compare.py` 是确定性且可回归的，方向正确。可迁移的只是其**索引结构**（见 D5），不是"照抄执行"本身。

### 4.3 不作生产主路径，但保留兜底通道

**agent 即兴编排（agent-improvised pipeline）**

原表用"本次两个失败点"否定 agent 式管线，**立论不成立**——两个失败点都证明不了该结论：

| 失败点 | 真实性质 |
| --- | --- |
| CSS 漏 `position: absolute` | **任何无验证的实现**都会犯，与是否 agent 编排无关。能挡住它的是 **D1 物证门**，不是"非 agent 架构"。 |
| 子代理撞 token 上限 | **供应商配额**问题，与架构无关。账本只能让中断可续跑，不能让配额不撞。 |

更关键的是，全面否定会**关掉 UBT 的一个正当逃生舱**。UBT 的自动化建立在启发式解析之上，而 Docling/PDFium 在**畸形文档**上必然有边界；恰恰在那类文档上，agent 逐页视觉判读 + 即兴处理可能是**唯一能完成**的路径。`engine_selector.py` / `router_mode.py` 的存在本身就说明 UBT 承认需要路由。

**修正后的立场：**

- **不**将 agent 即兴编排作为**生产交付主路径**（不可复现、无账本、成本不可测）；
- **应**保留它作为"管线判死"文档的**兜底通道**；
- 其产出**必须强制过一遍 D1 物证门**后方可视为交付——用这条换取灵活性，而不牺牲保证。

---

## 5. 建议落地顺序

| 阶段 | 内容 | 理由 |
| --- | --- | --- |
| ~~**P0-0**~~ ✅ | ~~修 §7.1 的 Typst 生成缺陷~~ → **已修（`a98a315`，11:50，本轮复核通过）**；~~遗留一条**单语全本可复现命令**~~ → 已由 `scripts/formula_matrix.sh` + `tests/integration/test_formula_engine_matrix.py`（公开 CI 停用期间改按需运行，见 `docs/guides/CI_AND_QUALITY_GATES.md`）闭合（2026-09-22 注） | 止血项已消除 |
| **P0** | D1 + **D2′** + D6 | 三者零 token、零新依赖。D1 堵"内部全绿、产物全错"；D2′ 堵本类文件上**唯一真实存在**的静默退化（D2 原设计对此零命中）；D6 已被两次实测证明会烧完整本书的 token |
| **P1** | D2（收缩后）+ D5 + D3 + D8 | D2 退守空格/词边界一类；D5 沉淀已有调试知识；D3 对齐 overlay 引擎的保真目标；D8 把 `ToUnicode` 覆盖率变成规划输入而非事后诊断 |
| **P1** | D4 | 需要 KPI 对比实验，工作量中等 |
| ~~**P1**~~ ✅ | ~~D9~~ **已实现**（2026-09-20 `f621705`） | usage 落盘进 `job_meta.usage_totals`，形态与提案的 `usage_snapshot` 表不同、实质相同 |
| **P2** | D7 | 报告字段调整，小改动 |

---

## 6. 附录：本次实测证据（2026-09-18）

**源文件。** `docs/chapter-3.pdf` — 26 页，540 × 665.972 pts（B5 书版），文字原生，含 19 张去重内嵌 JPEG（图表，**其中 2 张为 CMYK JPEG / 300 dpi**）。

> **v3 更正（影响题面理解）。** 该文件 **不是 LaTeX 产物**：`Creator: PDFPatcher 1.1.0.4572`、`Producer: macOS … Quartz PDFContext`；27 支字体全为 Adobe `AdvP*`/`AdvOT*` + Arial + `SymbolMT`，**无一支 Computer Modern（CMMI/CMSY/cmr10）**，且 **12 支完全没有 `/ToUnicode`**，`Form: none`，poppler 报 12 次 `Illegal annotation destination`。故"保持 LaTeX 公式"只能理解为**对输出的要求**（公式必须是正确数学排版），不是"保住输入里的 LaTeX 源码"—— 后者不存在。完整档案存于 git 历史（原 `docs/PDF_AGENT_SKILLS_VS_UBT.md` §1，已删除）。

**抽取退化证据（D2 的来源）。**

```text
pdfplumber chars → "Intheimplementationofcircuitsimulators,compactmodelsarepreferredoverother"
pdftotext 默认   → "In the implementation of circuit simulators, compact models are preferred over other"
```

**v3 追加：D2 的同类还有更严重的一种，且 D2 抓不到（→ D2′）。** 五条抽取通道在同一文件上的公式乱码计数几乎相同（279 / 279 / 280 / 279 / 277），20/26 页受损：

```text
p3: … ψ B ¼ V tm lnðN ch =ni Þ. Note that Eq. (3.1)
p4: Nch ¼ 1 % 1015 cm$3, TFIN ¼ 20 nm, tox ¼ 1 nm, and Vch ¼ 0 V …
p9: p##################### ψ s ðyÞ$ψ B $Vch ðyÞ
→ 真实内容：ψ_B = V_tm·ln(N_ch/n_i)   /   Nch = 1 × 10¹⁵ cm⁻³, TFIN = 20 nm, tox = 1 nm
```

错读映射：`¼`→`=`、`ð/Þ`→`(`/`)`、`$`→`−`、`%`→`×`、`ffi`→`≈`、`#`串→引导点。**根因是字体缺 ToUnicode 后各库共享同一 MacRoman 回退先验**，因此换库无效、交叉比对恒等。详见 D2′ 与其三项验收标准。

**失败 1 — 覆盖层被遮盖块压住（D1 的来源）。**
`generate2.py` 的 CSS 只给 `.cover` / `.banner` / `.bg` 写了 `position: absolute`，`.title` / `.big-num` / `.body-text` / `.footer` / `.pagenum` 均遗漏。静态元素上 `left/top` 不生效，且 **`z-index` 对静态元素无效**，于是中文层落回正常文档流，被绝对定位的白色遮盖块（`z-index: 1`）**盖在下面**。

产物表现：英文原文照旧可见、中文整块消失、蓝色横幅内中文折行溢出。
而 `pdfinfo` 显示 **26 页 / 540 × 666 pts，与原版逐项吻合** —— 页数、页尺寸、编译状态全部正常。**自证式门禁对此完全无感**，这正是 D1 的必要性证据。

**失败 2 — 翻译子代理中断。**
后台 worker 报 `已达到 Token Plan 用量上限`，26 页中仅完成 3 页（手译）。无账本可续跑，只能重来。

**体积数据点（§4.2 ① 的方向性依据，非技术路线固有成本）。**
150 dpi 整页 PNG base64 内联 → HTML 10.6 MB → PDF 8.4 MB（26 页）。**该数字量的是这次未优化的原型实现，不是"区域像素回退"或任何技术路线的固有开销。**

**产物特征观察。**
`docs/` 现有中文产物的 Producer 分别为：`chapter-3-zh.pdf` = LaTeX/xdvipdfmx（24 页 A4）、`chapter-1-zh.pdf` = LaTeX with hyperref、`chapter3-中文翻译.pdf` = WeasyPrint 70.0（28 页 A4）。**无一份是 Typst 产物** —— 说明历史上真正跑通过的是 LaTeX 与 HTML 两条路。建议尽快用一份完整章节把 UBT 的 Typst reflow 路径验实，使 §2 的评分基于实测而非设计。

> **v3 更新：这条建议已经有了答案，而且答案是负的。** 账本记录 `job_fc1d7bd7b799_zh`（chapter-3，209 块）状态为 `failed`，日志两次于 Stage 6 Typst 编译 exit 1（见 §7.1）。**在修好该崩溃之前，"用一份完整章节验实 Typst 路线"这个动作本身就会失败。**

**悬空文档引用（v3：已修；2026-09-22 再清）。** 原引用的 `docs/pdf-layout-comparison-and-sota-architecture.md`（`visual_gate.py` docstring）、`docs/formula-engine-acceptance-plan.md` 与 `tests/fixtures/formula_ocr_damage.json`（`scripts/formula_matrix.sh`）**均不存在**。曾改指 `docs/PDF_AGENT_SKILLS_VS_UBT.md`，该文档亦已于 2026-09-22 删除——相关代码注释（`visual_gate.py`、`formula_matrix.sh`）已改为自含表述。`formula_ocr_damage.json` **不会再补写**——D2′ 已实现，用例内联在 `tests/unit/test_formula_ocr_damage.py`（`tests/fixtures/` 已于 `13568a7` 删除）。

---

## 7. v3 新增实测（2026-09-18 第二轮；原载已删除的 `PDF_AGENT_SKILLS_VS_UBT.md`，数据存于 git 历史）

对比工作扫出的、属于 UBT 自身的三条事实：

| # | 事实 | 出处 | 影响 |
| --- | --- | --- | --- |
| 7.1 | chapter-3 最近两次运行**在 Stage 6 Typst 编译失败**（11:38 / 11:43），而 209 块已在 **3 分 03 秒**内全部译完 → token 花完、零产物。成因非"typst 缺失"（本机 0.15.1 在装），而是生成的 `.typ` 里 `#` 未转义 + 数学被四反引号包成 raw text | `.ubt/logs/ubt-tui-20260918-114201.log:559`；`tmp/output/chapter-3_bilingual.typ:137`；`.ubt/ledgers/job_fc1d7bd7b799_zh.sqlite` | D6 升 P0；pre-flight 必须是**真实样本编译**而非二进制探测 |
| 7.2 | 五条抽取通道在同一文件上的公式乱码计数**几乎相同**（279/279/280/279/277）→ 交叉见证对这类缺陷零命中 | 对比文档 §3.1 表 | D2 收缩，D2′ 提出并验证 |
| 7.3 | **成本计量已接通但不落盘**：`export.py` → `estimate_cost_usd()` → `reporter.py` 全链在；20 份报告为 `$0.00000` 的真实原因是所用模型 `muse-spark-1.3-contributor` / `free-best` 在价表里解析为 `(0.0, 0.0)`（`pricing.py` 最长前缀落空）。**当时缺的是持久化**：账本三表无任何 usage/latency/cost 列 → 成本只活在进程内，跑完即失。**（2026-09-22 注：已于 `f621705` 解决，usage 落盘进 `job_meta.usage_totals`，见 D9 行）** | `sqlite3 .schema`；`resolve_model_prices()` 实测 | 新增 D9；§6 金额标"推导"的理由 |

**耗时锚点（从账本时间戳反推，此前未有人记录）**：chapter-3 209 块 = **3m03s**；chapter-1-zh 92 块 = **11m10s**；chapter-1 84 块 = 2h05m（含人工中断，**不可当纯吞吐引用**）。

**最省力的下一步**：`bash scripts/formula_matrix.sh --pages 1-6`（默认 `--dry-run`，**零 API 成本**）一次跑完 6 个 math_backend × formula_render 组合，即可同时补齐 7.3 的时延数据与 D4 的三档 KPI 对比表。
