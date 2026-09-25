# 类 Jev 类型化决策模型作为翻译 QE 层评估（含 Jev 本体引入决策）

> **文档类型**：技术调研 / 模型引入决策支持
> **日期**：2026-09-25 · **基线**：当前工作区（仓库未含 `.git` 元数据；按 `docs/README.md` §3 一律以符号名引用）
> **评估对象**：
> 1. **TypeSafe AI Jev**（System One Model，闭源云 API，2026-09-15 发布）；
> 2. **开源复刻权重**（Open-Jev / SemIf / JEV-CPU / NanoJev / JevLite / Jev-Style 等）；
> 3. **"类 Jev"这一 readout 范式本身**——非生成、一次前向、类型化、带校准置信度的决策头——作为 UBT 的 QE 层。
> **信息来源与证据等级**：
> - 一手（高）：TypeSafe AI 官方博文；arXiv 论文（`2609.26758`、`2609.24052`、`2609.23959`、`2609.22793`、`2603.10775`、`2510.20780`、`2510.08870`、`2609.13611`、`2608.20925`）。
> - 社区（中，非同行评审）：`awesome-jev` 索引、`jev-fidelity`/`GeekLink Jev Subtitle Translator` 仓库自报数据、Reddit/X 讨论。
> - 结论与数字按来源标注，社区自报数字不外推为事实。
> **状态**：评估完成。**默认建议 = 形态借鉴并自建候选进基准门；Jev 云 API 仅作教师/实验，不进生产依赖；不替换 UBT 现有分层 QE 架构。** 本文档仅为决策支持，未改动任何源码。

---

## 0. TL;DR 决策摘要

**核心问题**：类 Jev 的类型化决策模型，能否成为翻译 QE 的"最终一次性方案"，取代 UBT 现在这套复杂 QE 设计？

**答案**：**不能取代，也不存在"最终一次性方案"。**

| 层面 | 结论 |
|---|---|
| Jev 本体（TypeSafe 云 API） | ❌ **不进生产**。闭源无自托管授权、厂商容量风险（发布后两次暂停注册）、文本外发第三方、无许可自持 |
| 开源复刻权重（Apache-2.0 / MIT） | ⚠️ **可作实验基线**，但它们的训练数据是 Banking77/SST5/BoolQ 等公共分类任务，**不是翻译 QE**，属"复刻形态"而非"复刻能力" |
| 类 Jev readout 形态（自建决策头） | ✅ **值得做**：作为 UBT QE 级联中"比免费规则懂语义、比 LLM judge 快/便宜一个数量级"的**中间层（Tier 1.5）** |
| 取代 UBT 分层 QE（`FastPassFilter` / `HeuristicQERunner` / `LLMJudgeQERunner` / `SubprocessQERunner`） | ❌ **不成立**。见 §6：它替代不了确定性精确门，也吸收不了缺陷分类法/多语言/块类型/成本分级/`fail-closed` 人工路由 |
| "最终最好的 QE 方案" | ❌ **不成立**。文献与社区都收敛到同一形态：**模型判定 + 确定性门 + 校准弃权 + 人工复核预算**的级联；类 Jev 只是这条路径上的一个新 readout，不是终点 |

**一句话**：类 Jev 给 UBT 的净收益是**一个更便宜、更快、可校准的语义判定 readout**，用来插进现有级联的中间层；它**既不取代底层确定性规则，也不取代整个 QE 栈**。

---

## 1. Jev 是什么（一手：TypeSafe AI 博文，2026-09-15）

**定位**：不是一个聊天模型，而是"软件可直接消费的模糊决策函数"。输入非结构化 `state` + 类型化问题，输出类型化概率决策，**不生成任何 token**。

| 维度 | 事实 |
|---|---|
| 输出类型 | `Choice`（≤255 候选的分布）、`Score`（2–10 有序档位的分布 + 期望）、`Noul`（p(yes) 布尔） |
| 训练方法 | **RLCD**（Reinforcement Learning for Calibrated Decisions） |
| 关键卖点 | 不生成字符串 → **结构上不可能类型错误/幻觉**；每次输出都带校准置信度 |
| 价格 / 速度 | **$0.042 / M input token，output 免费**；端到端 **70–500 ms**（自称比 LLM 快 40–200×、省约两个数量级） |
| 模态 | **纯文本**（官方 FAQ 原文 "not on images (yet…)"） |
| 获取 | 闭源 API；经 OpenRouter / Cloudflare AI Gateway / Workers AI(`typesafe/jev`)；发布后两次暂停注册 |

官方自认的两条**诚实边界**（对 UBT 尤其重要）：
1. 193.6× / 444.6× 的数字来自 4 个 workflow eval，参考答案是 GPT-6 Astra + Fable 5.1 的平均，"可能高估也可能低估"；
2. **"no type errors" 是数学必然，但不等于回答正确**——类型安全 ≠ 校准。

---

## 2. 社区在用它做什么（`awesome-jev` 分类，非同行评审）

生态已数百个项目，全部落在"固定 harness 里的判定角色"，而非开放生成：

| 场景 | 代表 | 做什么 |
|---|---|---|
| Agent 安全门禁（50+） | `jev-guard`、`pi-jev`、`Reflex`、`jev-axi` | 每个 tool call 执行前判"是否越权/破坏性/外泄"，允许/询问/拒绝 |
| 验证与护栏（34+） | `jev-fidelity`、`Sniff Test`、`taste-lint` | 逐句/逐段判定是否保留原意、是否有 AI 味、叠 boolean 阈值 |
| 分类与路由（41+） | `Notra`、`jev-router` | 把品牌分类器从 LLM 换成 Jev Boolean（目标 p50 300ms）；按复杂度路由模型 |
| 评分排序（35+） | `JevGate`、`Canny` | 对 diff/测试结果打分，`.80` 阈值以上变成 review/block |
| 数据标注 | `jev()` PostgreSQL/DuckDB 扩展 | 自然语言行分类，1000 行 ~10s |
| 实时应用 | `NanoJev`、`jev-trader` | 极低延迟决策回路 |
| **⭐ 翻译字幕质检** | `GeekLinkDev/jev-subtitle-translator` | 逐条字幕"源-译"对问 Jev 一个 `Noul`：**这条是否需要人工复核**；本地确定性检查先行 |
| **⭐ 编辑保真** | `klauswg/jev-suite/jev-fidelity` | 逐 fact 问 `Noul`(是否可核查事实) + `Choice`(preserved/equivalent/drift/lost)，置信度 <0.70 转 `REVIEW` |

### 社区两条翻译相关先例的**自报**数据（可作方向，不可作定论）

`jev-fidelity`（2026-09-23 标定，55 样本 / 110 fact 标签）：

| 方案 | 规则 | acc | precision | recall | 弃权 |
|---|---|---|---|---|---|
| 相似度基线 | 关键词重叠 <0.5 → 缺陷 | 0.706 | 1.000 | 0.200 | 0 |
| Jev 单独 | 原始 choice | 0.927 | 0.833 | 1.000 | 0 |
| **组合（上线版）** | **置信度 ≥0.70 门控** | **0.989**（91/92） | **0.975** | **1.000** | 17（15%） |

注入测试 0/20 被翻转。它的架构口号正是 UBT 的纪律：**"Jev classifies, code gates"**、**"degrade to human review, never to pass"**。

> 关键观察：**相似度基线只抓到 20% 的缺陷，而"模型 + 确定性门 + 弃权"抓到 100%（precision 0.975）**。这说明有价值的是**级联形态**，不是"用模型取代一切"。

---

## 3. 主流翻译 QE 的真实图景（论文）

问题里说"主流是 神经 QE → LLM-as-judge"——**方向对，但真实形态是一条级联，且正在发生"LLM judge 反向蒸馏回快速模型"的收敛**。

| 层 | 代表 | 特点 | 证据 |
|---|---|---|---|
| 有参考指标 | COMET-22、MetricX | 需参考译文，离线评测 | WMT metrics shared task 常客 |
| **学习式无参考 QE** | COMETKiwi、xCOMET、MetricX-QE、TransQuest | **非生成、毫秒级、标量分数**；生产主力 | `2602.06546`（CometKiwi 是英文-希伯来最强单模型）、`2510.08870`（SLIDE） |
| **LLM-as-judge** | GEMBA(-DA)、MQM 提示、TransEvalnia | 语义更强、贵、慢、**过度思考/高估、校准差** | `2510.20780`（LRM-as-judge 首次系统分析，需校准思考轨迹）、`Agentic AI Translate 2605.17041` |
| **收敛方向：LLM 当标注器/奖励，训回快速模型** | — | LLM judge 生成 MQM 标注 → 训练 COMET；RL 做 error-aware QE | `2603.10775`（LLM as Annotators for MTQE，训 COMET 达竞争水平）、`2602.08600`（ALOPE-RL error-aware QE）、`2609.22793`（diagnose-then-repair） |

**类 Jev 的坐标**：它不是第三条独立路线，而是**第三种 readout 形状**——在"学习式无参考 QE"（标量回归）与"LLM judge"（生成式推理）之间，提供 **非生成 + 类型化多问题 + 并行 + 校准置信度**。

### 三条必须记住的论文结论

1. **`2609.13611`（In the Blind）**：QE 引导的选择有已知失败模式——**指标会把"目标语言错误但流畅"的输出排到正确译文之上**，必须加确定性语言识别惩罚才压到 0。→ 纯模型判定不能没有确定性护栏。
2. **`2608.20925`（Source-Free MT Evaluation Is Not MT Evaluation）**：充分性必须对照**源文**判定；呼吁把 QE 从"没参考时的备胎"提升为主方案。→ UBT 的 `(src, mt)` 成对判定方向正确。
3. **`2609.22793`（Diagnose, Then Repair）**：把 MQM 评估**结构化到 span 级 + 显式编辑契约**，再做受限后编辑，比"judge-and-refine 一把梭"更可控、漂移更小，COMET-22/COMETKiwi 全面更好。→ **诊断与修复必须分离**，这正是 UBT `score_policy`/`triage` 的设计。

---

## 4. 决定性的新证据：类 Jev 做判定的**能力与边界**

### 4.1 反证（最重要）：`2609.26758` *Type-Safe Is Not Error-Free*

> 决定头**跟随"选项名"而非"绑定到选项名的 rubric"**。

实验：只改选项名到 rubric 的**分配**（问题、state、rubric 文本、选项名集合全部不变），把两个选项从 `0/1` 改名为 `no/yes`：

- 每 100 个判决多翻转 **70.4 个**（95% CI [67.6, 73.1]）；
- **AUC 从 .94 掉到 .23**（判定排序系统性反转，不是"更不确定"）；
- 中性命名影响很小；选项数越多越严重；
- readout 几何相关：mean-pool 的模型家族翻转少 4.1×；
- **整个过程中 type-error rate 始终为 0%。**

→ **对 UBT 的直接含义**：`Choice` 的选项命名/顺序/rubric 绑定必须在基准里显式做**命名敏感性测试**；"类型安全"绝不能当作"正确"。

### 4.2 规模落地：`2609.24052` *Calibrated Decisions at Scale*（Jev 编码交通事故叙述）

- 覆盖 499,500 条 Texas 叙述，用 **27 问 schema** 编码 195,857 条；
- **成本由 schema 大小决定，而不是输入长度**；
- 对盲测人工标签 **F1 0.908**；两个前沿 LLM 一个只高 0.059、一个与它无显著差异；
- **校准因模型而异、不因范式而异 → 每个模型都必须单独审计**；同标签重校准使校准误差降低 **3.3×**；
- 给出"**flagged records 的人工复核预算**（每变量每年需读多少条）"；
- 与既存编码字段的一致性**低估**了对叙述的保真度（kappa 中位数差 0.26）。

### 4.3 增益在 readout 而非模型类别：`2609.23959` *Open-Jev on CallScreenBench*

- Qwen3-4B + LoRA，读两个答案标签 logits 的温度缩放 softmax 得 P(scam)；
- 41 场景 / 577 逐轮决策：AUROC **.974**、校准误差 **.052**，在预注册 .02 边际下**非劣于 LLM judge（MiniMax-M3）**；单张消费级 GPU **64.5 ms/决策**，比"同底座微调成生成式"**快 4.9×**；
- 作者明说：**"增益在 readout 与校准，不在准确率；一个微调过的 ModernBERT 编码器并不显著更差"**；且承认"无架构创新"、配方有测试集暴露、来电者全为合成。

→ **对 UBT 的含义**：不需要"Jev 这个模型"，需要的是**非生成 readout + 校准**；而 UBT 的 `SubprocessQERunner`（COMET）**本就是**非生成标量 readout。

### 4.4 其他相关新论文

- `2609.26532` REFLEX with Jev：LLM Agent 的选择性控制；
- `2609.23886` this-that-model-1.0：30 ms、百万分之一美分的类型化决策模型；
- `2609.27607` Can Jev Judge Radiology Reports：临床事实性判定（领域迁移需实测）；
- `2609.24395` JEVQA：用通用决策模型做视频质量。

---

## 5. UBT 现状（代码核证）

UBT 的 QE 已经是一条**分层、成本分级、fail-closed** 的级联，与 §3 的主流形态同构：

| 符号 | 层 | 性质 | 关键契约 |
|---|---|---|---|
| `FastPassFilter` / `FastPassDecision`（`qe/fast_pass.py`） | Tier 0 | **0-token 确定性精确门**（echo/near-echo、重复环、数值一致性、HTML delta、脚本密度、公式 span 等价） | 免费、精确、可解释 |
| `HeuristicQERunner`（`qe/comet_runner.py`） | Tier 0.5 | 离散缺陷类打分 | **`is_calibrated()` 返回 `False`**（离散缺陷类不足以支撑 best-of-n 排序） |
| `LLMJudgeQERunner`（`qe/llm_judge.py`） | Tier 1 | LLM-as-judge，逐 `(src,mt)` 一行 `score:`，temperature 0 | **`is_calibrated()` 返回 `False`**（"LLM opinions are not calibrated enough to rank repair candidates"） |
| `SubprocessQERunner`（COMET） | Tier 2 | 神经 QE，非生成标量 | `is_calibrated()` 为 `True`（可驱动 best-of-n rerank） |
| `TieredQERunner`（`qe/__init__.py` 导出） | 编排 | 分层调度 | — |
| `BaseQERunner`（`qe/base.py`） | 抽象 | `score_pairs` / `with_languages` / `is_glossary_aware` / `is_calibrated` / `reset_residency` | 依赖倒置；管线只依赖接口 |
| `score_policy.py` | 报表 | `mtqe_score` 真实分数过滤（剔除 skip/TM 的 1.0 占位） | 防止"1 个 0.30 混 499 个占位读成 0.9994" |
| `defect_taxonomy.py` | 缺陷法 | echo/near-echo、结构/关键缺陷判定 | **天然的 `Choice` 选项集** |
| `omission.py` / `added_content.py` / `term_drift.py` / `term_metrics.py` | 专项 QE | 漏译/增译/术语漂移 | 目前以规则 + 神经/LLM 混合 |
| `stages/triage.py` | 分级 | MQM 严重度 → HITL/PE 队列；critical escape rate 0 | 分级 + fail-closed 人工 |
| `qe/mt_gate.py` | 路由 | `is_mt_suitable` 精度优先硬 AND（**尚未接线**） | 脆布尔，最适合作概率化的对象 |
| `docs/golden-set.md` + `tests/baselines/` | 数据 | 4 语料 golden + `TokenEchoMockProvider`/`MockQERunner` | **确定性、离线、可复算的标定底座** |

**结论：UBT 的"复杂 QE 设计"不是过度工程，而是主流 QE 级联的完整实现**（确定性门 + 学习式 QE + LLM judge + 分级 + 成本策略 + 人工路由）。类 Jev 的落点是**插入**，不是**替换**。

---

## 6. 能取代 UBT 的复杂 QE 吗？——逐条评估

### 6.1 不能取代的四条硬理由

1. **不能取代确定性精确门。** `FastPassFilter` 查的是 echo、数值一致性、HTML delta、公式 span 等价——这些是**精确、0-token、可解释**的判定。`jev-fidelity` 的标定自己证明：有价值的是"模型 + 确定性门"的**组合**，纯相似度只有 20% 召回。把 `FastPassFilter` 换成概率模型只会**更差更贵**。
2. **不能取代成本分级。** UBT 的复杂度一半来自"免费/便宜/贵"的分层与预算控制。类 Jev 是新增一层，不是合并层。
3. **不能吸收缺陷分类法/多语言/块类型。** UBT 的 `Choice` 空间包含 CJK 特有、术语漂移、公式/表格/代码块——`2609.26758` 证明**选项越多、命名越敏感，翻转越严重**。这不是一个 `Noul` 能覆盖的。
4. **不能替代 fail-closed 人工路由。** `triage.py` 的 MQM 分级 + `critical escape rate 0` 是产品合同。类 Jev 的贡献恰恰是**为这套路由提供更好的校准分数与弃权信号**，而不是取消它。

### 6.2 它能做好的**窄切片**（真正的价值）

| UBT 现状 | 类 Jev 形态能做什么 | 依据 |
|---|---|---|
| `llm_judge` 逐对语义打分（贵、慢、`is_calibrated()=False`） | 一次前向的 `Noul`/`Score`，带校准置信度 | `2609.23959`（AUROC .974、64.5ms、非劣于 LLM judge）；`2609.24052`（F1 .908、成本随 schema 不随长度） |
| `triage.py` 靠 flags 推 MQM 严重度 | `Choice(severity)` + `Noul(数值失真)`，置信门控到 repair/human | `jev-fidelity`（0.70 门控 → acc 0.989） |
| `mt_gate.is_mt_suitable` 硬 AND 布尔（未接线） | 改成"NMT 安全概率"→ 阈值/成本权衡 | `2609.24052`（每模型审计 + 重校准） |
| `defect_taxonomy` 已有选项集 | 直接作为 `Choice` schema | 零设计成本 |
| best-of-n rerank 需要**可校准**分数 | 类 Jev 头天然可校准，补足 `LLMJudgeQERunner.is_calibrated()=False` 的空缺 | `2609.23959` 的"增益在校准" |

### 6.3 一句话边界

> **类 Jev 能取代的是"LLM judge 这个又贵又不校准的语义判定"，不是"UBT 的 QE 设计"。**

---

## 7. "最终一次性方案"评估：不是，而且方向相反

### 7.1 反"最终方案"的四条证据

1. **增益在 readout，不在模型类别。** `2609.23959` 明说微调 ModernBERT 不显著更差；`2609.24052` 说校准**因模型而异、必须逐模型审计**。→ 没有"某类模型一劳永逸"。
2. **类型安全 ≠ 正确。** `2609.26758`：改名让 AUC 从 .94 掉到 .23，type-error 仍 0%。
3. **纯模型选择会翻车。** `2609.13611`：QE 引导选择把"目标语言错误却流畅"排到正确译文之上。
4. **充分性必须对照源文。** `2608.20925`：单靠生成式 LLM judge 或单靠参考都不完整。

### 7.2 真正的收敛形态（UBT 已经在这个形态上）

文献与社区的共识轨迹是：

```
LLM-as-judge（贵、慢、不校准）
        │  蒸馏/标注（2603.10775）/ 奖励（2602.08600）
        ▼
非生成、快速、可校准的判定头（COMET 类 或 类 Jev 决策头）
        │  必须外包
        ▼
确定性精确门（Tier 0）+ 置信度门 + 弃权/人工复核预算（jev-fidelity / 2609.24052）
```

**这就是"最终形态"——一个带校准弃权的级联**，而不是某个"一次性最好的模型"。UBT 的 `FastPassFilter → TieredQERunner → triage` 已是这个骨架；类 Jev 只是为它补一个更便宜、可校准的语义中间层。

### 7.3 与上一轮结论的对齐

- **LensVLM**（`2026-09-25` 评估）：模型本体否决（`apple-amlr` 非商用、能力域错配），只抄方法论。
- **Jev**（本文）：**形态值得抄、可自建落地**，因为它命中的正是 UBT 的判定层；但**不能取代 QE 架构**。

---

## 8. 若引入，落点与形态（分级推荐）

| 优先级 | 落点 | 形态 | 许可/隐私 |
|---|---|---|---|
| P0（先验证） | QE 中间层（Tier 1.5） | 在 `FastPassFilter` 之后、`LLMJudgeQERunner` 之前插入一个**非生成可校准判定头** | 自建 head（最稳）或 `com-kotobalabs/open-jev-deberta-v3-large`（Apache-2.0，435M，CPU 可跑） |
| P1 | `mt_gate` 概率化 | 硬 AND → "NMT 安全概率"阈值 | 自建 |
| P2 | `triage` 分级 | `Choice(severity)` + 置信度门 | 自建 |
| P3 | rerank 校准分 | 补 `LLMJudgeQERunner.is_calibrated()=False` 的空缺 | 自建 |
| ❌ | Jev 云 API 作生产依赖 | — | 闭源、厂商容量风险、文本外发 |

**形态建议（照 `VlmDriver`/`ModelProfile` 先例）**：新增 `DecisionProfile`（声明式：问题 schema、选项集、置信度门、弃权语义、是否可校准、许可/版本），注册表 fail-closed；选择写入 ledger 以便追溯"这个判定来自哪条引擎链"。

---

## 9. 红线与风险

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| J1 | 把"类型安全"当成"正确"而放松 fail-closed | **高** | `2609.26758` 为强制测试项；不确定一律降级人工/LLM |
| J2 | 选项命名/顺序敏感性导致判定翻转 | **高** | 上线前跑命名敏感性测试（`0/1` vs `no/yes` vs 中性命名 vs 随机字符串） |
| J3 | 通用类 Jev 模型在翻译专用 QE 上打不过现有免费规则 | **高** | 必须先过 `golden-set` 基准门（§10），不许先接线 |
| J4 | 依赖闭源云厂商（暂停注册已发生） | 中 | 只作教师/实验；生产用自建/permissive 权重 |
| J5 | 校准因模型而异，误信未经审计的校准 | 中 | 逐模型审计 + 重校准（`2609.24052`，误差可降 3.3×） |
| J6 | 训练数据与 UBT 域不符（复刻权重训于公共分类） | 中 | 用 golden-set + ledger 自标；不直接采用其分数 |
| J7 | 中文/CJK 无公开拆分证据 | 中 | 基准语料必须含中文书稿（沿 W6） |

---

## 10. 决策门与实验方案

**门 1（核心，先跑再说）**：在 `tests/baselines` + `docs/golden-set.md` 的确定性底座上，比较三档：

- **基线 A（免费）**：`FastPassFilter` + `HeuristicQERunner`
- **基线 B（贵）**：`LLMJudgeQERunner`
- **候选 C**：非生成可校准判定头（先用 Apache-2.0 复刻权重起步，再自建）

指标（顺序即权重，沿用 `WEVISDOC` §5 方案 D 的分通道纪律）：
1. **相对 A 的增量**：能否在 A 漏掉的语义缺陷上真正加分；
2. **相对 B 的成本/延迟**：token 与 p50/p95 延迟；
3. **校准**：ECE / reliability curve（**不是只看准确率**）；
4. **弃权率与 fail-closed 正确性**：低置信度必须落到人工/LLM，绝不落到 pass；
5. **选项命名敏感性**：`2609.26758` 复现；
6. **分通道**：text / table / formula / CJK 分别计分（不能整书均值掩盖局部损伤）；
7. **中文书稿**：必须包含。

**通过条件**：候选 C 在语义缺陷召回上显著优于 A，且接近 B；成本/延迟至少比 B 低一个数量级；ECE 可接受；命名敏感性不导致系统性反转；弃权率可控（<=20%，且弃权全部 fail-closed）。全部满足才考虑 P0 接线。

**门 2**：`mt_gate` 概率化 A/B（`NOT_WIRED` 阶段先做实验台）；
**门 3**：`triage` 严重度 `Choice` 可行性。

---

## 11. 决策清单

- [ ] **立即**：门 1 实验台（复用 mock provider + golden，不引入新依赖）——先只测"Apache-2.0 复刻权重 vs `FastPassFilter` vs `LLMJudgeQERunner`"。
- [ ] **立即**：把 `2609.26758` 的选项命名敏感性测试固化为 UBT QE 基准的一部分（无论是否上类 Jev 都该有）。
- [ ] **排期**：若门 1 通过，设计 `DecisionProfile` + 注册表（照 `VlmDriver`/`ModelProfile` 先例）。
- [ ] **长期**：自建决策头（LoRA + 读 option logits），用 golden/ledger 的免费确定性信号当奖励——与 `WEVISDOC §10.4` 的 RLVR→QE 塔同路。
- [ ] **不做**：Jev 云 API 进生产依赖；用类 Jev 取代 `FastPassFilter`；在未过门 1 前接线任何 QE 路径。

---

## 附：证据索引

| 证据 | 来源 |
|---|---|
| Jev 定义 / RLCD / 价格 / 速度 / 纯文本 | TypeSafe AI 博文 *Introducing System One Models & Jev*（2026-09-15，一手） |
| 社区分类与翻译 QE 先例 | `yibie/awesome-jev`（categories：verification-guardrails、related-practices-discussions；非同行评审） |
| `GeekLinkDev/jev-subtitle-translator`（逐字幕 Noul 质检） | GitHub README（自报） |
| `klauswg/jev-suite/jev-fidelity`（0.70 门控标定表） | GitHub README + `klauswg/jev-suite:docs/calibration-report.md`（自报） |
| **类型安全≠正确 / 命名敏感性** | arXiv `2609.26758` *Type-Safe Is Not Error-Free* |
| **规模化落地 / 逐模型审计 / 复核预算 / schema 成本** | arXiv `2609.24052` *Calibrated Decisions at Scale* |
| **增益在 readout 与校准** | arXiv `2609.23959` *Open-Jev Judgments on CallScreenBench* |
| 诊断-修复分离 / MQM span | arXiv `2609.22793` *Diagnose, Then Repair* |
| LLM 标注器蒸馏回 COMET | arXiv `2603.10775` *LLMs as Annotators for MTQE* |
| error-aware RL QE | arXiv `2602.08600` *Beyond Scalar Scores* |
| LRM-as-judge 校准问题 | arXiv `2510.20780` *Are Large Reasoning Models Good Translation Evaluators?* |
| 文档级 QE reranking | arXiv `2510.08870` *Quality Estimation Reranking for Document-Level Translation* |
| QE 选择失败模式（错语言排高） | arXiv `2609.13611` *In the Blind* |
| 充分性必须对照源文 | arXiv `2608.20925` *Source-Free MT Evaluation Is Not MT Evaluation* |
| UBT QE 分层与契约 | `ubt/core/qe/{base,fast_pass,llm_judge,comet_runner,score_policy,defect_taxonomy,mt_gate}.py`、`ubt/core/engine/stages/triage.py` |
| UBT 标定底座 | `docs/golden-set.md`、`tests/baselines/`、`tests/mock_providers.py` |
| 方法论移植先例 | `docs/WEVISDOC_ADOPTION_ASSESSMENT_2026-09-20.md` §5 方案 D |
| LensVLM 对照（同批调研，未落盘为独立文档） | 结论：模型本体否决（`apple-amlr` 非商用、能力域错配），仅方法论移植（廉价全局视觉定位→选择性重处理） |

> **报告结束。** 核心结论两句话：**类 Jev 不是"最终一次性 QE 方案"，而是 UBT QE 级联里一个有价值的可校准非生成中间层；它能取代的是"又贵又不校准的 LLM judge 判定"，取代不了确定性门，也取代不了整条 QE 架构。** 引入前必须过 golden-set 基准门与选项命名敏感性测试，且所有低置信度一律 fail-closed 到人工。
