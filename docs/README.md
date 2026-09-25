# docs/ — 文档索引与引用规约

## 1. 活文档（随代码演进更新）

| 文档 | 内容 |
| --- | --- |
| [USER_GUIDE.md](USER_GUIDE.md) | 用户指南与参考手册：CLI 全命令、配置字典、服务端接口 |
| [ENGINE_ROUTING_AND_DISASTER_PREVENTION_GUIDE.md](ENGINE_ROUTING_AND_DISASTER_PREVENTION_GUIDE.md) | 引擎全景、auto 路由决策与参数避坑 |
| [golden-set.md](golden-set.md) | 基线语料与 KPI golden 的契约与再生成规约 |
| [evaluation-and-comparison-guide.md](evaluation-and-comparison-guide.md) | 真实模型评测（L1/L2/L3 分层、成本基准） |
| [knob-calibration-protocol.md](knob-calibration-protocol.md) | 可调旋钮标定与晋升协议 |
| [COST-ACCOUNTING-DESIGN.md](COST-ACCOUNTING-DESIGN.md) | 成本核算分层设计与重构路线 |
| [benchmarks/](benchmarks/) | 可提交的无正文成本度量记录（约定见其 README） |

## 2. 时点快照与历史归档

带日期的文档（`*_2026-09-*.md`、`LAYOUT_PRESERVATION_MASTERPLAN_2026-09.md`、
`PDF_SKILL_BORROWINGS.md`、`WEVISDOC_*`、`INPLACE_TECH_SURVEY_*`）与
[history/](history/) 下的 PRD/设计稿是**成文当日的评估记录**，不随后续演进更新；
引用其中的数字与结论前先对照代码现状。

## 3. 引用规约

指向代码一律用 **符号名**（函数/类/字段/常量）+ 基线 commit，**不写行号**——行号是
全仓最易漂移的引用形式。

### 3.1 状态行可复核
活文档顶部若标注状态（如"决策已定/重构未开始"），改状态时必须附依据：符号名或
commit，使读者可以当场验证，而不是相信一句无法核对的话。

### 3.2 悬空引用即债务
文档引用另一个文件/符号前，先确认它存在。引用被删对象的句子要么随删并清理，
要么改写为自述——不留下指向虚无的链接。

### 3.3 案例与现状分层
事故复盘、翻车案例保留其历史叙事，但若当前实现已改变案例机制（修复、调度变更），
必须在案例旁加对齐说明，注明"当时的机制"与"现在的判据"。

### 3.4 代码与文档的权威序
两者不一致时，以代码符号为准修文档；只有代码确实错了才修代码，并同步更新文档。

### 3.5 活文档的状态标记
活文档顶部标注 🟢；快照文档标注日期。不要让快照冒充活文档，也不要给活文档
留下已失效的段落。

### 3.6 数值不写快照值
易变的度量（条目数、耗时、覆盖率）不写死进正文——引用产生它的符号或目录
（如 `calibration_summary()`、`_SIZE_RATCHETS`），需要数字时当场跑一次。

## 4. 测试权威
测试约束以仓库根 [AGENTS.md](../AGENTS.md) 为准，优先级高于本文与任何其他文档。
