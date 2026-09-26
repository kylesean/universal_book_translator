# docs/ — 文档索引与引用规约

本目录只保留**随代码维护的活文档**，按角色分组：

- `guides/` — 使用/操作指南（活文档，顶部标 🟢）。
- `design/` — 设计/契约/规范（活文档，顶部标 🟢）。
- `benchmarks/` — 可提交的**无正文成本度量记录**（约定见其 README）。

> 合成测试语料 `tests/fixtures/synthetic-*.pdf` **不是文档、也不在本目录**：由
> `scripts/make_sample_corpus.py` 生成、`tests/conftest.py` 在 pytest 启动时按需重建，
> 被 `.gitignore`（`tests/fixtures/*.pdf`）忽略，且随 `tests/` 一起被 sdist 排除，
> **不随仓库分发**。

> **历史评估与已废弃的设计稿不再保留在树内**（原 `assessments/`、`history/`：
> INPLACE_WORKBENCH PRD/设计、INPLACE_TECH_SURVEY、WEVISDOC、JEV、
> PDFIUM_THREAD_SAFETY）。需要时用 git 历史检索：
> `git log --diff-filter=D --name-only -- docs/`。在役契约一律内联到代码符号或本目录
> 活文档，不再依赖时点快照。

## 1. 活文档

| 目录 | 文档 | 内容 |
| --- | --- | --- |
| guides/ | [USER_GUIDE.md](guides/USER_GUIDE.md) | 用户指南与参考手册：CLI 全命令、配置字典、服务端接口 |
| guides/ | [ENGINE_ROUTING_AND_DISASTER_PREVENTION_GUIDE.md](guides/ENGINE_ROUTING_AND_DISASTER_PREVENTION_GUIDE.md) | 引擎全景、auto 路由决策与参数避坑 |
| guides/ | [evaluation-and-comparison-guide.md](guides/evaluation-and-comparison-guide.md) | 真实模型评测（L1/L2/L3 分层、成本基准） |
| guides/ | [CI_AND_QUALITY_GATES.md](guides/CI_AND_QUALITY_GATES.md) | 本地门禁与远程 CI 的分阶段落地约定（当前仅本地 pre-commit/pre-push） |
| design/ | [golden-set.md](design/golden-set.md) | 基线语料与 KPI golden 的契约与再生成规约 |
| design/ | [knob-calibration-protocol.md](design/knob-calibration-protocol.md) | 可调旋钮标定与晋升协议 |
| design/ | [COST-ACCOUNTING-DESIGN.md](design/COST-ACCOUNTING-DESIGN.md) | 成本核算分层设计与重构路线 |
| design/ | [LAYOUT_PRESERVATION_MASTERPLAN_2026-09.md](design/LAYOUT_PRESERVATION_MASTERPLAN_2026-09.md) | 版面保持 / 出版级翻译总体方案（rigid/reflow 里程碑） |
| design/ | [PDF_SKILL_BORROWINGS.md](design/PDF_SKILL_BORROWINGS.md) | 从 `pdf` skill 抽取的硬化设计 D1–D11 权威定义 |
| benchmarks/ | [benchmarks/](benchmarks/) | 可提交的无正文成本度量记录（约定见其 README） |

## 2. 引用规约

指向代码优先用 **符号名**（函数/类/字段/常量）+ 基线 commit。确需定位时可附
`file:line`，但行号是全仓最易漂移的引用形式：改动涉及该行时必须同步更新，且不得
让它成为唯一的定位手段（符号名必须同时在场）。

### 2.1 状态行可复核
活文档顶部若标注状态（如"决策已定/重构未开始"），改状态时必须附依据：符号名或
commit，使读者可以当场验证，而不是相信一句无法核对的话。

### 2.2 悬空引用即债务
文档引用另一个文件/符号前，先确认它存在。引用被删对象的句子要么随删并清理，
要么改写为自述——不留下指向虚无的链接。删除文档时，必须同时清理代码/测试/其他
文档中对它的引用（`tests/unit/test_reference_resolvability.py` 会扫描 `ubt/`、`tests/`
里反引号包裹的 `docs/...` 路径）。

### 2.3 在役契约内联，不留时点快照
需要在代码里长期遵守的契约（并发、字体策略、计价、路由）应写进**代码 docstring 或
活文档**，而不是某份带日期的评估；评估类文档一旦其结论被采纳或推翻，即可删除，
git 历史保留证据。

### 2.4 代码与文档的权威序
两者不一致时，以代码符号为准修文档；只有代码确实错了才修代码，并同步更新文档。

### 2.5 数值不写快照值
易变的度量（条目数、耗时、覆盖率、测试用例数）不写死进正文——引用产生它的符号或目录
（如 `calibration_summary()`、`_SIZE_RATCHETS`、`pytest --collect-only`），需要数字时当场跑一次。

## 3. 测试权威
测试约束以仓库根 [AGENTS.md](../AGENTS.md) 为准，优先级高于本文与任何其他文档。
