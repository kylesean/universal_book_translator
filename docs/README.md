# UBT 设计文档索引

`docs/` 只存在于工作副本中（`.gitignore` 的 `/docs/`），clone 不含这些笔记。
本索引是它们的导航，并标出每份文档的性质：**活文档**描述当前系统（以代码为准），
**设计记录**记录已落地的重构依据与实施日志，**提案/草案**描述尚未开工的方向，
**当前待办**见 [`guides/OPEN_ITEMS.md`](guides/OPEN_ITEMS.md)。

## 活文档（描述当前系统）

| 文档 | 内容 |
| --- | --- |
| [`guides/CI_AND_QUALITY_GATES.md`](guides/CI_AND_QUALITY_GATES.md) | 质量门禁在哪执行：本地 pre-commit/pre-push、CI 各阶段、覆盖率下限与结构护栏 |
| [`guides/TESTING_STRATEGY.md`](guides/TESTING_STRATEGY.md) | 测试如何写、写到哪、怎么跑：分层标记、目录约定、覆盖清单、结构护栏 |
| [`guides/MANUAL_E2E_INTEGRATION_TEST_SPEC.md`](guides/MANUAL_E2E_INTEGRATION_TEST_SPEC.md) | 全流程人工集成测试与端到端验收步骤 |
| [`guides/PDF_INGEST_AND_VLM_ROUTING.md`](guides/PDF_INGEST_AND_VLM_ROUTING.md) | 一个 PDF 从输入到 `IRBlock` 的解析/VLM 决策点与三条路径（当前行为地图） |
| [`guides/PDF_ENGINE_ROUTING_REDESIGN.md`](guides/PDF_ENGINE_ROUTING_REDESIGN.md) | PDF 解析引擎路由重构：为何把默认改为 pdfium、Docling 何时启用、扫描件逃逸、资源解耦（2026-10-10 落地记录） |
| [`guides/PDF_RENDER_LAYOUT_AND_FIDELITY.md`](guides/PDF_RENDER_LAYOUT_AND_FIDELITY.md) | 产物 PDF 如何由 IR 重建并由 Typst 排版：数学、表格、内联样式、语言方向、已知限制 |
| [`guides/PADDLEOCR_VL_SERVICE.md`](guides/PADDLEOCR_VL_SERVICE.md) | PaddleOCR-VL 作为表格/公式高保真读图通道：实测、服务部署、区域级接入与边界 |
| [`guides/SEMANTIC_JUDGE_SETUP.md`](guides/SEMANTIC_JUDGE_SETUP.md) | 语义判官：开关、provider/TOML 配置、模型能力实测、句级比较、成本与边界 |
| `benchmarks/README.md` | `scripts/cost_benchmark.py` 写出的无正文成本记录约定（本地证据，不进入 git，故此处不设链接） |

## 产品文档

| 文档 | 内容 |
| --- | --- |
| `UBT_OPERATOR_CONSOLE_PRD.md` | 操作台与审校台的产品需求文档（控制台已落地，实现细节以源码为准；本地文档，不进入 git） |

## 已落地的设计记录

| 文档 | 内容 |
| --- | --- |
| [`guides/MODE_SPACE_CONSOLIDATION_RFC.md`](guides/MODE_SPACE_CONSOLIDATION_RFC.md) | **RFC-2026-03**：模式空间收敛——16 个模式字段收敛为 2 个用户选择；overlay 语义清除（§7 实施日志）；含公式静默丢失等 4 个真 bug 的实测记录 |
| [`guides/SEMANTIC_FIRST_REARCHITECTURE_RFC.md`](guides/SEMANTIC_FIRST_REARCHITECTURE_RFC.md) | **RFC-2026-02（当前重构依据）**：语义优先重构——所有输入统一转 `DocumentTree`，翻译在语义层，渲染从语义树重排；含实证验证产物对比（同一份 PDF：中文占比 31%→58%、未翻译块 32→0、表格 0%→中文） |

## 提案 / 草案

| 文档 | 内容 |
| --- | --- |
| [`guides/FICTION_PRESET_AND_LITERARY_OPTIMIZATION.md`](guides/FICTION_PRESET_AND_LITERARY_OPTIMIZATION.md) | 小说/文学 preset 与文学化优化建议（草案待评审） |

## 当前待办

| 文档 | 内容 |
| --- | --- |
| [`guides/OPEN_ITEMS.md`](guides/OPEN_ITEMS.md) | 活动待办与修复记录：U-10 加密 PDF 报错（2026-10-20 已修复，含回归测试）＋从第二轮审查迁移的 triage 清单 |

## 当前评审记录

| 文档 | 内容 |
| --- | --- |
| `reviews/2026-10-20-round4-deep-review.md` | 第四轮深度审查：50 条问题台账与对抗性复核；基线 `fec23f0` + 同批未提交改动 |
| `reviews/2026-10-20-round4-deep-review-verdict.md` | 对第四轮审查的独立复核结论：50 条中 44 条成立、5 条 PARTIAL、1 条核心事实错误 |

> 提案文档中的文件与行号是**计划落点**，未必已存在；活文档里的引用以 `ubt/` 源码为准。
> 手动核对活文档的 `path:line` 引用：`python scripts/check_doc_refs.py`。
