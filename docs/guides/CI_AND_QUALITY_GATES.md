# CI 与质量门禁策略（分阶段落地）

> **状态**：🟢 活文档（随流程演进更新；命令与文件路径为准）

本项目私有、单人高速迭代，**远程 CI 已分阶段激活**（阶段 1，2026-10-03）：接口稳定后
先在 CI 复现本地门禁，再视需要逐级加作业。原则不变——任何作业加入前，命令必须在当前
`main` 全绿；CI 只复现已通过的检查。本文是这个约定的唯一权威说明。

---

## 0. 原则

1. **门禁不消失，只换执行位置。** 删掉远程 CI 的同时必须保留本地门禁，否则等接回 CI 时
   会积压大量本可早发现的问题。
2. **只加能保持常绿的作业。** 任何门禁（本地或 CI）加入前，其命令必须在当前 `main` 上
   已经通过。禁止"先铺红再还债"。
3. **单一工具版本。** 本地钩子、手动运行、未来 CI 都用仓库锁定（`uv.lock`）里的同一套
   ruff / mypy / pytest，避免"本地绿、CI 红"。

---

## 1. 当前阶段：本地门禁（已启用）

配置在 `.pre-commit-config.yaml`，通过 `pre-commit` 接入 git 的
`pre-commit`（每次提交）与 `pre-push`（每次推送）两个阶段。

| 阶段 | 钩子 | 命令 | 预算 |
| --- | --- | --- | --- |
| pre-commit | `ruff-check` | `uv run ruff check --fix` | 秒级 |
| pre-commit | `ruff-format` | `uv run ruff format` | 秒级 |
| pre-push | `mypy` | `uv run mypy --strict ubt tests` | ~5s |
| pre-push | `pytest-fast` | `uv run pytest -m fast -q -n auto --dist=loadscope --cov=ubt --cov-fail-under=70` | ~43s |
| pre-push | `layering-invariant` | `uv run python scripts/dump_compiler_edges.py` | 秒级 |

只对**本次改动**的文件做 lint/format（自动修复后需重新 `git add`）；类型、快测与
分层不变量在推送时**跑全仓**，避免"改的文件恰好干净就算过"。分层检查是全树性质
（import 图的属性，不是单文件的），所以同样 `pass_filenames: false`。

### 测试分层与“默认层”的移除

`addopts` 里的过滤器是 `-m 'not slow and not legacy_drift'`，但 CI 的两条 pytest
步骤都显式传 `-m`，而 pytest 的 `-m` 是**替换**而非叠加。因此“默认层”（既非 `fast`
也非 `slow`）在 CI 里**从未被执行**：历史上 12 个文件、77 个测试处于这一状态，
包括 `tests/integration/test_pipeline_smoke.py` 这个端到端彩排。它们手工跑全绿，
所以没有任何信号。

处置：把这 77 个测试全部标为 `fast`（它们离线、无重依赖、合计 1.3 秒），
并删除“默认层”这个概念——现在每个测试必须带 `fast` 或 `slow`。
`tests/unit/test_marker_hygiene.py` 用纯 AST 守住这两条契约（必须有档、不得同时
带 `fast` 与 `slow`）。

### 死代码与覆盖率门禁（2026-10 审计后新增）

一次审计发现约 2,500 行**死模块**（一个已退役的显示公式转换器、一个从未接线的 witness 缓存、
一个无人调用的 advisor、一个孤儿资产抽取器）在 lint/类型/测试全绿的情况下长期存活——
因为没有任何门禁检查"还有没有东西没人用"。两道新门禁堵住这类腐烂：

1. **孤儿模块测试**：`tests/unit/test_no_orphaned_modules.py`（fast 档）用纯 AST 构建包内
   import 图，任何无人 import 的模块都会让推送失败。仅靠字符串可达的子进程入口（`python -m ...`）
   在测试内的 `_DYNAMIC_ENTRY_POINTS` 白名单里显式登记。
2. **覆盖率下限**：fast 档带 `--cov-fail-under=70`，比当前基线（~78%）低约 8 个百分点，
   日常改动不会触发，但"大删减不带测试"或"新增无人调用的模块"会被拦下。
   （下限曾在 2026-10 审计后设为 58，比当时基线低 15 个百分点——那个宽度足以藏下
   约 4,700 条未覆盖语句，等于没有护栏；随后按"基线减若干个百分点"的规则提到 70。）

这两道门禁是**回归护栏，不是质量目标**：它们只保证"不再退化"，不要求追高覆盖率。

`docs/` 本身不版本化，其中的 `path:line` 引用会随重构静默漂移。需要时手动跑
`python scripts/check_doc_refs.py`（exit 1 表示有引用指向不存在的 UBT 文件或行号越界）；
它不检查行号的小幅偏移，因为文档是散文引用而非编译器。

### 安装与使用

```bash
uv sync --extra dev          # 提供 pre-commit / ruff / mypy / pytest
uv run pre-commit install    # 一次即可，安装 pre-commit 与 pre-push 两个钩子
uv run pre-commit run --all-files   # 手动对全仓跑一遍（排查用）
```

`pre-commit` 钩子调用 `uv run ...`（`language: system`），因此**需要 `uv` 在 PATH 上**。
若用缺少 PATH 的图形化 git 客户端提交，钩子会失败——改用终端提交，或先装好 `uv`。
不要用 `--no-verify` 常态化绕过；确有紧急情况时，请在随后的提交里补跑一次全仓门禁。

### 为什么 format 也必须常绿

历史上 `ruff format --check` 曾在 44 个文件上变红（ruff 版本演进导致的既有漂移），这正是
"CI 一直红"的来源之一。本阶段已一次性 `ruff format .` 抹平，故本地门禁自出生即为绿色；
新增改动由 pre-commit 阶段自动格式化，防止再次漂移。

---

## 2. 分阶段路线（稳定后再搬回远程）

| 阶段 | 触发时机 | 范围 |
| --- | --- | --- |
| **0** | 私有 / 高速迭代 | 仅本地 pre-commit + pre-push（本文 §1） |
| **1（✅ 已激活，2026-10-03）** | 接口稳定、或引入协作者前 | 单个 Ubuntu 作业：lint + format + `mypy --strict` + `pytest -m fast` |
| **1.5（✅ 已激活）** | 渲染路径需要回归网 | 在阶段 1 作业内追加：checksum-pinned `typst` + `pytest -m slow` 的合成 PDF 渲染 e2e（40 条收集、26 条必跑：render e2e、crash/resume、页面预览、re-export、golden 符号探针、合成语料 gate）；分层不变量（`dump_compiler_edges.py`）也接入 CI。需要 gitignore 语料的 corpus harness 在语料缺失时自动跳过，需要 `pdf` extra 的引擎差分在 extra 缺失时整模块跳过 |
| **2** | 渲染 / 工具链稳定 | 加 `pdf` extra、Windows/macOS 矩阵、pandoc 的 golden 渲染 |
| **3** | 有真实发布 / 长期协作者 | nightly 真模型 smoke、LLM Judge 等真实 API 调用的作业 |

阶段 1 已于 2026-10-03 激活：`.github/workflows/ci.yml`，随 `ci: activate the
quality-gate workflow` 提交落地（激活前 `uv run pre-commit run --all-files` 在
`main` 全绿的前提成立）。`.github/workflows.disabled/` 已随之清空删除。

CI 只**复现**本地已通过的检查。若某天它变红，按"红一次修一次"处理，而不是加宽
容忍或关掉作业。

---

## 3. 旧工作流与恢复

原 `ci.yml`（4 矩阵 + `pdf`/`macos` 作业）与 `nightly-smoke.yml`（真模型 / 本地模型 /
docling network）于 2026-09-26 删除，完整内容仍在 git 历史中：

```bash
git log --diff-filter=D --name-only -- .github/workflows   # 找到删除提交
git show <删除提交>^:.github/workflows/ci.yml              # 查看删除前内容
git checkout <删除提交>^ -- .github/workflows/ci.yml       # 按需恢复
```

阶段 2/3 若需要这些作业，从中挑选**当前能保持常绿**的部分，而不是整份还原。
