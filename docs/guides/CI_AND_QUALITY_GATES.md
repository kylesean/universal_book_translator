# CI 与质量门禁策略（分阶段落地）

> **状态**：🟢 活文档（随流程演进更新；命令与文件路径为准）

本项目私有、单人高速迭代，**远程 CI 目前有意停用**：接口天天变时红叉只反映"还没写完"，
而不是"坏了"；私有仓库 Actions 按分钟计费，失败也照计。但"关掉 CI"不等于"关掉门禁"——
门禁搬到本地，等接口稳定后再**逐级**搬回 CI。本文是这个约定的唯一权威说明。

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
| pre-push | `mypy` | `uv run mypy --strict ubt tests` | ~2s |
| pre-push | `pytest-fast` | `uv run pytest -m fast -q` | ~6s |

只对**本次改动**的文件做 lint/format（自动修复后需重新 `git add`）；类型与快测在推送时
**跑全仓**，避免"改的文件恰好干净就算过"。

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
| **0（现在）** | 私有 / 高速迭代 | 仅本地 pre-commit + pre-push（本文 §1） |
| **1** | 接口稳定、或引入协作者前 | 单个 Ubuntu 作业：lint + format + `mypy --strict` + `pytest -m fast` |
| **2** | 渲染 / 工具链稳定 | 加 `pdf` extra、Windows/macOS 矩阵、pinned typst/pandoc 的 golden 渲染 |
| **3** | 有真实发布 / 长期协作者 | nightly 真模型 smoke、CometKiwi 等烧额度的作业 |

阶段 1 的作业**已写好但处于惰性状态**：`.github/workflows.disabled/ci.yml`。
GitHub 只加载 `.github/workflows/*.yml`，该目录不在其列，因此不会触发任何运行。

### 激活阶段 1

前提：`uv run pre-commit run --all-files` 在 `main` 上全绿。

```bash
git mv .github/workflows.disabled/ci.yml .github/workflows/ci.yml
git commit -m "ci: activate stage-1 gate"
git push
```

激活后，CI 只**复现**本地已通过的检查。若某天它变红，按"红一次修一次"处理，而不是加宽
容忍或关掉作业。`.github/workflows.disabled/README.md` 保留了同样的说明。

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
