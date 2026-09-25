# 成本核算设计（方向与重构路线）

> **状态：📋 决策已定，重构未开始。** 2026-09-24 定案三条：**(1) 计价功能保留**；
> **(2) 价格表外置且可配置**；**(3) 重构放在后期**。本文是唯一的设计权威出处——
> 后续迭代与重构按本文推进，状态行必须可复核（写明依据的符号/commit）。
>
> 引用规约承 `docs/README.md` §3：指向代码用 **符号名 + 基线 commit**，不写行号。

## 0. 为什么需要重构（现状核证，2026-09-24）

现有实现分两半，**token 采集这半是可靠的，价格这半是空白**：

| 能力 | 现状 | 依据 |
| --- | --- | --- |
| 从真实响应取 token | ✅ 可用 | `transports/openai_chat.py` 读 `usage.prompt_tokens/completion_tokens`；`openai_responses.py` 读 `input_tokens/output_tokens`；`anthropic.py` 读 `input_tokens/output_tokens/cache_read_input_tokens/cache_creation_input_tokens`；缓存命中由 `transports/base.py::_extract_cached_tokens` 兜住 4 种 shape |
| 跨续跑累计 + 落盘 | ✅ 可用 | `engine/usage.py::JobBill` → `engine/ledger.py` 的 `job_meta.usage_totals` → 质量报告 |
| **价格来源** | ❌ **硬编码** | 全仓唯一价格源是 `router/pricing.py::MODEL_PRICES_USD_PER_MTOK` 的 30 条字面量；`UBTConfig` **无任何价格字段**；用户要加价只能改源码 |
| 价格覆盖度 | ⚠️ 有缺口 | 30 条 `startswith` 前缀匹配。实测探测 15 个真实模型名，`grok-4` / `kimi-k2` / `llama-3.3-70b` / `mistral-large` / `command-r-plus` 未收录 → 报"未知" |
| 计价维度 | ⚠️ 部分建模 | Batch 折扣 `BATCH_API_DISCOUNT = 0.5` 与缓存输入价均为硬编码常量；Anthropic 缓存**写入** 1.25x 溢价未建模（`pricing.py` 自述 `not modelled`） |

**一句话定性**：现在的实现是"用模型名前缀猜价格"，而真实世界的价格属于 **(厂商, 模型, 渠道, 计费维度)** 四元组，且随时间变化。

## 1. 定案：两层模型（token 观测 / 价格换算）

把"我花了多少 token"和"这些 token 值多少钱"**彻底分层**，各自独立可信：

```
        ┌─────────────────────────────────────────────┐
L1 观测层 │ 真实响应 usage → 账本（永远准确）           │
        │  prompt / completion / cached / batch / unmeasured │
        └───────────────────┬─────────────────────────┘
                            │  按模型 token 数（可配置的价格表）
        ┌───────────────────▼─────────────────────────┐
L2 换算层 │ token × 单价 → 估算金额（可失真，必须标注）  │
        └─────────────────────────────────────────────┘
```

**不变量（重构也不得破坏）**：
- **I-1** L1 不依赖 L2。没有价格表时，token 数照样精确、照样可观测、可落盘、可导出。
- **I-2** L2 永远标注为**估算**，且价格来源可追溯（内置/用户文件/端点自报，见 §3）。
- **I-3** 拿不到价格 ≠ 拿不到 token。缺价时金额报"未知"，**token 数照常给**。
- **I-4** 已实现的端点感知（`pricing.py::endpoint_is_local`、`price_is_known`）保留：自托管端点价 $0，局域网端点需 `UBT_LOCAL_ENDPOINTS` 显式声明，回环上的付费网关可用 `UBT_BILL_LOCAL_ENDPOINT=1` 反转。

## 2. L1（token 观测）——已可用，方向性增强待做

现状已覆盖三个官方 transport。**待补强项**（非阻塞，按需迭代）：

| 项 | 现状 | 目标 |
| --- | --- | --- |
| 第三方网关 usage 兼容性 | 不认识的 shape → `unmeasured`，金额报未知 | 网关自报多少认多少（可信度最高，§3 渠道 C）；仍认不出才回落 `unmeasured` |
| 自部署端点 | 同上 | 明确"无 usage 即不可核算"是**预期行为**而非故障，不该 WARN 成故障 |
| 导出 | 仅在质量报告内 | 提供 `--usage-json` 导出原始 token 账本，供用户自行核账 |
| 维度完整性 | input/output/cached/batch 已有 | 补 reasoning tokens（OpenAI `completion_tokens_details.reasoning_tokens` 已在采，Anthropic thinking 侧待确认） |

**不做**：本地 token 化估算（`len//4` 之类）作为金额依据。宁可"未知"，不可"看起来准的错数"。

## 3. L2（价格换算）——外置可配，本次重构主体

### 3.1 价格表形态：`prices.toml`（与 `ubt.toml` 同构，可版本控制、可提 PR）

```toml
# 优先级：UBT_PRICES_FILE > 内置默认值
# 匹配用最长前缀；同长度时"精确名"优先于"族前缀"

[prices.gpt-4o]                    # 精确模型名
input = 2.50                        # USD / 1M input tokens
output = 10.00
cached_input = 1.25                 # 可选；缺省 = input
batch_discount = 0.5                # 可选；缺省 1.0（Batch 半价）
source = "openai"                   # 厂商标识，进质量报告
verified_at = "2026-09-20"          # 该价格的核证日期，过期可提示

[prices.gpt-4]                      # 族前缀：所有 gpt-4* 继承
input = 30.00
output = 60.00

[prices.claude]
input = 3.00
output = 15.00
cached_input = 0.30
```

**为什么是 toml 而非 env**：30+ 条模型 × 4 个维度塞不进环境变量；且价格需要 review/PR 流程，文件天然支持。

### 3.2 三种价格来源，按可信度排序

| 渠道 | 优先级 | 说明 |
| --- | --- | --- |
| **A. 端点自报** | 最高 | 第三方网关（OpenRouter、one-api 等）返回的 `usage` 里若含金额字段，**以它为准**，覆盖表值。这是唯一"等于你实际付的钱"的数据源 |
| **B. 用户价格文件** | 次高 | `UBT_PRICES_FILE` 指向的 `prices.toml`，覆盖内置值 |
| **C. 内置默认表** | 兜底 | 今天的 `MODEL_PRICES_USD_PER_MTOK` 形态，随包发布，标注 `verified_at` |

冲突时按 A > B > C，并在报告中写明本次金额用了哪一档。

### 3.3 匹配规则修正

- 现状：纯 `startswith`，最长前缀胜出。问题：`qwen` 前缀会命中本地 `qwen3:8b`（已由 §1 I-4 的端点判定兜住，但匹配规则本身仍需修）。
- 目标：**精确名 > 族前缀 > 端点自报**；`verified_at` 超过 N 天（如 180）→ 报告里提示"价格可能已过期"。

### 3.4 未知价的处理（保持 fail-closed）

| 场景 | 行为 |
| --- | --- |
| 有 token、无价格 | 金额"未知"，**照常给 token 数**（I-3）；`--budget-usd` 场景下按现状 refuse（这是防超支的正确默认） |
| 无 usage（网关不返回） | `unmeasured`，金额未知；**自托管端点**不因此告警（预期行为） |
| 端点自报了金额 | 直接采信，标注"渠道自报" |

## 4. 重构路线（分三期，每期独立可交付）

| 期 | 范围 | 交付物 | 阻塞性 |
| --- | --- | --- | --- |
| **P0（已完成，2026-09-24）** | 端点感知计价：本地端点 $0、`price_is_known`、删 3 条假模型名条目、`run_real_benchmark.sh` 探针修复 | 见 git 变更 | 已落地 |
| **P1 价格外置** | `prices.toml` 解析器 + `UBT_PRICES_FILE`；内置表迁到包内 `prices.toml`；`UBTConfig` 增价格文件路径字段；`has_price_entry` → `price_is_known` 已是前置 | 价格文件可覆盖任意模型 | 不阻塞发布 |
| **P2 渠道与匹配** | 端点自报金额（渠道 A）；匹配规则改"精确名优先"；`verified_at` 过期提示；`--usage-json` 导出 | 网关场景金额等于实付 | 不阻塞发布 |
| **P3（可选）** | 价格表自动同步（抓厂商定价页/官方 API）；`ubt doctor` 增"价格覆盖率"体检项 | 覆盖率提升 | 需先定数据源与合规边界 |

**每期的失败优先要求**（承根 `AGENTS.md`）：先写会失败的断言（本地端点免费 / 远端 fail-closed / 精确名优先 / 渠道 A 优先），再改实现。

## 5. 明确不做（避免范围蔓延）

- ❌ 不做"本地按字数估 token 换算成金额"——猜测的金额比没有金额危险。
- ❌ 不为 Ollama/llama.cpp 单独写 transport——它们都走 OpenAI 兼容面（见 `deployment_backend` 泛化，`capabilities.py::ModelProfile`）。
- ❌ 不在价表里硬编码某个用户的自部署模型价——那是 `UBT_PRICES_FILE` 的用途。
- ❌ 不把"价格未知"降级成"免费"——那会让 `--budget-usd` 静默失效（I-2/I-3）。

## 6. 维护

- 状态行可复核：改状态时附上依据的符号名/commit（承 `docs/README.md` §3.1）。
- 价格数值**不写快照值**进本文正文，引用 `prices.toml` 现状（规约 §3.6）。
- 与 `evaluation-and-comparison-guide.md` §5（cost benchmark）同域：本文管"怎么算"，那篇管"怎么测"。
