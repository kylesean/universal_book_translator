# docs/benchmarks/ — 可提交的成本度量记录

本目录存放 [`scripts/cost_benchmark.py`](../../scripts/cost_benchmark.py) 写出的
**无正文成本记录**（text-free metrics JSON），约定如下：

## 落盘内容与命名

- 文件名：`<job_id>.json`（`--job-id` 缺省时为 `benchmark_<unix 时间戳>`）。
- 每份记录只含**计数与配置**，绝不含书稿文本：语料 `sha256`、模型与单价
  （`--price-input/--price-cache-hit/--price-output`，无价时写 `null` 而非 `0.0`）、
  墙钟耗时、分阶段调用数/token/延迟、cache 命中量、估算美元、块完成/失败计数。
- 除本 JSON 外，完整产物（成品、质量报告、明细调用日志）一律落 `/tmp`，
  不进入仓库。

## 为什么刻意提交进仓库

没有可复现的真实账单，`--budget-usd` 就无从校准、内置价表错价也无处对账。
这些 JSON 是价表校准与预算红线的历史证据，随代码一起 review。

## 用法

```bash
uv run python scripts/cost_benchmark.py path/to/book.md --metrics-dir docs/benchmarks
# --metrics-dir 传空串则跳过写入；目录不存在时脚本自行创建
```

字段语义、评测判据与 L1/L2/L3 质量分层见
[evaluation-and-comparison-guide.md](../evaluation-and-comparison-guide.md) §5；
成本核算的分层设计与不变量见 [COST-ACCOUNTING-DESIGN.md](../COST-ACCOUNTING-DESIGN.md)。
