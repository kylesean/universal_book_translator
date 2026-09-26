# Knob 校准协议（规范本体）

> **状态**：🟢 活文档（随 `layout_policy.py` 演进更新；引用前以代码符号为准）

> **文档类型**：制度规范（normative）。本文重建 `ubt/core/policy/layout_policy.py` docstring 引用、
> 却在 `13568a7` 批量删除中丢失、引用一直没跟着清理的那份本体（来历见 §7）；
> 存量清点在本文**附录 B**（原独立文档 `KNOB_HARDCODE_AUDIT_2026-09-20.md` 已于 2026-09-22 并入此处并删除）。
> **日期**：2026-09-20 · **最近更新**：2026-09-22 · **适用**：`ubt/` 全量 · **状态**：§5 的执法已撤销，§4 的 sweep 覆盖面为**部分**（见该节诚实声明）
> **一句话**：一个魔数要么进表带证据，要么不进表但被 CI 拦下；不允许第三种。

---

## 1. 什么算 knob

机读定义（与 §5 的执法脚本一致，二者不得各说各话）：

| 计入 knob | 不计入 |
|---|---|
| 模块级 `int` / `float` 字面量绑定 | `bool` 开关、`str` 常量 |
| 模块级 `re.compile(...)` 绑定 | `frozenset` 词表、`str.maketrans` 表 |
| 上述两种的带注解写法（`X: float = 5.5`）、多重赋值（`A = B = 5`） | 函数体内的局部常量 |

两条边界的理由必须讲明白，否则会被当成漏检：

- **词表/布尔不算**：sweep 是"±邻域内改一个数看判决"，标点集合没有邻域；把非数值条目灌进制度只会淹没数值。
- **函数内常量不算**：`_env_*` 覆盖点在 import 期生效，局部常量 import 时不可见 —— 它**天生扫不动**。所以协议不执法它，但**评审要求就地写明来历**（哪个语料、哪次观测）。若某个局部数字重要到需要仪表，正确的动作是把它提到模块级并进表，而不是给它开后门。

---

## 2. 三档证据等级

定义抄自 `Calibration` 的 docstring（`layout_policy.py` 模块头部），此处为规范文本：

| 档 | 准入判据（满足其一） | 允许的动作 | 失败含义 |
|---|---|---|---|
| `PROVEN` | 2+ 篇不同文档验证；或合成属性测试；或 PDF/Typst 规范机制（非判断性取值） | 可依赖；改动仍需复跑 §4 | 无 |
| `SINGLE_DOC` | 仅在 chapter-1（或单本语料）上定标 | **禁止**为另一本书"顺手调"；须先过 §6 泛化关 | 换书即失明，这是已记账的债 |
| `HYPOTHESIS` | 有原理、无实证（例如文献均值） | 只能作为待验证项存在 | 值错了没人知道 |

**证据必须可复查**，`rationale` 字段缺一即视为不合规。三条齐全才算记过账：

1. **语料名**（`chapter-1` / `nistir4653` / `chapter-3 的 68 条渲染公式`）；
2. **样本量**（`206 blocks 中 3 处溢出`）；
3. **日期**（`2026-09-09`）。

范例（现存最好与最差的对照）：

```python
"PUNCT_SQUEEZE_CAP": KnobMeta(H, "6% cap held: matrix 9-cell sweep shows zero delta "
    "(corpus too clean to test, 3 overflows/206 blocks); mechanism unit-tested, "
    "needs stress corpus")           # ✅ 语料+样本+判决+下一步都写着
"EM_ASCII_ADV": KnobMeta(H, "0.55 literature average, uncalibrated vs fontTools")
                                     # ⚠️ 有出处无样本，属待补
```

**禁止**：只写"经验值""默认如此""看起来合适"。这类 rationale 等同没记账。

---

## 3. 落位规则

1. **新 knob 只能落在 `ubt/core/policy/layout_policy.py`**，且在 `CALIBRATION` 里带 `KnobMeta(status, rationale)`。
   这一条就是 `layout_policy.py` 原先写作 "routing-table §3" 所指的内容 —— 那份"routing table"文档在全仓从未有迹可循（`llm_judge.py:38` 的 "routing table" 是另一码事），本文接管它，该引用已改指本文。
2. **adapters / core 其他模块**若确实需要模块级常量（引擎专属、不宜进 policy 的），必须走 §5 的豁免登记，并在该文件的登记段下写清"为什么不是 policy 层的 tunable"。
3. 覆盖点名以 `UBT_` 起头、全局唯一，且**只允许**经 `_env_float` / `_env_int` 读取：默认值永远是被测对象，env 只用于扫描，非法值静默回落，保证打错字不会弄坏一次真实运行。
   （不与常量名严格对应：`SHORT_CHAIN_MAX_PAGES` 的覆盖点是 `UBT_SHORT_MAX_PAGES` —— 常量改过名而 env 是用户接口，保留旧名，见 `tests/unit/test_knob_sweep.py`。）
4. 表内自洽由 `tests/unit/test_layout_policy.py::test_every_knob_has_calibration` 把守（每条必须有合法状态、`calibration_summary()` 计数与表长一致）。

---

## 4. sweep 协议（改动前后的测量）

**现状必须先说清楚，否则这一节是空头支票**：目前只有 **5 个** knob 有覆盖点 ——
`UBT_PUNCT_SQUEEZE_CAP`、`UBT_PUNCT_SQUEEZE_PER_PUNCT`、`UBT_ROW_MERGE_GAP_PT`、
`UBT_ROW_MERGE_Y_TOL`（均在 `layout_policy.py`，经 `_env_float` 读取）、
`UBT_SHORT_MAX_PAGES`。表内条目的精确数**以 `calibration_summary()` 现值为准**；
未接 `_env_*` 覆盖点的条目一律**扫不动**。K-1（扰动测试）的第一批产出就是把 policy 层的 knob 接上覆盖点。

### 4.1 一次扫描的最小闭环

手工步骤如下；日常直接用 §4.5 的 `scripts/knob_sweep.py`，它把 1)–3) 一次跑完并报告动了哪些测试。

```bash
# 0) 无覆盖点的 knob：先在 layout_policy 里把常量改成 _env_float("UBT_X", 现值)
#    —— 默认值不变，这是"改动前"与"改动后"必须同源的前提。

# 1) 基线（三条 KPI 黄金，MockProvider 驱动，跨机稳定）
#    注：rigid 计划黄金不在这里 —— 它的唯一执行者是
#    tests/unit/test_rigid_overlay_golden.py（126c714 把它从 anchored 改名为 rigid，
#    并非删除；2026-09-22 二次复核更正）。扫 knob 时若 knob 可能影响 rigid 路线，
#    把该文件并列进命令。
uv run pytest tests/baselines -o addopts="" -q

# 2) 三点：中心、×1.5、÷1.5（K-1 采用的耐受带定义）
UBT_ROW_MERGE_GAP_PT=36  uv run pytest tests/baselines -o addopts="" -q
UBT_ROW_MERGE_GAP_PT=16  uv run pytest tests/baselines -o addopts="" -q

# 3) 单点复现（定位是哪个语料敏感时按名跑）
uv run pytest "tests/baselines/test_baselines.py::test_baseline_call_of_the_wild_streaming_e2e" \
    -o addopts="" -q
```

`UBT_UPDATE_GOLDENS=1` 会**重写**黄金值，只允许在人已看过 diff 的定标动作中使用；扫描期间严禁使用（否则等于把回归写成通过）。

### 4.2 固定条件（一次标定内不许变）

沿用 2026-09-09 首轮的四条，缺一条则该轮数据不可比：

1. **模型固定**：同一 draft/repair 模型跑完整轮；小模型只做冒烟，不进标定。
2. **语料固定**：标定必须用**能触发被旋钮路径**的语料。零溢出的语料进冒烟不进标定 —— 触发不了救援路径就没有测量意义（首轮正是栽在这里：CAP 的 9 格矩阵全零 delta，因为语料只有 3 个溢出块）。
3. **隔离固定**：每 run 独立 `--db-dir`，旧 ledger 永不复用（stale 标记会污染计数）。
4. **先跑 `base`**：全默认值一轮，作为所有 delta 的分母。

### 4.3 判决与记账

| 扫描结果 | 结论 | 必须做的动作 |
|---|---|---|
| 三点 KPI 全等 | 该 knob 在现存语料上**不敏感** | 记进 rationale（"matrix sweep: zero delta"）；列为删除候选（不敏感的值是伪装成参数的常量） |
| 某点越界即红 | 记录**耐受带**（例如 `24pt` 耐受至 `36pt`，`16pt` 崩两列页） | 写进 rationale；越界场景进 §6 泛化关 |
| 红但归因不到该 knob | 归因链断了 | 补一条按名引用该常量的单测（§5 的豁免理由不成立） |

**纪律条款**：改任何 knob 的**取值**，必须同一次提交里更新它的 `rationale`（加新证据 + 日期）。只改数不改账，视同 §2 的"没记账"。

### 4.4 否决位（guardrail）

数值收益不能换版面正确性：任一候选值让精度指标跌破基线，该值**直接否决**，不看它的 rescue 有多高。目检项（版面是否"膏药感"）**一票否决**，数字只作证。这两条源自首轮，属协议本体而非风格建议。

### 4.5 采集 harness

`scripts/knob_sweep.py`（K-1，2026-09-20）是现行 harness：按消费者选定 7 个 target（KPI 黄金 + 排版/分域/路由测试，条目数随测试演化波动，以 `pytest --collect-only` 为准），对每个有覆盖点的 knob 跑中心 / ×factor / ÷factor，报告**哪些测试动了**。默认 `--factor 1.5` 即本协议的耐受带；`--factor 1e6` 是"这 knob 到底测不测得到"的探针。退出码恒 0 —— 它是测量不是门禁（多数 knob 现在就不敏感，做成红只会教人忽略红）。2026-09-22 修正：target 列表里的两个 `anchored_*` 测试文件名已随 `126c714` 改名为 `rigid_*`，此前 harness 因指向不存在路径直接报错。

首轮的 harness 是 `tools/calibrate_knobs.py` + `collect_knob_metrics.py` + inplace 引擎的 per-page 日志行，三者在引擎收敛中已删除；新 harness 因此改用**测试是否变红**作为信号，而不是重建成指标聚合。

### 4.6 2026-09-20 首轮（现行 harness）结果

| knob | 中心 | ±1.5 判决 | ±1e6 判决 |
|---|---|---|---|
| `SHORT_CHAIN_MAX_PAGES` | 30 | **唯一在协议带上被测到的**：20 → `test_short_born_digital_chapter` 红；45 绿 | 同（低侧 0 红） |
| `PUNCT_SQUEEZE_PER_PUNCT` | 0.02 | 无感 | 2e4 → 3 项红（含 anchored overlay 黄金，即现 `test_rigid_overlay_golden.py`） |
| `ROW_MERGE_GAP_PT` | 24.0 | 无感（16–36 全绿） | 2.4e7 → `test_build_zones_claims_side_column_continuation` 红 |
| `PUNCT_SQUEEZE_CAP` | 0.06 | 无感 | **仍无感**：6e-8 与 6e4 都绿；只有 `cap<=0`（text_fit.py:114 的关闭分支）会红 —— 被测到的是开关，不是 6% |
| `ROW_MERGE_Y_TOL` | 0.5 | 无感 | **无感**：0.0 与 99.0 全绿 —— 现存语料对该值零测量 |

三条结论：

1. 5 个可 sweep 的 knob 里，**协议带宽（±1.5）只有 1 个被测到**。其余的要嘛只在荒谬值才动，要嘛完全不动。
2. `ROW_MERGE_Y_TOL` 与 `PUNCT_SQUEEZE_CAP` 的**取值**目前没有任何测试约束 —— 这正是附录 B 所记录的原审计 §4.1 结论："改 knob → 基线不变红 ≠ 改对了"。两条已按 §4.3 写回 rationale。
3. 提高可测性的路不是继续扫，是 §4.2 第 2 条：**语料不触发路径就测不到**（首轮 9 格矩阵全零 delta 是同一个坑）。要么给这两处补会红的合成属性测试，要么等 K-4 的 stress 语料。

---

## 5. 机器执法（K-5，**2026-09-20 撤销**）

原本这一节靠 AST 扫描守卫强制"制度外新增 = 0"：

| 件 | 状态 |
|---|---|
| `tests/unit/test_knob_registration.py` | **已删** |
| `tests/unit/data/knob_registration_baseline.txt`（340 条豁免 / 73 文件） | **已删** |

撤销理由（记录在此，防止后来人以为是不小心丢的）：它不是行为测试，是把一条
lint 规则塞进 pytest。代价有三条，都落在日常开发上——

1. **它挡的和 knob 无关。** 任何人在任何文件新增任何模块级数值都会红，包括
   根本不是可调旋钮的字面量；出路是去改那张 340 行的豁免表，于是**表只增不减**，
   与它自己的设计意图相反。
2. **它自认不还债。** 文件头原话："This test does not repay that debt. It stops
   it growing." 一个只提高改动成本、不降低存量风险的守卫，性价比不成立。
3. **它每次全量 `pytest` 都 AST 扫整个 `ubt/`**，和另外几个同族"仓库扫描守卫"
   合计约 900 行、900+ token 的失败输出。同批撤销的六个是
   `test_doc_references` / `test_sys_modules_guard` / `test_license_guard` /
   `test_core_ports_isolation` / `test_run_metadata_contract` / `test_cost_benchmark`。
   **2026-09-22 复核：这份清单曾经过期，别再照抄。** 六个里三个（`test_sys_modules_guard` /
   `test_license_guard` / `test_core_ports_isolation`）后来被恢复，现都在 `tests/unit/`
   （`test_license_guard.py:13` 自述 "Restored after commit 0492c7d"）；另外两个
   （`test_doc_references` / `test_run_metadata_contract`）确实仍不存在；
   `test_cost_benchmark.py` 之后也已在 `tests/unit/` 恢复（覆盖 `scripts/cost_benchmark.py` 的价格解析）。

撤销**不等于**"never inline"条文作废。条文仍在
`ubt/core/policy/layout_policy.py` 模块 docstring，`CALIBRATION` 表仍在，
`tests/unit/test_layout_policy.py::test_every_knob_has_calibration` 仍在校验表内
自洽（每个 knob 都有状态）——**表内的纪律有测试，表外的增量靠 review**。
若日后要恢复增量执法，正确的位置是 `scripts/` 下一个独立的 lint 步骤（按需运行），
不是单测路径。撤销时留下的事实供参考：未登记数值集中且不限于
`ubt/core/qe/fast_pass.py`（17 个）与 `ubt/core/validators/math_guard.py`（15 个），
即**制定"never inline"纪律的那一层恰恰是存量最大的地方**——这个判断不依赖被删的守卫，
重新数一遍即可复现。

---

## 6. 晋升与销账

`calibration_summary()`（`layout_policy.py` 末尾的燃尽函数）是燃尽指标。数字随条目增删漂移，**引用前先当场跑一次 `calibration_summary()`**，本文不写死现值（引用规约 §3.6）；2026-09-20 的"79 条"与同期另一份快照的"33+ 条"分别对应全表与 policy 层早期子集，均为当时实数。

- **晋升**只能由证据触发：`SINGLE_DOC → PROVEN` 需要第二篇不同语料（或 §4 的耐受带记录 + 合成属性测试）；**never by feel**。
- **`HYPOTHESIS` 不得长存**：一个从未被 §4 扫过的 `HYPOTHESIS`，要么补证据、要么删掉。审计口径下当前 20 条全部属于"有原理无实证"。
- 泛化关的输入是**外部语料**（K-4）：首选 ForMaT（arXiv 2605.15794，3,956 PDF × 15 语向）或自建扩样语料；SINGLE_DOC 占比下降要有逐档量化记录，不接受"感觉更稳了"。（原出处 `RESEARCH_RADAR_2026-09.md` R-1 已删除，见 git 历史。）
- 引擎切换时，**rationale 里点到具体引擎名或具体语料名的 knob 全部列为强制重校准项** —— 那些句子里的"pdfium 产物""chapter-1 观察"在新引擎下未必仍成立。（pypdfium2→pdf_oxide 的切换已发生：`23f9755`/`3f69833`，原评估文档 `PDF_OXIDE_ADOPTION_ASSESSMENT_2026-09-19.md` 已删除。）

---

## 7. 本文的来历，以及它没写完的那半件事

**这份文档不是新写的。** 仓库里原本就有一份 `docs/design/knob-calibration-protocol.md`（156 行，含 2026-09-09 的 9 格矩阵首轮结果），连同 40 个其它文档在 `13568a7 "update"` 里被一次批量删除（该提交：41 文件、6898 行删除、11 行新增，顺带删掉了 `tools/calibrate_knobs.py`、`tools/collect_knob_metrics.py`、`inplace_engine.py`、`docs/golden/manifest.json` 与两本真实语料 PDF）。审计 §2.4 说"该文件全仓不存在"，是对的 —— 但它**曾经存在**，而代码里的引用一直没跟着删。

所以本文的正文是重写，附录 A 是从被删版本里抢救回来的首轮测量记录（那是若干 `SINGLE_DOC` rationale 唯一的证据来源；不救回来，那些 rationale 就变成指向虚无的引用）。

**仍未解决、需要作者定夺的两件事**：

1. **`P11`**：被删的原版 §4.3 把它定义为 `layout_spills`（"P11 溢出块数"），即一次 run 的版面溢出计数；而 `layout_policy.py` docstring 写的是"needs the **P11 golden corpus**"、`calibration_summary()` 注释写"（P11 burn-down metric）" —— 同一个代号在一处是**指标**、另一处是**语料/里程碑**。三处引用没有一处能指到定义（定义随 `13568a7` 没了）。本文按"指标"理解并据此写 §6 的燃尽口径；若 P11 另有所指（外部工单编号），请就地改名，别再留专有名词。
2. **`docs/formula-corruption-diagnosis-and-fix.md`**：同一次批量删除的受害者，1174 行。**作者已裁定保持删除**（该文档不再需要）。它此前被 `ubt/core/qe/added_content.py:5` 当作"这个失败类的原始诊断（D2）"引用着，是唯一的悬空引用。**2026-09-20 已清账**：那句引用改成了代码内的自述（失败类本身在 docstring 里已完整描述，不依赖外部文档），细节需要时可从 `13568a7^` 取回。原先登记这条债务的 `test_doc_references.py` 也已随 §5 一并撤销。

---

## 附录 A：2026-09-09 首轮标定记录（自 `13568a7^:docs/design/knob-calibration-protocol.md` 抢救）

`CALIBRATION` 里若干 rationale 写着 "matrix 9-cell sweep shows zero delta"、
"calibrated 2026-09-09: matrix per01==base on books 2+3" —— **它们指向的就是这张表**。
表不在了，那些 rationale 就成了自我引用的空壳，故恢复。

条件：books 2+3（golden `manifest.json` 两行，共 206 blocks），`--draft-model hy-mt2-7b-4k`
本地 ollama，harness `tools/calibrate_knobs.py all`，四旋钮全开为 `base`。

| cell | rendered | overflow | reflow | rerouted | spills | fp_min |
|---|---|---|---|---|---|---|
| `base` | 134 | 2 | 2 | 1 | 3 | 0.0132 |
| `nosqueeze` / `cap03` / `cap10` / `per01` / `reflow1` / `reflow5` / `gap10` | 134 (+0) | 2 (+0) | 2 (+0) | 1 (+0) | 3 (+0) | 0.0132 |
| `gap3` | 133 (−1) | 3 (+1) | 2 (+0) | 0 (−1) | 3 (+0) | 0.0132 |

判决（与 `CALIBRATION` 现值对照）：

- **`PUNCT_SQUEEZE_CAP` 留 `HYPOTHESIS`**：全零 delta 不代表无效，是**没测到** —— 语料只有 3 个溢出块，layer 1 无施展空间。等 stress 语料（K-4）。
- **`PUNCT_SQUEEZE_PER_PUNCT` 升 `SINGLE_DOC`**：`per01 == base`，速率项理论上就不敏感，矩阵只是确认。
- **`REFLOW_MAX_EXTRA_LINES` 曾升 `SINGLE_DOC`**（`reflow1 == reflow5 == base`，3 行封顶在天花板之上）—— 但该旋钮**已随 inplace 引擎删除**，今天源码里只剩一个陈旧 `.pyc` 命中。别去找它。
- **`REROUTE_GAP_PT` 留 `HYPOTHESIS`，且值本身不可信**：`gap3` 的 −1 不是"挤坏了版面"。15 页像素 diff 显示 base 与 gap3 只差在 p6，而差异**不在**页底续排条 —— 两版都没有可见续排条，"膏药感"前提不成立。真实机制：base 的文本层里有续排译文（y≈80/756）被空白盖板闷住，即**幽灵续排：文本层有名、视觉无文**。协议因此**不以 gap 值作晋升依据**，当时提出改为加一条可见性断言 `_veto_buried_strips`（strip 文字盒与任一 cover 相交超 1pt² 即整块否决回 overflow）—— 该断言**最终未实现**，且 `REROUTE_GAP_PT` 本身也已随引擎重构删除，此处仅存当时的方法论结论。
  → 本轮唯一的方法论收获：**目检否定的不是数字，是"这个数字在测什么"的假设。**

采集陷阱（重跑前必读）：pipeline 默认 WARNING 级，per-page 的 INFO 计数行**落不了盘**；首轮是改用 harness 直调 render 函数取报告对象（跳过 export 门禁 —— 影响绝对值，不影响组间 delta）。

---

## 附：本文引用的可执行事实

| 事实 | 位置 |
|---|---|
| 三档枚举与 "never inline" 条文 | `ubt/core/policy/layout_policy.py` 模块 docstring |
| `CALIBRATION` 表（条目数以 `calibration_summary()` 为准） | 同文件 `CALIBRATION` 定义处 |
| `calibration_summary()` 燃尽函数 | 同文件 |
| 覆盖点仅 5 处 | 同文件 grep `_env_float\|_env_int` |
| 表内自洽测试 | `tests/unit/test_layout_policy.py::test_every_knob_has_calibration` |
| 增量执法 + 豁免登记 | **已于 2026-09-20 撤销**，见 §5 |
| 黄金基线与 KPI golden | `tests/baselines/test_baselines.py`（无 `slow` 标记，属于每编辑一次的快档；用例数与耗时以 `pytest tests/baselines --collect-only` 与实跑为准）、`tests/baselines/*/metrics.golden.json` |
| rigid 计划黄金 | 数据在 `tests/baselines/rigid/plan.golden.json`（原 `anchored-overlay/` 已随 rigid 更名迁走），**执行**在 `tests/unit/test_rigid_overlay_golden.py` |
| 存量清点（制度外数字债务） | 本文**附录 B**（原独立文档 `KNOB_HARDCODE_AUDIT_2026-09-20.md` 已并入并删除） |
| 被删原版（本文附录 A 的出处） | `git show 13568a7^:docs/design/knob-calibration-protocol.md` |

---

## 附录 B：制度外数字清点与整改路线终态（2026-09-22 并入；原 `KNOB_HARDCODE_AUDIT_2026-09-20.md` 已删除，全文见 `git show 62fcd75^:docs/KNOB_HARDCODE_AUDIT_2026-09-20.md`）

**判定框架不变**（原审计 §1）：区分"设计"与 "hack" 的是三问 —— 记没记账（Q1）、
有没有仪表（Q2）、泛化有没有语料说明（Q3）。结论"有认识论的启发式，账记了一半"仍成立。

### B.1 制度外（Tier-3）债务现状（2026-09-22 复核）

| 位置 | 事实 | 状态 |
|---|---|---|
| `ubt/adapters/pdf/overlay_text.py` | 单文件 **20 条模块级编译正则**（`_CJK` 字符类、`_MERGE_BREAK_RE` 中英标点集、`_MATHY_RE → _LATEX_CMD_RE → _FRAC_RE/_SQRT_RE` LaTeX 串行链），零 KnobMeta | 仍在；串行链的**补偿性漂移**（改前一条位移后几条的匹配面，端到端测试看不见"两个错凑成一个对"）是归因盲区 |
| 语言判定硬编码 | 字符类/标点集/caption 正则与多语言战略冲突（新语言 = 改代码发版） | **部分落地（K-3）**：`ubt/core/language_profile.py`（`PROFILES`/`get_profile`，FastPass 与 validator 已走 profile）；`overlay_text.py` 正则链仍 inline |
| 超时魔数五处各写各的 | `artifact_parity.py` 30 / `font_probe.py` 20 / `math_renderer.py` 20+60（render）/ `svg_diagram.py` 90、120 / `typst_healer.py` 120 | 均未配置化。（原审计所列"360"无出处，为笔误；行号一律以符号名现查为准） |
| `svg_diagram.py` | `_RASTER_DPI=300`、`_MIN_DIAGRAM_EDGE_PT=20.0`、`_CROP_PAD_PT=2.0` | 仍在，无来历注释升级 |
| config/router 名字匹配 | `api.anthropic.com` 子串判定、`draft_model.startswith("muse-")`、`pricing.py` 模型名前缀价目表 | 配置面、低危；`router/capabilities.py` 的 `ModelProfile` 是收口方向 |
| 未登记数值集中地 | `core/qe/fast_pass.py`（17 个）、`core/validators/math_guard.py`（15 个） | 见 §5——"never inline"纪律层恰是存量最大处 |

总量参照（2026-09-20 实测，`re.compile` 296 处 / 模块级大写数值常量 187 处）只作量级感，
精确数现查：`grep -rc 're\.compile' ubt/`。

### B.2 整改路线 K-1..K-5 终态（2026-09-22）

| # | 行动 | 终态 |
|---|---|---|
| K-1 | 扰动测试（±1.5× 耐受带） | **部分**：5 个覆盖点 + 首轮判决（§4.6）；`tests/unit/test_knob_sweep.py` 已按名引用 `PUNCT_SQUEEZE_CAP/PER_PUNCT`、`ROW_MERGE_GAP_PT/Y_TOL`、`SHORT_CHAIN_MAX_PAGES`（原审计"按名引用 = 0"已不成立，`dbdcdc7` 起） |
| K-2 | 补写本协议文档 | **已完成**（本文；`layout_policy.py` 引用已可解析） |
| K-3 | 语言资源外置 | **部分**（`language_profile.py`）；正则链未迁 |
| K-4 | 外部语料泛化关（ForMaT） | **未开始**（见 §6） |
| K-5 | 机械执法（CI 拦新增） | **已撤销**（§5），恢复的正确位置是 `scripts/` 下按需 lint，不是单测 |
