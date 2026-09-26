# `regressions/` —— 按缺陷来源归档，不是按模块

这里放**复现某个真实缺陷**的回归测试：每个用例都对应一次实际坏过的行为，
文件头的 docstring 写清了它复现的是什么、以及修法失效时它会怎么红。

## 归属规则

**一种行为只有一个家。**

- 新缺陷的回归用例，先问"`tests/unit/test_<模块>.py` 存在吗"。存在就加到那里，
  不要在本目录新建文件。
- 只有当被测类**还没有**规范测试文件时，才在这里落一个以**被测模块**命名的文件
  （如 `test_glossary_enforcer.py`），并在下次触碰时并进去。
- **不要按审查日期建文件。** 曾经有 12 个 `test_review_2026-09-XX_regressions.py`，
  结果是一轮审查碰了哪些模块，这些用例就散在哪些不相关的文件里：查
  `fast_pass.py` 覆盖到什么，需要翻 5 个文件；于是下一轮审查看不见已有覆盖，
  又重测一遍。`test_review_fixes.py` / `test_review_architecture_fixes.py` 已因此
  被拆回 `test_cleaners` / `test_ledger` / `test_validators` / `test_api` /
  `test_pipeline` / `test_memory` / `test_mqm_triage` / `test_qe`。

## 关于删除

**本目录没有任何"不得删改"的特殊地位。** 这里的用例和别处一样，唯一的价值是
"它会在真实缺陷重现时变红"。所以：

- 与规范文件里已有用例覆盖同一分支的 → 删掉较弱的那个，不是两个都留。
- 被测代码已经不存在、且该行为不再可能的 → 直接删。
- 只有断言精确字符串/精确源码文本、改格式即红却抓不到 bug 的 → 换成行为断言。

过去这里写过"严禁随意删除或精简本目录中的测试用例"，并把用例数钉成一个数字。
那句话的实际效果是让本目录只增不减（钉的是 80+，实到 150），并且让"该不该留"
这个本该由人做的判断变成禁忌。**删掉那句。**

判断标准始终是：如果这行代码明天写错，这个用例会变红吗？会 → 留；不会 → 它不是
测试，是负载。

## 现状（2026-09-26 完成）

**所有日期/轮次名文件已并回 `test_<模块>.py`。** 本目录现在只剩本说明和
`__init__.py`，不再有任何测试文件。

并回分四批完成，每批单独提交、`pytest` 全绿、用例总数不变（仅位置变化）：

1. `test_review_2026_09_25_{regressions,final_round}.py` → 语言档案 / 成本预检 /
   added-content gate / rigid visibility / ledger flusher / QE / validators …
2. 五个 `test_round2_*_fixes.py` → cleaners / engine stages / PDF adapters /
   transports / OCR drivers …
3. `test_review_2026_09_24_fixes.py`、`test_reflow_guardrails_and_fixes.py`、
   `test_rigid_coverage_fixes.py` → 各被测模块。
4. 其余 9 个 `regressions/` 存量（audit / d2_echo / 09_17 / 09_18 / 09_18_round2 /
   09_21 / 09_21_pdf_adapters / 09_22 / security_and_isolation）→ 各被测模块。

并回时同步修正了两个迁移陷阱：跨文件同名 helper（按来源加前缀重命名），以及
`Path(__file__).parents[N]` 的目录深度假设（从 `regressions/` 移到 `tests/unit/`
后 `parents[3]` 失效，改为 `parents[2]`）。

**不要再新增日期名文件。** 约束的权威出处是仓库根目录 `AGENTS.md`
（§1 "One Behavior, One Home"）；本文件只记录 `regressions/` 的历史与例外。
