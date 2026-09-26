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

## 现状（2026-09-25 复核）

目录里共 **11 个**仍带审查轮次/日期标识的文件。它们不是"内容重复"（跨文件几乎
没有同名用例），而是**组织散射**：同一模块的守卫被拆进了多个 review 文件。

- **规则确立前的存量 —— 豁免但冻结**（下次触碰其模块时并走）：
  `test_audit_2026_09_20_regressions.py`、`test_review_2026_09_17_regressions.py`、
  `test_review_2026_09_18_regressions.py`、`test_review_2026_09_18_round2_regressions.py`、
  `test_review_2026_09_21_pdf_adapters.py`、`test_review_2026_09_21_regressions.py`、
  `test_review_2026_09_22_regressions.py`。
- **规则确立之后新增的违例**（证明"只写文档挡不住下一次"）：
  `test_review_2026_09_25_final_round.py`、`test_review_2026_09_25_regressions.py`。
  它们**没有豁免**，必须并回 `test_<模块>.py`。
- **不以日期命名、但仍按"缺陷来源"归档**：`test_d2_echo_regression.py`、
  `test_review_security_and_isolation.py`（后者应按被测模块重命名）。

`tests/unit/` 下另有 8 个同类轮次/修复名文件（`test_review_2026_09_24_fixes.py`、
`test_round2_*_fixes.py`、`test_rigid_coverage_fixes.py`、`test_reflow_guardrails_and_fixes.py`），
同样违反"按被测模块命名"。

**不要再新增日期名文件。** 测试组织约束的权威出处是仓库根目录 `AGENTS.md`
（§1 "One Behavior, One Home"）；本文件只记录 `regressions/` 的历史与例外。
