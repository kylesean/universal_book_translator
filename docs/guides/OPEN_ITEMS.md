# 当前待办与未闭环结论

本文件记录 `docs/` 里仍活动的待办，以及刚从待办转为已修复的结论（保留以便追溯）。
当前审查报告见 [`../reviews/2026-10-20-round4-deep-review.md`](../reviews/2026-10-20-round4-deep-review.md)
与 [`../reviews/2026-10-20-round4-deep-review-verdict.md`](../reviews/2026-10-20-round4-deep-review-verdict.md)；
那 50 条问题以报告为准，这里只保留不在其中的事项和历史遗留的 triage 清单。

## 1. 已修复：加密 PDF 在 translate 路径的报错不可行动（第三轮 U-10）

**状态**：2026-10-20 已修复（工作树）；回归测试见
`tests/unit/adapters/test_pdf_encryption_guard.py`。

修复前：Docling 把加密 PDF 报成
`ConversionError: Conversion failed ... docling-parse could not load document`，
同时 pdf_oxide 刷出大量 `Decryption failed ... Not authenticated`；用户无法判断
是加密还是文件损坏。

修复后：`translate` 在适配器选择/清单阶段就拒绝加密 PDF，CLI 输出：

```text
Pipeline Execution Failed: PDF is password-protected or encrypted; decrypt it before translating: /tmp/ubt-encrypted.pdf
```

不再出现 `ConversionError` 或 pdf_oxide 的逐对象解密噪声。`assess` 原有的中文提示保持不变。

改动：

- `ubt/adapters/pdf/encryption.py`：新增 `reject_encrypted_pdf()`，只拦截
  `pikepdf.PasswordError`；损坏文件仍交给解析器报告，空口令可打开的加密 PDF 继续处理。
- `ubt/adapters/pdf/docling_parser.py`（`extract_manifest`）：在探测页数前调用守卫，
  Docling 与 pdfium 两个引擎共用。
- `ubt/adapters/factory.py`（`_resolve_pdf_adapter`）：在路由探测前调用守卫，
  避免 pdf_oxide 对加密文件刷解密噪声。
- `ubt/core/job_options.py`（`resolve_profile_name`）：`--profile auto` 对加密 PDF
  跳过 archetype 采样，让适配器守卫给出最终错误。
- `tests/unit/adapters/test_pdf_encryption_guard.py`：manifest / parse_stream ×
  两个引擎 + translate CLI，共 5 条回归。

复现（本地、无付费模型）：

```bash
.venv/bin/python - <<'PY'
import pikepdf
with pikepdf.open('/home/kyle/Downloads/chapter-3.pdf') as pdf:
    pdf.save('/tmp/ubt-encrypted.pdf', encryption=pikepdf.Encryption(owner="o", user="u", R=6))
PY
.venv/bin/python -m ubt translate /tmp/ubt-encrypted.pdf -l zh --dry-run -o /tmp/ubt-encrypted-out
.venv/bin/python -m ubt assess /tmp/ubt-encrypted.pdf
```

**优先级**：P1（已完成）。

## 2. 第二轮审查遗留清单（从归档迁移，供 triage）

> 以下清单摘录自 2026-10-18 第二轮审查的 §9.5/§9.6。部分条目可能已被后续提交修复，
> 不是全部仍待办；逐条对照当前 HEAD 后再排期。原始截图与完整复现证据已随归档删除，
> 清理前的完整 `docs/` 备份保存在 `/tmp/ubt-docs-backup-20261020-023441.tar.gz`
> （临时目录，不保证持久；需要保留请自行转存）。

### 2.1 可执行工程清单（原 §9.5）

> 编号前缀对应 §9.2 的“修复”列。每条格式：**问题 → 改动 → 回归 → 验收**。
> 所有条目都不引入新依赖、不改 UI 设计语言。

#### A. Phase 0 — 信任 blocker

**A1 CLI 作业补 finalize 指针（P0，约 10 行）**

- 问题：`ubt/cli/main.py:161,163` 两条分支都没传 `finalize_job`，CLI 作业的 `job_meta.metadata_json`
  没有 `output_file/report_file`，`/report`、`/deliverables` 409。
- 改动：改用 `ubt/core/engine/job_runner.py:38` 的 `build_job_orchestrator(config, router=None, qe_runner=None,
  rate_limiter=None, finalize_job=lambda ev, led: persist_progress_metadata(ev, led, ev.job_id), rehearse=dry_run)`。
  注意 `create_dry_run_orchestrator` 已支持 `**orchestrator_kwargs` 转发 `finalize_job`，不需要新写。
- 回归：`tests/unit/engine/test_job_runner.py` 已有“hook 到达两种 wiring”的测试；新增 `tests/integration/`
  一条“CLI dry-run → ledger 有 `output_file` → TestClient `/report` 200、`/deliverables` 200”。
- 验收：`UBT_DB_DIR=<tmp> uv run ubt translate <md> --dry-run --job-id t1` 之后
  `sqlite3 <tmp>/t1.sqlite "select json_extract(metadata_json,'$.output_file') from job_meta"` 非空。

**A2 Quality Gate 三态（P0，约 40 行 TSX）**

- 问题：`QualityGateView.tsx:199,213` 在 `report=null` 时 `failedBlocks=0 → isPassed=true`；首帧即假 PASS；
  `delivery_verdict` 缺失时回退 `failed==0`，旧 report 仍假 PASS。
- 改动：引入 `type ReportState = 'loading' | 'ready' | 'missing' | 'error'`；
  `verdict = report?.delivery_verdict ?? null`；**`verdict === null` 时一律 UNKNOWN**，不显示 PASS/BLOCKED；
  pillars 在非 `ready` 时渲染 `n/a`；`deliverables` 的 409 单独成状态（不要 `.catch(() => [])`）；
  warning 走 amber “Needs review”，`GATE BLOCKED` 只留给 `verdict === 'blocked'`。
- 回归：新增 vitest（`web/` 目前零测试）：① `report=null` 不出现 “PRODUCTION READY”；②
  `delivery_verdict='warning'` 不出现 “GATE BLOCKED”；③ `deliverables` 409 显示“报告/产物未生成”而非“No deliverables”。
- 验收：`cd web && pnpm vitest run && pnpm build`。

**A3 rehearsal 全链路（P0/P1，约 30 行 Python + 20 行 TSX）**

- 问题：`JobSummary` 无 `rehearsal`；`/status` 的 ledger 回退不填 `rehearsal`；UI 不读；CLI 的 `_mock` 只加在派生 id 上。
- 改动：
  1. `ubt/api/models.py` 的 `JobSummary` 加 `rehearsal: bool = False`；
  2. `ubt/api/job_catalog.py:68-82` 从 `metadata.run_identity.mock_run` 读；
  3. `ubt/api/routes/job.py:820-828` 的 `JobStatusResponse` 也填 `rehearsal=bool(meta.get('run_identity', {}).get('mock_run'))`；
  4. 前端在 Mission Control 列表行、详情头、Quality Gate 标题、下载文件名统一加 `REHEARSAL — not a translation`。
- 回归：`tests/unit/api/test_api_contract.py` 加“ledger-only job 的 `/status` 与 `/jobs` 都带 `rehearsal=true`”；
  前端加一条 vitest 断言徽章存在。
- 验收：CLI `--dry-run` 作业在 `/jobs`、`/status`、UI 三处都显示 rehearsal。

**A4 缺图合同（P1，约 15 行 Python）**

- 问题：`docling_render.py:168-170` 把**资产路径**当 block id 传给 `apply_render_skip_flags`；
  该函数按 `block.id` 建索引（`export.py:137`），于是 `by_id.get(asset_path)` 永远 None，返回 `[]`；
  `adapt.py` 走 `else: PRESERVED_OPAQUE`，`contract.missing_assets=0`、`violations=[]`。
- 改动：把 `stats.assets_missing` 从 `list[str]` 改成 `list[tuple[str, str]]`（`block.id`, asset），
  或让 `docling_render` 用渲染器已知的 block 关联；`apply_render_skip_flags` 增加可选的
  `asset_to_block: Mapping[str, str]`。**不要**在 `export.py` 里做路径模糊匹配。
- 回归：`tests/unit/core/test_export_render_quality_flags.py` 加一条“缺图 → checkpoint 非空 → `contract.missing_assets==1`
  → `violations` 含 `ASSET_MISSING`”；现有 `apply_render_skip_flags` 的纯函数测试也要加“未知 id 仍被忽略”的反例。
- 验收：把某个 figure 的资产路径改成不存在的文件，跑一次导出，`*_contract.json` 的 `missing_assets` 应为 1。

**A5 坏缓存准入 + 源覆盖守卫 + partial 契约（P0，约 60 行 Python）**

- 问题（三段）：
  1. `read_docling_document` 命中即返回，不校验页/文本覆盖；一条坏转换会被永久复用；
  2. `map_iterated_items` 之后没有“源页覆盖”检查；`_check_completion_ratio` 只数已抽出块；
  3. `--pages` 作业的 quality report 没有 `partial`/`page_scope`，UI 不显示。
- 改动：
  1. `docling_cache.write_docling_document` 额外存 `coverage = {pages, texts, pages_with_text}`；
     `read_docling_document` 命中后校验 `pages_with_text / pages >= 0.5`（阈值可配），不达标就 `logger.warning` + 返回 `None`（触发重转）。
  2. 新增 `assert_source_coverage(manifest, blocks)`（放 `ubt/adapters/pdf/docling_parser.py` 或 `ubt/core/content/`）：
     用 `manifest.metadata["page_count"]` 与 `{b.provenance.source_page for b in blocks}` 算覆盖率；
     `ingest` 阶段调用一次，`export` 阶段再调用一次（防止重导出旧 ledger 时绕过）。
     低于 `UBT_SOURCE_COVERAGE_MIN`（默认 0.8）时先 CPU 重跑；仍低则 `IntegrityViolationError`。
  3. `QualityReport.route` 旁边加 `page_scope: str | None` 与 `partial: bool`；
     `export.py` 从 `ctx.config.get_selected_pages()` + `manifest.metadata["page_count"]` 填。
     `delivery_verdict_from_blocks` 不变（`partial` 不是 verdict 维度），但 UI 必须单独显示。
- 回归：
  - 用 `corpus/documents/chapter-3.pdf` 的坏缓存 fixture 造一个单测：`read_docling_document` 丢弃它；
  - 单测：造 3 页 manifest、只给 1 页的 blocks，`assert_source_coverage` 抛错；
  - 单测：`--pages` 的 report 有 `partial=True` 且 `page_scope='1-6'`。
- 验收：`uv run pytest -m fast -q tests/unit/adapters/test_docling_cache_coverage.py tests/unit/core/test_source_coverage.py`。

**A6 写锁三处（P0，约 25 行 Python）**

- 问题与修法见 §9.3 更正 1、2。
- 回归：`tests/unit/test_writer_lock.py` 加两条：
  ① “被接管后 `renew()` 必须抛 `LedgerWriterLockConflictError` 且 `_held=False`”；
  ② “legacy `--fresh` 打开 ledger 后，原 holder 仍持有锁 / 新 holder 被拒”。
- 验收：`uv run pytest -q tests/unit/test_writer_lock.py`。

**A7 截断 fail-open（P1，约 20 行 Python）**

- 问题：`ubt/core/router/router.py:805-814` 在 `finish_reason == "length"` 时续写（`_MAX_CONTINUATION_ROUNDS = 3`，`router_errors.py:21`）；
  `:951-963` 续写调用异常（非 budget/cancel）直接 `return text`；`:967-972` 在 3 轮后仍截断只 `logger.warning` 然后 `return text`。
  于是调用方（`draft.py:617` 的 `parse_bold`）把半句话当完整译文 finalize。复现（脚本 provider 恒返回 `length`）：
  `calls: 4 | returned: 'truncated text '`，无异常、无标记、无 `needs_human`。
  另：未覆写 `generate_with_finish_reason` 的 provider 走基类，恒返回 `finish_reason=None`，截断**根本检测不到**。
- 改动：续写耗尽/失败时返回 `(text, truncated: bool)` 或抛 `ModelProviderError`（由现有失败路径处理）；
  `draft` 阶段把 `truncated` 写成 `error_flags` 并置 `NEEDS_HUMAN`，不要静默 finalize。
- 回归：`tests/unit/core/router/test_router_funnel.py` 现有 `test_...` 把“失败保留 partial”固化为预期行为，
  加一条“耗尽后必须可被调用方看见”；`tests/unit/core/router/test_batch_transports.py:212-220` 的 provider 能力矩阵加一条“不报告 finish_reason
  的 provider 不得声称检测过截断”。
- 验收：`uv run pytest -q tests/unit/core/router/test_router_funnel.py`。

**A8 粗体哨兵吞公式（P1，修法要绕开一条已固化的行为）**

- 问题：`ubt/core/ir/emphasis.py:34` 的 `_MARKER_RE` 把 `\llbracket B \rrbracket` 当成 `⟦B⟧`；
  `parse_bold(r"$\llbracket B \rrbracket$ 的投影")` → `('$$ 的投影', (InlineRun(text='$ 的投影', bold=True),))`，
  公式里的 `B` 被删、留下空 `$$`。
- 触发面有两个：模型把真哨兵转写成 LaTeX（设计意图，**必须继续剥**），以及**源公式本身合法使用
  `\llbracket…\rrbracket`**（语义括号在学术文本常见，**必须保留**）。两者在文本上不可区分——
  这正是这条 bug 难修的原因。
- **⚠ 复核纠正：我最初写的“按 `_MATH_SPAN_RE` 跳过数学跨度”会破坏一条已固化的测试。**
  `tests/unit/core/test_emphasis.py:143-149` 的 `test_a_marker_inside_a_formula_leaves_no_math_residue`
  断言 `$(\Delta, \llbracket B \rrbracket A \llbracket /B \rrbracket)$` 里的 marker **必须被剥掉**、
  runs 只剩 `["A"]`——那是 mamba 的真实形状，不剥就会让 Typst 报 `unknown variable: lbracket` 整篇编译失败。
  按我原方案改，这条测试会红，并重新引入那个整篇失败。
  “marker 不与 `$` 相邻”同样无效：mamba 形状里首尾 marker 正贴 `$`。
- **可行的修法**：把“源文里的合法 `\llbracket`”与“模型转写的哨兵”在**上游**区分开，而不是在 `parse_bold` 里猜：
  1. **源侧**：`mark_bold_spans` / `_source_with_bold_markers` 在写入哨兵前，先对源文里的
     `\llbracket…\rrbracket` 做转义或占位（它属于 `$…$` 内的合法内容，`MathMasker` 本来就会整段掩码）；
     实测 `default_placeholder_engine().mask()` 确实把 `$\llbracket B \rrbracket$` 整段收进 `math_map`，
     但 **restore 之后又原样交给 `parse_bold`**，所以泄漏点在 restore→parse_bold 之间。
  2. **渲染侧兜底**：`artifact_guard._LATEX_EMPHASIS_RE` 会把合法语义括号也认成 `emphasis` 哨兵并转义
     （`ubt/render/typst/artifact_guard.py:71`、`ubt/render/typst/document_renderer.py` 的 `has_sentinel` 分支），所以只修 `parse_bold`
     会把“静默删除”变成“可见转义 + 泄漏告警”。
- 建议的最小实现：`parse_bold(text, *, math_spans=())` 增加一个可选的“已确认数学区间”参数，
  由调用方（`draft.finalize_draft`）传入**从 `MaskedSource.math_map` 恢复出的原始区间**——
  这样 mamba 那条测试（直接调用 `parse_bold`、不传区间）行为不变，而生产路径能保护合法语义括号。
- 回归：`tests/unit/core/test_emphasis.py` 现有 18 条测试**全部保持**（特别是 `:143-149` 那条）；
  新增两条：(a) `parse_bold(restored_text, math_spans=[…])` 保留 `B`；(b) 不传 `math_spans` 时行为与今天一致。
- 验收：`uv run pytest -q tests/unit/core/test_emphasis.py tests/unit/render/test_artifact_guard.py`。

**B1 数字门：源侧授权表（`ubt/core/validators/consistency.py`）**

- 问题：`Part II → 第二部分`、`May 1962 → 1962 年 5 月`、`January 5, 2020 → …1 月…` 被判“凭空造数”。
  根因是**源侧从不把这些词读成数值**，目标侧折叠是对的。
- 改动：在 `_invented_target_numbers` 的授权集里加两张表（放在 `src_view` 上匹配）：
  - `_EN_MONTH_VALUES = {"January": 1, ... "December": 12}`（只授权该月对应的数值，且必须与年份/日期结构同现，
    避免 `May` 的助动词义被读成 5——用一个 `\b(?:in|on|of)?\s*Month\b` + 邻近数字的守卫）；
  - `_ROMAN_VALUES`：`I..XXXIX` 的合法罗马数字，只在 `^(Part|Chapter|Book|Volume|Section|World War)\s+` 之后或
    纯标题上下文中授权，且**同时要求目标侧该数值出现在对应中文标签附近**（`第…部分/章/卷/节`、`世界大战`）。
- 回归：`tests/unit/validators/test_consistency.py` 加 6 条正例（Part II / World War II / Chapter II / May 1962 /
  January 5, 2020 / 第三部分 vs Part III）+ 3 条反例（`Chapter 10 → 第二章` 必须仍失败、
  `May I help you → 我可以帮你吗` 不得授权 5、`He is Part II of the story` 这类非标题上下文不得授权）。
- 验收：`FALSE REJECTS` 里 `Part I/II`、`WWII`、`date` 三类消失。

**B2 数字门：目标侧量词保护（`ubt/core/validators/consistency.py`）**

- 问题：`5 千米` 被 `_CN_NUMERAL_CONTEXT_RE` 的 `[数词]+(?=度量词)` 分支匹配到裸 `千`，
  折算成 `5 1000米`，于是源 `5` 丢失、目标多出 `1000`。
- 改动：在 `normalize_for_numeric_matching` 里，先把“数字/空格 + 数词 + 多字度量词”整体掩码（复用
  `CN_MEASURE_WORDS` 的多字项，按最长优先），掩码后再做数词折叠，最后还原。`_CN_UNIT_PREFIX_SUFFIXES`
  的现有逻辑保留（它处理 `10千克` 这种无空格形态）。
- 回归：`tests/unit/validators/test_consistency.py` 加 `5 千米`、`速度（千米/小时）`、`3 千瓦`、
  `10 千克`、`250 万`（必须仍折算）五组；并断言 `1 千米 == 1000 米` 这类“真的换算”仍能被接受。
- 验收：`Speed (km/h) → 速度（千米/小时）` 与 `5 km → 5 千米` 都 pass。

**B3 标题/短句膨胀门（`ubt/core/qe/fast_pass.py`）**

- 问题：`max_length_ratio=1.5` + 单字 CJK 按 2.2 折算，导致 `A/甲`、`1984/一九八四`、
  `Introduction/引言：本书的结构与方法`、`Preface/前言：致中国读者` 全部判“膨胀”。
- 改动：给 heading 设**独立上限**（`heading_max_length_ratio`，默认 3.0），并加一条“标题可含副标题”的豁免：
  目标以 `：` / `:` / `—` / `（` 开头分段且第一段长度 ≤ 源的 2 倍时放行。**不要**全局抬高 `max_length_ratio`。
- 回归：`tests/unit/qe/test_fast_pass.py` 加 4 条正例 + 2 条反例（`Introduction → 引言` 仍 pass；
  `Method → 我们提出了一种全新的、基于……的方法，它包含三个部分` 必须仍失败）。
- 验收：`heading_expand` 3 条 + `title` 里 `1984` 从 FALSE REJECTS 消失。

**B4 人名/术语（`ubt/core/qe/term_drift.py` + `ubt/core/validators/consistency.py`）**

- 问题：`GlossaryTerm` 无 `kind`；`normalize_glossary` 把 `kind` 丢掉；`GlossaryConsistencyValidator` 对 person
  仍要求每块出现目标串。
- 改动：
  1. `GlossaryTerm` 增加 `kind: str = "term"`；`normalize_glossary` 透传 `entry.get("kind", "term")`；
  2. `detect_term_drift` 对 `kind == "person"` 走“弱校验”：源出现人名、目标**没有**该渲染时，若块内存在
     代词（他/她/他们/她们/它）或该人名在**章节内已渲染过**，判 `rendered=True`；
  3. 报告里仍保留 `person` 的 drift 统计（供审校看），但不作为 `GLOSSARY_DRIFT` 的失败原因。
- 回归：`tests/unit/qe/test_term_drift.py` 加 person 正例（`Darcy nodded. → 他点了点头。`、
  `Elizabeth smiled. She did not answer. → 她笑了笑，没有回答。`）+ 反例（`Darcy → 伊丽莎白` 必须仍 drift）。
- 验收：`name` 类别 4 条从 FALSE REJECTS 消失；`kind="term"` 的术语行为不变。
- **补充（复核实测）**：`Mr. Darcy → 达西先生` 与 `Mr. Darcy → 达西` **都**通过（alias 机制已覆盖），
  所以“alias 只认全称”这句不成立；真正的失败只有“源人名 → 纯代词”一种。
  另外 `inflected_variants` 已存在（`416ffb0`/`d9855e3` 用它折多渲染），person 的弱校验可以直接复用它，
  不需要新的“章节内已渲染”状态机——先把 `kind` 透传 + 代词判定做出来，再谈跨块记忆。

**B5 repair 采用条件（`ubt/core/engine/repair_loop.py` + `ubt/core/validators/span_repair.py`）**

- 问题：`not error_spans` 时 `is_preserved=True`，整块重写被当成 tier 1，未校准 runner 直接采用。
- 改动：
  1. `splice_repairs` 的返回值扩成 `(text, spliced, rewrote_out_of_span)`；`not spans` 时第三项恒 `True`
     （可保留旧三元组签名，用命名元组或新方法 `splice_repairs_detailed`）；
  2. `repair_single_block` 里 `rewrote_out_of_span` 的候选**一律进 tier 2**；
  3. 未校准 runner 下，tier 2 采用条件是：
     `_adoptable(new, old)` **且** `len(new) >= 0.9*len(old)` **且** 句数不减少 **且**
     通过 glossary validator；否则保留原稿 + `error_flags`，状态 `NEEDS_HUMAN`；
  4. 术语类缺陷（`Glossary term violation`）额外要求“目标包含渲染串 + 长度不退化”。
- 回归：把 §2.3 的假 provider 复现写成 `tests/unit/engine/test_repair_no_span_overwrite.py`：
  正确原稿 + 更差候选 ⇒ `target_text` 不变、`status != repaired`。
- 验收：该测试 + 现有 `tests/unit/engine/test_repair_residual_defects.py` 全绿。

**B6 judge 闭环（`ubt/core/qe/defect_taxonomy.py` + `quality_gate.py`）**

- 问题：soft flag 被归为 `DETERMINISTIC_DEFECT`，judge 没有机会放行；`repair` 后重打分但不再问 judge。
  另外 `7a820d6` 之后 judge **只能降分**（`min(base, judge)`，`qe_judge_allow_upgrade` 已删）——这正是想要的，
  但它意味着“judge 放行”必须通过**新的缺陷类**实现，而不是靠抬分。
- 改动：
  1. `classify_defect` 增加第四类 `SOFT_AMBIGUOUS`：短标题膨胀、两句/三句合句、person 代词不渲染、
     `numeric_advisory`——这类不直接进 repair，先给 judge 一次裁决；
     放行条件用 **`score_policy.is_shippable`**（`7a820d6` 新增，`max(SHIPPABLE_FLOOR, qe_threshold)` + 无 flag），
     不要在这里另写一个 `>= 0.85` 的魔法数；
  2. `quality_gate` 在 `suspicious_blocks` 里区分“硬缺陷”和“soft”，后者只标 `REPAIR_PENDING` 但不消耗 repair 轮次；
  3. 只有 judge 触发的 repair，在 repair 后**再判一次** judge（用 `judge_again` 标记，避免无限循环）。
- 回归：`tests/unit/qe/test_defect_taxonomy.py` 加 `SOFT_AMBIGUOUS` 的判定表；
  `tests/unit/qe/test_llm_judge.py` 加“soft flag 被 judge 放行后不再 repair”；
  再加一条“judge 不能把分数抬过 `is_shippable` 的门槛”（防止有人把 upgrade 加回来）。
- 验收：编辑对抗集里 soft 类误杀降为 0，且 `qe_judge_calls` 只对 soft/抽样块增长。

**B8 “shippable” 规则的消费者（`ubt/core/qe/score_policy.py` 已存在，补测试与接线）**

- 问题：`7a820d6` 新增 `shippable_threshold` / `is_shippable`，并在 `triage.py:403`、`repair.py:194`、
  `repair_loop.py:76` 接了线；但 `classify_defect` 仍用调用方传入的 `threshold`，两条路径对“可发布”的定义
  还没有单点。
- 改动：`classify_defect` 的 `threshold` 参数改为内部调用 `score_policy.shippable_threshold(threshold)`；
  `quality_gate` 的 demote 判断同样走 `is_shippable`。这是 3 行改动，但能把“改了一处忘了另一处”的概率降到 0。
- **不对称是真实的，但后果不可达（复核纠正）**：`classify_defect` 直接用调用方传入的原始 `threshold`，不走 floor：

  ```
  classify_defect([], score=0.0,  threshold=0.0)  -> PASS
  classify_defect([], score=0.25, threshold=0.2)  -> PASS
  classify_defect([], score=0.55, threshold=0.75) -> SEMANTIC_DOUBT
  ```

  位置：`ubt/core/engine/repair_loop.py:76` 存了 `self.terminal_pass_floor = shippable_threshold(qe_threshold)`，
  但同一对象的 `:140-144` 仍把**原始的** `self.qe_threshold` 传给 `classify_defect`。

  **但“会把 0.25 分的块放出”这个推论不成立**：`classify_defect` 全仓唯一调用点是
  `repair_loop.select_repair_candidates`，返回 `PASS` 只表示“不进 repair 队列”，**从不写 `MTQE_PASSED`**；
  该选择器只接收非终态块（`ledger_mixins.py:1405-1430` 排除 `TERMINAL_STATUSES`），而一个
  `score=0.25, flags=[]` 的块在 `qe_threshold=0.2` 下已经是 `MTQE_PASSED`（终态），根本进不了选择器。
  即使进了，`triage` 的 `is_shippable` 也会判 `NEEDS_HUMAN`。
  **所以这是单元级 latent 不一致，不是线上泄漏。** 修法仍值得做（3 行），但不要当 P0 叙事。
- 回归：`tests/unit/qe/test_defect_taxonomy.py` 加“`qe_threshold=0.2` 时 PASS 仍要求 >= 0.5”。
- 验收：`uv run pytest -q tests/unit/qe/test_defect_taxonomy.py tests/unit/qe/test_score_policy.py`。

**B9 `rerank_k` 的处置（`ubt/core/config.py` + `ubt/core/engine/repair_loop.py`）**

- 问题：`rerank_k` 接受 1–5（`config.py:447`）、有文档、被 `RepairLoop` 读取；`b27ee66` 只在 docstring 里
  承认它 inert，**行为未变**，也没有任何校验拒绝 >1（`UBTConfig(rerank_k=3)` 仅告警通过）。它**不在**
  `PROVIDER_ALLOWED_KEYS`（`providers.py:93-107`），所以 `[defaults]` 根本设不了，只能走 `UBT_RERANK_K`。
- **⚠ 复核纠正：“设 >1 只告警、无行为”不是全局事实。** `_rerank_enabled()` 要求 `qe_runner.is_calibrated()`，
  而 **`MockQERunner` 继承 `BaseQERunner` 的默认 `True`**（`base.py:71`），`--dry-run` 用的正是它
  （`dry_run.py:108`）。实测 `RepairLoop(qe_runner=MockQERunner(), rerank_k=3)._rerank_enabled() == True`，
  一次 repair 会发 3 次模型调用。也就是说：**dry-run 路径上 `rerank_k` 是真起作用的**，生产路径上是 inert，
  而配置告警在生产路径上是对的、在 dry-run 上是错的。
- 改动：二选一，不要停在中间态：
  (a) 删除该字段（最干净；生产路径没有 calibrated runner 就没有排序）；
  (b) 保留但改成 `Literal[1]`，并让 `MockQERunner.is_calibrated()` 返回 `False`（mock 分数不是质量估计），
      这样 dry-run 也不再假排序。**(b) 更适合本项目——dry-run 是演练，不应变更真实的 repair 调用次数。**
- 回归：`tests/unit/core/test_provider_defaults.py` 已有“拒绝已删旋钮”的模式，照此加一条；
  再加一条“dry-run 下 `rerank_k>1` 不增加 provider 调用次数”。
- 验收：`UBTConfig(rerank_k=3)` 要么抛错，要么 dry-run 的 repair 调用次数与 `rerank_k=1` 相同。

**B7 EPUB 内部实体（`ubt/adapters/epub/adapter.py`）**

- 问题：`_parse_xhtml` 对 `<!ENTITY>` 返回空 soup；多章书可静默丢一章。
- 改动：解析失败或 `soup.body is None` 时抛 `DocumentParseError`（带文件名/章节），不要返回空 soup；
  若确实要容错，则 `logger.error` + 该章标记 `parse_failed` 并在 report 里 fail-closed。
- 回归：在 `tests/unit/adapters/test_format_adapters.py` 加“内部实体 → 抛错”；不要假设已有“空文件 → 抛错”用例
  （实测该文件没有这条测试，需要一并补上）。
- 验收：`uv run pytest -q tests/unit/adapters/test_format_adapters.py`。

#### C. Phase 2 — 上下文/提示词

**C1 macro 接受 strategy/block_type（`ubt/core/router/prompts.py` + `ubt/core/router/router.py`）**

- 问题：`build_macro_chunk_draft_prompt` 无 `strategy`/`block_type`，`prompt_strategy_override` 到不了主路径。
- 改动：给 macro 增加 `strategy` 与 `per_block_types: list[tuple[str, str]]`；`draft_macro_chunk` 从
  `_get_profile(model)` 读 strategy 并传入；`block_type` 以属性形式挂在 `<block>` 上（不影响输出协议）。
  若 strategy 是 `minimal`，macro 仍应保留 glossary/neighbor，但**删掉 rich 的 7 条规则**，只留 3 条核心。
- 回归：`tests/unit/core/router/test_registry_strategy.py` 加“macro 的 system prompt 随 strategy 变化”。
- 验收：同一条 prompt 在 `minimal`/`rich` 下 system prompt 长度差异 > 30%。

**C2 同批前文用译文（`ubt/core/engine/stages/draft.py` + `neighbor_window.py`）**

- 问题：`finalize_draft` 返回新对象，`current_batch` 里的原块 `target_text` 不更新。
- 改动：`draft_macro_chunk_group`/`draft_single_block` 在 finalize 后把 `target_text` 回写到 `current_batch`
  中同一 `id` 的块（用 `in_place` 修改或 dict 覆盖）；或让 `draft_single_block`/`draft_macro_chunk_group`
  accept 一个 `neighbor_provider`，按 `id` 取“已 finalize 的译文”。**不要**改 `neighbor_window` 的读取逻辑。
- 回归：`tests/unit/engine/test_draft_neighbor_context.py` 造 3 块批次，断言第 2 块的 `neighbor_ctx` 里含第 1 块的**译文**。
- 验收：同批相邻块的前文文本不再等于 `source_text`。

**C3 记忆与提示词身份（`ubt/core/engine/stages/draft.py` + `ubt/core/memory/tm.py`）**

- 问题：rolling 按章数开关（≤1 章、>40 章、academic 关闭）；`compute_tm_context` 无 `doc_id/strategy/domain`；
  `PROMPT_VERSION` 不含四套 system prompt。
- 改动：
  1. `resolve_draft_policy` 改为按“字符/块数”判断（例如 `total_chars >= 20_000 and chapters <= 200`），
     不再用 `chapters <= 1` 直接关；
  2. `compute_tm_context` 增加 `doc_id`（可空，空则退化为旧行为）、`strategy`、`domain`；
  3. `_PROMPT_LAYOUT_PARTS` 增加四套 prompt 的 `style_desc`/`rules` 摘要哈希。
- 回归：`tests/unit/test_tm.py` 加“doc_id/strategy/domain 不同则 exact 不命中”；
  `tests/unit/engine/test_draft_policy.py`（新增）加“51 页 PDF 有 rolling、50 页没有”的边界。
- 验收：`UBT_TM` 单测 + 边界测试全绿。

**C4 preset 与 judge 解耦（`ubt/core/presets.py`）**

- 问题：所有 preset 都写 `qe_engine=heuristic`，`high_fidelity` 也用不上 `tiered`。
- 改动：`PRESETS[…].qe_engine` 改为 `None`（不覆盖），让 config/env 决定；如果确实要“high_fidelity 默认 tiered”，
  改成 `qe_engine="tiered"` 但 `qe_judge_enabled` 仍由用户显式开（并在 doctor 里 warning）。
- 回归：`tests/unit/core/test_preset_routing.py` 加“env `UBT_QE_ENGINE=tiered` + preset high_fidelity 仍为 tiered”。
- 验收：`overrides_from_request({"preset":"high_fidelity"}).get("qe_engine")` 为 `None` 或 `tiered`，不是 `heuristic`。

#### D. Phase 3 — 成书体验

**D1 heading level + outline（`ubt/adapters/pdf/docling_parser.py`）**

- 问题：`map_iterated_items` 的 `make_element(...)` 不传 `level`，25 个 heading 全 level 1；
  PDF outline 24 项全 depth 0。
- 改动：在 `type_docling_blocks` 里按 Docling 的 `text.level`/字号/编号形状推导 `level`；
  **复用** `ubt/adapters/pdf/docling_outline.py:242-246` 的映射规则（`max(1, level+1)`），不要新写一套。
  `document_renderer` 已按 `level` 输出 `#heading(level: N)`，Typst **自动**生成分层 PDF 书签——
  已用最小 `.typ` 验证（`level:1`/`level:2` → `pypdf` 读到 depth `[0,1,0]`），所以**不需要**加 `#outline()`。
- 注意区分：正文里的**目录页**是另一件事（`#outline(...)` 或继续用 `render_toc`），且 `resolve_toc_mode("auto")`
  对 academic 默认关闭。D1 只负责书签层级。
- 回归：`tests/unit/adapters/test_docling_heading_levels.py`：`1.` → 1、`2.1.` → 2、`2.1.1.` → 3；
  并对 `jc_023404` 的 25 个 heading 断言 level 集合非 `{1}`。
- 验收：重新导出 2606，`pypdf` 读 outline 的 depth 集合包含 `{0,1,2}`。

**D2 display title / 元数据 / 页眉（`ubt/render/typst/` + `export.py`）**

- 问题：PDF `/Title` 为空；页眉用 `manifest.title`（源文件名）；HTML `<title>` 是常量 `UBT translation`。
- 改动：`manifest` 增加 `display_title`（译文标题，来自首个 level-1 heading 的 `target_text`）；
  `TypstDocumentRenderer` 的 `paper_title` 与 `#set document(title: ...)` 都用它；
  `_write_html_view` 传 `title=display_title`；EPUB 的 `dc:title` 同步。
- **已实测可行性**：全仓 `grep "set document"` 零命中，说明这个元数据入口从未被用过；但 Typst 本身支持——
  用一个最小 `.typ`（`#set document(title: "测试标题", author: "UBT")`）编译后，`pypdf` 读到
  `{'/Title': '测试标题', '/Author': 'UBT', ...}`。所以 D2 不需要改渲染架构，只在 preamble 加一行。
- 回归：`tests/unit/render/test_preamble_compile.py` 断言 `#set document(title:` 出现；实际产物用 `pypdf` 读 `/Title`。
- 验收：`2606.23344v1_mono.pdf` 的 `/Title` 是中文标题，不是 `None`。

**D3 列表 marker / 斜体 / `\(...\)`（`ubt/render/typst/document_renderer.py` + `textgeom.py`）**

- 问题：`ListItem` 走 prose 分支，marker 丢失；`styled_runs_in_box` 的 italic 位硬编码 `False`；
  行内 `\(...\)` 不被 `_MATH_SPAN_RE` 识别。
- 改动：
  1. `document_renderer` 加 `block_type == "list_item"` 分支：有 `marker` 时渲染为 `#list.item` 或
     `#block[...]` 前缀，无 marker 时用 `- `；
  2. `CharStyle` 加 `italic: bool`，`extract_char_styles` 从 `fontname`/`italic` flag 填；
     `styled_runs_in_box` 的 `current` 用 `char.italic`；
  3. `_MATH_SPAN_RE` 增加 `\\\((?:.|\n)*?\\\)` 与 `\\\[(?:.|\n)*?\\\]` 两个分支（注意与 `$` 分支的优先级）。
- 回归：`tests/unit/render/test_list_marker.py`（`1.`/`2.` 输出含编号）、`tests/unit/render/test_italic_runs.py`、
  `tests/unit/render/test_math_delimiters.py`。
- 验收：合成样张 PDF 里列表有编号、斜体有倾斜、`\(x^2\)` 不再字面输出。

**D4 宽表 + 公式图注（`ubt/render/typst/document_renderer.py`）**

- 问题：宽表按最长单元格给权重，14 列时每列一字；公式图后 ≤320 字段落被当 caption。
- 改动：
  1. `render_table` 增加 `max_weight`（例如 `<= 2 * 平均权重`）与 `min_font_size` 降级；
     `columns >= 10` 时改 `#place`/横排页；
  2. `_Figure` 增加 `is_formula: bool`（由 block 的 region/来源判断），`is_formula` 时**不**绑定无标签短段落，
     除非显式 `图/Figure/表/Table` 标签。
- 回归：`tests/unit/render/test_table_columns.py`（14 列权重比 ≤ 2）、
  `tests/unit/render/test_formula_following_prose.py`（公式后正文不变成 caption）。
- 验收：2606 第 8–10 页的表格文本层不再“一列一字”；公式后的“其中…”回到正文。

**D5 HTML/EPUB 结构（`ubt/render/html_view.py` + `epub_view.py`）**

- 问题：HTML `<title>` 常量；工具栏写死中文；`<ol>` 丢失；`<br>` 被 escape；`alt=""`；`figcaption` 在 `figure` 外；
  EPUB nav 只有书名、无 CSS、缺 `dcterms:modified`。
- 改动：
  1. `compose_html` 的 `title` 由调用方传 `display_title`；工具栏文案按 `lang` 取（zh/en/ja 至少三套）；
  2. `render_fragment` 对连续 `ListItem` 按 `marker` 判断 `ol`/`ul`，`<li value=…>`；
  3. `_render_table` 把单元格里的字面 `<br>` 还原成真正的换行标签（其余内容仍 escape）；
  4. `_render_element(Figure)` 的 `alt` 用该 figure 的 caption 文本；`figcaption` 放进 `<figure>` 内；
  5. `compose_epub` 生成按 heading 的 nav、加 `_STYLE_CSS` 与 `<link rel=stylesheet>`、metadata 加
     `<meta property="dcterms:modified">`。
- 回归：`tests/unit/render/test_html_view.py` 加 `<ol>`/`<br>`/`alt`/`figcaption` 四条；
  `tests/unit/render/test_epub_view.py`（新增）解包断言 nav 条目数 = heading 数、存在 CSS、存在 dcterms。
- 验收：解包 EPUB 后 `nav.xhtml` 有章节，`content.opf` 有 `dcterms:modified` 与 stylesheet。

#### E. Phase 4 — 产品 UI/UX

**E1 Wizard 就绪门 + `/setup`（前端 + 后端）**

- 改动：`NewJobWizard` 顶部 `getDoctor()`，credential/model FAIL 时阻断提交并给“复制 fix 命令 / 我已配置，重试”；
  新增 `ubt setup` CLI 与 `/setup` 路由（provider/base_url/api_mode/key/draft/repair/judge，写
  `~/.config/ubt/config.toml` 0600，Test connection）；仅在 loopback 或已认证时可写。
- 回归：`tests/unit/api/test_console_contract.py` 加 `/setup` 的 OpenAPI 存在性与鉴权矩阵。
- 验收：新用户在 UI 内完成配置并提交第一个作业（不计 60s KPI，先保证“不离开 UI”）。

**E2 Mission Control 交付物 + SSE（前端）**

- 改动：`has_output === false` 时禁用 Download；deliverables 改成菜单（用 `item.label/media_type`）；
  `html` 加进 `DELIVERABLE_LABELS` 与 `deliverable_paths`；SSE 订阅前先 `GET /status`，404/terminal 不订阅；
  `onerror` 到达 401/403/404 时停止并显示原因。
- 回归：vitest 两条（hidden download when `has_output=false`；terminal job 不订阅）。
- 验收：CLI 作业点下载不再弹“not ready”，而是按钮禁用 + 提示原因。

**E3 Review 脏稿守卫 + 编辑入口（前端）**

- 改动：显式 Edit 按钮 + `tabIndex/onKeyDown`；`useBlocker` + `beforeunload` 脏稿确认；
  术语级联后用 `block_id` 重新定位，不清空 `segments`/`issueCursor`。
- 回归：vitest（编辑中离开触发 confirm；级联后 `issueCursor` 保持）。
- 验收：手动测试：编辑 → 切换路由 → 弹确认。

**E4 资产页真实分页 + a11y/i18n 最低集（前端）**

- 改动：`/assets/tm` 增加 `q`/`offset` 参数（后端 `list_tm_entries` 已支持 limit/offset）；
  glossary 未配置时显示可操作状态；删除加 `ConfirmDialog`；**删掉** fuzzy 阈值滑杆（后端无参数）；
  dropzone 改 `<button>`；删除按钮加 `aria-label`；`document.documentElement.lang` 跟随语言；
  `ISSUE_LABELS` 与残留 raw 文案入 i18n。
- 回归：`tests/unit/api/test_asset_import.py` 加 `q` 搜索；vitest 两条（dropzone 可键盘、删除需确认）。
- 验收：`pnpm lint && pnpm build && pnpm vitest run`。

### 2.2 回归测试清单（原 §9.6）

| 测试 | 类型 | 覆盖 | 优先级 |
| --- | --- | --- | --- |
| `tests/unit/core/test_source_coverage.py` | 新 | A5 源覆盖门 | P0 |
| `tests/unit/adapters/test_docling_cache_coverage.py` | 新 | A5 坏缓存丢弃 | P0 |
| `tests/unit/test_writer_lock.py::test_renew_after_takeover_fails` | 加 | A6 renew | P0 |
| `tests/unit/test_writer_lock.py::test_fresh_reset_keeps_the_lock_row` | 加 | A6 `--fresh` | P0 |
| `tests/integration/test_cli_to_api.py` | 新 | A1 CLI→API 指针 | P0 |
| `tests/unit/core/test_export_render_quality_flags.py::test_missing_asset_is_missing` | 加 | A4 缺图 | P1 |
| `tests/unit/core/router/test_router_funnel.py::test_truncation_is_visible_to_the_caller` | 加 | A7 截断 | P1 |
| `tests/unit/core/test_emphasis.py`（数学内 marker） | 加 | A8 粗体吞公式 | P1 |
| `tests/unit/qe/test_defect_taxonomy.py::test_pass_respects_the_shippable_floor` | 加 | B8 | P1 |
| `tests/unit/core/test_provider_defaults.py`（`rerank_k`） | 加 | B9 | P2 |
| `web/src/views/quality-gate/QualityGateView.test.tsx` | 新 | A2 三态 | P0 |
| `tests/unit/engine/test_repair_no_span_overwrite.py` | 新 | B5 整块覆盖 | P1 |
| `tests/unit/validators/test_consistency.py`（6 正 + 3 反） | 加 | B1/B2 | P1 |
| `tests/unit/qe/test_fast_pass.py`（4 正 + 2 反） | 加 | B3 | P1 |
| `tests/unit/qe/test_term_drift.py`（person） | 加 | B4 | P1 |
| `tests/unit/engine/test_draft_neighbor_context.py` | 新 | C2 | P2 |
| `tests/unit/engine/test_draft_policy.py` | 新 | C3 rolling 边界 | P2 |
| `tests/unit/render/test_html_view.py`（`<ol>`/`<br>`/`<figcaption>`） | 加 | D5 | P2 |
| `tests/unit/render/test_epub_view.py`（**新增**；全仓目前没有 EPUB view 测试） | 加 | D5 | P2 |
| `tests/unit/adapters/test_docling_heading_levels.py` | 新 | D1 | P2 |

