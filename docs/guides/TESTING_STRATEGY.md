# 测试策略：分层、契约优先

> **状态**：🟢 活文档（随测试演进更新）
> 本文是测试**如何写、写到哪、怎么跑**的唯一权威说明；
> 与 `docs/guides/CI_AND_QUALITY_GATES.md`（门禁在哪执行）配套阅读。

---

## 0. 原则（先读这一节）

1. **测契约，不测现状。** 断言“架构要求必须成立”的不变量（如源切片无损、渲染永不丢内容），
   而不是“当前实现恰好如此”。契约测试在重构中存活，现状测试只会随实现漂移。
2. **已确认缺陷用 `xfail(strict=True)` 记录。** 已知 bug 不是“先跳过”——写一个会命中的
   失败用例并标注原因；修好后它变成 `XPASS` 并**让套件变红**，强制移除标记。禁止 `skip`。
3. **隔离外部世界。** 后端、校验器、模型、网络都用测试替身注入；替身只替换**边界**，
   不 mock 被测对象内部。
4. **一个测试一个失败原因。** 用 `parametrize` 覆盖闭合集合（如 `TEXT_ELEMENTS`），
   而不是把多个断言塞进一个用例。
5. **会挂死或污染全局的用例放子进程。** 配硬超时（见 H1 回归），否则一个死锁会拖垮整个会话。
6. **用定向变异证明测试有效。** 每个模块写完后，对产品代码注入一组定向缺陷（改门限、翻转
   布尔、去掉守卫），确认套件**变红**；未能捕获的变异要么补用例，要么记录为冗余防御。
   还原后必须 `git diff --exit-code` 零残留。**坑：** 变异脚本**每次跑 pytest 前、以及最终
   还原后**都必须清理 `ubt/**/__pycache__`——同尺寸（`0.0`↔`1.0`）且同 mtime 的还原会让
   Python 复用**变异的字节码**，导致还原后测试仍按变异行为运行（曾两次踩中：一次是还原后
   测试仍失败，一次是"未捕获"的假象）。

---

## 1. 分层标记（与 `pyproject.toml` 的 markers 一致）

| 标记 | 语义 | 运行命令 | 预算 | pre-push |
| --- | --- | --- | --- | --- |
| `fast` | 纯逻辑单元测试，无网络 / 无重依赖 | `uv run pytest -m fast -q` | ~30s（`-n auto` ~10s；含覆盖率 ~43s） | ✅ 必跑 |
| `slow` | 重子进程（typst 编译、`python -m ubt` 子进程）或渲染 A/B | `uv run pytest -m slow -q` | ~2.5 分钟 | CI 必跑 |
| `network` | 需真实网络（如 Docling 模型下载） | `uv run pytest -m network -q` | 视网络 | — |
| `legacy_drift` | 隔离的遗留检查，阻塞且考古成本 > 15min | 全量运行时带出 | — | — |

**每个测试必须携带 `fast` 或 `slow`。** CI 的两条 pytest 步骤都显式传 `-m`，而
`-m` 是**替换**而非叠加 `addopts` 里的过滤器，所以 CI 实际只跑这两个档。既无 `fast`
也无 `slow` 的测试**不被任何门禁执行**——手工跑仍然通过，于是不会变红，只是静默地
不再保护代码。`tests/unit/test_marker_hygiene.py` 用纯 AST 拦截这一情况（以及同时带
两个标记的矛盾情况）。

```bash
uv run pytest -m fast -q              # 内循环：改代码后立刻跑
uv run pytest -o addopts="" -q        # 全量本地：清空标记过滤与 --timeout，跑所有层
uv run pytest -m slow -q              # 只跑重子进程层
```

> `-m fast` / `-m slow` 在命令行**替换** `addopts` 里的 `-m 'not slow and not
> legacy_drift'`（pytest 的 `-m` 是覆盖，不是取交集），因此两者合起来并不等于全量。
> `-o addopts=""` 会**同时**清掉标记过滤和 `--timeout=600`，这是有意为之。

---

## 2. 目录约定（镜像 `ubt/`）

```
tests/
├── conftest.py        # 全局保证：禁网 + UBT_* 环境隔离（见 §3）
├── pdf_builders.py    # 合成 PDF 写入器（pypdf，独立于被测引擎）
├── unit/              # 纯逻辑，标 fast —— 与 ubt/ 的模块一一对应
└── integration/       # 重子进程 / 端到端，标 slow
```
- `network/` 与 `legacy/` 目录**尚未建立**：`network` / `legacy_drift` 两个标记已声明在
  `pyproject.toml`，但当前分别只有 0 个用例使用（`network` 的豁免机制仍由 conftest
  的 `_block_network` 提供并测试）。需要时再建目录，不要预留空目录。
- 文件名 `test_<被测模块>.py`；目录结构镜像 `ubt/`，便于定位。
- 测试名描述**契约**（`test_a_block_kept_in_source_shows_its_source_text`），不描述实现。
- 合成 PDF 夹具由 `tests/pdf_builders.py` 在 `tmp_path` 下按运行生成（见
  `tests/unit/adapters/test_pdf_adapter_e2e.py`），不写入 `tests/fixtures/`，因此无需入库。

`tests/unit/test_no_orphaned_modules.py` 与 `tests/unit/test_reference_resolvability.py`
是两道结构护栏：前者用纯 AST 拦截无人 import 的孤儿模块，后者拦截注释/文档里指向已删
文件或符号的引用。二者都必须在 fast 档保持绿。

---

## 3. 全局保证（写在 `conftest.py`，不在每个测试里重复）

`tests/conftest.py` 提供两条项目级保证：

- **禁网**：autouse 的 `_block_network` 让任何 `socket.connect` 直接抛错；确实需要网络的
  用例显式标 `@pytest.mark.network` 才豁免。这样 `fast` 层保证离线可跑。
- **确定性**：autouse 的 `clean_ubt_env` 清除全部 `UBT_*` 变量。`UBTConfig` 是 pydantic-settings 对象，
  开发者 shell 里残留的 `UBT_*` 会悄悄改变构造结果（包括 provider 与 key 选择，进而改变 keyless 提交是否自动彩排）。

单测**不得**依赖网络或可选重依赖（`docling`/`torch`/`rapidocr`）；需要时归入
`slow`/`network` 层并在文档标注所需 extra。

---

## 4. 覆盖清单（按被测模块）

fast 档的单元测试与 `ubt/` 的模块一一对应；下表列出每个模块钉住的核心契约。

| 文件 | 钉住的契约 |
| --- | --- |
| `tests/unit/test_writer_lock.py` | 锁存放在账本数据库的 `job_locks` 表中，基于 `(db_path, job_id)` 的租约机制；acquire 记录 holder pid 与 token；第二个写者被 `LedgerWriterLockConflictError` 拒绝且 `details` 带 `job_id`/`holder_pid`；acquire/release 幂等、context manager 异常也释放；支持过期租约原子回收（防止进程崩溃永久死锁）；**跨进程**互斥 |
| `tests/unit/test_job_queue.py` | 入队幂等（live/completed 原样返回、failed/cancelled 同租户原地重排、跨租户不抢占）；深度上限 `QueueDepthExceededError` 仅计 QUEUED 行；claim 按优先级+FIFO、attempts 递增、设租约；全局/租户运行上限与跳过；过期租约先回收再选取；心跳/进度/取消仅限当前 owner；`complete` 拒绝非终态、丢租约返回 False、`progress=None` 保留旧值；`release_claim` 退还 attempt 且尊重待处理取消；`reclaim_stale` 重排/达上限失败/尊重取消、到期边界严格 `<`；查询与跨重开持久化 |
| `tests/unit/test_ledger.py` | schema 落在 `TARGET_SCHEMA_VERSION` 且 `_REQUIRED_BLOCK_COLUMNS` 齐全；重开幂等、`read_only` 缺文件抛 `LedgerError`、`close` 幂等且惰性重开；`blocks_seq` 仅写递增；manifest 初始化幂等；job_id/doc_id 解析；指纹与 metadata 往返；usage 往返与 `merge_usage_totals` 纯函数；`finalize_job` completed 拒绝非终态块、failed 宽松、failed 不覆盖 cancelled；**复合主键**跨 job 同 block_id 不串行；append upsert 与 total_blocks；checkpoint 全字段；batch 更新计数、省略键 COALESCE 保留、显式 `[]` 清空、非终态不复活终态行；keyset 分页、排除 skip_translate；`reset_blocks_to_pending` 清空全部裁决；`get_job_stats` 计数；上下文尾/首；batch 持久化/liveness/`reserve_batch_job` 与陈旧预留回收 |
| `tests/unit/test_config_construction.py` | `UBTConfig()` 在任何环境下都可构造；能力覆盖不产生重入死锁（回归直接断言，套件无 `xfail`） |
| `tests/unit/cleaners/test_maskers.py` | math/code/citation 三类掩码共享同一 protect-restore 契约：往返恢复原文；`UnmaskReport` 六桶各自可命中、`clean` 仅在零桶时为真；改写校验和→mismatched、删除→missing、泄漏→duplicated、交换→reordered；无校验和回显时 math 恢复标 `unverified`、citation 拒绝标 `mutated`；分隔符/货币/`inline_math` 守卫 |
| `tests/unit/cleaners/test_math_text.py` | `extract_text_spans`（命令族/平衡花括号/跳过不匹配/不下潜嵌套）、`is_translatable_text`、`translatable_spans`；`skeleton`/`skeleton_holds`（命令区分、空白归一、计数/外部改动拒绝）；`reassemble`；异步 `translate_math_text` 四种结果全部 fail-closed |
| `tests/unit/cleaners/test_html_sanitizer.py` | 允许列表清洗 fail-closed：fragment 丢弃危险容器、剥离事件处理器/js/data 导航、保留安全链接与 `data:` 图片、转义解码尖括号、保留 `List<T>`、幂等；inline 只改标签不动文本；`scrub_source_document` 字节级不变 / 幂等；`scrub_css_text`；`strip_html_mark_tags` |
| `tests/unit/cleaners/test_cjk_spacing.py` | CJK 空格删除、保留 CJK-Latin 空格与换行、spaceless script 门、pangu 插入、`sin(x)` 不破坏、受保护 span、货币非数学、出版标点；组合管线幂等；全角句号 `。` 归一 |
| `tests/unit/cleaners/test_skip_rules.py` | `classify_skip` 的 precision-first 契约两侧：handle/arxiv/括号引用/`in_bibliography`/byline/debris 命中；near-miss **保持可翻译**；`is_heading` 抑制 byline；byline 扫描对抗尾串线性（ReDoS 回归护栏） |
| `tests/unit/cleaners/test_dynamic_boilerplate.py` | "删 chrome 不删正文"两侧：页标记/罗马页码/running header 被删；**4 位年份、`IV.` 标题、罗马字母拼成的词、整块即数字**必须保留；尾部照片署名与 disclaimer 被删，仅前缀相似的正文句保留（0.85 相似度守卫） |
| `tests/unit/test_translate_cache.py` | `step_key` 的身份性；`DiskCacheStore` 缺失即 miss、分片路径往返、`get_or_compute` 命中不计算/未命中只计算一次；`TranslationEngine` 缓存键覆盖 masked source **+ model + prompt_version + prompt context**，任一变化即 miss；fail-open（损坏条目/键不匹配/不可用 store 均回退 provider）；非草稿值缓存按 `kind` 隔离 |
| `tests/unit/test_tm.py` | TM 纯函数：**极性守卫**、`compute_tm_context` 指纹、`normalize_for_tm`、`_fts_query_terms`、`format_few_shot_reference`；**SQLite 存储**：writeback→`lookup_exact` 往返、语言对隔离、**最新译文胜出但 `human_pe` 标签不被降级**、写回不计 reuse、身份行拒绝、context/domain 读门、**`human_pe` 排序压过精确 context 机器行**、fuzzy scan、FTS5 预筛；**跨实例 generation** |
| `tests/unit/test_contract.py` | 交付契约 `reconcile`：文本 TRANSLATED/VERBATIM 无违规、SOURCE_KEPT 默认 warning 但 **space-failure 升级为 error**、SKIPPED→`TEXT_UNDELIVERED`、PENDING→`TEXT_UNACCOUNTED`；资产 PRESERVED_OPAQUE/DROPPED 计入、MISSING error、未验证 warning、corrupt error；report 的 totals/errors/warnings/`passed`/`summary_line` |
| `tests/unit/qe/test_fast_pass.py` | 0-token 快检门：`is_verbatim_echo`、`is_near_verbatim_echo`、`grid_columns`/`markdown_grid_shape`、`_detect_line_repetition_loop`；`FastPassFilter.evaluate` 的干净通过/空目标/XML 泄漏/回显/循环拒绝 |
| `tests/unit/qe/test_score_policy.py` | QE 计分策略（**按 provenance 而非魔法分数值排除**）：`is_qe_scored`（`None`/`skip_translate`/`tm_hit` 任一即排除、**真实 1.0 计入**）；`qe_scored_values`、`average_qe_scored`、`QE_SCORED_SQL` 谓词形状 |
| `tests/unit/qe/test_defect_taxonomy.py` | 缺陷旗标分类单一真源：**三种"未翻译"措辞同时属于 structural 与 critical**；masker token 损坏标记为 structural+critical；术语违规/视觉差异 structural 但非 critical；format-only 类仅当每个旗标都是 format-only 才为真；`is_transient_failure`/`is_repair_only_transient_failure`/`is_transient_lifecycle_only`；**critical 表 ⊆ structural 表** |
| `tests/unit/qe/test_term_shape.py` | 源逐字术语形状与句计数：`is_identifier_shaped`（结构标签非标识符、数字/camelCase/下划线/点为标识符、**3+ 全大写缩写是**、**1–2 字符全大写与常见大写词不是**）；`is_verbatim_carryover`；`count_sentences`（缩写/小数不裂句、CJK 全/半角终止符无空白即裂） |
| `tests/unit/qe/test_term_drift.py` | 术语漂移的**单一检测原语**（五个消费方共用）：`normalize_glossary`、`GlossaryTerm.source_surfaces`、`TermDriftFinding` 真值表；`detect_term_drift`（**源侧严口径 `[source]+aliases`**、仅源命中才扫目标、**首命中 surface 胜出且源优先于 alias**、双侧大小写折叠、双侧受保护区间遮蔽、**inflected 变体在规范形缺席时满足渲染**）；`detect_target_term_violations` |
| `tests/unit/qe/test_term_metrics.py` | 术语 TP/TF 指标：`TermDrift.drift_rate`；`_fuzzy_match`（**<3 字符仅精确**、阈值门控）；`summarize_drift`（按 source 聚合、**只保留确有漂移的术语**、**按漂移率→出现次数→source 最坏优先排序**）；`evaluate_terms` |
| `tests/unit/qe/test_added_content.py` | 0-token 增内容门：`reference_tokens`（引用/列表/区间展开、**超长区间不展开**、docling 空格版本号折叠）；`markdown_headings`（0–3 空格缩进、**4 空格不算**）；`AddedContentGate.evaluate`（**裸括号量词豁免**、**目标独有标题→scaffold 泄漏**） |
| `tests/unit/qe/test_consistency_enforce.py` | 术语漂移→有界受约束重译计划：`plan_consistency_tasks`（**非正 cap 禁用**、**精确命中不规划**、重复命中折叠、**按 `(block_id, source)` 稳定排序**） |
| `tests/unit/qe/test_omission.py` | 0-token 遗漏门（三信号）：`singular_variant`、`_digit_glued`、`_merge_candidates`、`_is_hyphen_ghost`、`identifier_terms`、`_math_term_variants`、`_match_term_surfaces` **验收阶梯**、`_is_table_content`；`OmissionGate.evaluate`（**句比信号**、**短源不布防**、**表格豁免**、**单术语不布防**） |
| `tests/unit/memory/test_cjk_matcher.py` | 术语匹配与块选择：`count_term_in_text`（拉丁用词边界、CJK 用子串、**单字 CJK 一律拒绝**）；`select_terms_for_chunk`（本地命中优先于全局 top-N、**CJK 复合片段降权但不丢弃**、≥8 词 Aho 预筛语义不变） |
| `tests/unit/memory/test_abbreviation_miner.py` | "全称 (缩写)" 0-token 挖掘：`_find_expansion_for`（**贪心假前缀拒绝**）；`mine_abbreviations`（**首次出现胜出**）；`mine_abbreviations_stream`（与整体 join 等价、块边界跨接仍可挖） |
| `tests/unit/memory/test_glossary_table.py` | 术语表两形态：`build_global_glossary_table`（**专名优先→词频→source 确定性 tie-break**）；`build_chunk_glossary_table`（术语表 + **仅当块内出现的本地缩写块**） |
| `tests/unit/cleaners/test_inline_math.py` | 货币感知的行内数学：`is_math_content`（空/纯数字/货币区间→非数学、**括号粘连散文→非数学**）；`inline_math_spans`（`$5 and $10` 不成对、**被拒候选不遮蔽后续 span**） |
| `tests/unit/cleaners/test_boilerplate_catalog.py` | 多语言样板目录：`_primary_lang_code`、各语言 running header/legal/photo-credit 模式、**区域子标签用主语言模式**、未知语言回退英文、lru_cache 同对象 |
| `tests/unit/cleaners/test_lnds_pruner.py` | LNDS 页码剪枝 + Calibre 清洗 + 学术 PDF 数学修复 + 块去重：`is_ascii_digit_line`（**ASCII 门严于 `isdigit()`**）；`detect_page_number_lines`（双门）；`collect_dropped_line_indices`（**数字后接标题受保护**、**冒号引导的有序列表整段保护**）；`normalize_academic_pdf_math`（**`\x00SPAN` 哨兵存活**、**`$`/`ð`/`Þ` 修复受等式证据门控**）；`dedup_duplicate_blocks`（**同归一化文本 + IoU≥0.6**） |
| `tests/unit/cleaners/test_soup_math.py` | 无定界符 unicode 数学汤的**精确优先**检测：`_qualifies`（**裸标识符与散文恒假**）；`find_soup_spans`（**尾随单位词守卫**、**已存在 token 内部被扣除**）；`SoupMathMasker`（`unmask` 恒等往返） |
| `tests/unit/validators/test_glossary_enforcer.py` | 确定性术语强制：`find_term_occurrences`（词边界、`case_insensitive` 不改偏移、受保护区间排除）；`DeterministicGlossaryEnforcer`（alias→canonical、**alias 与已批准译文冲突不改写**、**CJK 复合守卫**、受保护区间绝不改写、幂等、**最长匹配优先**） |
| `tests/unit/validators/test_consistency.py` | 0-token 数值/术语一致性（**匹配视图，存储文本永不改动**）：`_cn_numeral_value`、`_expand_scientific_not`、`canonicalize_numeric_token`、`normalize_for_numeric_matching`（**词内数词不误折**）、`scale_equivalent_values`（**SI 符号大小写敏感**、**`10千克` 单位前缀守卫**）；`NumericConsistencyValidator`（**惯用语豁免按出现位置作用域**）；`GlossaryConsistencyValidator`（**按 source 长度降序排序**） |
| `tests/unit/validators/test_span_repair.py` | GEMBA-MQM 误差跨度标注与确定性拼接：`severity_for_error_type`（**子串回退**）；`MQMSpanAnnotator.annotate_draft`（**属性与内容 HTML 转义**、**CJK 扩张别名不标注**、**重叠跨度折叠为最先命中**）；`SpanRepairSplicer.splice_repairs`（**不可拼接时优先原草稿而非协议标记**、多跨度**从右到左**拼接） |
| `tests/unit/validators/test_html_delta.py` | HTML/Markdown 结构增量（**结构错误硬失败，格式漂移仅告警**）：缺失/多余图片为 `HTML_DELTA_MISMATCH`；**转义 `\![]()` 不计**；**脏源忠实翻译相消**；新引入的畸形 `<img>` 硬失败；**自闭合 `<img/>` 与大小写不敏感** |
| `tests/unit/validators/test_math_guard.py` | 共享数学碎片检测（Gate 1）与公式不变式（Gate 3）：`novel_unsupported_latex_commands`（**可渲染命令放行**、**黑板字母身份比较**）；`looks_like_math_debris`（**运行词否决**）；`formula_skeleton_intact`（**`\text{}` 散文跨度翻译合法**）；`apply_math_guards`（**幂等**、**绝不改块状态**） |
| `tests/unit/metrics/test_schema.py` | 版本化 KPI schema：`SCHEMA_VERSION` 钉死；定义名唯一且与 `KPI_BY_NAME` 索引一致；**floor 只用于 higher-is-better、ceiling 只用于 lower-is-better**；`KpiSet` 默认 `schema_version=0` 与空映射、**frozen** |
| `tests/unit/metrics/test_compare.py` | 方向感知的 KPI 回归门：`compare_kpis`（**候选新增 KPI 记入 `unknown_kpis` 且永不回归**、`strict_names` 把缺失基线变违规、**非有限值违规**）；`compare_kpi_sets`（结构 `details` 键精确比较、**schema 版本不匹配违规**）；`check_thresholds`（floor/ceiling 违规、**边界值通过**） |
| `tests/unit/metrics/test_collect.py` | 从报告工件派生 KPI：`_div`（零分母守卫）、`_flag_count`、`_severity_counts`、`_visual_page_count`（**取最大命名页而非计数**）；`collect_kpis`（**空分母守卫**、成本按字符缩放、视觉比率用页数分母）；`save_metrics_report`/`load_kpis` 磁盘往返 |
| `tests/unit/memory/test_rolling_summary.py` | 跨章滚动摘要：`extract_chapter_id`（**无分隔符→"" 关闭跨章摘要**）；`collect_chapter_text`（**优先 target→draft→source**）；`deterministic_summary`；`summarize_chapter`（**长章回退摘要有 400 上限**、**超长摘要在 800 内按句界截断**） |
| `tests/unit/memory/test_neighbor_window.py` | 只读邻域滑动窗口：`extract_excerpts`（**前文取尾部、后文取头部**）；`extract_from_blocks`（**严格 flow 隔离**、**前文优先 finished target 否则 source**、**后文用 source**、**跨过窗口两端用 fallback 补齐**） |
| `tests/unit/memory/test_skeleton_extractor.py` | LLM 文档骨架术语抽取：`_count_term_occurrences`（**词界 + 末词屈折**、**前缀词不误命中**）；`build_document_skeleton`（**尾部片段仅当 ≥120 字符余量才保留**）；`extract_skeleton_terms_llm`（**骨架 <30 字符不调用 LLM**、**锚定文档并按真实出现次数排名**） |
| `tests/unit/memory/test_bible.py` | 术语圣经条目清洗与合并：`clean_bible_entry`（**CJK 源 >12 字符丢弃**、**字母源 >4 词丢弃**、**剔除自别名**）；`merge_bible_entries`（**按 `source.lower()` 合并**、首次译文胜出但可回填空译文、别名并集且排序） |
| `tests/unit/memory/test_abbreviation_backfill.py` | 缩写译文批量回填：`_clean_translation`（**大小写无关回显缩写→None**）；`_apply_response`（**同一缩写首次胜出**、**不覆盖已有译文**、**`1.2.` 小数编号不被剥离**）；`backfill_abbreviation_translations`（**共享 key 的变体归组且一次调用填全部**、**跨批次响应不泄漏到他批**） |
| `tests/unit/memory/test_seed_glossary.py` | 领域种子术语表与外部表加载：`seed_entries_for_profile`（**外部资源目录优先于内置种子**、**路径穿越/分隔符 profile 名被拒并告警**）；`load_external_glossary`（JSON 映射式/列表式、CSV 带表头且列可重排、**条目经 clean 守卫过滤超长口号**） |
| `tests/unit/policy/test_adaptive_policy.py` | 粒度策略解析：`RENDER_ENGINE == "overlay"` 为常量；`resolve_adaptive_policy`（短文档三门全开、长文档全关且 `visual_blocking` 跟随配置） |
| `tests/unit/policy/test_bilingual_advisor.py` | 零 LLM 双语模式顾问：词表与阈值、`_PROFILE_SENSITIVITY`、每条规则的 **fires 边界**且每规则对 inline 惩罚 ≥ alternating ≥ monolingual；`collect_signals`；`advise_layout`（**tier 跟随被请求模式自身分数而非榜首**、**0.70 / 0.40 精确边界**）；`assess_difficulty`；`downgrade_mode`；`resolve_effective_mode` |
| `tests/unit/policy/test_layout_policy.py` | 统一 policy/knob 注册表：**每条 `CALIBRATION` 都是带非空 rationale 的 `KnobMeta`**；`calibration_summary()` 计数；`rigid_min_font_pt_for`（**caption/footnote 阶梯降到 6.5**）；`is_non_prose_degradable`（**prose 永不降级**、**致命 prompt 泄漏标记仍拒降级**）；词表集合；判据/续接正则 |
| `tests/unit/layout/test_theme_rtl.py` | 方向检测表：RTL 基语言、**script 子标签压过基语言**（`ar-Latn`=LTR、`en-Arab`=RTL）；RTL 目标 resolve 出 RTL theme 且字体栈非空；fragment typesetter 的 `_dir_line` 仅 RTL 非空 |
| `tests/unit/segment/test_xliff_roundtrip.py` | XLIFF 2.1 双射：每段 source/target/state/占位符全字段往返；头部 src/trg/original 幸存；各 kind 的 original 精确回取；`xml_safe` 只剥 XML 1.0 非法控制字节 |
| `tests/unit/analyze/test_bridge_roundtrip.py` | IRBlock↔typed AST 往返：全字段无损、spine 序恢复、同 region 连续分组的边界切分；**执行态刻意不桥接** |
| `tests/unit/translate/test_engine_echo_vandal.py` | 引擎双通过：echo provider 全段 TRANSLATED、target==`xml_safe`(原文)；vandal provider 剥掉全部 token 后**凡带占位符的段必 BLOCKED 且带旗标** |
| `tests/unit/analyze/test_reader_md_spans.py` | 原生 Markdown 读者：每元素 `Span.chars` 从 `CanonicalSource.text` 切出**恰好自身文本**；桥往返无损 |
| `tests/unit/analyze/test_reader_formats.py` + `tests/unit/adapters/test_format_adapters.py` | HTML/EPUB/DOCX 原生读者的 span 精确性 + 桥往返 + 结构下限，夹具全部合成；三个 adapter 的 parse 块数下限 + monolingual/bilingual 渲染 |
| `tests/unit/render/test_html_view.py` | 视图：HTML 携带每个块的交付文本（重建走译文、保源走原文）、**placement 总数==块数**；EPUB 是合法 OCF 容器；**保源块显示源文本** |
| `tests/unit/test_artifact.py` | `DeliveredArtifact` 双身份（sidecar/companion 键在*返回*文件、sibling 键在*请求* target）；`_present` 的 token 重叠探针（噪声下限 + `>=` 阈值）；`ArtifactReport` 的 missing/passed/summary |
| `tests/unit/test_run_stages.py` | 阶段计划：全序 + render/cost 预检先于 billable bible；`chapter_streaming` 门替换 draft 阶梯、`c_text`/`consistency` 门各只关自己的阶段；事件按阶段顺序冒出；export 终态事件**先跑 hook 再 yield** |
| `tests/unit/test_render_plan_fields.py` | 字段归属守卫：模式簇在 `RenderPlan`/`RenderOutcome` 上，`RunMetadata` 不得携带 |

### slow 层

`tests/integration/` 下的重子进程/端到端测试（typst 渲染、页面预览、合成 PDF e2e）标 `slow`，
默认不跑、CI 必跑。CI 装有 checksum-pinned `typst`，因此该层在 CI 里**真实执行**（~26 条
必跑；40 条里另有 8 条因依赖 skip、6 条参数化因语料缺失收敛为 0——见下方 skip 清单）；
需要 gitignore 语料的 harness（`test_corpus_acceptance.py`、
`test_publish_corpus_sweep.py`）在语料缺失时自动 skip，但语料 gate 的机器由
`test_synthetic_corpus_gate.py` 用自建 corpus 在 CI 里执行（`ubt verify --corpus --run`
的正/负两条路径都钉死）。

**CI 里 skip 的三组（有意为之，不是遗漏；2026-10 实测核账）**：

- `test_pdf_engine_differential.py`（6 条）：需要 `pdf` extra（Docling），CI 不装该 extra
  （HuggingFace 模型体积/下载时间不划算），整模块 skip → **双引擎差分守卫只在本地跑**。
- `test_corpus_acceptance.py`（2 条）：需要 gitignored 的 `corpus/documents/` 真实语料
  （`scripts/fetch_semantic_gold.py` 拉取），CI 无语料 → **真实文档 gate 只在本地跑**。
- `test_publish_corpus_sweep.py`（6 条参数化）：同样依赖真实语料，CI 收敛为 0 条。

这三组的"守什么"由各自的 fast 档替代物与 `test_synthetic_corpus_gate.py` 承接；在本地跑
`uv run pytest -m slow -o addopts=""` 才是完整守护面。CI 数值（26 必跑 + 8 skip + 6 收敛）
若与实际不符，以 `pytest -m slow -q -rs` 的输出为准。
`tests/integration/test_pipeline_smoke.py`
用 dry-run 编排器驱动真实全链路，零 token、零网络，因此标 `fast` 而不是 `slow`：它
是端到端的，但不需要任何重依赖，且 1 秒内跑完。

### 测试系统（区别于"测试集合"）

除模块契约外，四类跨模块不变量已有 fast/slow 覆盖，各自经过定向变异验证（改坏
被保护的行为必须变红）：

- **属性测试**：`tests/unit/qe/test_mask_transpile_properties.py`——种子化
  `random.Random`（失败打印种子）生成语料，钉掩码往返无损、伪造 token fail-clean、
  LaTeX→Typst 转换的配平/不泄漏/正向转换不变量。
- **引擎差分**：`tests/integration/test_pdf_engine_differential.py`——pdfium 与
  Docling 各自对照**源文本**断言字符级无损（缺字能定位到哪个引擎）。
- **crash/resume e2e**：`tests/integration/test_crash_resume_e2e.py`——中断后的
  ledger 可恢复、resume 不重计费已入账的 draft。
- **golden 几何**：`tests/unit/adapters/test_golden_geometry.py`——/Rotate、双栏、
  CropBox≠MediaBox 的精确坐标，帧算术回归即刻变红。

## 5. 与质量门禁的关系

本地 `pre-push`（`.pre-commit-config.yaml`）执行：

```bash
uv run mypy --strict ubt tests     # 类型：全仓，含测试
uv run pytest -m fast -q           # 快测：全仓，内循环契约（带覆盖率下限）
```

两者都 `pass_filenames: false`——推送前跑**全仓**，避免"改动的文件恰好干净就算过"。
CI 复现同一套命令，见 `docs/guides/CI_AND_QUALITY_GATES.md`。

> 快测层当前约 55 秒、约 4,465 passed（`-n auto` 并发下 ~15s；含覆盖率统计 ~107s）。
> pre-push 的 `pytest-fast` 带 `--cov=ubt --cov-fail-under=70`（回归护栏，非质量目标；
> 下限 = 基线（~78%）减 8 个百分点，见 `docs/guides/CI_AND_QUALITY_GATES.md`）。
>
> `tests/unit/test_marker_hygiene.py` 用纯 AST 守三条标记契约：`slow` 测试不得
> 继承 `fast`（`pytestmark` 会把模块标记与到每个测试上，把重子进程循环拖进内循环），
> 每个测试都必须带 `fast` 或 `slow`（否则任何门禁都不会执行它），且标了 `fast`
> 的模块若 spawn 子进程必须在显式 allowlist 里注明理由。

---

## 6. 结构护栏

除契约测试外，fast 档有三道自动护栏，防止"测试全绿但代码腐烂"：

- `tests/unit/test_no_orphaned_modules.py`：纯 AST 构建包内 import 图，任何无人 import
  的模块都让推送失败（仅靠字符串可达的子进程入口在白名单内显式登记）。
- `tests/unit/test_reference_resolvability.py`：拦截注释/文档里指向已删文件或符号的引用、
  `scripts/` 里 shell harness 调用的已删 pytest 目标，以及按"评审轮次"而非被测模块
  命名的测试文件。
- `scripts/check_doc_refs.py`：只读检查 `docs/` 里的 `path:line` 引用（手动运行）。

测试文件清单以 `git ls-files tests/` 为准，此处不再重复维护。
