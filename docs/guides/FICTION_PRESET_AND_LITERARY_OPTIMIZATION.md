# 小说/文学 Fiction Preset 与文学化优化建议

> 状态：草案待评审（novel/literary 尚未开始测试和优化）
> 方法：CodeGraph MCP（`codegraph explore` 双仓复核）+ 4 并行子代理（TBL 风格 / TBL 润色选型 / UBT 缺口 / 方案设计）
> 对照仓：TBL = `/home/kyle/Projects/python/TranslateBooksWithLLMs`，UBT = 本仓
>
> **时效提示**：本文写于预设只有 4 档（`publication / standard / preview / fast`）时。
> 现已增至 5 档，新增的 `publication_math` 只改 `formula_enrichment=on`（公式转矢量 LaTeX），
> 与本文论证的"缺 literary 旋钮"无关；下文所有"现有 4 preset"的表述按当时状态阅读。

---

## 0. 结论先行

1. UBT 现有 4 preset（`ubt/core/presets.py:69-118`）全部是论文公式向，无小说档。`fiction` 目前只是一个 prompt 层的字符串（`genre_profile`），不在质量分层里。
2. TBL 在小说上真正领先的三件套是：风格提取卡 + assembler 双块、文学重写二遍 refine、文学 benchmark 选型。不是模型更多，而是约束更对。
3. 建议新增 `fiction_light / fiction_prestige` 两档 preset，不动现有四档，不关任何正确性门（遵守 `presets.py:1-15` 正交原则），只调“花多少钱做文学质量”。

---

## 1. TBL 做得好的地方（可直接复用）

### 1.1 风格系统：学笔法不学故事

- 采样：`10000ch / 6段 / 每段≥1200ch` 分布式摘录，多文件均分预算。
  取证：TBL `src/api/blueprints/custom_instruction_routes.py:55-58,381-394`，`src/core/auto_prep.py:40-47,279-285`。
  作用：覆盖起承转合，不只学序章声腔。短书整本直送，长书均匀跳采。
- 提取 prompt：`src/prompts/prompts.py:477-623`，`STYLE_TAG_IN=<STYLE_JSON>`。
  五禁令：禁引号、禁例词（`such as/e.g.`）、禁专有名词、禁一词可满足的指令、只描述比例/频率/位置/对比。
  配额：`6-14条 / 每维度≤3 / 每条≤240ch`，宁缺毋滥。
- Schema 与解析：`src/core/style/extractor.py:32-101,141-232`。
  宽进严出：`STYLE_JSON → fence → find_balanced → try_repair_json`，`dimension` 不在白名单归 `other`，`instruction>500` 按词截断，`>40条` 截断，去重，`context>600` 截断且永不进 lint。
- 9 维度：`src/core/style/dimensions.py:7-17`
  `register / narrative_voice / sentence_rhythm / lexicon / imagery / dialogue / punctuation / formatting / other`。标签只做 UI 分组，永不写入 prompt。
- lint 五旗标：`src/core/style/lint.py:58-143`
  `quoted_example / example_marker / word_list / proper_noun / too_specific（<25ch）`。纯函数，宁误报不漏报。
  小说意义：禁复读金句、禁词库复用、禁意象口头禅、禁 A 书人名污染 B 书、禁口号式规则。
- 组装：`src/core/style/assembler.py:56-107`
  确定性无 LLM，产 `translation / refinement` 双块。无 context 时 `<preamble + body + ANTI_TIC>`，有 context 时插入 `## Setting + SETTING_GUARD + ## Style`。空规则回 `{None,None}`，guard 永不单独发射。
  双 guard 必须照抄原文：`_ANTI_TIC_GUARD:35-40`（倾向非词库，冲突让位自然）、`_SETTING_GUARD:45-48`（禁时代错位词）。
- mode 差异：`prompts.py:507-531`，`assembler.py:9-31`
  `source` = 保住这本书自己的声音+时代，setting 必填；`model` = 只借腔不借命，setting 强制空。auto 恒 `source`。
- auto 免配置：`src/core/auto_prep.py:241-312,315-405`
  只留 `flags==[]` 规则，双块分别进 `custom_instructions / refinement_instructions`，超时 420s 降级裸译，至多两条日志。
- 注入点：`src/prompts/prompts.py:242-309` 翻译走 system 头 `# STYLE INSTRUCTIONS` + 全程有效句；`src/core/translator.py:599-620` 润色走 `# ADDITIONAL REFINEMENT INSTRUCTIONS`。两 key 不互通。

### 1.2 润色二遍 refine：重写而非修复

- 人设：`src/prompts/prompts.py:712-756`
  `elite literary editor`，输入定性为 bad first draft，`You are NOT translating - you are REWRITING`。优先级 `flow > idiom > lexicon > rhythm > fidelity`，与一遍保义优先相反。特有同根词改写规则。
- 上下文链：`src/core/translator.py:887-909,942-954`
  `last_refined_context = 末25词` 滑动，失败清零阻断污染。试读对比时 `prev=""` 可接受，不可推导整书质量。
- 术语双块：`src/core/translator.py:36-130`，`src/core/glossary/filter.py:60-123`，`injector.py:45-177`
  `glossary` 按 chunk 过滤命中才发，`cast` 性别表全量每 chunk 都发（代词省略处最需要）。默认 `max_entries=50 / max_cast=80`，拉丁词界 CJK 子串，user 顶部注入保 system 可缓存。二遍扫的是 draft（目标语），若重写人名会漏检，移植时需改扫原文+映射。
- 自适应窗：`src/core/context_optimizer.py:322-329,356-540`
  初值 2048 步长 2048 上限 131072，二遍 floor 永不低于 4096，`was_truncated` 增窗，`RepetitionLoop` 双增，满窗且 `max_usage≤0.8` 才缩容。
- 失败策略：`src/core/translator.py:935-966,910-923`
  失败保留 draft，`mark_failed`，占位损坏回退，429 存 partial 抛给上层落盘。EPUB 用 `generate_post_processing_prompt(has_placeholders=True)`，TXT 通用二遍 `has_placeholders=False`，强度不可混用。

### 1.3 选型 benchmark：分数可比比模型多更重要

- 四维：`docs/JUDGE_RUBRIC_V2.md:32-116`
  `accuracy / fluency / style / overall`，`overall` 永不取平均，有硬顶（`acc<6→overall≤6` 等），无参考译文 cap 9.0，扣分表（contresens -2.0 等），feedback `≤200 ASCII` 禁引源文。
  旧 `benchmark/evaluator.py:70-146` 裸四维无锚定会导致全 8 分，不可再用。
- 单评委：`scripts/rejudge_all_via_poe.py:81-86`
  `JUDGE_ID=claude-opus-4-7-rubric-v2-poe`，`temp=0.1`。换 rubric 必须发新 judgments 文件。流程 `run(--no-evaluate) → add-translations → rejudge → aggregate`，译文不可变评分可重跑。
- 成本：`src/core/pricing/estimator.py:12-135`
  `560/chunk + 50×链` 开销，语种比（`en→zh 0.5-0.8` 等），`refine=+1 pass` 即双倍成本。未知模型标 `unknown` 不编 0。
- 血泪：`docs/PROVIDERS.md:106-112,253-273`
  OpenRouter 推理默认开可翻 2-10 倍成本，Poe 另有 web_search 税。对策：读目录按 catalog 关最低推理 + 关 search。小说整书不关即灾难。`refine` 应用比 `translate` 高一档的模型，否则二遍改不动文风。
- 遗传优化：`tools/prompt_optimizer/`
  四变异 `CORRECT / SIMPLIFY / REFORMULATE / RADICAL`，权重 `0.35/0.30/0.20/0.15`，罚项 variance + 超 500 字 + 专名 + train-test gap。`specificity_penalty` 词表机制必须保留，否则进化出的规则会把本书人名写死。

---

## 2. UBT 现状与小说缺口（8 个未测试风险）

### 2.1 现有 preset 与 prompt 三档

- `ubt/core/presets.py`：`publication / standard / preview / fast`，旋钮 `exec_mode + prompt_strategy + formula_enrichment + qe_engine + table_render_mode + emit_both + pdf_engine`。其中 3 个与公式/表格相关，无语气/人称/对话旋钮。
- `ubt/core/router/capabilities.py:8-13`：`RICH / MINIMAL / HYBRID`。`ubt/core/router/prompts.py:134-176,179-277,280-392`：
  `minimal` 无 system、无记忆；`hybrid` 短 system + 全记忆；`rich` 长 system（8 Rules + Bible 服从 + 去翻译腔 + OCR 去噪）+ 三层缓存 + `<translation>` 信封。注册表默认全 RICH，小模型被标 MINIMAL 会静默失忆。
- `fiction` 分支仅一句：`prompts.py:168-172,212-216,304-308` `Use standard X terminology / acclaimed literary translator`，记忆/门限/修复不跟随切换。

### 2.2 记忆在小说的短板

- L1 邻窗 `ubt/core/memory/neighbor_window.py:5,68-103` 前后各 300 字，同 flow 隔离，后文只用源文。对话短轮次易截断声线，代词易错。
- L2 `step_chars=3500` + L3 `l3_epoch_steps=4 / _MAX_L3_CHARS=1200 / 最早优先`（`hierarchical_memory.py:31-36,196-249`）是论文连续论证假设。小说伏笔跨数万字，确定性截断把人物弧光压成名词堆。`rolling_summary.py:38-55` 只取章 head 3000 字，章尾反转型系统性失明。
- Bible `ubt/core/memory/bible.py:8-96` 本体是术语表（人/地/词），无关系/声线/时态/敬语槽。首译胜出 + 全局 100 条（`glossary_table.py:29-59`）对低频关键凶手名不友好。
- TM `ubt/core/memory/tm.py:534-630` 句子精确复用 + `fuzz 0.85` 整串比。小说故意重复（复沓/咒语）被当复用，文学改写永远命中不了，只有人名条款生效。

### 2.3 QE 对文学的 6 处误伤（均已定位源码）

1. 长度比 `fast_pass.py:1030-1070` + `language_profile.py:183-218`：诗化压缩/铺陈无豁免，`omission.py:413-435` 合句/分句敏感。
2. 术语 `quality_gate.py:59-64` + `term_drift.py:176-258`：昵称/变称/代称判 drift，`consistency` 默认 off（`config.py:400`），开 repair 上限 50 条不够整书。
3. 重复 `fast_pass.py:797-884`：变奏复沓豁免失败 → `0.20` → repair 洗稿破韵律。
4. 脚本密度 `fast_pass.py:1071-1141`：有意保留异语段无作者意图通道，只能预标 `skip_translate`。
5. 标识符 recall `omission.py:177-244`：音译名无豁免。
6. `0.92 pass` 永不复审：`config.py:440-459` 默认 `judge 关闭 + pass_sample=0`，文学“正确但无味”出厂即 `MTQE_PASSED`。

### 2.4 repair ≠ refine

`RepairLoop.repair_single_block:152-509` 是缺陷驱动最小原位修正，Tier1 拼接守栏惩罚 out-of-span 改写。全仓 `grep refine` 唯一命中是 `extractor.py:71` 输出前缀兼容，无独立抛光阶段。人称统一/节奏/隐喻若未编码成 flag，永不触发改写。

取证缺口：`tests/` 下 `grep fiction|novel|literary` 仅命中 boilerplate，零 fiction prompt/记忆/QE 用例。

---

## 3. Fiction Preset 设计（可直接实施）

### 3.1 新增两档，不动旧四档

```python
# ubt/core/presets.py 新增
class FictionPreset(StrEnum):
    FICTION_LIGHT = "fiction_light"      # 类型网文，成本优先
    FICTION_PRESTIGE = "fiction_prestige"  # 纯文学，质量优先
```

建议初值（首轮影子跑后再调）：

| 参数 | fiction_light | fiction_prestige |
|---|---|---|
| exec_mode | auto | auto |
| prompt_strategy | hybrid_fiction（见 3.2） | rich_fiction |
| chunk | 1800 token / overlap 250，按 `\n\n`/引号闭合切 | 1500 / 350，同左 |
| temperature | 0.35 | 0.55 |
| memory_recall_k | 6 | 10 |
| tm_fuzzy | 0.82 | 0.75 |
| repair_max_rounds | 1 | 2 |
| polish | 关 | 开（见 3.3） |
| qe.heuristic_floor | 0.68 | 0.65 |
| qe.consistency | 0.95 | 0.98（一票否决） |
| qe.style | 0.70 | 0.80 |
| judge | tier1 only | tiered 必跑灰带+抽检 |

`minimal` 在 fiction 下禁用，强制升级 hybrid，防止丢人物。

### 3.2 style 资产与 prompt 插槽（不改原三档，包装）

新增 `ubt/assets/style_fiction.yaml`：

```yaml
style_card: {register, rhythm, narrator_distance, dialogue_markers}
character_voices: [{id, speech_style, sample_src, sample_tgt, taboo}]
```

新增 `ubt/core/router/prompts_fiction.py`，复用原 `build_*` 后追加插槽 `{style_voice} {character_voices} {rhythm_notes} {bible_persons}`。

- `rich_fiction`：`[术语/人名块] + [文风/人物声线块]`，TBL assembler 同款。
- `hybrid_fiction`：默认仅注 `bible_persons + style_voice 80字摘要`，省 token。
- 预读：全书前 5k token 跑一次 `extract_style_card`（抄 TBL `10000ch/6段` 分布式采样 + 五禁令 + lint 五码），存记忆顶层，chunk 级只读。

Bible 人物表扩展：`Person{id, name_src, name_tgt_locked, aliases[], gender, pronouns_tgt, speech_style, relations, first_chapter}`。`name_tgt_locked` 一旦锁定只读，代词错即 heuristic fail，成长改称走显式解锁而非机械替换。

### 3.3 repair vs polish 两阶段

| | repair（语义修复） | polish（文学润色，prestige 专属） |
|---|---|---|
| 触发 | FastPass/heuristic fail | repair 后且 QE 过线 |
| 输入 | src + mt + 错误定位 | src + repair 译文 + style_card + 相邻 2 chunk |
| 约束 | 最小 diff | 禁增删情节人名时间线，diff>15% 打回 |
| 模型 | 低温 0.1-0.2 小模型 | 高温 0.5-0.6 大模型，TBL refine prompt 同款 |
| 轮数 | ≤2 | 仅 1 轮 |

实现：`repair()` 后加 `polish()`，`polish_enabled` 由 preset 控制。无 style_card 不得进 polish。

### 3.4 QE 与成本照搬清单

- 评分照搬 TBL v2：四维 + 扣分表 + overall 硬顶 + `feedback≤200 ASCII`，`temp=0.1`。旧裸四维不可用。
- 单 Judge + `judge_id` 版本化，译文不可变评分可重跑。
- 成本：`560/chunk + 语种比 + refine×2`，Judge 约 `$0.007/段`。关 OpenRouter 推理 + 关 Poe search，否则整书成本翻倍且上下文被污染。
- 评测对：`quick 8` 做 A/B，`standard 16` 做发布，`full` 仅放行前。固定 `pair_set + judge_id + rubric` 三元组。

---

## 4. 落地路线（灰度三阶段 + 验收）

- P0 影子 1 周：fiction preset 跑历史 10 本，不上线，对比旧档调阈值。准入 P1：人名一致率达标。
- P1 5% 类型小说 2 周：仅 `fiction_light`，关 polish，监控 cost/latency/QE fail。回滚：fail>论文档 1.5x 或 cost+30%。
- P2 50% + 纯文学内测 2 周：开 `prestige+polish`，人工抽检 5%。准入 GA：人名 100%，代词≥98/96%，漏译 0，TM 污染<2%，流畅/声线/对话≥4.2，polish regression<5%。

实施清单（3-5d）：`presets.py` 加两档 → `prompts_fiction.py + extract_style_card` → 拆 `repair/polish` → Bible 升级 + 代词 heuristic → `evals/fiction/` + TBL scorer 移植。

---

## 附：关键文件索引

- UBT：`ubt/core/presets.py:31,69-118`，`ubt/core/router/prompts.py:134-392,527-748`，`capabilities.py:8-13`，`memory/hierarchical_memory.py:61-311`，`memory/bible.py:8-96`，`memory/tm.py:534-630`，`qe/fast_pass.py:624-1107`，`qe/llm_judge.py:203-346`，`engine/stages/quality_gate.py:118-284`，`engine/repair_loop.py:152-509`，`config.py:421-463`
- TBL：`src/core/style/extractor.py:104-138`，`assembler.py:56-107`，`lint.py:58-143`，`dimensions.py:7-17`，`src/prompts/prompts.py:477-623,626-798`，`src/core/translator.py:602-607,734-966`，`context_optimizer.py:322-329`，`glossary/filter.py:60-123`，`injector.py:45-177`，`benchmark/*`，`docs/JUDGE_RUBRIC_V2.md`
