# 目标语言文案技术债（I18N String Debt）

- **状态**：Open（部分收敛）
- **记录日期**：2026-11-01
- **关联提交**：`256300d fix(facsimile): keep furniture, preserve corrupt tables and localize labels`
- **适用范围**：`zh` / `zh-Hant` / `ja` / `ko` 优先，未来扩展其他目标语言

## 1. 结论与原则

**不是最佳实践：用户可见的目标语言文案不能散落在 renderer、adapter 或 export 代码里。**

需要区分三类中文：

1. **目标语言输出文案**（图注前缀、表注前缀、附录标题、续表标记、CIP/版权页等）
   - 必须按 `target_lang` 从语言资源取。
   - 未知目标语言应回退英文，不能回退中文。
2. **源语言识别规则**（`图/表/図/표/그림/コードリスト` 等 caption 标记）
   - 属于输入识别，不是目标输出；可以包含多语言字符，但建议集中为 source-marker 资源。
3. **注释、测试 fixture、术语表、QE 词表**
   - 不影响交付产物，不属于同一类问题；不要为了“去中文”而误删测试数据或词表。

判定标准：

> 同一段代码在 `target_lang=ja` 或 `ko` 时，是否会写出中文？
> 会，就是技术债；不会，则只是注释/数据/识别规则。

## 2. 本次已收敛（2026-11-01）

- 删除 `ubt/adapters/pdf/docling_render.py` 里的 `_CAPTION_LABELS` 中文字面量。
- `FontConfig` 新增 `listing_prefix`，caption 标签统一从 `resolve_font_config(target_lang)` 取：
  - `zh`：`代码清单`
  - `zh-Hant`：`程式碼清單`
  - `ja`：`コードリスト`
  - `ko`：`코드 목록`
  - 其他语言：各自 profile 或默认 `Listing`
- 新增 `ubt/core/target_labels.py`，集中管理表格附录、溢出附录的标题/计数/说明/页码模板/表头：
  - 已覆盖 `zh` / `ja` / `ko` / `en`；
  - 未知目标语言回退 `en`，不回退 `zh`。
- `export.py` 的 `_write_table_appendix` 与 `_write_spill_appendix` 不再内联中文标题。
- caption 源标记正则补充 `圖/図/표/그림/コードリスト/코드 목록`；这是输入识别，不是输出文案。

## 3. 未完成技术债

### TD-I18N-01：表格续表标记只有 zh/en

**位置**

- `ubt/core/tables/models.py:14-16`
- `ubt/core/tables/models.py:22-32`
- 使用方：
  - `ubt/render/table_compiler.py`
  - `ubt/render/typst/document_renderer.py`
  - `ubt/cli/commands/table_cmd.py`
  - `ubt/core/tables/styler.py`

**现状**

`TableContinuationConfig` 只有：

```text
continued_header_text_zh = "（续表）"
continued_header_text_en = "(Continued from previous page)"
next_page_footer_text_zh = "（续下页）"
next_page_footer_text_en = "(Continued on next page)"
```

`get_continued_header_text()` / `get_next_page_footer_text()` 的规则是：

```python
if str(lang).lower().startswith("zh"):
    return self.continued_header_text_zh
return self.continued_header_text_en
```

因此 `ja`、`ko` 目标会得到英文 `(Continued on next page)`，与全文目标语言不一致。这属于用户可见的目标语言文案，不是源语言识别。

**处理方案**

1. 将续表标记迁入统一语言资源：
   - 推荐 `ubt/core/target_labels.py` 新增 `continuation_header` / `continuation_footer`；
   - 或扩展 `FontConfig`，与 `figure_prefix` / `table_prefix` / `listing_prefix` 同源。
2. 补齐 `ja` / `ko` / `zh-Hant`：
   - `ja`：`（前ページから続く）` / `（次ページに続く）`
   - `ko`：`(앞 페이지에서 계속)` / `(다음 페이지로 계속)`
   - `zh-Hant`：`（續表）` / `（續下頁）`
3. 保留 `TableContinuationConfig` 的 `_zh` / `_en` 字段为兼容层，但 `get_*` 改为委托语言资源，避免旧调用方一次性全改。
4. 测试矩阵：`zh`、`zh-Hant`、`ja`、`ko`、`en`、未知语言（如 `es` 回退 `en`）。

**验收标准**

- `target_lang=ja` 的续表与续下页标记不含英文、不含中文；
- `target_lang=ko` 同理；
- `target_lang=en` 保持英文；
- 未知语言回退英文；
- `pytest -m fast` 全绿。

**优先级**：P1（直接影响表格分页可读性）

---

### TD-I18N-02：Colophon/CIP 默认值是中文出版语境

**位置**

- `ubt/core/colophon/models.py:11-36`
- `ubt/core/colophon/models.py:44-67`
- `ubt/core/colophon/generator.py:65-72`
- `ubt/core/colophon/generator.py:93-125`
- `ubt/cli/commands/colophon_cmd.py`

**现状**

`ColophonMetadata` 的默认值直接是中文出版信息：

```text
title = "未命名图书"
publisher = "开源翻译出版工坊"
edition = "第 1 版"
impression = "第 1 次印刷"
subject_headings = ["程序设计", "人工智能", "自然语言处理"]
format_size = "16开 (185mm × 260mm)"
legal_disclaimer = "版权所有 · 侵权必究。印装差错，负责调换。"
printer = "中华商务联合印刷有限公司"
price = "89.00 元"
```

`render_colophon_typst()` 还硬编码了：

- `图书在版编目（CIP）数据`
- `中国国家版本馆 CIP 数据核字（2026）第 089123 号`
- `书　　名` 等中文表单标签
- `千字` / `字` 的中文计量

这不是“漏了一个语言分支”，而是**中国出版语境专属功能**。如果对 `ja` / `ko` / `en` 交付也默认套用，会生成错误的版权页、错误的 CIP、错误的 ISBN/价格/开本语义。

**处理方案**

1. 明确边界：
   - CIP/Copyright page 是 **`zh` 市场 profile**，不是通用 colophon；
   - 通用 colophon 只保留 title/authors/translators/publisher/date/ISBN 等跨语言字段。
2. 拆模型：
   - `ColophonMetadata`：跨语言基础字段，默认值清空或 `None`；
   - `ZhCipMetadata`（或 `ZhColophonProfile`）：CIP、CLC、开本、印张、价格、印刷厂、免责声明等；
   - `render_colophon_typst(meta, profile)`：按 profile 选择模板。
3. 行为约束：
   - `render_colophon_typst` 只允许在 `target_lang` 为 `zh`（或显式 `zh` profile）时调用；
   - `ja` / `ko` / `en` 不生成 CIP 卡片；没有真实数据时不要填占位 CIP 号、价格、印刷厂；
   - CLI `ubt colophon generate` 增加 `--language` 或从 job 的 `target_lang` 读取，不能默认中文。
4. 兼容：
   - 保留 `ColophonMetadata` 的旧字段名一个版本，内部委托 `ZhCipMetadata`；
   - 旧 CLI 调用继续输出中文，但必须显式 `--language zh` 或检测到 zh job。

**验收标准**

- 构造通用 `ColophonMetadata()` 不再自动产生中文出版社/价格/开本；
- `target_lang=ja/ko/en` 不输出 CIP 卡片和中文计量；
- `target_lang=zh` 输出与当前中文版权页一致；
- 单元测试覆盖 `zh` / `ja` / `ko` / `en` 四条路径。

**优先级**：P2（独立 `colophon` 功能，不影响主翻译链路，但一旦使用会输出错误出版信息）

---

### TD-I18N-03：缺少“禁止新增目标语言文字量”的守卫

**位置**

- 目标范围：
  - `ubt/render/**`
  - `ubt/adapters/pdf/**`
  - `ubt/core/engine/stages/export.py`
  - `ubt/core/tables/**`
- 现有测试没有拦截新增中文字面量。

**风险**

本次 `_CAPTION_LABELS` 就是绕过语言资源新增的；如果没有守卫，下一次仍可能在图注、附录、表头、导出提示中继续写死中文。

**处理方案**

1. 新增静态检查脚本，例如 `scripts/check_target_language_literals.py`：
   - 用 AST 扫描目标目录的 **字符串字面量**（不扫描注释/docstring）；
   - 命中 CJK 字符即报错；
   - 白名单只允许：
     - `source_marker` / 源语言识别正则；
     - 语言资源模块（`ubt/core/target_labels.py`、`ubt/core/language_profile.py` 等）；
     - 测试 fixture；
     - 术语表/QE 数据。
2. 或者在 `tests/unit/` 加一个 AST 测试：
   - 失败信息给出文件、行号、字面量；
   - 基线 allowlist 文件可审计、只减不增。
3. CI 接入：
   - pre-commit 与 `pytest -m fast` 同时跑；
   - 新违规直接失败，不允许“先合并后补”。

**验收标准**

- 在 `ubt/render/` 新增一个 `"图 1 | ..."` 字面量会让检查失败；
- 语言资源模块中的多语言文案不误报；
- 注释、测试 fixture、术语表不误报；
- CI 与本地 pre-commit 结果一致。

**优先级**：P1（防止债务继续扩散，成本最低）

## 4. 非目标（Non-goals）

以下内容不属于本技术债，不要为了“去中文”而修改：

- 代码注释、docstring、测试说明中的中文；
- `tests/` 里的中文 fixture；
- `ubt/core/memory/seed_glossary.py`、QE 词表、术语校验数据中的中文；
- 源语言 caption/TOC 识别正则中的 `图/表/図/표/그림`；
- CLI/Console 的 UI 国际化（它有自己的 UI 语言开关，和 `target_lang` 是两条轴）。

## 5. 相关文件索引

- 已资源化：
  - `ubt/core/language_profile.py`（`FontConfig.figure_prefix` / `table_prefix` / `listing_prefix`）
  - `ubt/core/target_labels.py`（附录标题/计数/说明/表头）
  - `ubt/adapters/pdf/docling_render.py`（`_normalise_caption_label` 按 `target_lang` 解析）
  - `ubt/core/engine/stages/export.py`（附录写入使用 `appendix_labels`）
- 待迁移：
  - `ubt/core/tables/models.py`
  - `ubt/core/colophon/models.py`
  - `ubt/core/colophon/generator.py`
- 守卫：
  - 待新增 `scripts/check_target_language_literals.py` 或 `tests/unit/test_target_language_literals.py`

## 6. 验证命令

```bash
uv run pytest -m fast -q
uv run ruff check ubt tests
uv run mypy --strict ubt tests
```

本次关联提交已通过：

- `ruff check --fix` / `ruff format`
- `mypy --strict`
- `pytest -m fast`（5879 passed, 8 skipped, 50 deselected）
- core→compiler layering
- console bundle matches source
