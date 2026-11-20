# Universal Book Translator (UBT)

[![CI](https://github.com/kylesean/universal_book_translator/actions/workflows/ci.yml/badge.svg)](https://github.com/kylesean/universal_book_translator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Typst: 0.15.1](https://img.shields.io/badge/typst-0.15.1-239dad.svg)](https://typst.app/)

> **开发状态提示**：本项目目前处于 **Pre-1.0 积极迭代开发阶段**。核心流水线与排版引擎已具备出版级质量，但在多格式分章灵活性、依赖轻量化与极端版面自适应上仍有已知缺陷与限制（详见后文[已知局限与问题](#已知局限与缺陷-known-limitations)）。欢迎试用、提出 Issue 或提交 PR 参与建设！

---

## 项目简介

**Universal Book Translator (UBT)** 是一个自托管、编译器架构的工业级图书与学术文档翻译重排引擎。

不同于市面上常见的“调用 LLM API + 简单正则替换”的机翻脚本，UBT 真正采用**编译器思想**构建：
1. **统一文档解析 (Ingestion)**：将 PDF、EPUB、DOCX、Markdown、HTML 等多源文档提取并规范化为高精度的中间表示（Document IR）；
2. **多阶段 LLM 编译流水线 (Translation Ladder)**：基于支持 ACID 特性的 SQLite WAL 账本驱动，结合分层上下文记忆与术语库，进行高质量翻译与自动化质量门禁（MTQE）靶向修复；
3. **出版级排版与交付 (Typesetting & Delivery)**：`auto` 按交付契约选引擎：**faithful** 走原位内容流引擎，逐行擦除原文并在源坐标写回译文，完整保留矢量图/公式/表格；**readable** 由现代排版语言 **Typst** 重建双语对照与结构化版面。两条路径都输出可检索、可复制的出版级 PDF 及 HTML/EPUB 伴生格式。

---

## 核心优势与特性

- **双交付契约 + 三代 PDF 引擎**：
  - **原生原位引擎（faithful / born-digital PDF）**：直接擦除源页文字指令，把译文按实测字宽、字形与避头尾规则写回原坐标系；矢量图、公式、表格与页码位置原样保留，输出仍是可检索、可复制的文字层；
  - **`readable`（Typst 重排）**：依据语义 IR 由 Typst 重新编排，统一正文字号、自适应表格、原生数学公式；`auto` 仅在扫描件、字体损坏或非 PDF 输入上选择 `docling`，干净的原生 PDF 默认留在原位引擎；
  - **`faithful`（版面保真）**：原位引擎天然满足页数、分页与图/公式/表位置不变；`faithful` 请求在不可行时会在消耗模型调用前拒绝，`--allow-degrade` 可改走 `readable`；
  - 路由不是拍脑袋：PDF 结构普查（分栏、公式、扫描、字体损伤）与页面级探针共同决定契约；探针测不出来时如实标注 `eligibility.provisional`，而不是当成通过。
- **ACID 级状态账本与零损耗断点续跑**：底层基于 SQLite WAL 账本维护逐块（Block-level）状态机。任务中途若遭遇断网或中断，直接凭借 `--job-id` 即可恢复，**已翻译分块绝不重复扣费**。
- **分层上下文记忆体系**：
  - **L1 局部滑动窗口**：感知紧邻语义流，避免脚注干扰正文；
  - **L2 跨章节滚动宏观摘要**：保持长文主线剧情与学术推导一致性；
  - **L3 宏观 Epoch 摘要**：结合 System Prompt 前缀缓存，兼顾长书记忆与 Token 成本。
- **术语挖掘与质量门禁**：内置 Bible 挖掘阶段自提取全书专有名词；集成 FastPass 启发式初筛与 MQM 严重度分级门禁，自动识别漏译、幻觉并进入二次修复循环。
- **现代化开箱即用 Web 控制台**：前端基于 React 19 + TypeScript + Radix UI 构建，**已预编译打包并随 Python wheel 内嵌分发**。用户无需安装 Node.js/pnpm，一条命令即可启动可视化向导与审校工作台。
- **创新的 Agent 协同范式 (MCP)**：
  - **自主流水线模式**：通过标准 MCP 工具调度 UBT 执行长任务；
  - **Agent-Hosted 模式**：外部大模型（如 Claude / GPT-4）**自身充当翻译大脑**，UBT 专职负责文档 AST 解析、样式占位符保护、上下文分包认领与 Typst 排版编译。

---

## 安装指南

当前项目尚未正式推送至公共 PyPI（预发版），推荐通过源码检出或 Docker 容器运行：

### 1. 源码本地安装（推荐使用 `uv`）

要求 **Python >= 3.12**：

```bash
git clone https://github.com/kylesean/universal_book_translator.git
cd universal_book_translator

# 推荐使用现代 Python 包管理器 uv（超快依赖解析）：
uv sync --extra dev                            # 开发与基础运行环境
uv sync --all-extras                           # 安装所有扩展（dev, pdf, ocr, academic, mcp 等）

# 或使用标准 pip 安装：
pip install .                                  # 基础安装（含 Docling 语义解析）
pip install '.[mcp]'                           # 安装 MCP Server 依赖
pip install '.[ocr]'                           # 安装图片扫描件 OCR 依赖
pip install '.[all]'                           # 全功能完整安装
```

> **外部排版编译器 Typst**：PDF 渲染需要系统 `PATH` 中具备 `typst`（推荐 v0.15.1）。
> - Linux: 可通过 GitHub Releases 下载预编译可执行文件放入 `/usr/local/bin`，并安装 CJK 字体：`sudo apt-get install fonts-noto-cjk`；
> - macOS: `brew install typst`；
> - 运行 `ubt doctor` 可自动检查环境中的 Typst 与中文字体就绪情况。

---

### 2. Docker 容器一键部署（免配置 Typst 与字体）

预构建容器内已集成 **Typst 0.15.1** 与 **Noto CJK 中文字体**：

```bash
# 控制台默认 fail-closed：未设置 UBT_API_KEY 会拒绝启动（它能读取 /data 下的
# 任何文档、并消耗你的 LLM 密钥）。先生成一个访问令牌：
echo "UBT_API_KEY=$(openssl rand -hex 24)" > .env

# 使用 Docker Compose 启动 Web 控制台与 API 服务（宿主端口默认只绑 127.0.0.1）
docker compose up -d

# 或直接通过 Docker CLI 运行
docker run -d \
  --name ubt-app \
  -p 127.0.0.1:8000:8000 \
  -v $(pwd)/data:/data \
  -e UBT_API_KEY=<你生成的令牌> \
  -e UBT_PROVIDER=openai-chat \
  -e UBT_LLM_API_KEY=sk-... \
  -e UBT_DRAFT_MODEL=gpt-4o-mini \
  universal-book-translator:latest
```

> 只有在你确认网络环境可信、且愿意承担风险时，才把 `-p 127.0.0.1:8000:8000` 改为 `-p 8000:8000` 暴露到局域网；如需无鉴权的本地实例，显式设置 `UBT_ALLOW_NO_AUTH=1`（务必保留回环绑定）。

服务就绪后，直接在浏览器中访问 `http://localhost:8000` 即可使用 Web 控制台。
更多容器化启动与持久化挂载说明参见 [`deploy/docker/README.md`](deploy/docker/README.md)。

### 3. 部署姿势与信任模型

安全默认值随部署方式变化，共三种姿势（下表为代码强制行为，非使用建议）：

| 姿势 | 进入方式 | 网页填写 LLM Key | 网页编辑 Provider | 启动 / 访问门槛 |
|------|----------|------------------|-------------------|------------------|
| **本机**（个人默认） | `ubt console`，绑定 `127.0.0.1` |  可直接填写 |  完全可编辑（含已存钥匙的块） | 免鉴权：`ubt console` 自动带上 `UBT_ALLOW_NO_AUTH`，无需任何令牌 |
| **局域网 / 暴露** | 绑定 `0.0.0.0`、Docker 映射宿主端口 |  拒绝，请写入 `ubt.toml` 或 `UBT_LLM_API_KEY` |  可新建/编辑**无钥匙**的 provider 块； 已存钥匙的块不可跨机改写端点 | 必须配好 `UBT_API_KEY` 与 `UBT_ALLOWED_DIRS`（缺任一，非回环请求一律 403），并用 `UBT_TRUSTED_HOSTS` 登记对外主机名 |
| **严格** | `UBT_STRICT_AUTH=1` 或 `UBT_ENV=production` |  即使本机也拒绝 | 同暴露模式 | 覆盖 `UBT_ALLOW_NO_AUTH`，必须设置 `UBT_API_KEY`；开启鉴权后 `/docs` 与 `/openapi.json` 一并关闭 |

**反向代理下的客户端判定**：控制台把"请求来自本机"作为放开凭据写入的条件，而 `request.client.host` 在反向代理后面永远显示为回环。因此经由代理暴露时，须设置 `UBT_TRUST_PROXY=1`，让判定改用 `X-Forwarded-For` 链；不设置时，带代理头的请求一律按非本机处理（拒绝改写已存钥匙的 provider 端点）。同时建议反向代理只对可信来源设置转发头，因为链路本身可被伪造。

**信任模型**：持有 `UBT_API_KEY` 的人等同于**对 LLM 花费完全信任的操作员**——他能提交消耗真实 token 的长任务，也能新建指向任意端点的 provider 块（环境变量中的 `UBT_LLM_API_KEY` 会流向他所指定的端点）。控制台门禁防的是家庭网络里的**意外**：访客设备、其他机器上的恶意软件、误暴露的端口；它不防有门禁卡的预谋内部攻击。若需把控制台交给不完全信任的人，请让改动走服务器上的配置文件，而不是发放门禁卡。

---

## 快速上手

### 1. 配置模型凭证

UBT 采用**单键规范（Single-Key Design）**以确保安全性（不读取 `OPENAI_API_KEY` 等第三方混淆变量），请通过 `UBT_*` 环境变量或 `ubt.toml` 文件配置：

```bash
export UBT_LLM_API_KEY="sk-..."                 # 访问大模型的 API Key
export UBT_BASE_URL="https://api.openai.com/v1" # 接口端点（兼容各种中转网关/自建服务）
export UBT_DRAFT_MODEL="gpt-4o-mini"            # 一级初翻模型
export UBT_REPAIR_MODEL="gpt-4o"                # 二级审校修复模型（可选）
```

> **提示**：UBT 内建 4 种底层线缆协议（`openai-chat`, `openai-responses`, `anthropic-messages`, `gemini-native`）。如需使用自建端点（如 vLLM、Ollama 或中转服务），可参考 [`ubt.toml.example`](ubt.toml.example) 在当前目录或 `~/.config/ubt/config.toml` 中声明 `[providers.<name>]` 配置块。

> **质量与成本默认值**：`consistency_enforce=report`（零 token 报告全书术语漂移，不自动重译）；`c_text_enabled=true`（翻译公式内 `\text{}`，可用 `UBT_C_TEXT_ENABLED=false` 关闭）；`memory_sliding_window=false`（L3 保留最早 epoch 作为长程锚点，近期上下文由 L2 负责）；TM 精确命中按文档隔离，设置 `UBT_TM_CROSS_BOOK=true` 才会跨书复用机器命中。语义 judge 仍为显式 opt-in（`UBT_QE_JUDGE_ENABLED=true`）；报告会明确区分“未测量”和“已测量”。

---

### 2. CLI 命令行使用

#### 翻译书籍
```bash
# 将书籍翻译为中文（-l 目标语言，-s 源语言，默认源语言为 en）
ubt translate sample.pdf -l zh

# 零成本全真演练（使用内置 Mock Echo 引擎，不消耗任何 Token，快速验证排版流水线）
ubt translate sample.pdf -l zh --dry-run

# 仅翻译前 5 页进行抽样验证
ubt translate sample.pdf -l zh --pages 1-5

# 默认会拒绝交付仍待人工复核（NEEDS_HUMAN）的块；确认风险后显式放行
ubt translate sample.pdf -l zh --allow-needs-human
```

执行开始后终端会打印专属任务 ID：
```text
Job ID: job_da4c06edac50_zh (resume with: ubt translate sample.pdf --job-id job_da4c06edac50_zh)
```
若任务因故中断，再次运行上述带 `--job-id` 的命令即可**原地秒级继续断点翻译**；若想推倒重来，追加 `--fresh` 参数即可。

#### 实用工具命令
```bash
# 零 Token 计费全书预检（预估所需 Token 费用、耗时、公式密度与版面风险）
ubt assess book.pdf

# 查看已提交任务的进度与状态统计
ubt status job_da4c06edac50_zh

# 环境健康体检（检查 Typst 编译器、系统字体、账本路径、模型连通性）
ubt doctor

# 离线验证内置排版样本（无需任何 API Key，验证 Typst 编译链路）
uv run python scripts/verify_sample_book.py
```

---

### 3. 可视化 Web 控制台 (Operator Console)

无需记忆繁杂的 CLI 参数，运行控制台命令即可唤起沉浸式交互界面：

```bash
ubt console
```
命令会自动打开浏览器访问 `http://127.0.0.1:8000/`：
- **向导式新建任务 (`/wizard`)**：拖拽上传文档，右侧实时显示基于 `/jobs/assess` 的全书预检与预算核算；
- **全景观控中心 (`/jobs/{id}`)**：实时 SSE 数据流展示当前翻译阶梯（Extract → Segment → TM → Translate → QE → Repair → Render）；
- **L3 人工审校工作台 (`/jobs/{id}/review`)**：双语对齐列表沉浸式校对，集成 KaTeX 实时预览公式。在某一处修改译文后，支持**全书术语一键传播（Propagate）**，并可直接触发**无损重新排版导出（Re-export，不消耗任何模型 Token）**；
- **系统诊断与 Provider 可视化编辑 (`/system`)**：免终端直观配置各类大模型提供商端点与密钥。

---

### 4. AI Agent 集成 (MCP Server)

UBT 提供了标准的 stdio 模式 Model Context Protocol (MCP) 服务，可无缝接入 Claude Desktop、Cursor 等支持 MCP 的 AI 开发工具中：

```bash
pip install 'universal-book-translator[mcp]'
ubt-mcp
```

在客户端工具中配置对应的 `ubt-mcp` 可执行路径后，即可直接赋予智能体以下工具能力：
- `ubt_translate_book` / `ubt_job_status` / `ubt_cancel_job`：长任务自动托管；
- `ubt_assess_book` / `ubt_inspect_book`：全书目录与预检感知；
- `ubt_agent_prepare` / `ubt_agent_claim` / `ubt_agent_submit` / `ubt_agent_finish`：Agent-Hosted 交互模式。

---

## 排版与交付模式说明

输出文档由任务的交付模式（Delivery Contract）驱动，`auto` 的裁决规则是：

- **原生原位（faithful）**：任何文本层可信的原生 PDF（论文、教材、报告、长书）默认走 `native` 内容流引擎：逐行擦除原文、按源 BBox 与实测字宽排入译文，页数、分页、图/公式/表位置保持不变。`auto` 同时带有 `allow_degrade`，个别译文实在放不进源框时才降级为 readable。
- **`readable`（流式重排）**：扫描件、字体编码损坏、以及 EPUB/DOCX/MD 等非 PDF 输入走 Typst 重排；`auto` 为这类输入选择 `--pdf-engine docling`。显式指定 `readable` 时也走这条路径。
- **`faithful`（版面保真）**：显式请求原位 1:1 交付。没有可信文本层（扫描/字体损坏）的文档会在预检阶段直接拒绝（`--allow-degrade` 可降级到 `readable`），而不是先花完整本书的模型费用再失败。

**它承诺什么，不承诺什么**：原位/`faithful` 承诺页数不变、分页不变、图/公式/表位置不变、文字锚定在原位置；它**不**承诺中文字形与英文逐像素一致——字形本身就不同。译文在源框内仍可能溢出：溢出单元会如实记录（`page_overflow`）并进入有界压缩/复核，而不是悄悄压小字号或裁掉半句。

**实测与未测的区分**：交付计划里的 `eligibility.provisional` 只在探针未能测量该文档时出现（扫描件、加密、无文字层、解析失败）。看到 provisional 就表示"没量过"，不是"量过没问题"。

输出模式通过 `--dual-mode` 控制（默认 `mono`）：
- `auto`：由版面顾问按文档结构决定，结构化/图表密集的文档倾向 `mono`；
- `dual`：严格中英双语上下逐段对照；
- `mono`：纯目标语言译文排版。

### 交付门禁（fail-closed）

- `NEEDS_HUMAN`（已分诊、带提示标记、建议人工复核）的块**默认随文交付**，但质量报告的 `summary.needs_human_block_ids` 会逐一列出它们，审校工作台也会高亮。要求严格阻断时用 `--no-allow-needs-human`（`UBT_EXPORT_ALLOW_NEEDS_HUMAN=0`）；不可修复的 critical 缺陷仍以 `BLOCKED_HUMAN` 隔离，不会因这个默认值放行。
- `--min-completion-ratio`（默认 0.5）与产物门禁的致命项（不可读 PDF、空白页、文本不可搜索、目标语言缺失）同样会在导出前拒绝。
- 质量报告的 `score_metrics.scored_coverage` 是 `avg_qe` 的计算样本占比：默认启发式引擎只对通过块抽样打分，看到 0.14 就意味着那个均值只代表 14% 的块。

### 产物门禁（确定性）

PDF 交付不再有像素/VLM/阈值门禁。导出前只做确定性检查：

- 文件能打开、页数正确；
- 有内容的页面能提取文本（`/ToUnicode` 生效）；
- 源页非空时目标页不为空；
- `faithful` 交付保留源页数；
- 不出现 U+FFFD、目标语言缺失。

`blank_page`、`unreadable_pdf`、`text_not_searchable` 等不可读项 fail-closed；其余 finding 只记录、不阻断。`[gate]` 配置段已删除。

---

## 已知局限与缺陷 (Known Limitations)

为了秉持开源求实精神，以下为当前实现中客观存在的局限与已知边界，请在实际使用中留意：

1. **输入格式支持度差异**：
   - **Word (DOCX)**：当前版本中整篇 DOCX 文档被视为单一逻辑章节处理，超长 Word 书籍暂无法享受多章节并行流式并发；
   - **Markdown / TXT**：章节划分严格依赖一级标题（`# `），若文档全部采用二级标题（`## `）起头，会被整体作为单章节处理；
   - **PDF 引擎边界**：`auto` 对文本层可信的 PDF 默认走原位 1:1，扫描件/损坏文本层走 docling 重排；显式指定 `docling` 时走 reflow 契约。轻量几何 reader 已删除，PDF 只有 native 与 scan 两条路径；
   - **格式转换单向性**：输入何种格式即输出何种格式（PDF 额外附带 HTML/EPUB 伴生格式），暂不支持将 Markdown 或 DOCX 直接转换为 Typst PDF。
2. **依赖体积与外部环境**：
   - 默认安装同时带有 Docling 版面引擎、PP-DocLayoutV2 ONNX 版面检测（`rapid-layout` + `onnxruntime`）与 HarfBuzz 字形度量，初次安装体积较大；
   - 原生原位引擎需要目标语言字体；仓库/镜像自带 Noto CJK，源码安装若系统缺少 CJK 字体，中文译文可能缺字（`ubt doctor` 会检查）；
   - Typst 重排仍依赖宿主机 `PATH` 中的外部二进制 `typst`。
3. **版面与特殊内容暂不支持**：
   - 暂不支持垂直排版（日文/中文传统竖排文本）；
   - 暂不支持带密码加密的 PDF（提取阶段直接拒绝）；
   - 暂不支持深色背景/复杂彩色艺术水印背景的 PDF（可能存在背景反差不足问题）；
   - 暂不支持从栅格位图插图中自动擦除并翻译内嵌文字。
4. **质量门禁的覆盖边界（如实说明）**：
   - 产物门禁检查的是**可读性与可搜索性**，不做像素级遮挡、字号或版面美观判定；
   - 字体覆盖检查（缺字方块的确定性检测）在无法解析的字体格式上会**记为未测**，不会伪装成通过；
   - 位图插图内的文字、纸张纹理、印刷套印偏差不在门禁的判定范围内；
   - 交付契约（`*_delivery_contract.json`）负责文本覆盖与版面几何；两者相互独立，任何一项 critical 失败都会阻断交付。

---

## 参与贡献

项目文档以根目录的 [`GLOSSARY.md`](GLOSSARY.md) 为准。
欢迎查阅：
- [术语与领域词典 (`GLOSSARY.md`)](GLOSSARY.md)
- [贡献者指南 (`CONTRIBUTING.md`)](CONTRIBUTING.md)

在提交 Pull Request 前，请确保本地通过核心检查：
```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict ubt tests
uv run python scripts/dump_compiler_edges.py
uv run pytest -m fast -q
```

---

## 开源许可证

本项目采用 [MIT 许可证](LICENSE) 开源发布。
