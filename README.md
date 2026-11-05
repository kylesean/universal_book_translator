# Universal Book Translator (UBT)

[![CI](https://github.com/kylesean/universal_book_translator/actions/workflows/ci.yml/badge.svg)](https://github.com/kylesean/universal_book_translator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Typst: 0.15.1](https://img.shields.io/badge/typst-0.15.1-239dad.svg)](https://typst.app/)

> ⚠️ **开发状态提示**：本项目目前处于 **Pre-1.0 积极迭代开发阶段**。核心流水线与排版引擎已具备出版级质量，但在多格式分章灵活性、依赖轻量化与极端版面自适应上仍有已知缺陷与限制（详见后文[已知局限与问题](#已知局限与缺陷)）。欢迎试用、提出 Issue 或提交 PR 参与建设！

---

## 📖 项目简介

**Universal Book Translator (UBT)** 是一个自托管、编译器架构的工业级图书与学术文档翻译重排引擎。

不同于市面上常见的“调用 LLM API + 简单正则替换”的机翻脚本，UBT 真正采用**编译器思想**构建：
1. **统一文档解析 (Ingestion)**：将 PDF、EPUB、DOCX、Markdown、HTML 等多源文档提取并规范化为高精度的中间表示（Document IR）；
2. **多阶段 LLM 编译流水线 (Translation Ladder)**：基于支持 ACID 特性的 SQLite WAL 账本驱动，结合分层上下文记忆与术语库，进行高质量翻译与自动化质量门禁（MTQE）靶向修复；
3. **出版级排版重构 (Typst Typesetting)**：彻底摆脱传统版面拉伸错位的问题，利用现代排版语言 **Typst** 重新编排，原生重建双语对照、数学公式与结构化表格，输出符合印刷标准的出版级 PDF 及伴生格式。

---

## ✨ 核心优势与特性

- 🎨 **印刷级排版重构**：基于 Typst 编译重排，支持 LaTeX 原生数学公式转换（无需光栅化截图）与自适应表格重建，支持双语并排（`dual`）与纯译文（`mono`）。
- 💾 **ACID 级状态账本与零损耗断点续跑**：底层基于 SQLite WAL 账本维护逐块（Block-level）状态机。任务中途若遭遇断网或中断，直接凭借 `--job-id` 即可恢复，**已翻译分块绝不重复扣费**。
- 🧠 **分层上下文记忆体系**：
  - **L1 局部滑动窗口**：感知紧邻语义流，避免脚注干扰正文；
  - **L2 跨章节滚动宏观摘要**：保持长文主线剧情与学术推导一致性；
  - **L3 宏观 Epoch 摘要**：结合 System Prompt 前缀缓存，兼顾长书记忆与 Token 成本。
- 🛡️ **术语挖掘与质量门禁**：内置 Bible 挖掘阶段自提取全书专有名词；集成 FastPass 启发式初筛与 MQM 严重度分级门禁，自动识别漏译、幻觉并进入二次修复循环。
- 🖥️ **现代化开箱即用 Web 控制台**：前端基于 React 19 + TypeScript + Radix UI 构建，**已预编译打包并随 Python wheel 内嵌分发**。用户无需安装 Node.js/pnpm，一条命令即可启动可视化向导与审校工作台。
- 🤖 **创新的 Agent 协同范式 (MCP)**：
  - **自主流水线模式**：通过标准 MCP 工具调度 UBT 执行长任务；
  - **Agent-Hosted 模式**：外部大模型（如 Claude / GPT-4）**自身充当翻译大脑**，UBT 专职负责文档 AST 解析、样式占位符保护、上下文分包认领与 Typst 排版编译。

---

## 🛠️ 安装指南

当前项目尚未正式推送至公共 PyPI（预发版），推荐通过源码检出或 Docker 容器运行：

### 1. 源码本地安装（推荐使用 `uv`）

要求 **Python >= 3.12**：

```bash
git clone https://github.com/kylesean/universal_book_translator.git
cd universal_book_translator

# 推荐使用现代 Python 包管理器 uv（超快依赖解析）：
uv sync --extra dev                            # 开发与基础运行环境
uv sync --all-extras                           # 安装所有扩展（dev, pdf, ocr, cels, mcp 等）

# 或使用标准 pip 安装：
pip install .                                  # 基础安装（含 Docling 语义解析）
pip install '.[mcp]'                           # 安装 MCP Server 依赖
pip install '.[ocr]'                           # 安装图片扫描件 OCR 依赖
pip install '.[all]'                           # 全功能完整安装
```

> 📌 **外部排版编译器 Typst**：PDF 渲染需要系统 `PATH` 中具备 `typst`（推荐 v0.15.1）。
> - Linux: 可通过 GitHub Releases 下载预编译可执行文件放入 `/usr/local/bin`，并安装 CJK 字体：`sudo apt-get install fonts-noto-cjk`；
> - macOS: `brew install typst`；
> - 运行 `ubt doctor` 可自动检查环境中的 Typst 与中文字体就绪情况。

---

### 2. Docker 容器一键部署（免配置 Typst 与字体）

预构建容器内已集成 **Typst 0.15.1** 与 **Noto CJK 中文字体**：

```bash
# 使用 Docker Compose 启动 Web 控制台与 API 服务
docker compose up -d

# 或直接通过 Docker CLI 运行
docker run -d \
  --name ubt-app \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  -e UBT_PROVIDER=openai-chat \
  -e UBT_LLM_API_KEY=sk-... \
  -e UBT_DRAFT_MODEL=gpt-4o-mini \
  universal-book-translator:latest
```

服务就绪后，直接在浏览器中访问 `http://localhost:8000` 即可使用 Web 控制台。
更多容器化启动与持久化挂载说明参见 [`deploy/docker/README.md`](deploy/docker/README.md)。

---

## 🚀 快速上手

### 1. 配置模型凭证

UBT 采用**单键规范（Single-Key Design）**以确保安全性（不读取 `OPENAI_API_KEY` 等第三方混淆变量），请通过 `UBT_*` 环境变量或 `ubt.toml` 文件配置：

```bash
export UBT_LLM_API_KEY="sk-..."                 # 访问大模型的 API Key
export UBT_BASE_URL="https://api.openai.com/v1" # 接口端点（兼容各种中转网关/自建服务）
export UBT_DRAFT_MODEL="gpt-4o-mini"            # 一级初翻模型
export UBT_REPAIR_MODEL="gpt-4o"                # 二级审校修复模型（可选）
```

> 💡 **提示**：UBT 内建 4 种底层线缆协议（`openai-chat`, `openai-responses`, `anthropic-messages`, `gemini-native`）。如需使用自建端点（如 vLLM、Ollama 或中转服务），可参考 [`ubt.toml.example`](ubt.toml.example) 在当前目录或 `~/.config/ubt/config.toml` 中声明 `[providers.<name>]` 配置块。

> 🎛️ **质量与成本默认值**：`consistency_enforce=report`（零 token 报告全书术语漂移，不自动重译）；`c_text_enabled=true`（翻译公式内 `\text{}`，可用 `UBT_C_TEXT_ENABLED=false` 关闭）；`memory_sliding_window=false`（L3 保留最早 epoch 作为长程锚点，近期上下文由 L2 负责）；TM 精确命中按文档隔离，设置 `UBT_TM_CROSS_BOOK=true` 才会跨书复用机器命中。语义 judge 仍为显式 opt-in（`UBT_QE_JUDGE_ENABLED=true`）；报告会明确区分“未测量”和“已测量”。

---

### 2. CLI 命令行使用

#### ① 翻译书籍
```bash
# 将书籍翻译为中文（-l 目标语言，-s 源语言，默认源语言为 en）
ubt translate sample.pdf -l zh

# 零成本全真演练（使用内置 Mock Echo 引擎，不消耗任何 Token，快速验证排版流水线）
ubt translate sample.pdf -l zh --dry-run

# 仅翻译前 5 页进行抽样验证
ubt translate sample.pdf -l zh --pages 1-5
```

执行开始后终端会打印专属任务 ID：
```text
Job ID: job_da4c06edac50_zh (resume with: ubt translate sample.pdf --job-id job_da4c06edac50_zh)
```
若任务因故中断，再次运行上述带 `--job-id` 的命令即可**原地秒级继续断点翻译**；若想推倒重来，追加 `--fresh` 参数即可。

#### ② 实用工具命令
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

## 🎨 排版与交付模式说明

输出文档由任务的交付模式（Delivery Contract）驱动：
- **`readable`（流式重排，默认推荐）**：依据语义 IR 由 Typst 重新编排版面，具有统一优美的正文字号（10.5pt）、自适应表格及原生数学公式，阅读体验最佳；
- **`faithful`（版面保真）**：基于原 PDF 页面几何坐标进行精准图文擦除回贴，适用于需要严格保留原始几何版式的排版场景。

输出模式通过 `--dual-mode` 控制：
- `auto`：默认模式，尝试生成中英双语对照版；
- `dual`：严格中英双语上下逐段对照；
- `mono`：纯目标语言译文排版。

---

## ⚠️ 已知局限与缺陷 (Known Limitations)

为了秉持开源求实精神，以下为当前实现中客观存在的局限与已知边界，请在实际使用中留意：

1. **输入格式支持度差异**：
   - **Word (DOCX)**：当前版本中整篇 DOCX 文档被视为单一逻辑章节处理，超长 Word 书籍暂无法享受多章节并行流式并发；
   - **Markdown / TXT**：章节划分严格依赖一级标题（`# `），若文档全部采用二级标题（`## `）起头，会被整体作为单章节处理；
   - **PDF 几何模式限制**：若强制使用纯轻量的 `pdfium` 引擎，解析时会跳过文档内的复杂表格、插图与数学公式；
   - **格式转换单向性**：输入何种格式即输出何种格式（PDF 额外附带 HTML/EPUB 伴生格式），暂不支持将 Markdown 或 DOCX 直接转换为 Typst PDF。
2. **依赖体积与外部环境**：
   - 默认安装绑定了深度学习版面引擎 `docling`，初次安装会拉取 PyTorch 等依赖，环境体积较大；
   - 依赖宿主机 `PATH` 中的外部二进制 `typst` 与中文字体，若环境缺失字体可能导致排版回退。
3. **版面与特殊内容暂不支持**：
   - 暂不支持垂直排版（日文/中文传统竖排文本）；
   - 暂不支持带密码加密的 PDF（提取阶段直接拒绝）；
   - 暂不支持深色背景/复杂彩色艺术水印背景的 PDF（可能存在背景反差不足问题）；
   - 暂不支持从栅格位图插图中自动擦除并翻译内嵌文字。

---

## 🤝 参与贡献

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

## 📄 开源许可证

本项目采用 [MIT 许可证](LICENSE) 开源发布。
