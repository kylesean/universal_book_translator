# PDFium 线程安全缺陷：诊断、修复与演进路线

日期：2026-09-20 · **2026-09-22 复核**：`anchored`→`rigid` 更名已跟进；收口清单补
`render_fidelity`；本闸口的辖域是 **pypdfium2**——`pdf_oxide`（Rust/PyO3 同样封装
pdfium）走的是另一条原生路径，线程策略不同（`&mut self` borrow → open-per-call，
不依赖 `PDFIUM_LOCK`），见 `ubt/adapters/pdf/oxide_render.py` 模块注释。
**2026-09-26 复核**：`docling-parse` 并非 pypdfium2 调用方——其扩展模块
`readelf -d` 显示静态链接自带的 pdfium、无 `libpdfium.so` 依赖，故不在此锁辖域内；
而 **docling（Python 包）自身**的 pypdfium2 路径（栅格/大纲）原用另一把
`docling.utils.locks.pypdfium2_lock`，已通过
`pdfium_gate.unify_docling_pdfium_lock()` 重绑为 `PDFIUM_LOCK`（见 §3）。
状态：**已实施双层防御**（PdfiumGateway 串行闸口 + 钉扎替换字体集）；
sidecar 进程隔离为第三层，见文末
相关崩溃：`SIGSEGV in __tree_balance_after_insert (libpdfium.so)`、
glibc `double free or corruption (!prev)` abort

## 1. 现象

人工测试（本地 llama-server 翻译 arXiv 论文 `2609.20519v1.pdf`）时，
`ubt translate` 两次运行均在 **MTQE 质检完成后 1 秒内**（进入导出/资产抽取
阶段的瞬间）原生崩溃，无任何 Python 异常栈，仅一行
`double free or corruption (!prev)`。两次崩溃均留下完整 core dump
（`coredumpctl`：PID 1234948 @16:25、1245683 @16:27，均为 python3.12 SIGSEGV）。

## 2. 根因（来自 core dump 的实证）

崩掉线程的栈（符号完整）：

```
FPDF_LoadPage → CPDF_Font::GetStockFont → LoadSubstFont
  → CFX_FolderFontInfo::EnumFontList → ScanPath → ReportFace
  → std::map __tree_balance_after_insert          ← SIGSEGV
```

同一 core 中**另一线程正在 libpdfium 内解析同一 PDF 的页面内容**
（`CPDF_StreamContentParser::Parse`，两层 XObject 嵌套），两线程皆为
Python `thread_run` 起的工作线程（ctypes → `FPDF_LoadPage`）。

机制：pdfium 的字体替换表（`CFX_FolderFontInfo`）是**进程级惰性单例**，
首次遇到未内嵌的标准 Type1 字体时扫描系统字体目录并写入全局 std::map；
该初始化与页面内容解析都不带内部锁。UBT 通过十几处
`asyncio.to_thread`/`run_in_executor` 把 pdfium 工作扔进共享线程池，
两个线程并发进入 libpdfium 即构成数据竞争，撞坏原生堆。

**为什么以前没暴露**（三个条件第一次同时凑齐）：

1. 测试文档使用了未内嵌的标准字体（LaTeX 常见），触发字体替换路径；
   字体全内嵌的文档永远不会走到 `EnumFontList`。
2. 本机为桌面环境，系统字体目录大，字体扫描耗时数百毫秒，竞争窗口
   从接近零放大到可命中。
3. 崩溃点在 MTQE 之后——导出阶段 asset 抽取 / witness / 预检等多个
   pdfium 任务密集 fan-out，此前测试多止步于 draft 或走单线程路径。

因此这是一个**依赖文档特征 + 机器环境 + 调度时序**的概率性 bug，
不是新引入的回归；CI 用内嵌字体的合成 PDF、单线程居多，测不到它。

## 3. 已实施：双层防御

`ubt/adapters/pdf/pdfium_gate.py` 是唯一的 pdfium 闸口（PdfiumGateway）。
新代码的规范入口是 `open_document(path)` 上下文管理器；存量入口按下面
两种模式收口。

**第一层 · 进程内串行闸口。** 进程级
`threading.RLock（PDFIUM_LOCK）` + `@pdfium_serialized` 装饰器。
所有进入 libpdfium 的代码路径持锁执行（RLock 允许同线程嵌套，
如 `rigid.extract.extract_pages → textgeom.extract_lines`）。
懒初始化的字体扫描在锁内完成，因此天然只热一次、无竞争，
不需要单独的预热步骤。

收口方式（入口**数量随重构漂移，不在此钉数**——清单以
`grep -rn "PdfDocument(" ubt/adapters/` 与 `tests/unit/test_pdfium_gate.py` 的
AST 守护为准；2026-09-22 实测 pypdfium2 实例化 16 处）：

- **整函数装饰**（函数体是纯 pdfium/CPU 工作）：
  `pdfium_adapter._extract_with_pdfium`、`short_doc.probe_pdf_pages`、
  `page_profiler._pdfium_facts`、`rigid.extract.extract_pages`、
  `textgeom.extract_lines`、`asset_extractor.extract_pdf_figures`、
  `formula_tags._page_text_boxes`
  （位于 `@lru_cache` 内侧，缓存命中不占锁）、`visual_scalpel.crop_block_pil`、
  `extraction_witness.inspect_pdf`、`engine_selector.inspect_pdf_route_plan`
- **只锁 pdfium 区段**（函数还含网络/子进程调用，**严禁持锁等待外部服务**）：
  `vlm.transcribe.transcribe_page_to_blocks`（VLM `driver.recognize` 在锁外）、
  `docling_parser` VLM fallback 的 probe 段、`formula_witness` 栅格化段
  （typst 子进程在锁外）、`render_fidelity`（源页与产物页两束光栅在
  `with PDFIUM_LOCK:` 内完成，2026-09-22 补录——首版清单未含此新增点）、
  页面采样链 `core.advisor` → `ubt/core/archetype.py::sample_document` →
  `ubt/core/ports.py::sample_pdf_pages` → `plain_text_extractor`（经
  `open_document` 持锁；2026-09-20 起逐层下沉，锁语义不变）。

**docling 的两条 pdfium 路径（2026-09-26 补充）**：`docling-parse`（文本/坐标抽取）
把 pdfium **静态链进自己的扩展模块**，与 pypdfium2 的 `libpdfium.so` 是两份独立
原生库、独立进程状态，因此**不需要**本锁；docling（Python 包）自身的 pypdfium2
路径（`docling.backend.*` 的页面栅格、`docling.utils.pdf_outline` 的大纲）才与本
闸共用同一份 `libpdfium.so`，原持有的是它自带的 `threading.Lock`
（`docling.utils.locks.pypdfium2_lock`）。两把锁守一个非线程安全库正是本闸要消除
的竞争，故 `docling_parser.extract_with_docling` 在 docling 导入后调用
`pdfium_gate.unify_docling_pdfium_lock()`，把 docling 各模块的 `pypdfium2_lock`
重绑为 `PDFIUM_LOCK`（仅在原生调用期间持锁，不横跨其模型推理）。

### 第二层：钉扎替换字体集（`pdfium_gate.install_font_policy`）

Linux 上 pdfium 默认**自己遍历系统字体目录**读每个文件的 name 表攒替换表
（不走 fontconfig），桌面机动辄数百字体——扫描慢、结果依赖宿主、且首次
触发时的懒初始化正是竞争窗口。gate 模块在 pypdfium2 首次导入前
hook `FPDF_InitLibraryWithConfig`，注入 `m_pUserFontPaths`：

- 默认取 `/usr/share/fonts` 下的 `liberation`（顶 Helvetica/Arial）、
  `gsfonts`（Nimbus，顶 Times）、`dejavu`（顶 Courier）、`noto-cjk`；
- `UBT_PDFIUM_FONT_DIRS`（`os.pathsep` 分隔）覆盖，供容器镜像只装
  这十几个字体的部署形态；目录全部不存在时自动退回宿主扫描（不误伤）。
- 语义为**替换而非追加**（pdfium `FPDF_LIBRARY_CONFIG` 约定），本机
  空目录实验确认：不引用任何宿主字体仍能正常提取/渲染。
- 若 pypdfium2 先于 gate 被导入，记 WARN 并退回第一层单防（不崩）。

### 守护测试

`tests/unit/test_pdfium_gate.py` 用 AST 扫描**整个 ubt/ 包**：任何文件
import pypdfium2 而未先 import gate（或顺序颠倒导致字体策略错过安装
窗口）→ 测试直接红。另含锁重入、6 线程互斥、open_document 往返测试。

### 已验证

- 改动文件编译/导入通过；`PDFIUM_LOCK` 重入正常；
- 102 个单元测试全过（gate 守卫 + 全部受影响 adapter 测试）；
- 字体策略 `state=applied`，文本提取/渲染输出不变，首屏 render
  0.149s → 0.133s（收益主要是确定性与宿主隔离，本机字体缓存热时
  扫描本就只有一百毫秒级）；
- 真实并发压测：8 线程 × 20 轮页面加载 + 渲染混合负载，2.2s 零错误
  零新 core（修复前同一文档导出阶段两次必崩）。

### 压测中暴露的既有性能缺陷（独立于本 bug，未修）

`textgeom.extract_lines` 在页面文本 rect 数量异常大时呈**二次方级慢**：
本论文第 1 页有 9395 个 rect（多为逐词矩形），单页耗时 271s，而同文档
其他页 <0.1s。锁无过错——无锁原始函数同样慢。这会让 rigid/visual
路径在病理页上卡死分钟级，值得单独立项（线性化 row-merge 或按 rect
预算降级）。

### 已知的代价

- 进程内所有 pdfium 工作串行化。单 job 场景无感（瓶颈在 LLM 延迟）；
  docling 的栅格/大纲等 pypdfium2 调用现与 UBT 共用一把锁（见 §3 的锁统一），
  只在这些**原生调用**期间持锁，不横跨其模型推理——不再有"整书转换持锁
  分钟级"的阻塞。
- 多用户服务化时这是真实吞吐瓶颈，见下节。

## 4. 演进路线（第二阶段，未实施）

1. **worker 进程化**（推荐下一步）：`ubt worker` 已是租约
   队列。让每个 worker **进程**领取 job，锁自动退化为进程内竞争，按
   CPU 核数横向扩展，改动最小。
2. **pdfium sidecar 子进程**（服务可用性的终态）：本 bug 证明了
   libpdfium 一颗段错误就能带走整个 API 进程。参照 ocr-sidecar 的现成
   模式，把 pdfium 调用关进子进程：崩溃只死 sidecar，主服务 IPC 超时
   重试即可。同时天然解开全局锁的串行化代价（sidecar 内可多实例）。
3. **更细的锁粒度**（收益小，不建议优先）：字体表初始化一次后全局只读，
   理论上可做"初始化全局锁 + 后续 per-document 锁"，但需精确掌握
   pdfium 哪些路径碰共享状态，维护成本高且上游行为随版本变化。
4. **上游化**：给 pdfium 的 `CFX_FolderFontInfo` 惰性初始化竞争提 issue
   或查是否已有修复版本；pypdfium2 文档明确 pdfium 仅"thread-compatible"，
   并发责任历来甩给调用方，长期看升级 pdfium 不保证消除该问题。

## 5. 排查方法备忘

- `coredumpctl list | tail` 确认崩溃是否同签名列队；
- `coredumpctl info <pid>` 直接给出全线程栈——本例中"另一线程同在
  pdfium 内"就是决定性证据；
- 触发条件核查：`pdffonts <doc.pdf>` 出现 `(no file)` 的 Type1 标准
  字体名 = 走替换路径 = 有竞争暴露面。
