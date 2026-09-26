# 技术设计 V1: 原位翻译工作台 + 保版导出 (TDD)

> **⚠️ SUPERSEDED（2026-09-22 归档，不再是开发依据）。** 本文的模块地图（events.py、
> `adapters/pdfops/`、`composers/inplace.py`、`quality/abbor.py`、`desktop/`）零落地；
> pdf_oxide redact/text_in_rect 主路线已被 `rigid` 引擎的 pikepdf stream-strip + Typst overlay
> 取代（见 `ubt/adapters/pdf/rigid/`、`ubt/adapters/pdf/stream_strip.py` 与
> [docs/design/LAYOUT_PRESERVATION_MASTERPLAN_2026-09.md](../design/LAYOUT_PRESERVATION_MASTERPLAN_2026-09.md)）。
> §8.3 的离散步长字号策略与实现相反（实际为二分搜索 + fontTools 字宽，见 `text_fit.py`）；
> PyMuPDF"逃生门"被 license guard 测试明令禁止。仅存设计推理参考价值。
- 状态: ~~Final (开发依据文档)~~ → 已归档，勿作为开发输入
- 日期: 2026-09-19
- 上游: INPLACE_WORKBENCH_PRD_V1.md (需求与验收) · docs/assessments/INPLACE_TECH_SURVEY_2026.md (选型证据)
- 读者: 引擎( Python )、桌面(Rust/Tauri)、前端(React) 三条开发线

---

## 1. 模块地图

```
ubt/
├── core/
│   ├── events.py                    # [新] EventBus + 契约 dataclass (schema_version)
│   ├── engine/pipeline.py           # [改] 状态迁移处 publish()（≤10 行侵入）
│   └── composers/
│       └── inplace.py               # [新] 保版导出合成器（策略 A，调 adapter）
├── adapters/
│   └── pdfops/                      # [新] PDF 操作 vendor 适配层（pdf_oxide 单点收敛）
│       ├── __init__.py              # PdfOps Protocol 定义（唯一对外接口）
│       ├── pdf_oxide_backend.py     # 主实现（锁版本）
│       ├── pymupdf_backend.py       # 逃生门实现（商业授权后启用）
│       └── color_sample.py          # S7 背景主色采样
├── quality/
│   └── abbor.py                     # [新] FR-13 ABBOR/IoU 计算 + 审校队列生成
└── desktop/                         # [新] Tauri 工作台
    ├── src-tauri/                   # Rust 壳: sidecar 生命周期 + 事件桥
    └── web/                         # React + pdf.js + overlay
        └── src/
            ├── store/trans.ts       # 单一事实源 store
            ├── layers/              # L1 pdf.js / L2 textlayer / L3 overlay
            └── bridge/events.ts     # 契约消费（listen → store 迁移）
```

依赖许可基线：pdf_oxide (MIT/Apache) · pikepdf (MPL) · pdf.js (Apache) · 零新增 GPL/AGPL。
pdf_oxide 版本锁定：`pdf_oxide==0.3.78`（升级走 S5 回归流程，见 §3.4）。

---

## 2. 事件契约 v1（全端唯一接口）

### 2.1 Schema（`ubt/core/events.py`，dataclass → JSON）

```python
SCHEMA_VERSION = 1

# 所有事件公共信封:
# { "v": 1, "seq": <单调递增int>, "ts": <epoch float>, "type": "<事件名>", "payload": {...} }

JobStarted      payload: {doc_id: str, total_pages: int, total_blocks: int}
PrepProgress    payload: {current: int, total: int, stage: "parse"|"ocr"|"layout"}
SegmentApplied  payload: {block_id: str, page: int, source: str, target: str,
                          bbox: [x0,y0,x1,y1],            # pt, 左下原点
                          overflow_ratio: float,          # 1.0 = 完美装下
                          scale_factor: float,            # FR-8 搜索结果
                          fill_rgb: [r,g,b] | None,       # S7 采样结果
                          text_rgb: [r,g,b] | None,       # 原文颜色继承
                          qe: float | None}               # 供 FR-14 路由
SegmentFailed   payload: {block_id: str, page: int, error_code: str}
JobSpeed        payload: {blocks_per_min: float, eta_seconds: float}   # 节流 ≥1s
JobCancelled    payload: {reverted_blocks: [str], kept_blocks: [str]}
JobFinished     payload: {stats: {applied: int, failed: int, kept_source: int},
                          abbor: float | None}            # 导出后填充
ReviewEdited    payload: {block_id: str, new_target: str, editor: str}  # 前端→引擎方向
```

### 2.2 时序铁律（PRD §4 不变量的实现约定）

1. 任何终态（JobFinished/JobCancelled）前，保证已 publish 全部在途 SegmentApplied。
2. seq 由引擎侧单点递增；前端丢弃 `seq <= last_seen` 的乱序事件。
3. 断线重连（S4）：前端携带 `last_seq`，引擎从环形缓冲（容量 2048）重放；
   缓冲溢出则前端降级为全量状态快照请求（`state.snapshot` RPC）。

---

## 3. PdfOps 适配层（S5 定案）

### 3.1 Protocol（唯一被 composer/UI 后端引用的接口）

```python
# ubt/adapters/pdf_oxide/__init__.py
class WordGeom(TypedDict):  # span 级几何
    text: str
    bbox: tuple[float, float, float, float]  # (x,y,w,h) pt
    font_name: str
    font_size: float
    rotation: float


class PdfOps(Protocol):
    def open(self, path: str) -> "Doc": ...

    # Doc 协议:
    #   page_count() -> int
    #   page_size(page) -> (w, h)
    #   extract_words(page) -> list[WordGeom]
    #   redact_destructive(page, rects: list[Rect], fill: RGB | None) -> RedactReport
    #   render_png(page, dpi) -> bytes
    #   write_overlay(page_size, items: list[OverlayItem]) -> bytes   # 生成透明 overlay PDF 页
    #   save(path) / close()


class OverlayItem(TypedDict):
    rect: tuple[float, float, float, float]
    text: str
    font_size: float
    text_rgb: tuple[float, float, float] | None
    align: Literal["left", "center", "right"]
```

### 3.2 pdf_oxide_backend 实现要点（全部经 S1/S2 实测）

```python
# redact_destructive:
doc.add_redaction(page=page, rect=(x0, y0, x1, y1), fill=fill)  # fill 必传(见 S7)
report = doc.apply_redactions_destructive()
# ⚠️ 已知行为: RuntimeError("refuses composite/Type0") 仅存在于过时 docstring,
#    实测 Identity-H CIDFontType2 正常; 若上游真抛错 → 记 error_code=REDACT_REFUSED
#    并回退该块为"保留原文+角标", 不中断整篇。

# write_overlay:
b = po.DocumentBuilder().register_embedded_font("UBT-CJK", po.EmbeddedFont.from_file(CJK_FONT_PATH))
page = b.page(w, h).font("UBT-CJK", size)
for it in items:
    page = page.text_in_rect(x, y, w, h, text, align=...)  # 签名: (x,y,w,h,text,align)
page.done().save(tmp)  # DocumentBuilder 无 to_bytes, 走临时文件
# ⚠️ TextWord.bbox 是 (x,y,w,h) 非 (x0,y0,x1,y1) —— adapter 内统一转换, 对外只暴露 x0y0x1y1。
# ⚠️ EmbeddedFont handle 一次性(消费后 RuntimeError), 每次 write_overlay 重新 from_file。
```

### 3.3 合成（pikepdf，MPL）

```python
base = pikepdf.open(redacted_path)
ov = pikepdf.open(io.BytesIO(overlay_bytes))
# 页尺寸一致性断言: base.pages[i].MediaBox == overlay 页 MediaBox (容差 0.5pt)
base.pages[i].add_overlay(ov.pages[i])
```

### 3.4 版本锁与升级回归（S5 流程，替代持续验证）

- `requirements-desktop.txt` 锁 `pdf_oxide==0.3.78`；升级 PR 必须跑 §8.2 黄金集全量。
- adapter 层单测 mock PdfOps——composer 与 UI 后端**永不 import pdf_oxide**（CI 加 import 守卫，
  与现有 license guard 同族）。
- 逃生门：`pymupdf_backend.py` 实现同一 Protocol（redaction=page.add_redact_annot+
  apply_redactions；写入=insert_textbox）。启用条件 = 商业授权采购 or pdf_oxide 上游崩塌。

---

## 4. 保版导出 Compositor（`ubt/core/composers/inplace.py`）

### 4.1 主流程（伪代码，逐页）

```python
def compose_inplace(pdf_path, ir, out_path, *, enable_vlm=False) -> InplaceReport:
    doc = backend.open(pdf_path)
    for page in doc.pages():
        blocks = ir.blocks_on(page)  # 只处理 status ∈ {translated, accepted}
        rects, items = [], []
        for b in blocks:
            if b.block_type == FORMULA:
                continue  # FR-11: 公式永不进 overlay
            fill = sample_bg_color(doc, page, b.bbox)  # §5.2 (S7)
            rgb = extract_source_color(b)  # FR-16: 原文色继承
            size = fit_font_size(b, doc, page)  # §5.1 (S6, FR-8)
            rects.append(RedactRect(b.bbox, fill=fill))
            items.append(OverlayItem(b.bbox, b.target, size, rgb, b.align))
        doc.redact_destructive(page, rects)  # §3.2
        ov_bytes = doc.write_overlay(doc.page_size(page), items)
        composite(page, ov_bytes)  # §3.3
    abbor = compute_abbor(ir, out_path)  # §5.3 (FR-13)
    if enable_vlm:
        defects = vlm_inspect(out_path)  # FR-15, 默认关
    return InplaceReport(per_block=..., abbor=..., review_queue=abbor.low_blocks)
```

### 4.2 错误处理矩阵

| 故障 | 策略 | 事件 |
|---|---|---|
| 单块 redact 抛 REDACT_REFUSED | 该块保留原文 + 角标，继续整篇 | SegmentFailed(REDACT_REFUSED) |
| overlay 页 MediaBox 不一致 | 断言失败 → 整页跳过合成（原件输出） | SegmentFailed(PAGE_MISMATCH) |
| 字体注册失败 | 降级到系统 Noto 路径列表；全失败 → 终止并报 FONT_MISSING | JobFinished(error) |
| 导出中途取消 | 页级原子：当前页完成才落盘，已完成页保留 | JobCancelled |

---

## 5. 算法定案（S6/S7/S3/S4）

### 5.1 S6 · 缩放因子迭代搜索（FR-8 ②）

```python
def fit_font_size(block, doc, page) -> tuple[float, float]:  # (size, scale_factor)
    bbox = block.bbox_pt
    # 预算: 优先用翻译期注入的字符预算 (FR-8 ①); 此处兜底
    for scale in [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]:  # BabelDOC 式离散步长
        size = block.source_font_size * scale
        if fits(bbox, block.target, size, leading=size * 1.35):
            return size, scale
    return size_floor, 0.6  # 触底 → overflow_ratio>1 → 黄标 (FR-8 ③)


def fits(bbox, text, size, leading) -> bool:
    # 行宽估算: CJK 字符 ≈ 1.0em, Latin ≈ 0.5em (fontTools hmtx 精化, 见 §8.3 优化项)
    lines = wrap(text, width=bbox.w / size)  # 贪心断词, 与 text_in_rect 同规则
    return len(lines) * leading <= bbox.h + 0.5
```

DeepL 四级约束优先级实现约定：搜索顺序**先整块后全局**——单块搜索互不通信（保"版面位置"），
导出末尾若同页字号方差 > 3 级，仅记录警告不改版（保"同页>位置"压倒"全文一致"）。

### 5.2 S7 · bbox 背景主色采样

```python
def sample_bg_color(doc, page, bbox, dpi=150) -> RGB:
    # 1) render_png(page, dpi) 一次/页, 缓存复用 (整页只渲染一次, 供 S7+VLM 共用)
    # 2) 取 bbox 外扩 3pt 的边框环带像素 (排除文字笔画: 亮度处于极端 5% 的像素丢弃)
    # 3) RGB 量化到 4bit/通道 → 直方图取众数 → 反量化取该桶均值
    # 失败(透明/超大图) → (1,1,1) 白, 并记 warning
```

### 5.3 FR-13 · ABBOR / BBox IoU

```python
def block_iou(orig_bbox, placed_bbox) -> float:
    # placed_bbox = 导出后对 final.pdf 重新 extract_words 聚合该块译文行的实际外接框
    # (复用 adapter 的 extract_words, 对译文 overlay 同样有效——写入即矢量)
    inter / union

def compute_abbor(ir, final) -> AbborResult:
    per_block = [...]; page_means = group_by_page(per_block)
    return AbborResult(global_mean=Σ, low_blocks=[b for b if iou < 0.8])  # 审校队列
```

### 5.4 S3 · pdf.js 几何对齐（设计定案，误差策略代替前置验证）

- 权威源 = **IRBlock.bbox**（Docling 产出，pt 空间）。pdf.js textContent 仅用于
  hover 命中测试与原文选择，**不参与 overlay 定位**——从架构上消除对齐依赖。
- 旋转页：`viewport.convertToViewportRectangle(bbox)` 处理 rotate；180/90/270 一律走
  pdf.js API 换算，不自算矩阵。
- 运行时自检（替代验证）：M2 起每页加载后计算"IR bbox ∪ vs textContent 外接框"的 IoU，
  < 0.5 的页打 `geom_suspect` 角标并进遥测——把 S3 变成线上指标而非上线门槛。

### 5.5 S4 · Tauri sidecar 事件桥（设计定案）

```
生命周期: Rust spawn(python -m ubt.desktop.sidecar) → 双向 stdio
  · 下行: JSON-lines 命令 {cmd: "start_job"|"cancel"|"snapshot"|..., id}
  · 上行: §2 事件信封 JSON-lines
  · 心跳: 2s ping/pong, 3 次丢失 → Rust 重启 sidecar + 前端 state.snapshot 恢复
背压: 引擎侧 SegmentApplied 合并窗口 100ms (同块多次更新只发末值)
开发模式: sidecar 可独立以 HTTP/SSE 起 (ubt/desktop/sidecar.py --http 8765),
  前端脱离 Tauri 用浏览器开发, 事件桥抽象为 Transport trait (Sse | Stdio)
```

---

## 6. 前端架构（React + pdf.js，PRD §3 三层模型落地）

```
<App>
 ├─ <WorkspaceStore>          zustand: {running, stage, cur, total} + blocks: Map<id, SegState>
 ├─ <PageVirtualizer>         仅挂载 current±2 页
 │   └─ <Page n>
 │       ├─ <PdfCanvasLayer>  pdf.js render → canvas (L1)
 │       ├─ <TextLayer>       pdf.js TextLayer (L2, 命中测试/hover 原文)
 │       └─ <OverlayLayer>    (L3) blocks_on(page) → 绝对定位 div
 │           ├─ 白/采样色底 + 译文 + scale 变换 (与导出共用 fit_font_size 结果回传)
 │           ├─ 状态角标: 黄(overflow) 蓝(低置信) 红(failed) 绿勾(accepted)
 │           └─ contenteditable (FR-7) → debounce 800ms → ReviewEdited RPC
 ├─ <Toolbar>                 视图三态(FR-9) / 停止 / 导出 / 审校队列入口
 └─ <Bridge>                  Transport 事件 → store 迁移 (唯一写入口)
```

约定：
- OverlayLayer 的定位公式：`left = bbox.x0 * vpw`，`top = (pageH - bbox.y1) * vpw`
  （pt→px 统一经 pdf.js viewport，FR-12 由此保证）。
- 审校编辑只改 L3 DOM + 发事件，**永不**直接改 canvas/IR 文件（单一数据流）。
- hover 原文（FR-4）：L3 div `pointer-events` 穿透到 L2 选择文本；L3 自身监听
  modifier+hover → opacity 0.35 露出 L1。零请求。

---

## 7. 引擎侧集成（≤10 行侵入）

- `pipeline.py` 现有 block 状态迁移点（draft→qe→applied）处插 `bus.publish(SegmentApplied)`；
  ledger 表**不加列**——applied/accepted 状态已够，人工版本写现有 target 字段 +
  status=accepted（复用 added_content_gate 的"人工优先"守卫）。
- 导出入口：`composers/inplace.py::compose_inplace` 注册为 `output_mode="inplace"` 的
  preset 路由（与 publication/preview 并列），CLI/TUI/sidecar 三端共享。
- budget prompt（FR-8 ①）：翻译阶段前，`bbox 面积×字号 → 目标字符预算` 注入
  `prompts.py` 的 segments 载荷（新增可选字段 `budget_chars`，模型不支持时自然忽略）。

## 8. 测试计划

### 8.1 单测（CI 门禁）
- PdfOps mock：composer 全分支（含 REDACT_REFUSED / PAGE_MISMATCH / FONT_MISSING）。
- fit_font_size / sample_bg_color / block_iou 纯函数黄金用例。
- 事件契约：dataclass ↔ JSON round-trip + schema_version 断言；seq 单调性 property test。

### 8.2 黄金集回归（pdf_oxide 升级必跑）
fixture 四件：Type1 页(chapter-3) · Type0 CID 页(自生成 overlay) · 旋转 90° 页 ·
彩色底白字段（FR-16 回归）。断言：redact 后 pdfminer 零残留 + ABBOR 页均值 ≥0.85 +
渲染像素 diff（阈值 SSIM<0.02 变化）。

### 8.3 E2E（M3 起）
sidecar --http 模式 + Playwright：拖入 PDF → 试译 → 断言 SegmentApplied 序列与
overlay DOM 一致 → 停止 → 断言回滚。性能门禁：首段 ≤ LLM+200ms、取消 ≤1s。
（优化项：行宽估算换 fontTools hmtx 精确度量——M4 前用启发式即可。）

## 9. 开发任务分解（按 PRD §9 里程碑展开）

| # | 任务 | 产出文件 | 依赖 | 验收 |
|---|---|---|---|---|
| T1 | EventBus + 契约 | core/events.py | — | §8.1 契约测试绿 |
| T2 | PdfOps Protocol + pdf_oxide 实现 | adapters/pdfops/* | — | §8.2 黄金集 4/4 |
| T3 | S7 采样 + S6 搜索 + IoU | color_sample.py, quality/abbor.py | T2 | 单测 + chapter-3 e2e 出"第三章" |
| T4 | Compositor | composers/inplace.py | T1-T3 | CLI `--output-mode inplace` 跑通基准论文 |
| T5 | pipeline publish | pipeline.py 插桩 | T1 | 事件序列快照测试 |
| T6 | sidecar + Transport | desktop/sidecar.py | T5 | SSE 模式收全事件 |
| T7 | 前端三层 + store | desktop/web/* | T6 | FR-1/2/3/4/9 手测清单 |
| T8 | 审校写回 + 队列 UI | ReviewEdited 链路 | T4,T7 | FR-7/14 验收 |
| T9 | 停止/回滚/续传 | 全链 cancel 贯通 | T7 | FR-5/6 + §8.3 性能门禁 |
| T10 | VLM 巡检(可延) | quality/vlm_inspect.py | T4 | FR-15 四类缺陷召回 |

## 10. 开放决策（不阻塞开发，M2 前定）

1. CJK 字体分发：随包 Noto Sans CJK（OFL 可嵌入）vs 用户自选——建议随包 + 设置项覆盖。
2. overlay 页缓存位置：workspace cache 复用 SHA-256 目录结构（建议）vs 临时目录。
3. pdf_oxide 上游 issue 策略：告警噪音（"Dictionary used where Stream expected"）
   是否提 issue——建议 M4 后统一提，附最小复现。
