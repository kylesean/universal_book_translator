# PDF 解析引擎路由重构：从「自动加载」到「建议 + 显式授权」

> **状态**：✅ 主体已落地（L1/L2/L4/L5 已实现并验证；L3 告知部分随 L5 一并完成，见 §5）
>
> 当前行为地图见 [`PDF_INGEST_AND_VLM_ROUTING.md`](PDF_INGEST_AND_VLM_ROUTING.md)；
> 本文保留重构动因、逐层方案与实施日志。
>
> **进度**（2026-10-10）：
> - ✅ **L1** 修 `formula_debris_share` 人名缩写误判
> - ✅ **L2** 修采样偏差（末页改 0.92 + 信号需 ≥2 页）
> - ✅ **L4** 解耦 Docling 的「使用」与「资源」（`UBT_DOCLING_DEVICE`）
> - ✅ **L5** 预设控制 `pdf_engine` + 默认改 `pdfium` + 扫描逃逸
> - ✅ **L3** 告知（强制 pdfium 但探测建议 docling 时告警；扫描逃逸时告警）
>
> **性质**：本文档是**设计提案**。文中 `path:line` 是**计划落点**，可能随实现漂移；
> 现有系统行为的引用以 `ubt/` 源码为准。与活文档 [`PDF_INGEST_AND_VLM_ROUTING.md`](PDF_INGEST_AND_VLM_ROUTING.md)
> （描述当前系统）互补：那份说"现在怎么走"，这份说"应该怎么走、为什么、怎么落地"。
>
> **触发**：2026-10-10 一次真实的翻译失败（本地 LLM 500），暴露了引擎路由的三个结构性缺陷。

---

## 0. TL;DR

**默认引擎保持 `auto`（探测选引擎）。**

> ⚠️ **2026-10-10 反转**：本文档初稿主张把默认改为 `pdfium`，并已实现。
> 用户随后用真实文档（arXiv 2610.02185）跑默认参数，产物**无法交付**，
> 根因就是这次改动 —— 见 §0.1。默认已回退为 `auto`。
>
> ⚠️ **2026-10-20 变更（以源码为准）**：`auto` 现在**无条件**选 docling，
> 不再"探测后分流到 pdfium"。探测信号仍被计算，但只作为**关于文档的
> 证据**（`ubt assess` 报告用它），不再参与引擎投票。因此：
> - 干净单栏 born-digital PDF **也**走 docling；
> - 没有装 `pdf` extra 时 `auto` **直接失败**，不会静默降级到 pdfium；
> - 想走快路必须显式 `--pdf-engine pdfium`（或 `preview`/`fast` 预设）。
> 权威实现见 `ubt/adapters/pdf/engine_selector.py` 与 `ubt/adapters/factory.py`。

三层优先级：

```
CLI --pdf-engine  >  预设 pdf_engine  >  config 默认 (auto)
```

| 入口 | 引擎 |
|---|---|
| 默认（无 flag） | **`auto`** = **docling（无条件）**；探测结果只进 `assess` 报告 |
| `--preset high_fidelity` / `standard` | `auto` |
| `--preset preview` / `fast` | `pdfium` |
| `--pdf-engine docling` | `docling`（最高优先级） |
| **扫描件**（任意模式） | `docling`（能力边界，pdfium 抽 0 块） |

### 0.1 为什么默认必须是 `auto`（实测教训）

同一篇 `2610.02185v1.pdf`，两个引擎的块级对比：

| | `pdfium` | `docling` |
|---|---|---|
| 总块数 | **1801** | **350** |
| 表格块 | **0**（被拆成碎片） | **19**（结构化） |
| 公式块 | 2 | 6 |
| 列表块 | 15 | 57 |
| 标题 | `'d L eco n'`、`'g oo e p fo'` 碎片 | `'Decoding Looped Transformers Better for (Almost) Free'` ✅ |
| 可翻译块中碎裂>30% | 201 / 1796 (11%) | 19 / 230 (8%) |

`pdfium` 把这篇切成 **5 倍多的碎片块**（表格被拆成 413 heading + 1371 narrative）。
碎片块里的数字/符号错乱，触发 `Numeric fidelity failure` 等校验 →
**244 块被 quarantine（14%）→ 233 个 `【待人工审校】` 标记被画进 PDF**。

**结论**：对表格/公式/多栏文档，`pdfium` 不是"质量略差"，而是"碎片化到不可交付"。
默认必须交付体验最好（用户不懂参数），所以是 `auto`，不是 `pdfium`。

> **docling 用了哪些模型**（回答"为何没走 VLM"）：docling 一跑就带两个**判别式视觉模型** ——
> `docling-layout-heron`（版面）与 `TableFormer`（表格，重建出这 19 个表格）。
> 唯一的**生成式 VLM** `CodeFormulaV2` 反而**被关闭**：这篇被判 `formula_heavy=True`，
> 而 `resolve_formula_enrichment` 对公式密集文档会跳过冗余的 VLM OCR（保留原始矢量公式）。
> 所以"效果好"来自版面+表格模型，与 VLM 开关无关。

---

## 1. 问题陈述

### 1.1 现场：一次真实的失败

```
$ uv run ubt translate /home/kyle/Downloads/2610.02185v1.pdf --provider index-translate
...
HTTP Request: POST http://127.0.0.1:9090/v1/chat/completions "HTTP/1.1 500 Internal Server Error"  (×N)
```

网关日志（`/tmp/swap.log`）：

```
[WARN] group: starting index-translate-9b failed: upstream command exited prematurely
→ HTTP 500
```

直接复现上游，拿到根因：

```
E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4812.25 MiB on device 0:
    cudaMalloc failed: out of memory
E llama_model_load: error loading model: unable to allocate CUDA0 buffer
```

**显存不足，本地 LLM 的权重都装不进去。** 因为同一个 `ubt` 进程里，Docling 的解析先占了 GPU：

```
nvidia-smi: 12282 total, 5687 used, 6183 free
  ubt python (docling CUDA context + 版面/表格模型)  1634 MiB   ← 就是这次 translate 进程
  chrome gpu-process                                 1932 MiB
  ...
```

而 `index-translate-9b`（`CTX=32768 / NP=16`）需要 ~6.4 GiB（权重 4812 + KV 544 + RS buffer 804 + compute 216）。

### 1.2 用户视角的矛盾

用户报告的原始疑问（三连问，全部成立）：

1. **"provider 没有吗？"** → 有。`index-translate` 在 `ubt.toml:50` 声明，网关 `/v1/models` 也有。500 与 provider 无关。
2. **"我之前就这样用，一切正常"** → 属实。配置、二进制、模型**逐字节未变**。变的是**文档**：新论文首次解析，未命中 Docling 缓存。
3. **"为什么非公式密集的场景反而启动 Docling？这违背常识"** → **成立，且是 bug**（见 §1.3）。

### 1.3 已复现的两个缺陷

#### 缺陷 A：`formula_debris_share` 把人名缩写当成公式碎片

`2609.22978v1.pdf`（31 页）被判定 `has_formulas=True` → 路由到 Docling。
逐页实测 `formula_debris_share`：

| 页 | share | 内容 |
|---|---|---|
| 1–24 | 0.002 – 0.138 | 正文 |
| **25** | 0.138 | 引用段 |
| **26** | 0.104 | 参考文献 |
| **27** | **0.382** | `D. Guo, D. Yang, H. Zhang, J. Song, P. Wang...` |
| **28** | 0.324 | 同上 |
| **29** | 0.344 | 同上 |
| **30** | 0.150 | 同上 |
| **31** | 0.182 | `P. Zijlstra, J. Fernandes, and V. Pillai...` |

**7 页全是参考文献页**，而真正的正文页最高只有 0.067。检测器数的是"孤立单字母 token"
（`ubt/core/policy/layout_policy.py:367`）：

```python
singles = sum(1 for t in tokens if len(t) == 1 and t not in PROSE_SINGLE_LETTERS)
```

作者缩写 `P.` `J.` `V.` 正好命中。手工验算：

```
tokens = ['P','Zijlstra','J','Fernandes','and','V','Pillai', ...]
单字母 = ['P','J','V']  →  share = 3/21 = 0.143  （整页 0.182，阈值 0.10）
```

**它把参考文献的作者缩写当成了碎裂的公式符号。**

#### 缺陷 B：采样分数 `1.0` 永远命中末页 = 论文的参考文献页

`ubt/core/policy/layout_policy.py:252`：

```python
PROBE_SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)   # 1.0 = 末页
```

学术论文的末页几乎必然是参考文献。实测四篇论文的末页：

| 文档 | 末页 share | 内容 |
|---|---|---|
| `2609.22978v1` | **0.182** ← 命中 | 参考文献 |
| `2608.25512v1` | **0.087** ← 差 0.013 | `[124] G. H. Cooper and S. Krishnamurthi...` |
| `2609.20519v1` | 0.008 | 正文 |
| `2610.02185v1` | 0.010 | 正文 |

`2608.25512` 只差 **0.013** 就会同样被误判。**这是系统性偏差，不是偶发。**

#### 缺陷 C（连锁）：误判反向关掉了公式 VLM

`ubt/adapters/pdf/docling_parser.py:148`：

```
formula_heavy=True  →  关闭公式 VLM
```

所以 `2609.22978`（**真有零星公式、正需要 VLM 保护**）因为一页参考文献被关掉了 VLM；
而真正公式密集的文档本来就不需要 VLM。**逻辑完全反了。**

### 1.4 缺陷 D：`--preset fast` 承诺"零重型依赖"却仍加载 Docling

`FAST` 预设的自我描述（`ubt/core/presets.py:105`）：

```python
Preset.FAST: PresetPolicy(
    label="极速论文",
    tagline="极速开箱即用：零重型依赖，轻量几何抽取，短链快速翻译",
    description="... 零重型环境门槛",
    ...
)
```

但 `pdf_engine` **不在**预设控制范围内（`ubt/core/presets.py:64`）：

```python
PRESET_ENGINE_FIELDS = ('exec_mode', 'prompt_strategy', 'formula_enrichment',
                        'qe_engine', 'table_render_mode', 'emit_both')
#                        ↑ pdf_engine 不在这里
```

**实测确认**（clean 环境）：

```bash
$ uv run ubt translate ~/Downloads/2609.20519v1.pdf --preset fast --dry-run --yes -o /tmp/fasttest
$ sqlite3 .ubt/ledgers/job_7f3910c43a49_...sqlite \
    "select json_extract(metadata_json,'$.pdf_parser_engine') from job_meta;"
docling          ← 就是 docling
```

`resolve_engine_params(Preset.FAST, ...)` 的返回值里**根本没有 `pdf_engine` 键**。
所以 `--preset fast` 完全不影响解析器选择，`auto` 探测照样跑，照样可能选中 Docling 并占 GPU。

---

## 2. 根因分析

缺陷 A–D 只是症状。真正的病根有三个，都在**设计层**。

### 2.1 病根一：可靠性天差地别的信号被一视同仁地 `or`

`ubt/adapters/pdf/engine_selector.py:226`：

```python
needs_docling = (
    scanned_detected or has_formulas or has_multicolumn or has_vector_diagrams
)
```

四个信号的可靠性**不在一个量级**：

| 信号 | 判据 | 可靠性 | 失败模式 |
|---|---|---|---|
| **扫描件** | 过半抽样页 `n_chars < 32` | **高**：客观二值，几乎不可能误判 | 漏判 → **空书**（灾难） |
| 矢量图密集 | 过半抽样页 `path_ops >= 200` | 较高 | 漏判 → 图内文字丢失 |
| **多栏** | `right_share > 0.25` | **低**：连续值，图内文字污染 | 误判 → 慢 + 占显存 |
| **公式密集** | `formula_debris_share > 0.10` | **低**：连续值，人名缩写污染 | 误判 → 慢 + 占显存 |

**把"客观二值"和"模糊连续"用 `or` 缝在一起，等于让最不可靠的信号拥有否决权。**

### 2.2 病根二：解析引擎与 GPU 可用性形成循环依赖

```
选 docling → 开 CUDA → 占 1.6G 显存 → 本地 LLM 加载失败 → 500
                 ↑
        formula_enrichment 又依赖 has_accelerator()
```

**翻译引擎（LLM）与解析引擎（Docling）在抢同一块显存，但代码里这两个决策互相看不见。**
这是架构级的气味，**不是调阈值能修的**。

### 2.3 病根三：重型资源自动加载，没有"预算"概念

`ubt assess` 会告诉用户预期花多少 token、多少钱 —— 唯独**不告诉"这一步会占 1.6G 显存"**。
而它恰恰是唯一会跟本地 LLM 抢资源的步骤。

### 2.4 决定性事实：选错引擎是**静默错误**

这是整个重构的**地基**。把多栏论文强制走 pdfium：

```
WARNING  UBT_PDF_ENGINE='pdfium' forced ... degraded equation fidelity
INFO     Delivery contract: [PASS]
         text 88/146 delivered (58 verbatim, 0 source-kept, ...) | 0 error(s), 0 warning(s)
```

**契约报 [PASS]，0 错误 0 警告。** 但 pdfium 实际抽出的是：

```
heading  'Research environments Research environmen g (( g Research en'
heading  'g g g g g Optimization D ED E'
heading  'KhKhKhKhhh KhKhKhKhhh hhhhhh Keep changeKeep change Environm'
heading  '$1$1$17$17$1741$1741 $1$1$17$17$1741 $$$$$$ $1,7$1,7 AP'
```

两栏文字**逐行交错**（`Research environmen g (( g`），图内数字**重复拼接**（`$1$1$17$17$1741`）。
而**契约说 PASS** —— 因为它只检查"文字有没有丢"（`ubt/core/content/contract.py:26`），
不检查"文字对不对"。

**推论**：引擎选择的错误不会响亮失败，会产出一本**看起来交付成功、实际是垃圾**的书。
这决定了**不能让不可靠的启发式自动触发重型路径**。

### 2.5 不对称性：必须诚实面对

| 情况 | 后果 |
|---|---|
| 该用 Docling 却没用 | **静默垃圾**（契约还 PASS） |
| 不该用 Docling 却用了 | 慢一点、占点显存，但**输出正确** |

这个不对称意味着"拿不准就用 Docling"在**质量**上是合理的保守策略。

**所以当前设计的问题不是"保守"，而是保守错了维度**：

- 在**质量**上保守（宁可用重的，不冒出错风险）→ ✅ 对
- 在**资源**上冒进（自动加载 2GB CUDA 模型，不问用户）→ ❌ 错

**它把两个不同维度的风险混成了一个决策。**

---

## 3. 设计原则

### 3.1 核心原则：检测应「说话」，不应「动手」

| 角色 | 行为 | 类比 |
|---|---|---|
| **探测** | 继续跑，产出建议与告知 | 体检报告 |
| **决策** | 由用户显式授权，或由**可靠信号**自动 | 处方需医生签字 |

### 3.2 与既有先例一致

本仓库已有"最重的特性默认关闭"的先例：

```python
# ubt/adapters/pdf/vlm/transcribe.py:34
#: Four-tier fallback switch: "off" (default), "missing", "weak", "all".
FALLBACK_ENV_VAR = "UBT_VLM_SCAN_FALLBACK"      # 默认 OFF
```

自研 VLM 扫描兜底（比 Docling 更重：调云端、花钱、出网）**默认 OFF**。
Docling 同样重（torch + CUDA）、同样占资源、同样抢显存 —— **却默认 `auto` 自动触发**。

**同一个项目里，两个重量级特性用了两套相反的设计哲学。** 本次重构是把 Docling 拉回一致的标准。

### 3.3 显式化重型资源的两个正交维度

```
「用不用 Docling」   ← --pdf-engine / 预设档位        （质量维度）
「占不占 GPU」       ← UBT_DOCLING_DEVICE=auto|cpu|cuda （资源维度）
```

两个维度**必须可独立控制**。当前的 bug 就是它们被绑死了。

### 3.4 信号分级

| 级别 | 信号 | 效力 |
|---|---|---|
| **自动**（可靠 + 失败灾难性） | 扫描件 | 直接路由 Docling |
| **建议**（模糊） | 多栏 / 公式 / 矢量 | 告知 + 建议，不自动加载重型引擎 |

---

## 4. 方案设计

### 4.0 总览：分四层，可独立落地

| 层 | 内容 | 类型 | 依赖 |
|---|---|---|---|
| **L1** | 修 `formula_debris_share` 的人名缩写误判 | Bug 修复 | 无 |
| **L2** | 修采样偏差（末页 + 单页否决） | Bug 修复 | 无 |
| **L3** | 信号分级 + 探测降级为「建议 + 告知」 | 行为变更 | L1/L2 |
| **L4** | 解耦 Docling 的「使用」与「资源」 | 架构变更 | 无（可先行） |

**L1、L2、L4 相互独立，可并行。** L3 依赖 L1/L2（否则会保留误判）。

---

### L1：修 `formula_debris_share` 的人名缩写误判

**落点**：`ubt/core/policy/layout_policy.py:367`

**问题**：`P. Zijlstra` / `D. Guo` / `J. K. Rowling` 形态的作者缩写被当成公式碎片。

**方案**：在统计前剔除"姓名首字母缩写"形态。

```python
# 新增模块级常量
#: 姓名首字母缩写：`P. Zijlstra` / `D. Guo` / `J. K. Rowling`。
#: 这类 token 是参考文献的作者，不是碎裂公式的孤立符号 —— 见
#: docs/guides/PDF_ENGINE_ROUTING_REDESIGN.md §1.3。
_NAME_INITIAL_RE = re.compile(r"\b[A-Z]\.(?:\s*[A-Z]\.)*\s+[A-Z][a-z]+")


def formula_debris_share(text: str) -> float:
    """...（docstring 增补一句：先剔除姓名缩写）"""
    text = _NAME_INITIAL_RE.sub(" ", text)   # 剔除人名缩写后再统计
    tokens = ASCII_WORD_RE.findall(text)
    if not tokens:
        return 0.0
    singles = sum(1 for t in tokens if len(t) == 1 and t not in PROSE_SINGLE_LETTERS)
    return singles / len(tokens)
```

**为什么正则形态安全**：真正的公式碎片（`K = x plus`）没有"大写字母 + 句点 + 大写开头的单词"形态。
`K = x` 里 `K` 后跟的是 `=`，不匹配 `[A-Z]\.`。

**校准登记更新**（`ubt/core/policy/layout_policy.py` 的 `CALIBRATION`）：

```python
"PROBE_FORMULA_SHARE": KnobMeta(
    S, "0.10 separates formula pages (0.16+) from prose (<0.08); "
       "name-initials excluded at the source (2026-10-10 regression)"
),
```

**验证**：

```bash
uv run python -c "
from ubt.core.policy.layout_policy import formula_debris_share
# 参考文献页（当前误判 0.182）
t = 'P. Zijlstra, J. Fernandes, and V. Pillai. Core scheduling, 2021.'
print(formula_debris_share(t))   # 期望 ≈ 0.0
# 真公式碎片（必须仍然命中）
print(formula_debris_share('K = x plus'))   # 期望 ≈ 0.667
"
```

**测试**（`tests/unit/policy/test_layout_policy.py`）：

```python
def test_formula_debris_excludes_name_initials() -> None:
    # 参考文献的作者缩写不是公式碎片 —— 2026-10-10 回归
    assert formula_debris_share("P. Zijlstra, J. Fernandes, and V. Pillai.") < 0.05
    assert formula_debris_share("D. Guo, D. Yang, H. Zhang, J. Song") < 0.05
    assert formula_debris_share("J. K. Rowling wrote this") < 0.05

def test_formula_debris_still_catches_real_debris() -> None:
    # 真实公式碎片必须仍然命中（回归护栏）
    assert formula_debris_share("K = x plus") == pytest.approx(2 / 3)
```

**风险**：低。需确认不误伤"公式里恰好有大写缩写"的极端情况 —— 但那本来就是公式页，仍会被其它信号覆盖。

---

### L2：修采样偏差

**落点**：`ubt/core/policy/layout_policy.py:252` + `ubt/adapters/pdf/engine_selector.py:226`

两个独立的子问题，建议都修。

#### L2a：避开末页（参考文献）

```python
# 原：1.0 永远采样末页，而论文末页几乎必然是参考文献（见 §1.3）
PROBE_SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 0.92)
```

#### L2b：从「单页否决」改为「≥2 页命中」

当前是"**任一**抽样页命中即翻转全书路由"。改成**至少 2 页命中**：

```python
# layout_policy.py
#: 一个模糊的版面信号（公式碎片 / 右栏边缘）必须出现在至少这么多抽样页上，
#: 才能翻转整本书的路由。一页是噪声，两页是信号。
PROBE_MIN_SIGNAL_PAGES = 2
```

**为什么不是"过半"**：初稿曾用 `>50%`，但**实测证明这会造成真实回归**。
`book.pdf` 是 296 页数学教材，41% 的页面含公式碎片，但 5 页采样只命中 2 页（40%），
`>50%` 规则会把它推给 pdfium 并**打碎其公式**。

实测数据（2026-10-10，`~/Downloads` 21 篇）：

| 规则 | 路由到 docling | 说明 |
|---|---|---|
| 原：任一命中 | 19 / 21 | 误判多（参考文献页翻转） |
| 初稿：>50% | 9 / 21 | **回归**：数学教材被推给 pdfium |
| **≥2 页** | **13 / 21** | 修正 6 篇误判，保留教材 |

**为什么 ≥2 是正确阈值**：5 页采样下，"两页"既排除了单页噪声（参考文献页、图页），
又不会因为长文档的稀疏采样而漏判（教材只有 41% 页面含公式，采样必然稀疏）。

**保留的例外**：**扫描件**继续用"过半抽样页无文本"判据（`_scan_detected_from_probe`），
因为它的判据本来就是多数、且失败模式是空书。整本无文本仍须 fail-closed 报错
（`docling_parser.py` 的 `_reject_empty_book`，当前约 :1275 起），不能静默出空书。

**测试**：

```python
def test_probe_ignores_single_reference_page() -> None:
    """单页参考文献不应翻转全书路由（2026-10-10 回归）。"""

def test_probe_majority_still_routes_multicolumn() -> None:
    """真多栏论文（过半页命中）必须仍路由 docling。"""
```

**风险**：中。改变聚合口径会影响所有 PDF 的路由。**必须先跑全语料对比**（见 §6 验收）。

---

### L3：信号分级 + 探测降级为「建议 + 告知」

**落点**：`ubt/adapters/factory.py:73`（`_resolve_pdf_adapter`）、`ubt/core/assess.py`、CLI

**核心变更**：把 `needs_docling` 拆成两级。

```python
# 概念示意（非最终实现）
@dataclass(frozen=True)
class PDFRoutePlan:
    primary_engine: str
    # 新增：区分"自动路由"与"仅建议"
    auto_route: bool          # True=可靠信号，直接路由；False=仅建议
    advisory: str | None      # 给用户看的建议文案
    ...
```

**分级规则**：

| 信号 | auto_route | 行为 |
|---|---|---|
| 扫描件 | `True` | 直接路由 Docling（失败模式是空书，必须兜住） |
| 多栏 / 公式 / 矢量 | `False` | 输出**建议**，默认仍走 pdfium |

**「告知」接到 `translate` 的前置提示**：用户直接跑 `translate` 时（而非两步 `assess` + `translate`）
就能看到：

```
⚠ 检测到本文档可能为多栏排版（抽样 4/5 页命中）。
  pdfium 快路径会把两栏文字逐行交错。建议加 --pdf-engine docling。
  当前使用 pdfium（快速路径）。如结果异常，请重跑并加该参数。
```

**已有的资产**：`ubt assess` 已经在展示这个信息（`ubt/cli/commands/assess.py:222`）：

```
│ 结构: 主引擎 docling · 文本层覆盖 100% · 扫描页占比 0%
```

问题只是它是**独立命令**，没接到 `translate` 的流程里。L3 要做的就是把这份告知**内联到 translate 的前置输出**。

**注意**：L3 会让"复杂 PDF 默认走 pdfium"——**会退化多栏/公式文档**。
所以 L3 必须与 **L4 的预设档位**配合：`--preset high_fidelity`/`standard` 仍自动用 Docling，
只有明确要求"快/省资源"时才退回 pdfium。**默认预设的行为需要单独决策**（见 §5 开放问题）。

---

### L4：解耦 Docling 的「使用」与「资源」

**落点**：`ubt/adapters/pdf/docling_parser.py:663`（`extract_with_docling`）

**问题**：只要走 Docling 就开 CUDA，且解析后不释放（PyTorch 缓存 CUDA context），与本地 LLM 抢显存。

**方案 A（推荐，最小改动）**：新增设备开关。

```python
# ubt/core/config.py
#: Docling 加速器设备：'auto'（默认，有 CUDA 就用）、'cpu'（不占显存）、'cuda'。
docling_device: str = "auto"
```

```python
# ubt/adapters/pdf/docling_parser.py（extract_with_docling 内）
options.accelerator_options.device = docling_device   # 替代隐式的 "auto"
```

用法：

```bash
UBT_DOCLING_DEVICE=cpu uv run ubt translate book.pdf --provider index-translate
```

**方案 B（更主动）**：检测到本地 LLM provider 时自动降级。

```python
# 若 provider 的 base_url 指向 127.0.0.1/localhost，且 docling_device == "auto"
# → 自动切 cpu，并打印告知
```

**推荐先做 A**（简单、可预测、用户可控）；B 可作为后续增强，但要注意"自动"再次引入隐式行为 —— 
与本设计"显式优先"的原则有张力，需谨慎。

**测试**：

```python
def test_docling_device_override_reaches_pipeline_options() -> None:
    """UBT_DOCLING_DEVICE=cpu 必须写进 accelerator_options.device。"""
```

**风险**：低。默认 `auto` 保持现有行为，不改变任何现有用户的路径。

---

### L5（可选）：让预设真正控制 `pdf_engine`

**落点**：`ubt/core/presets.py:64`

```python
@dataclass(frozen=True)
class PresetPolicy:
    ...
    pdf_engine: str = "auto"        # 新增字段

PRESETS[Preset.FAST]    = PresetPolicy(..., pdf_engine="pdfium")
PRESETS[Preset.PREVIEW] = PresetPolicy(..., pdf_engine="pdfium")
# high_fidelity / standard 保持 auto（质量优先）

PRESET_ENGINE_FIELDS = (..., "pdf_engine")   # 加入字段表
```

**注意**：`ubt/adapters/factory.py:85` 已有"强制 pdfium 但探测建议 docling"的告警逻辑，
所以行为是安全的（会 `logger.warning` 提示公式会降级）。

**这是缺陷 D 的直接修复**，但**需要产品决策**（见 §5）。

---

## 5. 已定决策（2026-10-10）

产品决策已定，记录如下（取代初稿的开放问题）：

| # | 问题 | 决策 | 理由 |
|---|---|---|---|
| Q1 | `pdf_engine` 默认值 | **`pdfium`** | 快路径无 torch/CUDA 依赖，不与本地 LLM 抢显存 |
| Q2 | `high_fidelity`/`standard` 预设 | **`auto`**（质量优先探测路由） | 选这两个预设本身就是对质量的授权 |
| Q3 | `preview`/`fast` 预设 | **`pdfium`**（强制快路径） | "快/零重型依赖"名实相符 |
| Q4 | 扫描件 | **始终逃逸到 docling** | 这是**能力边界**，不是偏好：pdfium 对扫描件抽 0 块 |

**关键洞察**：扫描件与"多栏/公式"是两类不同信号。

- 多栏/公式/矢量 = **偏好**：pdfium 能跑，只是质量差 → 尊重用户选择
- 扫描件 = **能力边界**：pdfium 完全读不了（0 块）→ 必须自动逃逸

所以 `pdfium` 模式不是"无条件用 pdfium"，而是"用 pdfium，除非它做不到"。

### 5.1 三层优先级

```
CLI --pdf-engine  >  预设 pdf_engine  >  config 默认 (pdfium)
```

实测验证（`2609.20519v1.pdf`，多栏信号）：

| 命令 | 实际引擎 |
|---|---|
| 默认（无 flag） | `pdfium` |
| `--preset standard` | `docling` |
| `--preset fast` | `pdfium` |
| `--preset fast --pdf-engine docling` | `docling` |
| 扫描件（任意模式） | `docling` |

---

## 6. 落地路线图

### 阶段 0：基线数据（必须先做）

**在改任何代码前**，跑全语料建立"当前路由"基线：

```bash
uv run python -c "
from pathlib import Path
from ubt.adapters.pdf.engine_selector import inspect_pdf_route_plan
import glob
for f in sorted(glob.glob('/home/kyle/Downloads/*.pdf')):
    pl = inspect_pdf_route_plan(Path(f))
    print(f'{Path(f).name:44} {pl.primary_engine:8} multi={pl.has_multicolumn} formula={pl.has_formulas} vec={pl.has_vector_diagrams}')
"
```

**已知基线（2026-10-10 实测）**：21 篇中 2 篇 pdfium、19 篇 docling；其中 **7 篇仅因 formula 信号**进 docling：

```
20261006-182205-...-2609.22978v1_bilingual.pdf
20261007-042355-491c135b-book_bilingual.pdf
2609.22978v1.pdf
BabelDOCv0.6.3.pdf
Understanding_KV_Cache_Handbook_02.pdf
Understanding_KV_Cache_Handbook_02_zh-CN.pdf
book.pdf
```

**L1/L2 修复后，这 7 篇的路由应重新评估**（预期其中若干改走 pdfium）。

### 阶段 1：L1 + L2（纯 bug 修复，独立可发）—— ✅ 已完成

- [x] L1：`_NAME_INITIAL_RE` + 剔除逻辑
- [x] L1：单元测试（人名缩写排除 + 真碎片护栏）
- [x] L2a：`PROBE_SAMPLE_FRACTIONS` 末页改 `0.92`
- [x] L2b：聚合改「≥2 页信号」（`PROBE_MIN_SIGNAL_PAGES = 2`）
- [x] L2：单元测试（单页不翻转 + 真多栏仍路由 + 两页信号）
- [x] 跑全语料对比，确认修复效果与无回归
- [x] 更新 `CALIBRATION` 登记

**验收结果**：`2609.22978v1` 全篇最高 debris 从 0.382 降到 0.067，不再被判 `has_formulas=True`。
路由从 19/21 docling 降到 13/21；翻转的 6 篇经逐页渲染核实均为误判修复
（参考文献页、图页、单页公式），**无真实回归**（`book.pdf` 数学教材保留在 docling）。

### 阶段 2：L4（解耦资源，独立可发）—— ✅ 已完成

- [x] `docling_device` 配置字段 + 环境变量映射
- [x] `extract_with_docling` 写入 `accelerator_options.device`
- [x] 测试 + 文档

**验收结果**：`UBT_DOCLING_DEVICE=cpu` 时 Docling 真实转换完成，`nvidia-smi` 显示显存
**零增长**（3407 → 3407 MiB）。

> 注：`ubt doctor` 展示该设置尚未做，可归入阶段 3。

### 阶段 3：L3（信号分级 + 告知）

- [ ] `PDFRoutePlan` 增加 `auto_route` / `advisory`
- [ ] `_resolve_pdf_adapter` 按分级决策
- [ ] `translate` 前置输出建议文案
- [ ] 测试 + 文档

**验收**：`translate` 一份多栏论文（默认配置）时，打印建议但**不**自动加载 Docling。

### 阶段 3：L3 + L5（路由语义 + 告知）—— ✅ 已完成

- [x] `pdf_engine` 加入 `PRESET_ENGINE_FIELDS`
- [x] 默认 `pdf_engine` 改 `pdfium`
- [x] `high_fidelity`/`standard` → `auto`；`preview`/`fast` → `pdfium`
- [x] 扫描逃逸：`pdfium` 模式下扫描件自动转 docling
- [x] `factory._resolve_pdf_adapter` 重写（区分偏好 vs 能力边界）
- [x] `PDFRoutePlan.has_scan` 新增
- [x] 告知：强制 pdfium 但探测建议 docling 时 `logger.warning`
- [x] CLI `--pdf-engine` / `--preset` 帮助文本更新
- [x] `ubt doctor` 文案更新
- [x] 测试（预设路由 + 扫描逃逸 + 显式覆盖）

**验收结果**（`2609.20519v1.pdf`，多栏信号）：

| 命令 | 实际引擎 |
|---|---|
| 默认（无 flag） | `pdfium` |
| `--preset standard` | `docling` |
| `--preset fast` | `pdfium` |
| 扫描件（任意模式） | `docling` |

### 阶段 4：修 `textgeom` 逐字形 bug（默认 pdfium 的前提）—— ✅ 已完成

把默认改成 `pdfium` 后暴露了一个**既有 bug**（不是本次重构引入的，`git stash` 对比
改动前后均为 224/226）：

`ubt/adapters/pdf/textgeom.py` 在"每字符一个 rect"的 PDF 上产出 99% 碎裂文本：

```
pdfium 原始 get_text_range : 'LOOPCD Introduction\r\n2\r\nLooped Transformers offer...'
textgeom 重建后            : 'd O O C 1 n t r o d u c t o n L o o pe d T a n s...'
```

**两个叠加的 bug**：

1. **`_drop_scatter_debris` 丢真字符**：`_is_debris_rect` 用 `高<5pt 或 宽<3pt`
   判定碎片，而逐字形页的字母 rect 恰好 3–5pt（`L`=3.13×4.87，`I`=0.73×4.87），
   于是 61% 的真实文本被当"矢量碎片"删除。修复：加一道**中位高**门 ——
   真 marker cloud 的字形 ~1pt（散点），逐字形正文 ~5pt（小写字母），
   只有中位高 < 2pt 的页才当 cloud。
2. **`_glue_run` 在字形间插空格**：`" ".join` 把 `LOOPCD` 拼成 `L O O P C D`。
   修复：识别"字形行"（单字符片段占比 > 20%），按**字形间隙**拼接 ——
   间隙 > 0.4×中位字形高才插空格。

**校准**：`_GLYPH_ROW_GAP_RATIO` 用词级 ground truth 扫描（19/20 锚点词命中，
对比 0.5 的 15/20）。

**验收结果**（`~/Downloads` 20 篇语料，逐字形页碎片率）：

| 文档 | 修复前 | 修复后 |
|---|---|---|
| `2610.02185v1.pdf` | 0.963 | **0.208** |
| `Pi-Durable-Technical-Manual.pdf` | 0.991 | **0.215** |
| `BabelDOCv0.6.3.pdf`（CJK） | 1.000 | 0.728（ASCII 词已正确：`KV`/`LLM`/`PagedAttention`） |
| `Understanding_KV_Cache_Handbook_02_zh-CN.pdf` | 0.766 | 0.680 |

端到端：`2610` 的 pdfium 正文块从 **224/226 碎裂降到 83/586**，
且样例已可读：`'a LoopCD reads an earlier state through b Accuracy at full depth'`。

> **重要**：旧的 `auto` 路由把这 4 篇全部送去 docling，恰好掩盖了这个 bug。
> 这也是为什么必须先修 textgeom、再切默认 pdfium。

---

## 6.1 已定规格（用户原则推导，无需再问）

用户的原则：**默认必须交付体验最好；用户不懂参数；只有高级用户才调参**。
由此直接得出下表，没有真正的开放决策：

| 入口 | 引擎 | 推导 |
|---|---|---|
| 默认（无 flag） | `auto` | 实测反转：pdfium 对表格/公式文档碎片化到不可交付 |
| `high_fidelity` / `standard` | `auto` | 这两个预设就是"要质量"的授权 |
| `preview` / `fast` | `pdfium` | 排除法：docling 只由显式或 high_fidelity/standard 触发 |
| `--pdf-engine docling` | `docling` | 显式最高优先 |
| 扫描件（任意模式） | `docling` | **能力边界**：pdfium 抽 0 块，不是偏好 |

**实测补充**（支持默认 pdfium 是合理的）：pdfium 的双栏阅读顺序**可靠** ——
真实文档 `2101.03961v3` 与合成双栏均输出正确的"左栏完再右栏"顺序。
早先渲染图里看到的交错乱码是 `textgeom` bug（已修），不是 pdfium 的能力问题。
pdfium 默认的真实代价是：表格不重建、公式不结构化（靠 `math_guard` 兜底不丢）。

残余限制（可接受）：figure 标签文字不翻译；极少数词间空格丢失
（`RelatedWork`）—— gap 法的固有极限。

### 6.1.1 默认引擎：`auto`（实测反转）

初稿主张默认 `pdfium`，已实现并**被实测推翻**。用户用真实文档（arXiv 2610.02185）
跑默认参数，产物不可交付；根因就是默认改成了 `pdfium`。已回退为 `auto`。

| 指标 | `pdfium`（错误默认） | `docling`（`auto` 会选） |
|---|---|---|
| 总块数 | **1801** | **350** |
| 表格块 | 0 | **19** |
| 标题 | `'d L eco n'` 碎片 | 完整 ✅ |
| 可翻译块碎裂>30% | 201/1796 | 19/230 |
| 产物违规 | 23 | 1 |
| `blocked_human` | 244 | 39 |
| `【待人工审校】` 标记 | **233** | 38 |

**结论**：对表格/公式/多栏文档，`pdfium` 不是"质量略差"，而是"碎片化到不可交付"。
默认必须交付体验最好（用户不懂参数），所以是 `auto`。

### 6.1.2 `CodeFormulaV2`（公式 VLM）何时启动

`auto` 下依次判断（`docling_parser.py:120`）：

1. `formula_enrichment == 'off'` → 不启动（公式用源图裁剪，VLM 冗余）
2. 文档被判 `formula_heavy` → 不启动（矢量公式保留，VLM 冗余）
3. 无 CUDA/MPS → 不启动
4. 否则 → 启动

**注意**：docling 一跑就带的两个**判别式视觉模型**（`docling-layout-heron` 版面、
`TableFormer` 表格）与 VLM 开关**无关** —— 2610 的 19 个表格就来自 TableFormer。
用户感受到的"质量好"来自这两个模型，不是 VLM。

**盲点**：这个决策依赖 `content_flags`，而它用 `formula_debris_share` ——
L1 修的就是该函数的人名缩写误判。修前参考文献页会误判 `formula_heavy=True`
而**错误地关闭** VLM。

### 6.1.3 显存冲突的真正机制（实测）

**docling 是惰性加载的**：

| 情况 | GPU 增量 |
|---|---|
| 构建 converter（不 convert） | **0 MiB** |
| 缓存命中（不 convert） | **0 MiB** |
| 真正 `convert()` | **379.8 MiB** |

所以"之前能一起跑"是因为：**之前翻译的文档 docling 缓存已存在** → 不 `convert()` → 0 GPU。
新文档首次解析才会占用，与本地 LLM 抢显存。

### 6.1.4 Bug B：中文量词误报（已修）

`一` 在量词前是中文的不定冠词（`一种` = "a kind"），但 `normalize_for_numeric_matching`
把它折成 `1`，"发明数字"检查于是误报：

```
源: 'We introduce a training-free framework'   ← 无数字
译: '我们提出了一种无需训练的框架'
  → 误报 "Unauthorized numeric tokens: ['1']"
```

**36/39 个隔离块源于此。** 修复：`_CN_ARTICLE_QUANTIFIER_RE` 把"裸冠词"加进授权集
（`一` 后跟量词且前无其他数字字符），真计数（`其实有 1 项`）仍被抓住。

### 6.1.5 Bug C：隔离占位符被画进 PDF（已修）

`BLOCKED_HUMAN` 块的 target 是包裹**源文**的 HTML 标记，不是译文。
`_overlayable` 只看 `skip_translate`，于是把 `【待人工审校】` + 源文画到页面上
（233 个标记出库）。修复：`_overlayable` 对 `BLOCKED_HUMAN` 返回 `False`，
让块留在源图层 —— 源页原样透出，符合"未审校内容不得出库"。

### 6.1.6 最终验收（同一文档，默认参数）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| `【待人工审校】` 标记 | 233 | **0** |
| 契约违规 | 23 | **0** |
| `blocked_human` | 244 | 17 |
| 标题 | 碎裂 | 完整 ✅ |

---

## 7. 验收与测试

### 7.1 必须新增的回归测试

| 测试 | 钉住的行为 |
|---|---|
| `test_formula_debris_excludes_name_initials` | L1：人名缩写不触发 |
| `test_formula_debris_still_catches_real_debris` | L1：真碎片仍触发 |
| `test_probe_ignores_single_reference_page` | L2：单页参考文献不翻转 |
| `test_probe_majority_still_routes_multicolumn` | L2：真多栏仍路由 |
| `test_docling_device_override_reaches_pipeline_options` | L4：设备开关生效 |
| `test_advisory_route_does_not_load_docling` | L3：建议不等于自动加载 |

### 7.2 交付契约的盲区（建议同时修）

§2.4 暴露了一个**独立的**问题：**交付契约不检查"文字对不对"，只检查"有没有丢"**。

```
heading  'Research environmen g (( g'    ← 明显是交错垃圾
INFO     Delivery contract: [PASS]
```

这不是本次重构的直接目标，但**值得单独记一笔**：可以增加一个"块内重复/交错"检测信号。
建议作为后续 issue，不在本次范围。

### 7.3 全语料回归

```bash
uv run ubt verify --corpus corpus --run --require-all
```

关注 `corpus/cases.json` 里的 `max_source_kept` / `min_delivered_ratio` 是否仍满足。

---

## 8. 风险与回滚

| 风险 | 缓解 |
|---|---|
| L2 改聚合口径导致多栏文档退化 | 阶段 0 基线对比；L2b 保留"过半"而非"全无" |
| L3 让复杂 PDF 默认退化 | 与 L5 预设配合；`high_fidelity`/`standard` 仍自动用 Docling |
| L1 正则误伤真公式 | 护栏测试（真碎片必须仍触发） |
| L4 默认值改变现有行为 | 默认 `auto`，与现状一致 |

**回滚**：每层独立提交，可单独 revert。L1/L2 是常量与正则，回滚零风险。

---

## 9. 与其它文档的关系

| 文档 | 关系 |
|---|---|
| [`PDF_INGEST_AND_VLM_ROUTING.md`](PDF_INGEST_AND_VLM_ROUTING.md) | **活文档**，描述当前系统。本文档落地后需同步更新 |
| [`MANUAL_E2E_INTEGRATION_TEST_SPEC.md`](MANUAL_E2E_INTEGRATION_TEST_SPEC.md) | 端到端验收步骤，本设计的新行为需补充用例 |
| [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md) | 测试分层约定，新增测试须遵循 |

---

## 10. 一页总结

**问题**：Docling 的自动加载权，让一个不可靠的启发式分类器有权加载 2GB CUDA 模型，
与本地 LLM 抢显存，产出 500；且它误判（参考文献页当公式页）、它说谎（`--preset fast` 承诺零依赖）、
它的错误还是静默的（契约 PASS 但输出是垃圾）。

**方案**：把"自动加载"降级为"建议 + 告知"，保留可靠信号（扫描件）的自动路由，
并解耦"用不用 Docling"与"占不占 GPU"两个维度。

**原则**：检测应**说话**，不应**动手**。这与本仓库 VLM 扫描兜底默认 OFF 的先例一致。

**落地**：L1/L2 修 bug（独立可发）→ L4 解耦资源（独立可发）→ L3 分级告知 → L5 预设（待决策）。
