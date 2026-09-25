# 端到端文档 VLM 引入评估(WeVisDoc + LightOnOCR-2 横评)

> **文档类型**:技术调研 / 模型引入决策支持
> **日期**:2026-09-20 · **基线 commit**:`0760175` 之后的当前 main
> **评估对象**:① 腾讯微信视觉团队开源文档解析模型 **WeVisDoc**(2B / 4B,基于 Qwen3-VL,Apache-2.0,整页图片 → Markdown + LaTeX 公式 + HTML 表格,端到端路线);② **(v2 增补)LightOnOCR-2-1B**(LightOn,法国,1B,mistral3 血统,Apache-2.0,同范式端到端,arXiv 2601.14251)
> **修订记录**:**v2(2026-09-20)** 增补 §10 LightOnOCR-2-1B 横评候选(**一手来源直核**:HF 模型卡全文 + arXiv 摘要 + 官方博客,证据等级高于 WeVisDoc 的二手转述)与 §11 "可插拔多引擎/多模型"战略评估。
> **信息来源**:WeVisDoc 部分——小互 AI 解读站文章 №1579(https://best.xiaohu.ai/article/wevisdoc/,2026-09-20 抓取)——**二手解读,数据为其转述的 WeVisDoc 技术报告口径,未直接核对论文与权重**;权重可得性/HF 页面未验证。LightOnOCR-2 部分——一手(https://huggingface.co/lightonai/LightOnOCR-2-1B 、 https://arxiv.org/abs/2601.14251 、 https://huggingface.co/blog/lightonai/lightonocr-2 )。
> **姊妹文档**:`PDF_OXIDE_ADOPTION_ASSESSMENT_2026-09-19.md`(库替代评估;已随 `62fcd75` 删除,历史见 `git show 62fcd75^:docs/PDF_OXIDE_ADOPTION_ASSESSMENT_2026-09-19.md`)。两评估严格互补:pdf_oxide 动的是 born-digital 提取链路,WeVisDoc 动的是扫描/VLM 转写链路。
> **状态**:评估完成,**默认建议 = 模型本体挂触发器不引入,方法论移植立即执行**;v2 起候选池加入 LightOnOCR-2-1B(§10),门 1 基准改为**三方排位赛**。§11 回答上位问题:"这类模型作为 UBT 可插拔槽位的用户自选引擎"这一战略是否成立——**结论:成立,架构已备好插槽,但必须配"能力声明 + 准入分级 + 默认自动路由"三件套**。本文档仅为决策支持,未改动任何源码。
> **2026-09-22 跟进**:门 1 三方排位**未启动**(`vlm/registry` 无新增驱动,无 WeVisDoc/LightOn 痕迹);M-1 能力声明未实现;`formula_witness` 灰区 VLM 层缺口仍在(docstring 自述未实现);"VLM 禁供坐标"规则未破例。**本文全部结论至今无一被代码推翻,也无一被执行。**

---

## 1. TL;DR 决策摘要

**核心问题**:WeVisDoc 是否有必要引入 UBT?

**答案**:

| 层面 | 结论 |
|---|---|
| born-digital 主链路 | ❌ **无关**。UBT 主路径处理自带文本层的 PDF,不需要 OCR 模型;anchored overlay 引擎的命根子是行级坐标,而 WeVisDoc 输出整页 Markdown 无 bbox(§5 方案 C 详述) |
| 扫描页转写链路 | ⚠️ **候选,但先过基准门**。UBT 的 driver 插件面(`vlm/registry.py` + `VlmDriver` 契约)使接入成本极低(一个 sidecar 配置或 ~100 行 driver),但 WeVisDoc 的强项在公式/表格,**文字识别与阅读顺序在端到端组均非第一**(Real 赛道 TextEdit 第 4)——对以文字为主的书籍恰是其短板 |
| 公式 witness 灰区 | ⚠️ **真缺口、非现成解**。`formula_witness.py` 自述"单字形替换(V_fb→V_h)检不出,需要 VLM 灰区层,此处未实现"——这是 WeVisDoc 强项能命中的真实空白,但整页输出无法直接回答局部问题,需 crop 级推理设计 |
| 方法论移植 | ✅ **无条件净收益,零依赖**。其"按 文字/表格/公式 分通道测残差"的做法与 UBT §5.2(pdf_oxide 可见文本丢失,已实证)的难点逐字同构,可直接搬进黄金基线与 witness 打分(§5 方案 D) |

**一句话**:WeVisDoc 是一篇"训练方法论 > 模型本身"的工作;对 UBT 而言模型本体是**条件性可选项**(扫描语料基准门后定),方法论是**免费的立即收益**。

**v2 增补预告**:LightOnOCR-2-1B(§10)坐同一生态位但证据链高一档(一手核证、OlmOCR-Bench 83.2 公开可复算、数据集开源)——门 1 从"是否引入 WeVisDoc"升级为"**deepseek-ocr / WeVisDoc / LightOnOCR-2 三方排位**";§11 将问题升维到"端到端文档 VLM 作为 UBT 用户自选可插拔引擎"的战略评估。

---

## 2. WeVisDoc 画像(带 UBT 视角的解读)

- **形态**:Qwen3-VL-2B/4B 底座 + 两阶段训练的端到端文档解析模型;输入整页图,输出含 LaTeX(`$$` 包裹)与 HTML `<table>` 的 Markdown;vLLM / Transformers 部署;PDF 需先逐页转图。
- **强项**(PureDocBench 端到端组,三赛道):FormulaCDM 第 1、TableTEDS 第 1。
- **弱项(对 UBT 是决定性的)**:
  - Clean/Digital/Real 的 **TextEdit 分别第 3/3/4**(最好 0.151/0.198/0.298,WeVisDoc 0.213/0.241/0.336);**ROEdit(阅读顺序)第 3/3/并列 3**。综合分第一纯靠公式表格两项拉动——文章原文警告:"以文字为主的文档(合同、报告**、书籍**),应该看文字编辑距离那一项"。UBT 的业务恰是书籍。
  - 无区域/行级坐标输出 → 与 UBT 的 bbox 驱动架构(§3)天然不整。
- **失败模式警示(案例 10/12)**:真实拍摄页存在**幻觉**(输出跑到别家文档内容)与**字符级残留错误**(1a→la,结构对字不对)——这两类正是 UBT witness 体系要防的,引入后它只能当**被验证者**,不能当**验证者**。
- **数据缺口**:训练覆盖中英繁+多语混排,但**无按语言拆分的分数**;UBT 主战场是英↔中,基准必须在 UBT 自己语料上跑。
- **选型口径**(文章转述):干净/电子页 2B 够用,拍照翻拍多再上 4B;4B 相对 2B 的增益几乎全在 Real 赛道(+3.48)。

---

## 3. UBT 扫描/VLM 基础设施全景(代码核证)

结论:"有没有接入位"——**有,且是现成插件槽**。

1. **驱动契约**(`ubt/adapters/pdf/vlm/types.py:34`):`VlmDriver` Protocol——`recognize(image, page_size_pt, scale) -> PageTranscript`,注释明确"**no geometry promises**":检测器驱动(rapidocr)可带 `measured_box`;**LLM-VLM 驱动必须留 None**("hallucinated coordinates are worse than none",P9 锚定规则)。DeepSeek-OCR driver 即此模式(`measured_boxes = False`)。WeVisDoc 天然落入同一档。
2. **注册表**(`vlm/registry.py`):`register_driver(name, factory)`,已注册 `rapidocr` / `deepseek-ocr` / `sidecar` / `cloud`(两种)。**fail-closed**:未知名字抛 KeyError。新增驱动零侵入。
3. **锚定融合**(`vlm/anchor.py:56` + `transcribe.py:159`):VLM 行与 pdfium 文本行做 NFKC 归一的**包含匹配**(`_best_match`),产出 `AnchoredLine{provenance: "pdfium" | "pdfium+proofread" | "vlm-measured"}` 与 `AnchorStats{matched, vlm_only, pdfium_only}`——**双见证结构内建**。文本层存在时 VLM 只做 proofread(保 pdfium 几何),这是测试钉死的行为(`test_vlm_core.py`:proofread 丢插入行、recognition 拒收幻觉几何)。
4. **接入形态**:现成 sidecar 体系(`deploy/docker/ocr-sidecar/server.py`,FastAPI,`POST /v1/ocr` + `/health`,UBT 侧 `SidecarOcrDriver` 走 httpx + `PageBBoxResolver` 坐标归一)。WeVisDoc 官方支持 vLLM serving → 可仿该模板做 GPU sidecar。
5. **触发路由**:`page_profiler.PageKind.SCAN_IMAGE` → `engine_selector` → 扫描引擎链;`docling_parser.vlm_fallback_missing_pages`(docling_parser.py:661)在 docling 漏页时兜底调 VLM。
6. **公式线现状**(评估方案 B 的关键):
   - `formula_witness.py` docstring 自标定:检**布局级损伤**(一行源公式重排成两行、项丢失、9×9 数独级结构),**检不出单字形替换 V_fb→V_h**,"需要 gray-zone VLM tier,**not implemented here**";设计哲学是 fail-open + 零 token,**检出即换源图裁剪(lossless)**。
   - `page_profiler.formula_debris_share(text)`:现有公式残骸信号是**纯文本启发式**。
   - chapter-3 案例:68 个渲染公式的标定跑过,残留 flags 两条,均布局级。

---

## 4. 逐项对照:WeVisDoc 能力 × UBT 需求

| UBT 需求 | 现状 | WeVisDoc 对应 | 匹配度 |
|---|---|---|---|
| 扫描页文字转写 | rapidocr(检测框)/ DeepSeek-OCR-2(本地 4B,8GB VRAM 门槛)/ cloud VLM | 同类竞品,**文字维度不占优**(§2 弱项) | ⚠️ 待基准,先验不乐观 |
| 扫描页阅读顺序 | pdfium 几何 + `column_order` 启发式 | ROEdit 端到端组第 3,非第一 | ⚠️ 同上 |
| 公式结构识别(扫描/退化页) | 文本启发式 + 布局级 witness,**灰区 VLM 层缺位** | FormulaCDM 端到端组第 1 | ✅ 方向命中真实空白(但需 crop 化改造,方案 B) |
| 表格重建 | docling 兜底 + 空间表格检测器 | TableTEDS 端到端组第 1 | ✅ 有纸面优势,但表格主要喂 RAG/分析场景,非 UBT 主路径 |
| 行级 bbox(anchored 链路刚需) | pdfium/rapidocr 提供 | **不提供**(整页 Markdown) | ❌ 结构性不匹配,勿入 anchored 提取侧 |
| 逐行置信度 | `VlmLine.confidence` 字段现成 | 自回归 VLM 无原生置信度 | ❌ 缺口,risk triage 只能靠跨引擎分歧度替代 |
| 幻觉防护 | witness 多引擎面板 + `visual_gate` | 自身是幻觉案例主角 | ❌ 只能当被验证者 |

---

## 5. 候选接入方案与判定

### 方案 A:扫描转写第四 driver(witness 面板成员)—— **条件推荐,先过基准门**

- **形态**:vLLM 起 WeVisDoc-2B(干净页)/4B(退化页)→ 仿 `ocr-sidecar` 加 markdown→`VlmLine` 拆分端点;UBT 侧零改动(`SidecarOcrDriver` 已支持)或注册专用 driver(`measured_boxes=False`,走 proofread 锚定)。
- **判定门槛(§7 门 1)**:在 UBT 自有扫描语料上 vs `deepseek-ocr` 基线,**主指标必须是 TextEdit 与 ROEdit**(书籍=文字为主),公式/表格作加分项;幻觉率>0 即出局(只能进 witness 面板不能独任)。**v2:候选池加入 LightOnOCR-2-1B(§10),门 1 输出改为三方排位而非单项准入。**
- **成本**:接入 ≈ 0.5-1 天(插件面现成);**GPU 运行成本不熟→列为已知缺口**——`models/` 与 `deploy/` 中未见 WeVisDoc 权重/基准痕迹,DeepSeek-OCR 已有 `MIN_FREE_BYTES=8GiB` 的运维先例,WeVisDoc-2B 量级开销应可类比,但实际吞吐与显存需 §7 门 1 实测。

### 方案 B:公式灰区 VLM tier(补 formula_witness 的自述空白)—— **真需求,但 WeVisDoc 非显然最优**

- 需求真实:witness 检不出字形级替换,现策略是保守换源图(公式不翻,语义安全)。若加 VLM 复核层,正确形态是**对单个公式 crop 做"源图 vs 渲染输出图"的视觉等价判定**——整页 markdown 模型不吐局部判定,需要 crop 级 prompt 工程或换用逐公式识别模型(如 dots.mocr 类专模型,论文里 TextEdit 最好、OmniDocBench 表格超 WeVisDoc 的 1B 模型)。
- **判定**:先做**专模型调研**再决定是否用 WeVisDoc;WeVisDoc 整页推理对"两图对比"不是量身设计。此方案独立于 WeVisDoc 成立,不应绑死。

### 方案 C:阅读顺序/版面理解增强 —— **不推荐**

无 bbox 输出 + ROEdit 非第一 + anchored 链路对几何的苛刻要求(`_assert_unrotated` 级别的一致性执念),强行接入需为它发明新数据通路,违背"adapter-owns-geometry"契约。

### 方案 D:方法论移植(零依赖)—— **无条件推荐,立即执行**

1. **分类别残差度量**:把"整页编辑距离掩盖局部损伤"的教训落到 UBT——黄金基线(`metrics.golden.json`)与 extraction_witness 打分改为 **text-edit / (1−TEDS) / (1−CDM) 三通道分别对齐计分**。这直接服务于 `PDF_OXIDE_ADOPTION_ASSESSMENT`(已随 `62fcd75` 删除,见 git 历史) §6 保真度判据的落地:pdf_oxide"静默丢公式"担忧已在 v3.2 §5.2 定谳为**可见题注跨四 API 丢失**,与 WeVisDoc 团队"公式没了整页分看不出来"完全同构,分通道指标让这类丢失可测。UBT 已有 `formula_debris_share`(page_profiler:170)这一按类别信号先例,是自然延伸。
2. **三引擎分层复核**:MinerU/PaddleOCR-VL/dots.mocr 交叉标注、"两同一异→图像验证、三异→更强模型+人工"的分流,可映射进 UBT 的 triage 成本策略(与 witness 面板现有 matched/vlm_only/pdfium_only 计数合流)。
3. **合成数据双编译模式**(DOM→图 + DOM→精确标注):若 UBT 未来做公式/表格的模型侧评测集,这是免标注 ground truth 的现成配方。

---

## 6. 明确不做

1. **不替换 docling**(§6 硬边界 1 维持):docling 在 UBT 是模型级版面兜底,WeVisDoc 同为模型但赌注不同(端到端生成 vs 结构化解析),替换是产品决策不是库选择。
2. **不进 born-digital 链路**:有文本层的页面用 OCR 模型是倒退;pdf_oxide 评估文档的结论不受本文影响。
3. **不引入为"第四套权重"**:同机已有 DeepSeek-OCR(9GB 口径)+ rapidocr ONNX,再叠 2B/4B 需基准门先于安装。
4. **不让任何 VLM 提供坐标**:P9 锚定规则(`types.py:13-16` docstring)不因新模型松动——WeVisDoc driver 的 `measured_boxes=False` 是硬性。

---

## 7. 决策门与验证方案

**门 1(方案 A,先跑再说)**:
```text
语料:book2-nistir4653-artifact.pdf(真实扫描件)+ 合成扫描/拍照退化页若干
     (UBT 规矩:合成退化仿 WeVisDoc Stage I 退化流水线思路,不收第三方原图)
基线:deepseek-ocr driver、rapidocr driver
指标(按 WeVisDoc 自己的口径,顺序即权重):
  1. TextEdit(逐行对齐,主) 2. ROEdit(主)
  3. 公式 CDM / 表格 TEDS(加分)
  4. 幻觉率=输出内容与页面无对应关系的页数占比(>0 出局)
  5. 显存/吞吐/整书延迟
通过条件:TextEdit 与 ROEdit 均不劣于基线劣化 ≤5% 且幻觉率=0 → 注册 sidecar driver 进 witness 面板
```

**门 2(方案 B)**:dots.mocr / 专公式模型 vs WeVisDoc-crop-prompt 两方案在 chapter-3 的 68 个公式上对比字形级损伤召回(`formula_witness` 已知残留 flags 作种子负例);WeVisDoc 只有在无需新推理形态时才选它。

**门 3(方案 D,无门槛)**:直接开工——黄金基线三通道指标拆分 + §8.1 实验 E 用分通道指标重跑。

---

## 8. 风险登记册

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| W1 | 信息二手:结论基于解读文章,论文/权重/基准表未直核 | **中** | 任何门 1/2 启动前先核 HF 权重可得性与论文原表;文章数据仅作方向 |
| W2 | 文字/阅读顺序非第一却因"综合分第一"被引入 → 主路径劣化 | 高 | 门 1 主指标定为 TextEdit+ROEdit,综合分不做准入 |
| W3 | 幻觉流入计费前链路 | 高 | 只入 witness 面板(被交叉验证位),禁止独任转写;`needs_review` 标记沿用 |
| W4 | GPU 运维面扩张(第二套本地权重),CI/无 GPU 环境不可用 | 中 | sidecar 形态隔离主包;`registry` fail-closed + `probe_effective_driver` 已有降级路径 |
| W5 | 无置信度 → triage 无法按行分流 | 低 | 用跨引擎分歧度(anchor_stats)替代行级置信度,现成 |
| W6 | 中文 CJK 书籍精度无公开拆分数据 | 中 | 门 1 语料必须含中文扫描页 |

---

## 9. 决策清单

- [ ] **立即(零依赖)**:方案 D-1 落地——黄金基线/witness 打分按 text/table/formula 三通道拆分;用它重跑 pdf_oxide §5.2 提取保真 A/B(顺手推进姊妹文档 §6 判据)。
- [ ] **本周**:直核 WeVisDoc 技术报告与权重可得性(消 W1)。
- [ ] **排期后议**:门 1 基准(语料、脚本、与 deepseek-ocr A/B);通过才做 sidecar driver。
- [ ] **独立调研**:方案 B 先比 dots.mocr 类专模型,勿默认 WeVisDoc。
- [ ] **门 1 扩容(v2)**:候选从"是否引入 WeVisDoc"改为 **deepseek-ocr / WeVisDoc / LightOnOCR-2 三方排位**,LightOn 侧先直核 HF 权重下载可用性与 transformers>=5 升级面(§10 红线 ③)。
- [ ] **战略层(v2,§11)**:认可"用户自选可插拔引擎"方向,按 M-1..M-3 落地(能力声明 → 按语言实测上架分级 → 默认路由 + 进账本的专家覆盖;准入矩阵测试由门 1 承担)。
- [ ] **不做**:方案 C、docling 替换、born-digital 引入、VLM 坐标例外。

---

## 10. v2 横评候选:LightOnOCR-2-1B(一手来源直核)

> 来源:HF 模型卡全文(https://huggingface.co/lightonai/LightOnOCR-2-1B)+ arXiv 摘要(https://arxiv.org/abs/2601.14251)+ 官方博客(https://huggingface.co/blog/lightonai/lightonocr-2 ),2026-09-20 抓取。**证据等级高于本文 WeVisDoc 部分(后者为二手转述,风险 W1)。**

### 10.1 画像

- **出身**:LightOn(法国公司,BPI Scribe 项目资助),作者 3 人(Taghadouini / Cavaillès / Aubertin);1B 参数,mistral3 架构血统,Apache-2.0;HF 823 likes,含 demo、开源训练数据集(`LightOnOCR-mix-0126`)与 bbox-bench。
- **定位**:端到端"页图 → 干净、自然顺序文本",README 原话 "**without relying on brittle pipelines**"——**与 WeVisDoc 同一范式**(小参数端到端文档 VLM,抛弃检测+识别+排序管线)。
- **家族**:6 变体——RLVR 精炼版(主模型)/ base(微调底座)/ bbox / bbox-base / 两个 `-soup`(checkpoint averaging + task-arithmetic merging 的鲁棒性变体)。
- **语言**:en fr de es it nl pt sv da **zh ja**(11 语)。
- **自报数字**(全部一手但单方):OlmOCR-Bench **83.2 ± 0.9** 自称 SOTA,超 Chandra-9B 1.5+ 分而参数量 1/9;速度 3.3× Chandra / 1.7× OlmOCR / 5× dots.ocr / 2× PaddleOCR-VL-0.9B / **1.73× DeepSeekOCR**;单 H100 5.71 pages/s(≈493k 页/日,<$0.01/千页)。**关键差别:OlmOCR-Bench 有开源 harness,83.2 这个数第三方可复算**——WeVisDoc 的 PureDocBench 名次目前只有二手口径。
- **训练配方**:大规模高质量**蒸馏 mix**(强项自述:扫描件、法语、科学 PDF、LaTeX 处理);**RLVR 后训练**(bbox 用 IoU 奖励);transformers **>=5.0** 原生类(`LightOnOcrForConditionalGeneration`),vLLM OpenAI 兼容 serving。

### 10.2 与 WeVisDoc 的思路异同(UBT 视角)

| 轴 | WeVisDoc(微信) | LightOnOCR-2(LightOn) | 对 UBT 的含义 |
|---|---|---|---|
| 范式 | 端到端文档 VLM | 同 | 同一条赛道的直接竞品 |
| 下注 | **版面理解/视觉定位**(名字即 WeChat **Vis**ion **Doc**) | **纯转写质量 + 吞吐** | 书籍正文扫描场景,LightOn 的下注更贴 |
| 坐标输出 | 无文本级 bbox | bbox 变体**只预测内嵌图片区域**,非字符/行级几何 | **对 P9 锚定规则(§3.1)两者同档**:都坐不了 anchored 定位位,只能坐整页转写/见证位 |
| 后训练 | 监督式视觉指令微调 | **RLVR(可验证奖励)+ merging** | 方法论可移植(10.4),这是 WeVisDoc 没有的技术路线 |
| 证据 | 二手转述(W1) | 一手 + 可复算基准 + 开放数据集 | 门 1 若只测一家,先测 LightOn |
| 中文 | 训练含中英繁,**无按语言拆分分数** | zh 在列,但 mix 卖点是法语/arXiv/扫描件,基准是英语系 | **同一条一票否决项(W6 对两家都成立):门 1 语料必须含中文扫描页** |

### 10.3 生态位判定

1. **方案 A(扫描转写 driver)**:同生态位候选,且纸面更强(公式/扫描自述 standout、比 UBT 现有 deepseek_driver 小且自称快 1.73×、1B bf16 显存低于 `MIN_FREE_BYTES=8GiB` 门槛、sidecar 契约零摩擦)。进门 1 三方排位。
2. **方案 B(公式灰区 tier)**:比 WeVisDoc **更对口**——"old scans with math"与 arXiv 类是其自述提升最大的类目,恰打 `formula_witness` 字形级盲区;但仍需 crop 级推理设计,不豁免门 2。
3. **红线**:① 蒸馏 mix → 与 witness 面板其他模型 VLM 可能**共享教师、相关误差**,作"独立第二意见"的价值打折,入面板前先做损伤页分歧度实测;② 中文证据薄(10.2);③ transformers>=5.0 整条依赖升级的连带面(deepseek_driver 已有多处 compat shim,此依赖线本就敏感);④ 自回归无逐行置信度(与 WeVisDoc 同,W5 适用)。

### 10.4 借鉴价值(独立于是否引入)

- **RLVR → QE 塔**:UBT 手握一批**天然可验证奖励源**(术语命中/Aho-Corasick、词接缝、公式结构完整性、Typst 可编译性)——"用可确定性验证的奖励做小模型后训练"在 LightOn 兑现了质量,对应到 UBT 是长期选项:用免费层 QE 信号作奖励微调小型 QE/重排模型,替代部分 LLM-judge 计费调用。
- **`-soup` merging 白捡鲁棒性**:checkpoint averaging + task arithmetic,零推理开销;UBT 任何未来自训/微调组件可直接抄。
- **开放数据集与 bbox-bench**:方法论完全可审计,也是"合成退化数据"(方案 D-3)的公开参照实现。

---

## 11. 战略评估:"端到端文档 VLM 这类模型 = UBT 里用户自选的可插拔引擎"是否成立

**用户命题**:UBT 面向主流语言的工业级翻译,适配多格式、长/短文档,应把这类模型做成可插拔位,并让用户自选适合其语言的引擎/模型。

**判定:✅ 成立,且 UBT 架构正是按这个思路长出来的——但"自选"必须转译成三条纪律,否则选项变成给用户甩责任。**

### 11.1 可行性证据(代码核证:插件位不是设想,是已交付的两套现成机制)

1. **翻译主链路已有完整先例**:`ubt/core/router/` 一整套——`capabilities.py` 的 **`ModelProfile` 声明式能力档案**(docstring 原话:"Decouples model execution logic from brittle name matching, **enabling enterprise self-hosted models to configure behavior declaratively**":prompt 策略/输出抽取策略/参数兼容性逐项声明)+ `registry` + `provider` + `pricing` + `rate_limiter`。config 层已有 base_url 多供应商适配(Google/Anthropic/OpenAI 兼容)。**"用户自选适合其语言的翻译模型"在 LLM 侧已是既成设计。**
2. **扫描/VLM 转写层有契约、缺档案**:`VlmDriver` Protocol(`vlm/types.py`:`name`/`measured_boxes`/`recognize`)+ `registry` fail-closed + rapidocr/deepseek/sidecar/cloud 四驱动并存 + `SidecarOcrDriver` 外部进程隔离。**插槽已存在,WeVisDoc/LightOnOCR-2 都只是"往槽里放什么"的问题。**
3. **路由与见证已存在**:`page_profiler.PageKind` → `engine_selector` → 引擎链;pdf_oxide 评估文档已提议 `--pdf-engine oxide` 灰度枚举。**自选机制(枚举配置 + 默认路由)有 template 可循。**

### 11.2 三条纪律(缺一即退化为"用户背锅")

**M-1 能力声明(Capability Manifest),选型界面禁止靠模型名猜。**
把翻译层 `ModelProfile` 的模式横推到扫描层:每个驱动注册时声明——语言覆盖矩阵(非"支持多语"一句空话)、输入类型(印刷/扫描/拍照/退化页)、输出物(纯文本/Markdown/HTML 表格/图片区域/行级框)、`measured_boxes` 布尔、显存/吞吐画像、许可证、**锁定版本**。UI 与 `engine_selector` 只消费声明字段出选项;`VlmDriver` 现有字段(name/measured_boxes)即原型,扩成完整 manifest 是增量而非重构。

**M-2 准入分级(货架规则):过了门 1 也不等于能上架。**
分级:`default`(自动路由权威,质量地板由黄金基线+账本对账兜底,责任在 UBT)/ `advanced`(用户显式 opt-in,UI 标注差异)/ `experimental`(仅灰度/内部)。**"按语言适配"的工程落点=按语言列的基准分数是硬上架数据**——WeVisDoc 无按语言拆分(W6)、LightOnOCR-2 中文证据薄(§10.2),则两者中文货架不开,无论综合分多高。这条规则同时防住"综合分第一诱导入主路径"(W2 的泛化版)。

**M-3 默认自动路由 + 专家覆盖,覆盖行为进账本。**
普通用户按 PageKind × 语言 × 硬件画像由 selector 路由;**expert 覆盖不是静默开关**——选定引擎写入 ledger 元数据,使 QE 分数、`needs_review`、计费对账均可追溯"这个输出是哪条引擎链产的",出问题可复盘到引擎粒度。

### 11.3 反方与成本(如实记账)

- **组合爆炸**:引擎 × 语言 × 文档类型 × 格式。纪律是把爆炸吸收在**准入侧**(门 1 语料矩阵:每个想上架的模型×语言格都要有分),而非摊到用户侧。
- **运维面**:每个本地权重都是显存/磁盘配额、transformers/推理栈兼容 shim 维护费(§10.3 红线③)、CI 无 GPU 环境的降级路径(W4 已有 sidecar 隔离答案)。多一个模型 = 多一份永久维护负债,上架宁慢勿滥。
- **许可证**:本文两模型皆 Apache-2.0 干净,但"模型货架"会引来 CC-BY-NC 类权重——`test_license_guard` 需把模型权重头计入围裙(lora/adapter 同样适用)。
- **风险对冲**:模型不像库有 issue tracker 可观察,退役更干脆——manifest 里 `deprecated` 态 + 黄金基线保留其通道,拔除成本天然低(sidecar 隔离已保证)。

### 11.4 结论

命题成立,UBT 的护城河恰好让它可以把这件事做得比竞品负责任:竞品给用户一个模型下拉框就完事;UBT 可以给下拉框配上**声明式能力档案(M-1)+ 按语言实测上架(M-2)+ 可追溯路由(M-3)+ 账本对账兜底**。行动序:先 M-1(扫描层 manifest 扩字段,~1 天)→ 门 1 三方排位顺带产 M-2 首批数据 → M-3 随门 1 通过后落地。**不为任何单一模型(含 WeVisDoc/LightOn)修改这三条规则本身。**

---

## 附:证据索引

| 证据 | 来源 |
|---|---|
| WeVisDoc 全部数据 | 小互 №1579 文章(转述技术报告,未核一手) |
| VlmDriver 契约与"禁幻觉坐标"规则 | `ubt/adapters/pdf/vlm/types.py:1-51`(本会话全文读) |
| 注册表/降级探针 | `ubt/adapters/pdf/vlm/registry.py` |
| proofread/recognition 锚定语义(测试钉死) | `vlm/anchor.py`、`vlm/transcribe.py:125-204`、`tests/unit/test_vlm_core.py` 符号表 |
| formula_witness 能力边界("V_fb→V_h 检不出、灰区 VLM 层未实现") | `ubt/adapters/pdf/formula_witness.py` docstring(全文) |
| sidecar 接入形态 | `deploy/docker/ocr-sidecar/server.py`(`POST /v1/ocr`、/health)、`vlm/drivers/sidecar_driver.py` |
| DeepSeek-OCR 运维先例(8GB floor、prompt 模式) | `vlm/drivers/deepseek_driver.py:30-34,136-167` |
| 扫描路由 | `page_profiler.py`(PageKind.SCAN_IMAGE、formula_debris_share)、`docling_parser.py:661` |
| §8.1 悬案同构性 | `PDF_OXIDE_ADOPTION_ASSESSMENT_2026-09-19.md` v3.2 §5.2(定谳;文档已随 `62fcd75` 删除,见 git 历史) |
| **(v2)LightOnOCR-2 画像/数字/变体/语言表** | HF 模型卡 raw README 全文一手抓取(2026-09-20) |
| **(v2)RLVR/IoU 奖励/蒸馏 mix/merging 训练配方** | arXiv 2601.14251 摘要一手抓取;OlmOCR-Bench 83.2±0.9 与速度倍数:官方博客正文 |
| **(v2)翻译层可插拔先例(`ModelProfile` 声明式能力档案)** | `ubt/core/router/capabilities.py:24-`(docstring 明言"enabling enterprise self-hosted models to configure behavior declaratively")+ `router/{registry,provider,pricing,rate_limiter}.py` |

> **报告结束**。v1 生成于 2026-09-20;v2 同日增补 §10/§11。核心结论两句话:**模型本体挂门观察,门 1 升级为 deepseek/WeVisDoc/LightOn 三方排位(主指标=文字与阅读顺序+按语言列);而"端到端文档 VLM 作为用户自选可插拔引擎"的战略成立——它的工程形态就是 M-1 能力声明 + M-2 按语言实测上架 + M-3 默认路由与进账本的专家覆盖,不为任何单一模型松动这三条规则。**
