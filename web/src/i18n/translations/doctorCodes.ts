/**
 * Localized copy for the Engine Doctor checklist.
 *
 * The engine's `ubt doctor` prints English, and the console's panel is the same
 * checklist. Rather than translating prose on the client (which cannot parse it),
 * each check ships a stable `code` plus the values interpolated into its English
 * sentence, so the console renders its own text from this catalogue.
 *
 * Keys are the engine's codes verbatim (`ubt/cli/commands/doctor.py`). A code
 * missing here falls back to the engine's English copy — honest, if untranslated —
 * never a raw key. That fallback is why this file can lag the engine safely.
 */

export interface DoctorCodeTables {
  /** Section headings, keyed by `group_code`. */
  groups: Record<string, string>
  /** Check names, keyed by `code`. */
  names: Record<string, string>
  /** Diagnoses, keyed by `detail_code`. */
  details: Record<string, string>
  /** Remediations, keyed by `fix_code`. */
  fixes: Record<string, string>
}

export const doctorCodesZh: DoctorCodeTables = {
  groups: {
    credentials: '凭据与模型',
    storage: '存储与隐私',
    quality: '质量与可选层',
    rendering: '渲染与工具链',
  },
  names: {
    api_key: 'API 密钥',
    base_url: '接口地址',
    live_probe: '在线端点探测',
    models: '模型',
    cost_pricing: '计费定价',
    ledger_dir: '账本目录',
    ledger_permissions: '账本权限',
    translation_memory: '翻译记忆',
    disk_space: '磁盘空间',
    sqlite_wal: 'SQLite WAL 账本',
    page_images: '页面图像',
    qe: '质量评估',
    batch_api: 'Batch API',
    pdf_engine: 'PDF 引擎',
    ocr_engine: 'OCR 引擎',
    render: '渲染',
    typst: 'Typst',
    typst_math: 'Typst 数学',
    cjk_fonts_render: 'CJK 字体（渲染）',
    poppler: 'Poppler',
    pdf_raster: 'PDF 光栅化（oxide）',
    pillow: 'Pillow',
    cjk_font_metrics: 'CJK 字体（度量）',
  },
  details: {
    'api_key.missing': '未配置',
    'api_key.ok': '已配置',
    'api_key.local_not_needed': '本地端点无需密钥',
    'base_url.ok': '{base_url}',
    'live_probe.model_online': '{base_url} 可达（{rtt_ms}ms）；草稿模型「{model}」已确认在线',
    'live_probe.model_missing':
      '{base_url} 可达（{rtt_ms}ms），但在 {count} 个模型中未找到草稿模型「{model}」。可用：{preview}',
    'live_probe.reachable': '{base_url} 可达（{rtt_ms}ms）',
    'live_probe.auth_failed': '端点 {base_url} 返回 {status}（鉴权失败）',
    'live_probe.models_404': '端点 {base_url} 可达（{rtt_ms}ms），但 /models 返回 404',
    'live_probe.http_error': '端点 {base_url} 返回 HTTP {status}（{rtt_ms}ms）',
    'live_probe.unreachable': '无法连接 {base_url}：{error}',
    'models.missing': '未配置草稿模型',
    'models.ok': '草稿={draft} 修复={repair}',
    'cost_pricing.local_endpoint': '本地端点 {base_url} — 不产生费用，模型名无需价格表条目',
    'cost_pricing.unpriced':
      '{models} 没有价格表条目 — 成本报告为未知，且 --budget-usd 无法触发',
    'cost_pricing.ok': '草稿与修复模型均已定价',
    'ledger_dir.writable': '可写：{path}',
    'ledger_dir.not_writable': '不可写：{path}（{error}）',
    'ledger_permissions.exposed': '{count} 个含书籍正文的文件对 group/other 可读，例如 {example}',
    'ledger_permissions.ok': '仅所有者可读，位于 {db_dir}',
    'ledger_permissions.windows_acl': 'Windows 通过 NTFS ACL 管理权限，POSIX 权限位在此不可度量；账本目录位于当前用户目录下时，仅本账户可读。',
    'translation_memory.ok': '共享 tm.sqlite，模糊匹配阈值 {threshold}',
    'translation_memory.disabled': '已禁用',
    'disk_space.critical':
      '{db_dir} 仅剩 {free_gb} GB 可用（共 {total_gb} GB）— 临时渲染产物将失败',
    'disk_space.low': '{db_dir} 剩余 {free_gb} GB（共 {total_gb} GB）— 大型 PDF 编译空间不足',
    'disk_space.ok': '{db_dir} 剩余 {free_gb} GB（共 {total_gb} GB）',
    'disk_space.unknown': '无法查询 {db_dir} 的磁盘用量：{error}',
    'sqlite_wal.ok': 'journal_mode={mode}，写入锁已获取并正常释放',
    'sqlite_wal.failed': '{db_dir} 上的 WAL 锁探测失败：{error}',
    'page_images.disabled':
      '已关闭外发（UBT_ALLOW_PAGE_UPLOAD=false）：书页不会离开本机；cloud/vlm OCR 拒绝启动，视觉修复降级为文本',
    'page_images.allowed': '已允许；当前生效路径：{routes}',
    'qe.tiered': '歧义类 + {sample} 通过抽样，judge={judge}',
    // Same text with the provider block named. Picked in ``localizeDoctorCheck``
    // only when ``{provider}`` is non-empty; without it the sentence above is
    // used unchanged, so English (which has no details table) is unaffected.
    'qe.tiered.via': '歧义类 + {sample} 通过抽样，judge={judge}（provider 块 {provider}）',
    'qe.heuristic': '零 token 规则评分',
    'batch_api.enabled': '已启用 — 端点须实现 /v1/files + /v1/batches（离线不探测）',
    'batch_api.disabled': '已禁用',
    'pdf_engine.docling_ok': 'docling 可导入',
    'pdf_engine.docling_missing':
      '未安装 docling — UBT_PDF_ENGINE=docling 是显式请求，PDF 作业会以可操作的错误失败，而不会静默丢弃表格、图与公式',
    'pdf_engine.pdfium': 'pypdfium2 快速路径（内置依赖）；扫描版文档仍会自动转到 docling',
    'pdf_engine.auto_docling': 'auto = docling 版面解析（无条件）',
    'pdf_engine.auto_missing':
      'auto 选用 docling，但未安装 — PDF 作业回退到几何 pdfium 解析器，重建的 PDF 会丢弃表格、图与公式',
    'pdf_engine.registered': '已注册的引擎：{engine}',
    'pdf_engine.unregistered': '「{engine}」不是已注册的引擎',
    'ocr_engine.sidecar': 'sidecar 服务在线（http://localhost:8765）',
    'ocr_engine.rapidocr': '本地 rapidocr + onnxruntime 可用',
    'ocr_engine.rapidocr_no_backend':
      'rapidocr 已安装但缺少推理后端（onnxruntime）— 扫描版 PDF 无法 OCR',
    'ocr_engine.cloud_ok': 'cloud/vlm 外发已启用（{mode}）',
    'ocr_engine.cloud_blocked': 'ocr_mode=「{mode}」但 allow_page_upload=False — 扫描页无法外发',
    'ocr_engine.none':
      '没有可用的本地 OCR 引擎（sidecar 离线，rapidocr/onnxruntime 未安装）— 扫描版 PDF 将失败',
    'ocr_engine.off': 'ocr_mode=off',
    'render.ok': 'dual_mode={dual_mode}，pdf=typst-reflow',
    'typst.ok': 'PDF 高保真重排渲染可用',
    'typst.missing': 'PATH 中未找到 — PDF 渲染会在第 6 阶段失败',
    'typst_math.ok': 'Typst 原生公式微排版可用',
    'typst_math.unavailable': 'Typst 不可用；公式将保留为源图',
    'cjk_fonts_render.ok': '{count} 个可用 CJK 字族（例如 {example}）',
    'cjk_fonts_render.missing': '渲染器看不到任何支持 CJK 的字体 — zh/ja/ko PDF 会渲染成豆腐块',
    'cjk_fonts_render.no_probe': '无探测响应（typst/fc-list 均不可用）',
    'cjk_fonts_render.no_module': '字体探测模块不可用',
    'poppler.ok': 'pdftotext 可用',
    'poppler.missing':
      '缺少：{missing} — 这同时会关闭 fail-closed 产物一致性校验（目标语言缺失、页面尺寸、图像数量），未翻译的书可能被标记为已完成',
    'pdf_raster.ok': 'pdf_oxide 进程内页面光栅化',
    'pdf_raster.missing': 'pdf_oxide 是基础依赖但无法导入 — 像素门禁失效',
    'pillow.ok': '像素启发式可用',
    'pillow.missing': '未安装 — 像素空白/黑块检查将跳过',
    'cjk_font_metrics.ok': '{path}',
    'cjk_font_metrics.missing': '{error}',
    'cjk_font_metrics.no_module': '字体度量模块不可用',
  },
  fixes: {
    'api_key.missing': 'export UBT_LLM_API_KEY=sk-...（或在 ubt.toml 设 api_key / 传 --api-key）',
    'models.missing': '设置 --draft-model/--repair-model、UBT_DRAFT_MODEL，或 --provider',
    'live_probe.model_missing': '可从以下选择：{preview}',
    'live_probe.auth_failed': '检查 {provider} 的凭据',
    'live_probe.unreachable': '检查端点地址、网络连接，或本地模型服务状态',
    'cost_pricing.local_endpoint': '仅当本地 URL 后面是付费网关时才设 UBT_BILL_LOCAL_ENDPOINT=1',
    'cost_pricing.unpriced':
      '把每百万 token 价格加入 ubt/resources/prices.toml（旧版回退：pricing.py 中的 MODEL_PRICES_USD_PER_MTOK）',
    'ledger_permissions.exposed': 'chmod -R go-rwx {roots}',
    'disk_space.critical': '释放 UBT_DB_DIR 所在卷的磁盘空间',
    'disk_space.low': '为大批量编译预留至少 5 GB 空闲空间',
    'sqlite_wal.failed': '检查文件系统锁支持（NFS/网络挂载可能导致 SQLite WAL 锁失败；请用本地磁盘）',
    'page_images.allowed': '设 UBT_ALLOW_PAGE_UPLOAD=false 可让页面保留在本机',
    'pdf_engine.docling_missing': 'uv sync --extra pdf',
    'pdf_engine.auto_missing': 'uv sync --extra pdf，或设 UBT_PDF_ENGINE=pdfium 走快速路径',
    'ocr_engine.rapidocr_no_backend': 'uv sync --all-extras（或 uv pip install onnxruntime）',
    'ocr_engine.cloud_blocked': '开启页面外发，或安装本地 rapidocr/sidecar',
    'ocr_engine.none': 'uv sync --all-extras（或 docker run -d -p 8765:8765 ubt-ocr-sidecar）',
    'typst.missing': '安装它（brew install typst / cargo install typst-cli），或改用文本输出格式（.md / .docx / .epub）',
    'cjk_fonts_render.missing': '安装 CJK 字体（Linux 发行版包如 noto-fonts-cjk；macOS 可 brew install --cask font-noto-cjk；Windows 一般自带雅黑）',
    'poppler.missing': '安装 poppler 并加入 PATH（apt / brew / Windows 官方构建均可）',
    'pdf_raster.missing': '重装项目环境（uv sync）',
    'pillow.missing': '安装 pillow',
  },
}

export const doctorCodesEn: DoctorCodeTables = {
  groups: {
    credentials: 'Credentials & models',
    storage: 'Storage & privacy',
    quality: 'Quality & optional tiers',
    rendering: 'Rendering & toolchain',
  },
  names: {
    api_key: 'API key',
    base_url: 'Base URL',
    live_probe: 'Live Endpoint Probe',
    models: 'Models',
    cost_pricing: 'Cost pricing',
    ledger_dir: 'Ledger dir',
    ledger_permissions: 'Ledger permissions',
    translation_memory: 'Translation memory',
    disk_space: 'Disk space',
    sqlite_wal: 'SQLite WAL Ledger',
    page_images: 'Page images',
    qe: 'QE',
    batch_api: 'Batch API',
    pdf_engine: 'PDF engine',
    ocr_engine: 'OCR engine',
    render: 'Render',
    typst: 'Typst',
    typst_math: 'Typst Math',
    cjk_fonts_render: 'CJK fonts (render)',
    poppler: 'Poppler',
    pdf_raster: 'PDF raster (oxide)',
    pillow: 'Pillow',
    cjk_font_metrics: 'CJK font (metrics)',
  },
  // English needs no table: the engine's own copy is already English and more
  // precise than a paraphrase would be. An empty map means "use the engine text",
  // which is exactly the fallback path a missing code takes.
  details: {},
  fixes: {},
}
