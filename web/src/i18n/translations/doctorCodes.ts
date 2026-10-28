/**
 * Localized copy for the system-check checklist.
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
    credentials: '模型连接',
    storage: '存储与隐私',
    quality: '质量检查',
    rendering: '文件处理',
    deep: '深度环境与离线',
  },
  names: {
    api_key: '访问密钥',
    base_url: '服务地址',
    live_probe: '服务连接',
    models: '模型配置',
    cost_pricing: '费用计算',
    ledger_dir: '任务数据目录',
    ledger_permissions: '任务数据权限',
    translation_memory: '翻译记忆',
    disk_space: '可用空间',
    sqlite_wal: '运行数据存储',
    page_images: '页面图像隐私',
    qe: '质量检查',
    pdf_engine: 'PDF 解析',
    ocr_engine: '文字识别',
    render: 'PDF 排版',
    typst: '排版组件',
    typst_math: '公式排版',
    cjk_fonts_render: '中文字体',
    poppler: 'PDF 文本校验',
    pdf_raster: '页面预览',
    pillow: '图像检查',
    cjk_font_metrics: '中文字体度量',
    gpu_acceleration: 'GPU 硬件加速',
    system_memory: '系统物理内存',
    local_inference: '本地大模型端点',
    ghostscript: 'Ghostscript 优化工具',
    typst_version: 'Typst 编译器版本',
    cjk_coverage: '中文字体覆盖度',
    offline_assets: '离线资源完整性',
  },
  details: {
    'api_key.missing': '未配置',
    'api_key.ok': '已配置',
    'api_key.local_not_needed': '本地端点无需密钥',
    'base_url.ok': '{base_url}',
    'live_probe.model_online': '服务可达（{rtt_ms}ms）；模型「{model}」在线',
    'live_probe.model_missing':
      '服务可达（{rtt_ms}ms），但未找到模型「{model}」。可用：{preview}',
    'live_probe.reachable': '服务可达（{rtt_ms}ms）',
    'live_probe.auth_failed': '服务返回 {status}，请检查访问密钥',
    'live_probe.models_404': '服务可达（{rtt_ms}ms），但不支持模型列表查询',
    'live_probe.http_error': '服务返回 HTTP {status}（{rtt_ms}ms）',
    'live_probe.unreachable': '无法连接服务：{error}',
    'models.missing': '未配置草稿模型',
    'models.ok': '草稿={draft} 修复={repair}',
    'cost_pricing.local_endpoint': '本地模型服务 — 不产生费用',
    'cost_pricing.unpriced':
      '{models} 没有价格信息 — 成本将显示为未知，预算上限无法生效',
    'cost_pricing.ok': '草稿与修复模型均已定价',
    'ledger_dir.writable': '可正常写入：{path}',
    'ledger_dir.not_writable': '无法写入：{path}（{error}）',
    'ledger_permissions.exposed': '{count} 个含书籍正文的文件对 group/other 可读，例如 {example}',
    'ledger_permissions.ok': '仅所有者可读，位于 {db_dir}',
    'ledger_permissions.windows_acl': 'Windows 通过 NTFS ACL 管理权限，POSIX 权限位在此不可度量；账本目录位于当前用户目录下时，仅本账户可读。',
    'translation_memory.ok': '相似度阈值 {threshold}',
    'translation_memory.disabled': '已禁用',
    'disk_space.critical':
      '仅剩 {free_gb} GB 可用（共 {total_gb} GB）— 任务将无法生成交付文件',
    'disk_space.low': '剩余 {free_gb} GB（共 {total_gb} GB）— 大型 PDF 可能空间不足',
    'disk_space.ok': '剩余 {free_gb} GB（共 {total_gb} GB）',
    'disk_space.unknown': '无法查询可用空间：{error}',
    'sqlite_wal.ok': '运行数据可正常写入',
    'sqlite_wal.failed': '运行数据无法写入：{error}',
    'sqlite_wal.network_fs':
      '{db_dir} 位于网络文件系统 ({fs_type})，SQLite WAL 强依赖本地 POSIX 共享内存与文件锁，在网络存储上易导致死锁或静默损坏',
    'page_images.disabled':
      '书页图像不会离开本机；需要图像外发的检查会自动跳过。',
    'page_images.allowed': '已允许；当前生效路径：{routes}',
    'qe.tiered': '模型评分已开启；抽样比例 {sample}',
    // Same text with the provider block named. Picked in ``localizeDoctorCheck``
    // only when ``{provider}`` is non-empty; without it the sentence above is
    // used unchanged, so English (which has no details table) is unaffected.
    'qe.tiered.via': '模型评分已开启；抽样比例 {sample}（使用 {provider}）',
    'qe.heuristic': '规则检查，不调用模型',
    'pdf_engine.docling_ok': '高精度版面解析可用',
    'pdf_engine.docling_missing':
      '未安装高精度版面解析组件；需要该模式的任务会直接报错，而不是静默丢失表格、图与公式',
    'pdf_engine.pdfium': '快速解析可用；扫描版文档会自动改用高精度解析',
    'pdf_engine.auto_docling': '自动使用高精度版面解析',
    'pdf_engine.auto_missing':
      '自动模式需要高精度版面解析组件，但当前未安装；PDF 会退回基础解析，表格、图与公式可能丢失',
    'pdf_engine.registered': '解析方式：{engine}',
    'pdf_engine.unregistered': '「{engine}」不是可用的解析方式',
    'ocr_engine.sidecar': '本地 OCR 服务可用',
    'ocr_engine.rapidocr': '本地 OCR 可用',
    'ocr_engine.rapidocr_no_backend':
      'OCR 组件不完整，扫描版 PDF 暂时无法识别',
    'ocr_engine.vlm_ok': 'vlm 外发已启用',
    'ocr_engine.vlm_blocked': 'ocr_mode="vlm" 但 allow_page_upload=False — 扫描页无法外发',
    'ocr_engine.none':
      '没有可用的本地 OCR；扫描版 PDF 将无法处理',
    'ocr_engine.off': 'OCR 已关闭',
    'render.ok': 'PDF 重排渲染可用',
    'typst.ok': 'PDF 高保真重排渲染可用',
    'typst.missing': 'PDF 排版组件不可用；交付阶段会失败',
    'typst_math.ok': '公式排版可用',
    'typst_math.unavailable': '公式排版不可用；公式将保留为原图',
    'cjk_fonts_render.ok': '{count} 个可用 CJK 字族（例如 {example}）',
    'cjk_fonts_render.missing': '渲染器看不到任何支持 CJK 的字体 — zh/ja/ko PDF 会渲染成豆腐块',
    'cjk_fonts_render.no_probe': '无探测响应（typst/fc-list 均不可用）',
    'cjk_fonts_render.no_module': '字体探测模块不可用',
    'poppler.ok': 'PDF 文本校验可用',
    'poppler.missing':
      '缺少：{missing} — 交付一致性检查会被关闭，未翻译的书可能被误判为已完成',
    'pdf_raster.ok': '页面预览可用',
    'pdf_raster.missing': '页面预览组件不可用 — 视觉检查会被跳过',
    'pillow.ok': '像素启发式可用',
    'pillow.missing': '未安装 — 像素空白/黑块检查将跳过',
    'cjk_font_metrics.ok': '{path}',
    'cjk_font_metrics.missing': '{error}',
    'cjk_font_metrics.no_module': '字体度量模块不可用',
    'gpu_acceleration.cuda_ok': 'NVIDIA GPU 可用：{gpu_name}（剩余 {vram_free}GB / 共 {vram_total}GB）',
    'gpu_acceleration.metal_ok': 'Apple Silicon Metal (MPS) 硬件加速可用',
    'gpu_acceleration.low_vram': '显存较少（剩余 {vram_free}GB），运行大模型或密集 OCR 时可能受限',
    'gpu_acceleration.none': '未检测到独立 GPU；将使用 CPU 执行',
    'system_memory.ok': '物理内存充足：剩余 {ram_avail}GB / 共 {ram_total}GB',
    'system_memory.low': '物理内存偏紧（剩余 {ram_avail}GB），大文件排版可能存在压力',
    'system_memory.critical': '物理内存极低（剩余 {ram_avail}GB），解析大文件存在 OOM 风险',
    'local_inference.online': '本地模型服务在线（{engine} 位于 {url}）；模型「{model}」就绪',
    'local_inference.model_missing': '本地服务在线，但未找到模型「{model}」',
    'local_inference.unreachable': '无法连接配置的本地模型服务：{url}',
    'local_inference.none': '未检测到运行中的本地模型服务（Ollama / vLLM / llama.cpp）',
    'ghostscript.ok': 'Ghostscript {version} 可用，支持 PDF 高阶压缩与字体子集化',
    'ghostscript.missing': '未找到 Ghostscript；PDF 体积精简将降级为流重压缩',
    'typst_version.ok': 'Typst {version} 兼容（满足 >= 0.11.0 要求）',
    'typst_version.outdated': 'Typst {version} 版本过低；建议升级至 >= 0.11.0 以启用完整表格排版特性',
    'typst_version.missing': '未在系统 PATH 中找到 Typst 编译器',
    'cjk_coverage.complete': '中文字体完整覆盖：宋体 (Serif) 与黑体 (Sans) 均已就绪',
    'cjk_coverage.partial': '中文字体部分缺失：缺少 {missing_family} 字族',
    'cjk_coverage.missing': '未检测到中文字体 — 导出中文 PDF 会出现豆腐块',
    'offline_assets.ok': '核心离线资源验证通过（价格表、词典资源包等完整）',
    'offline_assets.missing': '缺失关键离线资源包文件：{missing_assets}',
  },
  fixes: {
    'api_key.missing': '在「系统」页的模型提供商中填写密钥',
    'models.missing': '在「系统」页为提供商填写模型名称',
    'live_probe.model_missing': '可从以下选择：{preview}',
    'live_probe.auth_failed': '检查 {provider} 的凭据',
    'live_probe.unreachable': '检查端点地址、网络连接，或本地模型服务状态',
    'cost_pricing.local_endpoint': '如果本地地址后是付费网关，请在服务端启用本地端点计费',
    'cost_pricing.unpriced':
      '在服务端为该模型补充价格信息',
    'ledger_permissions.exposed': 'chmod -R go-rwx {roots}',
    'disk_space.critical': '释放运行数据目录所在磁盘的空间',
    'disk_space.low': '为大批量编译预留至少 5 GB 空闲空间',
    'sqlite_wal.failed': '请确认运行数据目录位于支持文件锁的本地磁盘',
    'sqlite_wal.network_fs': '请将 DB_DIR 迁移至本地 NVMe/SSD 磁盘卷（网络挂载存在数据损坏风险）',
    'page_images.allowed': '关闭页面外发可让页面图像保留在本机',
    'pdf_engine.docling_missing': '在服务端安装 PDF 高精度解析组件',
    'pdf_engine.auto_missing': '安装高精度解析组件，或切换为基础解析模式',
    'ocr_engine.rapidocr_no_backend': '在服务端补全 OCR 组件',
    'ocr_engine.vlm_blocked': '开启页面外发，或配置本地 OCR',
    'ocr_engine.none': '在服务端安装本地 OCR，或启动 OCR 服务',
    'typst.missing': '在服务端安装 PDF 排版组件，或改用文本输出格式',
    'cjk_fonts_render.missing': '在系统中安装中文字体',
    'poppler.missing': '在服务端安装 PDF 文本校验组件',
    'pdf_raster.missing': '重新安装服务端依赖',
    'pillow.missing': '在服务端安装图像检查组件',
    'gpu_acceleration.none': '建议在带 GPU 的机器上运行以获得更快的本地推理性能',
    'gpu_acceleration.low_vram': '关闭占用显存的后台进程，或使用 4-bit 量化模型',
    'system_memory.low': '运行大部头翻译前关闭高内存占用的无关程序',
    'system_memory.critical': '释放系统内存或添加 Swap 交换分区以防进程被系统终止',
    'local_inference.model_missing': '在终端执行：ollama pull {model}',
    'local_inference.unreachable': '启动本地大模型服务或检查服务地址与端口',
    'ghostscript.missing': '在系统中安装 ghostscript',
    'typst_version.outdated': '升级 Typst 编译器至 >= 0.11.0',
    'typst_version.missing': '在系统中安装 Typst 编译器（>= 0.11.0）',
    'cjk_coverage.partial': '安装完整的宋体与黑体中文字体（如 fonts-noto-cjk）',
    'cjk_coverage.missing': '在系统中安装中文字体（如 fonts-noto-cjk / noto-fonts-cjk）',
    'offline_assets.missing': '重新安装 UBT 软件包以恢复 ubt/resources/ 资源文件',
  },
}

export const doctorCodesEn: DoctorCodeTables = {
  groups: {
    credentials: 'Credentials & models',
    storage: 'Storage & privacy',
    quality: 'Quality & optional tiers',
    rendering: 'Rendering & toolchain',
    deep: 'Deep offline & hardware',
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
    gpu_acceleration: 'GPU acceleration',
    system_memory: 'System memory',
    local_inference: 'Local inference engine',
    ghostscript: 'Ghostscript tool',
    typst_version: 'Typst version',
    cjk_coverage: 'CJK font coverage',
    offline_assets: 'Offline assets',
  },
  // English needs no table: the engine's own copy is already English and more
  // precise than a paraphrase would be. An empty map means "use the engine text",
  // which is exactly the fallback path a missing code takes.
  details: {},
  fixes: {},
}
