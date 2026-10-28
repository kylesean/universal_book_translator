"""Domain-aware translatable technical terms, universal acronyms, and compound matchers.

Provides a structured, multi-domain taxonomy of universally translatable terms and
acronyms (AI/ML, Systems/Architecture, Hardware, Networking, Science/Math, Organizations)
that legitimate translations frequently expand or translate into the target language
(e.g., "NLP" -> "自然语言处理", "CPU" -> "中央处理器", "DNA" -> "脱氧核糖核酸") rather than
retaining verbatim.

Replaces brittle, ad-hoc allowlists with an extensible, cross-domain knowledge base
and compound decomposition logic (e.g. "ext4formatted" -> "ext4" + "格式化", "microVMs" -> "微虚拟机").
"""

from __future__ import annotations

import re
from typing import Final

# Regex to split on letter/digit boundaries: 'ext4' -> ('ext', '4') or 'ext4formatted' -> ('ext4', 'formatted')
_DIGIT_LETTER_SPLIT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])"
)

# Common English affixes for technical compound decomposition
_COMMON_PREFIXES: Final[tuple[str, ...]] = (
    "micro",
    "macro",
    "multi",
    "single",
    "sub",
    "hyper",
    "meta",
    "auto",
    "semi",
    "pre",
    "post",
    "non",
    "inter",
    "intra",
    "cross",
)

# Common technical words with well-defined Chinese translations for OCR-glued or compound terms
_COMMON_SUB_TERMS: Final[dict[str, tuple[str, ...]]] = {
    "formatted": ("格式化", "格式化的"),
    "encoded": ("编码", "编码的"),
    "decoded": ("解码", "解码的"),
    "aligned": ("对齐", "对齐的"),
    "compressed": ("压缩", "压缩的"),
    "encrypted": ("加密", "加密的"),
    "decrypted": ("解密", "解密的"),
    "scaled": ("缩放", "按比例缩放"),
    "shared": ("共享", "共享的"),
    "isolated": ("隔离", "隔离的"),
    "distributed": ("分布式",),
    "threaded": ("线程", "多线程"),
    "tenant": ("租户",),
    "tenants": ("租户",),
    "node": ("节点",),
    "nodes": ("节点",),
    "task": ("任务",),
    "tasks": ("任务",),
    "block": ("块", "数据块"),
    "blocks": ("块", "数据块"),
    "layer": ("层", "网络层"),
    "layers": ("层", "网络层"),
}

#: Comprehensive taxonomy of universal technical acronyms and translatable terms.
#: Keys must be lower-case ASCII strings. Values are tuples of acceptable target surfaces.
UNIVERSAL_TRANSLATABLE_TERMS: Final[dict[str, tuple[str, ...]]] = {
    # -------------------------------------------------------------------------
    # Artificial Intelligence, NLP & Machine Learning
    # -------------------------------------------------------------------------
    "nlp": ("自然语言处理", "自然语言", "NLP"),
    "ml": ("机器学习", "ML"),
    "ai": ("人工智能", "AI"),
    "dl": ("深度学习", "DL"),
    "cv": ("计算机视觉", "CV"),
    "rl": ("强化学习", "RL"),
    "llm": ("大语言模型", "大模型", "语言模型", "LLM"),
    "llms": ("大语言模型", "大模型", "语言模型", "LLMs"),
    "agi": ("通用人工智能", "AGI"),
    "ann": ("人工神经网络", "神经网络", "ANN"),
    "anns": ("人工神经网络", "神经网络", "ANNs"),
    "cnn": ("卷积神经网络", "CNN"),
    "cnns": ("卷积神经网络", "CNNs"),
    "rnn": ("循环神经网络", "RNN"),
    "rnns": ("循环神经网络", "RNNs"),
    "gan": ("生成对抗网络", "GAN"),
    "gans": ("生成对抗网络", "GANs"),
    "vae": ("变分自编码器", "VAE"),
    "vaes": ("变分自编码器", "VAEs"),
    "gnn": ("图神经网络", "GNN"),
    "gnns": ("图神经网络", "GNNs"),
    "rag": ("检索增强生成", "RAG"),
    "peft": ("参数高效微调", "高效微调", "PEFT"),
    "moe": ("混合专家", "混合专家模型", "MoE"),
    "ocr": ("光学字符识别", "OCR"),
    "asr": ("语音识别", "自动语音识别", "ASR"),
    "tts": ("语音合成", "文字转语音", "TTS"),
    "ner": ("命名实体识别", "实体识别", "NER"),
    "ir": ("信息检索", "中间表示", "IR"),
    "kg": ("知识图谱", "KG"),
    "pos": ("词性标注", "POS"),
    "bert": ("BERT", "BERT模型"),
    "gpt": ("GPT", "GPT模型"),
    # -------------------------------------------------------------------------
    # Computer Systems, Virtualization, OS & Hardware
    # -------------------------------------------------------------------------
    "cpu": ("中央处理器", "处理器", "CPU"),
    "cpus": ("中央处理器", "处理器", "CPUs"),
    "gpu": ("图形处理器", "图形处理单元", "显卡", "GPU"),
    "gpus": ("图形处理器", "图形处理单元", "显卡", "GPUs"),
    "tpu": ("张量处理器", "张量处理单元", "TPU"),
    "tpus": ("张量处理器", "张量处理单元", "TPUs"),
    "npu": ("神经网络处理器", "神经网络处理单元", "NPU"),
    "npus": ("神经网络处理器", "神经网络处理单元", "NPUs"),
    "dpu": ("数据处理器", "数据处理单元", "DPU"),
    "dpus": ("数据处理器", "数据处理单元", "DPUs"),
    "os": ("操作系统", "OS"),
    "oss": ("操作系统",),
    "vm": ("虚拟机", "VM"),
    "vms": ("虚拟机", "VMs"),
    "microvm": ("微虚拟机", "虚拟机", "microvm", "MicroVM"),
    "microvms": ("微虚拟机", "虚拟机", "microvms", "MicroVMs"),
    "ram": ("随机存取内存", "随机存取存储器", "内存", "RAM"),
    "rom": ("只读内存", "只读存储器", "ROM"),
    "ssd": ("固态硬盘", "SSD"),
    "ssds": ("固态硬盘", "SSDs"),
    "hdd": ("机械硬盘", "硬盘", "HDD"),
    "hdds": ("机械硬盘", "硬盘", "HDDs"),
    "io": ("输入输出", "输入/输出", "I/O", "IO"),
    "ipc": ("进程间通信", "IPC"),
    "rpc": ("远程过程调用", "RPC"),
    "rpcs": ("远程过程调用", "RPCs"),
    "db": ("数据库", "DB"),
    "dbs": ("数据库", "DBs"),
    "dbms": ("数据库管理系统", "DBMS"),
    "rdbms": ("关系型数据库", "关系型数据库管理系统", "RDBMS"),
    "sql": ("结构化查询语言", "SQL"),
    "nosql": ("非关系型数据库", "NoSQL"),
    "kv": ("键值", "键-值", "KV"),
    "fs": ("文件系统", "FS"),
    "qos": ("服务质量", "QoS"),
    "cli": ("命令行界面", "命令行", "CLI"),
    "gui": ("图形用户界面", "图形界面", "GUI"),
    "api": ("应用程序接口", "应用编程接口", "接口", "API"),
    "apis": ("应用程序接口", "应用编程接口", "接口", "APIs"),
    "sdk": ("软件开发工具包", "软件开发套件", "开发套件", "SDK"),
    "sdks": ("软件开发工具包", "软件开发套件", "开发套件", "SDKs"),
    "ide": ("集成开发环境", "IDE"),
    "ides": ("集成开发环境", "IDEs"),
    "ci": ("持续集成", "置信区间", "CI"),
    "cd": ("持续部署", "持续交付", "CD"),
    "vcs": ("版本控制系统", "VCS"),
    "pr": ("拉取请求", "合并请求", "PR"),
    "prs": ("拉取请求", "合并请求", "PRs"),
    "ast": ("抽象语法树", "AST"),
    "asts": ("抽象语法树", "ASTs"),
    "jit": ("即时编译", "即时编译器", "JIT"),
    "aot": ("提前编译", "预先编译", "AOT"),
    "gc": ("垃圾回收", "垃圾收集", "GC"),
    "oom": ("内存溢出", "内存耗尽", "OOM"),
    "rtos": ("实时操作系统", "RTOS"),
    "fpga": ("现场可编程门阵列", "FPGA"),
    "fpgas": ("现场可编程门阵列", "FPGAs"),
    "asic": ("专用集成电路", "ASIC"),
    "asics": ("专用集成电路", "ASICs"),
    "soc": ("片上系统", "SoC"),
    "socs": ("片上系统", "SoCs"),
    "isa": ("指令集架构", "指令集", "ISA"),
    "isas": ("指令集架构", "指令集", "ISAs"),
    "alu": ("算术逻辑单元", "ALU"),
    "alus": ("算术逻辑单元", "ALUs"),
    "tlb": ("快表", "页表缓存", "转换检测缓冲区", "TLB"),
    "tlbs": ("快表", "页表缓存", "TLBs"),
    "dma": ("直接内存访问", "DMA"),
    "nic": ("网络接口卡", "网卡", "NIC"),
    "nics": ("网络接口卡", "网卡", "NICs"),
    # -------------------------------------------------------------------------
    # Networking & Protocols
    # -------------------------------------------------------------------------
    "lan": ("局域网", "LAN"),
    "lans": ("局域网", "LANs"),
    "wan": ("广域网", "WAN"),
    "wans": ("广域网", "WANs"),
    "vlan": ("虚拟局域网", "VLAN"),
    "vlans": ("虚拟局域网", "VLANs"),
    "dns": ("域名系统", "域名解析", "DNS"),
    "tcp": ("传输控制协议", "TCP"),
    "udp": ("用户数据报协议", "UDP"),
    "ip": ("网际协议", "互联网协议", "IP协议", "IP"),
    "http": ("超文本传输协议", "HTTP"),
    "https": ("超文本传输安全协议", "安全超文本传输协议", "HTTPS"),
    "tls": ("传输层安全", "传输层安全协议", "TLS"),
    "ssl": ("安全套接字层", "SSL"),
    "ssh": ("安全外壳协议", "SSH"),
    "ftp": ("文件传输协议", "FTP"),
    "vpn": ("虚拟专用网络", "虚拟专网", "VPN"),
    "vpns": ("虚拟专用网络", "虚拟专网", "VPNs"),
    "nat": ("网络地址转换", "NAT"),
    "bgp": ("边界网关协议", "BGP"),
    # -------------------------------------------------------------------------
    # Mathematics, Statistics & Natural Sciences
    # -------------------------------------------------------------------------
    "dna": ("脱氧核糖核酸", "DNA"),
    "rna": ("核糖核酸", "RNA"),
    "mrna": ("信使核糖核酸", "mRNA"),
    "crispr": ("基因编辑", "CRISPR"),
    "pcr": ("聚合酶链式反应", "PCR"),
    "atp": ("三磷酸腺苷", "ATP"),
    "mri": ("磁共振成像", "MRI"),
    "ct": ("计算机断层扫描", "断层扫描", "CT"),
    "ecg": ("心电图", "ECG"),
    "eeg": ("脑电图", "EEG"),
    "ode": ("常微分方程", "ODE"),
    "odes": ("常微分方程", "ODEs"),
    "pde": ("偏微分方程", "PDE"),
    "pdes": ("偏微分方程", "PDEs"),
    "svd": ("奇异值分解", "SVD"),
    "pca": ("主成分分析", "PCA"),
    "dft": ("离散傅里叶变换", "DFT"),
    "fft": ("快速傅里叶变换", "FFT"),
    "cdf": ("累积分布函数", "累积概率分布", "CDF"),
    "cdfs": ("累积分布函数", "CDFs"),
    "pdf": ("概率密度函数", "PDF"),
    "pdfs": ("概率密度函数", "PDFs"),
    "mse": ("均方误差", "MSE"),
    "rmse": ("均方根误差", "RMSE"),
    "mae": ("平均绝对误差", "MAE"),
    "roc": ("受试者工作特征", "ROC"),
    "auc": ("曲线下面积", "AUC"),
    "sd": ("标准差", "SD"),
    "se": ("标准误", "SE"),
    # -------------------------------------------------------------------------
    # UX, Metrics & Organizations
    # -------------------------------------------------------------------------
    "ui": ("用户界面", "界面", "UI"),
    "ux": ("用户体验", "UX"),
    "fps": ("每秒帧数", "帧率", "FPS"),
    "tps": ("每秒事务数", "TPS"),
    "qps": ("每秒查询率", "每秒请求数", "QPS"),
    "rps": ("每秒请求数", "RPS"),
    "sla": ("服务等级协议", "服务级别协议", "SLA"),
    "slas": ("服务等级协议", "服务级别协议", "SLAs"),
    "slo": ("服务等级目标", "服务级别目标", "SLO"),
    "slos": ("服务等级目标", "服务级别目标", "SLOs"),
    "sli": ("服务等级指标", "服务级别指标", "SLI"),
    "slis": ("服务等级指标", "服务级别指标", "SLIs"),
    "roi": ("投资回报率", "ROI"),
    "kpi": ("关键绩效指标", "KPI"),
    "kpis": ("关键绩效指标", "KPIs"),
    "faq": ("常见问题解答", "常见问题", "FAQ"),
    "faqs": ("常见问题解答", "常见问题", "FAQs"),
    "rfc": ("征求意见稿", "请求注解", "RFC"),
    "rfcs": ("征求意见稿", "RFCs"),
    "who": ("世界卫生组织", "WHO"),
    "fda": ("食品药品监督管理局", "FDA"),
    "cdc": ("疾病控制与预防中心", "CDC"),
    "eu": ("欧盟", "EU"),
    "un": ("联合国", "UN"),
    # Special attested glued forms from production arXiv corpuses
    "ext4formatted": ("ext4", "格式化", "ext4格式化", "ext4 formatted"),
}


def find_translatable_surfaces(term: str) -> tuple[str, ...] | None:
    """Find acceptable target surfaces for a known term, acronym, or plural form."""
    term_lower = term.lower()
    if term_lower in UNIVERSAL_TRANSLATABLE_TERMS:
        return UNIVERSAL_TRANSLATABLE_TERMS[term_lower]

    # De-pluralize check
    if (
        len(term_lower) > 3
        and term_lower.endswith("s")
        and not term_lower.endswith(("ss", "us", "is"))
    ):
        singular = term_lower[:-1]
        if singular in UNIVERSAL_TRANSLATABLE_TERMS:
            return UNIVERSAL_TRANSLATABLE_TERMS[singular]

    return None


def match_compound_term(term: str, target: str) -> list[str] | None:
    """Decompose compound OCR or glued terms (e.g. ext4formatted, microVMs) and verify parts."""
    term_lower = term.lower()

    # 1. Alphanumeric split: e.g. "ext4formatted" -> "ext4" + "formatted"
    parts = [p for p in _DIGIT_LETTER_SPLIT_RE.split(term_lower) if p]
    if len(parts) >= 2:
        matched_surfaces: list[str] = []
        all_matched = True
        for p in parts:
            # If the part is in translatable terms
            p_surfaces = find_translatable_surfaces(p) or _COMMON_SUB_TERMS.get(p)
            if p_surfaces:
                hit = next((s for s in p_surfaces if s in target), None)
                if hit:
                    matched_surfaces.append(hit)
                    continue
            # If part appears verbatim in target
            if p in target.lower():
                matched_surfaces.append(p)
                continue
            all_matched = False
            break
        if all_matched and matched_surfaces:
            return matched_surfaces

    # 2. Known prefix split: e.g. "microvm" -> "micro" + "vm"
    for prefix in _COMMON_PREFIXES:
        if term_lower.startswith(prefix) and len(term_lower) > len(prefix) + 1:
            rest = term_lower[len(prefix) :]
            rest_surfaces = find_translatable_surfaces(rest) or _COMMON_SUB_TERMS.get(rest)
            if rest_surfaces:
                hit = next((s for s in rest_surfaces if s in target), None)
                if hit:
                    return [hit]

    return None
