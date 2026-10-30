"""G2 Calibration Corpus: 80 standardized test cases across 4 layout roles.

Specification: ``docs/design/adaptive_delivery_prd.md`` §8.1 &
``docs/guides/ADAPTIVE_DELIVERY_TECH_DESIGN.md`` §5.2.

This module provides the golden calibration suite for MeasurementService:
- 20 Body cases (prose, CJK-Latin mixing, inline math formulas, varied lengths)
- 20 Heading cases (chapters, sections, subsections, varied font sizes)
- 20 Caption cases (figure/table labels, sub-figure enumerations, tight leading)
- 20 TableHeader / TableCell cases (narrow rigid containers, numbers, formulas)

Each case specifies text, width, style parameters, and expected bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ubt.delivery.measurement import MeasurementRequest, MeasurementStyle

RoleType = Literal["body", "heading", "caption", "table"]


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    case_id: str
    role: RoleType
    text: str
    width_pt: float
    size_pt: float
    leading_pt: float
    tracking_em: float = 0.0
    expected_min_lines: int = 1
    expected_max_lines: int = 20

    def to_request(
        self, font_family: str = "Noto Serif CJK SC", font_version: str = ""
    ) -> MeasurementRequest:
        style = MeasurementStyle(
            font_family=font_family,
            font_version=font_version,
            size_pt=self.size_pt,
            leading_pt=self.leading_pt,
            tracking_em=self.tracking_em,
            language="zh",
        )
        return MeasurementRequest(text=self.text, style=style, width_pt=self.width_pt)


def get_calibration_corpus() -> list[CalibrationCase]:
    """Return the 80 standardized calibration cases."""
    cases: list[CalibrationCase] = []

    # =========================================================================
    # 1. BODY (20 items): 8.4pt - 10.5pt, widths 160pt - 450pt
    # =========================================================================
    body_data = [
        # (id, text, width, size, leading, min_lines, max_lines)
        (
            "body_01",
            "本文提出了一种新型全耗尽纳米片晶体管模型，结合自适应网格细化技术，显著降低了数值求解的收敛震荡。",
            180.0,
            9.0,
            11.8,
            3,
            5,
        ),
        (
            "body_02",
            "在先进半导体工艺节点下，三维环绕栅极（GAA）架构逐渐取代传统的 FinFET 结构，成为继续推进摩尔定律的核心路径。",
            240.0,
            9.5,
            12.5,
            3,
            5,
        ),
        (
            "body_03",
            "通过漂移-扩散方程与量子修正势的自洽联合求解，可以精确评估沟道内部由于短沟道效应导致的载流子速度饱和特性。",
            360.0,
            10.0,
            13.0,
            2,
            4,
        ),
        (
            "body_04",
            "短段落测试。",
            180.0,
            9.0,
            11.8,
            1,
            1,
        ),
        (
            "body_05",
            "当器件漏极偏置电压满足 $V_{DS} > V_{GS} - V_{TH}$ 时，MOSFET 晶体管进入饱和导电区，电流主要受栅极电场调制。",
            180.0,
            8.5,
            11.2,
            3,
            5,
        ),
        (
            "body_06",
            "跨导 $g_m = \\frac{\\partial I_{DS}}{\\partial V_{GS}}$ 与输出电导 $g_{ds} = \\frac{\\partial I_{DS}}{\\partial V_{DS}}$ 是高频小信号等效电路中的关键参量。",
            220.0,
            9.0,
            12.0,
            3,
            6,
        ),
        (
            "body_07",
            "实验测得的阈值电压漂移 $\\Delta V_{TH}$ 随温度升高呈现负相关趋势，这与费米能级随温度变化理论计算预测一致。",
            300.0,
            9.5,
            12.5,
            2,
            4,
        ),
        (
            "body_08",
            "根据 BSIM-CMG 紧凑模型物理定义，非均匀栅氧化层厚度 $T_{ox}$ 的分布会引入表面势梯度的局部扰动。",
            200.0,
            8.4,
            11.0,
            3,
            5,
        ),
        (
            "body_09",
            "“自适应交付”（Adaptive Delivery）架构强调在阅读体验与物理保真之间取得平衡，拒绝粗暴缩减正文字号。",
            190.0,
            9.0,
            11.8,
            3,
            5,
        ),
        (
            "body_10",
            "基于 TSMC N3 工艺的物理测试晶圆表明，当栅长缩短至 12nm 以下时，DIBL（漏端诱导势垒降低）效应将显著增加静态待机漏电。",
            250.0,
            9.0,
            12.0,
            3,
            5,
        ),
        (
            "body_11",
            "长正文段落：在微电子器件模拟中，泊松方程与连续性方程的离散化方法决定了求解器在大偏置扫描下的鲁棒性。传统的有限差分法（FDM）在复杂非平面几何边缘处容易产生截断误差，而有限元法（FEM）通过非结构化四面体网格划分能够完美贴合多层金属互连线边界，但其刚度矩阵的求解需要消耗更多的内存带宽与浮点运算周期。",
            380.0,
            10.0,
            13.5,
            4,
            8,
        ),
        (
            "body_12",
            "为了抑制亚阈值摆幅（SS）恶化，通常需要在源漏两端引入陡峭的超浅掺杂结，并配合低温退火工艺激活杂质原子。",
            180.0,
            8.4,
            11.0,
            3,
            6,
        ),
        (
            "body_13",
            "在量子力学框架下，二维受限势阱中的能级分裂导致电子等效态密度增加，从而改变了准费米能级在导带底部的分布位置。",
            240.0,
            9.0,
            12.0,
            2,
            5,
        ),
        (
            "body_14",
            "考虑寄生电阻 $R_s$ 与 $R_d$ 的退化效应后，外特征电流方程需修正为 $I_{DS} = f(V_{GS} - I_{DS} R_s, V_{DS} - I_{DS}(R_s + R_d))$。",
            210.0,
            8.5,
            11.5,
            3,
            6,
        ),
        (
            "body_15",
            "英文混排短句：The device characteristics were measured using a Keysight B1500A parameter analyzer under ambient conditions.",
            220.0,
            9.0,
            12.0,
            2,
            4,
        ),
        (
            "body_16",
            "中文夹杂长公式：根据能量平衡传输模型，电子有效迁移率表示为 $\\mu_{\\text{eff}} = \\frac{\\mu_0}{1 + (E_{\\text{eff}} / E_c)^\\eta}$，其中参数 $\\eta$ 由声子散射强度决定。",
            200.0,
            8.5,
            11.5,
            4,
            7,
        ),
        (
            "body_17",
            "在测试中未发现界面态陷阱电荷引起的迟滞现象（Hysteresis < 2mV），证明高介电常数（High-k）绝缘层与金属栅极界面钝化良好。",
            320.0,
            9.5,
            12.5,
            2,
            4,
        ),
        (
            "body_18",
            "正文单行极值：仅有一句话的极短段落。",
            400.0,
            10.0,
            13.0,
            1,
            1,
        ),
        (
            "body_19",
            "多标点符号与括号：【注】根据《器件模型手册》（第 4 版，第 127 页）所述：该模型（BSIM-CMG v111.1.0）已包含对热载流子注入效应（HCI）的经验拟合参数项！",
            220.0,
            8.8,
            11.6,
            3,
            6,
        ),
        (
            "body_20",
            "紧凑列宽极限测试：双栏左栏极窄宽度下的紧密正文断行与对齐能力验证。",
            120.0,
            8.0,
            10.5,
            3,
            6,
        ),
    ]

    for item in body_data:
        cid, text, width, size, leading, min_l, max_l = item
        cases.append(CalibrationCase(cid, "body", text, width, size, leading, 0.0, min_l, max_l))

    # =========================================================================
    # 2. HEADING (20 items): 11.0pt - 18.0pt, bold/section headings
    # =========================================================================
    heading_data = [
        ("head_01", "第 3 章 纳米器件紧凑模型与几何缩放理论", 400.0, 16.0, 20.0, 1, 2),
        ("head_02", "3.1 全耗尽绝缘体上硅（FD-SOI）与多栅架构演进", 350.0, 14.0, 18.0, 1, 2),
        ("head_03", "3.2 基于漂移扩散方程的非均匀掺杂修正方法", 350.0, 14.0, 18.0, 1, 2),
        ("head_04", "3.2.1 漏端诱导势垒降低（DIBL）与表面势连续性分析", 300.0, 12.0, 15.0, 1, 3),
        ("head_05", "3.2.2 量子限制效应导致的有效电容退化机制", 300.0, 12.0, 15.0, 1, 3),
        ("head_06", "3.3 寄生电阻与结电容的高频提取技术", 350.0, 13.0, 16.5, 1, 2),
        ("head_07", "4.0 实验测试环境与参数拟合结果比对", 380.0, 15.0, 19.0, 1, 2),
        ("head_08", "引言", 200.0, 16.0, 20.0, 1, 1),
        ("head_09", "理论推导", 200.0, 14.0, 18.0, 1, 1),
        ("head_10", "数值模拟与实验验证", 250.0, 13.0, 16.5, 1, 1),
        ("head_11", "结论与未来展望", 250.0, 14.0, 18.0, 1, 1),
        ("head_12", "参考文献（References）", 300.0, 13.0, 16.5, 1, 1),
        ("head_13", "附录 A：主要物理符号与常数表", 320.0, 12.0, 15.0, 1, 2),
        ("head_14", "附录 B：边界条件推导与有限元弱形式表达式", 350.0, 12.0, 15.0, 1, 2),
        (
            "head_15",
            "超长跨栏标题测试：基于深亚微米三维全环绕栅极纳米片晶体管的多物理场耦合数值仿真与可靠性评估",
            450.0,
            16.0,
            21.0,
            1,
            3,
        ),
        ("head_16", "窄栏短标题：模型验证", 140.0, 11.5, 14.5, 1, 1),
        ("head_17", "含数学符号标题：关于阈值电压 $V_{TH}$ 的解析解推导", 280.0, 12.5, 16.0, 1, 2),
        (
            "head_18",
            "含参数标题：寄生电阻 $R_{ext}$ 对器件高频品质因数 $f_{max}$ 的影响分析",
            320.0,
            12.5,
            16.0,
            1,
            3,
        ),
        ("head_19", "致谢", 200.0, 14.0, 18.0, 1, 1),
        ("head_20", "作者简介与利益冲突声明", 300.0, 12.0, 15.0, 1, 2),
    ]

    for item in heading_data:
        cid, text, width, size, leading, min_l, max_l = item
        cases.append(CalibrationCase(cid, "heading", text, width, size, leading, 0.0, min_l, max_l))

    # =========================================================================
    # 3. CAPTION (20 items): 7.5pt - 9.0pt, tight leading, figures/tables
    # =========================================================================
    caption_data = [
        (
            "cap_01",
            "图 3.1 漏极电流 $I_{DS}$ 随栅极偏压 $V_{GS}$ 变化的实验测量与模型拟合曲线。",
            180.0,
            8.5,
            10.8,
            2,
            4,
        ),
        (
            "cap_02",
            "图 3.2 纳米线晶体管截面高分辨率透射电子显微镜（HR-TEM）成像与能谱分析图。",
            220.0,
            8.0,
            10.2,
            2,
            4,
        ),
        (
            "cap_03",
            "图 3.3 (a) 线性区输出特性；(b) 饱和区跨导随栅偏压的变化关系曲线。",
            200.0,
            8.0,
            10.2,
            2,
            4,
        ),
        (
            "cap_04",
            "图 3.4 亚阈值摆幅与栅极氧化层等效厚度（EOT）在不同衬底偏压下的对应关系。",
            240.0,
            8.5,
            10.8,
            2,
            4,
        ),
        ("cap_05", "表 3.1 紧凑模型主要拟合参数取值及其物理单位说明", 250.0, 8.5, 10.8, 1, 2),
        (
            "cap_06",
            "表 3.2 不同工艺角（TT/FF/SS）下关键器件性能指标蒙特卡洛统计方差",
            280.0,
            8.5,
            10.8,
            1,
            2,
        ),
        (
            "cap_07",
            "图 3.5 多栅纳米片器件在热载流子应力加压 1000 小时前后的阈值电压漂移对比分析。",
            210.0,
            8.0,
            10.0,
            2,
            5,
        ),
        ("cap_08", "FIG. 3.1", 80.0, 8.0, 10.0, 1, 1),
        ("cap_09", "FIG. 3.2 (continued)", 100.0, 8.0, 10.0, 1, 1),
        ("cap_10", "图 1", 50.0, 8.0, 10.0, 1, 1),
        (
            "cap_11",
            "长图注：图 3.6 实验测得的三维 FinFET 器件在不同栅长（从 14nm 至 32nm）下的亚阈值斜率（SS）与漏端诱导势垒降低（DIBL）数值统计散点图。实线代表本文修正紧凑物理模型的连续解析解，虚线代表未考虑量子受限修正的经典漂移扩散仿真结果。",
            360.0,
            8.0,
            10.0,
            3,
            6,
        ),
        (
            "cap_12",
            "图 3.7 (a) 沟道中心电场分布；(b) 漏端碰撞电离倍增系数分布对比。",
            190.0,
            8.0,
            10.0,
            2,
            4,
        ),
        (
            "cap_13",
            "图 3.8 寄生电容 $C_{gg}$ 与 $C_{gd}$ 随工作频率（1GHz 至 100GHz）的响应曲线。",
            220.0,
            8.0,
            10.2,
            2,
            4,
        ),
        (
            "cap_14",
            "表 4.1 本文模型与实测数据在全工作温区（-40°C 至 125°C）内的均方根误差（RMSE）统计",
            320.0,
            8.5,
            10.8,
            1,
            3,
        ),
        (
            "cap_15",
            "图 3.9 栅极诱导漏极漏电（GIDL）在反向偏置下的实验测量谱图。",
            180.0,
            8.0,
            10.0,
            2,
            4,
        ),
        ("cap_16", "图 3.10 硅纳米线反型层中的二维电子气能带结构图。", 180.0, 8.0, 10.0, 2, 4),
        (
            "cap_17",
            "图注英文夹杂：Fig. 3.11 Comparison of simulated and measured transfer characteristics at $V_{DS} = 0.05\\text{V}$ and $V_{DS} = 0.9\\text{V}$.",
            220.0,
            8.0,
            10.2,
            2,
            4,
        ),
        ("cap_18", "图 3.12 纳米片宽度对载流子有效注入速度的影响规律。", 190.0, 8.0, 10.0, 2, 4),
        ("cap_19", "表 5.2 各种前沿架构晶体管品质因数（FOM）对比总表", 250.0, 8.5, 10.8, 1, 2),
        ("cap_20", "极窄双栏图注：图 3.13 接触孔电阻模型验证。", 120.0, 7.5, 9.5, 1, 3),
    ]

    for item in caption_data:
        cid, text, width, size, leading, min_l, max_l = item
        cases.append(CalibrationCase(cid, "caption", text, width, size, leading, 0.0, min_l, max_l))

    # =========================================================================
    # 4. TABLE HEADER / CELL (20 items): 7.0pt - 8.5pt, narrow containers 35pt - 140pt
    # =========================================================================
    table_data = [
        ("tbl_01", "参数符号", 60.0, 8.0, 10.0, 1, 1),
        ("tbl_02", "标称值 (Nominal)", 75.0, 8.0, 10.0, 1, 2),
        ("tbl_03", "物理单位 (Unit)", 70.0, 8.0, 10.0, 1, 2),
        ("tbl_04", "相对误差 (\\%)", 65.0, 8.0, 10.0, 1, 2),
        ("tbl_05", "沟道长度 $L_{\\text{gate}}$ (nm)", 80.0, 7.5, 9.5, 1, 2),
        ("tbl_06", "阈值电压 $V_{TH}$ (mV)", 75.0, 7.5, 9.5, 1, 2),
        ("tbl_07", "最大跨导 $g_m$ (mS/μm)", 85.0, 7.5, 9.5, 1, 2),
        ("tbl_08", "饱和驱动电流 $I_{on}$", 70.0, 7.5, 9.5, 1, 2),
        ("tbl_09", "静态漏电 $I_{off}$ (pA/μm)", 80.0, 7.5, 9.5, 1, 2),
        ("tbl_10", "亚阈值摆幅 SS (mV/dec)", 90.0, 7.5, 9.5, 1, 2),
        ("tbl_11", "极窄表头：编号", 35.0, 7.0, 9.0, 1, 3),
        ("tbl_12", "极窄数值：12.50", 40.0, 7.5, 9.5, 1, 3),
        ("tbl_13", "单元格短文本：PASS", 45.0, 8.0, 10.0, 1, 3),
        ("tbl_14", "科学计数法：1.08e-12", 60.0, 7.5, 9.5, 1, 3),
        ("tbl_15", "工艺状态：TT corner", 65.0, 7.5, 9.5, 1, 3),
        (
            "tbl_16",
            "温度范围：$-40^\\circ\\text{C} \\sim 125^\\circ\\text{C}$",
            90.0,
            7.5,
            9.5,
            1,
            2,
        ),
        ("tbl_17", "电容密度：$C_{ox} = 1.45\\,\\mu\\text{F/cm}^2$", 100.0, 7.5, 9.5, 1, 5),
        ("tbl_18", "寄生电感 $L_s$ (pH)", 70.0, 7.5, 9.5, 1, 2),
        ("tbl_19", "品质因数 $f_T / f_{max}$", 80.0, 7.5, 9.5, 1, 2),
        (
            "tbl_20",
            "较长表格文字：基于非局部能带重整化的电子空穴等效质量修正系数",
            130.0,
            7.0,
            9.0,
            2,
            4,
        ),
    ]

    for item in table_data:
        cid, text, width, size, leading, min_l, max_l = item
        cases.append(CalibrationCase(cid, "table", text, width, size, leading, 0.0, min_l, max_l))

    return cases
