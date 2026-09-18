"""test_compare.py：钉死比较器的判定矩阵——compliant/non_compliant/B 拒答三类。

夹具用 corpus 里的真实条文文本（oracle 句），保证规则轨与语料演进同步。
"""
from agents.compare import detect_attribute, judge_item


def _c(clause_no, content, score=6.0):
    return [{"chunk_id": f"SPGT21001-2025:{clause_no}:0", "clause_no": clause_no,
             "content": content, "rerank": score}]


RA = _c("4.3.1", "表面粗糙度参数应优先选用轮廓算术平均偏差 Ra。"
                 "一般配合表面的 Ra 上限值为 1.6 μm；一般结合表面的 Ra 上限值为 6.3 μm；"
                 "非加工表面 Ra 上限值为 25 μm。")
RATIO = _c("4.1.2", "绘制图样时应采用 1:1、1:2、1:5、2:1、5:1 及其十进制倍数系列的比例，"
                    "不推荐使用 1:3、1:7 等中间比例。")
GROUND = [{"chunk_id": "SPGT21003-2025:4.3.2:0", "clause_no": "4.3.2",
           "content": "保护接地电阻应不大于 4 Ω；测量点为设备接地端子与最近接地极之间。",
           "rerank": 6.0}]
REINFORCE = [{"chunk_id": "SPGT21002-2025:4.2.1:0", "clause_no": "4.2.1",
              "content": "焊缝余高应控制在 0 ~ 3 mm 之间，余高过大将造成应力集中。",
              "rerank": 6.0}]
UNDERCUT = [{"chunk_id": "SPGT21002-2025:4.1.2:0", "clause_no": "4.1.2",
             "content": "咬边深度不得大于 0.5 mm，且连续长度不得大于 100 mm；"
                        "一级焊缝不允许存在咬边。", "rerank": 6.0}]
MISALIGN = [{"chunk_id": "SPGT21002-2025:4.2.2:0", "clause_no": "4.2.2",
             "content": "对接接头的错边量不得大于板厚的 10%，且最大不超过 2 mm。",
             "rerank": 6.0}]
RT_PCT = [{"chunk_id": "SPGT21002-2025:4.3.1:0", "clause_no": "4.3.1",
           "content": "一级焊缝应进行 100% 射线检测（RT）；二级焊缝进行不低于 20% 的"
                      "抽检探伤，抽检位置应覆盖交叉焊缝与变截面部位。", "rerank": 6.0}]
COLOR = [{"chunk_id": "SPGT21003-2025:4.1.2:0", "clause_no": "4.1.2",
          "content": "导体相序标识颜色：L1 为黄色、L2 为绿色、L3 为红色；"
                     "中性导体 N 为淡蓝色；保护导体 PE 必须为绿黄双色，"
                     "且绿黄双色不得用于其他用途。", "rerank": 6.0}]
LIVE = [{"chunk_id": "SPGT21003-2025:4.2.1:0", "clause_no": "4.2.1",
         "content": "操作面上高度低于 2300 mm 的裸露带电部分必须加装护罩或遮栏，"
                    "护罩应使用工具才能拆卸。", "rerank": 6.0}]
AISLE = [{"chunk_id": "SPGT21003-2025:4.2.2:0", "clause_no": "4.2.2",
          "content": "电气柜前维修通道宽度不得小于 800 mm；柜后通道不得小于 600 mm。",
          "rerank": 6.0}]
PE = [{"chunk_id": "SPGT21003-2025:4.3.1:0", "clause_no": "4.3.1",
       "content": "保护导体（PE）截面积：当相线截面积不大于 16 mm² 时，"
                  "PE 截面积应与相线相同；大于 16 mm² 时可为相线的一半，"
                  "且不小于 16 mm²。", "rerank": 6.0}]
DRY = [{"chunk_id": "SPGT21002-2025:4.4.1:0", "clause_no": "4.4.1",
        "content": "低氢型焊条使用前应在 350 ~ 400 ℃ 烘干 1 ~ 2 h，"
                   "并置于 100 ~ 150 ℃ 保温筒内随用随取。", "rerank": 6.0}]


def test_ra_over_limit():
    j = judge_item("阶梯轴配合表面粗糙度 Ra 3.2", RA)
    assert j.verdict == "non_compliant" and "1.6" in j.rationale


def test_ra_within_limit_and_kind_routing():
    assert judge_item("配合表面 Ra 1.6", RA).verdict == "compliant"
    assert judge_item("非加工表面 Ra 20", RA).verdict == "compliant"   # 换句：25 上限
    assert judge_item("结合表面 Ra 10", RA).verdict == "non_compliant"


def test_ratio_forbidden_not_whitelisted():
    """回归：'不推荐 1:3、1:7' 否定分句不得被当进允许集。"""
    assert judge_item("绘图比例 1:3", RATIO).verdict == "non_compliant"
    assert judge_item("绘图比例 1:2", RATIO).verdict == "compliant"


def test_ground_resistance():
    assert judge_item("电气柜保护接地电阻 5 Ω", GROUND).verdict == "non_compliant"
    assert judge_item("保护接地电阻 4 Ω", GROUND).verdict == "compliant"


def test_reinforcement_range():
    assert judge_item("焊缝余高 4 mm", REINFORCE).verdict == "non_compliant"
    assert judge_item("焊缝余高 2 mm", REINFORCE).verdict == "compliant"


def test_undercut_dual_bound():
    """双上限各归各：长度限 100 不得套深度限 0.5（评测二 15 条误报的回归钉）。"""
    j = judge_item("咬边深度 0.8 mm", UNDERCUT)
    assert j.verdict == "non_compliant" and "0.5" in j.rationale
    j = judge_item("咬边连续长度 80 mm", UNDERCUT)
    assert j.verdict == "compliant" and "100" in j.rationale
    j = judge_item("咬边连续长度 150 mm", UNDERCUT)
    assert j.verdict == "non_compliant" and "100" in j.rationale
    assert judge_item("咬边深度 0.3 mm", UNDERCUT).verdict == "compliant"


def test_misalignment_compound():
    j = judge_item("板厚 20 mm 错边量 2.5 mm", MISALIGN)
    assert j.verdict == "non_compliant"
    assert judge_item("板厚 10 mm 错边量 0.8 mm", MISALIGN).verdict == "compliant"


def test_rt_percent():
    assert judge_item("二级焊缝抽检探伤比例 15%", RT_PCT).verdict == "non_compliant"
    assert judge_item("二级焊缝抽检探伤比例 20%", RT_PCT).verdict == "compliant"


def test_color_code_mapping():
    assert judge_item("PE 保护导体采用红色", COLOR).verdict == "non_compliant"
    assert judge_item("PE 保护导体采用绿黄双色", COLOR).verdict == "compliant"


def test_live_height_guard():
    assert judge_item("高度 2000 mm 的裸露带电部分未加装护罩", LIVE).verdict == "non_compliant"
    assert judge_item("高度 2400 mm 的裸露带电部分未加装护罩", LIVE).verdict == "compliant"


def test_aisle_front_back():
    assert judge_item("电气柜前维修通道宽度 700 mm", AISLE).verdict == "non_compliant"
    assert judge_item("柜后通道宽度 500 mm", AISLE).verdict == "non_compliant"
    assert judge_item("柜后通道宽度 650 mm", AISLE).verdict == "compliant"


def test_pe_section_compound():
    assert judge_item("相线 25 mm²，PE 12 mm²", PE).verdict == "non_compliant"
    assert judge_item("相线 10 mm²，PE 10 mm²", PE).verdict == "compliant"


def test_dry_temp_range():
    assert judge_item("焊条烘干温度 300 ℃", DRY).verdict == "non_compliant"
    assert judge_item("焊条烘干温度 380 ℃", DRY).verdict == "compliant"


def test_quote_is_substring_for_gate():
    """比较器给的 quote 必须是条文正文的子串（证据门控第 2 道的前提）。"""
    for item, cand in [("配合表面 Ra 3.2", RA), ("PE 采用红色", COLOR),
                       ("板厚 20 mm 错边量 2.5 mm", MISALIGN)]:
        j = judge_item(item, cand)
        assert j.quote in cand[0]["content"]


def test_unparseable_clause_refuses():
    """命中术语卡（无约束）→ None → 上层走 B 级拒答。"""
    term = _c("3.1", "基本尺寸：设计给定的尺寸，图样上标注的理论值。")
    assert judge_item("配合表面 Ra 3.2", term) is None


def test_detect_attribute():
    assert detect_attribute("接地电阻 4 Ω") == "ground_res"
    assert detect_attribute("焊缝余高 3 mm") == "reinforcement"
    assert detect_attribute("与标准无关的一句话") is None
