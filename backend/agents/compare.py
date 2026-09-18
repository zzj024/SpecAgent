"""compare.py：确定性比较器——rule 模式的"支持性判定"，也是评测的离线 oracle。

定位（面试口径）：审查判定有两条可插拔的执行轨——
  rule 模式（默认，无 LLM API 也能全流程跑）：程序化数值/类别比对，本文件；
  llm  模式（配置 DEEPSEEK_API_KEY 后自动启用）：LLM 支持性判定，
       证据门控与结论落库逻辑两条轨完全一致（见 nodes.py）。
rule 轨的意义不止"没 KEY 也能跑"：它是单测的 oracle、评测七的回归基准、
以及 LLM 判定的交叉复核第二意见。

结构：detect_attribute（这句话在说哪个属性）→ 派发到对应 judge →
  judge 从**条款卡片正文**里抽约束（上限/下限/区间/枚举），从检查项里抽文档值，
  比对出 verdict。约束来自检索到的条文本身而不是代码里的硬编码表——
  换一份语料，约束跟着条文走。quote 取"含约束的那句话"，它是 content 的
  子串，天然过证据门控第 2 道（分块按句子边界拆，句子不会跨卡）。
"""
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Judgment:
    """一次判定的产物。verdict 用 core/schemas.py 的同一套词表。"""

    verdict: str                 # compliant | non_compliant | refusal_B
    confidence: float
    clause_no: str               # 判定依据条款（来自命中的候选卡）
    quote: str = ""              # 证据原文（content 的子串）
    rationale: str = ""
    evidence_score: float = 0.0  # 检索相关分，随证据进报告
    meta: dict = field(default_factory=dict)


_EPS = 1e-9


def _num(s: str) -> float:
    return float(s)


def _sentences(content: str) -> list[str]:
    """条款正文切句：。；后切、标点留前段——与 chunking._split_long 同规则，
    保证返回的每一句都是 content 的逐身子串。"""
    return [s for s in re.split(r"(?<=[。；])", content) if s.strip()]


# ---------------------------------------------------------------------------
# 约束抽取：从一句话里识别 上限 / 下限 / 区间 / 百分比
# ---------------------------------------------------------------------------

_UPPER_RE = re.compile(
    r"(?:不得大于|不得超过|最大不超过|应不大于|上限值为|不得高于|不宜超过|不超过)\s*"
    r"(\d+(?:\.\d+)?)")
_LOWER_RE = re.compile(
    r"(?:不得小于|应不小于|不小于|不得低于|不低于|不少于)\s*(\d+(?:\.\d+)?)")
_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[~～]\s*(\d+(?:\.\d+)?)")
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|μm|℃|Ω|%|h|次)?")


def extract_bounds(sentence: str) -> dict | None:
    """一句话 → {upper} / {lower} / {lo, hi}。识别不出数值约束返回 None。"""
    if m := _RANGE_RE.search(sentence):
        lo, hi = _num(m.group(1)), _num(m.group(2))
        if lo < hi:
            return {"lo": lo, "hi": hi}
    if m := _UPPER_RE.search(sentence):
        return {"upper": _num(m.group(1))}
    if m := _LOWER_RE.search(sentence):
        return {"lower": _num(m.group(1))}
    return None


def extract_value(text: str) -> float | None:
    """检查项/文档侧抽数值：最后一个带小数位的数（"Ra 3.2" → 3.2）。
    取最后不取最先：工程表述几乎都是"属性名 + 值"顺序。"""
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    return _num(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# 属性注册表：检查项关键词 → 属性 → 该属性在条文里的锚词
# ---------------------------------------------------------------------------

ATTRIBUTES: dict[str, dict] = {
    # attr: {"item_keys": 检查项侧关键词, "clause_keys": 条文侧锚词}
    "ra":               {"item_keys": ["粗糙度", "Ra"], "clause_keys": ["Ra"]},
    "flatness":         {"item_keys": ["平面度"], "clause_keys": ["平面度"]},
    "undercut":         {"item_keys": ["咬边"], "clause_keys": ["咬边"]},
    "reinforcement":    {"item_keys": ["余高"], "clause_keys": ["余高"]},
    "misalignment":     {"item_keys": ["错边"], "clause_keys": ["错边"]},
    "rt_percent":       {"item_keys": ["射线检测", "RT", "探伤"], "clause_keys": ["射线检测", "探伤"]},
    "rt_grade":         {"item_keys": ["评定等级", "II 级", "Ⅲ"], "clause_keys": ["评定"]},
    "dry_temp":         {"item_keys": ["烘干", "保温筒"], "clause_keys": ["烘干"]},
    "repair_count":     {"item_keys": ["返修"], "clause_keys": ["返修"]},
    "color_code":       {"item_keys": ["相序", "L1", "绿黄", "标识颜色", "导体颜色"],
                         "clause_keys": ["绿黄双色"]},
    "live_height":      {"item_keys": ["裸露带电", "带电部分"], "clause_keys": ["裸露带电部分"]},
    "aisle_width":      {"item_keys": ["通道宽度", "柜前", "柜后", "维修通道"],
                         "clause_keys": ["通道宽度"]},
    "pe_section":       {"item_keys": ["PE", "保护导体", "接地导体"],
                         "clause_keys": ["保护导体"]},
    "ground_res":       {"item_keys": ["接地电阻"], "clause_keys": ["接地电阻"]},
    "scale_ratio":      {"item_keys": ["比例"], "clause_keys": ["比例"]},
}


_COLOR_WORDS = ("绿黄双色", "淡蓝色", "黄色", "绿色", "红色")


def detect_attribute(item_text: str) -> str | None:
    """检查项文本 → 属性名。颜色词一票优先（消歧）："PE 保护导体采用红色"
    会同时命中 pe_section 的两个关键词，但只要出现颜色词，这句话说的
    一定是相序颜色不是截面积——强特征词短路弱计数。其余按命中数取胜。"""
    if any(w in item_text for w in _COLOR_WORDS):
        return "color_code"
    best, best_hits = None, 0
    for attr, spec in ATTRIBUTES.items():
        hits = sum(1 for k in spec["item_keys"] if k in item_text)
        if hits > best_hits:
            best, best_hits = attr, hits
    return best


def _numeric_upper(item_text: str, clause: dict) -> Judgment | None:
    """数值上限型：文档值 ≤ 条文上限 → 符合。Ra 按表面类别选句（配合/结合/非加工）。

    两道守卫（跨条款误判的来源，回归测试钉死）：
    1. 类别严格匹配：检查项说了"非加工"，条款里没有非加工句 → 本卡不判
       （让位给下一名候选），不能拿配合面上限套非加工面；
    2. 前置条件守卫：条款带 IT7/公差等级前置（4.3.2）而检查项没提等级 →
       本卡不判——它的上限只对高精度尺寸生效。"""
    if "IT" in clause["content"] and not re.search(r"IT\s*\d+", item_text):
        return None
    kind = next((k for k in ("非加工", "结合", "配合") if k in item_text), None)
    sents = _sentences(clause["content"])
    chosen = None
    if kind:
        chosen = next((s for s in sents if kind in s and extract_bounds(s)), None)
        if chosen is None:                    # 本条款不含该类别 → 不判，让位
            return None
    else:
        chosen = next((s for s in sents if extract_bounds(s)), None)
    if chosen is None:
        return None
    bounds = extract_bounds(chosen)
    val = extract_value(item_text)
    if val is None or "upper" not in bounds:
        return None
    ok = val <= bounds["upper"] + _EPS
    return Judgment(
        verdict="compliant" if ok else "non_compliant",
        confidence=0.9,
        clause_no=clause["clause_no"],
        quote=chosen,
        rationale=f"条文上限 {bounds['upper']}，文档值 {val}"
                  + ("，未超限。" if ok else "，超出上限。"),
        evidence_score=clause.get("rerank", 0.0),
    )


def _numeric_range(item_text: str, clause: dict) -> Judgment | None:
    """区间型（余高 0~3 / 烘干 350~400）：lo ≤ val ≤ hi。"""
    for s in _sentences(clause["content"]):
        b = extract_bounds(s)
        val = extract_value(item_text)
        if b and "lo" in b and val is not None:
            ok = b["lo"] - _EPS <= val <= b["hi"] + _EPS
            return Judgment(
                verdict="compliant" if ok else "non_compliant",
                confidence=0.9,
                clause_no=clause["clause_no"],
                quote=s,
                rationale=f"条文要求控制在 {b['lo']} ~ {b['hi']}，文档值 {val}"
                          + ("，在区间内。" if ok else "，越出区间。"),
                evidence_score=clause.get("rerank", 0.0),
            )
    return None


def _dual_bound(item_text: str, clause: dict) -> Judgment | None:
    """双上限型（咬边：深度 ≤0.5 / 连续长度 ≤100）。

    上限必须锚定关键词抽取：两条上限写在同一句里（"深度不得大于 0.5，
    且连续长度不得大于 100"），无脑取句中第一个"不得大于"会把深度限
    0.5 套到长度上（回归教训：评测二 15 条误报全源于此）。"""
    which = "长度" if ("长度" in item_text) else "深度"
    s = next((x for x in _sentences(clause["content"]) if which in x), None)
    if s is None:
        return None
    m = re.search(rf"{which}[^；。]*?不得大于\s*(\d+(?:\.\d+)?)", s)
    if not m:
        b = extract_bounds(s)
        if not b or "upper" not in b:
            return None
        limit = b["upper"]
    else:
        limit = _num(m.group(1))
    val = extract_value(item_text)
    if val is None:
        return None
    ok = val <= limit + _EPS
    return Judgment(
        verdict="compliant" if ok else "non_compliant",
        confidence=0.9, clause_no=clause["clause_no"], quote=s,
        rationale=f"咬边{which}条文上限 {limit}，文档值 {val}"
                  + ("，符合。" if ok else "，超限。"),
        evidence_score=clause.get("rerank", 0.0),
    )


def _misalignment(item_text: str, clause: dict) -> Judgment | None:
    """复合约束（错边量：≤板厚10% 且 ≤2mm 双上限取小）。"""
    s = next((x for x in _sentences(clause["content"]) if "错边" in x), None)
    if s is None:
        return None
    pct = re.search(r"板厚的\s*(\d+(?:\.\d+)?)\s*%", s)
    cap = _UPPER_RE.search(s.replace("最大不超过", "不得超过"))
    if not (pct and cap):
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", item_text)
    if len(nums) < 2:                       # 需要 板厚 + 错边量 两个数
        return None
    thickness, actual = _num(nums[0]), _num(nums[-1])
    limit = min(thickness * _num(pct.group(1)) / 100.0, _num(cap.group(1)))
    ok = actual <= limit + _EPS
    return Judgment(
        verdict="compliant" if ok else "non_compliant",
        confidence=0.9, clause_no=clause["clause_no"], quote=s,
        rationale=f"板厚 {thickness} → 限值 min(10%={thickness * _num(pct.group(1)) / 100:g}, "
                  f"2)={limit:g}，文档错边量 {actual}" + ("，符合。" if ok else "，超限。"),
        evidence_score=clause.get("rerank", 0.0),
    )


def _percent(item_text: str, clause: dict) -> Judgment | None:
    """百分比型（一级 100% RT 精确；二级 ≥20% 下限）。"""
    grade = "一级" if "一级" in item_text else "二级" if "二级" in item_text else None
    sents = _sentences(clause["content"])
    s = next((x for x in sents if grade and grade in x), None) or next(
        (x for x in sents if _PCT_RE.search(x)), None)
    if s is None:
        return None
    req = _PCT_RE.search(s)
    val_m = re.search(r"(\d+(?:\.\d+)?)\s*%", item_text)
    if not (req and val_m):
        return None
    need, val = _num(req.group(1)), _num(val_m.group(1))
    if "不低于" in s or "不得低于" in s or "不小于" in s:
        ok = val >= need - _EPS
    else:                                    # "应进行 100%" 精确要求
        ok = abs(val - need) < _EPS
    return Judgment(
        verdict="compliant" if ok else "non_compliant",
        confidence=0.85, clause_no=clause["clause_no"], quote=s,
        rationale=f"条文要求 {need}%（{grade or '该级焊缝'}），文档 {val}%"
                  + ("，满足。" if ok else "，不满足。"),
        evidence_score=clause.get("rerank", 0.0),
    )


_COLORS = ["绿黄双色", "淡蓝色", "黄色", "绿色", "红色"]
_CONDUCTORS = ["L1", "L2", "L3", "N", "PE"]


def _color_code(item_text: str, clause: dict) -> Judgment | None:
    """枚举映射型（相序颜色）：条文给 导体→颜色 映射，文档值查表比对。"""
    mapping = {}
    for c in _CONDUCTORS:
        if m := re.search(rf"{c}\s*(?:导体)?\s*(?:为|必须为|应采用)\s*({'|'.join(_COLORS)})", clause["content"]):
            mapping[c] = m.group(1)
    if not mapping:
        return None
    cond = next((c for c in _CONDUCTORS if c in item_text), None)
    color = next((c for c in _COLORS if c in item_text), None)
    if not (cond and color):
        return None
    ok = mapping.get(cond) == color
    s = next((x for x in _sentences(clause["content"]) if cond in x or "绿黄" in x), None) or ""
    return Judgment(
        verdict="compliant" if ok else "non_compliant",
        confidence=0.8, clause_no=clause["clause_no"], quote=s,
        rationale=f"条文规定 {cond} 应为 {mapping.get(cond)}，文档标注 {color}"
                  + ("，一致。" if ok else "，不一致。"),
        evidence_score=clause.get("rerank", 0.0),
    )


def _scale_ratio(item_text: str, clause: dict) -> Judgment | None:
    """集合成员型（绘图比例）：条文枚举允许系列，文档比例查成员资格。

    允许集只从"不推荐"之前的分句抽——条文的否定分句（"不推荐使用 1:3、
    1:7"）里同样有 比例 字面量，不切走会把禁用项洗白成允许项。"""
    s = next((x for x in _sentences(clause["content"]) if "比例" in x), None)
    if s is None:
        return None
    allowed_part = re.split(r"不推荐|不得使用|不应使用", s)[0]
    allowed = set(re.findall(r"(\d+:\d+)", allowed_part))
    if not allowed:
        return None
    val = re.search(r"(\d+:\d+)", item_text)
    if not val:
        return None
    ok = val.group(1) in allowed
    return Judgment(
        verdict="compliant" if ok else "non_compliant",
        confidence=0.8, clause_no=clause["clause_no"], quote=s,
        rationale=f"条文允许比例系列 {sorted(allowed)}，文档采用 {val.group(1)}"
                  + ("。" if ok else "，不在推荐系列内。"),
        evidence_score=clause.get("rerank", 0.0),
    )


def _live_height(item_text: str, clause: dict) -> Judgment | None:
    """复合条件型（裸露带电部分：低于 2300mm 必须加装护罩）。"""
    s = next((x for x in _sentences(clause["content"]) if "裸露带电" in x or "护罩" in x), None)
    if s is None:
        return None
    m = re.search(r"(?:低于|小于)\s*(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    threshold = _num(m.group(1))
    h = extract_value(item_text)
    if h is None:
        return None
    guarded = ("未加装" not in item_text) and ("无护罩" not in item_text)
    if h >= threshold:
        ok, why = True, f"高度 {h} 不低于条文阈值 {threshold}，不强制加装护罩。"
    else:
        ok = guarded
        why = (f"高度 {h} 低于 {threshold}，条文要求必须加装护罩，"
               + ("文档已设护罩，符合。" if ok else "文档未加装，不符合。"))
    return Judgment(
        verdict="compliant" if ok else "non_compliant",
        confidence=0.85, clause_no=clause["clause_no"], quote=s,
        rationale=why, evidence_score=clause.get("rerank", 0.0),
    )


def _aisle(item_text: str, clause: dict) -> Judgment | None:
    """下限型按位置选句（柜前 ≥800 / 柜后 ≥600）。"""
    pos = "后" if ("柜后" in item_text or "后通道" in item_text) else "前"
    s = next((x for x in _sentences(clause["content"])
              if ("通道" in x) and (("前" in x and pos == "前") or ("后" in x and pos == "后"))),
             None) or next((x for x in _sentences(clause["content"]) if "通道" in x), None)
    if s is None:
        return None
    b = extract_bounds(s)
    val = extract_value(item_text)
    if not (b and "lower" in b and val is not None):
        return None
    ok = val >= b["lower"] - _EPS
    return Judgment(
        verdict="compliant" if ok else "non_compliant",
        confidence=0.9, clause_no=clause["clause_no"], quote=s,
        rationale=f"柜{pos}通道条文下限 {b['lower']}，文档值 {val}"
                  + ("，满足。" if ok else "，不足。"),
        evidence_score=clause.get("rerank", 0.0),
    )


def _pe_section(item_text: str, clause: dict) -> Judgment | None:
    """复合分段型（PE 截面积：相线≤16→同截面；>16→≥一半且≥16）。"""
    s = next((x for x in _sentences(clause["content"]) if "截面积" in x), None)
    if s is None:
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", item_text)
    if len(nums) < 2:
        return None
    phase, pe = _num(nums[0]), _num(nums[-1])
    if "PE" not in item_text and "保护导体" not in item_text:
        return None
    if phase <= 16:
        ok = abs(pe - phase) < _EPS
        why = f"相线 {phase} ≤16 → PE 应与相线相同（{phase}），文档 PE {pe}。"
    else:
        ok = pe >= phase / 2 - _EPS and pe >= 16 - _EPS
        why = (f"相线 {phase} >16 → PE ≥ max({phase / 2:g}, 16)，"
               f"文档 PE {pe}。")
    return Judgment(
        verdict="compliant" if ok else "non_compliant",
        confidence=0.85, clause_no=clause["clause_no"], quote=s,
        rationale=why + ("符合。" if ok else "不符合。"),
        evidence_score=clause.get("rerank", 0.0),
    )


_JUDGES = {
    "ra": _numeric_upper, "ground_res": _numeric_upper, "repair_count": _numeric_upper,
    "reinforcement": _numeric_range, "dry_temp": _numeric_range,
    "undercut": _dual_bound, "misalignment": _misalignment,
    "rt_percent": _percent, "rt_grade": _percent,
    "color_code": _color_code, "scale_ratio": _scale_ratio,
    "live_height": _live_height, "aisle_width": _aisle, "pe_section": _pe_section,
}
# 平面度等未写专用 judge 的属性走数值通用轨
_JUDGES = {**{a: _numeric_upper for a in ATTRIBUTES}, **_JUDGES}


def judge_item(item_text: str, candidates: list[dict]) -> Judgment | None:
    """比较器总入口：检查项 × 候选条款 → 判定。

    按检索名次逐卡试判，第一张出非 None 判定的卡即治理条款：
    第一名常是术语卡或前置条件卡（抽不出约束/守卫不放行），
    自动让位给下一名——"检索第一名 ≠ 治理条款"。
    全部试完仍 None → 上层走 B 级拒答（相关但依据不足）或转 LLM 轨。"""
    attr = detect_attribute(item_text)
    for clause in _eligible(item_text, attr, candidates):
        j = _JUDGES.get(attr, _numeric_upper)(item_text, clause)
        if j is not None:
            return j
    return None


def _eligible(item_text: str, attr: str | None,
              candidates: list[dict]) -> list[dict]:
    """锚词过滤后的候选序列（保持名次序）。"""
    keyed = ATTRIBUTES.get(attr or "", {})
    if not attr:
        return list(candidates)
    return [c for c in candidates
            if any(k in c["content"] for k in keyed["clause_keys"])]
