"""test_parsing.py：用 21001 的真实结构（截取）钉死 v1 行为边界。"""
from rag.parsing import parse_markdown

SAMPLE = """# SPG/T 21001—2025

## 机械制图与尺寸公差技术要求

## 前言

本标准为 SpecAgent 项目自写模拟标准。

## 1 范围

本标准规定了机械产品图样绘制中的通用技术要求。

## 4 技术要求

### 4.2 尺寸公差

**4.2.1** 图样中未注公差的线性尺寸，按附录 A 中表 A.2 的中等级（m）执行。

**4.2.2** 配合制的选用应符合下列原则：
在一般情况下应优先选用基孔制配合。

### 4.3 表面粗糙度

**4.3.1** 一般配合表面的 Ra 上限值为 1.6 μm。

## 5 试验方法

**5.1** 表面粗糙度采用针描式粗糙度测量仪测量。
"""


def test_tree_shape():
    """树的整体形状：标题、标准号归一化、前言不入树。"""
    doc = parse_markdown(SAMPLE)
    assert doc.standard_id == "SPGT21001-2025"
    assert doc.title == "机械制图与尺寸公差技术要求"
    assert [c.title for c in doc.chapters] == ["1 范围", "4 技术要求", "5 试验方法"]


def test_clauses_and_multiline_text():
    """条款归位 + 跨行续行拼接（4.2.2 那种长条款靠续行拼完整）。"""
    doc = parse_markdown(SAMPLE)
    sec42 = doc.chapters[1].sections[0]
    assert sec42.title == "4.2 尺寸公差"
    assert [c.clause_no for c in sec42.clauses] == ["4.2.1", "4.2.2"]
    assert sec42.clauses[1].text.endswith("基孔制配合。")   # 两行拼成了一条


def test_reset_on_new_chapter():
    """★ 行的回归测试：5.1 必须挂在第 5 章，而不是 4.3 的末尾。

    若删掉换章时的 section=None：5.1 会追加到 4.3，ch5.sections 为空，
    本测试以 IndexError（而非 AssertionError）变红——结构空了就是挂错地方。"""
    doc = parse_markdown(SAMPLE)
    ch5 = doc.chapters[2]
    assert ch5.title == "5 试验方法"
    assert ch5.sections[0].clauses[0].clause_no == "5.1"
