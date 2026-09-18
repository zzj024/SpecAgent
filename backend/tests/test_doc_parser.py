"""test_doc_parser.py：待审文档解析——标签行/兜底行/扫描件置信度。"""
from agents.doc_parser import document_id_for, parse_document


DOC = """# 支架零件设计技术要求（示例）

[粗糙度] 阶梯轴配合表面粗糙度 Ra 3.2
[比例] 主视图绘图比例 1:3
[接地] 电气柜保护接地电阻 5 Ω

其余未打标签的背景说明行不产检查项。
未打标签但提到接地电阻的行会走兜底通道。
"""


def test_tagged_lines_become_items():
    items, conf = parse_document(DOC)
    texts = [i.text for i in items]
    assert any("Ra 3.2" in t for t in texts)
    assert conf == 1.0
    seqs = [i.seq for i in items]
    assert seqs == sorted(seqs)                    # seq 连续递增


def test_untagged_keyword_fallback_lower_confidence():
    items, _ = parse_document(DOC)
    fallback = [i for i in items if "兜底" in i.text]
    assert fallback and fallback[0].parse_confidence == 0.6


def test_scan_marker_lowers_doc_confidence():
    items, conf = parse_document("<!-- scan-confidence: 0.3 -->\n[粗糙度] Ra 3.2")
    assert conf == 0.3
    assert items[0].parse_confidence == 0.3        # 逐项继承文档级置信度


def test_document_id_stable():
    a = document_id_for("a.md", "内容")
    b = document_id_for("a.md", "内容")
    c = document_id_for("a.md", "内容2")
    assert a == b and a != c
