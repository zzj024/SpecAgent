"""test_chunking.py：钉死分块三定律——1:1、组地址、拆而不丢。"""
from rag.chunking import chunk_standard, _split_long
from rag.parsing import Chapter, Clause, Section, StandardDoc

LONG_TEXT = "配合代号由基本尺寸与公差带代号组成。" + "过渡配合用于常拆装连接。" * 100


def _doc() -> StandardDoc:
    """构造迷你树：短条款 / 千字长条款 / 透传节里的条款，三种情况齐活。"""
    return StandardDoc(
        standard_id="SPGT21001-2025",
        title="机械制图与尺寸公差技术要求",
        chapters=[Chapter(title="4 技术要求", sections=[
            Section(title="4.2 尺寸公差", clauses=[
                Clause(clause_no="4.2.1", text="未注公差按表 A.2 的中等级（m）执行。"),
                Clause(clause_no="4.2.3", text=LONG_TEXT),
            ]),
            Section(title="4 技术要求", clauses=[      # 透传节（节名=章名）
                Clause(clause_no="3.1", text="基本尺寸：设计给定的尺寸。"),
            ]),
        ])],
    )


def test_short_clause_single_card():
    """短条款一张卡：chunk_id 三段式，sub_idx=0，路径两段齐全。"""
    chunks = [x for x in chunk_standard(_doc()) if x.clause_no == "4.2.1"]
    assert chunks[0].chunk_id == "SPGT21001-2025:4.2.1:0"
    assert chunks[0].section_path == ["4 技术要求", "4.2 尺寸公差"]


def test_long_clause_split_shares_clause_no():
    """长条款拆多卡：sub_idx 连续编号，拼接后与原文逐字相等（拆而不丢）。"""
    chunks = [x for x in chunk_standard(_doc()) if x.clause_no == "4.2.3"]
    assert len(chunks) >= 2
    assert [x.chunk_id.rsplit(":", 1)[1] for x in chunks] == \
           [str(i) for i in range(len(chunks))]
    assert "".join(x.text for x in chunks) == LONG_TEXT


def test_passthrough_section_path_dedup():
    """透传节路径去重：3.1 的路径只有一章，不出现重复段。"""
    chunks = [x for x in chunk_standard(_doc()) if x.clause_no == "3.1"]
    assert chunks[0].section_path == ["4 技术要求"]


def test_split_long_keeps_punctuation():
    """零宽断言的疫苗：切分必须保标点、不丢字——门控第 2 查的命根。"""
    parts = _split_long("Ra≤1.6。结合面≤6.3。", 5)
    assert parts[0].endswith("。")
    assert "".join(parts) == "Ra≤1.6。结合面≤6.3。"
