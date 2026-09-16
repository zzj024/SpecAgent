"""chunk_standard：结构化分块——把标准文档树切成"卡片"（Chunk）。

按协作模式（AGENTS.md 2026-09-17 变更）AI 逐段代写，每段经用户走读确认：
  第 1 段：Chunk 数据结构——已确认写入
  第 2 段：遍历树产卡 + 超长条款按句子边界拆分（待确认）
  第 3 段：pytest 测试（待确认）

设计原则（对应课程"结构化分块"）：叶子条款 = 一张卡，chunk 与条款号约 1:1，
引用才能指准；超长条款按句子边界拆而不丢（共享 clause_no，sub_idx 区分）。
v1 只产 clause 卡；表格整卡、图描述并入、refs 正则抽取均为 v2。
"""
from dataclasses import dataclass, field
from typing import Literal

import re

from rag.parsing import StandardDoc


@dataclass
class Chunk:
    """一张卡片：向量检索与证据门控"子串校验"发生在这一层。

    chunk_id 三段式 = 组地址 + 页码，是将来 chunks 表的唯一键——
    幂等不写在 if 里，写在唯一键里（重复入库同卡，数据库直接拒绝）。"""
    chunk_id: str                 # "{standard_id}:{clause_no}:{sub_idx}"
    standard_id: str
    clause_no: str                # 超长条款拆卡后共享（组地址），靠 sub_idx 区分
    section_path: list[str]       # ["4 技术要求", "4.2 尺寸公差"]，溯源展示用
    text: str                     # 卡片正文，"引用必须是子串"验在这里
    kind: Literal["clause", "table", "figure"] = "clause"   # v1 只有 clause
    refs: list[str] = field(default_factory=list)           # 谁引用记谁，v2 正则填充


# ---- 第 2 段：遍历树产卡 + 超长条款拆分（2026-09-17 用户走读确认后写入）----

_MAX_LEN = 800   # 卡片正文长度阈值：BGE-M3 对 800 字内的中文语义保持最好


def _split_long(text: str, max_len: int) -> list[str]:
    """超长条款按句子边界拆段：在句号/分号后的"空隙"切，标点留在前段。

    必须用零宽断言 (?<=) 而不是普通 split：后者会把标点当分隔符吃掉，
    卡片文本与原文不再精确子串——证据门控第 2 查会误杀正常结论。"""
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    buf = ""
    for sentence in re.split(r"(?<=[。；])", text):
        if buf and len(buf) + len(sentence) > max_len:
            parts.append(buf)        # 再装就超 → 手里这段先落袋
            buf = sentence
        else:
            buf += sentence          # 还装得下 → 继续攒
    if buf:
        parts.append(buf)
    return parts


def chunk_standard(doc: StandardDoc, max_len: int = _MAX_LEN) -> list[Chunk]:
    """遍历文档树：叶子条款 → 卡片。透传节（节名=章名）的路径自动去重。

    已知妥协（v1）：单句超长时原样超长不硬切——中文标准几乎不存在
    单句超 800 字，不为极端情况引入切句复杂度。"""
    chunks: list[Chunk] = []
    for chapter in doc.chapters:
        for section in chapter.sections:
            path = [chapter.title]                     # 章节路径按"节"算
            if section.title != chapter.title:         # 透传节不重复记
                path.append(section.title)
            for clause in section.clauses:
                pieces = _split_long(clause.text, max_len)
                for i, piece in enumerate(pieces):     # i 就是 sub_idx
                    chunks.append(Chunk(
                        chunk_id=f"{doc.standard_id}:{clause.clause_no}:{i}",
                        standard_id=doc.standard_id,
                        clause_no=clause.clause_no,
                        section_path=path,
                        text=piece,
                    ))
    return chunks
