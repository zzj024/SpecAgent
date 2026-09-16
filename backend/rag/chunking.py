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
