"""parse_markdown：把语料 md 解析成"标准文档树"。

按协作模式（AGENTS.md 2026-09-17 变更）AI 逐段代写，每段经用户走读确认：
  第 1 段：数据结构（StandardDoc/Chapter/Section/Clause）——已确认写入
  第 2 段：逐行扫描解析逻辑（待确认）
  第 3 段：pytest 测试

v1 范围纪律：只解析 章/节/条款 三级；附录、表格、图占位、refs 抽取是 v2。
"""
from dataclasses import dataclass, field


@dataclass
class Clause:
    """条款：树的最小叶子，一条能被单独引用的规则。

    clause_no 是它的门牌号（如 "4.2.1"）。解析层只负责把原文装进来；
    将来超长条款拆卡共享条款号，那是 chunking 层的事，这里不管。"""
    clause_no: str
    text: str


@dataclass
class Section:
    """节：条款的容器，如 4.2 尺寸公差。它自己不是条款。"""
    title: str                       # 带编号的标题，如 "4.2 尺寸公差"
    clauses: list[Clause] = field(default_factory=list)


@dataclass
class Chapter:
    """章：节的容器，如 4 技术要求。"""
    title: str                       # 如 "4 技术要求"
    sections: list[Section] = field(default_factory=list)


@dataclass
class StandardDoc:
    """树根：整份标准。standard_id 是全文唯一标识，如 "SPGT21001-2025"。"""
    standard_id: str
    title: str
    chapters: list[Chapter] = field(default_factory=list)
