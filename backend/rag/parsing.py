"""parse_markdown：把语料 md 解析成"标准文档树"。

按协作模式（AGENTS.md 2026-09-17 变更）AI 逐段代写，每段经用户走读确认：
  第 1 段：数据结构（StandardDoc/Chapter/Section/Clause）——已确认写入
  第 2 段：逐行扫描解析逻辑——已确认写入
  第 3 段：pytest 测试（待确认）

v1 范围纪律：只解析 章/节/条款 三级；附录、表格、图占位、refs 抽取、
第 3 章术语条（**3.1　术语名** 格式特殊）均为 v2。
"""
from dataclasses import dataclass, field
import re


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


# ---- 第 2 段：逐行扫描解析逻辑（2026-09-17 用户走读确认后写入）----

_CHAPTER_RE = re.compile(r"^##\s+(\d+)\s+(.+)$")              # "## 4 技术要求"
_SECTION_RE = re.compile(r"^###\s+(\d+\.\d+)\s+(.+)$")        # "### 4.2 尺寸公差"
_CLAUSE_RE  = re.compile(r"^\*\*(\d+(?:\.\d+)+)\*\*\s*(.*)$") # "**4.2.1** 正文……"
_TITLE_RE   = re.compile(r"^#\s+(.+)$")                       # "# SPG/T 21001—2025"


def _normalize_id(raw: str) -> str:
    """'SPG/T 21001—2025' → 'SPGT21001-2025'：去斜杠和空格，长横线改短横线。"""
    return re.sub(r"[/\s]", "", raw).replace("—", "-")


def parse_markdown(src: str) -> StandardDoc:
    """核心循环：逐行扫描，谁匹配了规则就归谁，都不匹配就当条款的续行。

    doc / chapter / section 是循环仅有的三个"手"：边读边搭树。
    ★ 换章时 section=None 不可删——否则下一章开头第一条条款会挂到
      上一章最后一节（test_reset_on_new_chapter 专门防它）。"""
    doc: StandardDoc | None = None
    chapter: Chapter | None = None
    section: Section | None = None

    for raw_line in src.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if (m := _TITLE_RE.match(line)) and doc is None:   # 第一行：标准号
            doc = StandardDoc(standard_id=_normalize_id(m.group(1)), title="")
            continue

        if m := _CHAPTER_RE.match(line):                   # 开新章
            chapter = Chapter(title=f"{m.group(1)} {m.group(2)}")
            doc.chapters.append(chapter)
            section = None                                  # ★换章清空当前节
            continue

        if line.startswith("## "):                          # 无编号 ##：标题/前言/附录
            if not doc.title:                               # 第一个无编号 ## 是文档标题
                doc.title = line[3:].strip()
            section = None                                  # 前言/附录 v1 跳过
            continue

        if m := _SECTION_RE.match(line):                    # 开新节
            section = Section(title=f"{m.group(1)} {m.group(2)}")
            chapter.sections.append(section)
            continue

        if m := _CLAUSE_RE.match(line):                     # 开新条款
            if section is None:                             # 章下直接出现条款（无 ###）
                section = Section(title=chapter.title)      # 建透传节接住
                chapter.sections.append(section)
            section.clauses.append(Clause(clause_no=m.group(1), text=m.group(2)))
            continue

        if section and section.clauses:                     # 普通行：条款的续行
            section.clauses[-1].text += line                # （4.2.2 跨行长条款靠这拼）
        # 其他（表格行、引用块等）v1 忽略

    return doc
