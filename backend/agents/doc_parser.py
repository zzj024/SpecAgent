"""doc_parser.py：待审文档 → 检查项清单。

约定的文档格式（demo 语料，evals/gen_docs.py 同格式生成）：
  - `[标签] 描述行`：一行一个检查项，标签用于路由与属性归一；
  - 其余行是背景说明，不产检查项；
  - `<!-- scan-confidence: 0.x -->`：扫描件质量标记——文档级解析置信度，
    低于阈值时全单走 C 级拒答（"输入解析存疑，请人工复核"）。

无标签兜底：含属性关键词（粗糙度/接地电阻/咬边…）的裸行也收，
parse_confidence 降到 0.6——真实文档不会都打标签，兜底行制造
"可解析但不确定"的中间态，正好是评测三 C 级样本的来源。
"""
import hashlib
import re

from agents.compare import detect_attribute
from core.schemas import CheckItem

_SCAN_MARK = re.compile(r"<!--\s*scan-confidence:\s*([0-9.]+)\s*-->")
_TAGGED = re.compile(r"^\[([^\]]+)\]\s*(.+)$")


def parse_document(text: str) -> tuple[list[CheckItem], float]:
    """返回 (检查项列表, 文档级解析置信度)。"""
    conf_match = _SCAN_MARK.search(text)
    doc_conf = float(conf_match.group(1)) if conf_match else 1.0

    items: list[CheckItem] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<!--"):
            continue
        if m := _TAGGED.match(line):
            tag, body = m.group(1), m.group(2)
            items.append(CheckItem(
                seq=len(items), text=body, tag=tag,
                attribute=detect_attribute(body) or "",
                value_text=body, parse_confidence=doc_conf,
            ))
        elif detect_attribute(line):                    # 无标签兜底
            items.append(CheckItem(
                seq=len(items), text=line, tag="",
                attribute=detect_attribute(line) or "",
                value_text=line, parse_confidence=0.6,
            ))
    return items, doc_conf


def document_id_for(filename: str, text: str) -> str:
    """内容哈希做 document_id：同一文档重复上传 → 同 id → 幂等。"""
    return "doc-" + hashlib.sha1(f"{filename}:{text}".encode()).hexdigest()[:12]
