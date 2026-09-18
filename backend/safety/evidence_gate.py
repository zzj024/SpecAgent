"""evidence_gate.py：程序化证据门控——无证据不得下结论（L5，代码级强制）。

为什么门控必须是代码而不是 LLM 的自觉（面试高频）：
  审查 Agent 的 LLM 可能幻觉出一个"看起来很像"的条款号或引用原文；
  门控独立于产生结论的 LLM 调用，用数据库事实做两道硬校验：
    第 1 道：条款真实存在（chunk_id + clause_no 在库里查得到）；
    第 2 道：引用原文逐字是卡片正文的子串（防"改写式引用"漂移）。
  两道全过才允许结论进入报告，否则强制降级为 refusal_B（相关但依据不足）。
  第 3 道"证据是否支持结论"属语义判定，rule 模式由比较器复核、llm 模式由
  校验 Agent 判定，见 agents/nodes.py 的 verify 节点。
"""
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row


@dataclass
class GateResult:
    passed: bool
    reason: str = ""          # 通过时为空；失败时给"哪道门、为什么"


def _quote_in_content(quote: str, content: str) -> bool:
    """子串校验。空白容忍：引用可能因排版折行多一个换行/空格，
    压掉全部空白后比较——压完仍不等就是真漂移，不放行。"""
    if quote in content:
        return True
    return "".join(quote.split()) in "".join(content.split())


def verify_evidence(dsn: str, chunk_id: str, clause_no: str, quote: str) -> GateResult:
    """两道程序化硬门。任何一道不过 = 证据链断裂。"""
    if not chunk_id or not clause_no:
        return GateResult(False, "gate1: 缺少条款定位（chunk_id/clause_no 为空）")
    if not quote:
        return GateResult(False, "gate2: 缺少引用原文")

    with psycopg.connect(dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT chunk_id, clause_no, content FROM chunks
                WHERE chunk_id = %s AND clause_no = %s""",
            (chunk_id, clause_no),
        )
        row = cur.fetchone()

    if row is None:
        return GateResult(False, f"gate1: 条款不存在（{chunk_id} / {clause_no}）——疑似幻觉引用")
    if not _quote_in_content(quote, row["content"]):
        return GateResult(False, "gate2: 引用原文不是条款正文的子串——疑似改写漂移")
    return GateResult(True)
