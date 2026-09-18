"""test_evidence_gate.py（integration）：门控两道硬校验对真实库的 chunks。"""
import pytest

from core.config import DSN
from safety.evidence_gate import verify_evidence

pytestmark = pytest.mark.integration


def _real_chunk():
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(DSN, row_factory=dict_row) as conn, conn.cursor() as cur:
        return cur.execute(
            "SELECT chunk_id, clause_no, content FROM chunks LIMIT 1").fetchone()


def test_valid_evidence_passes():
    row = _real_chunk()
    gate = verify_evidence(DSN, row["chunk_id"], row["clause_no"],
                           row["content"][:20])
    assert gate.passed


def test_hallucinated_clause_rejected():
    gate = verify_evidence(DSN, "SPGT21001-2025:9.9.9:0", "9.9.9", "随便引一句")
    assert not gate.passed and "gate1" in gate.reason


def test_fabricated_quote_rejected():
    """改写式引用（不是原文子串）必须被第 2 道门拦下。"""
    row = _real_chunk()
    gate = verify_evidence(DSN, row["chunk_id"], row["clause_no"],
                           "这句根本不在条文原文里——LLM 幻觉引用")
    assert not gate.passed and "gate2" in gate.reason


def test_whitespace_tolerant_match():
    row = _real_chunk()
    quote = row["content"][:20]
    gate = verify_evidence(DSN, row["chunk_id"], row["clause_no"],
                           quote[:5] + "  \n " + quote[5:])     # 排版折行多空白
    assert gate.passed
