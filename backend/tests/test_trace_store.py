"""test_trace_store.py（integration）：唯一写入口的幂等与恢复游标。"""
import pytest

from core.config import DSN
from reliability.trace_store import TraceStore

pytestmark = pytest.mark.integration

RID = "rev-test-trace"


@pytest.fixture()
def store():
    s = TraceStore(DSN)
    with s._conn() as conn, conn.cursor() as cur:      # 测试数据隔离：清旧单据
        for t in ("findings", "review_steps", "reviews"):
            cur.execute(f"DELETE FROM {t} WHERE review_id = %s", (RID,))
        cur.execute("DELETE FROM check_items WHERE document_id LIKE 'doctest%'")
        cur.execute("DELETE FROM documents WHERE document_id LIKE 'doctest%'")
        cur.execute(                                     # 本套夹具共用的文档行
            """INSERT INTO documents (document_id, filename, content)
               VALUES ('doctest-1', 't.md', 'x')""")
    return s


def test_record_step_idempotent(store):
    store.create_review(RID, "doctest-1")
    assert store.record_step(RID, 1, "router", {"a": 1})
    assert not store.record_step(RID, 1, "router", {"a": 1})   # 重放撞键被跳过
    node, snap, seq = store.load_progress(RID)
    assert node == "router" and snap == {"a": 1} and seq == 1


def test_cursor_moves_with_seq(store):
    store.create_review(RID, "doctest-1")
    store.record_step(RID, 1, "router", {"x": 1})
    store.record_step(RID, 2, "retriever", {"x": 2})
    node, _, seq = store.load_progress(RID)
    assert (node, seq) == ("retriever", 2)
    assert len(store.list_steps(RID)) == 2


def test_save_finding_no_duplicate(store):
    store.create_review(RID, "doctest-1")
    with store._conn() as conn, conn.cursor() as cur:
        cur.execute(                              # 文档行由 fixture 建好，只补检查项
            """INSERT INTO check_items (item_id, document_id, seq, text)
               VALUES ('doctest-1:0', 'doctest-1', 0, '测试检查项')""")
    payload = {"item_id": "doctest-1:0", "item_seq": 0, "item_text": "测试检查项",
               "verdict": "non_compliant", "confidence": 0.9,
               "evidence": {"chunk_id": "c", "clause_no": "4.3.1", "quote": "q"},
               "rationale": "r", "verify": "passed"}
    assert store.save_finding(RID, payload)
    assert not store.save_finding(RID, payload)                # 断点重放不双写
    assert len(store.list_findings(RID)) == 1
