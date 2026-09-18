"""test_api.py：接口层测试。/health 快测；/search 属集成（标 integration）。"""
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from api.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_rejects_short_query():
    """参数校验：查询过短直接 422，不进模型（FastAPI Query 约束生效）。"""
    assert client.get("/search", params={"q": "a"}).status_code == 422


def test_unknown_review_404():
    """不存在的审查单：404 而不是 500（区分'没有'与'坏了'）。"""
    assert client.get("/reviews/rev-nonexistent").status_code == 404
    assert client.get("/reviews/rev-nonexistent/steps").status_code == 404


@pytest.mark.integration
def test_review_endpoint_end_to_end():
    """同步审查接口：完整跑 rule 轨，报告结构与 executor 直调一致。"""
    resp = client.post("/reviews", json={
        "doc_text": "[接地] 电气柜保护接地电阻 5 Ω", "use_memory": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["stats"]["total"] == 1
    assert body["findings"][0]["verdict"] == "non_compliant"
    assert body["findings"][0]["evidence"]["clause_no"] == "4.3.2"
    rid = body["review_id"]
    steps = client.get(f"/reviews/{rid}/steps").json()
    assert [s["agent"] for s in steps] == [
        "router", "retriever", "reviewer", "verifier", "writer"]


@pytest.mark.integration
def test_sse_stream_events():
    """SSE 流：node 事件逐节点推送 + report 收尾。"""
    with client.stream("POST", "/reviews/stream", json={
            "doc_text": "[接地] 电气柜保护接地电阻 5 Ω", "use_memory": False}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        seen_nodes, got_report = [], False
        for line in resp.iter_lines():
            if line.startswith("event: node"):
                seen_nodes.append(line)
            elif line.startswith("event: report"):
                got_report = True
    assert len(seen_nodes) == 5 and got_report


@pytest.mark.integration
def test_search_returns_ranked_clauses():
    resp = client.get("/search", params={"q": "配合面粗糙度最多允许多少", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    assert body[0]["clause_no"] == "4.3.1"
    assert {"standard_id", "clause_no", "content", "score"} <= set(body[0])
