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


@pytest.mark.integration
def test_search_returns_ranked_clauses():
    resp = client.get("/search", params={"q": "配合面粗糙度最多允许多少", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    assert body[0]["clause_no"] == "4.3.1"
    assert {"standard_id", "clause_no", "content", "score"} <= set(body[0])
