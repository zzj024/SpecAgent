"""test_retrieve.py：RRF 融合纯函数单测 + 混合检索集成测（标 integration）。"""
import os

import pytest

# 集成测试依赖本地模型，强制离线（HF_HUB_OFFLINE 必须在导入前生效）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from rag.retrieve import hybrid_search, rrf_merge  # noqa: E402


def test_consensus_beats_single_path_leader():
    """向量路第一名若无 BM25 认可，输给两路都认可的卡——共识压倒单路狂热。"""
    vec = [{"chunk_id": "X:1:0", "content": "x"}, {"chunk_id": "Y:1:0", "content": "y"}]
    kw = [{"chunk_id": "Y:1:0", "content": "y"}]
    merged = rrf_merge([vec, kw], k=2)
    assert [m["chunk_id"] for m in merged] == ["Y:1:0", "X:1:0"]
    assert merged[0]["rrf"] > merged[1]["rrf"]


def test_k_truncation_and_missing_path():
    """k 截断生效；空路（未命中）不报错、不贡献分数。"""
    vec = [{"chunk_id": f"C{i}:1:0"} for i in range(5)]
    merged = rrf_merge([vec, []], k=3)
    assert len(merged) == 3
    assert [m["chunk_id"] for m in merged] == ["C0:1:0", "C1:1:0", "C2:1:0"]


def test_rrf_score_values():
    """手算对账：两路都是第 1 名 → 2×(1/61)≈0.03279（夹具与断言必须一致）。"""
    vec = [{"chunk_id": "A:1:0"}]
    kw = [{"chunk_id": "A:1:0", "content": "a"}]
    merged = rrf_merge([vec, kw], k=1)
    assert abs(merged[0]["rrf"] - 2 * (1 / 61)) < 1e-9


@pytest.mark.integration
def test_hybrid_finds_roughness_clause():
    """集成：口语查询必须在 top5 命中 4.3.1（需库与模型，-m integration 运行）。"""
    top5 = hybrid_search("配合面粗糙度最多允许多少", k=5)
    assert any(h["clause_no"] == "4.3.1" for h in top5)
