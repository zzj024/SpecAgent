"""L0 检索接口：GET /search（B 档收尾 M1）。

- /health：无模型依赖的存活探针（容器编排/前端探活用）
- /search ：人话查询 → 混合检索(+可选 rerank) → 带条款号与分数的卡片列表

模型全局只加载一次（进程级缓存）；score 字段统一取当档分数
（rerank 开启时为相关分 0~1，关闭时为 RRF 分），前端只认 score。"""
from fastapi import FastAPI, HTTPException, Query
from rag.retrieve import hybrid_search, load_model, rerank

app = FastAPI(title="SpecAgent 检索 API", version="0.1.0")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search")
def search(
    q: str = Query(min_length=2, description="自然语言查询"),
    k: int = Query(5, ge=1, le=20),
    rerank_on: bool = Query(True, description="是否启用 rerank 精排"),
) -> list[dict]:
    try:
        model = get_model()
        hits = hybrid_search(q, k=20, model=model)
        if rerank_on:
            hits = rerank(q, hits, k=k)
        return [
            {
                "standard_id": h["chunk_id"].split(":")[0],
                "clause_no": h["clause_no"],
                "content": h["content"],
                "score": round(h.get("rerank", h.get("rrf", 0.0)), 4),
            }
            for h in hits[:k]
        ]
    except Exception as e:  # noqa: BLE001 —— 接口层兜底，错误信息不外泄内部细节
        raise HTTPException(status_code=500, detail=f"检索失败: {type(e).__name__}") from e
