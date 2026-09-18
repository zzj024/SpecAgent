"""L0 接口层：检索 API + 审查流水线 API（同步 / SSE 流式）+ 前端静态托管。

端点：
  GET  /health                    存活探针
  GET  /search                    混合检索（M1）
  POST /reviews                   同步跑全流水线，返回结构化报告
  POST /reviews/stream            SSE 流式：节点过程事件逐个推送，末尾推报告
  GET  /reviews/{rid}             从库读报告（断点续跑后的恢复读取）
  GET  /reviews/{rid}/steps       执行轨迹（审计回放）
  GET  /                          web/index.html（Vue3 免构建前端）

SSE 实现：流水线在后台线程跑，on_event 回调把事件塞进线程安全队列，
生成器逐条 yield——HTTP 流不阻塞业务线程，前端实时看到 agent 流转。
"""
import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.graph import run_review
from core.config import DSN
from rag.retrieve import hybrid_search, load_model, rerank
from reliability.trace_store import TraceStore

app = FastAPI(title="SpecAgent 标准符合性审查 API", version="0.2.0")
_model = None
_WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


class ReviewRequest(BaseModel):
    doc_text: str = Field(min_length=4, description="待审文档全文（[标签] 行为检查项）")
    filename: str = "uploaded.md"
    mode: str | None = Field(None, description="rule | llm，缺省按 KEY 可用性自动")
    use_memory: bool = True


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


@app.post("/reviews")
def create_review(req: ReviewRequest) -> dict:
    try:
        # API 层每次请求发新单号：重复审同一文档是合法的新审查；
        # 幂等（同 id 续跑/免重跑）留给显式携带 review_id 的调用方（评测/恢复）
        review_id = f"rev-{uuid.uuid4().hex[:12]}"
        return run_review(req.doc_text, document_name=req.filename,
                          review_id=review_id,
                          mode=req.mode, use_memory=req.use_memory)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500,
                            detail=f"审查失败: {type(e).__name__}") from e


@app.post("/reviews/stream")
def stream_review(req: ReviewRequest) -> StreamingResponse:
    """SSE：event: node（agent 流转）→ 逐条推送；event: report 收尾。"""
    events: queue.Queue = queue.Queue()

    def on_event(agent: str, payload: list[dict]) -> None:
        for p in payload:
            events.put(("node", {"agent": agent, **p}))

    def worker() -> None:
        try:
            report = run_review(req.doc_text, document_name=req.filename,
                                review_id=f"rev-{uuid.uuid4().hex[:12]}",
                                mode=req.mode, use_memory=req.use_memory,
                                on_event=on_event)
            events.put(("report", report))
        except Exception as e:  # noqa: BLE001 —— 线程内兜底，错误也走流
            events.put(("report", {"error": f"{type(e).__name__}: {e}"}))
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            item = events.get()
            if item is None:
                break
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/reviews/{rid}")
def get_report(rid: str) -> dict:
    store = TraceStore(DSN)
    review = store.get_review(rid)
    if review is None:
        raise HTTPException(status_code=404, detail="审查单不存在")
    return _report(store, rid, review)


def _report(store: TraceStore, rid: str, review: dict) -> dict:
    findings = store.list_findings(rid)
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["verdict"]] = counts.get(f["verdict"], 0) + 1
    return {
        "review_id": rid, "document_id": review["document_id"],
        "status": review["status"], "domain": review["domain"],
        "mode": review["mode"], "stats": {"total": len(findings), **counts},
        "findings": findings, "steps": len(store.list_steps(rid)),
    }


@app.get("/reviews/{rid}/steps")
def get_steps(rid: str) -> list[dict]:
    steps = TraceStore(DSN).list_steps(rid)
    if not steps:
        raise HTTPException(status_code=404, detail="轨迹不存在")
    return [{"step_seq": s["step_seq"], "agent": s["agent"],
             "output_summary": s["output_summary"], "latency_ms": s["latency_ms"],
             "status": s["status"]} for s in steps]


if (_WEB_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
