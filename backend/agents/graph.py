"""graph.py：LangGraph 编排 + 执行器——图构建、轨迹落库、断点续跑。

图拓扑（02-技术方案 §2 的五 agent 流水线）：
  START → router → retriever → reviewer → verifier → writer → END

执行器 run_review() 是全流水线的唯一入口（API / 评测脚本 / MCP 都走它），
三件事：
  1. 幂等建档：documents / check_items 撞主键即复用（同文档重复提交同单）；
  2. 断点续跑：每节点完成即 record_step（含全量 state 快照）落 PG；
     进程 kill -9 后重跑同 review_id，从最后完成节点的下一个继续
     （评测六注入点：环境变量 SPECAGENT_CRASH_AFTER=<节点名>，
      该节点落库后立刻 os._exit(137) 模拟进程死亡）；
  3. 流式事件：graph.stream(values 模式) 每节点产出全量 state，
     新增 events 增量推给 on_event 回调（SSE 消费）。

为什么自建快照续跑而不是只靠 LangGraph checkpointer：checkpointer 的
快照在图执行器内部，kill -9 后由它恢复 state，但业务库（findings/
轨迹/游标）的幂等语义仍要自己保证；把快照写进自己的 review_steps 表，
"审计轨迹"与"恢复数据"就是同一份记录——少一份状态源，审计即恢复。
"""
import os
import time
from typing import Callable

import psycopg
from langgraph.graph import END, START, StateGraph

from agents.doc_parser import document_id_for, parse_document
from agents.llm import get_llm
from agents.nodes import NODE_FUNCS, NODE_ORDER
from agents.state import ReviewState
from core.config import DSN
from core.schemas import CheckItem, Finding
from reliability.trace_store import TraceStore

CRASH_AFTER = os.getenv("SPECAGENT_CRASH_AFTER", "")


def build_graph(nodes: list[str]):
    """按给定节点序列编译线性图。续跑时传"剩余节点"子序列——
    图拓扑随恢复点裁剪，已完成节点连函数都不进。"""
    assert [n for n in nodes if n in NODE_ORDER] == nodes, f"未知节点: {nodes}"
    g = StateGraph(ReviewState)
    for n in nodes:
        g.add_node(n, NODE_FUNCS[n])
    g.add_edge(START, nodes[0])
    for a, b in zip(nodes, nodes[1:]):
        g.add_edge(a, b)
    g.add_edge(nodes[-1], END)
    return g.compile()


# ---------------------------------------------------------------------------
# state ↔ JSONB 快照互转（Pydantic 对象不能直接进 JSONB）
# ---------------------------------------------------------------------------


def pack_state(state: dict) -> dict:
    out = dict(state)
    if out.get("items"):
        out["items"] = [i.model_dump() if isinstance(i, CheckItem) else i
                        for i in out["items"]]
    for key in ("findings", "verified"):
        if out.get(key):
            out[key] = [f.model_dump() if isinstance(f, Finding) else f
                        for f in out[key]]
    out.pop("report", None)                    # 报告由 writer 现算，不进快照
    return out


def unpack_state(snapshot: dict) -> dict:
    """DB 快照 → 可执行 state（Pydantic 对象重建）。"""
    state = dict(snapshot)
    if state.get("items"):
        state["items"] = [CheckItem(**i) for i in state["items"]]
    for key in ("findings", "verified"):
        if state.get(key):
            state[key] = [
                Finding(**{**f, "item": CheckItem(**f["item"])}) for f in state[key]
            ]
    return state


# ---------------------------------------------------------------------------


def _persist_document(dsn: str, document_id: str, filename: str, text: str,
                      conf: float, items: list[CheckItem]) -> None:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO documents (document_id, filename, parse_confidence, content)
               VALUES (%s, %s, %s, %s) ON CONFLICT (document_id) DO NOTHING""",
            (document_id, filename, conf, text),
        )
        cur.executemany(
            """INSERT INTO check_items (item_id, document_id, seq, text, tag,
                                        attribute, value_text, parse_confidence)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (item_id) DO NOTHING""",
            [(f"{document_id}:{i.seq}", document_id, i.seq, i.text, i.tag,
              i.attribute, i.value_text, i.parse_confidence) for i in items],
        )


def _save_findings(store: TraceStore, review_id: str, document_id: str,
                   findings: list[Finding]) -> None:
    for f in findings:
        store.save_finding(review_id, {
            "item_id": f"{document_id}:{f.item.seq}",   # 对齐 check_items 主键
            "item_seq": f.item.seq,
            "item_text": f.item.text,
            "verdict": f.verdict,
            "confidence": f.confidence,
            "evidence": f.evidence.model_dump(),
            "rationale": f.rationale,
            "verify": f.verify,
        })


def run_review(
    doc_text: str,
    document_name: str = "uploaded.md",
    review_id: str | None = None,
    mode: str | None = None,
    use_memory: bool = True,
    wm_strategy: str = "full",
    dsn: str = DSN,
    on_event: Callable[[str, list[dict]], None] | None = None,
) -> dict:
    """全流程入口。返回 writer 产出的报告 dict（失败时返回 error 报告）。"""
    items, doc_conf = parse_document(doc_text)
    document_id = document_id_for(document_name, doc_text)
    if mode is None:
        mode = "llm" if get_llm().available else "rule"
    if review_id is None:
        review_id = f"rev-{document_id}"
        if not use_memory:
            review_id += "-nomem"               # 评测四：消融组各自独立审查单
        if wm_strategy != "full":
            review_id += f"-{wm_strategy}"      # 评测五：三策略互不覆盖

    _persist_document(dsn, document_id, document_name, doc_text, doc_conf, items)
    store = TraceStore(dsn)
    store.create_review(review_id, document_id, mode=mode)

    last_node, snapshot, last_seq = store.load_progress(review_id)
    remaining = NODE_ORDER[NODE_ORDER.index(last_node) + 1:] if last_node else NODE_ORDER

    if not remaining:                            # 已跑完：直接从库里出报告
        return _report_from_db(store, review_id, mode)

    base: ReviewState = {                        # 快照（若有）优先，缺省字段补齐
        "review_id": review_id, "document_id": document_id,
        "document_name": document_name, "doc_text": doc_text,
        "doc_parse_confidence": doc_conf, "mode": mode,
        "use_memory": use_memory, "wm_strategy": wm_strategy,
        "items": items,
    }
    state = unpack_state(snapshot) if snapshot else dict(base)
    state.update({k: v for k, v in base.items() if k not in ("items",)})

    graph = build_graph(remaining)
    step_seq = last_seq
    n_events = 0
    final: dict = state
    t0 = time.perf_counter()

    try:
        for tick, updated in enumerate(graph.stream(state, stream_mode="values")):
            if tick == 0:                        # values 模式首个产出是初始 state
                continue
            final = updated
            agent = remaining[tick - 1]
            step_seq += 1
            store.record_step(
                review_id, step_seq, agent, pack_state(updated),
                output_summary=str(
                    (updated.get("events") or [{}])[-1].get("detail", ""))[:200],
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            if on_event and updated.get("events"):
                on_event(agent, updated["events"][n_events:])
                n_events = len(updated["events"])
            if agent == "writer":
                _save_findings(store, review_id, document_id,
                               updated.get("verified") or updated.get("findings") or [])
            if agent == CRASH_AFTER:             # 评测六：确定性崩溃注入点
                os._exit(137)
        store.finish_review(review_id, "completed")
        report = final.get("report") or _report_from_db(store, review_id, mode)
        report["elapsed_s"] = round(time.perf_counter() - t0, 2)
        return report
    except Exception as e:  # noqa: BLE001 —— 执行器兜底：任何节点异常都落库不裸抛
        store.finish_review(review_id, "failed", error=str(e)[:500])
        return {"review_id": review_id, "error": str(e)[:500], "mode": mode,
                "stats": {"total": 0}}


def _report_from_db(store: TraceStore, review_id: str, mode: str) -> dict:
    rows = store.list_findings(review_id)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {
        "review_id": review_id, "mode": mode,
        "stats": {"total": len(rows), **counts},
        "findings": rows, "recovered": True,
    }
