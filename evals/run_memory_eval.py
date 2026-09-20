"""评测四：长期记忆消融——同一批评查用例，有/无经验库两组对比。

方法：取评测二前 10 份文档，各跑两组（配置完全一致，仅 use_memory 不同）：
  baseline 组：use_memory=False（屏蔽记忆）
  treatment 组：先跑一遍沉淀经验（use_memory=True 本身边审边沉淀），再重跑
指标：漏报率 / 误报率 / 耗时。

诚实预期（结果如实记录，允许负结果）：rule 轨的记忆注入只进 rationale
不进判定（判定由确定性比较器完成），两组判定数字应完全一致——
这不是 bug 而是设计使然：经验影响的是 LLM 判定的上下文，
llm 轨（配置 DEEPSEEK_API_KEY）下消融才有意义。脚本输出会显式标注 mode。
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.graph import run_review                     # noqa: E402
from core.config import DSN                             # noqa: E402
from memory.long_term import MemoryStore                # noqa: E402
from reliability.trace_store import TraceStore          # noqa: E402

HERE = Path(__file__).parent
DOCS = HERE / "review_docs"
N = 10


def _purge(rid: str) -> None:
    s = TraceStore(DSN)
    with s._conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM findings WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM review_steps WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM reviews WHERE review_id = %s", (rid,))


def run_group(tag: str, use_memory: bool, ann: dict, mode: str = "rule") -> dict:
    miss = nc_pred = tp = fn = 0
    t0 = time.perf_counter()
    for d in range(N):
        doc_id = f"eval2-doc{d:02d}"
        rid = f"rev-eval4-{tag}-{doc_id}"
        _purge(rid)
        text = (DOCS / f"{doc_id}.md").read_text(encoding="utf-8")
        report = run_review(text, document_name=f"{doc_id}.md", review_id=rid,
                            mode=mode, use_memory=use_memory)
        findings = {f["item_seq"]: f["verdict"] for f in report.get("findings", [])}
        for seq, violated in ann[doc_id].items():
            pred_nc = findings.get(seq) == "non_compliant"
            nc_pred += pred_nc
            if violated and pred_nc:
                tp += 1
            elif violated and not pred_nc:
                fn += 1
    return {"miss_rate": round(fn / (tp + fn), 4) if tp + fn else 0.0,
            "n_nc_pred": nc_pred, "elapsed_s": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ann_raw = [json.loads(x) for x in
               (DOCS / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if x]
    ann: dict[str, dict[int, bool]] = {}
    for a in ann_raw:
        if a["doc_id"] < f"eval2-doc{N:02d}":
            ann.setdefault(a["doc_id"], {})[a["seq"]] = a["violated"]

    mem = MemoryStore(DSN)
    mem.clear()                                   # 基线从零记忆开始
    baseline = run_group("nomem", False, ann)

    mem.clear()
    treatment_first = run_group("mem-warm", True, ann)   # 第一遍：边审边沉淀
    n_mem = mem.count()
    treatment = run_group("mem-run", True, ann)          # 第二遍：带经验库重审

    result = {
        "date": date.today().isoformat(), "mode": "rule", "n_docs": N,
        "baseline": baseline, "treatment": treatment,
        "memory_entries": n_mem,
        "note": "rule 轨记忆只进 rationale 不进判定（设计使然）：预期两组判定"
                "一致；llm 轨（DEEPSEEK_API_KEY）下经验进 prompt，消融才有行为差",
    }
    out = HERE / "results" / f"memory_results_{date.today().isoformat()}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已存 {out}")


if __name__ == "__main__":
    main()
