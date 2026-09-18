"""评测二：审查准确率——人工埋错法（漏报率 / 误报率 / 证据一致性）。

用法：HF_HUB_OFFLINE=1 .venv/Scripts/python.exe evals/run_review_eval.py

方法（03-评测方案）：30 份埋错文档全量跑审查流水线（rule 轨，无 LLM 依赖，
数字离线可复现），比对 findings 与标注：
  漏报率 = 埋错项被判成非 non_compliant 的比例
  误报率 = 合规项被判 non_compliant 的比例
  证据一致性 = non_compliant 结论挂的条款号与治理条款库的命中率
BadCase 落归因表（解析/检索/门控三类），供 docs/问题档案复盘。
"""
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.graph import run_review                     # noqa: E402
from core.config import DSN                             # noqa: E402
from reliability.trace_store import TraceStore          # noqa: E402

HERE = Path(__file__).parent
DOCS = HERE / "review_docs"

# 属性 → 治理条款（人工埋错的 ground truth 侧写；BadCase 归因用）
GOVERNING = {
    "ra": "4.3.1", "ra_raw": "4.3.1", "scale_ratio": "4.1.2",
    "reinforce": "4.2.1", "undercut_d": "4.1.2", "undercut_l": "4.1.2",
    "misalign": "4.2.2", "rt_pct": "4.3.1", "dry_temp": "4.4.1",
    "repair": "6.3", "ground_res": "4.3.2", "aisle_front": "4.2.2",
    "aisle_back": "4.2.2", "pe_section": "4.3.1", "color_pe": "4.1.2",
    "live_h": "4.2.1",
}


def _purge(review_ids: list[str]) -> None:
    """清旧单：重跑必须从零判定，否则续跑机制会直接复用库里的旧结论。"""
    s = TraceStore(DSN)
    with s._conn() as conn, conn.cursor() as cur:
        for rid in review_ids:
            cur.execute("DELETE FROM findings WHERE review_id = %s", (rid,))
            cur.execute("DELETE FROM review_steps WHERE review_id = %s", (rid,))
            cur.execute("DELETE FROM reviews WHERE review_id = %s", (rid,))


def main() -> None:
    ann = [json.loads(x) for x in
           (DOCS / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if x]
    by_doc: dict[str, dict[int, dict]] = {}
    for a in ann:
        by_doc.setdefault(a["doc_id"], {})[a["seq"]] = a

    doc_ids = sorted(by_doc)
    review_ids = [f"rev-eval2-{d}" for d in doc_ids]
    _purge(review_ids)

    tp = fn = fp = tn = 0
    ev_hit = ev_total = 0
    badcases = []
    t0 = time.perf_counter()

    for doc_id, rid in zip(doc_ids, review_ids):
        text = (DOCS / f"{doc_id}.md").read_text(encoding="utf-8")
        report = run_review(text, document_name=f"{doc_id}.md", review_id=rid,
                            mode="rule", use_memory=False)
        findings = {f["item_seq"]: f for f in report.get("findings", [])}
        for seq, truth in by_doc[doc_id].items():
            f = findings.get(seq)
            pred_nc = bool(f) and f["verdict"] == "non_compliant"
            if truth["violated"] and pred_nc:
                tp += 1
            elif truth["violated"] and not pred_nc:
                fn += 1
                badcases.append({"doc": doc_id, "seq": seq, "type": "漏报",
                                 "truth": "不符合", "pred": f["verdict"] if f else "缺失",
                                 "attr": truth["attr"], "why": f["rationale"][:80] if f else ""})
            elif not truth["violated"] and pred_nc:
                fp += 1
                badcases.append({"doc": doc_id, "seq": seq, "type": "误报",
                                 "truth": "符合", "pred": f["verdict"] if f else "缺失",
                                 "attr": truth["attr"], "why": f["rationale"][:80] if f else ""})
            else:
                tn += 1
            if pred_nc and f:
                ev_total += 1
                ev_hit += f["evidence"]["clause_no"] == GOVERNING.get(truth["attr"])

    n = tp + fn + tn + fp
    result = {
        "date": date.today().isoformat(), "n_docs": len(doc_ids), "n_items": n,
        "miss_rate": round(fn / (tp + fn), 4) if tp + fn else 0.0,
        "false_alarm_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "evidence_clause_hit": round(ev_hit / ev_total, 4) if ev_total else 0.0,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "badcase_types": dict(Counter(b["type"] + ":" + b["attr"] for b in badcases)),
        "elapsed_min": round((time.perf_counter() - t0) / 60, 1),
        "mode": "rule",
    }
    out = HERE / "results" / f"review_results_{date.today().isoformat()}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "badcase_types"},
                     ensure_ascii=False, indent=2))
    print(f"BadCase 归因：{result['badcase_types']}")
    print(f"结果已存 {out}")


if __name__ == "__main__":
    main()
